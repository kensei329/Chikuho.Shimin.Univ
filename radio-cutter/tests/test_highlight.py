"""steps/s5_pick_highlight.py — ハイライト選定（SPEC Step 5）。

ここで守らせたいのは SPEC Step 5「採用後の後処理（必須）」の3段スナップ:

  1. start / end を最寄りの単語境界にスナップ
  2. start は文の先頭まで前方に、end は文末まで後方に拡張（`。？！` で判定）
  3. silencedetect で無音の谷に寄せる
  4. max_duration_sec を超えたら末尾の文を1つ落として再計算

「この3段スナップを飛ばすと、語尾が千切れて視聴に耐えないものが出る」と SPEC が
名指しで警告している箇所なので、純関数 `snap_highlight` を厚く固める。

`run` 側は SPEC 9章「ハイライト候補が本編の範囲外 → その候補を破棄し、次点を採用。
全滅なら停止」と、SPEC 8章 decisions.json（selected / alternatives / snapped_from /
llm_calls / warnings）の約束を検証する。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
import shutil
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import SILENCE_BACKOFF_SEC
from radio_cutter.errors import (
    HighlightError,
    LlmError,
    MissingArtifactError,
    RadioCutterError,
)
from radio_cutter.llm.client import StubLlmClient
from radio_cutter.models import CutPoint, HighlightCandidate, HighlightResult, Word
from radio_cutter.steps import s5_pick_highlight as s5

# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------


def W(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def cand(
    start: float,
    end: float,
    *,
    score: float = 90.0,
    hook: str = "フックの一言",
    reason: str = "理由",
) -> HighlightCandidate:
    return HighlightCandidate(start=start, end=end, score=score, hook_line=hook, reason=reason)


#: 3文。文末は「。」で、文の切れ目は 3.0 / 6.0 / 8.0 秒。
THREE_SENTENCES: list[Word] = [
    W("これは", 0.0, 1.0),
    W("テストの", 1.0, 2.0),
    W("文です。", 2.0, 3.0),
    W("次の", 3.0, 4.0),
    W("文は", 4.0, 5.0),
    W("ここまで。", 5.0, 6.0),
    W("三つ目の", 6.0, 7.0),
    W("文です。", 7.0, 8.0),
]

#: 1単語＝1文。文境界への拡張が効かないので「単語境界へのスナップ」だけを見られる。
STACCATO: list[Word] = [
    W("あ。", 0.0, 1.0),
    W("い。", 1.0, 2.0),
    W("う。", 2.0, 3.0),
    W("え。", 3.0, 4.0),
    W("お。", 4.0, 5.0),
]

#: 「？」「！」も文末（SPEC Step 5「`。？！` で判定」）。
QA_WORDS: list[Word] = [
    W("これは", 0.0, 1.0),
    W("なぜ", 1.0, 2.0),
    W("ですか？", 2.0, 3.0),
    W("だから", 3.0, 4.0),
    W("こうです！", 4.0, 5.0),
    W("最後の", 5.0, 6.0),
    W("文です。", 6.0, 7.0),
]

#: 全体で1文。しかも文末記号すら無い（ASR が句点を落とすことがある）。
ONE_SENTENCE_NO_PERIOD: list[Word] = [
    W("ずっと", 0.0, 2.0),
    W("しゃべり", 2.0, 4.0),
    W("続けている", 4.0, 6.0),
]


def make_recording_lookup(*, shift: float = 0.05, found: bool = True):
    """呼ばれた時刻を記録するフェイクの silence_lookup。

    start は shift だけ手前へ、end は shift だけ後ろへ寄せる（語頭・語尾を残す向き）。
    """
    calls: list[tuple[float, str]] = []

    def lookup(t: float, *, kind: str = "start") -> tuple[float, bool]:
        calls.append((round(float(t), 6), kind))
        if not found:
            return (float(t), False)
        return (float(t) - shift if kind == "start" else float(t) + shift, True)

    lookup.calls = calls  # type: ignore[attr-defined]
    return lookup


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def cuts() -> dict[str, CutPoint]:
    """合成エピソードのカット点（Step 4 の出力に相当）。main は 5.90〜43.90 秒。"""
    return {
        "A": CutPoint(
            anchor_id="A",
            raw_cut_time=fixtures.EXPECTED_ANCHOR_A_RAW,
            cut_time=fixtures.EXPECTED_CUT_A,
            silence_found=True,
            score=100.0,
        ),
        "B": CutPoint(
            anchor_id="B",
            raw_cut_time=fixtures.EXPECTED_ANCHOR_B_RAW,
            cut_time=fixtures.EXPECTED_CUT_B,
            silence_found=True,
            score=96.2,
        ),
    }


MAIN_LO = fixtures.EXPECTED_CUT_A   # 5.90
MAIN_HI = fixtures.EXPECTED_CUT_B   # 43.90

#: 本編の先頭・末尾の単語（words_in_bounds の結果の端）
FIRST_MAIN_WORD_START = 5.95
LAST_MAIN_WORD_END = 43.60


def highlight_stub(candidates: list[dict], *, model: str = "stub-model") -> StubLlmClient:
    """任意の候補を返す Step 5 用スタブ。"""
    return StubLlmClient({"highlight": {"candidates": candidates}}, model=model)


def raw_candidate(start: float, end: float, score: float, hook: str = "フック") -> dict:
    return {
        "start": start,
        "end": end,
        "score": score,
        "hook_line": hook,
        "reason": "テスト用の理由。",
    }


# ===========================================================================
# words_in_bounds — LLM に渡す／スナップに使う単語の範囲
# ===========================================================================


class TestWordsInBounds:
    """本編の外の単語を混ぜないこと。混ぜると文境界への拡張が本編の外へ出る。"""

    def test_開始時刻が範囲内の単語だけを返す(self, transcript):
        words = s5.words_in_bounds(transcript.words(), MAIN_LO, MAIN_HI)
        assert words[0].start == pytest.approx(FIRST_MAIN_WORD_START)
        assert words[-1].end == pytest.approx(LAST_MAIN_WORD_END)
        assert all(MAIN_LO <= w.start < MAIN_HI for w in words)

    def test_範囲は半開区間(self):
        words = [W("あ", 1.0, 2.0), W("い", 2.0, 3.0), W("う", 3.0, 4.0)]
        got = s5.words_in_bounds(words, 2.0, 3.0)
        assert [w.word for w in got] == ["い"]

    def test_本編の外の発話は入らない(self, transcript):
        """アンカーBの直後（エンディング）の「木原」はハイライトの材料にならない。"""
        text = "".join(w.word for w in s5.words_in_bounds(transcript.words(), MAIN_LO, MAIN_HI))
        assert "木原" not in text
        assert "えーっと" not in text


# ===========================================================================
# snap_highlight — SPEC Step 5「採用後の後処理（必須）」
# ===========================================================================


class TestSnapWordBoundary:
    """後処理1: LLM の秒数を最寄りの単語境界へ寄せる。"""

    def test_単語の途中で切られた秒数が単語境界に寄る(self):
        # 1単語＝1文なので、文境界への拡張は何も足さない。
        got, _, _ = s5.snap_highlight(
            cand(1.4, 3.6), STACCATO, bounds=(0.0, 5.0), max_duration=100.0, min_duration=0.0
        )
        assert got.start == pytest.approx(1.0)   # 1.4 は 1.0 の方が近い
        assert got.end == pytest.approx(4.0)     # 3.6 は 4.0 の方が近い

    def test_同距離なら手前の境界を採る(self):
        """語頭が欠けるより少し長い方が安全（SPEC Phase1「語頭が欠けていない」）。"""
        got, _, _ = s5.snap_highlight(
            cand(1.5, 3.5), STACCATO, bounds=(0.0, 5.0), max_duration=100.0, min_duration=0.0
        )
        assert got.start == pytest.approx(1.0)
        assert got.end == pytest.approx(3.0)

    def test_単語の境界そのものは動かない(self):
        got, _, _ = s5.snap_highlight(
            cand(2.0, 4.0), STACCATO, bounds=(0.0, 5.0), max_duration=100.0, min_duration=0.0
        )
        assert got.start == pytest.approx(2.0)
        assert got.end == pytest.approx(4.0)


class TestSnapSentenceBoundary:
    """後処理2: 文の先頭まで前方に、文末まで後方に拡張する（`。？！` で判定）。"""

    def test_文の途中から始まる指定は文頭まで戻る(self):
        got, _, _ = s5.snap_highlight(
            cand(1.4, 4.6), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        # 1文目の頭（0.0）から、4.6 が属する2文目の末尾（6.0）まで。
        assert got.start == pytest.approx(0.0)
        assert got.end == pytest.approx(6.0)

    def test_文の内側に収まる指定はその文ちょうどになる(self):
        got, _, _ = s5.snap_highlight(
            cand(3.2, 4.2), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))

    def test_疑問符と感嘆符も文末として扱う(self):
        """SPEC Step 5「`。？！` で判定」。? と ! が効かないと語尾が千切れる。"""
        got, _, _ = s5.snap_highlight(
            cand(3.2, 4.2), QA_WORDS, bounds=(0.0, 7.0), max_duration=100.0, min_duration=0.0
        )
        assert got.start == pytest.approx(3.0)   # 直前の「ですか？」で文が切れている
        assert got.end == pytest.approx(5.0)     # 「こうです！」で文が終わる

    def test_小数点は文末ではない(self):
        """「3.5倍」の半角ピリオドで文を切ると、文頭が「倍になりました。」になってしまう。"""
        words = [
            W("要約の", 0.0, 1.0),
            W("精度は", 1.0, 2.0),
            W("3.5", 2.0, 3.0),
            W("倍に", 3.0, 4.0),
            W("なりました。", 4.0, 5.0),
        ]
        got, _, _ = s5.snap_highlight(
            cand(3.2, 3.8), words, bounds=(0.0, 5.0), max_duration=100.0, min_duration=0.0
        )
        assert got.start == pytest.approx(0.0)
        assert got.end == pytest.approx(5.0)

    def test_文末の直後で終わる指定は次の文へ伸びない(self):
        got, _, _ = s5.snap_highlight(
            cand(0.0, 3.0), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert got.end == pytest.approx(3.0)


class TestSnapTrim:
    """後処理4: max_duration を超えたら末尾の文を1つずつ落とし、落とした数を返す。"""

    def test_上限内なら1文も落とさない(self):
        got, _, trimmed = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(0.0), pytest.approx(8.0))
        assert trimmed == 0

    def test_1文落とせば収まるなら1文だけ落とす(self):
        got, _, trimmed = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=7.0, min_duration=0.0
        )
        assert got.end == pytest.approx(6.0)     # 3文目（6.0〜8.0）が落ちた
        assert trimmed == 1
        assert got.duration <= 7.0

    def test_足りなければ2文落とす(self):
        got, _, trimmed = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=4.0, min_duration=0.0
        )
        assert got.end == pytest.approx(3.0)     # 1文目だけが残る
        assert trimmed == 2
        assert got.duration <= 4.0

    def test_落とすのは常に末尾で先頭は動かない(self):
        got, _, trimmed = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=4.0, min_duration=0.0
        )
        assert got.start == pytest.approx(0.0)
        assert trimmed == 2

    def test_これ以上落とせなければ上限超過のまま返す(self):
        """1文しか残っていないのに超過。無理に単語で切ると語尾が千切れるので、超過を許して返す。"""
        got, _, trimmed = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=2.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(0.0), pytest.approx(3.0))
        assert trimmed == 2
        assert got.duration > 2.0

    def test_文が1つしかなく上限を超えても無限ループしない(self):
        """落とせる文が無いことを1回で見切ること（ループ回数はスナップ呼び出し回数で数える）。"""
        lookup = make_recording_lookup()
        got, _, trimmed = s5.snap_highlight(
            cand(0.0, 6.0),
            ONE_SENTENCE_NO_PERIOD,
            bounds=(0.0, 6.0),
            max_duration=0.5,
            min_duration=0.0,
            silence_lookup=lookup,
        )
        assert trimmed == 0
        assert len(lookup.calls) == 2            # start と end を1回ずつ = ループ1周だけ
        assert got.duration > 0.5

    def test_下限を下回っても区間は削られない(self):
        """min_duration は下回っても落とさない（短すぎるかどうかは run() が警告する）。"""
        got, _, trimmed = s5.snap_highlight(
            cand(3.2, 4.2), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=30.0
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))
        assert trimmed == 0


class TestSnapBounds:
    """結果は必ず bounds の中（＝本編の中）に収まること。"""

    def test_文への拡張が本編をはみ出したらクランプされる(self):
        got, _, _ = s5.snap_highlight(
            cand(0.5, 7.5), THREE_SENTENCES, bounds=(0.5, 5.5), max_duration=100.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(0.5), pytest.approx(5.5))

    @pytest.mark.parametrize(
        "start,end",
        [
            (-50.0, 500.0),
            (0.0, 8.0),
            (1.7, 1.9),
            (7.9, 8.0),
            (-1.0, 2.0),
            (5.5, 100.0),
            (2.2, 6.6),
        ],
    )
    @pytest.mark.parametrize("bounds", [(0.0, 8.0), (1.0, 6.5), (2.5, 7.25)])
    def test_どんな候補でも範囲外には出ない(self, start, end, bounds):
        got, _, _ = s5.snap_highlight(
            cand(start, end),
            THREE_SENTENCES,
            bounds=bounds,
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=make_recording_lookup(shift=2.0),
        )
        assert bounds[0] - 1e-9 <= got.start <= bounds[1] + 1e-9
        assert bounds[0] - 1e-9 <= got.end <= bounds[1] + 1e-9
        assert got.start <= got.end

    def test_乱数で振っても必ず有限回で終わり範囲に収まる(self):
        """ASR の出方は読めない。どんな単語列・秒数でも止まり、本編の外へ出ないこと。"""
        rng = random.Random(20260830)
        for _ in range(300):
            n = rng.randint(1, 24)
            words: list[Word] = []
            t = 0.0
            for i in range(n):
                span = rng.uniform(0.05, 1.5)
                # だいたい4語に1語が文末。0語のときもある（句点を落とす ASR）。
                text = f"語{i}" + ("。" if rng.random() < 0.25 else "")
                words.append(W(text, round(t, 3), round(t + span, 3)))
                t += span
            lo = rng.uniform(0.0, t / 2 or 1.0)
            hi = lo + rng.uniform(0.1, t + 2.0)
            lookup = make_recording_lookup(shift=rng.uniform(0.0, 0.5))
            got, _, trimmed = s5.snap_highlight(
                cand(rng.uniform(-5.0, t + 5.0), rng.uniform(-5.0, t + 5.0)),
                words,
                bounds=(lo, hi),
                max_duration=rng.uniform(0.01, t + 1.0),
                min_duration=0.0,
                silence_lookup=lookup,
            )
            assert lo - 1e-9 <= got.start <= got.end <= hi + 1e-9
            assert 0 <= trimmed <= len(words)
            # 1周あたり start/end で2回。打ち切り上限を超えて回っていないこと。
            assert len(lookup.calls) <= 2 * (s5.MAX_TRIM_ITERATIONS + 1)


class TestSnapSilence:
    """後処理3: silence_lookup があれば無音の谷へ寄せる。"""

    def test_単語境界の時刻で呼ばれ結果が反映される(self):
        lookup = make_recording_lookup(shift=0.05)
        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2),
            THREE_SENTENCES,
            bounds=(0.0, 8.0),
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=lookup,
        )
        # 呼ばれたのは「文境界まで拡張したあとの時刻」。LLM の生の秒数ではない。
        assert lookup.calls == [(3.0, "start"), (6.0, "end")]
        assert got.start == pytest.approx(2.95)
        assert got.end == pytest.approx(6.05)
        assert snapped is True

    def test_無音が見つからなければ位置は動かない(self):
        lookup = make_recording_lookup(found=False)
        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2),
            THREE_SENTENCES,
            bounds=(0.0, 8.0),
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=lookup,
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))
        assert snapped is False
        assert lookup.calls == [(3.0, "start"), (6.0, "end")]

    def test_lookup_を渡さなければ呼ばれず寄せもしない(self):
        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2), THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))
        assert snapped is False

    def test_文を落とすたびに新しい終端で引き直す(self):
        """寄せた値を次の計算に持ち込まないこと（持ち込むと backoff が毎周ずれ込む）。"""
        lookup = make_recording_lookup(shift=0.05)
        got, snapped, trimmed = s5.snap_highlight(
            cand(0.5, 7.5),
            THREE_SENTENCES,
            bounds=(0.0, 8.5),
            max_duration=7.0,
            min_duration=0.0,
            silence_lookup=lookup,
        )
        ends = [t for t, kind in lookup.calls if kind == "end"]
        assert ends == [8.0, 6.0]                # 文境界そのもの。8.05 や 6.05 では引かない
        assert got.end == pytest.approx(6.05)
        assert trimmed == 1
        assert snapped is True

    def test_1引数だけの_lookup_も受け付ける(self):
        """契約上の型は Callable[[float], tuple[float, bool]]。kind を取らない実装も通ること。"""

        def simple(t: float) -> tuple[float, bool]:
            return (float(t) + 0.1, True)

        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2),
            THREE_SENTENCES,
            bounds=(0.0, 8.0),
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=simple,
        )
        assert (got.start, got.end) == (pytest.approx(3.1), pytest.approx(6.1))
        assert snapped is True

    def test_区間がつぶれるなら寄せる前の位置に戻す(self):
        """無音へ寄せた結果 end <= start になったら、単語・文境界の位置を使う。"""

        def collapse(t: float, *, kind: str = "start") -> tuple[float, bool]:
            return (4.0, True)

        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2),
            THREE_SENTENCES,
            bounds=(0.0, 8.0),
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=collapse,
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))
        assert snapped is False

    @pytest.mark.parametrize(
        "bad",
        [
            lambda t, *, kind="start": 3.0,                 # タプルでない
            lambda t, *, kind="start": (None, True),        # 数値でない
            lambda t, *, kind="start": (float("nan"), True),
            lambda t, *, kind="start": (1.0, 2.0, 3.0),     # 要素数が違う
        ],
    )
    def test_壊れた_lookup_でも落ちずに元の時刻を使う(self, bad):
        got, snapped, _ = s5.snap_highlight(
            cand(3.2, 4.2),
            THREE_SENTENCES,
            bounds=(0.0, 8.0),
            max_duration=100.0,
            min_duration=0.0,
            silence_lookup=bad,
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(6.0))
        assert snapped is False


class TestSnapMisc:
    def test_score_と_hook_line_と_reason_は書き換えない(self):
        original = cand(3.2, 4.2, score=88.5, hook="実は一番もったいない", reason="逆説がある")
        got, _, _ = s5.snap_highlight(
            original, THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert got.score == 88.5
        assert got.hook_line == "実は一番もったいない"
        assert got.reason == "逆説がある"

    def test_元の候補を破壊しない(self):
        original = cand(3.2, 4.2)
        s5.snap_highlight(
            original, THREE_SENTENCES, bounds=(0.0, 8.0), max_duration=100.0, min_duration=0.0
        )
        assert (original.start, original.end) == (3.2, 4.2)

    def test_単語が無ければクランプするだけ(self):
        got, snapped, trimmed = s5.snap_highlight(
            cand(3.0, 50.0), [], bounds=(0.0, 8.0), max_duration=1.0, min_duration=0.0
        )
        assert (got.start, got.end) == (pytest.approx(3.0), pytest.approx(8.0))
        assert snapped is False
        assert trimmed == 0

    def test_実データでも語頭語尾が文の境界に揃う(self, transcript):
        """合成エピソードの本編で、LLM が文の途中を指しても文単位に整うこと。"""
        main = s5.words_in_bounds(transcript.words(), MAIN_LO, MAIN_HI)
        got, _, trimmed = s5.snap_highlight(
            cand(26.5, 33.0), main, bounds=(MAIN_LO, MAIN_HI), max_duration=45.0, min_duration=20.0
        )
        assert got.start == pytest.approx(25.70, abs=0.01)   # 「実はAIに…」の頭
        assert got.end == pytest.approx(33.60, abs=0.01)     # 「…なんです。」の末尾
        assert trimmed == 0


# ===========================================================================
# check_within_bounds — SPEC 9章「ハイライト候補が本編の範囲外 → 破棄」
# ===========================================================================


class TestCheckWithinBounds:
    BOUNDS = (100.0, 200.0)

    def test_範囲内なら_None(self):
        assert s5.check_within_bounds(cand(120.0, 150.0), self.BOUNDS) is None

    def test_開始が本編より前なら理由を返す(self):
        reason = s5.check_within_bounds(cand(50.0, 150.0), self.BOUNDS)
        assert reason is not None and "開始" in reason

    def test_終了が本編より後なら理由を返す(self):
        reason = s5.check_within_bounds(cand(150.0, 900.0), self.BOUNDS)
        assert reason is not None and "終了" in reason

    def test_終了が開始以前なら理由を返す(self):
        assert s5.check_within_bounds(cand(150.0, 150.0), self.BOUNDS) is not None
        assert s5.check_within_bounds(cand(150.0, 140.0), self.BOUNDS) is not None

    def test_数値として不正なら理由を返す(self):
        assert s5.check_within_bounds(cand(float("nan"), 150.0), self.BOUNDS) is not None
        assert s5.check_within_bounds(cand(120.0, math.inf), self.BOUNDS) is not None

    def test_丸め誤差程度のはみ出しは破棄しない(self):
        """LLM は小数1桁で返す。許容内のはみ出しは破棄せずクランプに任せる。"""
        eps = s5.BOUNDS_TOLERANCE_SEC / 2.0
        assert s5.check_within_bounds(cand(100.0 - eps, 200.0 + eps), self.BOUNDS) is None

    def test_許容を超えたはみ出しは破棄する(self):
        over = s5.BOUNDS_TOLERANCE_SEC * 2.0
        assert s5.check_within_bounds(cand(100.0 - over, 150.0), self.BOUNDS) is not None
        assert s5.check_within_bounds(cand(120.0, 200.0 + over), self.BOUNDS) is not None


# ===========================================================================
# parse_candidates — LLM 応答の取り出し
# ===========================================================================


class TestParseCandidates:
    def test_score_の降順に並べ替える(self):
        payload = {
            "candidates": [
                raw_candidate(10.0, 20.0, 70),
                raw_candidate(30.0, 40.0, 95),
                raw_candidate(50.0, 60.0, 85),
            ]
        }
        got = s5.parse_candidates(payload)
        assert [c.score for c in got] == [95.0, 85.0, 70.0]

    def test_同点は_LLM_が返した順を保つ(self):
        payload = {
            "candidates": [
                raw_candidate(10.0, 20.0, 80, hook="先"),
                raw_candidate(30.0, 40.0, 80, hook="後"),
            ]
        }
        assert [c.hook_line for c in s5.parse_candidates(payload)] == ["先", "後"]

    def test_候補が0件なら_HighlightError(self):
        with pytest.raises(HighlightError):
            s5.parse_candidates({"candidates": []})

    def test_candidates_が無ければ_HighlightError(self):
        with pytest.raises(HighlightError):
            s5.parse_candidates({})

    def test_候補がオブジェクトでなければ_HighlightError(self):
        with pytest.raises(HighlightError):
            s5.parse_candidates({"candidates": ["842.5〜871.2"]})

    def test_必須キーが欠けていれば_HighlightError(self):
        with pytest.raises(HighlightError):
            s5.parse_candidates({"candidates": [{"start": 1.0, "score": 90}]})


# ===========================================================================
# make_silence_lookup — Step 4 と同じ silencedetect を前後 1.5 秒に
# ===========================================================================


class TestMakeSilenceLookup:
    """detect_silences を差し替えて、寄せ方の規則だけを見る。"""

    @staticmethod
    def _patch(monkeypatch, spans, seen=None):
        def fake(wav, *, start, end, noise_db, min_duration):
            if seen is not None:
                seen.append({"start": start, "end": end, "noise_db": noise_db, "min_duration": min_duration})
            return list(spans)

        monkeypatch.setattr(s5, "detect_silences", fake)

    def test_探索窓は前後15秒(self, monkeypatch):
        seen: list[dict] = []
        self._patch(monkeypatch, [], seen)
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        assert lookup(10.0, kind="start") == (10.0, False)
        assert seen == [
            {
                "start": 10.0 - s5.SILENCE_WINDOW_SEC,
                "end": 10.0 + s5.SILENCE_WINDOW_SEC,
                "noise_db": -32.0,
                "min_duration": 0.12,
            }
        ]

    def test_start_は直前の無音の終わりから_backoff_だけ手前(self, monkeypatch):
        self._patch(monkeypatch, [(9.0, 9.5)])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        snapped, found = lookup(10.0, kind="start")
        assert found is True
        assert snapped == pytest.approx(9.5 - SILENCE_BACKOFF_SEC)

    def test_end_は直後の無音の始まりから_backoff_だけ後ろ(self, monkeypatch):
        """語尾を切り落とさないよう、end だけ無音の始まりより後ろに置く。"""
        self._patch(monkeypatch, [(10.2, 10.6)])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        snapped, found = lookup(10.0, kind="end")
        assert found is True
        assert snapped == pytest.approx(10.2 + SILENCE_BACKOFF_SEC)

    def test_無音が無ければ動かさない(self, monkeypatch):
        self._patch(monkeypatch, [])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        assert lookup(10.0, kind="start") == (10.0, False)
        assert lookup(10.0, kind="end") == (10.0, False)

    def test_遠すぎる無音には寄せない(self, monkeypatch):
        """隣の文へ食い込む移動は採らない。"""
        far = 10.0 - s5.MAX_SILENCE_SHIFT_SEC - 1.0
        self._patch(monkeypatch, [(far - 0.3, far)])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        assert lookup(10.0, kind="start") == (10.0, False)

    def test_わずかに跨いだ無音は拾う(self, monkeypatch):
        """単語境界と silencedetect の境界は数十msずれる。厳密な不等号だと直近の谷を落とす。"""
        self._patch(monkeypatch, [(9.5, 10.0 + s5.SILENCE_MATCH_JITTER_SEC / 2.0)])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        snapped, found = lookup(10.0, kind="start")
        assert found is True
        assert snapped > 10.0 - SILENCE_BACKOFF_SEC

    def test_同じ時刻は引き直さない(self, monkeypatch):
        seen: list[dict] = []
        self._patch(monkeypatch, [(9.0, 9.5)], seen)
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        first = lookup(10.0, kind="start")
        second = lookup(10.0, kind="start")
        assert first == second
        assert len(seen) == 1
        lookup(10.0, kind="end")            # 向きが違えばキャッシュは別
        assert len(seen) == 2

    def test_負の時刻は返さない(self, monkeypatch):
        self._patch(monkeypatch, [(0.0, 0.02)])
        lookup = s5.make_silence_lookup("dummy.wav", noise_db=-32.0, min_duration=0.12)
        snapped, _ = lookup(0.03, kind="start")
        assert snapped >= 0.0


# ===========================================================================
# run — SPEC Step 5 本体
# ===========================================================================


class TestRunSelection:
    def test_範囲外を捨てて_score_最上位の残りを採用する(self, ctx, transcript, cuts, stub_llm):
        """SPEC 9章「本編の範囲外 → その候補を破棄し、次点を採用」。

        スタブは score 95（999〜1020秒＝本編外）・92・80 を返す。採用は 92。
        """
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert result.selected.score == 92.0
        assert result.selected.hook_line.startswith("実はAIに議事録を")

    def test_採用した候補は3段スナップ後の秒数になる(self, ctx, transcript, cuts, stub_llm):
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert result.selected.start == pytest.approx(25.70, abs=0.01)
        assert result.selected.end == pytest.approx(33.60, abs=0.01)
        assert result.trimmed_sentences == 0

    def test_snapped_from_に_LLM_の生の秒数が残る(self, ctx, transcript, cuts, stub_llm):
        """SPEC 8章 decisions.json の highlight.snapped_from。何を動かしたか追えること。"""
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert result.snapped_from.start == pytest.approx(26.5)
        assert result.snapped_from.end == pytest.approx(33.0)

    def test_採用区間は本編に収まる(self, ctx, transcript, cuts, stub_llm):
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert MAIN_LO <= result.selected.start < result.selected.end <= MAIN_HI

    def test_破棄した候補は警告に残る(self, ctx, transcript, cuts, stub_llm):
        """SPEC 8章 warnings。あとから「なぜこれを採ったのか」を追えること。"""
        s5.run(ctx, transcript, cuts, stub_llm)
        assert any("破棄" in w and "999" in w for w in ctx.warnings)

    def test_採用しなかった候補は_alternatives_に残る(self, ctx, transcript, cuts, stub_llm):
        """SPEC Step 5「残り2つは decisions.json に残し、差し替え候補にする」。"""
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert len(result.alternatives) == 2
        assert sorted(c.score for c in result.alternatives) == [80.0, 95.0]
        assert all(c.score != 92.0 for c in result.alternatives)

    def test_全候補が範囲外なら_HighlightError(self, ctx, transcript, cuts):
        """SPEC 9章「全滅なら停止」。勝手に本編の外を切らないこと。"""
        llm = highlight_stub(
            [
                raw_candidate(900.0, 930.0, 95),
                raw_candidate(1000.0, 1030.0, 90),
                raw_candidate(0.5, 3.0, 85),
            ]
        )
        with pytest.raises(HighlightError) as excinfo:
            s5.run(ctx, transcript, cuts, llm)
        message = str(excinfo.value)
        assert "43.9" in message                  # 本編の範囲を示すこと
        assert message.count("- ") >= 3           # 落ちた理由が候補ごとに並ぶこと

    def test_採用は_score_順で先頭から(self, ctx, transcript, cuts):
        """本編内の候補が複数あれば、必ずスコア最上位が採用されること。"""
        llm = highlight_stub(
            [
                raw_candidate(26.5, 33.0, 70, hook="低い方"),
                raw_candidate(34.5, 39.0, 88, hook="高い方"),
            ]
        )
        result = s5.run(ctx, transcript, cuts, llm)
        assert result.selected.hook_line == "高い方"
        assert [c.hook_line for c in result.alternatives] == ["低い方"]

    def test_本編の頭を指してもアンカーAより前には出ない(self, ctx, transcript, cuts):
        """文の先頭への拡張がオープニング側へ食い込まないこと（Phase1 の受け入れ基準）。"""
        llm = highlight_stub([raw_candidate(6.2, 10.0, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert result.selected.start >= MAIN_LO
        assert result.selected.start == pytest.approx(FIRST_MAIN_WORD_START, abs=0.01)

    def test_本編の末尾を指してもアンカーBを越えない(self, ctx, transcript, cuts):
        llm = highlight_stub([raw_candidate(41.0, 43.5, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert result.selected.end <= MAIN_HI
        assert result.selected.end == pytest.approx(LAST_MAIN_WORD_END, abs=0.01)

    def test_丸め誤差程度のはみ出しは破棄せずクランプする(self, ctx, transcript, cuts):
        eps = s5.BOUNDS_TOLERANCE_SEC / 2.0
        llm = highlight_stub([raw_candidate(MAIN_LO - eps, MAIN_HI + eps, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert MAIN_LO <= result.selected.start < result.selected.end <= MAIN_HI

    def test_尺が上限を超えたら末尾の文が落ちる(self, ctx, transcript, cuts):
        """本編ほぼ全体を指す候補は、max_duration_sec まで末尾の文を落として詰める。"""
        # config フィクスチャはセッション共有なので、必ず複製してから尺だけ差し替える。
        ctx.config = copy.deepcopy(ctx.config)
        ctx.config.highlight = dataclasses.replace(
            ctx.config.highlight,
            min_duration_sec=5.0,
            target_duration_sec=10.0,
            max_duration_sec=20.0,
        )
        llm = highlight_stub([raw_candidate(6.0, 43.5, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert result.trimmed_sentences >= 1
        assert result.selected.duration <= 20.0
        assert result.selected.start == pytest.approx(FIRST_MAIN_WORD_START, abs=0.01)

    def test_短すぎる採用は警告に残る(self, ctx, transcript, cuts, stub_llm):
        """採用区間は 7.9 秒。min_duration_sec は 20 秒。"""
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert result.selected.duration < ctx.config.highlight.min_duration_sec
        assert any("下回" in w for w in ctx.warnings)

    def test_探す範囲は_source_segment_で決まる(self, ctx, transcript, cuts):
        """SPEC 5章「アンカー語をコードにハードコードしないこと」。

        highlight.source_segment を "ending" にしたら、探索範囲は B〜終端に移ること
        （A〜B が決め打ちになっていないこと）。
        """
        ctx.config = copy.deepcopy(ctx.config)
        ctx.config.highlight = dataclasses.replace(
            ctx.config.highlight, source_segment="ending", min_duration_sec=5.0, target_duration_sec=6.0
        )
        llm = highlight_stub([raw_candidate(45.0, 50.0, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert MAIN_HI <= result.selected.start < result.selected.end <= transcript.duration
        assert "木原" in llm.calls[0]["prompt"]          # エンディングの発話が渡っている
        assert "このチャンネルは" not in llm.calls[0]["prompt"]

    def test_本編に単語が無ければ_HighlightError(self, ctx, transcript, cuts):
        empty_cuts = {
            "A": CutPoint("A", 58.0, 58.0, True, 100.0),
            "B": CutPoint("B", 59.0, 59.0, True, 96.0),
        }
        with pytest.raises(HighlightError):
            s5.run(ctx, transcript, empty_cuts, highlight_stub([raw_candidate(58.1, 58.9, 90)]))


class TestRunPrompt:
    """SPEC Step 5「本編区間の文字起こしのみをLLMに渡す」／11章「プロンプトは .md に外出し」。"""

    def test_プロンプトは_llm_prompts_highlight_md_から作る(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        assert len(stub_llm.calls) == 1
        call = stub_llm.calls[0]
        assert call["step"] == "highlight"
        assert "ハイライト区間の選定" in call["prompt"]     # highlight.md の見出し

    def test_プロンプトには本編の外の発話を入れない(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        prompt = stub_llm.calls[0]["prompt"]
        assert "このチャンネルは" in prompt          # 本編の頭は入る
        assert "木原" not in prompt                  # エンディングは入らない
        assert "また次回" not in prompt
        assert "えーっと" not in prompt              # オープニングも入らない

    def test_プロンプトに本編の範囲と尺の条件が埋まる(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        prompt = stub_llm.calls[0]["prompt"]
        assert "5.9 秒 〜 43.9 秒" in prompt
        assert "20 秒以上 45 秒以下" in prompt         # min / max（%g で整形）
        assert "3 個選び出してください" in prompt      # 候補は3つ（SPEC Step 5）
        assert "{{" not in prompt                     # 埋め残しが無いこと

    def test_文字起こしの行は開始秒つき(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        lines = [l for l in stub_llm.calls[0]["prompt"].splitlines() if l.startswith("[")]
        assert lines
        for line in lines:
            head, _, body = line.partition("] ")
            assert MAIN_LO - 0.1 <= float(head[1:]) <= MAIN_HI
            assert body


class TestFormatTranscriptForPrompt:
    def test_行は文末で折り返す(self):
        words = [
            W("いち", 0.0, 4.0),
            W("にです。", 4.0, 8.0),
            W("さん", 8.0, 11.0),
            W("しです。", 11.0, 14.0),
            W("ごです。", 14.0, 18.0),
        ]
        text = s5.format_transcript_for_prompt(words, min_line_sec=10.0, max_line_sec=20.0)
        assert text.splitlines() == ["[0.0] いちにです。さんしです。", "[14.0] ごです。"]

    def test_文末が来なくても上限秒で折り返す(self):
        words = [W(f"語{i}", float(i), float(i + 1)) for i in range(30)]
        lines = s5.format_transcript_for_prompt(words, min_line_sec=10.0, max_line_sec=12.0).splitlines()
        assert len(lines) >= 2
        assert lines[0].startswith("[0.0] ")

    def test_単語が無ければ空文字(self):
        assert s5.format_transcript_for_prompt([]) == ""

    def test_最後の端数も落とさない(self):
        words = [W("あ。", 0.0, 1.0), W("い。", 1.0, 2.0)]
        text = s5.format_transcript_for_prompt(words, min_line_sec=10.0, max_line_sec=20.0)
        assert text == "[0.0] あ。い。"


class TestRunLlmRecording:
    """SPEC 8章 decisions.json の llm_calls。"""

    def test_成功した呼び出しが記録される(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        assert len(ctx.llm_calls) == 1
        record = ctx.llm_calls[0]
        assert record.step == "highlight"
        assert record.model == "stub-model"
        assert record.ok is True
        assert record.retries == 0

    def test_LLM_が失敗したら握りつぶさず止まる(self, ctx, transcript, cuts):
        """SPEC 9章「握りつぶさない」。スタブに highlight の応答が無いので LlmError。"""
        llm = StubLlmClient({}, model="stub-model")
        with pytest.raises(RadioCutterError) as excinfo:
            s5.run(ctx, transcript, cuts, llm)
        # _call_llm の契約: 失敗も記録してから投げ直す（例外型は変えない）。
        assert isinstance(excinfo.value, LlmError)

    def test_失敗した呼び出しも記録される(self, ctx, transcript, cuts):
        llm = StubLlmClient({}, model="stub-model")
        with pytest.raises(RadioCutterError):
            s5.run(ctx, transcript, cuts, llm)
        assert len(ctx.llm_calls) == 1
        assert ctx.llm_calls[0].ok is False
        assert ctx.llm_calls[0].step == "highlight"
        assert ctx.llm_calls[0].error

    def test_LLM_が失敗したら_highlight_json_は書かれない(self, ctx, transcript, cuts):
        llm = StubLlmClient({}, model="stub-model")
        with pytest.raises(RadioCutterError):
            s5.run(ctx, transcript, cuts, llm)
        assert not s5.highlight_path(ctx).exists()


# ===========================================================================
# save / load — --from-step での再開
# ===========================================================================


class TestSaveLoad:
    def test_highlight_json_が書かれる(self, ctx, transcript, cuts, stub_llm):
        s5.run(ctx, transcript, cuts, stub_llm)
        path = s5.highlight_path(ctx)
        assert path == ctx.work_path("highlight.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) >= {"selected", "snapped_from", "alternatives", "silence_snapped", "trimmed_sentences"}

    def test_load_で復元できる(self, ctx, transcript, cuts, stub_llm):
        result = s5.run(ctx, transcript, cuts, stub_llm)
        loaded = s5.load(ctx)
        assert loaded.selected.start == pytest.approx(result.selected.start)
        assert loaded.selected.end == pytest.approx(result.selected.end)
        assert loaded.selected.score == result.selected.score
        assert loaded.selected.hook_line == result.selected.hook_line
        assert loaded.selected.reason == result.selected.reason
        assert loaded.snapped_from.start == pytest.approx(result.snapped_from.start)
        assert loaded.snapped_from.end == pytest.approx(result.snapped_from.end)
        assert [c.score for c in loaded.alternatives] == [c.score for c in result.alternatives]
        assert loaded.silence_snapped == result.silence_snapped
        assert loaded.trimmed_sentences == result.trimmed_sentences

    def test_中間ファイルが無ければ_MissingArtifactError(self, ctx):
        with pytest.raises(MissingArtifactError):
            s5.load(ctx)

    def test_JSON_が壊れていれば_MissingArtifactError(self, ctx):
        path = s5.highlight_path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ここから壊れている", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s5.load(ctx)

    def test_形が違えば_MissingArtifactError(self, ctx):
        path = s5.highlight_path(ctx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"alternatives": []}), encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s5.load(ctx)

    def test_秒数は小数3桁に丸めて保存する(self, ctx, transcript, cuts, stub_llm):
        """SPEC 11章「秒数は全て float（小数点以下3桁）」。"""
        s5.run(ctx, transcript, cuts, stub_llm)
        data = json.loads(s5.highlight_path(ctx).read_text(encoding="utf-8"))
        for key in ("start", "end"):
            value = data["selected"][key]
            assert round(value, 3) == value


# ===========================================================================
# 無音スナップの実行（ffmpeg が要る）
# ===========================================================================


class TestRunSilenceSnap:
    def test_音声が無ければ無音スナップを飛ばして警告する(self, ctx, transcript, cuts, stub_llm):
        assert not s5.audio_path(ctx).exists()
        result = s5.run(ctx, transcript, cuts, stub_llm)
        assert result.silence_snapped is False
        assert any("音声ファイルが無い" in w for w in ctx.warnings)
        assert any("無音が見つかりませんでした" in w for w in ctx.warnings)

    @pytest.mark.ffmpeg
    @requires_ffmpeg
    def test_音声があれば無音の谷へ寄る(self, ctx, transcript, cuts, stub_llm, episode_wav: Path):
        """SPEC Step 5 後処理3。語頭・語尾を残す向き（start は手前・end は後ろ）に動くこと。

        合成エピソードでは 25.40〜25.70 と 33.60〜33.90 が無音。
        文境界 25.70 / 33.60 は backoff 50ms だけ外側へ広がるはず。
        """
        ctx.ensure_dirs()
        shutil.copyfile(episode_wav, s5.audio_path(ctx))

        result = s5.run(ctx, transcript, cuts, stub_llm)

        assert result.silence_snapped is True
        assert result.selected.start == pytest.approx(25.70 - SILENCE_BACKOFF_SEC, abs=0.05)
        assert result.selected.end == pytest.approx(33.60 + SILENCE_BACKOFF_SEC, abs=0.05)
        # 無音へ寄せた結果、文の切れ目より外側に出ていること（語頭・語尾が残る）
        assert result.selected.start < 25.70
        assert result.selected.end > 33.60
        assert not any("無音が見つかりませんでした" in w for w in ctx.warnings)
        assert not any("音声ファイルが無い" in w for w in ctx.warnings)

    @pytest.mark.ffmpeg
    @requires_ffmpeg
    def test_無音へ寄せても本編からは出ない(self, ctx, transcript, cuts, episode_wav: Path):
        """アンカーB直前の文を採ると end は 43.90 の壁に当たる。越えないこと。"""
        ctx.ensure_dirs()
        shutil.copyfile(episode_wav, s5.audio_path(ctx))
        llm = highlight_stub([raw_candidate(41.0, 43.5, 90)])
        result = s5.run(ctx, transcript, cuts, llm)
        assert MAIN_LO <= result.selected.start < result.selected.end <= MAIN_HI


# ===========================================================================
# HighlightResult の往復（decisions.json / highlight.json の値）
# ===========================================================================


class TestHighlightResultRoundtrip:
    def test_dict_往復で値が保たれる(self):
        result = HighlightResult(
            selected=cand(10.0, 40.0, score=92.0, hook="フック", reason="理由"),
            snapped_from=cand(12.5, 41.2, score=92.0),
            alternatives=[cand(100.0, 130.0, score=85.0), cand(200.0, 230.0, score=70.0)],
            silence_snapped=True,
            trimmed_sentences=2,
        )
        again = HighlightResult.from_dict(result.to_dict())
        assert (again.selected.start, again.selected.end) == (10.0, 40.0)
        assert again.selected.hook_line == "フック"
        assert (again.snapped_from.start, again.snapped_from.end) == (12.5, 41.2)
        assert [c.score for c in again.alternatives] == [85.0, 70.0]
        assert again.silence_snapped is True
        assert again.trimmed_sentences == 2
        assert again.duration == pytest.approx(30.0)
