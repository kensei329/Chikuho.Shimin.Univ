"""steps/s3_find_anchors.py — アンカー検出（SPEC Step 3）。

SPEC が「ここが最重要かつ最も壊れやすい箇所」と名指ししている工程。
このファイルで守らせたいのは次の4つ。

1. 決まり文句を**正規化後の**あいまい一致で拾い、しきい値の上下で
   ちゃんと拾う／拾わないが切り替わること（SPEC Step 3-2, 3-3）。
2. 探索窓と `must_follow` で「それらしいだけの偽物」を落とすこと。
   合成エピソードの 19.95 秒には「ということで」のおとりが仕込んである（SPEC Step 3-4）。
3. 確定した候補の `raw_cut_time` が**語頭の単語 start** に一致すること。
   ここがずれると 02_main.mp4 の冒頭に前の文の尻尾が混ざる（SPEC Step 3-5 / Phase 1 受け入れ基準）。
4. 見つからないときは黙って代替位置を選ばず、次の一手が分かるメッセージで止まること
   （SPEC Step 3「失敗時の挙動」／SPEC 9章）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re

import pytest
from rapidfuzz import fuzz

import fixtures
from radio_cutter.config import AnchorConfig, Config
from radio_cutter.context import RunContext
from radio_cutter.errors import AnchorNotFoundError, AnchorOrderError, MissingArtifactError
from radio_cutter.models import AnchorCandidate, AnchorResult, Transcript, Word
from radio_cutter.steps import s3_find_anchors as s3
from radio_cutter.util.text_normalize import FlatText, build_flat, normalize_phrase

PHRASE_A = "このチャンネルは"
PHRASE_B = "ということで"

#: SPEC Step 3 の出力例が持つキー（work/anchors.json の契約）
SPEC_ANCHOR_KEYS = {
    "phrase",
    "matched_text",
    "score",
    "raw_cut_time",
    "candidates_found",
    "candidates_rejected",
    "context",
}


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------


def anchor(**overrides) -> AnchorConfig:
    """テスト用の AnchorConfig。設定ファイルと同じ経路（from_dict）で作る。"""
    d: dict = {"id": "A", "phrase": PHRASE_A, "fuzzy_threshold": 0.85}
    d.update(overrides)
    return AnchorConfig.from_dict(d)


def flat_of(transcript: Transcript) -> FlatText:
    return build_flat(transcript.words())


def flat_from(*specs: tuple[str, float, float]) -> FlatText:
    """(語, start, end) の並びから小さな FlatText を作る。"""
    return build_flat([Word(word=w, start=s, end=e) for w, s, e in specs])


def word_span_of(
    transcript: Transcript, phrase: str, *, occurrence: int = 0
) -> tuple[float, float]:
    """生テキスト上の phrase から「先頭単語の start」と「末尾単語の end」を返す。

    s3 の実装を一切使わずに期待値を出すための独立計算。
    テストが実装の写経にならないようにするのが目的。
    """
    words = transcript.words()
    flat = "".join(w.word for w in words)
    owner: list[int] = []
    for i, w in enumerate(words):
        owner.extend([i] * len(w.word))
    pos = -1
    for _ in range(occurrence + 1):
        pos = flat.find(phrase, pos + 1)
        assert pos >= 0, f"フィクスチャに phrase がありません: {phrase!r}"
    return (words[owner[pos]].start, words[owner[pos + len(phrase) - 1]].end)


def make_candidate(
    score: float,
    start_time: float,
    *,
    end_time: float | None = None,
    norm_start: int = 0,
    matched_text: str = PHRASE_B,
    context: str = "",
    rejected_reason: str | None = None,
) -> AnchorCandidate:
    """select / build_error_message を直接叩くための候補。"""
    end = start_time + 1.0 if end_time is None else end_time
    return AnchorCandidate(
        score=score,
        norm_start=norm_start,
        norm_end=norm_start + len(matched_text),
        word_start=0,
        word_end=1,
        start_time=start_time,
        end_time=end,
        matched_text=matched_text,
        context=context or f"…{matched_text}の前後…",
        rejected_reason=rejected_reason,
    )


def anchors_json(config: Config) -> list[dict]:
    """config/ai-radio.json の anchors を書き換え可能な形で取り出す。"""
    return copy.deepcopy(config.raw["anchors"])


def config_with(config: Config, anchors: list[dict]) -> Config:
    """anchors だけ差し替えた Config を作る（セッション共有の config は壊さない）。"""
    raw = copy.deepcopy(config.raw)
    raw["anchors"] = anchors
    return Config.from_dict(raw)


def ctx_with(ctx: RunContext, config: Config) -> RunContext:
    return dataclasses.replace(ctx, config=config)


def window_scores(flat: FlatText, phrase: str) -> list[float]:
    """しきい値なしで全窓のスコアを自前で出す（SPEC 指定の rapidfuzz.fuzz.ratio）。"""
    norm_phrase = normalize_phrase(phrase)
    m = len(norm_phrase)
    return [
        float(fuzz.ratio(flat.norm[i : i + m], norm_phrase))
        for i in range(len(flat.norm) - m + 1)
    ]


def error_ranks(message: str) -> list[tuple[int, float]]:
    """エラーメッセージ中の「N. スコア X」行を (順位, スコア) で拾う。"""
    return [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(r"(\d+)\.\s*スコア\s*([0-9.]+)", message)
    ]


def assert_gives_next_action(message: str) -> None:
    """SPEC Step 3「失敗時の挙動」が求める案内が入っているか。"""
    assert "下げ" in message, f"しきい値を下げる案内がありません:\n{message}"
    assert ("fuzzy_threshold" in message) or ("しきい値" in message), message
    assert ("phrase" in message) or ("フレーズ" in message), message


# ---------------------------------------------------------------------------
# find_candidates（SPEC Step 3-3 あいまい一致）
# ---------------------------------------------------------------------------


class TestFindCandidates:
    def test_完全一致はスコア100で語頭の単語に紐づく(self, transcript: Transcript) -> None:
        """完全一致は 100 点。時刻は「こ」を含む単語の start（SPEC Step 3-5）。"""
        flat = flat_of(transcript)
        got = s3.find_candidates(flat, PHRASE_A, 82.0)

        assert len(got) == 1
        c = got[0]
        assert c.score == pytest.approx(100.0)
        assert c.matched_text == PHRASE_A
        start, end = word_span_of(transcript, PHRASE_A)
        assert c.start_time == pytest.approx(start)
        assert c.end_time == pytest.approx(end)
        assert c.start_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)
        assert c.rejected_reason is None

    def test_しきい値100でも完全一致は拾える(self, transcript: Transcript) -> None:
        """しきい値は「以上」。config は fuzzy_threshold=1.0 を許すので、
        「超えた」判定にすると 1.0 の設定が永久に何も拾えなくなる。"""
        got = s3.find_candidates(flat_of(transcript), PHRASE_A, 100.0)
        assert [c.score for c in got] == [pytest.approx(100.0)]

    def test_一文字違いの想定スコアは87_5点(self, transcript: Transcript) -> None:
        """以降のしきい値テストの前提を明示しておく（8文字中1文字違い → 87.5）。"""
        assert fuzz.ratio(PHRASE_A, normalize_phrase("このチャンネルわ")) == pytest.approx(87.5)

    @pytest.mark.parametrize(
        "threshold,expected_hit",
        [
            (82.0, True),   # config/ai-radio.json のアンカーA と同じしきい値
            (87.5, True),   # 境界。ちょうど同点なら拾う
            (88.0, False),
            (90.0, False),
        ],
    )
    def test_一文字違いはしきい値次第で拾える(
        self, transcript: Transcript, threshold: float, expected_hit: bool
    ) -> None:
        """1文字の表記ゆれ（は/わ）は 87.5 点。しきい値の上下で挙動が切り替わること。"""
        got = s3.find_candidates(flat_of(transcript), "このチャンネルわ", threshold)
        assert bool(got) is expected_hit

    def test_かけ離れたフレーズは拾わない(self, transcript: Transcript) -> None:
        """しきい値未満は候補にしない。勝手に「近いもの」を拾わせない。"""
        assert s3.find_candidates(flat_of(transcript), "宇宙船の操縦マニュアル", 82.0) == []

    def test_空白と句読点は正規化で無視される(self, transcript: Transcript) -> None:
        """SPEC Step 3-2。config の phrase に空白や句点が混じっても同じ位置を指す。"""
        got = s3.find_candidates(flat_of(transcript), "この チャンネル は。", 82.0)
        assert len(got) == 1
        assert got[0].score == pytest.approx(100.0)
        assert got[0].matched_text == PHRASE_A
        assert got[0].start_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)

    def test_全角半角の揺れはNFKCで吸収される(self, transcript: Transcript) -> None:
        """SPEC Step 3-2「全角/半角の統一（NFKC正規化）」。"""
        got = s3.find_candidates(flat_of(transcript), "ＡＩの活用法", 90.0)
        assert len(got) == 1
        assert got[0].score == pytest.approx(100.0)
        assert got[0].matched_text == "AIの活用法"

    def test_候補が複数あるときは全部返る(self, transcript: Transcript) -> None:
        """おとり（19.95秒）と本物（43.95秒）の両方が候補として上がること。
        絞り込みは filter_candidates の仕事で、ここで間引いてはいけない。"""
        got = s3.find_candidates(flat_of(transcript), PHRASE_B, 85.0)
        starts = [c.start_time for c in got]
        assert starts == [pytest.approx(19.95), pytest.approx(fixtures.EXPECTED_ANCHOR_B_RAW)]
        assert all(c.score == pytest.approx(100.0) for c in got)

    def test_返り値は出現順に並ぶ(self, transcript: Transcript) -> None:
        got = s3.find_candidates(flat_of(transcript), PHRASE_B, 60.0)
        assert [c.norm_start for c in got] == sorted(c.norm_start for c in got)
        assert [c.start_time for c in got] == sorted(c.start_time for c in got)

    def test_空のflatでも落ちない(self) -> None:
        """Step 2 が空の文字起こしを返しても、ここで例外を投げずに候補0件とする。"""
        assert s3.find_candidates(build_flat([]), PHRASE_A, 82.0) == []

    def test_flatがフレーズより短くても落ちない(self) -> None:
        """窓が作れないケース。スライスの端で落ちないこと。"""
        assert s3.find_candidates(flat_from(("は", 0.0, 0.5)), PHRASE_A, 82.0) == []

    def test_フレーズと同じ長さのflatでも動く(self) -> None:
        flat = flat_from((PHRASE_A, 1.0, 2.0))
        got = s3.find_candidates(flat, PHRASE_A, 82.0)
        assert len(got) == 1
        assert got[0].start_time == pytest.approx(1.0)
        assert got[0].end_time == pytest.approx(2.0)

    def test_正規化すると空になるフレーズは候補なし(self, transcript: Transcript) -> None:
        """句読点だけの phrase。全窓が空文字と比較されて誤爆するのを防ぐ。"""
        assert s3.find_candidates(flat_of(transcript), "、。！", 82.0) == []


# ---------------------------------------------------------------------------
# 隣接候補の統合（SPEC Step 3-3「開始位置が3文字以内は最高スコアに統合」）
# ---------------------------------------------------------------------------


class TestMergeAdjacent:
    def test_隣接する窓は最高スコアの1件に統合される(self) -> None:
        """一致箇所の周りでは窓が数個続けてしきい値を超える。
        統合しないと同じ場所が複数候補として数えられ、occurrence の nth がずれる。"""
        flat = flat_from(("あああこのチャンネルは", 0.0, 2.0))
        merged = s3.find_candidates(flat, PHRASE_A, 40.0)
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(100.0)
        assert merged[0].matched_text == PHRASE_A

    def test_統合しなければ複数の窓が残る(self) -> None:
        """統合が効いていることの裏取り（merge_within=0 なら間引かれない）。"""
        flat = flat_from(("あああこのチャンネルは", 0.0, 2.0))
        raw = s3.find_candidates(flat, PHRASE_A, 40.0, merge_within=0)
        assert len(raw) > 1
        assert max(c.score for c in raw) == pytest.approx(100.0)

    def test_統合後の候補は3文字より離れている(self, transcript: Transcript) -> None:
        flat = flat_of(transcript)
        got = s3.find_candidates(flat, PHRASE_B, 60.0)
        starts = [c.norm_start for c in got]
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert all(g > s3.DEFAULT_MERGE_WITHIN for g in gaps), starts

    def test_離れた一致どうしは統合されない(self, transcript: Transcript) -> None:
        """おとりと本物は 100 文字以上離れている。まとめてしまうと B が消える。"""
        got = s3.find_candidates(flat_of(transcript), PHRASE_B, 85.0)
        assert len(got) == 2

    def test_残ったスコアは近傍の窓の最大値(self, transcript: Transcript) -> None:
        """「最高スコアのものに統合する」。近傍により高い窓が残っていたら統合が壊れている。"""
        flat = flat_of(transcript)
        scores = window_scores(flat, PHRASE_A)
        norm_phrase = normalize_phrase(PHRASE_A)
        for c in s3.find_candidates(flat, PHRASE_A, 60.0):
            assert c.score == pytest.approx(
                fuzz.ratio(flat.norm[c.norm_start : c.norm_start + len(norm_phrase)], norm_phrase)
            )
            lo = max(0, c.norm_start - s3.DEFAULT_MERGE_WITHIN)
            hi = min(len(scores), c.norm_start + s3.DEFAULT_MERGE_WITHIN + 1)
            assert c.score >= max(scores[lo:hi]) - 1e-9


# ---------------------------------------------------------------------------
# filter_candidates — 探索窓（SPEC Step 3-4）
# ---------------------------------------------------------------------------


class TestFilterBySearchWindow:
    @pytest.fixture
    def candidates(self, transcript: Transcript) -> list[AnchorCandidate]:
        return s3.find_candidates(flat_of(transcript), PHRASE_A, 82.0)

    @pytest.mark.parametrize(
        "window,expect_kept",
        [
            ((0, 600), True),      # config/ai-radio.json と同じ窓
            ((0, 5.95), True),     # 上端ちょうど。範囲は閉区間
            ((5.95, 600), True),   # 下端ちょうど
            ((0, 5.94), False),    # 5.95 は窓の外
            ((5.96, 600), False),
            ((20, 600), False),
        ],
    )
    def test_探索窓の内外で残る落ちるが決まる(
        self,
        transcript: Transcript,
        candidates: list[AnchorCandidate],
        window: tuple[float, float],
        expect_kept: bool,
    ) -> None:
        """候補の start_time が search_window_sec に入っているかで判定する。"""
        kept, rejected = s3.filter_candidates(
            candidates, anchor(search_window_sec=list(window)), flat_of(transcript)
        )
        assert bool(kept) is expect_kept
        assert bool(rejected) is not expect_kept

    def test_窓の外に落ちた候補には理由が入る(
        self, transcript: Transcript, candidates: list[AnchorCandidate]
    ) -> None:
        """あとでエラーメッセージに「なぜ落ちたか」を出せるようにする。"""
        _, rejected = s3.filter_candidates(
            candidates, anchor(search_window_sec=[0, 5.94]), flat_of(transcript)
        )
        assert len(rejected) == 1
        reason = rejected[0].rejected_reason
        assert reason
        assert "5.94" in reason, reason
        assert "5.95" in reason, reason

    def test_窓が未指定なら時刻では落ちない(
        self, transcript: Transcript, candidates: list[AnchorCandidate]
    ) -> None:
        kept, rejected = s3.filter_candidates(candidates, anchor(), flat_of(transcript))
        assert len(kept) == len(candidates)
        assert rejected == []

    def test_残った候補に理由は付かない(
        self, transcript: Transcript, candidates: list[AnchorCandidate]
    ) -> None:
        kept, _ = s3.filter_candidates(
            candidates, anchor(search_window_sec=[0, 600]), flat_of(transcript)
        )
        assert kept and all(c.rejected_reason is None for c in kept)

    def test_残ったぶんと落ちたぶんの合計は入力と一致する(self, transcript: Transcript) -> None:
        """候補を握りつぶさない。数が合わないと candidates_found の報告が嘘になる。"""
        flat = flat_of(transcript)
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        kept, rejected = s3.filter_candidates(cands, anchor(id="B", phrase=PHRASE_B), flat)
        assert len(kept) + len(rejected) == len(cands)
        assert [c.start_time for c in kept] == sorted(c.start_time for c in kept)


# ---------------------------------------------------------------------------
# filter_candidates — must_follow（SPEC 5章 / Step 3-4）
# ---------------------------------------------------------------------------


class TestFilterByMustFollow:
    def test_おとりはmust_followで落ちる(self, transcript: Transcript, config: Config) -> None:
        """19.95秒の「ということで、まず前回の…」には「木原」が続かないので落ちる。
        これが落ちないと 02_main.mp4 が 24 秒短くなる。"""
        flat = flat_of(transcript)
        anchor_b = config.anchor("B")
        cands = s3.find_candidates(flat, anchor_b.phrase, anchor_b.threshold_score)
        kept, rejected = s3.filter_candidates(cands, anchor_b, flat)

        assert [c.start_time for c in kept] == [pytest.approx(fixtures.EXPECTED_ANCHOR_B_RAW)]
        assert [c.start_time for c in rejected] == [pytest.approx(19.95)]
        reason = rejected[0].rejected_reason
        assert reason
        assert "木原" in reason, reason

    def test_must_followが無ければおとりも残る(self, transcript: Transcript) -> None:
        """おとりは must_follow 以外では本物と区別できない、ということの裏取り。"""
        flat = flat_of(transcript)
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        kept, rejected = s3.filter_candidates(cands, anchor(id="B", phrase=PHRASE_B), flat)
        assert len(kept) == 2
        assert rejected == []

    @pytest.mark.parametrize(
        "within_sec,expect_kept",
        [
            (4.0, True),
            (3.9, True),    # 境界。候補終端 11.0 + 3.9 = 14.9 に「木原」の start が乗る
            (3.8, False),
            (0.5, False),
        ],
    )
    def test_within_secの内側にあるかで決まる(self, within_sec: float, expect_kept: bool) -> None:
        """「候補終端から within_sec 以内に出現するか」（SPEC Step 3-4）。"""
        flat = flat_from(
            ("ということで", 10.0, 11.0),
            ("あああ", 11.0, 12.0),
            ("木原", 14.9, 15.4),
            ("さん", 15.4, 16.0),
        )
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        assert len(cands) == 1
        cfg = anchor(
            id="B", phrase=PHRASE_B, must_follow={"phrase": "木原", "within_sec": within_sec}
        )
        kept, rejected = s3.filter_candidates(cands, cfg, flat)
        assert bool(kept) is expect_kept
        if not expect_kept:
            assert rejected[0].rejected_reason and "木原" in rejected[0].rejected_reason

    def test_候補より前にあるmust_followは数えない(self) -> None:
        """「続く」フィルタなので、直前に出ていても満たしたことにはならない。"""
        flat = flat_from(
            ("木原さん", 5.0, 6.0),
            ("ということで", 10.0, 11.0),
            ("おわり", 11.0, 12.0),
        )
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        cfg = anchor(id="B", phrase=PHRASE_B, must_follow={"phrase": "木原", "within_sec": 10.0})
        kept, rejected = s3.filter_candidates(cands, cfg, flat)
        assert kept == []
        assert len(rejected) == 1

    def test_must_followもあいまい一致で確認する(self) -> None:
        """SPEC Step 3-4「同じあいまい一致で確認し」。
        must_follow.fuzzy_threshold を下げれば 1 文字違いでも通る。"""
        flat = flat_from(("ということで", 10.0, 11.0), ("気原さん", 11.0, 12.0))
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)

        strict = anchor(id="B", phrase=PHRASE_B, must_follow={"phrase": "木原", "within_sec": 4.0})
        loose = anchor(
            id="B",
            phrase=PHRASE_B,
            must_follow={"phrase": "木原", "within_sec": 4.0, "fuzzy_threshold": 0.5},
        )
        assert s3.filter_candidates(cands, strict, flat)[0] == []
        assert len(s3.filter_candidates(cands, loose, flat)[0]) == 1

    def test_実在しないmust_followなら全滅する(self, transcript: Transcript) -> None:
        """満たさない候補を残して繋いだりしない。全部落ちて Step 3 が止まるのが正しい。"""
        flat = flat_of(transcript)
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        cfg = anchor(id="B", phrase=PHRASE_B, must_follow={"phrase": "山田", "within_sec": 4.0})
        kept, rejected = s3.filter_candidates(cands, cfg, flat)
        assert kept == []
        assert len(rejected) == len(cands)
        assert all(c.rejected_reason for c in rejected)

    def test_探索窓とmust_followは両方効く(self, transcript: Transcript, config: Config) -> None:
        """窓で落ちた候補には窓の理由が、残ったものだけ must_follow が見られる。"""
        flat = flat_of(transcript)
        cfg = anchor(
            id="B",
            phrase=PHRASE_B,
            search_window_sec=[0, 30],
            must_follow={"phrase": "木原", "within_sec": 4.0},
        )
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        kept, rejected = s3.filter_candidates(cands, cfg, flat)
        assert kept == []
        reasons = {round(c.start_time, 2): c.rejected_reason or "" for c in rejected}
        assert "木原" in reasons[19.95]
        assert "30" in reasons[43.95]


# ---------------------------------------------------------------------------
# select_candidate（SPEC Step 3-4「occurrence に従って1つを確定」）
# ---------------------------------------------------------------------------


class TestSelectCandidate:
    @pytest.fixture
    def three(self) -> list[AnchorCandidate]:
        # わざとスコア順と時刻順を食い違わせる（occurrence は時刻で決めるべき）
        return [
            make_candidate(90.0, 10.0, norm_start=10),
            make_candidate(100.0, 20.0, norm_start=100),
            make_candidate(95.0, 30.0, norm_start=200),
        ]

    def test_firstは最も早い候補(self, three: list[AnchorCandidate]) -> None:
        """スコアが一番高い候補ではなく、時刻が一番早い候補を採る。"""
        got = s3.select_candidate(three, anchor(occurrence="first"))
        assert got.start_time == pytest.approx(10.0)

    def test_lastは最も遅い候補(self, three: list[AnchorCandidate]) -> None:
        got = s3.select_candidate(three, anchor(occurrence="last"))
        assert got.start_time == pytest.approx(30.0)

    @pytest.mark.parametrize("nth,expected", [(1, 10.0), (2, 20.0), (3, 30.0)])
    def test_nthは1始まりで時刻順(
        self, three: list[AnchorCandidate], nth: int, expected: float
    ) -> None:
        got = s3.select_candidate(three, anchor(occurrence="nth", nth=nth))
        assert got.start_time == pytest.approx(expected)

    def test_入力の並び順に依らない(self, three: list[AnchorCandidate]) -> None:
        shuffled = [three[2], three[0], three[1]]
        assert s3.select_candidate(shuffled, anchor(occurrence="first")).start_time == pytest.approx(10.0)
        assert s3.select_candidate(shuffled, anchor(occurrence="last")).start_time == pytest.approx(30.0)

    def test_nthが候補数を超えたら止まる(self, three: list[AnchorCandidate]) -> None:
        """足りないぶんを first で埋めたりしない（SPEC 9章「自動的に代替を選ばない」）。"""
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.select_candidate(three, anchor(occurrence="nth", nth=4))
        message = str(exc.value)
        assert "4" in message
        assert "3" in message  # 実際の候補数を伝える

    def test_候補0件は例外(self) -> None:
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.select_candidate([], anchor(id="B", phrase=PHRASE_B))
        message = str(exc.value)
        assert "B" in message
        assert PHRASE_B in message
        assert_gives_next_action(message)


class TestRawCutTimeOf:
    def test_beforeは候補の先頭単語のstart(self) -> None:
        """SPEC Step 3-5「先頭文字が属する単語の start を raw_cut_time とする」。"""
        c = make_candidate(100.0, 43.95, end_time=45.561)
        assert s3.raw_cut_time_of(c, anchor(cut="before")) == pytest.approx(43.95)

    def test_afterは候補の末尾単語のend(self) -> None:
        c = make_candidate(100.0, 43.95, end_time=45.561)
        assert s3.raw_cut_time_of(c, anchor(cut="after")) == pytest.approx(45.561)


# ---------------------------------------------------------------------------
# build_error_message（SPEC Step 3「失敗時の挙動」／SPEC 9章）
# ---------------------------------------------------------------------------


class TestBuildErrorMessage:
    def test_しきい値を下回った候補が上位3件まで出る(self, transcript: Transcript) -> None:
        """候補0件でも「一番惜しかったのはどこか」を出す。ここが無いと調整の起点が無い。"""
        flat = flat_of(transcript)
        cfg = anchor(phrase="宇宙船の操縦マニュアル", fuzzy_threshold=0.9)
        message = s3.build_error_message(cfg, [], flat)

        ranks = error_ranks(message)
        assert 1 <= len(ranks) <= s3.TOP_N_ON_ERROR
        assert [r for r, _ in ranks] == list(range(1, len(ranks) + 1))
        assert "文脈" in message
        assert "宇宙船の操縦マニュアル" in message
        assert_gives_next_action(message)

    def test_上位はスコアの降順(self, transcript: Transcript) -> None:
        cfg = anchor(phrase="宇宙船の操縦マニュアル", fuzzy_threshold=0.9)
        scores = [s for _, s in error_ranks(s3.build_error_message(cfg, [], flat_of(transcript)))]
        assert scores == sorted(scores, reverse=True)

    def test_文脈は前後30文字ぶん出る(self, transcript: Transcript) -> None:
        """SPEC Step 3「最高スコアの候補とその前後30文字の文脈を出し」。"""
        flat = flat_of(transcript)
        cands = s3.find_candidates(flat, PHRASE_B, 85.0)
        cfg = anchor(id="B", phrase=PHRASE_B, must_follow={"phrase": "山田", "within_sec": 4.0})
        _, rejected = s3.filter_candidates(cands, cfg, flat)
        message = s3.build_error_message(cfg, cands, flat)

        for c in cands:
            assert c.context in message
            assert len(c.context.strip("…")) > s3.CONTEXT_WIDTH
        # 落ちた理由も追える
        assert rejected and rejected[0].rejected_reason
        assert "山田" in message

    def test_4件以上あっても3件までに絞る(self, transcript: Transcript) -> None:
        pool = [make_candidate(90.0 - i, 10.0 * i, norm_start=50 * i) for i in range(6)]
        message = s3.build_error_message(anchor(), pool, flat_of(transcript))
        ranks = error_ranks(message)
        assert len(ranks) == s3.TOP_N_ON_ERROR
        assert [s for _, s in ranks] == [pytest.approx(90.0), pytest.approx(89.0), pytest.approx(88.0)]

    def test_候補が皆無でも落ちない(self) -> None:
        """文字起こしが空でもメッセージを作れること。ここで例外を投げると原因が消える。"""
        message = s3.build_error_message(anchor(), [], build_flat([]))
        assert isinstance(message, str) and message.strip()
        assert PHRASE_A in message
        assert_gives_next_action(message)

    def test_絞り込み条件もメッセージに出る(self, transcript: Transcript) -> None:
        """落ちた原因が phrase ではなく窓や must_follow のこともあるため。"""
        cfg = anchor(
            id="B",
            phrase=PHRASE_B,
            occurrence="last",
            search_window_sec=[0, 600],
            must_follow={"phrase": "木原", "within_sec": 4.0},
        )
        message = s3.build_error_message(cfg, [], flat_of(transcript))
        assert "last" in message
        assert "木原" in message
        assert "600" in message


# ---------------------------------------------------------------------------
# check_anchor_order（SPEC 9章「アンカーBがAより前 → 停止」）
# ---------------------------------------------------------------------------


def make_result(anchor_id: str, raw_cut_time: float, matched_text: str) -> AnchorResult:
    return AnchorResult(
        id=anchor_id,
        phrase=matched_text,
        matched_text=matched_text,
        score=100.0,
        raw_cut_time=raw_cut_time,
        candidates_found=1,
        candidates_rejected=0,
        context=f"…{matched_text}…",
    )


class TestCheckAnchorOrder:
    def test_順序どおりなら何も起きない(self) -> None:
        anchors = [anchor(id="A"), anchor(id="B", phrase=PHRASE_B)]
        results = {"A": make_result("A", 5.95, PHRASE_A), "B": make_result("B", 43.95, PHRASE_B)}
        s3.check_anchor_order(anchors, results)

    def test_逆転していたら止まる(self) -> None:
        anchors = [anchor(id="A"), anchor(id="B", phrase=PHRASE_B)]
        results = {"A": make_result("A", 43.95, PHRASE_A), "B": make_result("B", 5.95, PHRASE_B)}
        with pytest.raises(AnchorOrderError) as exc:
            s3.check_anchor_order(anchors, results)
        message = str(exc.value)
        assert "43.95" in message
        assert "5.95" in message
        assert "A" in message and "B" in message

    def test_同時刻は逆転扱いしない(self) -> None:
        anchors = [anchor(id="A"), anchor(id="B", phrase=PHRASE_B)]
        results = {"A": make_result("A", 10.0, PHRASE_A), "B": make_result("B", 10.0, PHRASE_B)}
        s3.check_anchor_order(anchors, results)

    def test_結果に無いアンカーは飛ばす(self) -> None:
        anchors = [anchor(id="A"), anchor(id="B", phrase=PHRASE_B)]
        s3.check_anchor_order(anchors, {"B": make_result("B", 43.95, PHRASE_B)})


# ---------------------------------------------------------------------------
# run（SPEC Step 3 の通し）
# ---------------------------------------------------------------------------


class TestRun:
    def test_合成エピソードでAとBが仕様どおりの位置に出る(
        self, ctx: RunContext, transcript: Transcript
    ) -> None:
        """Phase 1 の受け入れ基準そのもの。ここがずれたら先へ進む意味がない。"""
        got = s3.run(ctx, transcript)

        assert set(got) == {"A", "B"}
        assert got["A"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)
        assert got["B"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_B_RAW)
        assert got["A"].matched_text == PHRASE_A
        assert got["B"].matched_text == PHRASE_B
        assert got["A"].score == pytest.approx(100.0)
        assert got["B"].score == pytest.approx(100.0)

    def test_おとりの19秒は採用しない(self, ctx: RunContext, transcript: Transcript) -> None:
        """19.95秒の「ということで」を掴むと本編が24秒短くなる。"""
        got = s3.run(ctx, transcript)
        assert got["B"].raw_cut_time != pytest.approx(19.95)
        assert got["B"].candidates_found == 2
        assert got["B"].candidates_rejected == 1
        assert got["A"].candidates_found == 1
        assert got["A"].candidates_rejected == 0

    def test_raw_cut_timeは語頭の単語のstart(self, ctx: RunContext, transcript: Transcript) -> None:
        """文字起こしから独立に計算した単語 start と一致すること（SPEC Step 3-5）。"""
        got = s3.run(ctx, transcript)
        assert got["A"].raw_cut_time == pytest.approx(word_span_of(transcript, PHRASE_A)[0])
        assert got["B"].raw_cut_time == pytest.approx(
            word_span_of(transcript, PHRASE_B, occurrence=1)[0]
        )

    def test_cut_afterなら末尾単語のend(self, ctx: RunContext, transcript: Transcript, config: Config) -> None:
        """cut='after' は「フレーズを言い終わってから」切る。末尾文字が属する単語の end。"""
        a, b = anchors_json(config)
        a["cut"] = "after"
        got = s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        assert got["A"].raw_cut_time == pytest.approx(word_span_of(transcript, PHRASE_A)[1])
        assert got["A"].raw_cut_time > fixtures.EXPECTED_ANCHOR_A_RAW

    def test_anchors_jsonがSPECの形で書かれる(self, ctx: RunContext, transcript: Transcript) -> None:
        """SPEC Step 3 の出力例のキーを満たすこと。Step 4 以降と decisions.json がこれを読む。"""
        s3.run(ctx, transcript)
        path = ctx.work_path(s3.ANCHORS_FILE)
        assert path.exists()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"A", "B"}
        for anchor_id, entry in payload.items():
            assert SPEC_ANCHOR_KEYS <= set(entry), (anchor_id, sorted(entry))
            assert isinstance(entry["phrase"], str) and entry["phrase"]
            assert isinstance(entry["matched_text"], str) and entry["matched_text"]
            assert isinstance(entry["score"], (int, float))
            assert 0.0 <= float(entry["score"]) <= 100.0
            assert isinstance(entry["raw_cut_time"], (int, float))
            assert entry["candidates_found"] >= 1
            assert entry["candidates_rejected"] >= 0
            assert entry["candidates_found"] > entry["candidates_rejected"]
            assert isinstance(entry["context"], str) and entry["context"]

        assert payload["A"]["raw_cut_time"] == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)
        assert payload["B"]["raw_cut_time"] == pytest.approx(fixtures.EXPECTED_ANCHOR_B_RAW)

    def test_中間ファイルは人が読める日本語のまま(self, ctx: RunContext, transcript: Transcript) -> None:
        """中間ファイルはデバッグの起点（SPEC 11章）。\\uXXXX に潰さない。"""
        s3.run(ctx, transcript)
        text = ctx.work_path(s3.ANCHORS_FILE).read_text(encoding="utf-8")
        assert PHRASE_A in text
        assert "\\u" not in text

    def test_秒数は小数点以下3桁で保存される(self, ctx: RunContext, transcript: Transcript, config: Config) -> None:
        """SPEC 11章「秒数は全て float（小数点以下3桁）」。"""
        a, b = anchors_json(config)
        a["cut"] = "after"  # 8.36875... のように桁が伸びるケースを通す
        s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        payload = json.loads(ctx.work_path(s3.ANCHORS_FILE).read_text(encoding="utf-8"))
        for entry in payload.values():
            value = entry["raw_cut_time"]
            assert value == pytest.approx(round(float(value), 3))

    def test_表記ゆれ_空白と句読点入りのphraseでも同じ位置(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """config に「この チャンネル は。」と書かれていても同じアンカーを指す。"""
        a, b = anchors_json(config)
        a["phrase"] = "この チャンネル は。"
        got = s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        assert got["A"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)
        assert got["A"].matched_text == PHRASE_A

    def test_表記ゆれ_1文字違いのphraseでも語頭に合う(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """「は」を「わ」と書いた（＝ASR が1文字外した）ときでも、
        カット点は発話の語頭 5.95 秒でなければならない。

        あいまい一致の窓は1文字ずれても同点になるので、統合時にどちらを残すかで
        カット点が 1.9 秒手前に飛ぶ。手前に飛ぶと 02_main.mp4 の冒頭に前の文の尻尾
        （「…いきます。」）が混ざり、Phase 1 の受け入れ基準
        「02_main.mp4 の冒頭が『このチャンネルは』で始まっている」を満たせない。
        """
        a, b = anchors_json(config)
        a["phrase"] = "このチャンネルわ"
        got = s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        assert got["A"].matched_text == PHRASE_A
        assert got["A"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW)

    def test_must_followを外すとおとりが選ばれる(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """must_follow が本当に効いていることの裏取り。外すと 19.95 秒を掴む。"""
        a, b = anchors_json(config)
        b.pop("must_follow")
        b["occurrence"] = "first"
        got = s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        assert got["B"].raw_cut_time == pytest.approx(19.95)

    def test_見つからないフレーズで止まる(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """勝手に代替位置を選ばずに例外。メッセージには上位候補と次の一手を出す。"""
        a, b = anchors_json(config)
        a["phrase"] = "宇宙船の操縦マニュアル"
        target = ctx_with(ctx, config_with(config, [a, b]))
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.run(target, transcript)
        message = str(exc.value)
        assert "宇宙船の操縦マニュアル" in message
        assert error_ranks(message)
        assert_gives_next_action(message)
        assert not target.work_path(s3.ANCHORS_FILE).exists()

    def test_しきい値が高すぎても止まる(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """しきい値の締めすぎで 0 件になったときも、下げろと言える情報が出る。"""
        a, b = anchors_json(config)
        a["phrase"] = "このチャンネルわ"
        a["fuzzy_threshold"] = 0.95
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        message = str(exc.value)
        assert "0.95" in message
        assert_gives_next_action(message)

    def test_探索窓の外にしか無ければ止まる(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """窓の設定ミスで 0 件になったときも、「窓の外に 5.95 秒の候補がある」と分かること。"""
        a, b = anchors_json(config)
        a["search_window_sec"] = [30, 60]
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.run(ctx_with(ctx, config_with(config, [a, b])), transcript)
        message = str(exc.value)
        assert error_ranks(message)
        assert "5.95" in message      # 窓の外にあった候補の時刻
        assert "30" in message        # 効いていた窓
        assert_gives_next_action(message)

    def test_アンカーの順序が逆転していたら止まる(
        self, ctx: RunContext, transcript: Transcript, config: Config
    ) -> None:
        """SPEC 9章。B が A より前になる設定はミスの可能性が高いので停止し、
        両方の時刻を出して原因を切り分けられるようにする。"""
        a, b = anchors_json(config)
        swapped_a = dict(b, id="A")
        swapped_b = dict(a, id="B")
        with pytest.raises(AnchorOrderError) as exc:
            s3.run(ctx_with(ctx, config_with(config, [swapped_a, swapped_b])), transcript)
        message = str(exc.value)
        assert "43.95" in message
        assert "5.95" in message

    def test_文字起こしが空なら未検出で止まる(self, ctx: RunContext) -> None:
        """Step 2 が空を返しても、ここで潰れずに理由の分かる例外にする。"""
        empty = Transcript(language="ja", duration=0.0, segments=[])
        with pytest.raises(AnchorNotFoundError) as exc:
            s3.run(ctx, empty)
        assert_gives_next_action(str(exc.value))


# ---------------------------------------------------------------------------
# save / load（--from-step での再開）
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_書いたものを読み戻せる(self, ctx: RunContext, transcript: Transcript) -> None:
        saved = s3.run(ctx, transcript)
        loaded = s3.load(ctx)

        assert set(loaded) == set(saved)
        for anchor_id, result in saved.items():
            assert loaded[anchor_id].id == anchor_id
            assert loaded[anchor_id].phrase == result.phrase
            assert loaded[anchor_id].matched_text == result.matched_text
            assert loaded[anchor_id].score == pytest.approx(result.score)
            assert loaded[anchor_id].raw_cut_time == pytest.approx(result.raw_cut_time)
            assert loaded[anchor_id].candidates_found == result.candidates_found
            assert loaded[anchor_id].candidates_rejected == result.candidates_rejected
            assert loaded[anchor_id].context == result.context

    def test_中間ファイルが無ければ次の一手を示して止まる(self, ctx: RunContext) -> None:
        with pytest.raises(MissingArtifactError) as exc:
            s3.load(ctx)
        message = str(exc.value)
        assert s3.ANCHORS_FILE in message
        assert "step" in message.lower()

    def test_壊れたJSONは握りつぶさない(self, ctx: RunContext) -> None:
        path = ctx.work_path(s3.ANCHORS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{壊れている", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s3.load(ctx)

    def test_中身が空なら止まる(self, ctx: RunContext) -> None:
        path = ctx.work_path(s3.ANCHORS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s3.load(ctx)

    def test_必須キーが欠けていたら止まる(self, ctx: RunContext) -> None:
        """raw_cut_time の無いアンカーを Step 4 に渡さない。"""
        path = ctx.work_path(s3.ANCHORS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"A": {"phrase": PHRASE_A}}), encoding="utf-8")
        with pytest.raises(MissingArtifactError) as exc:
            s3.load(ctx)
        assert "A" in str(exc.value)
