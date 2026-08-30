"""時間軸まわりのユーティリティ（SPEC Step 5 の3段スナップ、Step 6-a の時刻変換とチャプター、時刻書式）。

ここには「秒 ⇄ 単語 ⇄ 文 ⇄ final.mp4 のタイムライン」の変換だけを置く。
ffmpeg も LLM も触らない純関数の集まりにしてあるのは、テストで単体検証できるようにするため。
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from typing import Mapping, Sequence

from ..config import SegmentConfig
from ..errors import ConfigError
from ..logging_util import get_logger
from ..models import Chapter, CutPoint, Word, r3

logger = get_logger(__name__)

#: 文末とみなす記号（SPEC Step 5 の後処理2「。？！ で判定」）
#
# 半角ピリオドは入れない。ASR が「3.5倍」のような小数をひとつの単語で返すため、
# 入れると「要約の精度は3.5」で文が切れ、拡張後の文頭が「倍になりました。」になる。
# 文頭が千切れるのは語尾が千切れるのと同じくらい視聴に耐えない。
SENTENCE_TERMINATORS = "。．？！?!"

#: 単語境界の指定に使えるキーワード
VALID_KINDS = ("start", "end")


class TimestampFormatError(ConfigError, ValueError):
    """タイムスタンプ文字列が `M:SS` / `H:MM:SS` の形式ではない。

    CLI が捕まえられるよう ConfigError を継承しつつ、素直に `ValueError` としても
    扱えるようにしてある（`parse_timestamp` は値の解釈失敗であり設定固有ではないため）。
    """


# ---------------------------------------------------------------------------
# 時刻の書式
# ---------------------------------------------------------------------------


def fmt_timestamp(sec: float) -> str:
    """秒を YouTube チャプターの書式にする。1時間未満は "M:SS"、1時間以上は "H:MM:SS"。

    秒は切り捨て（int(floor)）。負値・NaN・無限大は "0:00" に丸める。
    早回しの表示が出るより、先頭に寄る方が概要欄としては安全なため。
    """
    try:
        value = float(sec)
    except (TypeError, ValueError):
        return "0:00"
    if not math.isfinite(value) or value < 0:
        return "0:00"
    total = int(math.floor(value))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


_RE_HMS = re.compile(r"^(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$")
_RE_MS = re.compile(r"^(?P<m>\d+):(?P<s>\d{1,2}(?:\.\d+)?)$")
_RE_BARE = re.compile(r"^(?P<s>\d+(?:\.\d+)?)$")


def parse_timestamp(text: str) -> float:
    """`fmt_timestamp` の逆。"M:SS" / "H:MM:SS" / 秒だけ、を float 秒にする。

    description.txt のチャプター行を読み戻して検算するために使う。
    全角数字・全角コロンは NFKC で吸収する。
    """
    if not isinstance(text, str):
        raise TimestampFormatError(f"タイムスタンプが文字列ではありません: {text!r}")
    s = unicodedata.normalize("NFKC", text).strip()
    if not s:
        raise TimestampFormatError("タイムスタンプが空です。")

    m = _RE_HMS.match(s)
    if m:
        hours = int(m.group("h"))
        minutes = int(m.group("m"))
        seconds = float(m.group("s"))
        if minutes >= 60 or seconds >= 60.0:
            raise TimestampFormatError(f"分・秒は 0〜59 にしてください: {text!r}")
        return float(hours * 3600 + minutes * 60) + seconds

    m = _RE_MS.match(s)
    if m:
        minutes = int(m.group("m"))
        seconds = float(m.group("s"))
        if seconds >= 60.0:
            raise TimestampFormatError(f"秒は 0〜59 にしてください: {text!r}")
        return float(minutes * 60) + seconds

    m = _RE_BARE.match(s)
    if m:
        return float(m.group("s"))

    raise TimestampFormatError(
        f"タイムスタンプの形式が不正です: {text!r}（期待する形式: \"M:SS\" または \"H:MM:SS\"）"
    )


def clamp(value: float, lo: float, hi: float) -> float:
    """value を [lo, hi] に収める。lo > hi の指定は入れ違いとみなして交換する。"""
    v = float(value)
    low = float(lo)
    high = float(hi)
    if math.isnan(v):
        return low
    if high < low:
        low, high = high, low
    if v < low:
        return low
    if v > high:
        return high
    return v


# ---------------------------------------------------------------------------
# 単語境界
# ---------------------------------------------------------------------------


def _check_kind(kind: str) -> str:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind は {VALID_KINDS} のいずれかにしてください（実際: {kind!r}）。")
    return kind


def _starts(words: Sequence[Word]) -> list[float]:
    return [float(w.start) for w in words]


def _ends(words: Sequence[Word]) -> list[float]:
    return [float(w.end) for w in words]


def word_index_at_time(words: Sequence[Word], t: float, *, kind: str = "start") -> int:
    """時刻 t に対応する単語インデックスを二分探索で返す。

    kind="start" は「t 以上で最も近い単語」（無ければ最後の単語）、
    kind="end" は「t 以下で最も近い単語」（無ければ最初の単語）。
    words が空なら -1 を返す（呼び出し側が「単語が無い」と判定できるようにするため）。
    """
    _check_kind(kind)
    n = len(words)
    if n == 0:
        return -1
    value = float(t)
    if kind == "start":
        idx = bisect_left(_starts(words), value)
        if idx >= n:
            return n - 1
        return idx
    idx = bisect_right(_ends(words), value) - 1
    if idx < 0:
        return 0
    return idx


def snap_to_word_boundary(words: Sequence[Word], t: float, *, kind: str) -> float:
    """t を最寄りの単語境界へ寄せる。kind="start" なら単語の start、"end" なら end。

    LLM が返す秒数は語の途中で切れていることが多いので、まずここで整える（SPEC Step 5-1）。
    words が空なら t をそのまま返す。
    """
    _check_kind(kind)
    if not words:
        return float(t)
    value = float(t)
    bounds = _starts(words) if kind == "start" else _ends(words)
    idx = bisect_left(bounds, value)
    if idx <= 0:
        return bounds[0]
    if idx >= len(bounds):
        return bounds[-1]
    before = bounds[idx - 1]
    after = bounds[idx]
    # 同距離なら手前を採る（語頭が欠けるより少し長い方が安全）。
    return before if (value - before) <= (after - value) else after


def _expand_start_index(words: Sequence[Word], t: float) -> int:
    """t を含む（または t の直前に始まる）単語の index。拡張の起点に使う。"""
    if not words:
        return -1
    idx = bisect_right(_starts(words), float(t)) - 1
    return idx if idx >= 0 else 0


def _expand_end_index(words: Sequence[Word], t: float) -> int:
    """t を含む（または t の直後に終わる）単語の index。拡張の終点に使う。"""
    if not words:
        return -1
    ends = _ends(words)
    idx = bisect_left(ends, float(t))
    return idx if idx < len(ends) else len(ends) - 1


# ---------------------------------------------------------------------------
# 文境界
# ---------------------------------------------------------------------------


def _is_sentence_end(word: Word, terminators: str) -> bool:
    """終端記号を1文字でも含む単語は、その文の末尾とみなす。

    ASR は「です。」のように終端記号を単語の途中／末尾に含めて返すことがあるため、
    単語単位で判定しつつ「含んでいれば末尾」とする。
    """
    if not terminators:
        return False
    return any(ch in terminators for ch in word.word)


def sentence_bounds(
    words: Sequence[Word],
    start_idx: int,
    end_idx: int,
    terminators: str = SENTENCE_TERMINATORS,
) -> tuple[int, int]:
    """start_idx が属する文の先頭単語 index と、end_idx が属する文の末尾単語 index（含む）を返す。

    words が空なら (0, -1) を返す。範囲外の index は内側へ丸める。
    """
    n = len(words)
    if n == 0:
        return (0, -1)
    s = int(clamp(start_idx, 0, n - 1))
    e = int(clamp(end_idx, 0, n - 1))
    if e < s:
        e = s

    # 前方向: 直前に終端記号を含む単語があれば、その次が文頭。
    head = 0
    for i in range(s - 1, -1, -1):
        if _is_sentence_end(words[i], terminators):
            head = i + 1
            break

    # 後方向: end_idx 以降で最初に終端記号を含む単語が文末。
    tail = n - 1
    for i in range(e, n):
        if _is_sentence_end(words[i], terminators):
            tail = i
            break

    if tail < head:
        tail = head
    return (head, tail)


def expand_to_sentence(words: Sequence[Word], start: float, end: float) -> tuple[float, float]:
    """区間を、その区間が属する文の先頭・末尾まで広げる（SPEC Step 5-2）。

    語尾が千切れたハイライトを出さないための処理。words が空なら入力をそのまま返す。
    """
    if not words:
        return (float(start), float(end))
    s_idx = _expand_start_index(words, start)
    e_idx = _expand_end_index(words, end)
    if e_idx < s_idx:
        e_idx = s_idx
    head, tail = sentence_bounds(words, s_idx, e_idx)
    return (float(words[head].start), float(words[tail].end))


def drop_last_sentence(
    words: Sequence[Word], start: float, end: float
) -> tuple[float, float] | None:
    """末尾の文を1つ落とした (start, end) を返す。落とすと空になるなら None。

    ハイライトが max_duration_sec を超えたときの詰め方（SPEC Step 5-4）。
    """
    if not words:
        return None
    s = float(start)
    e = float(end)
    if e <= s:
        return None

    s_idx = _expand_start_index(words, s)
    e_idx = _expand_end_index(words, e)
    if e_idx < s_idx:
        return None

    # 末尾単語が属する文の先頭を探し、その1つ手前の単語末尾を新しい終端にする。
    last_head, _ = sentence_bounds(words, e_idx, e_idx)
    if last_head <= s_idx:
        return None  # 区間全体が1つの文。これ以上落とせない。

    new_end = float(words[last_head - 1].end)
    if new_end <= s:
        return None
    return (s, new_end)


# ---------------------------------------------------------------------------
# final.mp4 のタイムライン
# ---------------------------------------------------------------------------


def to_final_time(
    t: float, *, cut_a: float, cut_b: float, highlight_dur: float, main_dur: float
) -> float:
    """元動画の時刻 t を final.mp4 上の時刻に変換する（SPEC 6-a）。

    先頭にハイライトを足しているため、元動画の秒数をそのまま書くと全部ずれる。
    境界（t == cut_b）は本編ではなくエンディング側として扱う。
    """
    value = float(t)
    if value < float(cut_b):
        return float(highlight_dur) + (value - float(cut_a))
    return float(highlight_dur) + float(main_dur) + (value - float(cut_b))


def resolve_segment_bounds(
    seg: SegmentConfig, cuts: Mapping[str, CutPoint], total_duration: float
) -> tuple[float, float]:
    """segments[].from / to（"start" / "end" / アンカーID）を実時刻に解決する。"""
    start = _resolve_ref(seg, "from", seg.from_, cuts, total_duration)
    end = _resolve_ref(seg, "to", seg.to, cuts, total_duration)
    if start >= end:
        raise ConfigError(
            f"セグメント '{seg.name}' の区間が空です（開始 {start:.3f}秒 >= 終了 {end:.3f}秒）。"
            " アンカーの検出位置か config の from/to を確認してください。"
        )
    return (start, end)


def _resolve_ref(
    seg: SegmentConfig,
    label: str,
    ref: str,
    cuts: Mapping[str, CutPoint],
    total_duration: float,
) -> float:
    if ref == "start":
        return 0.0
    if ref == "end":
        return float(total_duration)
    cut = cuts.get(ref)
    if cut is None:
        known = ", ".join(sorted(cuts)) or "（なし）"
        raise ConfigError(
            f"segments[{seg.name}].{label} が参照するアンカー '{ref}' のカット点がありません。"
            f" 利用できるアンカーID: {known}（ほかに \"start\" / \"end\" が使えます）。"
        )
    return float(cut.cut_time)


# ---------------------------------------------------------------------------
# チャプター
# ---------------------------------------------------------------------------


def normalize_chapters(
    chapters: Sequence[Chapter],
    final_duration: float,
    *,
    min_gap: float = 10.0,
    min_count: int = 3,
) -> tuple[list[Chapter], list[str]]:
    """YouTube チャプターの成立条件（SPEC 6-a）に合わせて直し、(結果, 警告文) を返す。

    LLM の出力は昇順が崩れたり間隔が詰まったりするので、ここで機械的に整える。
    例外は投げない。直した内容は警告文として返し、decisions.json に残せるようにする。
    """
    warnings: list[str] = []
    duration = float(final_duration)
    has_duration = math.isfinite(duration) and duration > 0

    # 1) 使えない値を落とす
    items: list[tuple[float, str]] = []
    for ch in chapters:
        try:
            t = float(ch.time_sec)
        except (TypeError, ValueError):
            warnings.append(f"チャプター「{ch.label}」の時刻が数値ではないため除外しました。")
            continue
        if not math.isfinite(t):
            warnings.append(f"チャプター「{ch.label}」の時刻が不正（{ch.time_sec!r}）なため除外しました。")
            continue
        if t < 0:
            warnings.append(f"チャプター「{ch.label}」の時刻が負（{r3(t)}秒）なため除外しました。")
            continue
        if has_duration and t >= duration:
            warnings.append(
                f"チャプター「{ch.label}」（{fmt_timestamp(t)}）が動画の尺"
                f"（{fmt_timestamp(duration)}）以降を指しているため除外しました。"
            )
            continue
        label = str(ch.label).strip()
        if not label:
            warnings.append(f"{fmt_timestamp(t)} のチャプターにラベルがありません。")
        items.append((t, label))

    # 2) 昇順に整列（同時刻は後勝ち）
    items.sort(key=lambda x: x[0])
    deduped: list[tuple[float, str]] = []
    for t, label in items:
        if deduped and deduped[-1][0] == t:
            warnings.append(
                f"{fmt_timestamp(t)} のチャプターが重複していたため「{deduped[-1][1]}」を"
                f"「{label}」で置き換えました。"
            )
            deduped[-1] = (t, label)
            continue
        deduped.append((t, label))

    if not deduped:
        warnings.append("有効なチャプターが1つも残りませんでした。概要欄のチャプターは出力できません。")
        logger.debug("normalize_chapters: 有効なチャプターなし")
        return ([], warnings)

    # 3) 先頭は必ず 0:00（ラベルは活かす）
    if deduped[0][0] != 0.0:
        warnings.append(
            f"最初のチャプターが {fmt_timestamp(deduped[0][0])} だったため 0:00 に寄せました"
            f"（ラベル「{deduped[0][1]}」）。"
        )
        deduped[0] = (0.0, deduped[0][1])

    # 4) 間隔が min_gap 未満のものを落とす（先に来たものを残す）
    kept: list[tuple[float, str]] = [deduped[0]]
    for t, label in deduped[1:]:
        gap = t - kept[-1][0]
        if gap < float(min_gap):
            warnings.append(
                f"チャプター「{label}」（{fmt_timestamp(t)}）は直前の「{kept[-1][1]}」"
                f"（{fmt_timestamp(kept[-1][0])}）との間隔が {r3(gap)}秒 で"
                f" {r3(float(min_gap))}秒 未満のため除外しました。"
            )
            continue
        kept.append((t, label))

    # 5) 成立条件のチェック（直せないものは警告だけ）
    if len(kept) < int(min_count):
        warnings.append(
            f"チャプターが {len(kept)} 個しかありません。YouTube のチャプターは"
            f" {int(min_count)} 個以上必要です。"
        )
    if has_duration and kept and (duration - kept[-1][0]) < float(min_gap):
        warnings.append(
            f"最後のチャプター「{kept[-1][1]}」（{fmt_timestamp(kept[-1][0])}）から動画の終わりまでが"
            f" {r3(float(min_gap))}秒 未満です。YouTube に認識されない可能性があります。"
        )

    result = [Chapter(time_sec=r3(t), label=label) for t, label in kept]
    logger.debug("normalize_chapters: %d 件 -> %d 件", len(chapters), len(result))
    return (result, warnings)
