"""SPEC Step 3「アンカー検出」。決まり文句をあいまい一致で探し raw_cut_time を決める。

ここが最重要かつ最も壊れやすい箇所（SPEC Step 3 冒頭）。方針：
- SPEC の 1〜5 をその順で行う（フラット化 → 正規化 → あいまい一致 → 絞り込み → 時刻）。
- 候補が0件なら**勝手に代替位置を選ばず**例外で止める。エラーには上位3件のスコアと
  前後30文字の文脈を載せ、しきい値を下げるかフレーズを直すよう案内する（SPEC 9章）。
- 検出ロジックは ffmpeg も LLM も触らない純関数に分け、テストが直接叩けるようにする。
"""

from __future__ import annotations

import difflib
import json
from bisect import bisect_right
from typing import Callable, Mapping, Sequence

from ..config import AnchorConfig
from ..context import RunContext
from ..errors import AnchorNotFoundError, AnchorOrderError, MissingArtifactError
from ..logging_util import get_logger
from ..models import AnchorCandidate, AnchorResult, Transcript, read_json, write_json
from ..util.text_normalize import FlatText, build_flat, normalize_phrase
from ..util.timeline import fmt_timestamp

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 3
NAME: str = "アンカー検出"

#: work/<episode_id>/ に書く中間ファイル
ANCHORS_FILE = "anchors.json"
OUTPUTS: tuple[str, ...] = (ANCHORS_FILE,)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

#: 候補の前後何文字を文脈として残すか（SPEC Step 3 の出力例・エラーメッセージ用）
CONTEXT_WIDTH = 30

#: 隣接候補を1つに統合する距離（正規化後の文字数、SPEC Step 3-3「開始位置が3文字以内」）
DEFAULT_MERGE_WITHIN = 3

#: 失敗時に表示する候補の件数（SPEC 9章「候補スコア上位3件と文脈を表示」）
TOP_N_ON_ERROR = 3

#: エラーメッセージの締め。文言は SPEC Step 3「失敗時の挙動」に合わせる。
ERROR_ADVICE = "fuzzy_threshold を下げるか、config の phrase を実際の発話に合わせて修正してください。"

try:  # rapidfuzz が本命。無い環境でも動くようにしておく（契約: difflib へフォールバック）。
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:  # pragma: no cover - rapidfuzz は必須依存なので通常は通らない
    _rapidfuzz_fuzz = None

#: difflib フォールバックの警告は1回だけ出す（窓ごとに出すとログが埋まるため）
_difflib_warned = False


# ---------------------------------------------------------------------------
# スコアリング
# ---------------------------------------------------------------------------


def _warn_difflib_once() -> None:
    """rapidfuzz が無いことを1回だけ警告する。"""
    global _difflib_warned
    if _difflib_warned:
        return
    _difflib_warned = True
    logger.warning(
        "rapidfuzz が見つからないため difflib.SequenceMatcher による近似でアンカーを探します。"
        " スコアが rapidfuzz と一致しないため、fuzzy_threshold の調整が必要になることがあります"
        "（`pip install rapidfuzz` を推奨）。"
    )


def _make_scorer(norm_phrase: str) -> Callable[[str, float], float]:
    """正規化済みフレーズ固定のスコア関数を作る。

    `score_cutoff` を渡して足切りするのが要点。60分の音声だと正規化後の flat は
    数万文字になり、窓ごとに全計算すると無駄が大きいため。
    しきい値未満は 0.0 を返す（0.0 は候補にならないので区別しなくてよい）。
    """
    if _rapidfuzz_fuzz is not None:
        def score_rapidfuzz(window: str, cutoff: float) -> float:
            return float(_rapidfuzz_fuzz.ratio(window, norm_phrase, score_cutoff=cutoff))

        return score_rapidfuzz

    _warn_difflib_once()
    matcher = difflib.SequenceMatcher(None, "", norm_phrase, autojunk=False)

    def score_difflib(window: str, cutoff: float) -> float:
        # set_seq1 だけを差し替えると、フレーズ側の索引が再利用されて速い。
        matcher.set_seq1(window)
        if matcher.real_quick_ratio() * 100.0 < cutoff:
            return 0.0
        if matcher.quick_ratio() * 100.0 < cutoff:
            return 0.0
        value = matcher.ratio() * 100.0
        return value if value >= cutoff else 0.0

    return score_difflib


def _scan_windows(flat: FlatText, norm_phrase: str, cutoff: float) -> list[tuple[int, float]]:
    """正規化済み flat の上を1文字ずつスライドし、(開始位置, スコア) を集める（SPEC Step 3-3）。

    窓の長さは正規化済みフレーズと同じ文字数。cutoff 未満は捨てる。
    """
    text = flat.norm
    n = len(text)
    m = len(norm_phrase)
    if n == 0 or m == 0 or m > n:
        return []

    score_of = _make_scorer(norm_phrase)
    cut = max(0.0, float(cutoff))
    out: list[tuple[int, float]] = []
    append = out.append
    for i in range(n - m + 1):
        score = score_of(text[i : i + m], cut)
        if score > 0.0:
            append((i, score))
    return out


def _head_match_len(flat: FlatText | None, norm_phrase: str, start: int) -> int:
    """窓の先頭が phrase の先頭と何文字そろっているか。

    同点の候補を分けるための物差し。raw_cut_time は「一致箇所の先頭文字が属する単語の
    start」なので、頭がそろっているかどうかがそのままカット位置の正しさになる。
    """
    if flat is None or not norm_phrase:
        return 0
    window = flat.norm[start : start + len(norm_phrase)]
    count = 0
    for left, right in zip(window, norm_phrase):
        if left != right:
            break
        count += 1
    return count


def _merge_adjacent(
    scored: Sequence[tuple[int, float]],
    merge_within: int,
    *,
    flat: FlatText | None = None,
    norm_phrase: str = "",
) -> list[tuple[int, float]]:
    """開始位置が merge_within 文字以内で隣り合う候補を1グループにし、最良の1つだけ残す。

    一致箇所の周りでは窓が数個続けてしきい値を超えるので、まとめないと同じ場所が
    複数候補として数えられ、occurrence の `nth` がずれる。

    同点のときは「頭がそろっているほう」を採る。config のフレーズが実際の発話と
    1文字違うと（「このチャンネルは」に対して「このチャンネルわ」など）、
    正しい位置の窓と、2文字手前から始まる窓が同点になる。手前を採ると
    raw_cut_time が前の文の尻尾に飛び、02_main.mp4 の冒頭に「…います。」が混ざる。
    """
    if not scored:
        return []
    ordered = sorted(scored, key=lambda item: item[0])
    limit = max(0, int(merge_within))

    def rank(start: int, score: float) -> tuple[float, int]:
        return (score, _head_match_len(flat, norm_phrase, start))

    best: list[tuple[int, float]] = []
    group_best = ordered[0]
    group_rank = rank(*ordered[0])
    prev_start = ordered[0][0]
    for start, score in ordered[1:]:
        if start - prev_start <= limit:
            current = rank(start, score)
            # 完全に同じ物差しなら先に出たほうを残す（並びを安定させるため）。
            if current > group_rank:
                group_best = (start, score)
                group_rank = current
        else:
            best.append(group_best)
            group_best = (start, score)
            group_rank = rank(start, score)
        prev_start = start
    best.append(group_best)
    return best


def _top_windows(
    flat: FlatText, norm_phrase: str, *, limit: int, min_gap: int
) -> list[tuple[int, float]]:
    """しきい値なしでスコア上位を取り直す（エラーメッセージ用）。

    `_merge_adjacent` はしきい値で候補が疎になっている前提の連結処理なので、
    全窓を渡すと全体が1グループに潰れてしまう。ここでは「上位から採り、
    すでに採った位置から min_gap 未満のものは飛ばす」方式で重なりを抑える。
    """
    scored = _scan_windows(flat, norm_phrase, 0.0)
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[1], item[0]))
    gap = max(1, int(min_gap))
    picked: list[tuple[int, float]] = []
    for start, score in scored:
        if any(abs(start - other) < gap for other, _ in picked):
            continue
        picked.append((start, score))
        if len(picked) >= max(1, int(limit)):
            break
    return picked


# ---------------------------------------------------------------------------
# 候補の組み立て（SPEC Step 3-5「時刻の取得」）
# ---------------------------------------------------------------------------


def _build_candidate(
    flat: FlatText, norm_start: int, norm_end: int, score: float
) -> AnchorCandidate | None:
    """正規化後の一致位置から AnchorCandidate を作る。単語に紐付かなければ None。

    start_time は先頭文字が属する単語の start、end_time は末尾文字が属する単語の end。
    「こ」「と」の発話開始時刻をカット点にしたいので、必ず単語の境界に戻して取る。
    """
    if not flat.words or not flat.norm:
        return None
    word_start = flat.word_index_at_norm(norm_start)
    word_last = flat.word_index_at_norm(norm_end - 1)
    if word_start < 0 or word_last < 0:
        return None
    if word_last < word_start:
        word_last = word_start

    return AnchorCandidate(
        score=float(score),
        norm_start=int(norm_start),
        norm_end=int(norm_end),
        word_start=int(word_start),
        word_end=int(word_last) + 1,  # 終端は排他
        start_time=float(flat.words[word_start].start),
        end_time=float(flat.words[word_last].end),
        matched_text=flat.raw_text_for_norm(norm_start, norm_end),
        context=flat.context_for_norm(norm_start, norm_end, CONTEXT_WIDTH),
    )


def find_candidates(
    flat: FlatText,
    phrase: str,
    threshold_score: float,
    *,
    merge_within: int = DEFAULT_MERGE_WITHIN,
) -> list[AnchorCandidate]:
    """SPEC Step 3-3。正規化済み flat をスライド窓で走査し、しきい値以上の候補を返す。

    窓はフレーズと同じ文字数。`threshold_score` は 0〜100（rapidfuzz と同じ尺度）で、
    「以上」を候補とする（1.0 を指定したときに完全一致が拾えるようにするため）。
    返り値は出現順（正規化後の開始位置の昇順）。
    """
    norm_phrase = normalize_phrase(phrase)
    if not norm_phrase:
        logger.warning("フレーズ「%s」は正規化すると空になります。候補なしとして扱います。", phrase)
        return []
    if not flat.norm:
        return []

    m = len(norm_phrase)
    if m > len(flat.norm):
        logger.debug(
            "フレーズ「%s」（正規化後%d文字）が文字起こし全体（%d文字）より長いため候補なし。",
            phrase,
            m,
            len(flat.norm),
        )
        return []

    scored = _scan_windows(flat, norm_phrase, float(threshold_score))
    merged = _merge_adjacent(scored, merge_within, flat=flat, norm_phrase=norm_phrase)

    candidates: list[AnchorCandidate] = []
    for start, score in merged:
        candidate = _build_candidate(flat, start, start + m, score)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: c.norm_start)
    logger.debug(
        "フレーズ「%s」: 窓ヒット %d 件 → 統合後 %d 件（しきい値 %.1f）",
        phrase,
        len(scored),
        len(candidates),
        float(threshold_score),
    )
    return candidates


# ---------------------------------------------------------------------------
# 候補の絞り込み（SPEC Step 3-4）
# ---------------------------------------------------------------------------


def _fmt_sec(value: float) -> str:
    """秒数をメッセージ用に短く整形する（末尾の 0 を落とす）。"""
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _must_follow_threshold(anchor: AnchorConfig) -> float:
    """must_follow のしきい値（0〜100）。未指定ならアンカー本体のしきい値を使う。"""
    must_follow = anchor.must_follow
    if must_follow is None or must_follow.fuzzy_threshold is None:
        return anchor.threshold_score
    return float(must_follow.fuzzy_threshold) * 100.0


def _slice_after(
    flat: FlatText, candidate: AnchorCandidate, word_starts: Sequence[float], within_sec: float
) -> FlatText:
    """候補の直後 within_sec 以内に始まる単語だけで、小さな FlatText を作り直す。

    元の単語列のスライスから `build_flat` し直すのが要点。flat 全体に対して
    位置で範囲を切ると、正規化で消えた文字のぶんだけ境界がずれるため。
    下限は候補の末尾単語の次（`word_end`）にする。ASR のタイムスタンプは
    まれに前後の単語と重なるので、時刻ではなく単語 index で切るほうが安定する。
    """
    begin = max(0, int(candidate.word_end))
    limit_time = float(candidate.end_time) + float(within_sec)
    end = bisect_right(word_starts, limit_time)
    if end < begin:
        end = begin
    return build_flat(flat.words[begin:end])


def filter_candidates(
    candidates: Sequence[AnchorCandidate], anchor: AnchorConfig, flat: FlatText
) -> tuple[list[AnchorCandidate], list[AnchorCandidate]]:
    """SPEC Step 3-4。探索窓と must_follow で候補を絞り、(残った候補, 落ちた候補) を返す。

    落ちた候補には `rejected_reason` を入れる。あとでエラーメッセージに
    「なぜ落ちたか」を出せるようにするため（候補は消さずに残す）。
    """
    kept: list[AnchorCandidate] = []
    rejected: list[AnchorCandidate] = []

    window = anchor.search_window_sec
    must_follow = anchor.must_follow
    # must_follow の探索範囲を二分探索で切るための開始時刻の一覧（必要なときだけ作る）。
    word_starts: list[float] = (
        [float(w.start) for w in flat.words] if must_follow is not None else []
    )
    mf_threshold = _must_follow_threshold(anchor)

    for candidate in candidates:
        # 1) 探索窓の外を除外
        if window is not None:
            lo, hi = window
            if not (lo <= candidate.start_time <= hi):
                candidate.rejected_reason = (
                    f"探索窓 [{_fmt_sec(lo)}, {_fmt_sec(hi)}] の外"
                    f"（start_time={_fmt_sec(candidate.start_time)}）"
                )
                rejected.append(candidate)
                continue

        # 2) must_follow: 直後 within_sec 以内に指定フレーズが続くかを同じあいまい一致で確認
        if must_follow is not None:
            tail = _slice_after(flat, candidate, word_starts, must_follow.within_sec)
            hits = find_candidates(tail, must_follow.phrase, mf_threshold)
            if not hits:
                candidate.rejected_reason = (
                    f"must_follow '{must_follow.phrase}' が"
                    f" {_fmt_sec(must_follow.within_sec)} 秒以内に見つからない"
                )
                rejected.append(candidate)
                continue

        kept.append(candidate)

    logger.debug(
        "アンカー %s: 絞り込み %d 件 → 残り %d 件 / 除外 %d 件",
        anchor.id,
        len(candidates),
        len(kept),
        len(rejected),
    )
    return (kept, rejected)


# ---------------------------------------------------------------------------
# 候補の確定（SPEC Step 3-4「occurrence に従って1つを確定」）
# ---------------------------------------------------------------------------


def select_candidate(
    candidates: Sequence[AnchorCandidate], anchor: AnchorConfig
) -> AnchorCandidate:
    """occurrence（first / last / nth）に従って候補を1つに決める。

    0件なら AnchorNotFoundError。勝手に代替位置を選ばない（SPEC 9章）。
    詳しい文脈付きのメッセージが要る呼び出し側は `build_error_message` を使う。
    """
    if not candidates:
        raise AnchorNotFoundError(
            f"アンカー '{anchor.id}'（phrase=「{anchor.phrase}」）の候補が0件です。\n{ERROR_ADVICE}"
        )

    ordered = sorted(candidates, key=lambda c: (c.start_time, c.norm_start))
    occurrence = anchor.occurrence
    if occurrence == "first":
        return ordered[0]
    if occurrence == "last":
        return ordered[-1]
    if occurrence == "nth":
        index = int(anchor.nth or 1)
        if index < 1 or index > len(ordered):
            raise AnchorNotFoundError(
                f"アンカー '{anchor.id}' は occurrence='nth' で {index} 番目を求めていますが、"
                f" 候補は {len(ordered)} 件しかありません（phrase=「{anchor.phrase}」）。\n"
                f"  nth を 1〜{len(ordered)} の範囲にするか、{ERROR_ADVICE}"
            )
        return ordered[index - 1]

    # config.py が検証済みなので通常は届かない。届いたら黙って first にせず止める。
    raise AnchorNotFoundError(
        f"アンカー '{anchor.id}' の occurrence が不正です: {occurrence!r}"
        "（first / last / nth のいずれかにしてください）。"
    )


def raw_cut_time_of(candidate: AnchorCandidate, anchor: AnchorConfig) -> float:
    """SPEC Step 3-5。cut='before' なら候補の start_time、'after' なら end_time。"""
    return float(candidate.end_time) if anchor.cut == "after" else float(candidate.start_time)


# ---------------------------------------------------------------------------
# 失敗時のメッセージ（SPEC Step 3「失敗時の挙動」／SPEC 9章）
# ---------------------------------------------------------------------------


def _describe_conditions(anchor: AnchorConfig) -> str:
    """絞り込み条件を1行で説明する。落ちた原因が条件側のこともあるため。"""
    parts = [f"occurrence={anchor.occurrence}"]
    if anchor.occurrence == "nth":
        parts[0] += f"（{anchor.nth}番目）"
    if anchor.search_window_sec is not None:
        lo, hi = anchor.search_window_sec
        parts.append(f"search_window_sec=[{_fmt_sec(lo)}, {_fmt_sec(hi)}]")
    if anchor.must_follow is not None:
        parts.append(
            f"must_follow=「{anchor.must_follow.phrase}」"
            f"（{_fmt_sec(anchor.must_follow.within_sec)}秒以内）"
        )
    parts.append(f"cut={anchor.cut}")
    return " / ".join(parts)


def _fallback_candidates(
    flat: FlatText, norm_phrase: str, *, limit: int = TOP_N_ON_ERROR
) -> list[AnchorCandidate]:
    """しきい値なしで上位候補を取り直す。

    「しきい値を大きく下回るものすら無い」ときでもエラーメッセージを出せるようにする経路。
    """
    if not norm_phrase or not flat.norm:
        return []
    m = len(norm_phrase)
    top = _top_windows(flat, norm_phrase, limit=limit, min_gap=max(DEFAULT_MERGE_WITHIN + 1, m))
    out: list[AnchorCandidate] = []
    for start, score in top:
        candidate = _build_candidate(flat, start, start + m, score)
        if candidate is not None:
            out.append(candidate)
    return out


def build_error_message(
    anchor: AnchorConfig, all_candidates: Sequence[AnchorCandidate], flat: FlatText
) -> str:
    """アンカー未検出のエラー本文を組み立てる。

    しきい値を下回った候補も含めて上位3件を「スコア / 一致テキスト / 時刻 / 前後30文字の文脈」で並べ、
    次の一手（しきい値を下げる or phrase を直す）を必ず示す。
    """
    norm_phrase = normalize_phrase(anchor.phrase)
    pool = list(all_candidates)
    used_fallback = False
    if not pool:
        pool = _fallback_candidates(flat, norm_phrase)
        used_fallback = True
    pool.sort(key=lambda c: (-c.score, c.start_time))
    top = pool[:TOP_N_ON_ERROR]

    lines: list[str] = [
        f"アンカー '{anchor.id}' が見つかりませんでした（phrase=「{anchor.phrase}」）。",
        f"  正規化後のフレーズ: 「{norm_phrase}」（{len(norm_phrase)}文字）",
        f"  しきい値: fuzzy_threshold={anchor.fuzzy_threshold}"
        f"（スコア {anchor.threshold_score:.1f} 以上）",
        f"  絞り込み条件: {_describe_conditions(anchor)}",
    ]

    if not flat.norm:
        lines.append("  文字起こしに単語がありません。Step 2 の結果（work/transcript.json）を確認してください。")
    elif norm_phrase and len(norm_phrase) > len(flat.norm):
        lines.append(
            f"  フレーズ（{len(norm_phrase)}文字）が文字起こし全体（{len(flat.norm)}文字）より長いため、"
            "一致する窓がありません。"
        )

    if not top:
        lines.append("  スコア上位の候補: なし")
    else:
        header = (
            f"  スコア上位{len(top)}件（しきい値未満も含む）:"
            if used_fallback
            else f"  検出した候補のうちスコア上位{len(top)}件:"
        )
        lines.append(header)
        for rank, candidate in enumerate(top, start=1):
            lines.append(
                f"    {rank}. スコア {candidate.score:.1f}"
                f" / 一致テキスト「{candidate.matched_text}」"
                f" / 時刻 {_fmt_sec(candidate.start_time)}秒（{fmt_timestamp(candidate.start_time)}）"
            )
            lines.append(f"       文脈: {candidate.context}")
            if candidate.rejected_reason:
                lines.append(f"       除外理由: {candidate.rejected_reason}")

    lines.append(f"  {ERROR_ADVICE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# アンカー同士の順序（SPEC 9章「アンカーBがAより前 → 停止」）
# ---------------------------------------------------------------------------


def check_anchor_order(
    anchors: Sequence[AnchorConfig], results: Mapping[str, AnchorResult]
) -> None:
    """config の anchors の並び順に対して raw_cut_time が単調増加かを確かめる。

    逆転していたら設定ミスの可能性が高いので、両方の時刻を出して止める。
    """
    previous: AnchorResult | None = None
    for anchor in anchors:
        current = results.get(anchor.id)
        if current is None:
            continue
        if previous is not None and current.raw_cut_time < previous.raw_cut_time:
            raise AnchorOrderError(
                f"アンカー {current.id} ({current.raw_cut_time:.2f}s) が"
                f"アンカー {previous.id} ({previous.raw_cut_time:.2f}s) より前にあります。\n"
                f"  {current.id} の一致テキスト: 「{current.matched_text}」"
                f"（{fmt_timestamp(current.raw_cut_time)}）\n"
                f"  {previous.id} の一致テキスト: 「{previous.matched_text}」"
                f"（{fmt_timestamp(previous.raw_cut_time)}）\n"
                "  config の anchors の並び順、occurrence、search_window_sec を確認してください。"
            )
        previous = current


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def run(ctx: RunContext, transcript: Transcript) -> dict[str, AnchorResult]:
    """SPEC Step 3 の 1〜5 を順に行い、work/anchors.json を書いて結果を返す。"""
    words = transcript.words()
    flat = build_flat(words)  # 1) フラット化 + 2) 正規化（FlatText が両方を持つ）
    logger.info(
        "文字起こしを連結しました: 単語 %d 個 / 生 %d 文字 / 正規化後 %d 文字",
        len(words),
        len(flat.raw),
        len(flat.norm),
    )

    results: dict[str, AnchorResult] = {}
    for anchor in ctx.config.anchors:
        # 3) あいまい一致
        candidates = find_candidates(flat, anchor.phrase, anchor.threshold_score)
        # 4) 絞り込み
        kept, rejected = filter_candidates(candidates, anchor, flat)
        if not kept:
            raise AnchorNotFoundError(build_error_message(anchor, candidates, flat))
        chosen = select_candidate(kept, anchor)

        # 5) 時刻の取得
        raw_cut_time = raw_cut_time_of(chosen, anchor)
        word_index = chosen.word_end - 1 if anchor.cut == "after" else chosen.word_start

        results[anchor.id] = AnchorResult(
            id=anchor.id,
            phrase=anchor.phrase,
            matched_text=chosen.matched_text,
            score=chosen.score,
            raw_cut_time=raw_cut_time,
            candidates_found=len(candidates),
            candidates_rejected=len(rejected),
            context=chosen.context,
            word_index=word_index,
            rejected=rejected,
        )
        logger.info(
            "アンカー %s「%s」: 候補 %d 件（除外 %d 件）→ 採用 スコア %.1f /"
            " 一致テキスト「%s」/ raw_cut_time %.3f秒（%s）",
            anchor.id,
            anchor.phrase,
            len(candidates),
            len(rejected),
            chosen.score,
            chosen.matched_text,
            raw_cut_time,
            fmt_timestamp(raw_cut_time),
        )

    check_anchor_order(ctx.config.anchors, results)
    save(ctx, results)
    return results


def save(ctx: RunContext, result: Mapping[str, AnchorResult]) -> None:
    """work/<episode_id>/anchors.json に書く（SPEC Step 3 の出力例の形）。"""
    payload = {anchor_id: anchor.to_dict() for anchor_id, anchor in result.items()}
    path = ctx.work_path(ANCHORS_FILE)
    write_json(path, payload)
    logger.debug("アンカーを保存しました: %s（%d 件）", path, len(payload))


def load(ctx: RunContext) -> dict[str, AnchorResult]:
    """work/<episode_id>/anchors.json を読む（--from-step での再開用）。"""
    path = ctx.work_path(ANCHORS_FILE)
    if not path.exists():
        raise MissingArtifactError(
            f"アンカーの中間ファイルがありません: {path}\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step {STEP}` を実行してください"
            "（Step 3 は Step 2 の transcript.json を使います）。"
        )
    try:
        data = read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissingArtifactError(
            f"アンカーの中間ファイルの JSON が壊れています: {path}\n  {exc}\n"
            f"  このファイルを消して `radio-cutter run {ctx.input_path} --from-step {STEP}` で作り直してください。"
        ) from exc
    except OSError as exc:
        raise MissingArtifactError(f"アンカーの中間ファイルを読めませんでした: {path}\n  {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise MissingArtifactError(
            f"アンカーの中間ファイルの中身が空か、想定した形（アンカーIDをキーにしたオブジェクト）ではありません: {path}"
        )

    results: dict[str, AnchorResult] = {}
    for anchor_id, payload in data.items():
        if not isinstance(payload, dict):
            raise MissingArtifactError(
                f"アンカー '{anchor_id}' の内容がオブジェクトではありません: {path}"
            )
        try:
            results[str(anchor_id)] = AnchorResult.from_dict(str(anchor_id), payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise MissingArtifactError(
                f"アンカー '{anchor_id}' の内容が不正です: {path}\n  {exc}"
            ) from exc
    return results
