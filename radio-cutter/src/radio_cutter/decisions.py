"""SPEC 8章「decisions.json スキーマ」に対応。全ステップの判断を1ファイルに集約する。

ここは「各ステップの結果オブジェクトを受け取って JSON にする」だけを担う純粋な組み立て役。
ffmpeg も LLM も叩かない（唯一の I/O は入力ファイルの SHA-256 計算と書き出し）。
値が取れなかった部分は null を書かずキーごと省く。あとから読む人が
「取れなかった」と「0だった」を取り違えないようにするため。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .context import RunContext
from .errors import RadioCutterError
from .logging_util import get_logger
from .models import (
    AnchorResult,
    CutPoint,
    HighlightCandidate,
    HighlightResult,
    RenderResult,
    r3,
    write_json,
)
from .util.cache import sha256_file
from .util.ffmpeg import MediaInfo
from .util.timeline import resolve_segment_bounds

logger = get_logger(__name__)

#: out/<episode_id>/ に書くファイル名
DECISIONS_FILE = "decisions.json"

#: durations のキー。セグメントは config.segments の name をそのまま使う（決め打ちしない）。
HIGHLIGHT_KEY = "highlight"
FINAL_KEY = "final"

#: durations が実測値か想定値かを示す値
SOURCE_MEASURED = "measured"
SOURCE_ESTIMATED = "estimated"

#: スコアの丸め桁数（models.AnchorResult / HighlightCandidate と揃える）
SCORE_PRECISION = 2


# ---------------------------------------------------------------------------
# 時刻
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """現在時刻をローカルタイムゾーン付き ISO8601 で返す（例 "2026-08-30T14:20:11+09:00"）。

    decisions.json の `generated_at` に入れる。秒未満は落とす（読みやすさ優先）。
    """
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def decisions_path(ctx: RunContext) -> Path:
    """decisions.json の置き場（out/<episode_id>/decisions.json）。"""
    return ctx.out_path(DECISIONS_FILE)


# ---------------------------------------------------------------------------
# 小さなヘルパ
# ---------------------------------------------------------------------------


def _is_usable(value: Any) -> bool:
    """秒数として書ける値か（None・NaN・無限大を弾く）。"""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _score(value: Any) -> float | None:
    """スコアを丸めて返す。数値でなければ None（キーごと省くため）。"""
    if not _is_usable(value):
        return None
    return round(float(value), SCORE_PRECISION)


def _absolute_input(ctx: RunContext) -> str:
    """入力ファイルの絶対パス文字列。存在しなくても絶対パスにする。"""
    p = Path(ctx.input_path).expanduser()
    try:
        return str(p.resolve())
    except OSError:  # シンボリックリンクの循環など。絶対化だけして諦める。
        return str(p.absolute())


def _resolve_input_sha256(ctx: RunContext, input_sha256: str | None) -> str | None:
    """SHA-256 を決める。渡されていればそれを使い、無ければ入力から計算する。

    入力ファイルが無い／読めないときは None を返す（decisions.json 自体は必ず書きたいため）。
    """
    if input_sha256:
        return str(input_sha256)
    path = Path(ctx.input_path)
    if not path.is_file():
        logger.debug("入力ファイルが無いため input_sha256 を省略します: %s", path)
        return None
    try:
        return sha256_file(path)
    except RadioCutterError as exc:
        logger.warning("入力ファイルの SHA-256 を計算できませんでした: %s", exc)
        return None


def _total_duration(media: MediaInfo | None) -> float | None:
    """入力動画の総尺。probe できていなければ None。"""
    if media is None:
        return None
    if not _is_usable(media.duration):
        return None
    value = float(media.duration)
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------


def _anchor_order(
    ctx: RunContext,
    anchors: Mapping[str, AnchorResult] | None,
    cuts: Mapping[str, CutPoint] | None,
) -> list[str]:
    """出力するアンカーIDの並び。config の順を優先し、config に無いIDは後ろに回す。"""
    order: list[str] = []
    seen: set[str] = set()
    for anchor_id in ctx.config.anchor_ids():
        if anchor_id not in seen:
            seen.add(anchor_id)
            order.append(anchor_id)
    for mapping in (anchors, cuts):
        for anchor_id in mapping or {}:
            if anchor_id not in seen:
                seen.add(anchor_id)
                order.append(anchor_id)
    return [
        anchor_id
        for anchor_id in order
        if (anchors is not None and anchor_id in anchors) or (cuts is not None and anchor_id in cuts)
    ]


def _anchor_entry(anchor: AnchorResult | None, cut: CutPoint | None) -> dict[str, Any] | None:
    """アンカー1つ分（SPEC 8章）。cuts がまだ無ければ raw_cut_time だけを入れる。"""
    if anchor is None and cut is None:
        return None

    entry: dict[str, Any] = {}

    raw = anchor.raw_cut_time if anchor is not None else (cut.raw_cut_time if cut is not None else None)
    if _is_usable(raw):
        entry["raw_cut_time"] = r3(float(raw))

    if cut is not None and _is_usable(cut.cut_time):
        entry["cut_time"] = r3(float(cut.cut_time))
        entry["silence_found"] = bool(cut.silence_found)

    score = _score(anchor.score) if anchor is not None else None
    if score is None and cut is not None and cut.score:
        score = _score(cut.score)
    if score is not None:
        entry["score"] = score

    return entry or None


def _build_anchors(
    ctx: RunContext,
    anchors: Mapping[str, AnchorResult] | None,
    cuts: Mapping[str, CutPoint] | None,
) -> dict[str, Any]:
    """anchors と cuts を突き合わせて decisions.json の "anchors" を作る。"""
    out: dict[str, Any] = {}
    for anchor_id in _anchor_order(ctx, anchors, cuts):
        anchor = (anchors or {}).get(anchor_id)
        cut = (cuts or {}).get(anchor_id)
        entry = _anchor_entry(anchor, cut)
        if entry:
            out[anchor_id] = entry
    return out


# ---------------------------------------------------------------------------
# highlight
# ---------------------------------------------------------------------------


def _candidate_entry(candidate: HighlightCandidate) -> dict[str, Any]:
    """ハイライト候補1件（start / end / score / reason）。空の reason は省く。"""
    entry: dict[str, Any] = {}
    if _is_usable(candidate.start):
        entry["start"] = r3(float(candidate.start))
    if _is_usable(candidate.end):
        entry["end"] = r3(float(candidate.end))
    score = _score(candidate.score)
    if score is not None:
        entry["score"] = score
    reason = str(candidate.reason or "").strip()
    if reason:
        entry["reason"] = reason
    return entry


def _build_highlight(highlight: HighlightResult | None) -> dict[str, Any]:
    """decisions.json の "highlight"（採用・次点・スナップ前の区間）。"""
    if highlight is None:
        return {}

    out: dict[str, Any] = {}
    selected = _candidate_entry(highlight.selected)
    if selected:
        out["selected"] = selected

    alternatives = [_candidate_entry(c) for c in highlight.alternatives]
    alternatives = [a for a in alternatives if a]
    if alternatives:
        out["alternatives"] = alternatives

    snapped = highlight.snapped_from
    snapped_entry: dict[str, Any] = {}
    if snapped is not None:
        if _is_usable(snapped.start):
            snapped_entry["start"] = r3(float(snapped.start))
        if _is_usable(snapped.end):
            snapped_entry["end"] = r3(float(snapped.end))
    if snapped_entry:
        out["snapped_from"] = snapped_entry

    return out


# ---------------------------------------------------------------------------
# durations
# ---------------------------------------------------------------------------


def _duration_key_order(ctx: RunContext) -> list[str]:
    """durations のキー順。SPEC 8章の例に合わせて highlight → 各セグメント → final。"""
    return [HIGHLIGHT_KEY, *(seg.name for seg in ctx.config.segments), FINAL_KEY]


def _ordered_durations(ctx: RunContext, values: Mapping[str, float]) -> dict[str, float]:
    """既知のキーを SPEC の順に並べ、config に無いキーは後ろに回して返す。"""
    out: dict[str, float] = {}
    for key in _duration_key_order(ctx):
        if key in values and _is_usable(values[key]):
            out[key] = r3(float(values[key]))
    for key in sorted(values):
        if key not in out and _is_usable(values[key]):
            out[key] = r3(float(values[key]))
    return out


def _measured_durations(ctx: RunContext, render: RenderResult | None) -> dict[str, float]:
    """Step 7 が ffprobe で測った実尺。"""
    if render is None or not render.durations:
        return {}
    return _ordered_durations(ctx, render.durations)


def _estimated_durations(
    ctx: RunContext,
    cuts: Mapping[str, CutPoint] | None,
    highlight: HighlightResult | None,
    total_duration: float | None,
) -> dict[str, float]:
    """まだ書き出していない段階（--dry-run など）の想定尺を cuts と highlight から計算する。"""
    values: dict[str, float] = {}

    highlight_dur: float | None = None
    if highlight is not None and _is_usable(highlight.duration) and float(highlight.duration) > 0:
        highlight_dur = float(highlight.duration)
        values[HIGHLIGHT_KEY] = highlight_dur

    segment_total = 0.0
    all_segments_resolved = bool(ctx.config.segments)
    if cuts:
        for seg in ctx.config.segments:
            try:
                start, end = resolve_segment_bounds(seg, cuts, float(total_duration or 0.0))
            except RadioCutterError as exc:
                all_segments_resolved = False
                logger.debug("セグメント '%s' の想定尺を計算できませんでした: %s", seg.name, exc)
                continue
            duration = float(end) - float(start)
            values[seg.name] = duration
            segment_total += duration
    else:
        all_segments_resolved = False

    if all_segments_resolved and highlight_dur is not None:
        values[FINAL_KEY] = highlight_dur + segment_total

    return _ordered_durations(ctx, values)


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------


def _carry_over(
    previous: Mapping[str, Any] | None, input_sha256: Any
) -> tuple[list[Any], list[str]] | None:
    """前回の decisions.json から llm_calls と warnings を引き継ぐ（同じ入力のときだけ）。

    入力ファイルが違えば別のエピソードの記録なので混ぜない。
    SHA-256 が両方に載っているときだけ突き合わせる。
    """
    if not isinstance(previous, Mapping):
        return None
    old_sha = previous.get("input_sha256")
    if old_sha and input_sha256 and old_sha != input_sha256:
        return None
    old_calls = previous.get("llm_calls")
    old_warnings = previous.get("warnings")
    calls = [c for c in old_calls if isinstance(c, dict)] if isinstance(old_calls, list) else []
    notes = [str(w) for w in old_warnings] if isinstance(old_warnings, list) else []
    if not calls and not notes:
        return None
    return (calls, notes)


def build_decisions(
    ctx: RunContext,
    *,
    media: MediaInfo | None,
    anchors: Mapping[str, AnchorResult] | None,
    cuts: Mapping[str, CutPoint] | None,
    highlight: HighlightResult | None,
    render: RenderResult | None,
    generated_at: str,
    input_sha256: str | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict:
    """SPEC 8章の decisions.json を組み立てて dict で返す。

    途中のステップまでしか回っていなくても落ちない（取れた分だけ書く）。
    warnings と llm_calls だけは空でも必ず出す。「何も無かった」ことを記録として残すため。

    `previous` に前回の decisions.json を渡すと、同じ入力ファイルに対する記録として
    llm_calls と warnings を引き継ぐ。`--dry-run` で判断を確かめてから
    `--from-step 7` で書き出す、という普段の流れだと後半の実行では LLM を呼ばないので、
    引き継がないと「何回どのモデルを叩いたか」も前半で出た警告も消えてしまう。
    """
    payload: dict[str, Any] = {
        "episode_id": ctx.episode_id,
        "input": _absolute_input(ctx),
    }

    sha = _resolve_input_sha256(ctx, input_sha256)
    if sha:
        payload["input_sha256"] = sha

    total_duration = _total_duration(media)
    if total_duration is not None:
        payload["duration"] = r3(total_duration)

    payload["generated_at"] = str(generated_at)

    anchors_out = _build_anchors(ctx, anchors, cuts)
    if anchors_out:
        payload["anchors"] = anchors_out

    highlight_out = _build_highlight(highlight)
    if highlight_out:
        payload["highlight"] = highlight_out

    durations = _measured_durations(ctx, render)
    source = SOURCE_MEASURED
    if not durations:
        durations = _estimated_durations(ctx, cuts, highlight, total_duration)
        source = SOURCE_ESTIMATED
    if durations:
        payload["durations"] = durations
        payload["durations_source"] = source

    llm_calls = [record.to_dict() for record in ctx.llm_calls]
    warnings = list(ctx.warnings)
    if render is not None:
        # Step 7 の検算の警告は render.json にも残る。中間ファイルから読んだ回でも拾う。
        for message in render.warnings:
            if message not in warnings:
                warnings.append(message)

    carried = _carry_over(previous, payload.get("input_sha256"))
    if carried is not None:
        old_calls, old_warnings = carried
        llm_calls = [*old_calls, *llm_calls]
        warnings = [*[w for w in old_warnings if w not in warnings], *warnings]

    payload["llm_calls"] = llm_calls
    payload["warnings"] = warnings

    return payload


def write_decisions(ctx: RunContext, payload: dict) -> Path:
    """decisions.json を out/<episode_id>/ に書き、そのパスを返す。"""
    path = decisions_path(ctx)
    try:
        write_json(path, payload)
    except OSError as exc:
        raise RadioCutterError(f"decisions.json を書けませんでした: {path}\n{exc}") from exc
    logger.info("判断ログを書きました: %s", path)
    return path
