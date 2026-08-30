"""SPEC Step 7「書き出し」。ハイライトと各セグメントを再エンコードし、concat で final.mp4 を作る。

方針（SPEC Step 7・9章）：
- `-c copy` は最寄りのキーフレームまでカット位置がずれるので使わない。必ず再エンコードする。
- 映像コーデックは**最初に一度だけ**決め、すべての書き出しで同じパラメータを使う。
  concat demuxer は「同一パラメータでエンコード済み」が前提なので、ここがずれると連結が壊れる。
- final.mp4 の実尺は ffprobe で検算する。ずれても止めず、警告として decisions.json に残す
  （書き出し自体は成功しているので、判断はあとから人がやればよい）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..context import RunContext
from ..errors import MissingArtifactError, RenderError
from ..logging_util import fmt_duration, get_logger
from ..models import CutPoint, HighlightResult, RenderResult, r3, read_json, write_json
from ..util.ffmpeg import (
    MediaInfo,
    choose_video_codec,
    concat_files,
    encode_segment,
    media_duration,
    probe_media,
    require_binaries,
)
from ..util.timeline import clamp, fmt_timestamp, resolve_segment_bounds

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 7
NAME: str = "書き出し"

#: work/<episode_id>/ に書く中間ファイル
RENDER_FILE = "render.json"
OUTPUTS: tuple[str, ...] = (RENDER_FILE,)

#: RenderResult.files / durations のキー。セグメントは config の name をそのまま使う。
HIGHLIGHT_KEY = "highlight"
FINAL_KEY = "final"

#: out/<episode_id>/ に書く連結済み動画と、work/<episode_id>/ に書く concat リスト
FINAL_FILE = "final.mp4"
CONCAT_LIST_NAME = "concat.txt"


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def render_path(ctx: RunContext) -> Path:
    """書き出し結果の中間ファイル（decisions.json の durations がここから来る）。"""
    return ctx.work_path(RENDER_FILE)


def final_path(ctx: RunContext) -> Path:
    """連結済みの完成動画。"""
    return ctx.out_path(FINAL_FILE)


# ---------------------------------------------------------------------------
# 純関数（テストが直接叩く）
# ---------------------------------------------------------------------------


def clamp_span(label: str, start: float, end: float, total_duration: float) -> tuple[float, float]:
    """書き出し区間を [0, 総尺] に収める。収めた結果が空になるなら RenderError。

    アンカーやハイライトが動画の端に寄っていると `-ss` が尺の外に出て、
    ffmpeg が黙って0秒のファイルを吐くことがある。ここで先に気づけるようにする。
    """
    total = float(total_duration)
    if total <= 0:
        raise RenderError(f"{label}: 入力の総尺が取得できていないため書き出せません（{r3(total)}秒）。")
    lo = clamp(float(start), 0.0, total)
    hi = clamp(float(end), 0.0, total)
    if hi - lo <= 0:
        raise RenderError(
            f"{label}: 書き出す区間が空です（{r3(start)}〜{r3(end)}秒 / 総尺 {r3(total)}秒）。\n"
            "  アンカーの検出位置（work/anchors.json・work/cuts.json）を確認してください。"
        )
    if (lo, hi) != (float(start), float(end)):
        logger.warning(
            "%s: 区間 %.3f〜%.3f秒 が動画の外に出ていたので %.3f〜%.3f秒 に切り詰めました。",
            label,
            start,
            end,
            lo,
            hi,
        )
    return (lo, hi)


def concat_order(
    position: str, highlight_file: Path, segment_files: Sequence[Path]
) -> list[Path]:
    """連結順を決める。`prepend` ならハイライトが先頭、`append` なら末尾（SPEC 5章 highlight.position）。"""
    files = list(segment_files)
    if position == "append":
        return [*files, highlight_file]
    return [highlight_file, *files]


def duration_gap_warning(expected: float, actual: float, tolerance: float) -> str | None:
    """想定合計と実尺の差が許容を超えていれば警告文を返す（SPEC Step 7 の検算）。

    例外にはしない。書き出し自体は終わっているので、あとから人が判断できるよう
    decisions.json の warnings に残すのが目的。
    """
    gap = float(actual) - float(expected)
    if abs(gap) <= float(tolerance):
        return None
    return (
        f"{FINAL_FILE} の実尺 {r3(actual)}秒 が想定合計 {r3(expected)}秒 と "
        f"{r3(abs(gap))}秒 ずれています（許容 {r3(float(tolerance))}秒）。"
        " 連結した動画の切れ目を確認してください。"
    )


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def _duration_keys(ctx: RunContext) -> list[str]:
    """想定合計を出すためのキー順（連結順と同じ並び）。"""
    names = [seg.name for seg in ctx.config.segments]
    if ctx.config.highlight.position == "append":
        return [*names, HIGHLIGHT_KEY]
    return [HIGHLIGHT_KEY, *names]


def _check_key_collision(ctx: RunContext) -> None:
    """セグメント名が予約キーと衝突していないか確かめる。

    durations は {"highlight", <セグメント名>…, "final"} を1つの辞書に混ぜるので、
    セグメント名がこの2つと同じだと decisions.json の尺が上書きされてしまう。
    """
    reserved = {HIGHLIGHT_KEY, FINAL_KEY}
    clashes = [seg.name for seg in ctx.config.segments if seg.name in reserved]
    if clashes:
        raise RenderError(
            f"セグメント名が予約語と衝突しています: {', '.join(clashes)}\n"
            f"  '{HIGHLIGHT_KEY}' と '{FINAL_KEY}' は decisions.json の durations で使うため、"
            "config の segments[].name を別の名前にしてください。"
        )


def _resolve_total_duration(ctx: RunContext, media: MediaInfo | None) -> float:
    """入力の総尺。probe.json 由来の MediaInfo が無ければ ffprobe で取り直す。"""
    if media is not None and float(media.duration) > 0:
        return float(media.duration)
    logger.debug("総尺が渡されなかったので ffprobe で取り直します: %s", ctx.input_path)
    return float(probe_media(ctx.input_path).duration)


def _highlight_bounds(highlight: HighlightResult, total_duration: float) -> tuple[float, float]:
    """ハイライトの書き出し区間（Step 5 が選んだ selected の start/end）。"""
    selected = highlight.selected
    return clamp_span("ハイライト", float(selected.start), float(selected.end), total_duration)


def _encode_one(
    ctx: RunContext,
    label: str,
    key: str,
    out_name: str,
    start: float,
    end: float,
    *,
    use_fallback: bool,
    result: RenderResult,
) -> Path:
    """区間を1本書き出し、実尺を ffprobe で測って RenderResult に記録する。"""
    dst = ctx.out_path(out_name)
    requested = end - start
    logger.info(
        "%sを書き出します: %s〜%s（%.3f〜%.3f秒 / %s）→ %s",
        label,
        fmt_timestamp(start),
        fmt_timestamp(end),
        start,
        end,
        fmt_duration(requested),
        dst,
    )
    encode_segment(ctx.input_path, start, end, dst, ctx.config.render, use_fallback=use_fallback)

    actual = media_duration(dst)
    result.files[key] = str(dst)
    result.durations[key] = actual
    tolerance = float(ctx.config.render.duration_tolerance_sec)
    if abs(actual - requested) > tolerance:
        message = (
            f"{out_name} の実尺 {r3(actual)}秒 が指定した区間 {r3(requested)}秒 と "
            f"{r3(abs(actual - requested))}秒 ずれています（許容 {r3(tolerance)}秒）。"
        )
        ctx.warn(message)
        result.warnings.append(message)
        logger.warning("%s", message)
    logger.info("%sを書き出しました: %s（%s）", label, dst.name, fmt_duration(actual))
    return dst


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def run(
    ctx: RunContext,
    cuts: Mapping[str, CutPoint],
    highlight: HighlightResult,
    media: MediaInfo | None = None,
) -> RenderResult:
    """ハイライトと各セグメントを書き出し、連結して final.mp4 を作る（SPEC Step 7）。"""
    ctx.ensure_dirs()
    require_binaries()

    if highlight is None:
        raise MissingArtifactError(
            "ハイライトがありません。\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step 5` で"
            "Step 5（ハイライト選定）を実行してください。"
        )
    if not cuts:
        raise MissingArtifactError(
            "カット点が1つもありません。\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step 4` で"
            "Step 4（カット点の精密化）を実行してください。"
        )

    _check_key_collision(ctx)
    render_cfg = ctx.config.render
    total_duration = _resolve_total_duration(ctx, media)

    # コーデックの決定はここ1回だけ。すべての書き出しで同じパラメータを使わないと
    # concat demuxer（-c copy）で連結できない。
    codec, extra_args, used_fallback = choose_video_codec(render_cfg)
    if used_fallback:
        message = (
            f"映像コーデック {render_cfg.video_codec} が使えないため "
            f"{render_cfg.fallback_video_codec} にフォールバックしました（CPU エンコードになります）。"
        )
        ctx.warn(message)
        logger.warning("%s", message)
    logger.info(
        "書き出し設定: 映像 %s %s / 音声 %s %s",
        codec,
        " ".join(extra_args),
        render_cfg.audio_codec,
        render_cfg.audio_bitrate,
    )

    result = RenderResult(used_fallback_codec=used_fallback)

    # 1) ハイライト
    hl_cfg = ctx.config.highlight
    hl_start, hl_end = _highlight_bounds(highlight, total_duration)
    highlight_file = _encode_one(
        ctx,
        "ハイライト",
        HIGHLIGHT_KEY,
        hl_cfg.file,
        hl_start,
        hl_end,
        use_fallback=used_fallback,
        result=result,
    )

    # 2) 各セグメント（config の順）
    segment_files: list[Path] = []
    for seg in ctx.config.segments:
        start, end = resolve_segment_bounds(seg, cuts, total_duration)
        start, end = clamp_span(f"セグメント '{seg.name}'", start, end, total_duration)
        segment_files.append(
            _encode_one(
                ctx,
                f"セグメント '{seg.name}'",
                seg.name,
                seg.file,
                start,
                end,
                use_fallback=used_fallback,
                result=result,
            )
        )

    # 3) 連結（3本とも同一パラメータでエンコード済みなので -c copy でよい）
    order = concat_order(hl_cfg.position, highlight_file, segment_files)
    logger.info(
        "連結します（%s）: %s → %s",
        "ハイライトを先頭" if hl_cfg.position != "append" else "ハイライトを末尾",
        " + ".join(p.name for p in order),
        FINAL_FILE,
    )
    final = concat_files(order, final_path(ctx), ctx.work_dir, list_name=CONCAT_LIST_NAME)

    # 4) 実尺の検算（SPEC Step 7）。ずれても止めず警告に残す。
    expected_total = sum(result.durations[key] for key in _duration_keys(ctx))
    actual_total = media_duration(final)
    result.files[FINAL_KEY] = str(final)
    result.durations[FINAL_KEY] = actual_total

    logger.info(
        "%s: 実尺 %s（%.3f秒）／想定合計 %.3f秒",
        FINAL_FILE,
        fmt_duration(actual_total),
        actual_total,
        expected_total,
    )
    warning = duration_gap_warning(expected_total, actual_total, render_cfg.duration_tolerance_sec)
    if warning is not None:
        ctx.warn(warning)
        result.warnings.append(warning)
        logger.warning("%s", warning)

    save(ctx, result)
    return result


def save(ctx: RunContext, result: RenderResult) -> None:
    """work/<episode_id>/render.json に書く（run() の中から呼ぶ）。"""
    path = render_path(ctx)
    write_json(path, result.to_dict())
    logger.debug("書き出し結果を保存しました: %s（%d 本）", path, len(result.files))


def load(ctx: RunContext) -> RenderResult:
    """work/<episode_id>/render.json を読む（--from-step での再開用）。"""
    path = render_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"書き出し結果の中間ファイルがありません: {path}\n"
            f"  先に `radio-cutter run {ctx.input_path} --from-step {STEP}` を実行してください"
            f"（Step {STEP} は Step 4 の cuts.json と Step 5 の highlight.json を使います）。"
        )
    try:
        data = read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissingArtifactError(
            f"書き出し結果の中間ファイルの JSON が壊れています: {path}\n  {exc}\n"
            f"  このファイルを消して `radio-cutter run {ctx.input_path} --from-step {STEP}` で作り直してください。"
        ) from exc
    except OSError as exc:
        raise MissingArtifactError(
            f"書き出し結果の中間ファイルを読めませんでした: {path}\n  {exc}"
        ) from exc

    if not isinstance(data, dict) or "files" not in data:
        raise MissingArtifactError(
            f"書き出し結果の中間ファイルの中身が想定した形（files を持つオブジェクト）ではありません: {path}"
        )
    try:
        result = RenderResult.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingArtifactError(
            f"書き出し結果の中間ファイルの内容が不正です: {path}\n  {exc}\n"
            f"  Step {STEP} を実行し直してください。"
        ) from exc

    missing = [name for name, p in result.files.items() if not Path(p).exists()]
    if missing:
        logger.warning(
            "render.json に載っている出力が見つかりません: %s（Step %d を流し直すと作り直せます）",
            ", ".join(missing),
            STEP,
        )
    return result
