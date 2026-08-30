"""SPEC Step 8「確認用プレビュー」。カット点とハイライトの前後2秒だけを切り出す。

方針（SPEC Step 8）：
- 60分をレンダリングし直す前に、カットの当たりをここで確認できるようにするのが目的。
- 中間ファイルは作らない（`OUTPUTS = ()`）。成果物は out/<episode_id>/preview/ に直接置く。
- 動画の端をはみ出す窓はクランプする。クランプしても長さが残らない場合は
  例外にせずスキップして警告に残す（プレビューが1本欠けても本編の書き出しは止めない）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..context import RunContext
from ..logging_util import get_logger
from ..models import CutPoint, HighlightResult, r3
from ..util.ffmpeg import (
    MediaInfo,
    choose_video_codec,
    encode_segment,
    probe_media,
    require_binaries,
)
from ..util.timeline import fmt_timestamp

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 8
NAME: str = "確認用プレビュー"

#: work/ には何も書かない（成果物は out/<episode_id>/preview/ だけ）
OUTPUTS: tuple[str, ...] = ()

#: 出力先ディレクトリ名と、カット点の前後に取る幅（SPEC Step 8「前後2秒（計4秒）」）
PREVIEW_DIRNAME = "preview"
PREVIEW_MARGIN_SEC = 2.0

#: ハイライトの始点・終点のファイル名
HIGHLIGHT_IN_FILE = "highlight_in.mp4"
HIGHLIGHT_OUT_FILE = "highlight_out.mp4"


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def preview_dir(ctx: RunContext) -> Path:
    """プレビューの置き場（out/<episode_id>/preview/）。"""
    return ctx.out_path(PREVIEW_DIRNAME)


def _safe_name(anchor_id: str) -> str:
    """アンカーIDをファイル名に使える形にする。設定次第で記号が入りうるため。"""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(anchor_id))
    return cleaned or "unknown"


def cut_preview_name(anchor_id: str) -> str:
    """カット点1つぶんのプレビューのファイル名（SPEC Step 8 の cut_A.mp4 / cut_B.mp4）。"""
    return f"cut_{_safe_name(anchor_id)}.mp4"


# ---------------------------------------------------------------------------
# 純関数（テストが直接叩く）
# ---------------------------------------------------------------------------


def preview_window(
    center: float, total_duration: float, *, margin: float = PREVIEW_MARGIN_SEC
) -> tuple[float, float] | None:
    """[center - margin, center + margin] を動画の中に収めて返す。空になるなら None。

    カット点が動画の先頭や末尾に寄っていると窓がはみ出すので、必ずクランプする。
    """
    total = float(total_duration)
    start = max(0.0, float(center) - float(margin))
    end = float(center) + float(margin)
    if total > 0:
        end = min(end, total)
        start = min(start, total)
    if end - start <= 0:
        return None
    return (start, end)


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def _resolve_total_duration(ctx: RunContext, media: MediaInfo | None) -> float:
    """入力の総尺。probe.json 由来の MediaInfo が無ければ ffprobe で取り直す。"""
    if media is not None and float(media.duration) > 0:
        return float(media.duration)
    logger.debug("総尺が渡されなかったので ffprobe で取り直します: %s", ctx.input_path)
    return float(probe_media(ctx.input_path).duration)


def _render_preview(
    ctx: RunContext,
    label: str,
    filename: str,
    center: float,
    total_duration: float,
    *,
    use_fallback: bool,
) -> Path | None:
    """1本ぶんのプレビューを書き出す。窓が空ならスキップして警告に残す。"""
    window = preview_window(center, total_duration)
    if window is None:
        message = (
            f"{label}（{r3(center)}秒）のプレビューは、動画の範囲に収まる区間が無いため作りませんでした"
            f"（総尺 {r3(total_duration)}秒）。"
        )
        ctx.warn(message)
        logger.warning("%s", message)
        return None

    start, end = window
    dst = preview_dir(ctx) / filename
    logger.info(
        "%sのプレビュー: %s〜%s（%.3f〜%.3f秒 / %.1f秒）→ %s",
        label,
        fmt_timestamp(start),
        fmt_timestamp(end),
        start,
        end,
        end - start,
        dst.name,
    )
    encode_segment(ctx.input_path, start, end, dst, ctx.config.render, use_fallback=use_fallback)
    return dst


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def run(
    ctx: RunContext,
    cuts: Mapping[str, CutPoint],
    highlight: HighlightResult | None,
    media: MediaInfo | None = None,
) -> list[Path]:
    """カット点とハイライトの前後2秒を切り出す（SPEC Step 8）。書き出したファイルを返す。"""
    ctx.ensure_dirs()
    require_binaries()

    out_dir = preview_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_duration = _resolve_total_duration(ctx, media)

    # コーデックは Step 7 と同じ決め方で1回だけ決める（プレビューだけ別設定にしない）。
    codec, _extra_args, used_fallback = choose_video_codec(ctx.config.render)
    logger.debug("プレビューの映像コーデック: %s（フォールバック: %s）", codec, used_fallback)

    if not cuts:
        message = "カット点が無いため、カット点のプレビューは作りませんでした。"
        ctx.warn(message)
        logger.warning("%s", message)

    outputs: list[Path] = []

    # 1) 各カット点（config のアンカー順。config に無いIDも取りこぼさず後ろに回す）
    for anchor_id in _ordered_ids(ctx, cuts):
        cut = cuts[anchor_id]
        made = _render_preview(
            ctx,
            f"カット点 {anchor_id}",
            cut_preview_name(anchor_id),
            float(cut.cut_time),
            total_duration,
            use_fallback=used_fallback,
        )
        if made is not None:
            outputs.append(made)

    # 2) ハイライトの始点・終点
    if highlight is None:
        message = (
            "ハイライトが選ばれていないため、highlight_in.mp4 / highlight_out.mp4 は作りませんでした"
            "（Step 5 を実行すると作られます）。"
        )
        ctx.warn(message)
        logger.warning("%s", message)
    else:
        for label, filename, center in (
            ("ハイライト始点", HIGHLIGHT_IN_FILE, float(highlight.selected.start)),
            ("ハイライト終点", HIGHLIGHT_OUT_FILE, float(highlight.selected.end)),
        ):
            made = _render_preview(
                ctx, label, filename, center, total_duration, use_fallback=used_fallback
            )
            if made is not None:
                outputs.append(made)

    logger.info("プレビューを %d 本書き出しました: %s", len(outputs), out_dir)
    save(ctx, outputs)
    return outputs


def _ordered_ids(ctx: RunContext, cuts: Mapping[str, CutPoint]) -> list[str]:
    """config のアンカー順で並べる。config に無いIDは後ろに回す（取りこぼさない）。"""
    ids = [a.id for a in ctx.config.anchors if a.id in cuts]
    ids.extend(anchor_id for anchor_id in cuts if anchor_id not in ids)
    return ids


def save(ctx: RunContext, result: list[Path]) -> None:
    """このステップは work/ に中間ファイルを持たない（成果物が out/ に直接出る）。

    共通プロトコルに合わせて用意してあるだけなので、ここでは書き出した本数をログに残す。
    """
    logger.debug("プレビュー %d 本（中間ファイルなし）: %s", len(result), preview_dir(ctx))


def load(ctx: RunContext) -> list[Path]:
    """out/<episode_id>/preview/ にある mp4 を一覧で返す。

    中間ファイルが無いステップなので、無くても MissingArtifactError にはしない
    （プレビューはあくまで確認用で、後段のステップがこれに依存しないため）。
    """
    directory = preview_dir(ctx)
    if not directory.is_dir():
        logger.debug("プレビューのディレクトリがありません: %s", directory)
        return []
    files = sorted(p for p in directory.glob("*.mp4") if p.is_file())
    logger.debug("既存のプレビュー %d 本を読み込みました: %s", len(files), directory)
    return files
