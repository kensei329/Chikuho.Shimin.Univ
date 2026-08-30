"""SPEC Step 4「カット点の精密化」。raw_cut_time を直前の無音の谷に寄せる。

方針:
- raw_cut_time のまま切ると語の立ち上がりが削れるので、カット点の前後だけに
  silencedetect をかけ、**raw より前にある最後の無音区間の終わり - 50ms** を採用する。
- 無音が見つからなければ勝手に遠くの谷を探しに行かず、raw - 80ms で代用し、
  そのことを CutPoint と warnings の両方に残す（あとから何が起きたか追えるようにする）。
- 選択ロジックは ffmpeg を触らない純関数 `pick_cut_time()` に分け、テストが直接叩けるようにする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..config import (
    NO_SILENCE_BACKOFF_SEC,
    SILENCE_BACKOFF_SEC,
    SILENCE_LOOKAHEAD_SEC,
    SILENCE_LOOKBACK_SEC,
)
from ..context import RunContext
from ..errors import MissingArtifactError
from ..logging_util import get_logger
from ..models import AnchorResult, CutPoint, r3, read_json, write_json
from ..util.ffmpeg import detect_silences
from ..util.timeline import fmt_timestamp
from . import s1_extract_audio

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 4
NAME: str = "カット点の精密化"

#: work/<episode_id>/ に書く中間ファイル
CUTS_FILE = "cuts.json"
OUTPUTS: tuple[str, ...] = (CUTS_FILE,)


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def cuts_path(ctx: RunContext) -> Path:
    """カット点の中間ファイル（Step 5・7・8 と decisions.json が読む）。"""
    return ctx.work_path(CUTS_FILE)


# ---------------------------------------------------------------------------
# 純関数（テストが直接叩く）
# ---------------------------------------------------------------------------


def search_window(raw: float) -> tuple[float, float]:
    """raw_cut_time に対する無音の探索窓 [raw - 1.5秒, raw + 0.5秒]。下限は 0 でクランプ。

    全尺に silencedetect をかけると60分ぶん走査することになるため、カット点の周りだけを見る。
    """
    return (max(0.0, float(raw) - SILENCE_LOOKBACK_SEC), float(raw) + SILENCE_LOOKAHEAD_SEC)


def pick_cut_time(
    raw: float, silences: Sequence[tuple[float, float]]
) -> tuple[float, bool, tuple[float, float] | None]:
    """raw より前にある最後の無音区間を選び、その終了時刻 - 50ms をカット点にする（SPEC Step 4）。

    「raw より前」は `silence_start <= raw` で判定する。ふつうは無音が閉じたところで
    語が立ち上がるので `silence_end <= raw` になるが、ASR の時刻は数十ミリ秒ぶれるので、
    語頭が無音区間の内側に落ちることがある。そこで無音を捨てて 80ms のフォールバックに
    倒れると、無音を見つけているのに `silence_found: false` と記録され、
    しかも谷でないところで切ることになる。

    カット点は「無音の終わり」と raw の早いほうから 50ms 手前。
    無音が raw より前で閉じている通常の場合は SPEC どおり `silence_end - 50ms` そのままで、
    無音の内側に raw が落ちた場合だけ raw の 50ms 手前になる。どちらでも cut < raw は保たれる。
    0 未満にはせず、ミリ秒に丸めて返す
    （SPEC 11章。0.05 の減算で出る二進小数の端数をそのまま持ち回らない）。
    """
    raw_time = float(raw)
    chosen: tuple[float, float] | None = None
    for span in silences:
        start, end = float(span[0]), float(span[1])
        if end < start:
            # 壊れたログ由来の区間。採用すると尺が逆転するので捨てる。
            logger.debug("終了が開始より前の無音区間を無視しました（%.3f < %.3f）。", end, start)
            continue
        if start > raw_time:
            continue
        if chosen is None or (end, start) > (chosen[1], chosen[0]):
            chosen = (start, end)

    if chosen is None:
        return (r3(max(0.0, raw_time - NO_SILENCE_BACKOFF_SEC)), False, None)
    cut = min(chosen[1], raw_time) - SILENCE_BACKOFF_SEC
    return (r3(max(0.0, cut)), True, chosen)


def no_silence_warning(anchor_id: str) -> str:
    """無音が見つからなかったときの警告文（decisions.json の warnings に載る）。"""
    return (
        f"アンカー {anchor_id}: 無音が見つからず raw_cut_time - {NO_SILENCE_BACKOFF_SEC:g} 秒で代用しました"
    )


def refine_anchor(
    ctx: RunContext,
    anchor: AnchorResult,
    audio_path: Path,
) -> CutPoint:
    """アンカー1つぶんの無音検出とカット点の決定。無音が無ければ warnings にも残す。"""
    raw = float(anchor.raw_cut_time)
    win_start, win_end = search_window(raw)
    detected = detect_silences(
        audio_path,
        start=win_start,
        end=win_end,
        noise_db=ctx.silence.noise_db,
        min_duration=ctx.silence.min_duration_sec,
    )
    # ミリ秒に丸めてから判定する（SPEC 11章「秒数は小数点以下3桁」）。
    # silencedetect は無音の終わりを1サンプル（16kHz なら 62.5μs）ぶん行き過ぎて報告することがあり、
    # 生の値のまま `silence_end <= raw` を見ると、発話の直前でぴったり閉じた谷を
    # 「raw より後」と誤判定してフォールバックしてしまう。
    silences = [(r3(start), r3(end)) for start, end in detected]
    cut_time, silence_found, span = pick_cut_time(raw, silences)

    if silence_found and span is not None:
        logger.info(
            "アンカー %s: raw %.3f秒（%s）→ カット %.3f秒（%s）"
            "／無音 [%.3f, %.3f]（%d 区間中から採用、%.3f秒手前に寄せた）",
            anchor.id,
            raw,
            fmt_timestamp(raw),
            cut_time,
            fmt_timestamp(cut_time),
            span[0],
            span[1],
            len(silences),
            raw - cut_time,
        )
    else:
        message = no_silence_warning(anchor.id)
        ctx.warn(message)
        logger.warning(
            "%s（探索窓 [%.3f, %.3f] / n=%gdB / d=%g）。"
            "語頭が欠けるようなら --silence-db を上げるか --silence-dur を短くしてください。",
            message,
            win_start,
            win_end,
            ctx.silence.noise_db,
            ctx.silence.min_duration_sec,
        )
        logger.info(
            "アンカー %s: raw %.3f秒（%s）→ カット %.3f秒（%s）／無音なし",
            anchor.id,
            raw,
            fmt_timestamp(raw),
            cut_time,
            fmt_timestamp(cut_time),
        )

    return CutPoint(
        anchor_id=anchor.id,
        raw_cut_time=raw,
        cut_time=cut_time,
        silence_found=silence_found,
        score=float(anchor.score),
        silence_start=(span[0] if span is not None else None),
        silence_end=(span[1] if span is not None else None),
    )


# ---------------------------------------------------------------------------
# 順序（アンカーを config の並びに揃える）
# ---------------------------------------------------------------------------


def _ordered_ids(ctx: RunContext, anchors: Mapping[str, AnchorResult]) -> list[str]:
    """config のアンカー順で並べる。config に無いIDは後ろに回す（取りこぼさない）。"""
    ids = [a.id for a in ctx.config.anchors if a.id in anchors]
    ids.extend(anchor_id for anchor_id in anchors if anchor_id not in ids)
    return ids


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def run(ctx: RunContext, anchors: Mapping[str, AnchorResult]) -> dict[str, CutPoint]:
    """各アンカーの raw_cut_time を無音の谷に寄せ、work/cuts.json を書いて返す（SPEC Step 4）。"""
    ctx.ensure_dirs()

    if not anchors:
        raise MissingArtifactError(
            "アンカーが1つもありません。\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step 3` で"
            "Step 3（アンカー検出）を実行してください。"
        )

    audio = ctx.work_path(s1_extract_audio.AUDIO_FILENAME)
    if not audio.exists():
        raise MissingArtifactError(
            f"{s1_extract_audio.AUDIO_FILENAME} がありません: {audio}\n"
            "Step 4 の無音検出は Step 1 が作る音声を使います。\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step {s1_extract_audio.STEP}` を実行してください。"
        )

    logger.info(
        "無音検出の設定: n=%gdB / d=%g秒 / 探索窓 raw-%g秒 〜 raw+%g秒",
        ctx.silence.noise_db,
        ctx.silence.min_duration_sec,
        SILENCE_LOOKBACK_SEC,
        SILENCE_LOOKAHEAD_SEC,
    )

    cuts: dict[str, CutPoint] = {}
    for anchor_id in _ordered_ids(ctx, anchors):
        cuts[anchor_id] = refine_anchor(ctx, anchors[anchor_id], audio)

    save(ctx, cuts)
    return cuts


def save(ctx: RunContext, result: Mapping[str, CutPoint]) -> None:
    """work/<episode_id>/cuts.json に書く（形は {"A": {...}, "B": {...}}）。run() の中から呼ぶ。"""
    payload = {anchor_id: cut.to_dict() for anchor_id, cut in result.items()}
    path = cuts_path(ctx)
    write_json(path, payload)
    logger.debug("カット点を保存しました: %s（%d 件）", path, len(payload))


def load(ctx: RunContext) -> dict[str, CutPoint]:
    """work/<episode_id>/cuts.json を読む（--from-step での再開用）。"""
    path = cuts_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"カット点の中間ファイルがありません: {path}\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step {STEP}` を実行してください"
            "（Step 4 は Step 1 の audio.wav と Step 3 の anchors.json を使います）。"
        )
    try:
        data = read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissingArtifactError(
            f"カット点の中間ファイルの JSON が壊れています: {path}\n  {exc}\n"
            f"  このファイルを消して `radio-cutter run {ctx.input_path} --from-step {STEP}` で作り直してください。"
        ) from exc
    except OSError as exc:
        raise MissingArtifactError(f"カット点の中間ファイルを読めませんでした: {path}\n  {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise MissingArtifactError(
            f"カット点の中間ファイルの中身が空か、想定した形（アンカーIDをキーにしたオブジェクト）ではありません: {path}"
        )

    cuts: dict[str, CutPoint] = {}
    for anchor_id, payload in data.items():
        if not isinstance(payload, dict):
            raise MissingArtifactError(
                f"カット点 '{anchor_id}' の内容がオブジェクトではありません: {path}"
            )
        try:
            cuts[str(anchor_id)] = CutPoint.from_dict(str(anchor_id), payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise MissingArtifactError(
                f"カット点 '{anchor_id}' の内容が不正です: {path}\n  {exc}"
            ) from exc
    return cuts
