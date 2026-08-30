"""SPEC Step 1「音声抽出」。入力動画から 16kHz モノラル WAV を作り、素性を work/probe.json に残す。

方針:
- 音声トラックが無いファイルは、この先の全工程（文字起こし・無音検出）が成立しないのでここで止める。
- audio.wav が入力より新しければ作り直さない。60分の抽出を毎回やり直す理由がない。
"""

from __future__ import annotations

from pathlib import Path

from ..context import RunContext
from ..errors import FfmpegError, MissingArtifactError, RadioCutterError
from ..logging_util import fmt_duration, get_logger
from ..models import read_json, write_json
from ..util.ffmpeg import MediaInfo, extract_audio, probe_media, require_binaries

logger = get_logger(__name__)

STEP = 1
NAME = "音声抽出"

#: work/<episode_id>/ に作る中間ファイル名
AUDIO_FILENAME = "audio.wav"
PROBE_FILENAME = "probe.json"

OUTPUTS: tuple[str, ...] = (AUDIO_FILENAME, PROBE_FILENAME)


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def audio_path(ctx: RunContext) -> Path:
    """抽出した音声（Step 2 の文字起こしと Step 4 の無音検出が読む）。"""
    return ctx.work_path(AUDIO_FILENAME)


def probe_path(ctx: RunContext) -> Path:
    """ffprobe の結果を書く先。"""
    return ctx.work_path(PROBE_FILENAME)


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def _is_up_to_date(wav: Path, src: Path) -> bool:
    """既に抽出済みの音声がそのまま使えるか（入力より新しく、中身が空でないか）。"""
    try:
        if not wav.exists():
            return False
        wav_stat = wav.stat()
        if wav_stat.st_size <= 0:
            return False
        return wav_stat.st_mtime >= src.stat().st_mtime
    except OSError as exc:  # パーミッションなどで stat できないなら作り直したほうが安全
        logger.debug("音声の更新時刻を確認できませんでした（%s）。抽出し直します。", exc)
        return False


def _describe(media: MediaInfo) -> str:
    """ログ1行に収まる形で入力の素性をまとめる。"""
    parts = [f"尺 {fmt_duration(media.duration)}（{media.duration:.3f}秒）"]
    if media.width and media.height:
        parts.append(f"{media.width}x{media.height}")
    if media.fps:
        parts.append(f"{media.fps:.3f}fps")
    if media.video_codec:
        parts.append(f"映像 {media.video_codec}")
    parts.append(f"音声 {media.audio_codec}" if media.audio_codec else "音声なし")
    return " / ".join(parts)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run(ctx: RunContext) -> MediaInfo:
    """音声を抽出し、ffprobe で取った入力の素性を probe.json に保存する（SPEC Step 1）。"""
    ctx.ensure_dirs()

    src = ctx.input_path
    if not src.exists():
        raise RadioCutterError(f"入力ファイルが見つかりません: {src}")
    if src.is_dir():
        raise RadioCutterError(f"入力がディレクトリです（動画ファイルを渡してください）: {src}")
    require_binaries()

    media = probe_media(src)
    if not media.has_audio:
        raise FfmpegError(
            f"この動画に音声トラックがありません: {src}\n"
            "文字起こしも無音検出もできないため、ここで止めます。\n"
            "録音設定を確認し、音声入りのファイルを渡してください。"
        )
    save(ctx, media)
    logger.info("入力: %s", _describe(media))
    if not media.has_video:
        ctx.warn("入力に映像トラックがありません。Step 7 の書き出しが音声だけになります。")
        logger.warning("入力に映像トラックがありません: %s", src)

    wav = audio_path(ctx)
    if _is_up_to_date(wav, src):
        logger.info("既存の音声を使う（入力より新しいので抽出をスキップ）: %s", wav)
        return media

    logger.info("音声を抽出する: %s → %s（16kHz モノラル PCM）", src.name, wav)
    extract_audio(src, wav)
    try:
        size = wav.stat().st_size
    except OSError as exc:  # 直後に消えた・読めないなら抽出は失敗扱い
        raise FfmpegError(f"抽出した音声を確認できませんでした: {wav}\n{exc}") from exc
    if size <= 0:
        raise FfmpegError(f"抽出した音声が空です: {wav}\n入力の音声トラックを確認してください。")
    logger.info("音声を抽出した: %s（%.1f MB）", wav, size / (1024 * 1024))
    return media


def save(ctx: RunContext, result: MediaInfo) -> None:
    """probe.json を書く。run() の中から呼ぶ。"""
    path = probe_path(ctx)
    write_json(path, result.to_dict())
    logger.debug("入力の素性を保存した: %s", path)


def load(ctx: RunContext) -> MediaInfo:
    """probe.json から入力の素性を読む（--from-step での再開用）。"""
    path = probe_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"{PROBE_FILENAME} がありません: {path}\n"
            "Step 1（音声抽出）をまだ実行していません。"
            "`radio-cutter run <入力.mp4> --only-step 1` を先に流してください。"
        )
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        raise MissingArtifactError(
            f"{PROBE_FILENAME} を読めませんでした: {path}\n{exc}\n"
            "壊れている可能性があります。Step 1 を実行し直してください。"
        ) from exc
    if not isinstance(data, dict):
        raise MissingArtifactError(
            f"{PROBE_FILENAME} の中身が JSON オブジェクトではありません: {path}\nStep 1 を実行し直してください。"
        )
    try:
        media = MediaInfo.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingArtifactError(
            f"{PROBE_FILENAME} の内容が不正です: {path}\n{exc}\nStep 1 を実行し直してください。"
        ) from exc

    wav = audio_path(ctx)
    if not wav.exists():
        logger.warning(
            "%s がありません: %s（Step 2 の文字起こしと Step 4 の無音検出でつまずきます）", AUDIO_FILENAME, wav
        )
    return media
