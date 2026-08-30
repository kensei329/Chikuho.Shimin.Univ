"""ffmpeg / ffprobe の薄いラッパ（SPEC Step 1・Step 4・Step 7・Step 8 と doctor が使う）。

方針:
- ffmpeg が非ゼロ終了したら握りつぶさず FfmpegError に stderr を載せて投げる（SPEC 9章）。
- 秒数は全て float。フレーム番号には変換しない（SPEC 11章）。
- 外部依存を増やさない。標準ライブラリの subprocess だけで完結させる。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..config import RenderConfig
from ..errors import FfmpegError
from ..logging_util import get_logger

logger = get_logger(__name__)

#: FfmpegError に載せる stderr の最大文字数。ffmpeg の stderr は数MBになりうるため末尾だけ残す。
STDERR_TAIL_CHARS = 4000

#: 環境変数でバイナリのパスを差し替えられるようにしておく（Homebrew 版と静的ビルドの共存対策）。
FFMPEG_ENV = "RADIO_CUTTER_FFMPEG"
FFPROBE_ENV = "RADIO_CUTTER_FFPROBE"

#: `ffmpeg -encoders` の1行: " V....D libx264   libx264 H.264 ..." の 2 カラム目を拾う。
_ENCODER_LINE_RE = re.compile(r"^\s*[VAS][0-9A-Za-z.]{5}\s+(\S+)")

#: silencedetect のログ。1行に複数の情報が入る形（"silence_end: 3.5 | silence_duration: 0.4"）にも当たる。
_SILENCE_EVENT_RE = re.compile(
    r"silence_(?P<kind>start|end)\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

_VERSION_RE = re.compile(r"^ffmpeg version (\S+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# メディア情報
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    """ffprobe から取り出した入力メディアの素性。work/probe.json の中身になる。"""

    path: str
    duration: float
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_video: bool = False
    has_audio: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """work/probe.json に書く形。raw も残す（あとから何が起きたか追えるようにする）。"""
        return {
            "path": self.path,
            "duration": round(float(self.duration), 3),
            "fps": (round(float(self.fps), 6) if self.fps is not None else None),
            "width": self.width,
            "height": self.height,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MediaInfo":
        return cls(
            path=str(d.get("path", "")),
            duration=float(d.get("duration", 0.0)),
            fps=(float(d["fps"]) if d.get("fps") is not None else None),
            width=(int(d["width"]) if d.get("width") is not None else None),
            height=(int(d["height"]) if d.get("height") is not None else None),
            video_codec=(str(d["video_codec"]) if d.get("video_codec") is not None else None),
            audio_codec=(str(d["audio_codec"]) if d.get("audio_codec") is not None else None),
            has_video=bool(d.get("has_video", False)),
            has_audio=bool(d.get("has_audio", False)),
            raw=dict(d.get("raw", {}) or {}),
        )


# ---------------------------------------------------------------------------
# バイナリ
# ---------------------------------------------------------------------------


def ffmpeg_bin() -> str:
    """使う ffmpeg の実行ファイル名。環境変数で上書きできる。"""
    return os.environ.get(FFMPEG_ENV) or "ffmpeg"


def ffprobe_bin() -> str:
    """使う ffprobe の実行ファイル名。環境変数で上書きできる。"""
    return os.environ.get(FFPROBE_ENV) or "ffprobe"


def require_binaries() -> None:
    """ffmpeg / ffprobe が PATH にあるか確かめる。無ければ入れ方を添えて止める。"""
    missing: list[str] = []
    for name in (ffmpeg_bin(), ffprobe_bin()):
        if shutil.which(name) is None:
            missing.append(name)
    if missing:
        raise FfmpegError(
            "必要なコマンドが見つかりません: "
            + ", ".join(missing)
            + "\nmacOS なら `brew install ffmpeg` で入ります。"
            + f"\n別の場所にあるものを使う場合は環境変数 {FFMPEG_ENV} / {FFPROBE_ENV} で指定してください。"
        )


def ffmpeg_version() -> str | None:
    """ffmpeg のバージョン文字列。取得できなければ None（doctor が「無い」と表示するため）。"""
    try:
        proc = run_ffmpeg(["-version"], check=False)
    except FfmpegError:
        return None
    if proc.returncode != 0:
        return None
    m = _VERSION_RE.search(proc.stdout or "")
    if m:
        return m.group(1)
    first = (proc.stdout or "").strip().splitlines()
    return first[0] if first else None


# ---------------------------------------------------------------------------
# プロセス実行
# ---------------------------------------------------------------------------


def _tail(text: str, limit: int = STDERR_TAIL_CHARS) -> str:
    """例外メッセージ用に末尾だけ残す。ffmpeg の stderr は巨大になりうる。"""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return "…（先頭 %d 文字を省略）…\n" % (len(text) - limit) + text[-limit:]


def _run(cmd: list[str], *, check: bool, timeout: float | None) -> subprocess.CompletedProcess:
    """subprocess.run の共通処理。失敗は必ず FfmpegError に包む。"""
    logger.debug("実行: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise FfmpegError(
            f"コマンドが見つかりません: {cmd[0]}\nffmpeg / ffprobe を PATH に通してください。",
            cmd=cmd,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(
            f"コマンドがタイムアウトしました（{timeout}秒）: {cmd[0]}",
            cmd=cmd,
            stderr=_tail(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")),
        ) from exc

    if check and proc.returncode != 0:
        raise FfmpegError(
            f"{Path(cmd[0]).name} が異常終了しました。",
            cmd=cmd,
            stderr=_tail(proc.stderr or ""),
            returncode=proc.returncode,
        )
    return proc


def run_ffmpeg(
    args: Sequence[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """ffmpeg を実行する。`-hide_banner -nostdin` は常に付ける（対話待ちで固まらないように）。"""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", *[str(a) for a in args]]
    return _run(cmd, check=check, timeout=timeout)


def run_ffprobe(
    args: Sequence[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """ffprobe を実行する。ffprobe は `-nostdin` を解さないので付けない。"""
    cmd = [ffprobe_bin(), "-hide_banner", *[str(a) for a in args]]
    return _run(cmd, check=check, timeout=timeout)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    """ffprobe の -show_format -show_streams を JSON で取る。"""
    proc = run_ffprobe(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FfmpegError(
            f"ffprobe の出力が JSON として読めませんでした: {path}",
            stderr=_tail(proc.stdout or ""),
        ) from exc
    if not isinstance(data, dict):
        raise FfmpegError(f"ffprobe の出力が JSON オブジェクトではありません: {path}")
    return data


def _parse_fraction(value: Any) -> float | None:
    """"30000/1001" 形式（r_frame_rate）を float にする。0除算や "0/0" は None。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "/" in text:
            num_s, _, den_s = text.partition("/")
            num = float(num_s)
            den = float(den_s)
            if den == 0.0:
                return None
            return num / den
        return float(text)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def probe_media(path: str | Path) -> MediaInfo:
    """入力メディアの総尺・fps・解像度・コーデックを取る（SPEC Step 1）。"""
    p = Path(path)
    data = ffprobe_json(p)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video_stream: dict[str, Any] | None = None
    audio_stream: dict[str, Any] | None = None
    for st in streams:
        if not isinstance(st, dict):
            continue
        kind = st.get("codec_type")
        if kind == "video" and video_stream is None:
            # カバーアート（attached_pic）は映像トラックとして扱わない。
            disposition = st.get("disposition") or {}
            if int(disposition.get("attached_pic", 0) or 0) == 1:
                continue
            video_stream = st
        elif kind == "audio" and audio_stream is None:
            audio_stream = st

    # 尺は format.duration を優先。無ければ映像ストリームの duration（SPEC Step 1）。
    duration = _to_float(fmt.get("duration"))
    if duration is None and video_stream is not None:
        duration = _to_float(video_stream.get("duration"))
    if duration is None:
        raise FfmpegError(
            f"総尺を取得できませんでした: {p}\n"
            "format.duration も映像ストリームの duration も無いファイルです。"
            "破損しているか、ffprobe が対応していない形式の可能性があります。"
        )

    fps = None
    if video_stream is not None:
        fps = _parse_fraction(video_stream.get("r_frame_rate"))
        if fps is None:
            fps = _parse_fraction(video_stream.get("avg_frame_rate"))

    width = None
    height = None
    if video_stream is not None:
        try:
            width = int(video_stream["width"])
            height = int(video_stream["height"])
        except (KeyError, TypeError, ValueError):
            width = None
            height = None

    return MediaInfo(
        path=str(p),
        duration=float(duration),
        fps=fps,
        width=width,
        height=height,
        video_codec=(str(video_stream.get("codec_name")) if video_stream and video_stream.get("codec_name") else None),
        audio_codec=(str(audio_stream.get("codec_name")) if audio_stream and audio_stream.get("codec_name") else None),
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        raw=data,
    )


def media_duration(path: str | Path) -> float:
    """尺だけ欲しいとき用。final.mp4 の検算（SPEC Step 7）で使う。"""
    return probe_media(path).duration


# ---------------------------------------------------------------------------
# エンコーダ
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _encoder_names() -> frozenset[str]:
    """`ffmpeg -encoders` の実行結果。プロセス起動が重いのでモジュールレベルでキャッシュする。"""
    proc = run_ffmpeg(["-encoders"], check=False)
    if proc.returncode != 0:
        raise FfmpegError(
            "ffmpeg -encoders が失敗しました。ffmpeg のインストールを確認してください。",
            cmd=[ffmpeg_bin(), "-encoders"],
            stderr=_tail(proc.stderr or ""),
            returncode=proc.returncode,
        )
    names: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        m = _ENCODER_LINE_RE.match(line)
        if m:
            names.add(m.group(1))
    return frozenset(names)


def list_encoders() -> set[str]:
    """使えるエンコーダ名の集合。中身はキャッシュ済みで、毎回コピーを返す（呼び先の書き換えでキャッシュを壊さないため）。"""
    return set(_encoder_names())


#: テストや doctor が ffmpeg を差し替えたときにキャッシュを捨てられるようにしておく。
list_encoders.cache_clear = _encoder_names.cache_clear  # type: ignore[attr-defined]


def has_encoder(name: str) -> bool:
    """指定のエンコーダがこのビルドで使えるか（SPEC 2章の VideoToolbox 判定）。"""
    if not name:
        return False
    return name in _encoder_names()


def choose_video_codec(render: RenderConfig) -> tuple[str, tuple[str, ...], bool]:
    """使う映像コーデックと追加引数を決める。使えなければ CPU エンコードに落とす（SPEC Step 7）。

    返り値は (コーデック名, 追加引数, フォールバックしたか)。
    """
    if has_encoder(render.video_codec):
        return render.video_codec, ("-b:v", render.video_bitrate), False

    logger.warning(
        "映像コーデック %s がこの ffmpeg で使えません。%s にフォールバックします（CPU エンコードになります）。",
        render.video_codec,
        render.fallback_video_codec,
    )
    if not has_encoder(render.fallback_video_codec):
        logger.warning(
            "フォールバック先の %s も見つかりません。ffmpeg のビルドを確認してください。",
            render.fallback_video_codec,
        )
    return render.fallback_video_codec, tuple(render.fallback_extra_args), True


# ---------------------------------------------------------------------------
# Step 1: 音声抽出
# ---------------------------------------------------------------------------


def extract_audio(input_path: str | Path, out_wav: str | Path) -> Path:
    """SPEC Step 1 の音声抽出。16kHz モノラル PCM（ASR と silencedetect が使う）。"""
    src = Path(input_path)
    dst = Path(out_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i", str(src),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(dst),
        ]
    )
    if not dst.exists():
        raise FfmpegError(f"音声を抽出できませんでした（出力ファイルがありません）: {dst}")
    return dst


# ---------------------------------------------------------------------------
# Step 4: 無音検出
# ---------------------------------------------------------------------------


def parse_silence_log(
    stderr: str, *, offset: float = 0.0, window_end: float | None = None
) -> list[tuple[float, float]]:
    """silencedetect のログから無音区間を取り出し、絶対時刻の (開始, 終了) にして返す。

    atrim で切り出した窓の中の相対時刻がログに出るので、`offset`（＝窓の開始絶対時刻）を足す。
    1行に複数の情報が入る形にも、silence_end が先に来る崩れたログにも耐えるよう
    正規表現でイベント列として舐める。`window_end` は絶対時刻で、閉じられていない区間を閉じるのに使う。
    """
    spans: list[tuple[float, float]] = []
    pending_start: float | None = None

    for m in _SILENCE_EVENT_RE.finditer(stderr or ""):
        try:
            value = float(m.group("value"))
        except ValueError:  # pragma: no cover - 正規表現が通れば起きない
            continue
        abs_time = value + offset

        if m.group("kind") == "start":
            if pending_start is not None:
                # 閉じられないまま次の開始が来た。先に来た開始を活かす（そちらが実際の谷の入口）。
                logger.debug("silence_start が連続しました（%.3f を無視）。", abs_time)
                continue
            pending_start = abs_time
        else:
            if pending_start is None:
                # 対応する silence_start が無い（窓の頭が既に無音、またはログの順序が崩れている）。
                # 窓の先頭から始まっていたものとして扱い、捨てずに残す。
                start_abs = offset
            else:
                start_abs = pending_start
                pending_start = None
            if abs_time >= start_abs:
                spans.append((start_abs, abs_time))
            else:
                logger.debug("終了が開始より前の無音区間を捨てました（%.3f < %.3f）。", abs_time, start_abs)

    if pending_start is not None:
        if window_end is not None and window_end >= pending_start:
            spans.append((pending_start, float(window_end)))
        else:
            logger.debug("閉じていない無音区間を捨てました（開始 %.3f）。", pending_start)

    spans.sort(key=lambda s: (s[0], s[1]))
    return spans


def detect_silences(
    wav_path: str | Path,
    *,
    start: float,
    end: float,
    noise_db: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """SPEC Step 4 の silencedetect。指定区間だけを atrim で切り出して調べる。

    返り値は元音声の絶対時刻。全尺に silencedetect をかけると60分ぶん走査することになるため、
    カット点の前後だけを窓にして呼ぶ。
    """
    win_start = max(0.0, float(start))
    win_end = float(end)
    if win_end <= win_start:
        logger.debug("無音検出の窓が空です（start=%.3f, end=%.3f）。", win_start, win_end)
        return []

    af = (
        f"atrim=start={win_start:.3f}:end={win_end:.3f},"
        "asetpts=PTS-STARTPTS,"
        f"silencedetect=n={noise_db:g}dB:d={min_duration:g}"
    )
    proc = run_ffmpeg(["-i", str(wav_path), "-vn", "-af", af, "-f", "null", "-"])
    spans = parse_silence_log(proc.stderr or "", offset=win_start, window_end=win_end)
    logger.debug(
        "無音検出 [%.3f, %.3f] n=%gdB d=%g → %d 区間", win_start, win_end, noise_db, min_duration, len(spans)
    )
    return spans


# ---------------------------------------------------------------------------
# Step 7: 書き出し
# ---------------------------------------------------------------------------


def encode_segment(
    input_path: str | Path,
    start: float,
    end: float,
    out_path: str | Path,
    render: RenderConfig,
    *,
    use_fallback: bool | None = None,
) -> Path:
    """区間 [start, end) を再エンコードして書き出す（SPEC Step 7）。

    `-c copy` は最寄りのキーフレームまでカット位置がずれるので使わない。必ず再エンコードする。
    """
    src = Path(input_path)
    dst = Path(out_path)
    duration = float(end) - float(start)
    if duration <= 0:
        raise FfmpegError(
            f"書き出し区間の長さが0以下です: start={start:.3f}, end={end:.3f} → {dst.name}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)

    if use_fallback is None:
        codec, extra, _fell_back = choose_video_codec(render)
    elif use_fallback:
        codec, extra = render.fallback_video_codec, tuple(render.fallback_extra_args)
    else:
        codec, extra = render.video_codec, ("-b:v", render.video_bitrate)

    # -ss を -i の前に置くと高速シークになるが、出力タイムスタンプが 0 起点に振り直される。
    # そのため -to は「入力の絶対時刻」ではなく区間長として解釈され、意図とずれる。
    # 混乱を避けるため常に -t (end - start) を使う。
    args = [
        "-y",
        "-ss", f"{float(start):.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", codec,
        *extra,
        "-c:a", render.audio_codec,
        "-b:a", render.audio_bitrate,
        "-movflags", "+faststart",
        str(dst),
    ]
    run_ffmpeg(args)
    if not dst.exists():
        raise FfmpegError(f"書き出しに失敗しました（出力ファイルがありません）: {dst}")
    logger.debug("書き出し %s [%.3f, %.3f] codec=%s", dst.name, start, end, codec)
    return dst


def _concat_escape(path: Path) -> str:
    """concat demuxer のリスト用にパスをクォートする。

    `file '...'` 形式なので、パス中のシングルクォートは一度閉じて '\\'' で埋める必要がある。
    """
    return str(path.resolve()).replace("'", "'\\''")


def concat_files(
    files: Sequence[Path | str],
    out_path: str | Path,
    work_dir: str | Path,
    *,
    list_name: str = "concat.txt",
) -> Path:
    """同一パラメータでエンコード済みの動画を concat demuxer で連結する（SPEC Step 7）。

    ここだけは `-c copy` でよい。3本とも同じコーデック・同じ設定で書き出した直後だから。
    """
    paths = [Path(f) for f in files]
    if not paths:
        raise FfmpegError("連結する動画が1本もありません。")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FfmpegError("連結する動画が見つかりません: " + ", ".join(missing))

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    list_path = work / list_name
    body = "".join(f"file '{_concat_escape(p)}'\n" for p in paths)
    list_path.write_text(body, encoding="utf-8")

    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
    )
    if not dst.exists():
        raise FfmpegError(f"連結に失敗しました（出力ファイルがありません）: {dst}")
    logger.debug("連結 %d 本 → %s", len(paths), dst.name)
    return dst
