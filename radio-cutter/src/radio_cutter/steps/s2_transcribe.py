"""SPEC Step 2「文字起こし」。単語タイムスタンプ付きの work/transcript.json を作る。

方針（SPEC 6章 Step2）:
- 全工程の8割の時間を占めるので、入力とASR設定が同じならキャッシュで丸ごと飛ばす。
- 単語タイムスタンプが取れないバックエンドは使わない（Step 3 が単語 index から時刻を引くため）。
- バックエンドの生 dict → Transcript の変換は純関数 `normalize_asr_result()` に閉じ込める。
  whisperx と mlx-whisper で words の形が微妙に違うので、揺れの吸収を1箇所にまとめる。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..config import AsrConfig
from ..context import RunContext
from ..errors import MissingArtifactError, RadioCutterError, TranscriptionError
from ..logging_util import fmt_duration, get_logger
from ..models import Segment, Transcript, Word
from ..util.cache import (
    TranscriptCacheEntry,
    load_cache_entry,
    save_cache_entry,
    sha256_file,
    stable_hash,
    transcript_cache_key,
)
from ..util.ffmpeg import MediaInfo, media_duration
from . import s1_extract_audio

logger = get_logger(__name__)

STEP = 2
NAME = "文字起こし"

TRANSCRIPT_FILENAME = "transcript.json"
CACHE_FILENAME = "transcript_cache.json"

OUTPUTS: tuple[str, ...] = (TRANSCRIPT_FILENAME, CACHE_FILENAME)

#: 単語テキストが入っているキー。whisperx は "word"、mlx-whisper は "word" か "text"。
_WORD_TEXT_KEYS: tuple[str, ...] = ("word", "text")

#: config の asr.backend に書ける名前 → 実装名
BACKEND_ALIASES: dict[str, str] = {
    "whisperx": "whisperx",
    "whisper-x": "whisperx",
    "whisper_x": "whisperx",
    "mlx_whisper": "mlx_whisper",
    "mlx-whisper": "mlx_whisper",
    "mlx": "mlx_whisper",
    # SPEC が挙げる whispermlx（WhisperX の MLX バックエンド版）は API が mlx-whisper 互換なのでこちらへ寄せる。
    "whispermlx": "mlx_whisper",
    "whisper-mlx": "mlx_whisper",
}

#: backend="auto" のときに試す順（SPEC 6章 Step2 のフォールバック）
#
# mlx 系が主で、whisperx はフォールバック。設定の既定モデルが
# "mlx-community/whisper-large-v3-mlx" という MLX 用のIDなので、
# 順番を逆にすると、そのIDをそのまま whisperx に渡して落ちる。
AUTO_BACKEND_ORDER: tuple[str, ...] = ("mlx_whisper", "whisperx")


class _BackendUnavailable(TranscriptionError):
    """そのバックエンドがこの環境に入っていないだけ。次の候補を試してよい。"""


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def transcript_path(ctx: RunContext) -> Path:
    """work/<episode_id>/transcript.json。"""
    return ctx.work_path(TRANSCRIPT_FILENAME)


def cache_path(ctx: RunContext) -> Path:
    """work/<episode_id>/transcript_cache.json。"""
    return ctx.work_path(CACHE_FILENAME)


# ---------------------------------------------------------------------------
# ASR 結果の正規化（純関数・テストが直接叩く）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawWord:
    """バックエンドが返した単語1つ（時刻は欠けていることがある）。"""

    text: str
    start: float | None
    end: float | None


def _as_time(value: Any) -> float | None:
    """秒として読める値なら float に、読めなければ None。負値は 0 に寄せる。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f if f > 0.0 else 0.0


def _word_entry(raw: Any) -> _RawWord | None:
    """単語1件を _RawWord にする。テキストが取れなければ None（捨てる）。

    キーは "word" でも "text" でもよい。素の文字列だけが並ぶ形にも耐える。
    """
    if isinstance(raw, str):
        text: str | None = raw
        start: float | None = None
        end: float | None = None
    elif isinstance(raw, dict):
        text = None
        for key in _WORD_TEXT_KEYS:
            value = raw.get(key)
            if isinstance(value, str):
                text = value
                break
        if text is None:
            return None
        start = _as_time(raw.get("start"))
        end = _as_time(raw.get("end"))
    else:
        return None

    if not text.strip():
        return None
    return _RawWord(text=text, start=start, end=end)


def _known_before(
    starts: list[float | None], ends: list[float | None], index: int, fallback: float
) -> float:
    """index より前で分かっている最後の時刻。無ければ fallback（セグメント開始）。"""
    for k in range(index - 1, -1, -1):
        if ends[k] is not None:
            return float(ends[k])  # type: ignore[arg-type]
        if starts[k] is not None:
            return float(starts[k])  # type: ignore[arg-type]
    return float(fallback)


def _known_after(
    starts: list[float | None], ends: list[float | None], index: int, fallback: float
) -> float:
    """index 以降で最初に分かっている時刻。無ければ fallback（セグメント終了）。"""
    for k in range(index, len(starts)):
        if starts[k] is not None:
            return float(starts[k])  # type: ignore[arg-type]
        if ends[k] is not None:
            return float(ends[k])  # type: ignore[arg-type]
    return float(fallback)


def _fill_word_times(entries: list[_RawWord], seg_start: float, seg_end: float) -> list[Word]:
    """時刻が欠けている単語を前後から線形補間して Word に変換する。

    アライメントが外れた単語（whisperx は数字や記号でよく起きる）は start/end が欠けるが、
    欠けたままだと Step 3 が一致位置から時刻を引けないので、必ず埋める。
    前後に手がかりが無ければセグメントの start/end で挟む。
    """
    n = len(entries)
    starts: list[float | None] = [e.start for e in entries]
    ends: list[float | None] = [e.end for e in entries]

    i = 0
    while i < n:
        if starts[i] is not None and ends[i] is not None:
            i += 1
            continue
        # [i, j) が「どちらかの時刻が欠けている」連続区間
        j = i
        while j < n and (starts[j] is None or ends[j] is None):
            j += 1
        left = _known_before(starts, ends, i, seg_start)
        right = _known_after(starts, ends, j, seg_end)
        if right < left:
            right = left
        span = (right - left) / float(j - i)
        for k in range(i, j):
            if starts[k] is None:
                starts[k] = left + span * (k - i)
            if ends[k] is None:
                ends[k] = left + span * (k - i + 1)
        i = j

    words: list[Word] = []
    prev_start = 0.0
    for k, entry in enumerate(entries):
        start = max(0.0, float(starts[k]))  # type: ignore[arg-type]
        end = max(0.0, float(ends[k]))  # type: ignore[arg-type]
        # 単語列は開始時刻の昇順であることを前提に二分探索される（util/timeline.py）。
        if start < prev_start:
            start = prev_start
        if end < start:
            end = start
        words.append(Word(word=entry.text, start=start, end=end))
        prev_start = start
    return words


def _normalize_segment(raw: Any, index: int) -> Segment | None:
    """セグメント1件を Segment にする。中身が空なら None（捨てる）。"""
    if not isinstance(raw, dict):
        logger.debug("segments[%d] が dict ではないので捨てました（%s）。", index, type(raw).__name__)
        return None

    raw_words = raw.get("words")
    if not isinstance(raw_words, (list, tuple)):
        raw_words = ()
    entries: list[_RawWord] = []
    for w in raw_words:
        entry = _word_entry(w)
        if entry is not None:
            entries.append(entry)

    text = str(raw.get("text") or "").strip()

    seg_start = _as_time(raw.get("start"))
    if seg_start is None:
        known_starts = [e.start for e in entries if e.start is not None]
        seg_start = min(known_starts) if known_starts else None
    seg_end = _as_time(raw.get("end"))
    if seg_end is None:
        known_ends = [e.end for e in entries if e.end is not None]
        if not known_ends:
            known_ends = [e.start for e in entries if e.start is not None]
        seg_end = max(known_ends) if known_ends else None

    if seg_start is None:
        seg_start = 0.0
    if seg_end is None or seg_end < seg_start:
        seg_end = seg_start

    if entries:
        words = _fill_word_times(entries, seg_start, seg_end)
    elif text:
        # words が空のセグメントは、セグメント全体を1単語として補完する（SPEC Step 2 / 契約）。
        words = [Word(word=text, start=seg_start, end=seg_end)]
    else:
        logger.debug("segments[%d] に単語もテキストも無いので捨てました。", index)
        return None

    if words:
        seg_start = min(seg_start, words[0].start)
        seg_end = max(seg_end, words[-1].end)
    if not text:
        text = "".join(w.word for w in words).strip()
    return Segment(start=float(seg_start), end=float(seg_end), text=text, words=words)


def _merge_overlapping(segments: list[Segment]) -> list[Segment]:
    """時刻の重なるセグメントを1つにまとめ、単語列が必ず時刻順になるようにする。

    ASR はまれに重なったセグメントを返す。そのまま連結すると words() が時刻順にならず、
    Step 3 以降の二分探索が別の単語を指してカット位置がずれる。
    文の切れ目は単語テキストの終端記号で見ているので、まとめても後段の判断は変わらない。
    """
    merged: list[Segment] = []
    for seg in segments:
        # セグメント内の並びは触らない。逆行した時刻は _normalize_segment が
        # 単調になるよう均してあり、そこでの並びは発話順そのものだから。
        words = seg.words
        if merged and seg.start < merged[-1].end:
            prev = merged[-1]
            # どちらも時刻順なので、安定ソートは2列のマージと同じ働きになる。
            # 各セグメント内の相対順は保たれたまま、全体が時刻順に揃う。
            joined = sorted([*prev.words, *words], key=lambda w: (w.start, w.end))
            merged[-1] = Segment(
                start=min(prev.start, seg.start),
                end=max(prev.end, seg.end),
                text=f"{prev.text}{seg.text}",
                words=joined,
            )
            logger.debug(
                "重なったセグメントをまとめた: [%.3f, %.3f] と [%.3f, %.3f]",
                prev.start, prev.end, seg.start, seg.end,
            )
        else:
            merged.append(Segment(start=seg.start, end=seg.end, text=seg.text, words=words))
    return merged


def normalize_asr_result(raw: dict, duration: float, *, language: str = "ja") -> Transcript:
    """ASR バックエンドの生 dict を Transcript に揃える（whisperx / mlx-whisper 両対応）。

    単語のキーが "word" でも "text" でもよく、start/end が欠けていても前後から線形補間する。
    ここを純関数にしておくと、バックエンドが無い環境でも変換の挙動をテストできる。
    """
    if not isinstance(raw, dict):
        raise TranscriptionError(
            f"文字起こし結果が JSON オブジェクトではありません（{type(raw).__name__}）。"
        )

    raw_segments = raw.get("segments")
    if raw_segments is None:
        raw_segments = ()
    if not isinstance(raw_segments, (list, tuple)):
        raise TranscriptionError("文字起こし結果の 'segments' が配列ではありません。")

    segments: list[Segment] = []
    for index, raw_segment in enumerate(raw_segments):
        segment = _normalize_segment(raw_segment, index)
        if segment is not None:
            segments.append(segment)
    segments.sort(key=lambda s: (s.start, s.end))
    segments = _merge_overlapping(segments)

    detected = raw.get("language")
    lang = str(detected) if isinstance(detected, str) and detected.strip() else str(language or "ja")

    total = _as_time(duration) or 0.0
    if total <= 0.0:
        total = max((s.end for s in segments), default=0.0)
    return Transcript(language=lang, duration=float(total), segments=segments)


# ---------------------------------------------------------------------------
# バックエンド（どちらも「生 dict を返す」形に揃える）
# ---------------------------------------------------------------------------


def _torch_device() -> str:
    """whisperx に渡すデバイス名。torch が無ければ CPU 扱い。

    whisperx の裏の CTranslate2 は mps を扱えないので、Apple Silicon でも cpu を返す。
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # 壊れた CUDA 環境で例外が出ることがある
        logger.debug("CUDA の判定に失敗しました（cpu を使います）: %s", exc)
    return "cpu"


def _compute_type_for(requested: str, device: str) -> str:
    """CPU では float16 が使えないので落とす。"""
    value = (requested or "float16").strip()
    if device == "cpu" and value.lower() in ("float16", "fp16", "half"):
        logger.warning("CPU では %s が使えないため compute_type を int8 にします。", value)
        return "int8"
    return value


def _transcribe_with_whisperx(audio_path: str | Path, asr: AsrConfig) -> dict:
    """whisperx（faster-whisper バックエンド）で文字起こしし、生の dict を返す。

    素の transcribe は単語タイムスタンプを持たないので、必ず align まで通す。
    """
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise _BackendUnavailable("whisperx が入っていません（pip install whisperx）。") from exc

    device = _torch_device()
    compute_type = _compute_type_for(asr.compute_type, device)
    language = asr.language or None
    # initial_prompt はアンカー語をモデルにバイアスさせるための必須入力（SPEC Step 2）。
    asr_options: dict[str, Any] = {"initial_prompt": asr.initial_prompt or None}
    if asr.beam_size > 0:
        asr_options["beam_size"] = asr.beam_size

    logger.info(
        "whisperx で文字起こしする（model=%s, device=%s, compute_type=%s）", asr.model, device, compute_type
    )
    detected = language or "ja"
    try:
        model = whisperx.load_model(
            asr.model,
            device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
        )
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio)
        if not isinstance(result, dict):
            raise TranscriptionError("whisperx の transcribe が dict を返しませんでした。")
        detected = str(result.get("language") or language or "ja")
        align_model, metadata = whisperx.load_align_model(language_code=detected, device=device)
        aligned = whisperx.align(
            result.get("segments") or [],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except TranscriptionError:
        raise
    except Exception as exc:  # 外部ライブラリの例外はここで日本語に包む
        raise TranscriptionError(
            f"whisperx での文字起こしに失敗しました: {exc}\n"
            f"model={asr.model} / device={device} / compute_type={compute_type}"
        ) from exc

    out: dict[str, Any] = dict(aligned) if isinstance(aligned, dict) else {"segments": aligned}
    out.setdefault("language", detected)
    return out


def _transcribe_with_mlx(audio_path: str | Path, asr: AsrConfig) -> dict:
    """mlx-whisper（Apple Silicon）で文字起こしし、生の dict を返す。

    `word_timestamps=True` を必ず付ける。これが無いと Step 3 で使う単語時刻が取れない。
    """
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise _BackendUnavailable("mlx-whisper が入っていません（pip install mlx-whisper）。") from exc

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": asr.model,
        "word_timestamps": True,
        "verbose": False,
        # initial_prompt はアンカー語をモデルにバイアスさせるための必須入力（SPEC Step 2）。
        "initial_prompt": asr.initial_prompt or None,
    }
    if asr.language:
        kwargs["language"] = asr.language
    if asr.beam_size and asr.beam_size > 1:
        kwargs["beam_size"] = asr.beam_size

    logger.info("mlx-whisper で文字起こしする（model=%s）", asr.model)
    try:
        result = mlx_whisper.transcribe(str(audio_path), **kwargs)
    except TypeError as exc:
        # ビルドによっては beam_size を受け取らない。落として一度だけ試し直す。
        if "beam_size" not in kwargs:
            raise TranscriptionError(f"mlx-whisper での文字起こしに失敗しました: {exc}") from exc
        logger.warning("mlx-whisper が beam_size を受け取りませんでした（%s）。外して再試行します。", exc)
        kwargs.pop("beam_size", None)
        try:
            result = mlx_whisper.transcribe(str(audio_path), **kwargs)
        except Exception as retry_exc:
            raise TranscriptionError(
                f"mlx-whisper での文字起こしに失敗しました: {retry_exc}\nmodel={asr.model}"
            ) from retry_exc
    except Exception as exc:
        raise TranscriptionError(
            f"mlx-whisper での文字起こしに失敗しました: {exc}\nmodel={asr.model}"
        ) from exc

    if not isinstance(result, dict):
        raise TranscriptionError(
            f"mlx-whisper が dict を返しませんでした（{type(result).__name__}）。"
        )
    return result


#: 実装名 → 呼び出し関数
_BACKENDS: dict[str, Callable[[Path, AsrConfig], dict]] = {
    "whisperx": _transcribe_with_whisperx,
    "mlx_whisper": _transcribe_with_mlx,
}


def _backend_order(asr: AsrConfig) -> tuple[str, ...]:
    """試すバックエンドの順番を決める。config で固定されていればそれだけ。"""
    name = (asr.backend or "auto").strip().lower()
    if name in ("", "auto"):
        return AUTO_BACKEND_ORDER
    resolved = BACKEND_ALIASES.get(name)
    if resolved is None:
        raise TranscriptionError(
            f"asr.backend が不明です: {asr.backend!r}\n"
            "使えるのは 'auto' と " + ", ".join(sorted(set(BACKEND_ALIASES))) + " です。"
        )
    return (resolved,)


def _require_word_timestamps(raw: dict, backend: str) -> None:
    """単語タイムスタンプが1つも無い結果を先に弾く（SPEC Step 2「単語レベルが必要」）。"""
    segments = raw.get("segments") or ()
    if not segments:
        return
    for segment in segments:
        if isinstance(segment, dict) and segment.get("words"):
            return
    raise TranscriptionError(
        f"{backend} が単語タイムスタンプを返しませんでした。\n"
        "アンカー検出（Step 3）は単語ごとの start が要るため、このまま先へ進めません。\n"
        "whisperx なら align まで通るか、mlx-whisper なら word_timestamps に対応しているかを確認してください。"
    )


def _run_backend(audio: Path, asr: AsrConfig) -> tuple[dict, str]:
    """使えるバックエンドで文字起こしし、(生 dict, 使ったバックエンド名) を返す。"""
    order = _backend_order(asr)
    unavailable: list[str] = []
    for name in order:
        try:
            raw = _BACKENDS[name](audio, asr)
        except _BackendUnavailable as exc:
            logger.info("バックエンド %s は使えません: %s", name, exc)
            unavailable.append(f"{name}: {exc}")
            continue
        _require_word_timestamps(raw, name)
        return raw, name

    raise TranscriptionError(
        "文字起こしのバックエンドが見つかりません。次のどちらかを入れてください:\n"
        "  - Apple Silicon（推奨）: pip install mlx-whisper\n"
        "  - それ以外・フォールバック: pip install whisperx\n"
        "見つからなかった理由:\n  " + "\n  ".join(unavailable or ["（候補なし）"]) + "\n"
        "config の asr.backend で使うものを固定できます（auto / whisperx / mlx_whisper）。"
    )


# ---------------------------------------------------------------------------
# キャッシュ（SPEC 6章 Step2）
# ---------------------------------------------------------------------------


def _cache_key(ctx: RunContext) -> tuple[str, str, str] | None:
    """(入力のSHA-256, ASR設定のハッシュ, キャッシュキー)。入力が読めなければ None。"""
    try:
        input_sha = sha256_file(ctx.input_path)
    except RadioCutterError as exc:
        logger.warning("入力のハッシュを計算できませんでした（キャッシュを使いません）: %s", exc)
        return None
    payload = ctx.config.asr.cache_key_payload()
    return input_sha, stable_hash(payload), transcript_cache_key(input_sha, payload)


def _load_cached(ctx: RunContext, key_info: tuple[str, str, str] | None) -> Transcript | None:
    """キャッシュが一致すれば文字起こし済みの Transcript を返す。使えなければ None。"""
    if key_info is None:
        return None
    if ctx.force_transcribe:
        logger.info("再実行が指定されたので文字起こしキャッシュを無視する。")
        return None

    entry = load_cache_entry(cache_path(ctx))
    if entry is None:
        return None
    key = key_info[2]
    if not entry.matches(key):
        logger.info("入力か ASR 設定が変わっているので文字起こしをやり直す。")
        return None

    # キャッシュの transcript_file は work/ の中のファイル名としてだけ扱う（外へ出さない）。
    path = ctx.work_path(Path(entry.transcript_file).name)
    if not path.exists():
        logger.warning("キャッシュは一致しましたが %s がありません。文字起こしをやり直します。", path)
        return None
    try:
        transcript = Transcript.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("%s を読めませんでした（%s）。文字起こしをやり直します。", path, exc)
        return None

    logger.info(
        "文字起こしキャッシュを使う（key=%s…）: %s（%d セグメント / %d 単語）",
        key[:16],
        path,
        len(transcript.segments),
        sum(len(s.words) for s in transcript.segments),
    )
    return transcript


def _save_cache(ctx: RunContext, key_info: tuple[str, str, str] | None) -> None:
    """次回スキップできるようにキャッシュを書く。"""
    if key_info is None:
        return
    input_sha, asr_hash, key = key_info
    entry = TranscriptCacheEntry(
        input_sha256=input_sha,
        asr_hash=asr_hash,
        key=key,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        transcript_file=TRANSCRIPT_FILENAME,
    )
    save_cache_entry(cache_path(ctx), entry)


def _resolve_duration(ctx: RunContext, media: MediaInfo | None, audio: Path) -> float:
    """総尺を決める。probe.json → 音声ファイル → 文字起こし結果の順で当たる。"""
    if media is not None and media.duration > 0:
        return float(media.duration)
    try:
        probed = s1_extract_audio.load(ctx)
    except MissingArtifactError:
        probed = None
    if probed is not None and probed.duration > 0:
        return float(probed.duration)
    try:
        return float(media_duration(audio))
    except RadioCutterError as exc:
        logger.debug("音声の尺を取得できませんでした（%s）。文字起こし結果から求めます。", exc)
        return 0.0


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def run(ctx: RunContext, media: MediaInfo | None = None) -> Transcript:
    """音声を文字起こしして work/transcript.json に書く（SPEC Step 2）。"""
    ctx.ensure_dirs()

    # キャッシュ判定を最初に。ここが全工程の8割の時間を占めるため、無駄打ちを絶対に避ける。
    key_info = _cache_key(ctx)
    cached = _load_cached(ctx, key_info)
    if cached is not None:
        return cached

    audio = ctx.work_path(s1_extract_audio.AUDIO_FILENAME)
    if not audio.exists():
        raise MissingArtifactError(
            f"{s1_extract_audio.AUDIO_FILENAME} がありません: {audio}\n"
            "Step 1（音声抽出）をまだ実行していません。"
            "`radio-cutter run <入力.mp4> --only-step 1` を先に流してください。"
        )

    asr = ctx.config.asr
    if not asr.initial_prompt:
        logger.warning(
            "asr.initial_prompt が空です。アンカー語へのバイアスが効かず、Step 3 の検出が甘くなることがあります。"
        )

    duration = _resolve_duration(ctx, media, audio)
    logger.info(
        "文字起こしを開始する（音声 %s, model=%s, language=%s）",
        fmt_duration(duration),
        asr.model,
        asr.language or "auto",
    )

    started = time.perf_counter()
    raw, backend = _run_backend(audio, asr)
    elapsed = time.perf_counter() - started

    transcript = normalize_asr_result(raw, duration, language=asr.language or "ja")
    word_count = sum(len(s.words) for s in transcript.segments)
    logger.info(
        "文字起こしに %s かかった（backend=%s, %d セグメント / %d 単語）",
        fmt_duration(elapsed),
        backend,
        len(transcript.segments),
        word_count,
    )
    if duration > 0:
        logger.info("音声尺に対する所要時間の比は %.2f 倍だった。", elapsed / duration)

    if not transcript.segments:
        message = "文字起こし結果が空でした。音声の中身（無音・別トラック）を確認してください。"
        ctx.warn(message)
        logger.warning(message)

    save(ctx, transcript)
    _save_cache(ctx, key_info)
    return transcript


def save(ctx: RunContext, result: Transcript) -> None:
    """work/transcript.json を書く。run() の中から呼ぶ。"""
    path = transcript_path(ctx)
    result.save(path)
    logger.debug("文字起こしを保存した: %s", path)


def load(ctx: RunContext) -> Transcript:
    """work/transcript.json を読む（--from-step での再開用）。"""
    path = transcript_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"{TRANSCRIPT_FILENAME} がありません: {path}\n"
            "Step 2（文字起こし）をまだ実行していません。"
            "`radio-cutter transcribe <入力.mp4>` か `radio-cutter run <入力.mp4> --from-step 2` を"
            "先に流してください。"
        )
    try:
        return Transcript.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TranscriptionError(
            f"{TRANSCRIPT_FILENAME} を読めませんでした: {path}\n{exc}\n"
            "壊れている可能性があります。文字起こしをやり直してください。"
        ) from exc
