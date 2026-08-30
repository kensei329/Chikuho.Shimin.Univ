"""チャンネル設定（config/*.json）の読み込みと検証。

アンカー語はコードにハードコードしない（SPEC 5章）。
ここは「JSON を読んで型の付いた値にし、おかしければ ConfigError で止める」だけを担う。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------

DEFAULT_SILENCE_NOISE_DB = -32.0
DEFAULT_SILENCE_MIN_DUR = 0.12

#: Step 4 の探索窓（raw_cut_time の何秒前から何秒後までを見るか）
SILENCE_LOOKBACK_SEC = 1.5
SILENCE_LOOKAHEAD_SEC = 0.5

#: 無音区間の終了から何秒手前をカット点にするか
SILENCE_BACKOFF_SEC = 0.05

#: 無音が見つからなかったときのフォールバック量
NO_SILENCE_BACKOFF_SEC = 0.08

VALID_OCCURRENCE = ("first", "last", "nth")
VALID_CUT = ("before", "after")
VALID_POSITION = ("prepend", "append")


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"{where} に必須項目 '{key}' がありません。")
    return d[key]


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{where} は数値である必要があります（実際: {value!r}）。")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} は数値である必要があります（実際: {value!r}）。") from exc


def _as_int(value: Any, where: str) -> int:
    """整数として読む。設定ファイルの書き間違いは必ず ConfigError にする。

    素の int() が投げる ValueError をそのまま上げると、CLI が
    「設定のどこが悪いのか」を出せずにトレースバックで落ちる。
    """
    if isinstance(value, bool):
        raise ConfigError(f"{where} は整数である必要があります（実際: {value!r}）。")
    if isinstance(value, float):
        if value != int(value):
            raise ConfigError(f"{where} は整数である必要があります（実際: {value!r}）。")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} は整数である必要があります（実際: {value!r}）。") from exc


# ---------------------------------------------------------------------------
# アンカー
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MustFollow:
    """候補の直後に指定フレーズが続くことを求めるフィルタ。"""

    phrase: str
    within_sec: float
    fuzzy_threshold: float | None = None  # 未指定ならアンカー本体のしきい値を使う


@dataclass(frozen=True)
class AnchorConfig:
    id: str
    phrase: str
    occurrence: str = "first"           # first / last / nth
    nth: int | None = None              # occurrence == "nth" のときの 1 始まり順位
    search_window_sec: tuple[float, float] | None = None
    cut: str = "before"                 # before / after
    fuzzy_threshold: float = 0.85
    must_follow: MustFollow | None = None

    @property
    def threshold_score(self) -> float:
        """rapidfuzz のスコア（0〜100）に合わせたしきい値。"""
        return self.fuzzy_threshold * 100.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnchorConfig":
        anchor_id = str(_require(d, "id", "anchors[]"))
        where = f"anchors[{anchor_id}]"
        phrase = str(_require(d, "phrase", where))
        if not phrase.strip():
            raise ConfigError(f"{where}.phrase が空です。")

        occurrence = str(d.get("occurrence", "first"))
        if occurrence not in VALID_OCCURRENCE:
            raise ConfigError(
                f"{where}.occurrence は {VALID_OCCURRENCE} のいずれかにしてください（実際: {occurrence!r}）。"
            )
        nth = d.get("nth")
        if occurrence == "nth":
            if nth is None:
                raise ConfigError(f"{where}.occurrence が 'nth' のときは 'nth'（1始まり）が必要です。")
            nth = _as_int(nth, f"{where}.nth")
            if nth < 1:
                raise ConfigError(f"{where}.nth は1以上にしてください（実際: {nth}）。")
        else:
            nth = None

        window = d.get("search_window_sec")
        if window is not None:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ConfigError(f"{where}.search_window_sec は [開始秒, 終了秒] の2要素にしてください。")
            lo = _as_float(window[0], f"{where}.search_window_sec[0]")
            hi = _as_float(window[1], f"{where}.search_window_sec[1]")
            if hi <= lo:
                raise ConfigError(f"{where}.search_window_sec は 開始 < 終了 にしてください（{lo}, {hi}）。")
            window = (lo, hi)

        cut = str(d.get("cut", "before"))
        if cut not in VALID_CUT:
            raise ConfigError(f"{where}.cut は {VALID_CUT} のいずれかにしてください（実際: {cut!r}）。")

        threshold = _as_float(d.get("fuzzy_threshold", 0.85), f"{where}.fuzzy_threshold")
        if not 0.0 < threshold <= 1.0:
            raise ConfigError(f"{where}.fuzzy_threshold は 0 より大きく 1 以下にしてください（実際: {threshold}）。")

        mf_raw = d.get("must_follow")
        must_follow = None
        if mf_raw is not None:
            if not isinstance(mf_raw, dict):
                raise ConfigError(f"{where}.must_follow はオブジェクトにしてください。")
            mf_phrase = str(_require(mf_raw, "phrase", f"{where}.must_follow"))
            if not mf_phrase.strip():
                raise ConfigError(f"{where}.must_follow.phrase が空です。フィルタとして意味を成しません。")
            mf_within = _as_float(
                _require(mf_raw, "within_sec", f"{where}.must_follow"), f"{where}.must_follow.within_sec"
            )
            if mf_within <= 0:
                raise ConfigError(
                    f"{where}.must_follow.within_sec は 0 より大きくしてください（実際: {mf_within}）。"
                    "候補の終端から何秒以内に続くかを指す値です。"
                )
            mf_threshold = mf_raw.get("fuzzy_threshold")
            if mf_threshold is not None:
                mf_threshold = _as_float(mf_threshold, f"{where}.must_follow.fuzzy_threshold")
                if not 0.0 < mf_threshold <= 1.0:
                    raise ConfigError(
                        f"{where}.must_follow.fuzzy_threshold は 0 より大きく 1 以下にしてください"
                        f"（実際: {mf_threshold}）。"
                    )
            must_follow = MustFollow(phrase=mf_phrase, within_sec=mf_within, fuzzy_threshold=mf_threshold)

        return cls(
            id=anchor_id,
            phrase=phrase,
            occurrence=occurrence,
            nth=nth,
            search_window_sec=window,
            cut=cut,
            fuzzy_threshold=threshold,
            must_follow=must_follow,
        )


# ---------------------------------------------------------------------------
# セグメント / ハイライト
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentConfig:
    name: str
    file: str
    from_: str      # アンカーID または "start"
    to: str         # アンカーID または "end"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SegmentConfig":
        name = str(_require(d, "name", "segments[]"))
        where = f"segments[{name}]"
        return cls(
            name=name,
            file=str(_require(d, "file", where)),
            from_=str(_require(d, "from", where)),
            to=str(_require(d, "to", where)),
        )


@dataclass(frozen=True)
class HighlightConfig:
    file: str = "01_highlight.mp4"
    source_segment: str = "main"
    target_duration_sec: float = 30.0
    min_duration_sec: float = 20.0
    max_duration_sec: float = 45.0
    position: str = "prepend"
    allow_multi_cut: bool = False
    num_candidates: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HighlightConfig":
        where = "highlight"
        position = str(d.get("position", "prepend"))
        if position not in VALID_POSITION:
            raise ConfigError(f"{where}.position は {VALID_POSITION} のいずれかにしてください（実際: {position!r}）。")
        target = _as_float(d.get("target_duration_sec", 30.0), f"{where}.target_duration_sec")
        lo = _as_float(d.get("min_duration_sec", 20.0), f"{where}.min_duration_sec")
        hi = _as_float(d.get("max_duration_sec", 45.0), f"{where}.max_duration_sec")
        if not lo <= target <= hi:
            raise ConfigError(
                f"{where} の尺は min <= target <= max にしてください（min={lo}, target={target}, max={hi}）。"
            )
        return cls(
            file=str(d.get("file", "01_highlight.mp4")),
            source_segment=str(d.get("source_segment", "main")),
            target_duration_sec=target,
            min_duration_sec=lo,
            max_duration_sec=hi,
            position=position,
            allow_multi_cut=bool(d.get("allow_multi_cut", False)),
            num_candidates=_as_int(d.get("num_candidates", 3), f"{where}.num_candidates"),
        )


# ---------------------------------------------------------------------------
# ASR / LLM / レンダリング / YouTube / 無音検出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsrConfig:
    model: str = "mlx-community/whisper-large-v3-mlx"
    language: str = "ja"
    initial_prompt: str = ""
    backend: str = "auto"       # auto / whispermlx / whisperx / mlx_whisper
    compute_type: str = "float16"
    beam_size: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AsrConfig":
        return cls(
            model=str(d.get("model", "mlx-community/whisper-large-v3-mlx")),
            language=str(d.get("language", "ja")),
            initial_prompt=str(d.get("initial_prompt", "")),
            backend=str(d.get("backend", "auto")),
            compute_type=str(d.get("compute_type", "float16")),
            beam_size=_as_int(d.get("beam_size", 5), "asr.beam_size"),
        )

    def cache_key_payload(self) -> dict[str, Any]:
        """キャッシュキーに混ぜる ASR 設定（SPEC 6章 Step2）。"""
        return {
            "model": self.model,
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "backend": self.backend,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
        }


@dataclass(frozen=True)
class LlmConfig:
    """LLM の呼び出し設定。

    `max_retries` は「試行回数の上限」として扱う（3 なら API 呼び出しは最大3回、
    つまり投げ直しは2回まで）。SPEC 9章の「3回までリトライ」はどちらとも読めるので、
    呼び出し回数が読んだとおりになる側に寄せてある。
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_retries: int = 3
    max_tokens: int = 8000
    temperature: float = 1.0
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_sec: float = 300.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LlmConfig":
        retries = _as_int(d.get("max_retries", 3), "llm.max_retries")
        if retries < 1:
            raise ConfigError(f"llm.max_retries は1以上にしてください（実際: {retries}）。")
        return cls(
            provider=str(d.get("provider", "anthropic")),
            model=str(d.get("model", "claude-sonnet-4-6")),
            max_retries=retries,
            max_tokens=_as_int(d.get("max_tokens", 8000), "llm.max_tokens"),
            temperature=_as_float(d.get("temperature", 1.0), "llm.temperature"),
            api_key_env=str(d.get("api_key_env", "ANTHROPIC_API_KEY")),
            timeout_sec=_as_float(d.get("timeout_sec", 300.0), "llm.timeout_sec"),
        )


@dataclass(frozen=True)
class RenderConfig:
    video_codec: str = "h264_videotoolbox"
    video_bitrate: str = "12M"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    fallback_video_codec: str = "libx264"
    fallback_extra_args: tuple[str, ...] = ("-preset", "veryfast", "-crf", "20")
    duration_tolerance_sec: float = 0.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderConfig":
        extra = d.get("fallback_extra_args")
        if extra is None:
            extra_args: tuple[str, ...] = ("-preset", "veryfast", "-crf", "20")
        else:
            if not isinstance(extra, (list, tuple)):
                raise ConfigError("render.fallback_extra_args は文字列の配列にしてください。")
            extra_args = tuple(str(x) for x in extra)
        return cls(
            video_codec=str(d.get("video_codec", "h264_videotoolbox")),
            video_bitrate=str(d.get("video_bitrate", "12M")),
            audio_codec=str(d.get("audio_codec", "aac")),
            audio_bitrate=str(d.get("audio_bitrate", "192k")),
            fallback_video_codec=str(d.get("fallback_video_codec", "libx264")),
            fallback_extra_args=extra_args,
            duration_tolerance_sec=_as_float(d.get("duration_tolerance_sec", 0.5), "render.duration_tolerance_sec"),
        )


@dataclass(frozen=True)
class YoutubeConfig:
    channel_links: tuple[str, ...] = ()
    fixed_footer: str = ""
    hashtags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "YoutubeConfig":
        return cls(
            channel_links=tuple(str(x) for x in d.get("channel_links", [])),
            fixed_footer=str(d.get("fixed_footer", "")),
            hashtags=tuple(str(x) for x in d.get("hashtags", [])),
        )


@dataclass(frozen=True)
class SilenceConfig:
    noise_db: float = DEFAULT_SILENCE_NOISE_DB
    min_duration_sec: float = DEFAULT_SILENCE_MIN_DUR

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SilenceConfig":
        return cls(
            noise_db=_as_float(d.get("noise_db", DEFAULT_SILENCE_NOISE_DB), "silence.noise_db"),
            min_duration_sec=_as_float(d.get("min_duration_sec", DEFAULT_SILENCE_MIN_DUR), "silence.min_duration_sec"),
        )


# ---------------------------------------------------------------------------
# ルート
# ---------------------------------------------------------------------------


@dataclass
class Config:
    channel: str
    anchors: list[AnchorConfig]
    segments: list[SegmentConfig]
    highlight: HighlightConfig
    asr: AsrConfig
    llm: LlmConfig
    render: RenderConfig
    youtube: YoutubeConfig
    silence: SilenceConfig
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # ----- 参照ヘルパ -----

    def anchor(self, anchor_id: str) -> AnchorConfig:
        for a in self.anchors:
            if a.id == anchor_id:
                return a
        raise ConfigError(f"アンカー '{anchor_id}' が config にありません。")

    def anchor_ids(self) -> list[str]:
        return [a.id for a in self.anchors]

    def segment(self, name: str) -> SegmentConfig:
        for s in self.segments:
            if s.name == name:
                return s
        raise ConfigError(f"セグメント '{name}' が config にありません。")

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, path: Path | None = None) -> "Config":
        if not isinstance(d, dict):
            raise ConfigError("設定ファイルの中身が JSON オブジェクトではありません。")

        anchors_raw = d.get("anchors")
        if not isinstance(anchors_raw, list) or not anchors_raw:
            raise ConfigError("'anchors' は1つ以上の要素を持つ配列にしてください。")
        anchors = [AnchorConfig.from_dict(a) for a in anchors_raw]
        ids = [a.id for a in anchors]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"アンカーIDが重複しています: {ids}")

        segments_raw = d.get("segments")
        if not isinstance(segments_raw, list) or not segments_raw:
            raise ConfigError("'segments' は1つ以上の要素を持つ配列にしてください。")
        segments = [SegmentConfig.from_dict(s) for s in segments_raw]

        known = set(ids) | {"start", "end"}
        for seg in segments:
            for label, value in (("from", seg.from_), ("to", seg.to)):
                if value not in known:
                    raise ConfigError(
                        f"segments[{seg.name}].{label} が未知の参照です: {value!r}（使えるのは {sorted(known)}）。"
                    )
        seg_names = [s.name for s in segments]
        if len(set(seg_names)) != len(seg_names):
            raise ConfigError(f"セグメント名が重複しています: {seg_names}")
        seg_files = [s.file for s in segments]
        if len(set(seg_files)) != len(seg_files):
            raise ConfigError(f"セグメントの出力ファイル名が重複しています: {seg_files}")

        highlight = HighlightConfig.from_dict(d.get("highlight", {}) or {})
        if highlight.source_segment not in seg_names:
            raise ConfigError(
                f"highlight.source_segment '{highlight.source_segment}' が segments にありません（{seg_names}）。"
            )
        # 同じ out/ に書き出すので、名前がぶつかると片方が黙って消える。
        if highlight.file in seg_files:
            raise ConfigError(
                f"highlight.file '{highlight.file}' が segments の出力ファイル名と重複しています。"
                "同じディレクトリに書き出すため、どちらかが上書きされます。"
            )

        return cls(
            channel=str(d.get("channel", "")),
            anchors=anchors,
            segments=segments,
            highlight=highlight,
            asr=AsrConfig.from_dict(d.get("asr", {}) or {}),
            llm=LlmConfig.from_dict(d.get("llm", {}) or {}),
            render=RenderConfig.from_dict(d.get("render", {}) or {}),
            youtube=YoutubeConfig.from_dict(d.get("youtube", {}) or {}),
            silence=SilenceConfig.from_dict(d.get("silence", {}) or {}),
            path=path,
            raw=d,
        )


def load_config(path: str | Path) -> Config:
    """設定ファイルを読んで検証する。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {p}")
    if p.is_dir():
        raise ConfigError(f"設定ファイルではなくディレクトリが指定されています: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"設定ファイルを読めません: {p}\n{exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"設定ファイルが UTF-8 として読めません: {p}\n{exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"設定ファイルの JSON が壊れています: {p}\n{exc}") from exc
    return Config.from_dict(data, path=p)
