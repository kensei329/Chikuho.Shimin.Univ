"""radio-cutter の例外型。

方針（SPEC 9章）：
- 握りつぶさない。原因と次の一手を必ずメッセージに載せる。
- 自動で代替を選ばない失敗（アンカー未検出など）は専用の例外にする。
"""

from __future__ import annotations


class RadioCutterError(Exception):
    """このツールが投げる例外の基底。CLI はこれを捕まえて終了コード1で止まる。"""


class ConfigError(RadioCutterError):
    """設定ファイルが読めない・スキーマに合わない。"""


class MissingArtifactError(RadioCutterError):
    """--from-step で再開しようとしたが、必要な中間ファイルが無い。"""


class FfmpegError(RadioCutterError):
    """ffmpeg / ffprobe が非ゼロ終了した。stderr をそのまま持つ。"""

    def __init__(self, message: str, *, cmd: list[str] | None = None, stderr: str = "", returncode: int | None = None):
        super().__init__(message)
        self.cmd = cmd or []
        self.stderr = stderr
        self.returncode = returncode

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.cmd:
            parts.append("コマンド: " + " ".join(self.cmd))
        if self.returncode is not None:
            parts.append(f"終了コード: {self.returncode}")
        if self.stderr:
            parts.append("--- ffmpeg stderr ---\n" + self.stderr.rstrip())
        return "\n".join(parts)


class TranscriptionError(RadioCutterError):
    """文字起こしバックエンドが使えない、または失敗した。"""


class AnchorNotFoundError(RadioCutterError):
    """アンカー候補が0件。勝手に代替位置を選ばずここで止める。"""


class AnchorOrderError(RadioCutterError):
    """アンカーBがAより前にある。設定ミスの可能性が高いので止める。"""


class LlmError(RadioCutterError):
    """LLM 呼び出しがリトライ上限まで失敗した。"""


class HighlightError(RadioCutterError):
    """ハイライト候補が全滅した。"""


class RenderError(RadioCutterError):
    """書き出しに失敗した。"""
