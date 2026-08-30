"""1回の実行を通して持ち回る文脈（パス・設定・CLI上書き・警告の置き場）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, SilenceConfig
from .models import LlmCallRecord


@dataclass
class RunContext:
    """パイプライン全体で共有する状態。

    ステップ関数は必ずこれを第1引数に受け取る。中間ファイルの置き場と
    「今回の実行で何が起きたか」（warnings / llm_calls）をここに集める。
    """

    input_path: Path
    episode_id: str
    work_dir: Path          # work/<episode_id>
    out_dir: Path           # out/<episode_id>
    config: Config
    silence: SilenceConfig
    dry_run: bool = False
    force_transcribe: bool = False
    warnings: list[str] = field(default_factory=list)
    llm_calls: list[LlmCallRecord] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.work_dir = Path(self.work_dir)
        self.out_dir = Path(self.out_dir)

    # ----- パス -----

    def work_path(self, *parts: str) -> Path:
        return self.work_dir.joinpath(*parts)

    def out_path(self, *parts: str) -> Path:
        return self.out_dir.joinpath(*parts)

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 記録 -----

    def warn(self, message: str) -> None:
        """decisions.json に残る警告。同じ文言は1回だけ。"""
        if message not in self.warnings:
            self.warnings.append(message)

    def record_llm_call(self, record: LlmCallRecord) -> None:
        self.llm_calls.append(record)
