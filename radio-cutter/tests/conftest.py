"""pytest 共通フィクスチャ。

合成エピソード（tests/fixtures.py）を土台に、
work/out ディレクトリと RunContext をテストごとに用意する。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402
from radio_cutter.config import Config, SilenceConfig, load_config  # noqa: E402
from radio_cutter.context import RunContext  # noqa: E402
from radio_cutter.models import Transcript  # noqa: E402

CONFIG_PATH = ROOT / "config" / "ai-radio.json"

requires_ffmpeg = pytest.mark.skipif(
    not fixtures.ffmpeg_available(), reason="ffmpeg / ffprobe が PATH にありません"
)


@pytest.fixture(scope="session")
def config() -> Config:
    """同梱の ai-radio.json をそのまま使う。"""
    return load_config(CONFIG_PATH)


@pytest.fixture
def transcript() -> Transcript:
    return fixtures.build_transcript()


@pytest.fixture(scope="session")
def episode_video(tmp_path_factory) -> Path:
    """合成エピソードの mp4（セッション内で使い回す）。"""
    if not fixtures.ffmpeg_available():
        pytest.skip("ffmpeg / ffprobe が PATH にありません")
    out = tmp_path_factory.mktemp("episode") / "ep-test.mp4"
    return fixtures.build_test_video(out)


@pytest.fixture(scope="session")
def episode_wav(tmp_path_factory) -> Path:
    """合成エピソードの 16kHz モノラル WAV（ffmpeg 不要）。"""
    out = tmp_path_factory.mktemp("episode-wav") / "audio.wav"
    return fixtures.write_tone_wav(out)


@pytest.fixture
def ctx(tmp_path: Path, config: Config) -> RunContext:
    """入力ファイルは空でよいテスト用の RunContext。"""
    input_path = tmp_path / "ep-test.mp4"
    input_path.write_bytes(b"")
    return RunContext(
        input_path=input_path,
        episode_id="ep-test",
        work_dir=tmp_path / "work" / "ep-test",
        out_dir=tmp_path / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )


@pytest.fixture
def video_ctx(tmp_path: Path, config: Config, episode_video: Path) -> RunContext:
    """本物の mp4 を入力に持つ RunContext（ffmpeg が要るテスト用）。"""
    input_path = tmp_path / "ep-test.mp4"
    shutil.copyfile(episode_video, input_path)
    ctx = RunContext(
        input_path=input_path,
        episode_id="ep-test",
        work_dir=tmp_path / "work" / "ep-test",
        out_dir=tmp_path / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    return ctx


@pytest.fixture
def stub_llm():
    """合成エピソードに合わせた StubLlmClient。"""
    from radio_cutter.llm.client import StubLlmClient

    return StubLlmClient(fixtures.stub_responses(), model="stub-model")


@pytest.fixture
def stub_llm_file(tmp_path: Path) -> Path:
    """--stub-llm に渡せる JSON ファイル。"""
    p = tmp_path / "stub-llm.json"
    p.write_text(json.dumps(fixtures.stub_responses(), ensure_ascii=False), encoding="utf-8")
    return p
