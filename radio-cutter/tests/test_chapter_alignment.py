"""チャプターの時刻が final.mp4 上の正しい位置を指しているかを、実フレームで確かめる。

SPEC 10章 Phase 3 の受け入れ基準は「実際に再生して確認」。
それを機械でやるために、1秒ごとに明るさが変わる映像を入力に使う。
final.mp4 のチャプター位置のフレームと、元動画の対応する位置のフレームの
明るさが一致すれば、時刻変換（Step 6-a）と連結（Step 7）が噛み合っている。

ここが崩れると、概要欄のチャプターを押した視聴者が別の場面に飛ぶ。
アンカー検出からカット点の精密化、ハイライトのスナップ、
final.mp4 のタイムラインへの変換、書き出しと連結まで、全部が同時に正しくないと通らない。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import Config, SilenceConfig
from radio_cutter.context import RunContext
from radio_cutter.llm.client import StubLlmClient
from radio_cutter.pipeline import run_pipeline
from radio_cutter.steps import s1_extract_audio as s1

#: 明るさの一致とみなす差（再エンコードで数レベルは動く）
LUMA_TOLERANCE = 4

#: チャプター境界ちょうどはフレームの取り合いになるので、少し内側を見る
PROBE_OFFSET_SEC = 0.4


def build_ramp_video(path: Path, *, duration: float = fixtures.EPISODE_DURATION) -> Path:
    """1秒ごとに輝度が上がる映像＋合成エピソードと同じ無音配置の音声。"""
    wav = path.with_suffix(".ramp.wav")
    fixtures.write_tone_wav(wav, duration=duration)
    video = (
        f"color=c=black:s=64x64:r=10:d={duration},"
        "geq=lum='floor(T)*4':cb=128:cr=128,format=yuv420p"
    )
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi", "-i", video,
            "-i", str(wav),
            "-shortest",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"検証用の映像を作れませんでした:\n{proc.stderr[-2000:]}")
    wav.unlink(missing_ok=True)
    return path


def frame_luma(path: Path, at: float) -> int:
    """指定秒のフレームを 1x1 に潰して平均輝度を取る。"""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
            "-ss", f"{max(0.0, at):.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", "format=gray,scale=1:1",
            "-f", "rawvideo", "-",
        ],
        capture_output=True,
    )
    if not proc.stdout:
        raise AssertionError(f"{path.name} の {at:.3f}秒 のフレームを取れませんでした。")
    return proc.stdout[0]


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_チャプターは_final_mp4_の正しい位置を指す(
    tmp_path: Path, config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    source = build_ramp_video(tmp_path / "ep.mp4")

    ctx = RunContext(
        input_path=source,
        episode_id="ep",
        work_dir=tmp_path / "work" / "ep",
        out_dir=tmp_path / "out" / "ep",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    fixtures.build_transcript().save(ctx.work_path("transcript.json"))
    s1.run(ctx)

    result = run_pipeline(
        ctx, from_step=3, llm=StubLlmClient(fixtures.stub_responses(), model="stub")
    )

    assert result.cuts is not None and result.highlight is not None
    assert result.metadata is not None and result.metadata.chapters

    cut_a = result.cuts["A"].cut_time
    cut_b = result.cuts["B"].cut_time
    highlight = result.highlight.selected
    dh = highlight.duration
    dm = cut_b - cut_a

    def final_to_source(t: float) -> float:
        """final 上の秒を元動画の秒に戻す（既定構成の逆写像）。"""
        if t < dh:
            return highlight.start + t
        if t < dh + dm:
            return cut_a + (t - dh)
        return cut_b + (t - dh - dm)

    final = ctx.out_path("final.mp4")
    assert final.exists()

    mismatches: list[str] = []
    for chapter in result.metadata.chapters:
        probe_at = chapter.time_sec + PROBE_OFFSET_SEC
        expected_source = final_to_source(probe_at)
        got = frame_luma(final, probe_at)
        want = frame_luma(source, expected_source)
        if abs(got - want) > LUMA_TOLERANCE:
            mismatches.append(
                f"{chapter.time_sec:.1f}秒「{chapter.label}」は元動画の "
                f"{expected_source:.2f}秒 を指すはずが、輝度 {got}（期待 {want}）"
            )

    assert not mismatches, "チャプターがずれています:\n  " + "\n  ".join(mismatches)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_最初のチャプターはハイライトの冒頭を指す(tmp_path: Path, config: Config) -> None:
    """SPEC 6-a「`0:00` は必ずハイライト部分に割り当てる」。

    0:00 のフレームが、元動画のハイライト開始位置と同じ場面であること。
    """
    source = build_ramp_video(tmp_path / "ep.mp4")
    ctx = RunContext(
        input_path=source,
        episode_id="ep",
        work_dir=tmp_path / "work" / "ep",
        out_dir=tmp_path / "out" / "ep",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    fixtures.build_transcript().save(ctx.work_path("transcript.json"))
    s1.run(ctx)
    result = run_pipeline(
        ctx, from_step=3, llm=StubLlmClient(fixtures.stub_responses(), model="stub")
    )

    assert result.metadata is not None and result.highlight is not None
    assert result.metadata.chapters[0].time_sec == 0.0

    final = ctx.out_path("final.mp4")
    at = PROBE_OFFSET_SEC
    got = frame_luma(final, at)
    want = frame_luma(source, result.highlight.selected.start + at)
    assert abs(got - want) <= LUMA_TOLERANCE, (
        "0:00 がハイライトの冒頭を指していません"
        f"（輝度 {got} / 期待 {want}）"
    )
