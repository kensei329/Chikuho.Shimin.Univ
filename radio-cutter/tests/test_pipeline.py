"""SPEC 3章「全体パイプライン」と 7章「CLI仕様」の再開まわりを守るテスト。

このファイルが守らせたいこと:

- `--from-step N` を指定したら、N より前のステップは **中間ファイルから読む**（作り直さない）。
  SPEC 3章「各ステップは中間ファイルを work/<episode_id>/ に残し、--from-step N で
  途中から再実行できること」。
- 必要な中間ファイルが無いのに再開しようとしたら **止まる**。しかもエラーメッセージには
  「次にどのコマンドを打てばよいか」が書いてある（SPEC 9章「握りつぶさない」）。
- `--only-step N` はそのステップだけ、`--dry-run` は Step 6 まで、`--preview-only` は
  Step 8 だけ（SPEC 7章）。
- `decisions.json` は「あとから何が起きたか追えるようにする」ためのものなので、
  --dry-run でも、途中でステップが落ちても書かれる（SPEC 8章）。
- 所要秒数は必ず記録する（SPEC 11章「ログは各ステップの開始・終了・所要秒数を必ず出す」）。

この環境には ASR バックエンド（mlx-whisper / whisperx）が無いため、Step 2 の成果物である
`work/transcript.json` は tests/fixtures.py の合成文字起こしを直接置いてから Step 3 以降を流す。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import Config, SilenceConfig
from radio_cutter.context import RunContext
from radio_cutter.decisions import decisions_path
from radio_cutter.errors import MissingArtifactError, RadioCutterError, RenderError
from radio_cutter.llm.client import StubLlmClient
from radio_cutter.models import CutPoint, Transcript
from radio_cutter.pipeline import (
    DRY_RUN_LAST_STEP,
    FIRST_STEP,
    LAST_STEP,
    PREVIEW_STEP,
    plan_steps,
    run_pipeline,
    run_steps,
)
from radio_cutter.steps import s1_extract_audio as s1
from radio_cutter.steps import s2_transcribe as s2
from radio_cutter.steps import s3_find_anchors as s3
from radio_cutter.steps import s4_refine_cuts as s4
from radio_cutter.steps import s5_pick_highlight as s5
from radio_cutter.steps import s6_metadata as s6
from radio_cutter.steps import s7_render as s7

ALL_STEPS = list(range(FIRST_STEP, LAST_STEP + 1))

#: Step 8 が作るプレビュー（SPEC Step 8）
PREVIEW_FILES = ("cut_A.mp4", "cut_B.mp4", "highlight_in.mp4", "highlight_out.mp4")

#: Step 7 が作る成果物（config/ai-radio.json の segments / highlight / final）
RENDER_FILES = ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4", "final.mp4")

#: この環境に ASR バックエンドがあるか（あれば Step 2 は本当に走るので、落ちる前提のテストは飛ばす）
HAS_ASR_BACKEND = any(
    importlib.util.find_spec(name) is not None for name in ("mlx_whisper", "whisperx")
)


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def seed_transcript(ctx: RunContext) -> Transcript:
    """Step 2 の成果物（work/transcript.json）を先に置く。

    この環境に ASR バックエンドが無いので、合成エピソードの文字起こしを
    Step 2 が書いたことにして Step 3 以降を流せるようにする。
    """
    ctx.ensure_dirs()
    transcript = fixtures.build_transcript()
    transcript.save(ctx.work_path(s2.TRANSCRIPT_FILENAME))
    return transcript


def seed_anchors(ctx: RunContext) -> dict:
    """Step 3 の成果物（work/anchors.json）まで置く。ffmpeg は要らない。"""
    transcript = seed_transcript(ctx)
    return s3.run(ctx, transcript)


def forbid_step(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """そのステップの run() が呼ばれたら失敗させる。

    「load() で済ませたはず」を証明するために使う。中間ファイルがあるのに
    実行し直してしまったら、このテストが落ちる。
    """

    def boom(*args, **kwargs):  # pragma: no cover - 呼ばれたら即失敗
        raise AssertionError(
            f"Step {module.STEP}（{module.NAME}）は中間ファイルから読むはずで、実行してはいけません。"
        )

    monkeypatch.setattr(module, "run", boom)


def read_decisions(ctx: RunContext) -> dict:
    """out/<episode_id>/decisions.json を読む。"""
    return json.loads(decisions_path(ctx).read_text(encoding="utf-8"))


def out_exists(ctx: RunContext, *names: str) -> list[bool]:
    return [ctx.out_path(name).exists() for name in names]


@pytest.fixture
def ready_ctx(video_ctx: RunContext) -> RunContext:
    """Step 1・2 の中間ファイルが揃っていて、Step 3 から流せる RunContext。

    audio.wav / probe.json は本物の Step 1 で作り、transcript.json は合成文字起こしを置く。
    """
    s1.run(video_ctx)
    seed_transcript(video_ctx)
    return video_ctx


@pytest.fixture(scope="module")
def seeded_work_dir(tmp_path_factory, config: Config, episode_video: Path) -> Path:
    """Step 1〜6 の中間ファイルが全部揃った work/ を1回だけ作り、その場所を返す。

    「必要な中間ファイルが揃っていれば --from-step N はどこからでも再開できる」を
    総当たりで確かめるための土台。テストごとにここからコピーして使う（元は書き換えない）。
    """
    base = tmp_path_factory.mktemp("pipeline-seed")
    input_path = base / "ep-test.mp4"
    shutil.copyfile(episode_video, input_path)
    ctx = RunContext(
        input_path=input_path,
        episode_id="ep-test",
        work_dir=base / "work" / "ep-test",
        out_dir=base / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    s1.run(ctx)
    seed_transcript(ctx)
    # --dry-run（Step 6 まで）で anchors / cuts / highlight / metadata まで作る。
    run_pipeline(
        ctx,
        from_step=3,
        dry_run=True,
        llm=StubLlmClient(fixtures.stub_responses(), model="stub-model"),
    )
    return ctx.work_dir


def ctx_from_seed(
    seed: Path, tmp_path: Path, config: Config, episode_video: Path
) -> RunContext:
    """seeded_work_dir をコピーして、まっさらな out/ を持つ RunContext を作る。"""
    input_path = tmp_path / "ep-test.mp4"
    shutil.copyfile(episode_video, input_path)
    work_dir = tmp_path / "work" / "ep-test"
    shutil.copytree(seed, work_dir)
    ctx = RunContext(
        input_path=input_path,
        episode_id="ep-test",
        work_dir=work_dir,
        out_dir=tmp_path / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    return ctx


# ---------------------------------------------------------------------------
# plan_steps —— どのステップを実行するかの決定（SPEC 7章）
# ---------------------------------------------------------------------------


class TestPlanSteps:
    """CLI オプションの組み合わせから実行計画を決めるところ。ここが狂うと全部狂う。"""

    def test_default_runs_step_1_to_8_in_order(self) -> None:
        """既定は Step 1〜8 を順番に全部（SPEC 3章のパイプライン）。"""
        assert plan_steps() == [1, 2, 3, 4, 5, 6, 7, 8]

    @pytest.mark.parametrize("from_step", ALL_STEPS)
    def test_from_step_starts_at_n_and_runs_to_the_end(self, from_step: int) -> None:
        """--from-step N は「N から最後まで」。N より前は中間ファイルを再利用する。"""
        assert plan_steps(from_step=from_step) == list(range(from_step, LAST_STEP + 1))

    @pytest.mark.parametrize("only_step", ALL_STEPS)
    def test_only_step_runs_exactly_one_step(self, only_step: int) -> None:
        """--only-step N は N だけ（SPEC 7章「Nだけ実行」）。"""
        assert plan_steps(only_step=only_step) == [only_step]

    def test_dry_run_stops_after_step_6(self) -> None:
        """--dry-run は「Step 6まで実行し、書き出しを行わない」（SPEC 7章）。"""
        plan = plan_steps(dry_run=True)
        assert plan == list(range(FIRST_STEP, DRY_RUN_LAST_STEP + 1))
        assert 7 not in plan and 8 not in plan

    def test_dry_run_with_from_step_still_stops_at_step_6(self) -> None:
        """--from-step と --dry-run の併用でも書き出しには進まない。"""
        assert plan_steps(from_step=4, dry_run=True) == [4, 5, 6]

    def test_preview_only_runs_step_8_alone(self) -> None:
        """--preview-only は「Step 8のプレビューだけ生成」（SPEC 7章）。"""
        assert plan_steps(preview_only=True) == [PREVIEW_STEP] == [8]

    def test_from_step_and_only_step_cannot_be_combined(self) -> None:
        """矛盾する指定は走り出す前に止める。"""
        with pytest.raises(RadioCutterError):
            plan_steps(from_step=3, only_step=5)

    def test_dry_run_and_preview_only_cannot_be_combined(self) -> None:
        """--dry-run は書き出さない、--preview-only は書き出す。両立しない。"""
        with pytest.raises(RadioCutterError):
            plan_steps(dry_run=True, preview_only=True)

    @pytest.mark.parametrize("only_step", [7, 8])
    def test_dry_run_rejects_only_step_after_6(self, only_step: int) -> None:
        """--dry-run のまま Step 7・8 だけ実行しろ、は成立しない。"""
        with pytest.raises(RadioCutterError):
            plan_steps(only_step=only_step, dry_run=True)

    @pytest.mark.parametrize("from_step", [7, 8])
    def test_dry_run_rejects_from_step_after_6(self, from_step: int) -> None:
        """実行するステップが1つも無くなる指定は、走り出す前にエラーにする。"""
        with pytest.raises(RadioCutterError):
            plan_steps(from_step=from_step, dry_run=True)

    def test_preview_only_rejects_conflicting_only_step(self) -> None:
        """--preview-only は Step 8 固定。他のステップ指定とは併用できない。"""
        with pytest.raises(RadioCutterError):
            plan_steps(only_step=5, preview_only=True)

    @pytest.mark.parametrize("bad", [0, -1, 9, 42])
    def test_from_step_out_of_range_is_rejected(self, bad: int) -> None:
        """ステップ番号は 1〜8。範囲外は黙って丸めずエラーにする。"""
        with pytest.raises(RadioCutterError):
            plan_steps(from_step=bad)

    @pytest.mark.parametrize("bad", [0, -1, 9, 42])
    def test_only_step_out_of_range_is_rejected(self, bad: int) -> None:
        with pytest.raises(RadioCutterError):
            plan_steps(only_step=bad)

    def test_plan_is_always_ascending(self) -> None:
        """計画は必ず昇順。Step 4 の cuts を Step 3 より先に作ることはない。"""
        for from_step in ALL_STEPS:
            plan = plan_steps(from_step=from_step)
            assert plan == sorted(plan)


def test_run_steps_rejects_out_of_range_step(ctx: RunContext) -> None:
    """run_steps に 1〜8 の外を渡したら、何も実行せずエラーにする。"""
    with pytest.raises(RadioCutterError):
        run_steps(ctx, [3, 99])


# ---------------------------------------------------------------------------
# --from-step —— 前段は load() で済ませる
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_from_step_3_loads_step_1_and_2_artifacts(
    ready_ctx: RunContext, stub_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-step 3 では Step 1・2 を実行せず、probe.json と transcript.json を読む。

    文字起こしは全工程の8割の時間を占める（SPEC Step 2）。ここで作り直したら
    再開の意味が無いので、run() が呼ばれたら失敗するようにして確かめる。
    """
    forbid_step(monkeypatch, s1)
    forbid_step(monkeypatch, s2)

    result = run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)

    assert result.executed_steps == [3, 4, 5, 6]
    assert set(result.timings) == {3, 4, 5, 6}

    # Step 2 の中間ファイルから読めていること
    assert result.transcript is not None
    assert result.transcript.duration == pytest.approx(fixtures.EPISODE_DURATION)
    assert len(result.transcript.segments) == len(fixtures.UTTERANCES)

    # Step 1 の probe.json から読めていること
    assert result.media is not None
    assert float(result.media.duration) == pytest.approx(fixtures.EPISODE_DURATION, abs=0.5)

    # 読んだ中間ファイルを土台に Step 3・4 が正しく動いていること
    assert set(result.anchors or {}) == {"A", "B"}
    assert result.anchors["A"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW, abs=0.01)
    assert result.anchors["B"].raw_cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_B_RAW, abs=0.01)
    assert result.cuts["A"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.01)
    assert result.cuts["B"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.01)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_from_step_5_loads_transcript_and_cuts(
    ready_ctx: RunContext, stub_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-step 5 は Step 3・4 の結果（anchors.json / cuts.json）を読んで使う。"""
    run_pipeline(ready_ctx, only_step=3)
    run_pipeline(ready_ctx, only_step=4)

    forbid_step(monkeypatch, s3)
    forbid_step(monkeypatch, s4)

    result = run_pipeline(ready_ctx, from_step=5, dry_run=True, llm=stub_llm)

    assert result.executed_steps == [5, 6]
    assert result.cuts["A"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.01)
    assert result.cuts["B"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.01)
    assert result.highlight is not None


# ---------------------------------------------------------------------------
# 中間ファイルが無いときは止まる（SPEC 9章「握りつぶさない」）
# ---------------------------------------------------------------------------


def test_from_step_4_without_audio_wav_tells_the_next_command(ctx: RunContext) -> None:
    """Step 1 の audio.wav が無ければ Step 4 は止まり、次に打つコマンドを示す。

    Step 4 の無音検出は Step 1 が作る音声を読む（SPEC Step 4）。
    黙って silencedetect を諦めて raw_cut_time のまま進んではいけない。
    """
    seed_anchors(ctx)  # anchors.json はある。audio.wav だけが無い状態を作る。
    assert not s1.audio_path(ctx).exists()

    with pytest.raises(MissingArtifactError) as excinfo:
        run_pipeline(ctx, from_step=4)

    message = str(excinfo.value)
    assert s1.AUDIO_FILENAME in message
    assert "radio-cutter run" in message
    assert f"--only-step {s1.STEP}" in message
    assert str(ctx.input_path) in message
    # cuts.json は作られていない（中途半端な結果を残さない）
    assert not s4.cuts_path(ctx).exists()


def test_from_step_4_without_anchors_tells_the_next_command(ctx: RunContext) -> None:
    """Step 3 の anchors.json が無ければ Step 4 は止まり、Step 3 を促す。"""
    seed_transcript(ctx)

    with pytest.raises(MissingArtifactError) as excinfo:
        run_pipeline(ctx, from_step=4)

    message = str(excinfo.value)
    assert s3.ANCHORS_FILE in message
    assert "radio-cutter run" in message
    assert f"--only-step {s3.STEP}" in message


@pytest.mark.parametrize("from_step", [2, 3, 4, 5, 6, 7, 8])
def test_resume_without_any_artifacts_raises_missing_artifact(
    ctx: RunContext, from_step: int
) -> None:
    """中間ファイルが1つも無いのに再開しようとしたら、必ず MissingArtifactError で止まる。

    握りつぶして「空の結果」で先へ進んではいけない。
    メッセージには次に打つコマンドが入っていること。
    """
    ctx.ensure_dirs()

    with pytest.raises(MissingArtifactError) as excinfo:
        run_pipeline(ctx, from_step=from_step)

    message = str(excinfo.value)
    assert isinstance(excinfo.value, RadioCutterError)  # CLI が捕まえられる型であること
    assert "radio-cutter" in message
    assert ("--only-step" in message) or ("--from-step" in message) or ("transcribe" in message)


def test_from_step_7_without_highlight_tells_the_next_command(ctx: RunContext) -> None:
    """Step 5 の highlight.json が無ければ Step 7 は止まり、Step 5 を促す。

    cuts.json はあるので、足りないのがハイライトだけだと分かる形で止まること。
    """
    ctx.ensure_dirs()
    s4.save(
        ctx,
        {
            "A": CutPoint(
                anchor_id="A",
                raw_cut_time=fixtures.EXPECTED_ANCHOR_A_RAW,
                cut_time=fixtures.EXPECTED_CUT_A,
                silence_found=True,
            ),
            "B": CutPoint(
                anchor_id="B",
                raw_cut_time=fixtures.EXPECTED_ANCHOR_B_RAW,
                cut_time=fixtures.EXPECTED_CUT_B,
                silence_found=True,
            ),
        },
    )

    with pytest.raises(MissingArtifactError) as excinfo:
        run_pipeline(ctx, from_step=7)

    message = str(excinfo.value)
    assert s5.HIGHLIGHT_FILE in message
    assert f"--from-step {s5.STEP}" in message
    # ここまでで分かっている cut_time は decisions.json に残っている
    assert read_decisions(ctx)["anchors"]["A"]["cut_time"] == pytest.approx(
        fixtures.EXPECTED_CUT_A, abs=0.01
    )


def test_preview_only_without_cuts_raises_missing_artifact(ctx: RunContext) -> None:
    """--preview-only もカット点が無ければ止まる。プレビューは cuts.json が前提。"""
    ctx.ensure_dirs()

    with pytest.raises(MissingArtifactError) as excinfo:
        run_pipeline(ctx, preview_only=True)

    assert f"--only-step {s4.STEP}" in str(excinfo.value)


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.skipif(HAS_ASR_BACKEND, reason="ASR バックエンドがある環境では Step 2 は落ちない")
def test_from_step_1_fails_loudly_when_no_asr_backend(video_ctx: RunContext) -> None:
    """ASR バックエンドが無い環境では Step 2 が例外で止まる（空の文字起こしを作らない）。

    Step 1 は成功して audio.wav / probe.json を残す。Step 2 だけが落ちる。
    """
    with pytest.raises(RadioCutterError) as excinfo:
        run_pipeline(video_ctx, from_step=1)

    assert s1.audio_path(video_ctx).exists()
    assert s1.probe_path(video_ctx).exists()
    assert not video_ctx.work_path(s2.TRANSCRIPT_FILENAME).exists()
    assert "whisper" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# --only-step
# ---------------------------------------------------------------------------


def test_only_step_3_runs_just_step_3(ctx: RunContext) -> None:
    """--only-step 3 はアンカー検出だけ。Step 4 のカット精密化には進まない。"""
    seed_transcript(ctx)

    result = run_pipeline(ctx, only_step=3)

    assert result.executed_steps == [3]
    assert set(result.timings) == {3}
    assert ctx.work_path(s3.ANCHORS_FILE).exists()
    assert not s4.cuts_path(ctx).exists()
    assert result.cuts is None
    assert result.highlight is None
    assert result.render is None


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_only_step_4_runs_just_step_4_and_reuses_anchors(ctx: RunContext) -> None:
    """--only-step 4 は Step 3 をやり直さない。

    transcript.json をわざと消してから流す。Step 3 が再実行されるなら
    文字起こしが無くて落ちるはずなので、通ったこと自体が「読み直していない」証拠になる。
    """
    seed_anchors(ctx)
    fixtures.write_tone_wav(s1.audio_path(ctx))
    ctx.work_path(s2.TRANSCRIPT_FILENAME).unlink()

    result = run_pipeline(ctx, only_step=4)

    assert result.executed_steps == [4]
    assert set(result.timings) == {4}
    assert result.transcript is None
    assert result.cuts["A"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.01)
    assert result.cuts["B"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.01)
    assert result.cuts["A"].silence_found is True
    assert s4.cuts_path(ctx).exists()
    # Step 5 以降には進んでいない
    assert not s5.highlight_path(ctx).exists()
    assert not s6.description_path(ctx).exists()


def test_llm_client_is_not_built_before_step_5(
    ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 1〜4 は APIキー無しで回せる。LLM クライアントは Step 5 に入るまで作らない。

    SPEC 5章の llm 設定は Step 5・6 のためのもので、分割（Phase 1）には要らない。
    """
    import radio_cutter.pipeline as pipeline_module

    def boom(*args, **kwargs):  # pragma: no cover - 呼ばれたら即失敗
        raise AssertionError("Step 5 に入る前に LLM クライアントを作ってはいけません。")

    monkeypatch.setattr(pipeline_module, "build_client", boom)
    monkeypatch.delenv(ctx.config.llm.api_key_env, raising=False)
    seed_transcript(ctx)

    result = run_pipeline(ctx, only_step=3)

    assert result.executed_steps == [3]


# ---------------------------------------------------------------------------
# --dry-run（SPEC 7章の既定の運用フロー）
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_dry_run_skips_render_and_preview_but_writes_description(
    ready_ctx: RunContext, stub_llm
) -> None:
    """--dry-run は Step 7・8 を行わない。それでも description.txt は出る（SPEC 7章）。"""
    result = run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)

    assert result.executed_steps == [3, 4, 5, 6]
    assert 7 not in result.timings and 8 not in result.timings
    assert ready_ctx.dry_run is True

    # Step 6 の成果物は出る
    description = s6.description_path(ready_ctx)
    assert description.exists()
    assert description.read_text(encoding="utf-8").strip()
    assert s6.titles_path(ready_ctx).exists()

    # Step 7・8 の成果物は出ない
    assert out_exists(ready_ctx, *RENDER_FILES) == [False] * len(RENDER_FILES)
    assert not s7.final_path(ready_ctx).exists()
    assert not ready_ctx.out_path("preview").exists()
    assert result.render is None
    assert result.previews == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_dry_run_writes_decisions_json(ready_ctx: RunContext, stub_llm) -> None:
    """--dry-run でも decisions.json は書く。

    SPEC 7章の運用フローは「先に decisions.json とプレビューでカット点を確認し、
    問題なければ --from-step 7 で書き出す」。ここで書かれないとフローが成立しない。
    """
    run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)

    path = decisions_path(ready_ctx)
    assert path.exists()

    payload = read_decisions(ready_ctx)
    assert payload["episode_id"] == "ep-test"
    assert payload["input"] == str(ready_ctx.input_path.resolve())
    assert payload["generated_at"]

    # SPEC 8章: anchors には raw_cut_time / cut_time / silence_found / score が入る
    anchors = payload["anchors"]
    assert set(anchors) == {"A", "B"}
    assert anchors["A"]["raw_cut_time"] == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW, abs=0.01)
    assert anchors["A"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.01)
    assert anchors["B"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.01)
    assert anchors["A"]["silence_found"] is True

    # 書き出していないので尺は想定値
    assert payload["durations_source"] == "estimated"
    assert payload["durations"]["main"] == pytest.approx(
        fixtures.EXPECTED_CUT_B - fixtures.EXPECTED_CUT_A, abs=0.05
    )

    # ハイライトは採用1件＋次点（SPEC Step 5「候補を3つ返させ、残り2つは decisions.json に残す」）
    assert payload["highlight"]["selected"]["start"] < payload["highlight"]["selected"]["end"]
    assert len(payload["highlight"]["alternatives"]) == 2
    assert "snapped_from" in payload["highlight"]

    assert isinstance(payload["warnings"], list)
    assert [call["step"] for call in payload["llm_calls"]] == ["highlight", "metadata", "titles"]


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_metadata_failure_does_not_stop_the_render(ready_ctx: RunContext) -> None:
    """Step 6 が落ちても動画の書き出しは続ける（SPEC 9章）。

    「LLMがJSON以外を返す → 3回までリトライ。失敗したらそのステップだけ落とし、
    動画の書き出しは続行する（description.txt が無くても動画は出す）。」
    ここではハイライト用の応答だけを持つスタブを渡し、Step 6 の2回の呼び出しを失敗させる。
    """
    llm = StubLlmClient({"highlight": fixtures.stub_highlight_response()}, model="stub-model")

    result = run_pipeline(ready_ctx, from_step=3, llm=llm)

    assert result.executed_steps == [3, 4, 5, 6, 7, 8]
    # description.txt / titles.md は作れなかった
    assert not s6.description_path(ready_ctx).exists()
    assert not s6.titles_path(ready_ctx).exists()
    # それでも動画は出る
    assert out_exists(ready_ctx, *RENDER_FILES) == [True] * len(RENDER_FILES)
    assert sorted(p.name for p in ready_ctx.out_path("preview").glob("*.mp4")) == sorted(
        PREVIEW_FILES
    )

    # 何が落ちたかは decisions.json に残す（黙って無かったことにしない）
    payload = read_decisions(ready_ctx)
    failed = [call for call in payload["llm_calls"] if not call["ok"]]
    assert {call["step"] for call in failed} == {"metadata", "titles"}
    assert any("description.txt" in warning for warning in payload["warnings"])


# ---------------------------------------------------------------------------
# --preview-only（SPEC 7章 / Step 8）
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_preview_only_runs_step_8_alone(ready_ctx: RunContext, stub_llm) -> None:
    """--preview-only は Step 8 だけ実行する。書き出し（Step 7）はしない。"""
    run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)

    result = run_pipeline(ready_ctx, preview_only=True)

    assert result.executed_steps == [PREVIEW_STEP]
    assert set(result.timings) == {PREVIEW_STEP}

    preview_dir = ready_ctx.out_path("preview")
    assert sorted(p.name for p in preview_dir.glob("*.mp4")) == sorted(PREVIEW_FILES)
    assert sorted(p.name for p in result.previews) == sorted(PREVIEW_FILES)

    # Step 7 には触っていない
    assert out_exists(ready_ctx, *RENDER_FILES) == [False] * len(RENDER_FILES)
    assert result.render is None


# ---------------------------------------------------------------------------
# timings（SPEC 11章「各ステップの開始・終了・所要秒数を必ず出す」）
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_timings_cover_every_executed_step(ready_ctx: RunContext, stub_llm) -> None:
    """実行したステップの所要秒数が全部 timings に入る（実行していないステップは入らない）。"""
    result = run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)

    assert set(result.timings) == set(result.executed_steps)
    assert all(isinstance(step, int) for step in result.timings)
    for step, elapsed in result.timings.items():
        assert isinstance(elapsed, float), f"Step {step} の所要秒数が float ではありません"
        assert elapsed >= 0.0
    assert result.total_elapsed == pytest.approx(sum(result.timings.values()))


# ---------------------------------------------------------------------------
# 途中で落ちても decisions.json は残す
# ---------------------------------------------------------------------------


def test_decisions_written_when_step_4_fails(
    ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 4 が落ちても、そこまでに分かっていること（Step 3 のアンカー）は残す。

    decisions.json は「あとから何が起きたか追えるようにする」ためのもの（SPEC 8章）。
    落ちたときこそ要る。
    """
    seed_anchors(ctx)

    def boom(*args, **kwargs):
        raise RadioCutterError("わざと落とす（Step 4）")

    monkeypatch.setattr(s4, "run", boom)

    with pytest.raises(RadioCutterError, match="わざと落とす"):
        run_pipeline(ctx, from_step=4)

    payload = read_decisions(ctx)
    assert payload["anchors"]["A"]["raw_cut_time"] == pytest.approx(
        fixtures.EXPECTED_ANCHOR_A_RAW, abs=0.01
    )
    assert payload["anchors"]["B"]["raw_cut_time"] == pytest.approx(
        fixtures.EXPECTED_ANCHOR_B_RAW, abs=0.01
    )
    # Step 4 まで行っていないので cut_time はまだ無い（0 と「取れなかった」を取り違えない）
    assert "cut_time" not in payload["anchors"]["A"]


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_decisions_written_when_render_fails(
    ready_ctx: RunContext, stub_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 7（書き出し）が落ちても、カット点とハイライトの判断ログは残る。"""
    run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)
    decisions_path(ready_ctx).unlink()

    def boom(*args, **kwargs):
        raise RenderError("わざと落とす（Step 7）")

    monkeypatch.setattr(s7, "run", boom)

    with pytest.raises(RenderError, match="わざと落とす"):
        run_pipeline(ready_ctx, from_step=7, llm=stub_llm)

    payload = read_decisions(ready_ctx)
    assert payload["anchors"]["A"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.01)
    assert payload["anchors"]["B"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.01)
    assert "selected" in payload["highlight"]
    assert payload["durations_source"] == "estimated"  # 書き出せていないので実測値ではない


# ---------------------------------------------------------------------------
# --from-step の総当たり（どこからでも再開できること）
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("from_step", [3, 4, 5, 6, 7, 8])
def test_resume_from_any_step_when_artifacts_are_present(
    from_step: int,
    seeded_work_dir: Path,
    tmp_path: Path,
    config: Config,
    episode_video: Path,
    stub_llm,
) -> None:
    """前段の中間ファイルが揃っていれば、Step 3〜8 のどこからでも再開できる。

    SPEC 3章「各ステップは中間ファイルを work/<episode_id>/ に残し、
    --from-step N で途中から再実行できること」。
    """
    ctx = ctx_from_seed(seeded_work_dir, tmp_path, config, episode_video)
    before = sorted(p.name for p in ctx.work_dir.iterdir())

    llm = stub_llm if from_step <= 6 else None
    result = run_pipeline(ctx, from_step=from_step, llm=llm)

    assert result.executed_steps == list(range(from_step, LAST_STEP + 1))
    assert set(result.timings) == set(range(from_step, LAST_STEP + 1))
    assert decisions_path(ctx).exists()

    # Step 8 は必ず計画に入るのでプレビューは常に出る
    assert sorted(p.name for p in ctx.out_path("preview").glob("*.mp4")) == sorted(PREVIEW_FILES)

    if from_step <= 7:
        assert out_exists(ctx, *RENDER_FILES) == [True] * len(RENDER_FILES)
        assert result.render is not None
    else:
        assert not s7.final_path(ctx).exists()

    # SPEC 11章「中間ファイルは消さない」。再開しても前段の成果物は残っている。
    after = {p.name for p in ctx.work_dir.iterdir()}
    assert set(before) <= after


# ---------------------------------------------------------------------------
# SPEC 7章に書かれた運用フローそのもの
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_documented_workflow_dry_run_then_preview_then_render(
    ready_ctx: RunContext, stub_llm
) -> None:
    """「--dry-run → プレビューで確認 → --from-step 7 で書き出し」が通しで回る（SPEC 7章）。

    最後の書き出しは LLM を使わない。ここで APIキーを要求されたら運用が止まる。
    """
    # 1) --dry-run: 判断ログと概要欄まで
    run_pipeline(ready_ctx, from_step=3, dry_run=True, llm=stub_llm)
    assert decisions_path(ready_ctx).exists()
    assert not s7.final_path(ready_ctx).exists()

    # 2) --preview-only: カット点の確認用プレビュー
    run_pipeline(ready_ctx, preview_only=True)
    assert ready_ctx.out_path("preview", "cut_A.mp4").exists()
    assert ready_ctx.out_path("preview", "cut_B.mp4").exists()
    assert not s7.final_path(ready_ctx).exists()

    # 3) --from-step 7: 書き出し（llm は渡さない）
    result = run_pipeline(ready_ctx, from_step=7, llm=None)

    assert result.executed_steps == [7, 8]
    assert out_exists(ready_ctx, *RENDER_FILES) == [True] * len(RENDER_FILES)
    # 先に作った description.txt は消されていない
    assert s6.description_path(ready_ctx).exists()

    payload = read_decisions(ready_ctx)
    assert payload["durations_source"] == "measured"
    durations = payload["durations"]
    expected_final = durations["highlight"] + durations["main"] + durations["ending"]
    # SPEC Step 7「final.mp4 の実尺を検算し、差が0.5秒を超えたら警告を出す」
    assert durations["final"] == pytest.approx(expected_final, abs=0.5)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_result_lists_the_files_it_produced(ready_ctx: RunContext, stub_llm) -> None:
    """PipelineResult.artifacts に、今 work/ と out/ にあるものが並ぶ（デバッグの起点）。"""
    result = run_pipeline(ready_ctx, from_step=3, llm=stub_llm)

    names = {Path(p).name for p in result.artifacts}
    assert {
        s2.TRANSCRIPT_FILENAME,
        s3.ANCHORS_FILE,
        s4.CUTS_FILE,
        s5.HIGHLIGHT_FILE,
        s6.METADATA_FILE,
        "final.mp4",
        "description.txt",
        "titles.md",
        "decisions.json",
    } <= names
    assert all(Path(p).exists() for p in result.artifacts)


# ---------------------------------------------------------------------------
# decisions.json の尺の出どころ
# ---------------------------------------------------------------------------


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_step7が落ちた回は前回の実尺を実測として載せない(
    video_ctx: RunContext, stub_llm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """書き出しに失敗した回の decisions.json に、前回の実尺が「実測」として残らないこと。

    カット点を変えて流し直した直後だと、前回の実尺は今回の anchors と食い違う。
    「今回の書き出しの結果です」という顔で古い数字が載るのがいちばん困る。
    """
    fixtures.build_transcript().save(video_ctx.work_path("transcript.json"))
    s1.run(video_ctx)

    # 1回目: 通しで成功させ、render.json を残す
    ok = run_pipeline(video_ctx, from_step=3, llm=stub_llm)
    assert ok.render is not None
    first = json.loads(video_ctx.out_path("decisions.json").read_text(encoding="utf-8"))
    assert first["durations_source"] == "measured"

    # 2回目: Step 7 だけ落とす
    def boom(*args, **kwargs):
        raise RenderError("書き出しに失敗しました（試験用）")

    monkeypatch.setattr(s7, "run", boom)
    with pytest.raises(RenderError):
        run_pipeline(video_ctx, from_step=7, llm=stub_llm)

    after = json.loads(video_ctx.out_path("decisions.json").read_text(encoding="utf-8"))
    assert after["durations_source"] != "measured", (
        "Step 7 が落ちた回なのに、前回の実尺が「実測」として載っている"
    )
