"""SPEC 7章「CLI仕様」と SPEC 8章「decisions.json スキーマ」のテスト。

ここで守らせたいこと:

- CLI は「人が打ち間違えたとき」に親切に落ちる。使い方の誤りは終了コード2、
  想定内の失敗（入力が無い・中間ファイルが無い）はトレースバック無しの終了コード1。
- SPEC 7章に書かれたオプションが、そのまま実行時の設定に効く
  （`--silence-db` / `--silence-dur` が SilenceConfig に、`--episode-id` が置き場に）。
- `run --from-step 3 --stub-llm` を通すと SPEC 1章の成果物が**全部**揃う。
  ここが崩れると「ワンコマンドで揃う」というこのツールの存在理由が消える。
- SPEC 7章が既定の運用フローとして挙げる
  「--dry-run で確認 → --preview-only で目視 → --from-step 7 で書き出し」が回る。
- decisions.json は SPEC 8章のキーを持ち、セグメント名を "main"/"ending" と
  決め打ちせず config から取る（SPEC 5章「コードにハードコードしないこと」）。

テストは実装ではなく SPEC と各モジュールの docstring に書かれた契約を根拠に書く。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg

from radio_cutter import cli
from radio_cutter.cli import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, main
from radio_cutter.config import Config, SilenceConfig
from radio_cutter.context import RunContext
from radio_cutter.decisions import (
    DECISIONS_FILE,
    SOURCE_ESTIMATED,
    SOURCE_MEASURED,
    build_decisions,
    decisions_path,
    now_iso,
    write_decisions,
)
from radio_cutter.models import (
    AnchorResult,
    CutPoint,
    HighlightCandidate,
    HighlightResult,
    LlmCallRecord,
    RenderResult,
)
from radio_cutter.pipeline import FIRST_STEP, LAST_STEP, PipelineResult
from radio_cutter.util.ffmpeg import FFMPEG_ENV, FFPROBE_ENV, MediaInfo, has_encoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ai-radio.json"

#: SPEC 1章「ゴール」の成果物一覧（preview/ 以下は別途）
SPEC_OUTPUTS: tuple[str, ...] = (
    "01_highlight.mp4",
    "02_main.mp4",
    "03_ending.mp4",
    "final.mp4",
    "description.txt",
    "titles.md",
    "decisions.json",
)

#: SPEC Step 8 のプレビュー
SPEC_PREVIEWS: tuple[str, ...] = (
    "cut_A.mp4",
    "cut_B.mp4",
    "highlight_in.mp4",
    "highlight_out.mp4",
)


# ---------------------------------------------------------------------------
# 足場
# ---------------------------------------------------------------------------


def _reset_logging() -> None:
    """radio_cutter のログハンドラを外す。

    setup_logging() はハンドラを1度だけ作り、そのとき掴んだ sys.stderr を握り続ける。
    pytest のキャプチャと組み合わせると古いストリームに書き続けてしまうので、
    テストごとに作り直させる。
    """
    logger = logging.getLogger("radio_cutter")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _fresh_logging():
    _reset_logging()
    yield
    _reset_logging()


class CliEnv:
    """CLI を叩くための置き場一式。work / out は必ず tmp に閉じ込める。"""

    def __init__(self, base: Path, episode_id: str = "ep-cli") -> None:
        self.base = base
        self.episode_id = episode_id
        self.input = base / f"{episode_id}.mp4"
        self.work = base / "work"
        self.out = base / "out"
        self.stub = base / "stub-llm.json"
        #: 通しテスト用に、実行結果を持ち回るための置き場
        self.code: int | None = None
        self.step1_code: int | None = None
        self.stdout: str = ""
        self.stub.write_text(
            json.dumps(fixtures.stub_responses(), ensure_ascii=False), encoding="utf-8"
        )

    @property
    def work_dir(self) -> Path:
        return self.work / self.episode_id

    @property
    def out_dir(self) -> Path:
        return self.out / self.episode_id

    def common(self, *extra: str) -> list[str]:
        return [
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(self.work),
            "--out",
            str(self.out),
            *extra,
        ]


@pytest.fixture
def env(tmp_path: Path) -> CliEnv:
    """入力ファイルは作らない（存在しないときの挙動もここから試せるように）。"""
    return CliEnv(tmp_path)


def _capture_pipeline(monkeypatch) -> dict:
    """run_pipeline を差し替えて、CLI が組み立てた RunContext と引数を捕まえる。"""
    captured: dict = {}

    def fake_run_pipeline(ctx: RunContext, **kwargs):
        captured["ctx"] = ctx
        captured["kwargs"] = kwargs
        return PipelineResult(ctx=ctx)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    return captured


def _no_traceback(err: str) -> None:
    """SPEC 9章。想定内の失敗でトレースバックを見せない（人が読む出力を汚さない）。"""
    assert "Traceback" not in err, f"想定内の失敗なのにトレースバックが出ている:\n{err}"


def _prepare_transcript(env: CliEnv) -> None:
    """この環境には ASR バックエンドが無いので、Step 2 の成果物だけ先に置く。"""
    env.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures.build_transcript().save(env.work_dir / "transcript.json")


# ===========================================================================
# doctor（SPEC 2章 / 7章）
# ===========================================================================


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_doctor_returns_zero_and_reports_core_dependencies(capsys):
    """`doctor` は環境が揃っていれば0で終わり、ffmpeg と必須ライブラリの行を出す。

    SPEC 2章「初回に環境チェックコマンドを用意すること」の最低限の中身。
    """
    code = main(["doctor"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    for needle in ("ffmpeg", "ffprobe", "rapidfuzz", "jsonschema"):
        assert needle in out, f"doctor の出力に {needle} の行がない:\n{out}"


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_doctor_checks_videotoolbox_and_announces_cpu_fallback(capsys):
    """SPEC 2章。VideoToolbox の有無を確認し、無ければ CPU へ落ちる旨を警告する。"""
    main(["doctor"])
    out = capsys.readouterr().out

    assert "h264_videotoolbox" in out, f"ハードウェアエンコーダの確認結果が出ていない:\n{out}"
    if not has_encoder("h264_videotoolbox"):
        assert "フォールバック" in out
        assert "libx264" in out, "CPU エンコードの落とし先が示されていない"


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_doctor_looks_at_claude_code_not_the_api_key(monkeypatch, capsys):
    """既定は APIキーではなく、このパソコンの Claude Code を見ること。

    キーが無くても、Claude Code さえ入っていれば Step 5・6 は動く。
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/local/bin/claude" if name == "claude" else "/usr/bin/" + name)

    code = main(["doctor"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Claude Code" in out
    assert "APIキーは要りません" in out


def test_doctor_warns_when_claude_code_is_missing(monkeypatch, capsys):
    """Claude Code が無いのは「警告」であって「NG」ではない（Step 1〜4 と --stub-llm は動く）。"""
    real_which = cli.shutil.which
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: None if name == "claude" else real_which(name)
    )

    code = main(["doctor"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "claude" in out.lower()
    assert "ログイン" in out


def test_doctor_still_checks_the_api_key_for_the_anthropic_provider(
    tmp_path: Path, monkeypatch, capsys
):
    """`llm.provider` を anthropic にしたときだけ、環境変数を見に行くこと。"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["llm"]["provider"] = "anthropic"
    raw["llm"]["model"] = "claude-opus-5"
    config = tmp_path / "api.json"
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    code = main(["doctor", "--config", str(config)])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "ANTHROPIC_API_KEY" in out


def test_default_config_falls_back_to_the_bundled_one(tmp_path: Path, monkeypatch, capsys):
    """どこから起動しても `doctor` が設定を読めること。

    SPEC 7章の既定は `config/ai-radio.json`。カレントに無ければ同梱のものを見に行く。
    ここが効かないと「よその作業ディレクトリから doctor を打つと設定不明」になる。
    """
    monkeypatch.chdir(tmp_path)

    main(["doctor"])
    out = capsys.readouterr().out

    assert "AI活用法実験ラジオ" in out, f"同梱の設定を読めていない:\n{out}"


def test_config_in_the_current_directory_wins(tmp_path: Path, monkeypatch, capsys):
    """カレント直下の `config/ai-radio.json` があればそちらを優先する。"""
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "config" / "ai-radio.json"
    local.parent.mkdir(parents=True)
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["channel"] = "別チャンネル実験放送"
    local.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    main(["doctor"])
    out = capsys.readouterr().out

    assert "別チャンネル実験放送" in out
    assert "AI活用法実験ラジオ" not in out


def test_doctor_fails_when_ffmpeg_is_missing(monkeypatch, capsys):
    """ffmpeg が無ければ何も動かないので NG（終了コード1）にする。

    SPEC 2章「`ffmpeg` と `ffprobe` がPATH上にあること」。
    """
    monkeypatch.setenv(FFMPEG_ENV, "radio-cutter-no-such-ffmpeg")
    monkeypatch.setenv(FFPROBE_ENV, "radio-cutter-no-such-ffprobe")

    code = main(["doctor"])
    captured = capsys.readouterr()

    assert code == EXIT_ERROR
    assert "radio-cutter-no-such-ffmpeg" in captured.out
    assert "radio-cutter-no-such-ffprobe" in captured.out
    _no_traceback(captured.err)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_doctor_survives_a_broken_config(tmp_path: Path, capsys):
    """設定ファイルが壊れていても環境チェック自体は最後まで走る。

    doctor は「なぜ動かないか」を突き止めるためのコマンドなので、
    途中で投げ出すと肝心の ffmpeg の状態が分からなくなる。
    """
    broken = tmp_path / "broken.json"
    broken.write_text("{ これは JSON ではない", encoding="utf-8")

    code = main(["doctor", "--config", str(broken)])
    out = capsys.readouterr().out

    assert code == EXIT_OK, "設定ファイルの不備は警告であって環境の NG ではない"
    assert "設定ファイル" in out
    assert "ffmpeg" in out


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_doctor_reports_missing_asr_backend(capsys):
    """文字起こしバックエンドが1つも無いなら、その旨と入れ方を出す（SPEC 6章 Step2）。"""
    import importlib.util

    installed = [
        name
        for name in ("whisperx", "mlx_whisper")
        if importlib.util.find_spec(name) is not None
    ]
    main(["doctor"])
    out = capsys.readouterr().out

    if installed:
        pytest.skip(f"文字起こしバックエンドが入っている環境です: {installed}")
    assert "mlx-whisper" in out or "mlx_whisper" in out
    assert "whisperx" in out


# ===========================================================================
# run — 引数の検証（使い方の誤りは終了コード2）
# ===========================================================================


def test_no_command_is_a_usage_error():
    """サブコマンド無しは使い方の誤り。"""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_USAGE


def test_run_requires_an_input_file():
    """`run` は入力ファイルが必須（SPEC 7章 `radio-cutter run <input.mp4>`）。"""
    with pytest.raises(SystemExit) as exc:
        main(["run"])
    assert exc.value.code == EXIT_USAGE


def test_from_step_and_only_step_cannot_be_combined(env: CliEnv, capsys):
    """`--from-step` と `--only-step` は意味が矛盾するので、走り出す前に止める。"""
    env.input.write_bytes(b"")

    with pytest.raises(SystemExit) as exc:
        main(["run", str(env.input), "--from-step", "3", "--only-step", "5", *env.common()])

    assert exc.value.code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--from-step" in err and "--only-step" in err


@pytest.mark.parametrize(
    "option,value",
    [
        ("--from-step", "0"),
        ("--from-step", "9"),
        ("--from-step", "-1"),
        ("--only-step", "0"),
        ("--only-step", "9"),
        ("--from-step", "さん"),
        ("--only-step", "3.5"),
    ],
)
def test_step_number_outside_1_to_8_is_rejected(env: CliEnv, option: str, value: str):
    """ステップ番号は SPEC 3章の 1〜8 だけ。範囲外・非整数は走り出す前に弾く。"""
    env.input.write_bytes(b"")

    with pytest.raises(SystemExit) as exc:
        main(["run", str(env.input), option, value, *env.common()])

    assert exc.value.code == EXIT_USAGE
    assert (FIRST_STEP, LAST_STEP) == (1, 8), "SPEC 3章のステップ数が変わったらこのテストも直すこと"


@pytest.mark.parametrize("value", ["0", "-0.5", "ぜろ"])
def test_silence_duration_must_be_positive(env: CliEnv, value: str):
    """`--silence-dur` は silencedetect の `d=` に入る。0以下は意味を成さない。"""
    env.input.write_bytes(b"")

    with pytest.raises(SystemExit) as exc:
        main(["run", str(env.input), "--silence-dur", value, *env.common()])

    assert exc.value.code == EXIT_USAGE


def test_version_flag_prints_the_version():
    """`--version` はバージョンだけ出して正常終了する。"""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == EXIT_OK


def test_blank_episode_id_is_rejected(env: CliEnv, capsys):
    """空白だけの `--episode-id` は置き場が決まらないので、走り出す前に止める。"""
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--episode-id", "   ", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "--episode-id" in captured.err
    _no_traceback(captured.err)


def test_dry_run_with_from_step_7_says_there_is_nothing_to_do(env: CliEnv, capsys):
    """`--dry-run` は Step 6 まで。Step 7 以降からの再開とは両立しない。

    黙って何もせず0で終わると「書き出したつもり」になるので、必ず理由を言って止める。
    """
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--dry-run", "--from-step", "7", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "--dry-run" in captured.err
    _no_traceback(captured.err)


def test_dry_run_and_preview_only_cannot_be_combined(env: CliEnv, capsys):
    """`--preview-only` は Step 8（書き出し）を行うので `--dry-run` と矛盾する。"""
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--dry-run", "--preview-only", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "--dry-run" in captured.err and "--preview-only" in captured.err
    _no_traceback(captured.err)


# ===========================================================================
# run — オプションが実行時の設定に届くか（SPEC 7章）
# ===========================================================================


@pytest.mark.parametrize(
    "argv",
    [
        ["--silence-db=-30", "--silence-dur", "0.2"],
        ["--silence-db", "-30", "--silence-dur=0.2"],
    ],
    ids=["equals-form", "space-form"],
)
def test_silence_options_reach_the_run_context(env: CliEnv, monkeypatch, argv: list[str]):
    """`--silence-db` / `--silence-dur` が SilenceConfig に反映される（SPEC Step 4）。

    「収録環境のノイズフロアによって最適値が変わる」ためのオプションなので、
    渡した値がそのまま無音検出に届かないと存在意義が無い。
    """
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    code = main(["run", str(env.input), *argv, *env.common()])

    assert code == EXIT_OK
    silence = captured["ctx"].silence
    assert silence.noise_db == pytest.approx(-30.0)
    assert silence.min_duration_sec == pytest.approx(0.2)


def test_silence_defaults_come_from_the_config_file(env: CliEnv, monkeypatch, config: Config):
    """オプションを渡さなければ設定ファイル（既定 -32dB / 0.12秒）のまま。"""
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), *env.common()])

    silence = captured["ctx"].silence
    assert silence.noise_db == pytest.approx(config.silence.noise_db)
    assert silence.min_duration_sec == pytest.approx(config.silence.min_duration_sec)
    assert silence == SilenceConfig(noise_db=-32.0, min_duration_sec=0.12)


def test_partial_silence_override_keeps_the_other_default(
    env: CliEnv, monkeypatch, config: Config
):
    """片方だけ上書きしたとき、もう片方は設定ファイルの値のまま残る。"""
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), "--silence-db=-45", *env.common()])

    silence = captured["ctx"].silence
    assert silence.noise_db == pytest.approx(-45.0)
    assert silence.min_duration_sec == pytest.approx(config.silence.min_duration_sec)


def test_episode_id_defaults_to_the_input_stem(tmp_path: Path, monkeypatch):
    """`--episode-id` の既定は入力ファイル名の stem（SPEC 7章）。

    置き場が work/<episode_id> と out/<episode_id> になることも同時に確かめる（SPEC 3章・4章）。
    """
    input_path = tmp_path / "ep42.mp4"
    input_path.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(input_path), "--config", str(CONFIG_PATH)])

    ctx = captured["ctx"]
    assert ctx.episode_id == "ep42"
    assert ctx.work_dir == Path("work") / "ep42"
    assert ctx.out_dir == Path("out") / "ep42"


def test_episode_id_can_be_given_explicitly(env: CliEnv, monkeypatch):
    """`--episode-id` を明示したら入力ファイル名より優先される。"""
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), "--episode-id", "ep-2026-08-30", *env.common()])

    ctx = captured["ctx"]
    assert ctx.episode_id == "ep-2026-08-30"
    assert ctx.work_dir == env.work / "ep-2026-08-30"
    assert ctx.out_dir == env.out / "ep-2026-08-30"


def test_episode_id_keeps_dots_in_the_stem(tmp_path: Path, monkeypatch):
    """`2026-08-30.収録.mp4` のような名前でも「最後の拡張子だけ」を落とす。"""
    input_path = tmp_path / "2026-08-30.収録.mp4"
    input_path.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(input_path), "--config", str(CONFIG_PATH)])

    assert captured["ctx"].episode_id == "2026-08-30.収録"


@pytest.mark.parametrize(
    "argv,expected",
    [
        ([], {"from_step": FIRST_STEP, "only_step": None, "dry_run": False, "preview_only": False}),
        (["--from-step", "4"], {"from_step": 4, "only_step": None}),
        (["--only-step", "7"], {"from_step": FIRST_STEP, "only_step": 7}),
        (["--dry-run"], {"dry_run": True, "preview_only": False}),
        (["--preview-only"], {"dry_run": False, "preview_only": True}),
    ],
)
def test_step_options_are_forwarded_to_the_pipeline(
    env: CliEnv, monkeypatch, argv: list[str], expected: dict
):
    """SPEC 7章のオプションがそのままパイプラインの実行計画に渡る。"""
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), *argv, *env.common()])

    kwargs = captured["kwargs"]
    for key, value in expected.items():
        assert kwargs[key] == value, f"{key} が渡っていない（{kwargs}）"


def test_dry_run_flag_is_recorded_on_the_context(env: CliEnv, monkeypatch):
    """`--dry-run` は RunContext にも残る（後段が「書き出さない」と判断できるように）。"""
    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), "--dry-run", *env.common()])

    assert captured["ctx"].dry_run is True


def test_stub_llm_replaces_the_real_client(env: CliEnv, monkeypatch):
    """`--stub-llm` を渡すと LLM クライアントが差し替わり、APIキー無しで通せる。"""
    from radio_cutter.llm.client import StubLlmClient

    env.input.write_bytes(b"")
    captured = _capture_pipeline(monkeypatch)

    main(["run", str(env.input), "--stub-llm", str(env.stub), *env.common()])

    llm = captured["kwargs"]["llm"]
    assert isinstance(llm, StubLlmClient)
    assert set(llm.responses) >= {"highlight", "metadata", "titles"}


def test_missing_stub_llm_file_is_reported_without_a_traceback(env: CliEnv, capsys):
    """スタブ応答ファイルが無いのは打ち間違いなので、素直に読める形で落ちる。"""
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--stub-llm", str(env.base / "nope.json"), *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "nope.json" in captured.err
    _no_traceback(captured.err)


# ===========================================================================
# run — エラー処理（SPEC 9章）
# ===========================================================================


def test_run_with_a_missing_input_returns_1_without_a_traceback(env: CliEnv, capsys):
    """存在しない入力は「想定内の失敗」。終了コード1で、トレースバックは出さない。"""
    missing = env.base / "not-here.mp4"

    code = main(["run", str(missing), *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert str(missing) in captured.err
    _no_traceback(captured.err)


def test_run_with_a_directory_as_input_returns_1(env: CliEnv, capsys):
    """ディレクトリを渡したときも同じ扱い（ffmpeg に投げる前に気づく）。"""
    directory = env.base / "episodes"
    directory.mkdir()

    code = main(["run", str(directory), *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    _no_traceback(captured.err)


def test_from_step_without_intermediate_files_explains_what_to_run_first(
    env: CliEnv, capsys
):
    """中間ファイルが無いまま `--from-step` すると、何を先に流せばよいかを教えて止まる。

    SPEC 3章「`--from-step N` で途中から再実行できること」の裏側。
    勝手に前段をやり直さない（文字起こしは8割の時間を食うため）。
    """
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--from-step", "3", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "transcript.json" in captured.err
    assert "radio-cutter" in captured.err, "次に打つコマンドが案内されていない"
    _no_traceback(captured.err)


def test_decisions_json_is_written_even_when_the_step_fails(env: CliEnv, capsys):
    """Step 4 以降に踏み込んだら、失敗しても decisions.json は残す。

    SPEC 8章「あとから何が起きたか追えるようにする」。
    ここが書かれないと、落ちた実行だけ記録が消えることになる。
    """
    env.input.write_bytes(b"")

    code = main(["run", str(env.input), "--from-step", "7", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    _no_traceback(captured.err)

    written = env.out_dir / DECISIONS_FILE
    assert written.is_file(), "Step 7 に踏み込んだのに decisions.json が残っていない"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["episode_id"] == env.episode_id
    assert "warnings" in payload and "llm_calls" in payload


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_ffmpeg_failure_is_surfaced_not_swallowed(env: CliEnv, capsys):
    """SPEC 9章「ffmpegが非ゼロ終了 → stderrをそのまま表示して停止。握りつぶさない」。"""
    env.input.write_text("これは動画ファイルではありません\n" * 100, encoding="utf-8")

    code = main(["run", str(env.input), "--only-step", "1", *env.common()])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "ffprobe" in captured.err or "ffmpeg" in captured.err
    assert "ffmpeg stderr" in captured.err, "ffmpeg 自身の出力が握りつぶされている"
    _no_traceback(captured.err)


def test_anchor_not_found_stops_instead_of_guessing(env: CliEnv, capsys):
    """SPEC 9章「アンカー未検出 → 停止。候補スコア上位3件と文脈を表示。自動的に代替を選ばない」。"""
    env.input.write_bytes(b"")
    _prepare_transcript(env)

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["anchors"][0]["phrase"] = "この番組はお聞きの皆様の提供でお送りします"
    broken_config = env.base / "unmatched-anchor.json"
    broken_config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    code = main(
        [
            "run",
            str(env.input),
            "--from-step",
            "3",
            "--config",
            str(broken_config),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "しきい値" in captured.err, "しきい値を下げる案内が無い"
    assert "スコア" in captured.err, "候補のスコアと文脈が出ていない"
    _no_traceback(captured.err)
    assert not (env.work_dir / "anchors.json").exists(), "見つからなかったのに結果を書いている"


def test_unexpected_exception_shows_the_traceback_and_exits_2(env: CliEnv, monkeypatch, capsys):
    """想定外の例外は握りつぶさない（SPEC 9章）。トレースバックを出して2で落ちる。"""
    env.input.write_bytes(b"")

    def boom(ctx, **kwargs):
        raise ValueError("説明のつかない失敗")

    monkeypatch.setattr(cli, "run_pipeline", boom)

    code = main(["run", str(env.input), *env.common()])

    err = capsys.readouterr().err
    assert code == EXIT_USAGE
    assert "Traceback" in err
    assert "説明のつかない失敗" in err


def test_keyboard_interrupt_is_not_an_internal_error(env: CliEnv, monkeypatch, capsys):
    """Ctrl-C は不具合ではない。トレースバックを出さず、中間ファイルを残す旨だけ伝える。"""
    env.input.write_bytes(b"")

    def interrupted(ctx, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pipeline", interrupted)

    code = main(["run", str(env.input), *env.common()])

    err = capsys.readouterr().err
    assert code == EXIT_INTERRUPTED
    _no_traceback(err)


# ===========================================================================
# transcribe（SPEC 7章）
# ===========================================================================


def test_transcribe_with_a_missing_input_returns_1(env: CliEnv, capsys):
    """`transcribe` も入力が無ければ想定内の失敗として1で終わる。"""
    code = main(
        [
            "transcribe",
            str(env.base / "not-here.mp4"),
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    _no_traceback(captured.err)


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_transcribe_without_an_asr_backend_returns_1(
    tmp_path: Path, episode_video: Path, capsys
):
    """文字起こしバックエンドが入っていない環境では、入れ方を案内して1で止まる。

    SPEC 6章 Step2「mlx-whisper …／利用できない場合は whisperx にフォールバックする」。
    どちらも無いなら黙って空の transcript.json を書いてはいけない。
    """
    import importlib.util

    if any(importlib.util.find_spec(n) is not None for n in ("whisperx", "mlx_whisper")):
        pytest.skip("文字起こしバックエンドが入っている環境です")

    env = CliEnv(tmp_path)
    shutil.copyfile(episode_video, env.input)

    code = main(
        [
            "transcribe",
            str(env.input),
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "mlx-whisper" in captured.err or "whisperx" in captured.err
    _no_traceback(captured.err)
    assert not (env.work_dir / "transcript.json").exists(), (
        "文字起こしできていないのに transcript.json を作ってはいけない"
    )


# ===========================================================================
# titles（SPEC 7章 / 6-b）
# ===========================================================================


def _seed_metadata_and_highlight(env: CliEnv) -> None:
    """`titles` が読む中間ファイル（metadata.json / highlight.json）を用意する。"""
    from radio_cutter.models import Chapter, MetadataResult, TitleCandidate, write_json
    from radio_cutter.steps import s5_pick_highlight as s5
    from radio_cutter.steps import s6_metadata as s6

    ctx = RunContext(
        input_path=env.input,
        episode_id=env.episode_id,
        work_dir=env.work_dir,
        out_dir=env.out_dir,
        config=cli.load_config(CONFIG_PATH),
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()

    meta = MetadataResult(
        summary_lead="AIに議事録を書かせるのは、実はいちばんもったいない使い方でした。",
        body="今回は議事録の自動化をテーマに実験しました。",
        chapters=[Chapter(time_sec=0.0, label="今回の結論")],
        keywords=["AI議事録", "業務自動化"],
        titles=[TitleCandidate(direction="結論直球型", text="古いタイトル")],
    )
    write_json(s6.metadata_path(ctx), meta.to_dict())

    selected = HighlightCandidate(
        start=25.65,
        end=33.65,
        score=92.0,
        hook_line="実はAIに議事録を書かせるのは一番もったいない使い方なんです",
        reason="逆説を含む。",
    )
    highlight = HighlightResult(selected=selected, snapped_from=selected)
    write_json(s5.highlight_path(ctx), highlight.to_dict())


def test_titles_overwrites_titles_md(env: CliEnv, capsys):
    """`titles <ep-id>` は out/<ep>/titles.md を作り直す（SPEC 7章「タイトルだけ再生成」）。"""
    _seed_metadata_and_highlight(env)
    titles_md = env.out_dir / "titles.md"
    titles_md.write_text("# 古い内容（上書きされるはず）\n", encoding="utf-8")

    code = main(
        [
            "titles",
            env.episode_id,
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
            "--stub-llm",
            str(env.stub),
        ]
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert str(titles_md) in out

    text = titles_md.read_text(encoding="utf-8")
    assert "古い内容" not in text, "titles.md が上書きされていない"
    assert text.startswith("# タイトル候補")
    # SPEC 6-b: 30個を6方向 × 5個。各行に想定文字数を併記する。
    assert len(re.findall(r"^\d+\. ", text, re.MULTILINE)) == 30
    assert len(re.findall(r"^## ", text, re.MULTILINE)) == 6
    assert "（全角" in text


def test_titles_without_intermediate_files_fails_readably(env: CliEnv, capsys):
    """中間ファイルが無いまま `titles` を打ったら、何が足りないかを言って1で止まる。"""
    code = main(
        [
            "titles",
            env.episode_id,
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
            "--stub-llm",
            str(env.stub),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "metadata.json" in captured.err
    assert "radio-cutter" in captured.err, "次に打つコマンドが案内されていない"
    _no_traceback(captured.err)


def test_titles_reports_failure_when_generation_fails(env: CliEnv, capsys):
    """タイトルを作れなかったのに0で終わってはいけない（黙って古い titles.md が残る）。"""
    _seed_metadata_and_highlight(env)
    broken_stub = env.base / "stub-without-titles.json"
    broken_stub.write_text(
        json.dumps({"highlight": fixtures.stub_highlight_response()}, ensure_ascii=False),
        encoding="utf-8",
    )

    code = main(
        [
            "titles",
            env.episode_id,
            "--config",
            str(CONFIG_PATH),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
            "--stub-llm",
            str(broken_stub),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "titles" in captured.err.lower() or "タイトル" in captured.err
    _no_traceback(captured.err)


def test_titles_without_an_llm_backend_returns_1(env: CliEnv, tmp_path: Path, monkeypatch, capsys):
    """LLM が使えないとき、トレースバックではなく案内で落ちること。

    provider を anthropic にしたうえでキーを外し、確実に「使えない」状態を作る
    （既定の claude_agent_sdk はこのパソコンの Claude Code を使うので、
     入っている環境では成功してしまい、この筋道を確かめられない）。
    """
    import importlib.util

    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("anthropic SDK が入っている環境です")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["llm"]["provider"] = "anthropic"
    raw["llm"]["model"] = "claude-opus-5"
    config = tmp_path / "api.json"
    config.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    _seed_metadata_and_highlight(env)

    code = main(
        [
            "titles",
            env.episode_id,
            "--config",
            str(config),
            "--work",
            str(env.work),
            "--out",
            str(env.out),
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "anthropic" in captured.err or "ANTHROPIC_API_KEY" in captured.err
    _no_traceback(captured.err)


# ===========================================================================
# 通し（SPEC 1章の成果物一式）
# ===========================================================================


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> CliEnv:
    """`run --from-step 3 --stub-llm` を1回だけ通し、その結果を使い回す。

    Step 1（音声抽出・probe）は CLI の `--only-step 1` で本当に走らせる。
    Step 2 だけはバックエンドが無いので合成の transcript.json で代替する。
    """
    if not fixtures.ffmpeg_available():
        pytest.skip("ffmpeg / ffprobe が PATH にありません")

    _reset_logging()
    base = tmp_path_factory.mktemp("cli-full-run")
    env = CliEnv(base)
    fixtures.build_test_video(env.input)

    buffer = StringIO()
    with redirect_stdout(buffer):
        step1 = main(["run", str(env.input), "--only-step", "1", *env.common()])
        _prepare_transcript(env)
        code = main(
            ["run", str(env.input), "--from-step", "3", "--stub-llm", str(env.stub), *env.common()]
        )
    _reset_logging()

    env.step1_code = step1
    env.code = code
    env.stdout = buffer.getvalue()
    return env


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_only_step_1_extracts_audio_and_probe(full_run: CliEnv):
    """`--only-step 1` は音声と probe.json だけを作る（SPEC Step 1）。"""
    assert full_run.step1_code == EXIT_OK
    assert (full_run.work_dir / "audio.wav").is_file()
    assert (full_run.work_dir / "probe.json").is_file()

    probe = json.loads((full_run.work_dir / "probe.json").read_text(encoding="utf-8"))
    assert probe["duration"] == pytest.approx(fixtures.EPISODE_DURATION, abs=0.5)
    assert probe["has_audio"] is True


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_full_run_produces_every_spec_chapter1_artifact(full_run: CliEnv):
    """SPEC 1章の表にある成果物が「全部」揃うこと。ここが本体の受け入れ基準。"""
    assert full_run.code == EXIT_OK

    missing = [name for name in SPEC_OUTPUTS if not (full_run.out_dir / name).is_file()]
    assert not missing, f"SPEC 1章の成果物が足りない: {missing}"
    for name in SPEC_OUTPUTS:
        assert (full_run.out_dir / name).stat().st_size > 0, f"{name} が空"


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_full_run_produces_preview_clips(full_run: CliEnv):
    """SPEC Step 8。カット点2箇所とハイライトの始点・終点のプレビューが出る。"""
    preview = full_run.out_dir / "preview"
    assert preview.is_dir()

    missing = [name for name in SPEC_PREVIEWS if not (preview / name).is_file()]
    assert not missing, f"プレビューが足りない: {missing}"
    assert sorted(p.name for p in preview.glob("cut_*.mp4")) == ["cut_A.mp4", "cut_B.mp4"]


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_full_run_decisions_json_follows_the_spec_schema(full_run: CliEnv):
    """decisions.json が SPEC 8章のキーを持ち、実際のカット点を記録している。"""
    payload = json.loads((full_run.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))

    for key in (
        "episode_id",
        "input",
        "input_sha256",
        "duration",
        "generated_at",
        "anchors",
        "highlight",
        "durations",
        "llm_calls",
        "warnings",
    ):
        assert key in payload, f"SPEC 8章のキー {key} が無い"

    assert payload["episode_id"] == full_run.episode_id
    assert Path(payload["input"]).is_absolute()
    assert re.fullmatch(r"[0-9a-f]{64}", payload["input_sha256"])

    # SPEC Phase 1 の受け入れ基準そのもの: アンカーが正しい位置に出ていること。
    assert payload["anchors"]["A"]["raw_cut_time"] == pytest.approx(
        fixtures.EXPECTED_ANCHOR_A_RAW, abs=0.05
    )
    assert payload["anchors"]["A"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.05)
    assert payload["anchors"]["B"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.05)
    assert payload["anchors"]["A"]["silence_found"] is True
    assert payload["anchors"]["B"]["silence_found"] is True

    assert set(payload["highlight"]) >= {"selected", "snapped_from"}
    assert payload["highlight"]["snapped_from"]["start"] == pytest.approx(26.5)

    # 書き出し済みなので durations は実測（SPEC Step 7 の ffprobe 検算）。
    assert payload["durations_source"] == SOURCE_MEASURED
    assert set(payload["durations"]) == {"highlight", "main", "ending", "final"}
    parts = sum(payload["durations"][k] for k in ("highlight", "main", "ending"))
    assert payload["durations"]["final"] == pytest.approx(parts, abs=0.5)

    # LLM は Step 5 / 6-a / 6-b の3回に分かれる（SPEC Step 6「同時に投げると品質が落ちる」）。
    steps = [call["step"] for call in payload["llm_calls"]]
    assert steps == ["highlight", "metadata", "titles"]


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_full_run_description_has_chapters_starting_at_zero(full_run: CliEnv):
    """概要欄はそのまま貼れる形（SPEC 6-a）。チャプターは 0:00 始まりで3つ以上。"""
    text = (full_run.out_dir / "description.txt").read_text(encoding="utf-8")

    assert "■ チャプター" in text
    stamps = re.findall(r"^(\d+:\d{2}(?::\d{2})?) .+$", text, re.MULTILINE)
    assert stamps, f"チャプター行が無い:\n{text}"
    assert stamps[0] == "0:00", "最初のチャプターは必ず 0:00"
    assert len(stamps) >= 3, "YouTube のチャプターは3つ以上必要"
    for tag in ("#AI", "#AI活用", "#生成AI"):
        assert tag in text, "config の hashtags が概要欄に出ていない"


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_full_run_prints_the_produced_files(full_run: CliEnv):
    """通したあと、どこに何ができたかが標準出力で分かる。"""
    stdout = full_run.stdout

    assert full_run.episode_id in stdout
    assert str(full_run.out_dir) in stdout
    for name in ("final.mp4", "description.txt", "titles.md", DECISIONS_FILE):
        assert name in stdout, f"{name} が実行結果のまとめに出ていない"


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_silence_options_actually_change_the_detected_cut(
    tmp_path: Path, episode_video: Path, capsys
):
    """`--silence-dur` が silencedetect の `d=` まで本当に届いていること（SPEC Step 4）。

    合成エピソードの無音は 0.30〜0.35秒。最短無音長を5秒にすれば1つも見つからなくなり、
    SPEC の指示どおり `cut_time = raw_cut_time - 0.08` にフォールバックして
    `"silence_found": false` が decisions.json に残るはず。
    ctx に値が入っているだけでは「ffmpeg に渡していない」不具合を見逃すので、
    ここは実際に検出結果が変わることで確かめる。
    """
    env = CliEnv(tmp_path)
    shutil.copyfile(episode_video, env.input)

    assert main(["run", str(env.input), "--only-step", "1", *env.common()]) == EXIT_OK
    _prepare_transcript(env)
    assert main(["run", str(env.input), "--only-step", "3", *env.common()]) == EXIT_OK
    capsys.readouterr()

    # Step 3 だけでは判断ログを書かない（Step 4 以降に踏み込んでから）。
    assert not (env.out_dir / DECISIONS_FILE).exists()

    code = main(["run", str(env.input), "--only-step", "4", "--silence-dur", "5.0", *env.common()])
    capsys.readouterr()
    assert code == EXIT_OK

    payload = json.loads((env.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))
    for anchor_id, raw in (("A", fixtures.EXPECTED_ANCHOR_A_RAW), ("B", fixtures.EXPECTED_ANCHOR_B_RAW)):
        entry = payload["anchors"][anchor_id]
        assert entry["silence_found"] is False, "無音が見つからないはずなのに見つけたことになっている"
        assert entry["cut_time"] == pytest.approx(raw - 0.08, abs=0.001)
    assert payload["warnings"], "無音が見つからなかったことが警告に残っていない"

    # 既定値に戻せば、同じ入力で無音の谷に寄る（fixtures の期待値）。
    assert main(["run", str(env.input), "--only-step", "4", *env.common()]) == EXIT_OK
    capsys.readouterr()
    payload = json.loads((env.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))
    assert payload["anchors"]["A"]["silence_found"] is True
    assert payload["anchors"]["A"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.05)
    assert payload["anchors"]["B"]["cut_time"] == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.05)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.slow
def test_dry_run_then_preview_then_render(tmp_path: Path, episode_video: Path, capsys):
    """SPEC 7章が既定の運用フローとして挙げる3段構えが回ること。

    1. `--dry-run` … Step 6 まで。decisions.json とメタデータは作るが動画は書かない。
    2. `--preview-only` … Step 8 だけ。カット点を目視で確認する。
    3. `--from-step 7` … 問題なければ書き出す。
    """
    env = CliEnv(tmp_path)
    shutil.copyfile(episode_video, env.input)

    assert main(["run", str(env.input), "--only-step", "1", *env.common()]) == EXIT_OK
    _prepare_transcript(env)

    # ---- 1. --dry-run（SPEC 7章「Step 6まで実行し、書き出しを行わない」）----
    code = main(
        ["run", str(env.input), "--dry-run", "--from-step", "3", "--stub-llm", str(env.stub), *env.common()]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert (env.work_dir / "cuts.json").is_file()
    assert (env.work_dir / "highlight.json").is_file()
    assert (env.out_dir / DECISIONS_FILE).is_file()
    assert (env.out_dir / "description.txt").is_file()
    assert (env.out_dir / "titles.md").is_file()
    for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4", "final.mp4"):
        assert not (env.out_dir / name).exists(), f"--dry-run なのに {name} を書いている"
    assert not (env.out_dir / "preview").exists(), "--dry-run なのにプレビューを書いている"
    assert "--from-step 7" in out, "次に何をすればよいかが案内されていない"

    # --dry-run 時点の durations は書き出し前なので想定値（SPEC 8章）。
    payload = json.loads((env.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))
    assert payload["durations_source"] == SOURCE_ESTIMATED
    assert payload["durations"]["main"] == pytest.approx(
        fixtures.EXPECTED_CUT_B - fixtures.EXPECTED_CUT_A, abs=0.05
    )

    # ---- 2. --preview-only（Step 8 だけ）----
    assert main(["run", str(env.input), "--preview-only", *env.common()]) == EXIT_OK
    capsys.readouterr()
    for name in SPEC_PREVIEWS:
        assert (env.out_dir / "preview" / name).is_file(), f"{name} が作られていない"
    assert not (env.out_dir / "final.mp4").exists(), "--preview-only なのに本編を書き出している"

    # ---- 3. --from-step 7（書き出し）----
    assert main(["run", str(env.input), "--from-step", "7", *env.common()]) == EXIT_OK
    capsys.readouterr()
    for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4", "final.mp4"):
        assert (env.out_dir / name).is_file(), f"{name} が書き出されていない"

    payload = json.loads((env.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))
    assert payload["durations_source"] == SOURCE_MEASURED
    first_durations = payload["durations"]

    # ---- 4. もう一度書き出しても同じ結果になる（何度でも打ち直せること）----
    assert main(["run", str(env.input), "--from-step", "7", *env.common()]) == EXIT_OK
    capsys.readouterr()
    payload = json.loads((env.out_dir / DECISIONS_FILE).read_text(encoding="utf-8"))
    assert payload["durations"] == first_durations


# ===========================================================================
# decisions.py（SPEC 8章）
# ===========================================================================


def test_now_iso_is_timezone_aware_iso8601():
    """`generated_at` はタイムゾーン付き ISO8601（SPEC 8章の例は "+09:00" 付き）。

    タイムゾーンが無いと、あとから別マシンでログを突き合わせたときに時刻が意味を失う。
    """
    value = now_iso()

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)", value), value
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    assert parsed.microsecond == 0, "秒未満は落とす（読みやすさ優先）"


def _make_config(segment_names: tuple[str, str] = ("main", "ending")) -> Config:
    """アンカーID・セグメント名を差し替えた設定を作る（決め打ちの検出用）。"""
    first, second = segment_names
    return Config.from_dict(
        {
            "channel": "テストチャンネル",
            "anchors": [
                {"id": "A", "phrase": "このチャンネルは"},
                {"id": "B", "phrase": "ということで"},
            ],
            "segments": [
                {"name": first, "file": "02.mp4", "from": "A", "to": "B"},
                {"name": second, "file": "03.mp4", "from": "B", "to": "end"},
            ],
            "highlight": {"source_segment": first},
        }
    )


def _make_ctx(tmp_path: Path, config: Config, *, episode_id: str = "ep42") -> RunContext:
    input_path = tmp_path / f"{episode_id}.mp4"
    input_path.write_bytes(b"dummy")
    return RunContext(
        input_path=input_path,
        episode_id=episode_id,
        work_dir=tmp_path / "work" / episode_id,
        out_dir=tmp_path / "out" / episode_id,
        config=config,
        silence=SilenceConfig(),
    )


def _sample_pieces():
    """decisions.json を埋めるだけの一式（anchors / cuts / highlight / media）。"""
    media = MediaInfo(path="/tmp/ep42.mp4", duration=100.0, has_video=True, has_audio=True)
    anchors = {
        "A": AnchorResult(
            id="A",
            phrase="このチャンネルは",
            matched_text="このチャンネルは",
            score=100.0,
            raw_cut_time=10.0,
            candidates_found=1,
            candidates_rejected=0,
            context="…このチャンネルは…",
        ),
        "B": AnchorResult(
            id="B",
            phrase="ということで",
            matched_text="ということで",
            score=96.234,
            raw_cut_time=50.0,
            candidates_found=7,
            candidates_rejected=6,
            context="…ということで、木原さん…",
        ),
    }
    cuts = {
        "A": CutPoint(anchor_id="A", raw_cut_time=10.0, cut_time=9.9, silence_found=True, score=100.0),
        "B": CutPoint(anchor_id="B", raw_cut_time=50.0, cut_time=49.9, silence_found=False, score=96.234),
    }
    selected = HighlightCandidate(start=20.0, end=40.0, score=92.0, reason="逆説を含む。")
    highlight = HighlightResult(
        selected=selected,
        snapped_from=HighlightCandidate(start=21.5, end=39.2, score=92.0),
        alternatives=[HighlightCandidate(start=12.0, end=30.0, score=85.0, reason="次点。")],
    )
    return media, anchors, cuts, highlight


def test_build_decisions_has_every_spec_chapter8_key(tmp_path: Path):
    """SPEC 8章のキーが揃っていること。"""
    ctx = _make_ctx(tmp_path, _make_config())
    media, anchors, cuts, highlight = _sample_pieces()

    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=None,
        generated_at="2026-08-30T14:20:11+09:00",
    )

    for key in (
        "episode_id",
        "input",
        "input_sha256",
        "duration",
        "generated_at",
        "anchors",
        "highlight",
        "durations",
        "llm_calls",
        "warnings",
    ):
        assert key in payload, f"SPEC 8章のキー {key} が無い"

    assert payload["episode_id"] == "ep42"
    assert Path(payload["input"]).is_absolute()
    assert re.fullmatch(r"[0-9a-f]{64}", payload["input_sha256"])
    assert payload["duration"] == pytest.approx(100.0)
    assert payload["generated_at"] == "2026-08-30T14:20:11+09:00"

    assert list(payload["anchors"]) == ["A", "B"], "アンカーの並びは config の順"
    assert payload["anchors"]["A"] == {
        "raw_cut_time": 10.0,
        "cut_time": 9.9,
        "silence_found": True,
        "score": 100.0,
    }
    assert payload["anchors"]["B"]["silence_found"] is False

    assert payload["highlight"]["selected"]["start"] == pytest.approx(20.0)
    assert payload["highlight"]["snapped_from"] == {"start": 21.5, "end": 39.2}
    assert len(payload["highlight"]["alternatives"]) == 1


def test_warnings_and_llm_calls_are_always_present(tmp_path: Path):
    """`warnings` と `llm_calls` は空でも必ず出す。

    「何も無かった」ことも記録として残す。キーごと消えると
    「起きなかった」のか「記録し忘れた」のか区別できない。
    """
    ctx = _make_ctx(tmp_path, _make_config())

    payload = build_decisions(
        ctx,
        media=None,
        anchors=None,
        cuts=None,
        highlight=None,
        render=None,
        generated_at=now_iso(),
    )

    assert payload["warnings"] == []
    assert payload["llm_calls"] == []
    assert payload["episode_id"] == "ep42"


def test_warnings_and_llm_calls_are_taken_from_the_context(tmp_path: Path):
    """実行中に貯めた警告と LLM 呼び出しがそのまま載る。"""
    ctx = _make_ctx(tmp_path, _make_config())
    ctx.warn("ハイライトが短すぎます。")
    ctx.warn("ハイライトが短すぎます。")  # 同じ文言は1回だけ
    ctx.record_llm_call(
        LlmCallRecord(step="highlight", model="claude-sonnet-4-6", input_tokens=24810, retries=0)
    )

    payload = build_decisions(
        ctx,
        media=None,
        anchors=None,
        cuts=None,
        highlight=None,
        render=None,
        generated_at=now_iso(),
    )

    assert payload["warnings"] == ["ハイライトが短すぎます。"]
    call = payload["llm_calls"][0]
    for key in ("step", "model", "input_tokens", "retries"):
        assert key in call, f"SPEC 8章の llm_calls[] に {key} が無い"
    assert call["step"] == "highlight"
    assert call["input_tokens"] == 24810


def test_durations_are_estimated_before_rendering(tmp_path: Path):
    """まだ書き出していなければ、durations は cuts と highlight から求めた想定値。"""
    ctx = _make_ctx(tmp_path, _make_config())
    media, anchors, cuts, highlight = _sample_pieces()

    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=None,
        generated_at=now_iso(),
    )

    durations = payload["durations"]
    assert payload["durations_source"] == SOURCE_ESTIMATED
    assert durations["highlight"] == pytest.approx(20.0)          # 40.0 - 20.0
    assert durations["main"] == pytest.approx(40.0)               # 49.9 - 9.9
    assert durations["ending"] == pytest.approx(50.1)             # 100.0 - 49.9
    assert durations["final"] == pytest.approx(110.1)


def test_durations_are_measured_after_rendering(tmp_path: Path):
    """書き出し済みなら、durations は Step 7 が ffprobe で測った実尺を使う。

    想定値と実測値がずれていても実測を優先する。SPEC Step 7 が
    「実尺を ffprobe で検算し、0.5秒を超えたら警告」と言えるのはここが実測だから。
    """
    ctx = _make_ctx(tmp_path, _make_config())
    media, anchors, cuts, highlight = _sample_pieces()
    render = RenderResult(
        files={"highlight": "01.mp4", "main": "02.mp4", "ending": "03.mp4", "final": "final.mp4"},
        durations={"highlight": 20.42, "main": 39.87, "ending": 50.31, "final": 110.55},
    )

    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=render,
        generated_at=now_iso(),
    )

    assert payload["durations_source"] == SOURCE_MEASURED
    assert payload["durations"] == {
        "highlight": 20.42,
        "main": 39.87,
        "ending": 50.31,
        "final": 110.55,
    }


def test_duration_keys_follow_the_config_segment_names(tmp_path: Path):
    """durations のキーは config の segments[].name。"main"/"ending" と決め打ちしない。

    SPEC 5章「アンカー語をコードにハードコードしないこと。チャンネルごとにJSONで持つ」。
    セグメント構成もチャンネルごとに変わるので、同じ理屈が durations にも効く。
    """
    ctx = _make_ctx(tmp_path, _make_config(("honpen", "owari")))
    media, anchors, cuts, highlight = _sample_pieces()

    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=None,
        generated_at=now_iso(),
    )

    assert list(payload["durations"]) == ["highlight", "honpen", "owari", "final"]
    assert "main" not in payload["durations"]
    assert "ending" not in payload["durations"]
    assert payload["durations"]["honpen"] == pytest.approx(40.0)
    assert payload["durations"]["owari"] == pytest.approx(50.1)


def test_measured_duration_keys_also_follow_the_config(tmp_path: Path):
    """実測側（Step 7 の結果）も同じ名前で載ること。"""
    ctx = _make_ctx(tmp_path, _make_config(("honpen", "owari")))
    media, anchors, cuts, highlight = _sample_pieces()
    render = RenderResult(
        durations={"highlight": 20.0, "honpen": 40.0, "owari": 50.1, "final": 110.1}
    )

    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=render,
        generated_at=now_iso(),
    )

    assert list(payload["durations"]) == ["highlight", "honpen", "owari", "final"]


def test_anchor_order_follows_the_config_not_the_input_dict(tmp_path: Path):
    """anchors の並びは config のアンカー順。入力 dict の順に引きずられない。"""
    ctx = _make_ctx(tmp_path, _make_config())
    media, anchors, cuts, highlight = _sample_pieces()
    shuffled = {"B": cuts["B"], "A": cuts["A"]}

    payload = build_decisions(
        ctx,
        media=media,
        anchors=None,
        cuts=shuffled,
        highlight=None,
        render=None,
        generated_at=now_iso(),
    )

    assert list(payload["anchors"]) == ["A", "B"]


def test_seconds_are_rounded_to_three_decimals(tmp_path: Path):
    """秒数は小数点以下3桁（SPEC 11章）。"""
    ctx = _make_ctx(tmp_path, _make_config())
    media = MediaInfo(path="/tmp/ep42.mp4", duration=100.123456, has_audio=True)
    cuts = {
        "A": CutPoint(anchor_id="A", raw_cut_time=10.1234567, cut_time=9.9876543, silence_found=True),
    }

    payload = build_decisions(
        ctx,
        media=media,
        anchors=None,
        cuts=cuts,
        highlight=None,
        render=None,
        generated_at=now_iso(),
    )

    assert payload["duration"] == 100.123
    assert payload["anchors"]["A"]["raw_cut_time"] == 10.123
    assert payload["anchors"]["A"]["cut_time"] == 9.988


def test_missing_values_are_omitted_not_written_as_null(tmp_path: Path):
    """取れなかった値はキーごと省く。null を書くと「0だった」と取り違えられる。"""
    ctx = _make_ctx(tmp_path, _make_config())

    payload = build_decisions(
        ctx,
        media=None,
        anchors=None,
        cuts=None,
        highlight=None,
        render=None,
        generated_at=now_iso(),
    )

    assert "duration" not in payload
    assert "anchors" not in payload
    assert "highlight" not in payload
    assert None not in payload.values()


def test_write_decisions_writes_readable_json(tmp_path: Path):
    """decisions.json は out/<episode_id>/ に、そのまま読める JSON で書かれる。"""
    ctx = _make_ctx(tmp_path, _make_config())
    ctx.ensure_dirs()
    media, anchors, cuts, highlight = _sample_pieces()
    payload = build_decisions(
        ctx,
        media=media,
        anchors=anchors,
        cuts=cuts,
        highlight=highlight,
        render=None,
        generated_at=now_iso(),
    )

    path = write_decisions(ctx, payload)

    assert path == decisions_path(ctx) == ctx.out_dir / DECISIONS_FILE
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    # 日本語をエスケープしない（人が読む前提のファイル）
    assert "\\u" not in path.read_text(encoding="utf-8")
