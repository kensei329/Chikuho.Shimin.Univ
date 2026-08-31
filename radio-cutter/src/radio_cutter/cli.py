"""SPEC 7章「CLI仕様」。argparse で run / doctor / transcribe / titles を提供する。

方針:
- 外部依存を増やさないため argparse を使う（click などは入れない）。
- ここだけ `print()` を使ってよい。人に見せる結果は stdout、ログは stderr（logging_util が stderr 固定）。
- 例外は握りつぶさず、`RadioCutterError` はメッセージだけ出して終了コード 1、
  想定外の例外はトレースバックを出して 2 で落とす（SPEC 9章）。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable, Sequence

from . import __version__
from .config import Config, RenderConfig, SilenceConfig, load_config
from .context import RunContext
from .decisions import DECISIONS_FILE, decisions_path
from .errors import FfmpegError, RadioCutterError
from .llm.client import LlmClient, build_client, load_stub_responses
from .logging_util import get_logger, setup_logging
from .models import read_json
from .pipeline import FIRST_STEP, LAST_STEP, PipelineResult, run_pipeline, run_steps
from .steps import s1_extract_audio as s1
from .steps import s2_transcribe as s2
from .steps import s5_pick_highlight as s5
from .steps import s6_metadata as s6
from .util.ffmpeg import (
    MediaInfo,
    ffmpeg_bin,
    ffmpeg_version,
    ffprobe_bin,
    list_encoders,
    run_ffprobe,
)

logger = get_logger(__name__)

PROG = "radio-cutter"

#: SPEC 7章の既定値
DEFAULT_CONFIG_RELATIVE = Path("config") / "ai-radio.json"
DEFAULT_OUT_DIR = "out"
DEFAULT_WORK_DIR = "work"

#: doctor の表示記号
STATUS_OK = "[OK]"
STATUS_WARN = "[警告]"
STATUS_NG = "[NG]"

#: 終了コード
EXIT_OK = 0
EXIT_ERROR = 1        # RadioCutterError（想定内の失敗）
EXIT_USAGE = 2        # 使い方の誤り・想定外の例外
EXIT_INTERRUPTED = 130  # Ctrl-C

#: doctor が見る Python の下限（pyproject の requires-python と揃える）
MIN_PYTHON = (3, 11)

#: doctor が有無を確認するライブラリ: (import 名, 配布名, 必須か, 用途)
CHECKED_MODULES: tuple[tuple[str, str, bool, str], ...] = (
    ("rapidfuzz", "rapidfuzz", True, "Step 3 のあいまい一致"),
    ("jsonschema", "jsonschema", True, "LLM 応答のスキーマ検証"),
    ("claude_agent_sdk", "claude-agent-sdk", False, "Step 5・6（このパソコンの Claude Code を呼ぶ）"),
    ("anthropic", "anthropic", False, "Step 5・6（Anthropic API を直接叩く場合）"),
    ("whisperx", "whisperx", False, "Step 2 の文字起こし（faster-whisper 系）"),
    ("mlx_whisper", "mlx-whisper", False, "Step 2 の文字起こし（Apple Silicon 向け）"),
)

#: 文字起こしバックエンドの import 名（どれか1つあればよい）
ASR_MODULES: tuple[str, ...] = ("whisperx", "mlx_whisper")


# ---------------------------------------------------------------------------
# 引数の型
# ---------------------------------------------------------------------------


def _step_number(value: str) -> int:
    """--from-step / --only-step の値。範囲は 1〜8。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"ステップ番号は整数で指定してください（実際: {value!r}）。"
        ) from None
    if not FIRST_STEP <= number <= LAST_STEP:
        raise argparse.ArgumentTypeError(
            f"ステップ番号は {FIRST_STEP}〜{LAST_STEP} で指定してください（実際: {number}）。"
        )
    return number


def _positive_float(value: str) -> float:
    """--silence-dur の値。0 以下は silencedetect が受け付けない。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"秒数は数値で指定してください（実際: {value!r}）。") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"秒数は0より大きい値にしてください（実際: {number}）。")
    return number


def _float_value(value: str) -> float:
    """--silence-db の値（負の数を受け付ける）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"dB は数値で指定してください（実際: {value!r}）。") from None


# ---------------------------------------------------------------------------
# パスと文脈
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    """--config の既定値。カレント直下 → リポジトリ同梱 の順に探す。

    どこから起動しても `radio-cutter doctor` が設定を読めるようにするための保険。
    """
    here = Path.cwd() / DEFAULT_CONFIG_RELATIVE
    if here.exists():
        return here
    bundled = Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_RELATIVE
    if bundled.exists():
        return bundled
    return here


def _silence_config(config: Config, args: argparse.Namespace) -> SilenceConfig:
    """--silence-db / --silence-dur で SilenceConfig を上書きする（SPEC Step 4）。"""
    base = config.silence
    noise_db = getattr(args, "silence_db", None)
    min_dur = getattr(args, "silence_dur", None)
    if noise_db is None and min_dur is None:
        return base
    silence = SilenceConfig(
        noise_db=base.noise_db if noise_db is None else float(noise_db),
        min_duration_sec=base.min_duration_sec if min_dur is None else float(min_dur),
    )
    logger.info(
        "無音検出の設定を上書きしました: n=%gdB / d=%g秒（既定は %gdB / %g秒）",
        silence.noise_db,
        silence.min_duration_sec,
        base.noise_db,
        base.min_duration_sec,
    )
    return silence


def _episode_id(args: argparse.Namespace, input_path: Path) -> str:
    """--episode-id の既定は入力ファイル名の stem。"""
    given = getattr(args, "episode_id", None)
    episode_id = str(given).strip() if given else input_path.stem.strip()
    if not episode_id:
        raise RadioCutterError(
            f"エピソードIDが決められません（入力: {input_path}）。--episode-id で明示してください。"
        )
    return episode_id


def build_context(
    args: argparse.Namespace,
    config: Config,
    input_path: Path,
    episode_id: str,
) -> RunContext:
    """CLI 引数から RunContext を組み立てる。置き場は <work>/<ep> と <out>/<ep>。"""
    work_root = Path(getattr(args, "work", None) or DEFAULT_WORK_DIR).expanduser()
    out_root = Path(getattr(args, "out", None) or DEFAULT_OUT_DIR).expanduser()
    return RunContext(
        input_path=input_path,
        episode_id=episode_id,
        work_dir=work_root / episode_id,
        out_dir=out_root / episode_id,
        config=config,
        silence=_silence_config(config, args),
        dry_run=bool(getattr(args, "dry_run", False)),
        force_transcribe=bool(getattr(args, "force_transcribe", False)),
    )


def _stub_llm(args: argparse.Namespace, config: Config) -> LlmClient | None:
    """--stub-llm PATH が指定されていれば StubLlmClient を作る。

    JSON の形は {"highlight": {...}, "metadata": {...}, "titles": {...}}。
    APIキー無しで通しの動作確認ができるようにするための入口。
    """
    path = getattr(args, "stub_llm", None)
    if not path:
        return None
    responses = load_stub_responses(path)
    client = build_client(config.llm, stub_responses=responses)
    logger.info(
        "スタブ LLM を使います: %s（用意されている step: %s）",
        path,
        ", ".join(sorted(responses)) or "（なし）",
    )
    return client


def _recover_input_path(work_dir: Path, out_dir: Path, episode_id: str) -> Path:
    """`titles` のように入力ファイルを受け取らないコマンド用に、元の入力パスを復元する。

    エラーメッセージの案内（`radio-cutter run <input> ...`）を正しく出すためだけに使う。
    """
    probe = work_dir / s1.PROBE_FILENAME
    if probe.is_file():
        try:
            media = MediaInfo.from_dict(read_json(probe))
        except (OSError, ValueError, KeyError, TypeError):
            media = None
        if media is not None and media.path:
            return Path(media.path)

    decisions = out_dir / DECISIONS_FILE
    if decisions.is_file():
        try:
            data = read_json(decisions)
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("input"):
            return Path(str(data["input"]))

    return Path(f"{episode_id}.mp4")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """SPEC 3章のパイプラインを回す。"""
    if args.from_step is not None and args.only_step is not None:
        args._parser.error("--from-step と --only-step は同時に指定できません。どちらか一方にしてください。")

    config = load_config(args.config)
    input_path = Path(args.input).expanduser()
    episode_id = _episode_id(args, input_path)
    ctx = build_context(args, config, input_path, episode_id)
    llm = _stub_llm(args, config)

    result = run_pipeline(
        ctx,
        from_step=args.from_step if args.from_step is not None else FIRST_STEP,
        only_step=args.only_step,
        dry_run=bool(args.dry_run),
        preview_only=bool(args.preview_only),
        llm=llm,
    )
    _print_run_summary(result)
    return EXIT_OK


def _print_run_summary(result: PipelineResult) -> None:
    """人が見る用のまとめ。ログと違って stdout に出す（パイプで拾えるように）。"""
    ctx = result.ctx
    print(f"エピソード: {ctx.episode_id}")
    print(f"成果物: {ctx.out_dir}")
    print(f"中間ファイル: {ctx.work_dir}")

    outputs: list[Path] = []
    if result.render is not None:
        outputs.extend(Path(p) for p in result.render.files.values())
    for path in (s6.description_path(ctx), s6.titles_path(ctx)):
        if path.exists():
            outputs.append(path)
    if result.decisions is not None:
        outputs.append(decisions_path(ctx))
    for path in result.previews:
        outputs.append(Path(path))

    seen: set[Path] = set()
    for path in outputs:
        if path in seen:
            continue
        seen.add(path)
        print(f"  - {path}")

    if ctx.dry_run:
        print("--dry-run のため Step 7（書き出し）と Step 8（プレビュー）は行っていません。")
        print("カット点を確認したら `--from-step 7` で書き出してください。")

    if ctx.warnings:
        print(f"警告 {len(ctx.warnings)} 件:")
        for warning in ctx.warnings:
            print(f"  ! {warning}")


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Step 1・2 だけを実行して transcript.json のパスを表示する（SPEC 7章）。"""
    config = load_config(args.config)
    input_path = Path(args.input).expanduser()
    episode_id = _episode_id(args, input_path)
    ctx = build_context(args, config, input_path, episode_id)

    run_steps(ctx, [s1.STEP, s2.STEP])

    path = s2.transcript_path(ctx)
    if not path.exists():
        raise RadioCutterError(f"文字起こしの出力が見つかりません: {path}")
    print(str(path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# titles
# ---------------------------------------------------------------------------


def cmd_titles(args: argparse.Namespace) -> int:
    """work/<ep>/metadata.json と highlight.json から 6-b だけ回して titles.md を上書きする。"""
    config = load_config(args.config)
    episode_id = str(args.episode_id).strip()
    if not episode_id:
        raise RadioCutterError("エピソードIDが空です。")

    work_dir = Path(args.work).expanduser() / episode_id
    out_dir = Path(args.out).expanduser() / episode_id
    ctx = RunContext(
        input_path=_recover_input_path(work_dir, out_dir, episode_id),
        episode_id=episode_id,
        work_dir=work_dir,
        out_dir=out_dir,
        config=config,
        silence=config.silence,
    )

    meta = s6.load(ctx)
    highlight = s5.load(ctx)
    try:
        transcript = s2.load(ctx)
    except RadioCutterError as exc:
        # hook_line の補完に使うだけなので、無くても続ける。
        logger.debug("文字起こしを読めなかったので hook_line の補完は行いません: %s", exc)
        transcript = None

    llm = _stub_llm(args, config) or build_client(config.llm)

    before = list(ctx.warnings)
    meta = s6.regenerate_titles(ctx, meta, highlight, llm, transcript)
    new_warnings = [w for w in ctx.warnings if w not in before]
    if any("Step 6-b" in w for w in new_warnings):
        for warning in new_warnings:
            print(warning, file=sys.stderr)
        return EXIT_ERROR

    path = s6.titles_path(ctx)
    print(f"{path}（{len(meta.titles)} 個）")
    return EXIT_OK


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


class _Report:
    """doctor の1行ずつの結果。NG が1つでもあれば終了コード1にする。"""

    def __init__(self) -> None:
        self.ok = 0
        self.warned = 0
        self.failed = 0

    def add(self, status: str, message: str) -> None:
        if status == STATUS_NG:
            self.failed += 1
        elif status == STATUS_WARN:
            self.warned += 1
        else:
            self.ok += 1
        print(f"{status} {message}")


def _has_module(name: str) -> bool:
    """モジュールが入っているか。import はせず find_spec だけで判定する（重い依存を読み込まない）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _distribution_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # メタデータが壊れている環境でも doctor は止めない
        return None


def _ffprobe_version() -> str | None:
    """ffprobe のバージョン文字列。取れなければ None。"""
    try:
        proc = run_ffprobe(["-version"], check=False)
    except FfmpegError:
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"^ffprobe version (\S+)", proc.stdout or "", re.MULTILINE)
    if match:
        return match.group(1)
    lines = (proc.stdout or "").strip().splitlines()
    return lines[0] if lines else None


def _check_python(report: _Report) -> None:
    version = ".".join(str(v) for v in sys.version_info[:3])
    required = ".".join(str(v) for v in MIN_PYTHON)
    if sys.version_info >= MIN_PYTHON:
        report.add(STATUS_OK, f"Python {version}（{required} 以上）")
    else:
        report.add(STATUS_NG, f"Python {version} は古すぎます。{required} 以上を使ってください。")


def _check_config(report: _Report, config_path: Path) -> Config | None:
    try:
        config = load_config(config_path)
    except RadioCutterError as exc:
        report.add(STATUS_WARN, f"設定ファイルを読めません: {config_path}\n     {exc}")
        return None
    anchors = "、".join(f"{a.id}「{a.phrase}」" for a in config.anchors)
    report.add(
        STATUS_OK,
        f"設定ファイル {config_path}（チャンネル「{config.channel}」/ アンカー {anchors}）",
    )
    return config


def _check_ffmpeg(report: _Report) -> bool:
    """ffmpeg / ffprobe の有無とバージョン。どちらも無ければ何も動かないので NG。"""
    ok = True

    name = ffmpeg_bin()
    path = shutil.which(name)
    if path is None:
        ok = False
        report.add(
            STATUS_NG,
            f"{name} が見つかりません。macOS なら `brew install ffmpeg` で入れてください。",
        )
    else:
        version = ffmpeg_version() or "（バージョン不明）"
        report.add(STATUS_OK, f"ffmpeg {version}（{path}）")

    probe_name = ffprobe_bin()
    probe_path = shutil.which(probe_name)
    if probe_path is None:
        ok = False
        report.add(
            STATUS_NG,
            f"{probe_name} が見つかりません。ffmpeg と一緒に入るはずなので、"
            "インストールを確認してください。",
        )
    else:
        version = _ffprobe_version() or "（バージョン不明）"
        report.add(STATUS_OK, f"ffprobe {version}（{probe_path}）")

    return ok


def _check_encoders(report: _Report, render: RenderConfig) -> None:
    """VideoToolbox の有無（SPEC 2章）。無ければ CPU エンコードへのフォールバックを警告する。"""
    try:
        encoders = list_encoders()
    except RadioCutterError as exc:
        report.add(STATUS_WARN, f"ffmpeg のエンコーダ一覧を取得できませんでした: {exc}")
        return

    if render.video_codec in encoders:
        report.add(STATUS_OK, f"{render.video_codec} が使えます（ハードウェアエンコード）。")
        return

    report.add(
        STATUS_WARN,
        f"{render.video_codec} が見つかりません。"
        f"CPU エンコード（{render.fallback_video_codec}）にフォールバックします。",
    )
    if render.fallback_video_codec in encoders:
        report.add(
            STATUS_OK,
            f"{render.fallback_video_codec} が使えます（書き出しは遅くなりますが動きます）。",
        )
    else:
        report.add(
            STATUS_NG,
            f"フォールバック先の {render.fallback_video_codec} もありません。"
            "この ffmpeg では書き出しができません。",
        )


def _check_modules(report: _Report) -> None:
    """必須ライブラリと任意ライブラリの有無を1行ずつ出す。"""
    for module_name, dist_name, required, purpose in CHECKED_MODULES:
        if _has_module(module_name):
            version = _distribution_version(dist_name)
            suffix = f" {version}" if version else ""
            report.add(STATUS_OK, f"{module_name}{suffix}（{purpose}）")
        elif required:
            report.add(
                STATUS_NG,
                f"{module_name} がありません（{purpose}）。`pip install {dist_name}` で入れてください。",
            )
        else:
            report.add(
                STATUS_WARN,
                f"{module_name} がありません（{purpose}）。使うなら `pip install {dist_name}`。",
            )

    if not any(_has_module(name) for name in ASR_MODULES):
        report.add(
            STATUS_WARN,
            "文字起こしバックエンドが1つもありません（Step 2 を実行できません）。"
            "`pip install mlx-whisper` か `pip install whisperx` を入れてください"
            "（既存の transcript.json があれば Step 3 以降は動きます）。",
        )


def _check_llm(report: _Report, config: Config | None) -> None:
    """LLM を呼ぶ準備ができているか。無くても Step 1〜4 と --stub-llm は動く。

    既定の claude_agent_sdk は APIキーではなく Claude Code のログインを使うので、
    見るところが違う。設定に書かれたプロバイダに応じて確認先を変える。
    """
    from .llm.client import PROVIDER_ALIASES

    llm = config.llm if config is not None else None
    raw = (llm.provider if llm is not None else "claude_agent_sdk") or ""
    provider = PROVIDER_ALIASES.get(raw.strip().lower())

    if provider == "anthropic":
        env_name = llm.api_key_env if llm is not None else "ANTHROPIC_API_KEY"
        if os.environ.get(env_name):
            report.add(STATUS_OK, f"環境変数 {env_name} は設定されています。")
        else:
            report.add(
                STATUS_WARN,
                f"環境変数 {env_name} が未設定です。Step 5・6（LLM）が実行できません"
                "（--stub-llm を使うなら不要）。",
            )
        return

    if provider != "claude_agent_sdk":
        report.add(
            STATUS_NG,
            f"config の llm.provider が未対応です: {raw!r}"
            "（'claude_agent_sdk' か 'anthropic' にしてください）。",
        )
        return

    claude = shutil.which("claude")
    if claude:
        report.add(STATUS_OK, f"Claude Code が見つかりました（{claude}）。APIキーは要りません。")
    else:
        report.add(
            STATUS_WARN,
            "Claude Code（`claude` コマンド）が見つかりません。Step 5・6（LLM）が実行できません。"
            "https://claude.com/claude-code から入れて、一度 `claude` を起動して"
            "ログインしてください（--stub-llm を使うなら不要）。",
        )


def cmd_doctor(args: argparse.Namespace) -> int:
    """環境チェック（SPEC 2章）。NG が1つでもあれば終了コード1。"""
    report = _Report()
    print(f"{PROG} doctor — 実行環境を確認します")

    _check_python(report)
    config = _check_config(report, Path(args.config))
    render = config.render if config is not None else RenderConfig()

    if _check_ffmpeg(report):
        _check_encoders(report, render)
    else:
        report.add(STATUS_WARN, "ffmpeg が無いため、エンコーダの確認は飛ばしました。")

    _check_modules(report)
    _check_llm(report, config)

    print(f"— OK {report.ok} 件 / 警告 {report.warned} 件 / NG {report.failed} 件")
    if report.failed:
        print("NG の項目を解消してから実行してください。", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


# ---------------------------------------------------------------------------
# パーサ
# ---------------------------------------------------------------------------


def _common_parser() -> argparse.ArgumentParser:
    """全サブコマンド共通のログ設定。既定を SUPPRESS にして、サブコマンド側の既定で
    上書きされないようにする（`radio-cutter -v run ...` も `run ... -v` も効かせるため）。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="ログを詳しく出す（DEBUG レベル）",
    )
    common.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="警告以上だけ出す",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """SPEC 7章のコマンド体系を組み立てる。"""
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog=PROG,
        parents=[common],
        description="収録動画を決まり文句で自動分割し、YouTube 公開物一式を作る。",
        epilog=(
            "例:\n"
            f"  {PROG} run ep42.mp4 --dry-run\n"
            f"  {PROG} run ep42.mp4 --from-step 7\n"
            f"  {PROG} run ep42.mp4 --silence-db=-30 --silence-dur 0.15\n"
            f"  {PROG} doctor\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="コマンド", required=True)

    default_config = str(default_config_path())

    # ----- run -----
    run_p = subparsers.add_parser(
        "run",
        parents=[common],
        help="パイプライン（Step 1〜8）を実行する",
        description="収録動画を分割し、ハイライト・概要欄・タイトル候補まで作る。",
    )
    run_p.add_argument("input", help="入力の動画ファイル（mp4 など）")
    run_p.add_argument("--config", default=default_config, help=f"設定ファイル（既定: {default_config}）")
    run_p.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"成果物の置き場（既定: {DEFAULT_OUT_DIR}/）")
    run_p.add_argument("--work", default=DEFAULT_WORK_DIR, help=f"中間ファイルの置き場（既定: {DEFAULT_WORK_DIR}/）")
    run_p.add_argument(
        "--from-step",
        type=_step_number,
        default=None,
        metavar="N",
        help=f"Step N から再開する（前段は中間ファイルを再利用。{FIRST_STEP}〜{LAST_STEP}）",
    )
    run_p.add_argument(
        "--only-step",
        type=_step_number,
        default=None,
        metavar="N",
        help=f"Step N だけ実行する（{FIRST_STEP}〜{LAST_STEP}。--from-step とは併用不可）",
    )
    run_p.add_argument("--dry-run", action="store_true", help="Step 6 まで実行し、書き出しを行わない")
    run_p.add_argument("--preview-only", action="store_true", help="Step 8 のプレビューだけ生成する")
    run_p.add_argument(
        "--silence-db",
        type=_float_value,
        default=None,
        metavar="DB",
        help="silencedetect のしきい値（既定: 設定ファイルの値／-32）。負の数は --silence-db=-30 の形で",
    )
    run_p.add_argument(
        "--silence-dur",
        type=_positive_float,
        default=None,
        metavar="SEC",
        help="silencedetect の最短無音長（既定: 設定ファイルの値／0.12）",
    )
    run_p.add_argument("--episode-id", default=None, help="エピソードID（既定: 入力ファイル名の stem）")
    run_p.add_argument(
        "--stub-llm",
        default=None,
        metavar="PATH",
        help='LLM の代わりに使う JSON（{"highlight": {...}, "metadata": {...}, "titles": {...}}）',
    )
    run_p.add_argument(
        "--force-transcribe", action="store_true", help="キャッシュを無視して文字起こしをやり直す"
    )
    run_p.set_defaults(_handler=cmd_run, _parser=run_p)

    # ----- doctor -----
    doctor_p = subparsers.add_parser(
        "doctor",
        parents=[common],
        help="実行環境を確認する",
        description="ffmpeg・ライブラリ・APIキーが揃っているかを確認する。",
    )
    doctor_p.add_argument("--config", default=default_config, help=f"設定ファイル（既定: {default_config}）")
    doctor_p.set_defaults(_handler=cmd_doctor, _parser=doctor_p)

    # ----- transcribe -----
    tr_p = subparsers.add_parser(
        "transcribe",
        parents=[common],
        help="文字起こしだけ実行する（Step 1・2）",
        description="音声抽出と文字起こしだけを行い、transcript.json のパスを表示する。",
    )
    tr_p.add_argument("input", help="入力の動画ファイル")
    tr_p.add_argument("--config", default=default_config, help=f"設定ファイル（既定: {default_config}）")
    tr_p.add_argument("--work", default=DEFAULT_WORK_DIR, help=f"中間ファイルの置き場（既定: {DEFAULT_WORK_DIR}/）")
    tr_p.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"成果物の置き場（既定: {DEFAULT_OUT_DIR}/）")
    tr_p.add_argument("--episode-id", default=None, help="エピソードID（既定: 入力ファイル名の stem）")
    tr_p.add_argument(
        "--force-transcribe", action="store_true", help="キャッシュを無視して文字起こしをやり直す"
    )
    tr_p.set_defaults(_handler=cmd_transcribe, _parser=tr_p)

    # ----- titles -----
    ti_p = subparsers.add_parser(
        "titles",
        parents=[common],
        help="タイトル候補だけ作り直す（Step 6-b）",
        description="work/<ep>/metadata.json と highlight.json から titles.md を作り直す。",
    )
    ti_p.add_argument("episode_id", metavar="episode-id", help="エピソードID")
    ti_p.add_argument("--config", default=default_config, help=f"設定ファイル（既定: {default_config}）")
    ti_p.add_argument("--out", default=DEFAULT_OUT_DIR, help=f"成果物の置き場（既定: {DEFAULT_OUT_DIR}/）")
    ti_p.add_argument("--work", default=DEFAULT_WORK_DIR, help=f"中間ファイルの置き場（既定: {DEFAULT_WORK_DIR}/）")
    ti_p.add_argument(
        "--stub-llm",
        default=None,
        metavar="PATH",
        help='LLM の代わりに使う JSON（{"titles": {...}} を含むもの）',
    )
    ti_p.set_defaults(_handler=cmd_titles, _parser=ti_p)

    return parser


def _log_level(args: argparse.Namespace) -> int:
    """-v / -q からログレベルを決める。"""
    if getattr(args, "quiet", False):
        return logging.WARNING
    if int(getattr(args, "verbose", 0) or 0) >= 1:
        return logging.DEBUG
    return logging.INFO


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI の入口。終了コードは 0=成功 / 1=想定内の失敗 / 2=使い方の誤り・想定外 / 130=中断。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(_log_level(args))

    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "_handler", None)
    if handler is None:  # required=True なのでここには来ないが、念のため
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(handler(args))
    except RadioCutterError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("中断しました（Ctrl-C）。中間ファイルは残してあります。", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception:
        traceback.print_exc()
        print(
            "想定外のエラーで停止しました。上のトレースバックを添えて報告してください。",
            file=sys.stderr,
        )
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
