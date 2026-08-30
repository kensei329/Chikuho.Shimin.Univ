"""SPEC 3章「全体パイプライン」と 7章の再開オプション（--from-step / --only-step / --dry-run / --preview-only）。

方針:
- 実行するステップの決定（plan_steps）と、実際に回す処理（run_steps）を分けて持つ。
  どのステップを飛ばしたかをテストから確かめられるようにするため。
- 実行しないステップの結果は `load()` で中間ファイルから読む。読めなければ
  「どのコマンドを先に流せばよいか」を添えた MissingArtifactError で止める（勝手に作り直さない）。
- LLM クライアントは Step 5・6 に入るまで作らない。APIキーが無くても Step 1〜4 は回せるようにするため。
- decisions.json は Step 4 以降に踏み込んだら必ず書く。途中で落ちても「何が起きたか」を残す。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from .context import RunContext
from .decisions import build_decisions, decisions_path, now_iso, write_decisions
from .errors import LlmError, MissingArtifactError, RadioCutterError
from .llm.client import LlmClient, build_client
from .logging_util import fmt_duration, get_logger, step_timer
from .models import (
    AnchorResult,
    CutPoint,
    HighlightResult,
    MetadataResult,
    RenderResult,
    Transcript,
)
from .steps import s1_extract_audio as s1
from .steps import s2_transcribe as s2
from .steps import s3_find_anchors as s3
from .steps import s4_refine_cuts as s4
from .steps import s5_pick_highlight as s5
from .steps import s6_metadata as s6
from .steps import s7_render as s7
from .steps import s8_preview as s8
from .util.ffmpeg import MediaInfo

logger = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# ステップ表
# ---------------------------------------------------------------------------

FIRST_STEP: int = 1
LAST_STEP: int = 8

#: --dry-run で実行する最後のステップ（SPEC 7章「Step 6まで実行し、書き出しを行わない」）
DRY_RUN_LAST_STEP: int = 6

#: --preview-only が実行するステップ
PREVIEW_STEP: int = 8

#: このステップ以降に踏み込んだら decisions.json を必ず書く
DECISIONS_FROM_STEP: int = 4

#: ステップ番号 -> ステップモジュール
STEP_MODULES: dict[int, Any] = {m.STEP: m for m in (s1, s2, s3, s4, s5, s6, s7, s8)}

#: LLM を使うステップ（ここに入るまでクライアントを作らない）
LLM_STEPS: frozenset[int] = frozenset({s5.STEP, s6.STEP})


def step_label(step: int) -> str:
    """ログ用の "3（アンカー検出）" のような表記。"""
    module = STEP_MODULES.get(int(step))
    return f"{step}（{module.NAME}）" if module is not None else str(step)


def _check_range(label: str, value: int) -> int:
    """ステップ番号が 1〜8 に収まっているか確かめる。"""
    number = int(value)
    if not FIRST_STEP <= number <= LAST_STEP:
        raise RadioCutterError(
            f"{label} は {FIRST_STEP}〜{LAST_STEP} で指定してください（実際: {number}）。"
        )
    return number


# ---------------------------------------------------------------------------
# 実行計画
# ---------------------------------------------------------------------------


def plan_steps(
    *,
    from_step: int = FIRST_STEP,
    only_step: int | None = None,
    dry_run: bool = False,
    preview_only: bool = False,
) -> list[int]:
    """実行するステップ番号を並び順で返す（SPEC 7章）。

    矛盾する指定（--from-step と --only-step の併用など）はここで止める。
    走り出してから「実は何も実行しない」と分かるより、先に言った方が親切なため。
    """
    from_step = _check_range("--from-step", from_step)
    if only_step is not None:
        only_step = _check_range("--only-step", only_step)

    if only_step is not None and from_step != FIRST_STEP:
        raise RadioCutterError(
            "--from-step と --only-step は同時に指定できません。どちらか一方にしてください。"
        )
    if dry_run and preview_only:
        raise RadioCutterError(
            "--dry-run と --preview-only は同時に指定できません"
            f"（--preview-only は Step {PREVIEW_STEP} の書き出しを行います）。"
        )

    if preview_only:
        if only_step is not None and only_step != PREVIEW_STEP:
            raise RadioCutterError(
                f"--preview-only は Step {PREVIEW_STEP} だけを実行します。"
                f"--only-step {only_step} とは併用できません。"
            )
        return [PREVIEW_STEP]

    if only_step is not None:
        if dry_run and only_step > DRY_RUN_LAST_STEP:
            raise RadioCutterError(
                f"--dry-run は Step {DRY_RUN_LAST_STEP} までです。"
                f"--only-step {only_step} とは併用できません。"
            )
        return [only_step]

    last = DRY_RUN_LAST_STEP if dry_run else LAST_STEP
    if from_step > last:
        raise RadioCutterError(
            f"--dry-run は Step {DRY_RUN_LAST_STEP} までなので、"
            f"--from-step {from_step} では実行するステップがありません。"
        )
    return list(range(from_step, last + 1))


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """1回の実行で得られたもの全部。CLI とテストはここだけ見ればよい。"""

    ctx: RunContext
    media: MediaInfo | None = None
    transcript: Transcript | None = None
    anchors: dict[str, AnchorResult] | None = None
    cuts: dict[str, CutPoint] | None = None
    highlight: HighlightResult | None = None
    metadata: MetadataResult | None = None
    render: RenderResult | None = None
    previews: list[Path] = field(default_factory=list)
    decisions: dict | None = None
    timings: dict[int, float] = field(default_factory=dict)

    #: 実際に着手したステップ番号（load で済ませたものは含まない）
    executed_steps: list[int] = field(default_factory=list)

    #: 書き出した／今そこにある成果物・中間ファイル
    artifacts: list[Path] = field(default_factory=list)

    @property
    def total_elapsed(self) -> float:
        """全ステップの所要秒数の合計。"""
        return sum(self.timings.values())


# ---------------------------------------------------------------------------
# 実行本体
# ---------------------------------------------------------------------------


class _Pipeline:
    """ステップの実行と、前段の中間ファイル読み込みをまとめて持つ。

    「実行したステップの結果」と「load で読んだ結果」を同じキャッシュに入れることで、
    後段のステップからは両者の区別が要らなくなる。
    """

    def __init__(self, ctx: RunContext, *, llm: LlmClient | None = None) -> None:
        self.ctx = ctx
        self.result = PipelineResult(ctx=ctx)
        self._highlight_failed = False
        self._llm = llm
        self._reached: list[int] = []

    # ----- 計測 -----

    def _timed(self, module: Any, call: Callable[[], T]) -> T:
        """ステップを step_timer で囲み、所要秒数を timings に残す（失敗しても残す）。"""
        holder: dict[str, Any] = {}
        try:
            with step_timer(module.STEP, module.NAME) as info:
                holder["info"] = info
                return call()
        finally:
            info = holder.get("info")
            if info is not None:
                self.result.timings[int(module.STEP)] = float(info.get("elapsed", 0.0))

    # ----- 中間ファイルの読み込み -----

    def _hint(self, message: str, step: int) -> str:
        """「先にこれを流してください」の案内を足す（既に書いてあれば足さない）。"""
        if f"--only-step {step}" in message or f"--from-step {step}" in message:
            return message
        return (
            f"{message}\n"
            f"  先に `radio-cutter run {self.ctx.input_path} --only-step {step}` を実行してください。"
        )

    def _load(self, module: Any) -> Any:
        """前段の中間ファイルを読む。無ければ案内付きの MissingArtifactError。"""
        logger.info("Step %s の中間ファイルを読み込みます。", step_label(module.STEP))
        try:
            return module.load(self.ctx)
        except MissingArtifactError as exc:
            raise MissingArtifactError(self._hint(str(exc), int(module.STEP))) from exc

    def _load_optional(self, module: Any) -> Any:
        """あれば読む、無ければ None。decisions.json の穴埋めなど「無くても進める」用途に使う。"""
        try:
            return module.load(self.ctx)
        except MissingArtifactError as exc:
            logger.debug("Step %s の中間ファイルは読めませんでした: %s", module.STEP, exc)
            return None

    # ----- 各ステップの入力 -----

    def media(self) -> MediaInfo | None:
        """入力の素性（probe.json）。無くても後段は ffprobe で取り直せるので必須にしない。"""
        if self.result.media is None:
            self.result.media = self._load_optional(s1)
        return self.result.media

    def total_duration(self) -> float | None:
        """入力動画の総尺。probe.json が無ければ None（後段が transcript から補う）。"""
        media = self.media()
        if media is None:
            return None
        duration = float(media.duration)
        return duration if duration > 0 else None

    def transcript(self) -> Transcript:
        if self.result.transcript is None:
            self.result.transcript = self._load(s2)
        return self.result.transcript

    def anchors(self) -> dict[str, AnchorResult]:
        if self.result.anchors is None:
            self.result.anchors = self._load(s3)
        return self.result.anchors

    def cuts(self) -> dict[str, CutPoint]:
        if self.result.cuts is None:
            self.result.cuts = self._load(s4)
        return self.result.cuts

    def highlight(self, *, required: bool = True) -> HighlightResult | None:
        if self.result.highlight is None:
            self.result.highlight = self._load(s5) if required else self._load_optional(s5)
        return self.result.highlight

    def llm(self) -> LlmClient:
        """LLM クライアントを遅延生成する。Step 5・6 に入るまで作らない（APIキー不要で回せるように）。"""
        if self._llm is None:
            cfg = self.ctx.config.llm
            logger.info("LLM クライアントを作ります（provider=%s / model=%s）。", cfg.provider, cfg.model)
            self._llm = build_client(cfg)
        return self._llm

    # ----- 各ステップ -----

    def _step1(self) -> None:
        self.result.media = self._timed(s1, lambda: s1.run(self.ctx))

    def _step2(self) -> None:
        media = self.media()
        self.result.transcript = self._timed(s2, lambda: s2.run(self.ctx, media))

    def _step3(self) -> None:
        transcript = self.transcript()
        self.result.anchors = self._timed(s3, lambda: s3.run(self.ctx, transcript))

    def _step4(self) -> None:
        anchors = self.anchors()
        self.result.cuts = self._timed(s4, lambda: s4.run(self.ctx, anchors))

    def _step5(self) -> None:
        transcript = self.transcript()
        cuts = self.cuts()
        total = self.total_duration()
        llm = self.llm()
        try:
            self.result.highlight = self._timed(
                s5, lambda: s5.run(self.ctx, transcript, cuts, llm, total_duration=total)
            )
        except LlmError as exc:
            # SPEC 9章: LLM が答えを返せなくても、そのステップだけ落として書き出しは続ける。
            # 60分ぶんのエンコードを API の一時的な不調で捨てないため。
            # 候補が全部本編の外だった場合（HighlightError）はここで拾わず、仕様どおり止まる。
            message = (
                f"Step 5（ハイライト選定）に失敗したため、ハイライト無しで続けます: {exc}"
            )
            self.ctx.warn(message)
            logger.error("%s", message)
            logger.warning(
                "01_highlight.mp4 は作られず、final.mp4 は本編以降だけの連結になります。"
                "あとから `--from-step 5` でやり直せます。"
            )
            self.result.highlight = None
            self._highlight_failed = True

    def _step6(self) -> None:
        if self._highlight_failed:
            message = "Step 6（メタデータ生成）は Step 5 が失敗したため飛ばしました。"
            self.ctx.warn(message)
            logger.warning("%s", message)
            return
        transcript = self.transcript()
        cuts = self.cuts()
        highlight = self.highlight()
        total = self.total_duration()
        llm = self.llm()
        self.result.metadata = self._timed(
            s6, lambda: s6.run(self.ctx, transcript, cuts, highlight, llm, total_duration=total)
        )

    def _step7(self) -> None:
        cuts = self.cuts()
        # Step 5 がこの実行で落ちたときだけハイライト無しで進む。
        # 中間ファイルが最初から無い（--from-step 7 をいきなり叩いた）場合は、
        # 黙って欠けたまま書き出さずに「先に Step 5 を回して」と言って止まる。
        highlight = None if self._highlight_failed else self.highlight()
        media = self.media()
        self.result.render = self._timed(s7, lambda: s7.run(self.ctx, cuts, highlight, media))

    def _step8(self) -> None:
        # Step 8 は Step 7 の出力に依存しない（元動画から直接切り出す）。
        # そのため --preview-only は Step 4 の cuts.json と Step 5 の highlight.json だけで動く。
        cuts = self.cuts()
        highlight = self.highlight(required=False)
        if highlight is None:
            logger.warning(
                "ハイライトの中間ファイルが無いため、カット点のプレビューだけを作ります"
                "（Step 5 を実行すると highlight_in / highlight_out も作られます）。"
            )
        media = self.media()
        self.result.previews = self._timed(s8, lambda: s8.run(self.ctx, cuts, highlight, media))

    def _dispatch(self, step: int) -> None:
        handlers: dict[int, Callable[[], None]] = {
            1: self._step1,
            2: self._step2,
            3: self._step3,
            4: self._step4,
            5: self._step5,
            6: self._step6,
            7: self._step7,
            8: self._step8,
        }
        handlers[step]()

    # ----- decisions.json -----

    def _should_write_decisions(self) -> bool:
        """Step 4 以降に着手していれば書く（--dry-run でも、途中で落ちても書く）。"""
        return any(step >= DECISIONS_FROM_STEP for step in self._reached)

    def _write_decisions(self) -> None:
        """decisions.json を組み立てて out/<episode_id>/ に書く。"""
        anchors = self.result.anchors if self.result.anchors is not None else self._load_optional(s3)
        cuts = self.result.cuts if self.result.cuts is not None else self._load_optional(s4)
        highlight = (
            self.result.highlight if self.result.highlight is not None else self._load_optional(s5)
        )
        # 書き出しを回していない実行では、前回の render.json を「実測」として載せない。
        # カット点を変えて --dry-run し直した直後だと、前回の実尺は今回の anchors と
        # 食い違う。decisions.json の中で辻褄の合わない数字が並ぶより、
        # 今回のカット点から出した想定値を載せるほうが読み手を誤らせない。
        render = self.result.render
        if render is None and int(s7.STEP) in self.result.timings:
            render = self._load_optional(s7)
        if render is None and self._load_optional(s7) is not None:
            logger.debug(
                "前回の書き出し結果はありますが、今回 Step %s を回していないので"
                "decisions.json の尺は想定値にします。",
                s7.STEP,
            )

        payload = build_decisions(
            self.ctx,
            media=self.media(),
            anchors=anchors,
            cuts=cuts,
            highlight=highlight,
            render=render,
            generated_at=now_iso(),
        )
        self.result.decisions = payload
        write_decisions(self.ctx, payload)

    # ----- 成果物の一覧 -----

    def artifact_paths(self) -> list[Path]:
        """今 work/ と out/ にある中間ファイル・成果物を、ステップ順に並べて返す。"""
        found: list[Path] = []

        def add(path: Path | str | None) -> None:
            if path is None:
                return
            p = Path(path)
            if p.exists() and p not in found:
                found.append(p)

        for step in sorted(STEP_MODULES):
            for name in getattr(STEP_MODULES[step], "OUTPUTS", ()):
                add(self.ctx.work_path(name))

        if self.result.render is not None:
            for path in self.result.render.files.values():
                add(path)
        else:
            add(s7.final_path(self.ctx))

        add(s6.description_path(self.ctx))
        add(s6.titles_path(self.ctx))

        previews = self.result.previews or s8.load(self.ctx)
        for path in previews:
            add(path)

        add(decisions_path(self.ctx))
        return found

    def _log_artifacts(self) -> None:
        """「どのファイルを作ったか」をログに出す（SPEC 11章「中間ファイルは消さない」の受け皿）。"""
        paths = self.artifact_paths()
        self.result.artifacts = paths
        if not paths:
            logger.info("生成されたファイルはありません。")
            return
        logger.info("生成物（work / out に今あるもの）: %d 件", len(paths))
        for path in paths:
            logger.info("  - %s", path)

    def _log_timings(self) -> None:
        """ステップごとの所要秒数をまとめて出す（文字起こしに何分かかったかを見るため）。"""
        if not self.result.timings:
            return
        parts = [
            f"Step {step} {STEP_MODULES[step].NAME} {fmt_duration(self.result.timings[step])}"
            for step in sorted(self.result.timings)
            if step in STEP_MODULES
        ]
        logger.info("所要時間: %s（合計 %s）", " / ".join(parts), fmt_duration(self.result.total_elapsed))

    def _warn_missing_api_key(self, plan: Sequence[int]) -> None:
        """LLM を使うステップが計画に入っているのに APIキーが無ければ、走り出す前に言う。

        文字起こしに何十分もかけたあとで Step 5 が落ちるのが一番もったいないため。
        """
        if self._llm is not None:
            return
        llm_steps = [step for step in plan if step in LLM_STEPS]
        if not llm_steps:
            return
        env_name = self.ctx.config.llm.api_key_env
        if not env_name or os.environ.get(env_name):
            return
        logger.warning(
            "環境変数 %s が未設定です。Step %s に入った時点で LLM 呼び出しに失敗します"
            "（--stub-llm を使うか、APIキーを設定してください）。",
            env_name,
            "・".join(str(step) for step in llm_steps),
        )

    # ----- 実行 -----

    def execute(self, steps: Sequence[int], *, dry_run: bool = False) -> PipelineResult:
        """指定されたステップを順に実行する。"""
        plan = [_check_range("ステップ番号", step) for step in steps]
        ctx = self.ctx
        ctx.ensure_dirs()

        logger.info("エピソード: %s", ctx.episode_id)
        logger.info("入力: %s", ctx.input_path)
        logger.info("中間ファイル: %s / 成果物: %s", ctx.work_dir, ctx.out_dir)
        if plan:
            logger.info("実行するステップ: %s", " → ".join(step_label(step) for step in plan))
        else:
            logger.warning("実行するステップがありません。")
        if dry_run:
            logger.info(
                "--dry-run のため Step %s と Step %s は行いません（書き出しは `--from-step 7` で）。",
                step_label(7),
                step_label(8),
            )
        self._warn_missing_api_key(plan)

        started = time.perf_counter()
        try:
            for step in plan:
                self._reached.append(step)
                self.result.executed_steps.append(step)
                self._dispatch(step)
        finally:
            if self._should_write_decisions():
                try:
                    self._write_decisions()
                except Exception as exc:  # 元の失敗を隠さないよう、ここでは記録だけして続ける
                    logger.error("decisions.json を書けませんでした: %s", exc)
            try:
                self._log_timings()
                self._log_artifacts()
            except Exception as exc:  # 一覧の作成で本体の結果を潰さない
                logger.error("生成物の一覧を作れませんでした: %s", exc)
            logger.info("全体の所要時間: %s", fmt_duration(time.perf_counter() - started))

        return self.result


def run_steps(
    ctx: RunContext,
    steps: Sequence[int],
    *,
    dry_run: bool = False,
    llm: LlmClient | None = None,
) -> PipelineResult:
    """任意のステップ列を実行する（`transcribe` サブコマンドのように途中だけ回したいとき用）。"""
    if dry_run:
        ctx.dry_run = True
    return _Pipeline(ctx, llm=llm).execute(steps, dry_run=dry_run)


def run_pipeline(
    ctx: RunContext,
    *,
    from_step: int = FIRST_STEP,
    only_step: int | None = None,
    dry_run: bool = False,
    preview_only: bool = False,
    llm: LlmClient | None = None,
) -> PipelineResult:
    """SPEC 3章のパイプラインを回す。

    `from_step` より前のステップは中間ファイルから読み、`only_step` はそのステップだけ実行する。
    `dry_run` は Step 6 まで、`preview_only` は Step 8 だけ。
    `llm` を渡さなければ Step 5・6 に入った時点で `build_client()` で作る。
    """
    plan = plan_steps(
        from_step=from_step,
        only_step=only_step,
        dry_run=dry_run,
        preview_only=preview_only,
    )
    return run_steps(ctx, plan, dry_run=dry_run, llm=llm)
