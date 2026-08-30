"""SPEC Step 5「ハイライト選定」。本編からフックに使う区間を LLM に選ばせ、3段スナップで整える。

方針（SPEC Step 5・9章）：
- LLM に渡すのは本編区間（cut_time_A 〜 cut_time_B）の文字起こしだけ。秒数は元動画の絶対秒。
- LLM が返す秒数は文の途中で切れていることが多いので、**単語境界 → 文境界 → 無音の谷** の順に
  寄せ、超過分は末尾の文を落として詰める。この3段スナップを飛ばすと語尾が千切れて視聴に耐えない。
- 候補は score 降順に見て、本編の範囲外なら破棄して次点。全滅なら HighlightError で止める。
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import SILENCE_BACKOFF_SEC
from ..context import RunContext
from ..errors import HighlightError, LlmError, MissingArtifactError
from ..llm.client import LlmClient, load_prompt, render_prompt
from ..llm.schemas import HIGHLIGHT_SCHEMA
from ..logging_util import get_logger
from ..models import (
    CutPoint,
    HighlightCandidate,
    HighlightResult,
    LlmCallRecord,
    Transcript,
    Word,
    r3,
    read_json,
    write_json,
)
from ..util.ffmpeg import detect_silences
from ..util.timeline import (
    SENTENCE_TERMINATORS,
    clamp,
    drop_last_sentence,
    expand_to_sentence,
    fmt_timestamp,
    resolve_segment_bounds,
    snap_to_word_boundary,
)
from .s1_extract_audio import audio_path

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 5
NAME: str = "ハイライト選定"

#: work/<episode_id>/ に書く中間ファイル
HIGHLIGHT_FILE = "highlight.json"
OUTPUTS: tuple[str, ...] = (HIGHLIGHT_FILE,)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

#: decisions.json の llm_calls に載るステップ名／使うプロンプト（llm/prompts/highlight.md）
LLM_STEP = "highlight"
PROMPT_NAME = "highlight"

#: プロンプトに渡す文字起こしの1行の長さ（秒）。細かすぎると行数が増えてトークンを食い、
#: 長すぎると LLM が秒数を指しづらくなるので 10〜20 秒に畳む。
PROMPT_LINE_MIN_SEC = 10.0
PROMPT_LINE_MAX_SEC = 20.0

#: 無音スナップ（SPEC Step 5 後処理3）で silencedetect をかける前後の幅
SILENCE_WINDOW_SEC = 1.5

#: 「この時刻の無音」とみなす許容ずれ。単語境界と silencedetect の境界は数十msずれるため、
#: 厳密な不等号で拾うと直近の谷を取りこぼして遠い谷に飛んでしまう。
SILENCE_MATCH_JITTER_SEC = 0.15

#: 無音スナップで動かしてよい最大量。これを超える移動は「隣の文へ食い込んでいる」ので採らない。
MAX_SILENCE_SHIFT_SEC = 1.0

#: 候補が本編の範囲をこの秒数まではみ出しているのは丸め誤差とみなし、破棄せずクランプする。
BOUNDS_TOLERANCE_SEC = 1.0

#: 末尾の文を落とす繰り返しの上限（無限ループ防止）
MAX_TRIM_ITERATIONS = 64


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def highlight_path(ctx: RunContext) -> Path:
    """work/<episode_id>/highlight.json の場所。"""
    return ctx.work_path(HIGHLIGHT_FILE)


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def _fmt_sec(value: float) -> str:
    """プロンプトに埋める秒数。30.0 は "30" と書きたいので %g で整える。"""
    return f"{float(value):g}"


def _ends_sentence(word: Word) -> bool:
    """この単語で文が終わるか（「。」「？」「！」を含むか）。"""
    return any(ch in SENTENCE_TERMINATORS for ch in (word.word or ""))


def words_in_bounds(words: Sequence[Word], start: float, end: float) -> list[Word]:
    """本編区間に開始時刻が入る単語だけを返す。

    スナップの対象をこの範囲に限ることで、文境界への拡張が本編の外へ出ないようにする。
    """
    lo = float(start)
    hi = float(end)
    return [w for w in words if lo <= float(w.start) < hi]


# ---------------------------------------------------------------------------
# プロンプト用の文字起こし整形
# ---------------------------------------------------------------------------


def format_transcript_for_prompt(
    words: Sequence[Word],
    *,
    min_line_sec: float = PROMPT_LINE_MIN_SEC,
    max_line_sec: float = PROMPT_LINE_MAX_SEC,
) -> str:
    """本編の単語を「[秒数] テキスト」の行に畳む（SPEC Step 5・トークン節約のため）。

    1行が min_line_sec を超えたあと最初に来る文末で改行し、max_line_sec を超えたら
    文末を待たずに改行する。行頭の秒数は元動画の絶対秒（LLM にはこの秒で返させる）。
    """
    lines: list[str] = []
    buffer: list[str] = []
    line_start: float | None = None

    for word in words:
        text = (word.word or "").strip()
        if line_start is None:
            line_start = float(word.start)
        if text:
            buffer.append(text)
        if not buffer:
            continue
        span = float(word.end) - line_start
        if (span >= float(min_line_sec) and _ends_sentence(word)) or span >= float(max_line_sec):
            lines.append(f"[{line_start:.1f}] {''.join(buffer)}")
            buffer = []
            line_start = None

    if buffer and line_start is not None:
        lines.append(f"[{line_start:.1f}] {''.join(buffer)}")
    return "\n".join(lines)


def build_prompt(ctx: RunContext, words: Sequence[Word], bounds: tuple[float, float]) -> str:
    """llm/prompts/highlight.md を読み、本編の文字起こしと尺の条件を埋める。"""
    hl = ctx.config.highlight
    variables: dict[str, Any] = {
        "channel": ctx.config.channel or "（名称未設定のチャンネル）",
        "num_candidates": max(1, int(hl.num_candidates)),
        "target_duration_sec": _fmt_sec(hl.target_duration_sec),
        "min_duration_sec": _fmt_sec(hl.min_duration_sec),
        "max_duration_sec": _fmt_sec(hl.max_duration_sec),
        "main_start": f"{bounds[0]:.1f}",
        "main_end": f"{bounds[1]:.1f}",
        "transcript": format_transcript_for_prompt(words),
    }
    return render_prompt(load_prompt(PROMPT_NAME), variables)


# ---------------------------------------------------------------------------
# 候補の取り出しと範囲チェック
# ---------------------------------------------------------------------------


def parse_candidates(payload: Mapping[str, Any]) -> list[HighlightCandidate]:
    """LLM の応答（スキーマ検証済み）から候補を取り出し、score の降順に並べる。

    同点は LLM が返した順を保つ（安定ソート）。
    """
    raw = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list) or not raw:
        raise HighlightError(
            "LLM がハイライト候補を1件も返しませんでした。"
            "llm/prompts/highlight.md の出力形式の指示を確認してください。"
        )
    candidates: list[HighlightCandidate] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise HighlightError(f"ハイライト候補 {index} 件目がオブジェクトではありません: {item!r}")
        try:
            candidates.append(HighlightCandidate.from_dict(item))
        except (KeyError, TypeError, ValueError) as exc:
            raise HighlightError(f"ハイライト候補 {index} 件目の内容が不正です: {exc}") from exc
    return sorted(candidates, key=lambda c: float(c.score), reverse=True)


def check_within_bounds(
    candidate: HighlightCandidate,
    bounds: tuple[float, float],
    *,
    tolerance: float = BOUNDS_TOLERANCE_SEC,
) -> str | None:
    """候補が本編の範囲に収まっているかを調べる。収まっていれば None、駄目なら理由を返す。

    SPEC 9章「ハイライト候補が本編の範囲外 → その候補を破棄し、次点を採用」の判定。
    tolerance 以内のはみ出しは LLM の丸め誤差とみなし、破棄せずクランプに任せる。
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    start = float(candidate.start)
    end = float(candidate.end)

    if not (math.isfinite(start) and math.isfinite(end)):
        return f"開始・終了が数値として不正です（start={candidate.start!r}, end={candidate.end!r}）。"
    if end <= start:
        return f"終了 {r3(end)}秒 が開始 {r3(start)}秒 以前になっています。"
    if start < lo - float(tolerance):
        return (
            f"開始 {r3(start)}秒（{fmt_timestamp(start)}）が本編の開始"
            f" {r3(lo)}秒（{fmt_timestamp(lo)}）より前です。"
        )
    if end > hi + float(tolerance):
        return (
            f"終了 {r3(end)}秒（{fmt_timestamp(end)}）が本編の終了"
            f" {r3(hi)}秒（{fmt_timestamp(hi)}）より後です。"
        )
    return None


# ---------------------------------------------------------------------------
# 無音スナップ（SPEC Step 5 後処理3）
# ---------------------------------------------------------------------------


def _accepts_kind(lookup: Callable[..., Any]) -> bool:
    """silence_lookup が `kind` キーワード（start / end の別）を受け取れるか。

    契約上の型は `Callable[[float], tuple[float, bool]]` なので、単純な1引数の関数も
    そのまま渡せる。start と end で寄せ方が違うため、受け取れる実装にだけ kind を渡す。
    """
    try:
        signature = inspect.signature(lookup)
    except (TypeError, ValueError):  # 組み込み関数などは内省できない
        return False
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "kind" and param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


def _apply_silence(
    lookup: Callable[..., tuple[float, bool]] | None, t: float, kind: str
) -> tuple[float, bool]:
    """silence_lookup を呼び、(寄せた時刻, 見つかったか) を返す。返り値が壊れていれば無視する。"""
    value = float(t)
    if lookup is None:
        return (value, False)
    try:
        result = lookup(value, kind=kind) if _accepts_kind(lookup) else lookup(value)
    except TypeError as exc:  # 想定と違うシグネチャ。ここで落とさず元の時刻を使う。
        logger.warning("無音スナップの関数を呼べませんでした（%s）。元の時刻を使います。", exc)
        return (value, False)

    if not isinstance(result, tuple) or len(result) != 2:
        logger.warning("無音スナップの返り値が (時刻, 見つかったか) ではありません: %r", result)
        return (value, False)
    snapped, found = result
    try:
        snapped_value = float(snapped)
    except (TypeError, ValueError):
        logger.warning("無音スナップが数値でない時刻を返しました: %r", snapped)
        return (value, False)
    if not math.isfinite(snapped_value):
        return (value, False)
    return (snapped_value, bool(found))


def make_silence_lookup(
    wav_path: str | Path,
    *,
    noise_db: float,
    min_duration: float,
    window: float = SILENCE_WINDOW_SEC,
    backoff: float = SILENCE_BACKOFF_SEC,
    max_shift: float = MAX_SILENCE_SHIFT_SEC,
) -> Callable[..., tuple[float, bool]]:
    """Step 4 と同じ silencedetect を [t-window, t+window] にかける lookup を作る。

    start は「t より前の最後の無音の終了 - backoff」、
    end は「t より後の最初の無音の開始 + backoff」に寄せる。
    end だけ無音の始まりより後ろへ置くのは、語尾を切り落とさないため。
    同じ時刻を何度も引く（尺の詰め直しで再評価する）ので結果はメモ化する。
    """
    wav = Path(wav_path)
    cache: dict[tuple[str, float], tuple[float, bool]] = {}

    def lookup(t: float, *, kind: str = "start") -> tuple[float, bool]:
        """t を無音の谷へ寄せた時刻と、無音が見つかったかどうかを返す。"""
        value = float(t)
        direction = "end" if kind == "end" else "start"
        key = (direction, round(value, 3))
        cached = cache.get(key)
        if cached is not None:
            return cached

        win_start = max(0.0, value - float(window))
        win_end = value + float(window)
        spans = detect_silences(
            wav,
            start=win_start,
            end=win_end,
            noise_db=float(noise_db),
            min_duration=float(min_duration),
        )

        result = (value, False)
        if spans:
            if direction == "end":
                after = [s for s in spans if s[0] >= value - SILENCE_MATCH_JITTER_SEC]
                picked = min(after, key=lambda s: s[0]) if after else None
                snapped = (picked[0] + float(backoff)) if picked else None
            else:
                before = [s for s in spans if s[1] <= value + SILENCE_MATCH_JITTER_SEC]
                picked = max(before, key=lambda s: s[1]) if before else None
                snapped = (picked[1] - float(backoff)) if picked else None

            if snapped is None:
                logger.debug("%.3f秒（%s）の近くに寄せられる無音がありませんでした。", value, direction)
            elif abs(snapped - value) > float(max_shift):
                logger.debug(
                    "%.3f秒（%s）の無音が %.3f秒 離れているため寄せませんでした。",
                    value,
                    direction,
                    abs(snapped - value),
                )
            else:
                result = (max(0.0, snapped), True)

        cache[key] = result
        return result

    return lookup


# ---------------------------------------------------------------------------
# 3段スナップ（SPEC Step 5 後処理1〜4）★ハイライトの品質はここで決まる
# ---------------------------------------------------------------------------


def snap_highlight(
    candidate: HighlightCandidate,
    words: Sequence[Word],
    *,
    bounds: tuple[float, float],
    max_duration: float,
    min_duration: float,
    silence_lookup: Callable[[float], tuple[float, bool]] | None = None,
) -> tuple[HighlightCandidate, bool, int]:
    """LLM の秒数を視聴に耐える区間へ整える。返り値は (スナップ後候補, 無音に寄せたか, 落とした文の数)。

    SPEC Step 5 の後処理をその順で行う:
      1) start / end を最寄りの単語境界へスナップ
      2) start はその単語が属する文の先頭へ、end は文末へ拡張
      3) silence_lookup があれば無音の谷へ寄せる
      4) max_duration を超えるなら末尾の文を1つ落として再計算（落とせなくなるまで）
    結果は必ず bounds の中に収める。min_duration は下回っても落とさない
    （短すぎることを警告するかは run() 側が決める）。
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        lo, hi = hi, lo

    limit = float(max_duration)
    if not math.isfinite(limit) or limit <= 0:
        limit = math.inf

    def _finish(start: float, end: float) -> HighlightCandidate:
        return HighlightCandidate(
            start=float(start),
            end=float(end),
            score=candidate.score,
            hook_line=candidate.hook_line,
            reason=candidate.reason,
        )

    if not words:
        logger.warning("本編区間に単語がないため、ハイライトのスナップを行いませんでした。")
        return (
            _finish(clamp(candidate.start, lo, hi), clamp(candidate.end, lo, hi)),
            False,
            0,
        )

    # 1) 単語境界へスナップ
    start = snap_to_word_boundary(words, float(candidate.start), kind="start")
    end = snap_to_word_boundary(words, float(candidate.end), kind="end")

    # 2) 文の先頭・末尾まで拡張
    start, end = expand_to_sentence(words, start, end)
    logger.debug(
        "スナップ: %.3f〜%.3f秒 → 単語・文境界 %.3f〜%.3f秒",
        candidate.start,
        candidate.end,
        start,
        end,
    )

    trimmed = 0
    silence_snapped = False
    final_start, final_end = clamp(start, lo, hi), clamp(end, lo, hi)

    for _ in range(MAX_TRIM_ITERATIONS + 1):
        # 3) 無音の谷へ寄せる（構造上の位置は start/end のまま保ち、寄せた値は毎回作り直す）
        snapped_start, found_start = _apply_silence(silence_lookup, start, "start")
        snapped_end, found_end = _apply_silence(silence_lookup, end, "end")
        current_start = clamp(snapped_start, lo, hi)
        current_end = clamp(snapped_end, lo, hi)
        if current_end <= current_start:
            # 無音へ寄せた結果つぶれた（窓が狭い・境界に張り付いた）ので寄せる前へ戻す。
            logger.debug("無音へ寄せると区間がつぶれるため、単語境界の位置を使います。")
            current_start = clamp(start, lo, hi)
            current_end = clamp(end, lo, hi)
            found_start = found_end = False

        final_start, final_end = current_start, current_end
        silence_snapped = bool(found_start or found_end)

        if (final_end - final_start) <= limit:
            break

        # 4) 尺が超過している。末尾の文を1つ落として再計算する。
        dropped = drop_last_sentence(words, start, end)
        if dropped is None:
            logger.warning(
                "ハイライトが %.1f秒 で上限 %.1f秒 を超えていますが、"
                "1つの文しか無いためこれ以上詰められません。",
                final_end - final_start,
                limit,
            )
            break
        start, end = dropped
        trimmed += 1
        logger.debug("尺が上限を超えたため末尾の文を落としました（%d 文目 / 新しい終端 %.3f秒）", trimmed, end)
    else:
        logger.warning(
            "末尾の文を落とす処理が上限 %d 回に達したため打ち切りました。", MAX_TRIM_ITERATIONS
        )

    if final_end <= final_start:
        logger.warning(
            "スナップの結果ハイライトの区間が空になりました（%.3f〜%.3f秒）。", final_start, final_end
        )

    if trimmed:
        logger.info(
            "ハイライトが上限 %.1f秒 を超えていたため、末尾の文を %d つ落としました。", limit, trimmed
        )
    if (final_end - final_start) < float(min_duration):
        logger.debug(
            "スナップ後の尺 %.1f秒 が下限 %.1f秒 を下回っています。",
            final_end - final_start,
            float(min_duration),
        )
    return (_finish(final_start, final_end), silence_snapped, trimmed)


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def _build_silence_lookup(ctx: RunContext) -> Callable[..., tuple[float, bool]] | None:
    """work/audio.wav があれば無音スナップ用の lookup を作る。無ければ None（スナップは省略）。"""
    wav = audio_path(ctx)
    if not wav.exists():
        message = (
            f"音声ファイルが無いため、ハイライトの無音スナップを省略しました: {wav}"
            "（Step 1 を実行すると語尾の切れ方が良くなります）"
        )
        logger.warning(message)
        ctx.warn(message)
        return None
    return make_silence_lookup(
        wav,
        noise_db=ctx.silence.noise_db,
        min_duration=ctx.silence.min_duration_sec,
    )


def _call_llm(ctx: RunContext, llm: LlmClient, prompt: str) -> dict[str, Any]:
    """LLM を呼び、decisions.json 用に呼び出し記録を残す。失敗も記録してから投げ直す。"""
    model = getattr(llm, "model", "") or ""
    try:
        response = llm.complete_json(step=LLM_STEP, prompt=prompt, schema=HIGHLIGHT_SCHEMA)
    except LlmError as exc:
        ctx.record_llm_call(LlmCallRecord(step=LLM_STEP, model=model, ok=False, error=str(exc)))
        raise

    ctx.record_llm_call(
        LlmCallRecord(
            step=LLM_STEP,
            model=model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            retries=response.retries,
            ok=True,
        )
    )
    logger.info(
        "ハイライト候補を LLM（%s）から受け取りました（入力 %d トークン / 再試行 %d 回）",
        model or "不明なモデル",
        response.input_tokens,
        response.retries,
    )
    return response.data


def run(
    ctx: RunContext,
    transcript: Transcript,
    cuts: Mapping[str, CutPoint],
    llm: LlmClient,
    *,
    total_duration: float | None = None,
) -> HighlightResult:
    """本編からハイライト区間を選び、3段スナップで整えて work/highlight.json に書く（SPEC Step 5）。"""
    hl = ctx.config.highlight
    segment_config = ctx.config.segment(hl.source_segment)

    duration = float(total_duration) if total_duration is not None else float(transcript.duration)
    if duration <= 0:
        all_words = transcript.words()
        duration = float(all_words[-1].end) if all_words else 0.0
        logger.warning(
            "文字起こしに総尺がありません。単語の末尾 %.3f秒 を総尺として使います。", duration
        )

    bounds = resolve_segment_bounds(segment_config, cuts, duration)
    main_words = words_in_bounds(transcript.words(), bounds[0], bounds[1])
    logger.info(
        "本編区間 %s〜%s（%.3f〜%.3f秒 / %.1f秒）から %d 単語を LLM に渡します。",
        fmt_timestamp(bounds[0]),
        fmt_timestamp(bounds[1]),
        bounds[0],
        bounds[1],
        bounds[1] - bounds[0],
        len(main_words),
    )
    if not main_words:
        raise HighlightError(
            f"本編区間（{r3(bounds[0])}〜{r3(bounds[1])}秒）に文字起こしの単語が1つもありません。\n"
            "  Step 3 のアンカー位置か Step 2 の文字起こしを確認してください。"
        )

    prompt = build_prompt(ctx, main_words, bounds)
    logger.debug("ハイライト用プロンプト: %d 文字", len(prompt))
    payload = _call_llm(ctx, llm, prompt)
    candidates = parse_candidates(payload)
    logger.info("ハイライト候補 %d 件をスコア降順で検査します。", len(candidates))

    silence_lookup = _build_silence_lookup(ctx)

    selected: HighlightCandidate | None = None
    source: HighlightCandidate | None = None
    silence_snapped = False
    trimmed = 0
    rejections: list[str] = []

    for index, candidate in enumerate(candidates, start=1):
        label = f"{index}件目（score {float(candidate.score):g} / {r3(candidate.start)}〜{r3(candidate.end)}秒）"
        reason = check_within_bounds(candidate, bounds)
        if reason is not None:
            rejections.append(f"{label}: {reason}")
            message = f"ハイライト候補 {label} は本編の範囲外なので破棄しました: {reason}"
            logger.warning(message)
            ctx.warn(message)
            continue

        snapped, found_silence, dropped = snap_highlight(
            candidate,
            main_words,
            bounds=bounds,
            max_duration=hl.max_duration_sec,
            min_duration=hl.min_duration_sec,
            silence_lookup=silence_lookup,
        )
        if snapped.duration <= 0:
            rejections.append(f"{label}: スナップした結果、区間が空になりました。")
            logger.warning("ハイライト候補 %s はスナップ後に区間が空になったので破棄しました。", label)
            continue

        selected = snapped
        source = candidate
        silence_snapped = found_silence
        trimmed = dropped
        break

    if selected is None or source is None:
        detail = "\n".join(f"  - {line}" for line in rejections) or "  - （候補がありませんでした）"
        raise HighlightError(
            f"ハイライト候補 {len(candidates)} 件すべてを採用できませんでした"
            f"（本編は {r3(bounds[0])}〜{r3(bounds[1])}秒）。\n"
            f"落ちた理由:\n{detail}\n"
            "  アンカーの検出位置（work/anchors.json）と、"
            "llm/prompts/highlight.md の「本編の範囲」の指示を確認してください。"
        )

    # SPEC Step 5 の「残り2つは decisions.json に残し、UI追加時の差し替え候補にする」。
    # 本編の外だったものも含めて全部残す。LLM が何を出してきたかは
    # プロンプトを直すときの手がかりになるし、破棄した理由は warnings 側に出ている。
    alternatives = [c for c in candidates if c is not source]
    result = HighlightResult(
        selected=selected,
        snapped_from=source,
        alternatives=alternatives,
        silence_snapped=silence_snapped,
        trimmed_sentences=trimmed,
    )

    logger.info(
        "ハイライト採用: %.3f〜%.3f秒（%s〜%s / %.1f秒）score %.1f%s%s",
        selected.start,
        selected.end,
        fmt_timestamp(selected.start),
        fmt_timestamp(selected.end),
        selected.duration,
        float(selected.score),
        "・無音へスナップ済み" if silence_snapped else "・無音は見つからず",
        f"・末尾の文を{trimmed}つ削除" if trimmed else "",
    )
    logger.info("フック: %s", selected.hook_line or "（LLM が hook_line を返しませんでした）")

    if selected.duration < hl.min_duration_sec:
        ctx.warn(
            f"ハイライトの尺が {r3(selected.duration)}秒 で下限 {_fmt_sec(hl.min_duration_sec)}秒 を"
            "下回っています。冒頭フックとして短すぎないか preview/highlight_in.mp4 で確認してください。"
        )
        logger.warning(
            "ハイライトが短すぎます（%.1f秒 < %.1f秒）。", selected.duration, hl.min_duration_sec
        )
    if selected.duration > hl.max_duration_sec:
        ctx.warn(
            f"ハイライトの尺が {r3(selected.duration)}秒 で上限 {_fmt_sec(hl.max_duration_sec)}秒 を"
            "超えています。末尾の文をこれ以上落とせませんでした。"
        )
        logger.warning(
            "ハイライトが長すぎます（%.1f秒 > %.1f秒）。", selected.duration, hl.max_duration_sec
        )
    if not silence_snapped:
        ctx.warn(
            "ハイライトの始点・終点の近くに無音が見つかりませんでした。"
            "語頭・語尾の切れ方を preview/highlight_in.mp4 と highlight_out.mp4 で確認してください。"
        )

    save(ctx, result)
    return result


def save(ctx: RunContext, result: HighlightResult) -> None:
    """work/<episode_id>/highlight.json に書く（run() の中から呼ぶ）。"""
    path = highlight_path(ctx)
    write_json(path, result.to_dict())
    logger.debug("ハイライトを保存しました: %s", path)


def load(ctx: RunContext) -> HighlightResult:
    """work/<episode_id>/highlight.json を読む（--from-step での再開用）。"""
    path = highlight_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"ハイライトの中間ファイルがありません: {path}\n"
            f"  先に `radio-cutter run {ctx.input_path} --from-step {STEP}` を実行してください"
            f"（Step {STEP} は Step 2 の transcript.json と Step 4 の cuts.json を使います）。"
        )
    try:
        data = read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissingArtifactError(
            f"ハイライトの中間ファイルの JSON が壊れています: {path}\n  {exc}\n"
            f"  このファイルを消して `radio-cutter run {ctx.input_path} --from-step {STEP}` で作り直してください。"
        ) from exc
    except OSError as exc:
        raise MissingArtifactError(f"ハイライトの中間ファイルを読めませんでした: {path}\n  {exc}") from exc

    if not isinstance(data, dict) or "selected" not in data:
        raise MissingArtifactError(
            f"ハイライトの中間ファイルの中身が想定した形（selected を持つオブジェクト）ではありません: {path}"
        )
    try:
        return HighlightResult.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingArtifactError(
            f"ハイライトの中間ファイルの内容が不正です: {path}\n  {exc}\n"
            f"  Step {STEP} を実行し直してください。"
        ) from exc
