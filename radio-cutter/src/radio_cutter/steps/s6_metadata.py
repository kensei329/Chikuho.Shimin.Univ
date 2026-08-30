"""SPEC Step 6「メタデータ生成」。概要欄・チャプター（6-a）とタイトル候補30個（6-b）を作る。

方針:
- LLM 呼び出しは 6-a と 6-b の**2回に分ける**（同時に投げると片方の品質が落ちる、SPEC 6章）。
- チャプターの秒数は必ず `final.mp4` のタイムラインに直してから LLM に渡し、返ってきた値も
  `normalize_chapters` で YouTube の成立条件に合わせる。`0:00` は必ずハイライトに割り当てる。
- `description.txt` / `titles.md` の最終レイアウトはコード側で組み立てる（LLM に作らせると順序が崩れる）。
- 6-a / 6-b のどちらが失敗しても例外を外に投げない。作れたものだけ書き、動画の書き出しは止めない
  （SPEC 9章「そのステップだけ落とし、動画の書き出しは続行する」）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..config import YoutubeConfig
from ..context import RunContext
from ..errors import LlmError, MissingArtifactError, RadioCutterError
from ..llm.client import LlmClient, load_prompt, render_prompt
from ..llm.schemas import (
    METADATA_SCHEMA,
    TITLE_DIRECTIONS,
    TITLES_PER_DIRECTION,
    TITLES_SCHEMA,
)
from ..logging_util import get_logger
from ..models import (
    Chapter,
    CutPoint,
    HighlightResult,
    LlmCallRecord,
    MetadataResult,
    TitleCandidate,
    Transcript,
    r3,
    read_json,
    write_json,
)
from ..util.text_normalize import normalize_phrase, zenkaku_length
from ..util.timeline import (
    fmt_timestamp,
    normalize_chapters,
    resolve_segment_bounds,
    to_final_time,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ステップの約束（steps/sN_*.py 共通プロトコル）
# ---------------------------------------------------------------------------

STEP: int = 6
NAME: str = "メタデータ生成"

#: work/<episode_id>/ に書く中間ファイル
METADATA_FILE = "metadata.json"
OUTPUTS: tuple[str, ...] = (METADATA_FILE,)

#: out/<episode_id>/ に書く成果物
DESCRIPTION_FILE = "description.txt"
TITLES_FILE = "titles.md"

#: LLM 呼び出しの step 名（decisions.json の llm_calls とスタブ応答のキーになる）
STEP_NAME_METADATA = "metadata"
STEP_NAME_TITLES = "titles"

# ---------------------------------------------------------------------------
# 組み立てに使う定数
# ---------------------------------------------------------------------------

#: description.txt の罫線（SPEC 6-a のレイアウトそのまま。全角罫線15本）
RULE_LINE = "━" * 15

#: チャプター見出し
CHAPTER_HEADING = "■ チャプター"

#: 文字起こし要約の丸め単位（SPEC 6-a「トークン節約のため30秒単位に丸めた要約でよい」）
SUMMARY_WINDOW_SEC = 30.0

#: 要約1行あたりの最大文字数。壊れた文字起こしで1行が異常に長くなるのを防ぐ保険。
MAX_SUMMARY_LINE_CHARS = 600

#: 0:00 のチャプターに付ける既定ラベル（hook_line から作れなかったとき）
DEFAULT_HIGHLIGHT_LABEL = "今回の結論"

#: 0:00 のチャプターラベルの上限（全角換算）
HIGHLIGHT_LABEL_MAX_ZENKAKU = 20

#: hook_line が空のときに文字起こしから作る代替フックの上限（全角換算）
HOOK_FALLBACK_MAX_ZENKAKU = 60

#: decisions.json に載せる LLM エラー本文の上限
MAX_RECORDED_ERROR_CHARS = 500

#: 「0秒のチャプターがハイライトを表していない」と判定する語。
#: SPEC 6-a の悪い例（`{"time_sec": 0, "label": "オープニング"}`）に該当するものを弾く。
#: 部分一致で見る語（日本語のみ。ラテン文字を部分一致にすると "Copilot" が "op" に当たる）。
_NOT_HIGHLIGHT_SUBSTRINGS: tuple[str, ...] = (
    "オープニング",
    "イントロ",
    "あいさつ",
    "挨拶",
    "自己紹介",
    "番組紹介",
    "はじめに",
    "雑談",
    "本編",
    "前置き",
)

#: 完全一致で見る語（短すぎて部分一致にすると誤爆するもの）
_NOT_HIGHLIGHT_EXACT: frozenset[str] = frozenset(
    {"op", "opening", "intro", "冒頭", "導入", "導入部", "オープン", "トーク", "その1"}
)


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def metadata_path(ctx: RunContext) -> Path:
    """メタデータの中間ファイル（`titles` サブコマンドが読み直す）。"""
    return ctx.work_path(METADATA_FILE)


def description_path(ctx: RunContext) -> Path:
    """YouTube 概要欄の出力先。"""
    return ctx.out_path(DESCRIPTION_FILE)


def titles_path(ctx: RunContext) -> Path:
    """タイトル候補の出力先。"""
    return ctx.out_path(TITLES_FILE)


# ---------------------------------------------------------------------------
# 文字列の補助（純関数）
# ---------------------------------------------------------------------------


def trim_to_zenkaku(text: str, limit: int, *, ellipsis: str = "…") -> str:
    """全角換算 limit 字に収める。収まらなければ末尾を落として省略記号を付ける。

    タイトルではなくチャプターラベル用。長さの数え方は `zenkaku_length` と同じにする。
    """
    s = str(text).strip()
    if not s or limit <= 0:
        return s
    if zenkaku_length(s) <= limit:
        return s
    budget = max(1, limit - zenkaku_length(ellipsis))
    cut = 0
    for i in range(1, len(s) + 1):
        if zenkaku_length(s[:i]) > budget:
            break
        cut = i
    head = s[:cut]
    # 読点で切れる位置が十分後ろにあるなら、そこで切ったほうが語の途中で千切れない。
    comma = max(head.rfind("、"), head.rfind("，"), head.rfind(","))
    if comma >= 0 and zenkaku_length(head[:comma]) >= budget * 0.6:
        head = head[:comma]
    trimmed = head.rstrip("、，,・ 　")
    if not trimmed:
        return s[: max(1, cut)]
    return trimmed + ellipsis


def first_sentence(text: str) -> str:
    """最初の文だけを取り出す。チャプターラベルとフックの短縮に使う。"""
    s = str(text).strip()
    if not s:
        return ""
    for i, ch in enumerate(s):
        if ch in "\n\r":
            return s[:i].strip()
        # 半角ピリオドは入れない。「実は3.5倍になるんです」が「実は3」に千切れる。
        if ch in "。．？?！!":
            # 疑問符・感嘆符は文の意味に効くので残し、句点は落とす。
            return (s[: i + 1] if ch in "？?！!" else s[:i]).strip()
    return s


def highlight_label(hook_line: str, *, fallback: str = DEFAULT_HIGHLIGHT_LABEL) -> str:
    """hook_line から 0:00 チャプターの短いラベルを作る。作れなければ既定ラベル。"""
    label = trim_to_zenkaku(first_sentence(hook_line), HIGHLIGHT_LABEL_MAX_ZENKAKU)
    return label or fallback


def label_represents_highlight(label: str) -> bool:
    """0:00 のチャプターラベルがハイライト部分を指しているとみなせるか（発見的な判定）。

    「オープニング」「本編」のような、冒頭ハイライトではなく元動画の頭を指すラベルを弾く。
    ここで弾いたものは `ensure_highlight_chapter` がハイライト側のラベルに差し替える。
    """
    normalized = normalize_phrase(str(label))
    if not normalized:
        return False
    if normalized in _NOT_HIGHLIGHT_EXACT:
        return False
    return not any(normalize_phrase(word) in normalized for word in _NOT_HIGHLIGHT_SUBSTRINGS)


# ---------------------------------------------------------------------------
# 文字起こしの要約（6-a の入力）
# ---------------------------------------------------------------------------


def summarize_transcript_window(
    transcript: Transcript,
    start: float,
    end: float,
    *,
    window: float = SUMMARY_WINDOW_SEC,
) -> list[dict]:
    """区間 [start, end) の文字起こしを window 秒単位に丸めて `[{"start", "text"}]` にする。

    60分ぶんの単語タイムスタンプをそのまま渡すとトークンが跳ねるため、
    30秒バケツに畳んで「どの秒に何を話したか」だけを残す（SPEC 6-a）。
    `start` は元動画の絶対秒のまま返す。final.mp4 への変換は呼び出し側で行う。
    """
    lo = float(start)
    hi = float(end)
    size = float(window)
    if size <= 0:
        raise ValueError(f"window は正の秒数にしてください（実際: {window!r}）。")
    if hi <= lo:
        return []

    buckets: dict[int, list[str]] = {}
    for segment in transcript.segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        # 区間と少しでも重なるセグメントを対象にする（境界の文を落とさない）。
        if seg_end <= lo or seg_start >= hi:
            continue
        text = str(segment.text).strip()
        if not text:
            text = "".join(w.word for w in segment.words).strip()
        if not text:
            continue
        anchor = seg_start if seg_start >= lo else lo
        index = int(math.floor((anchor - lo) / size))
        buckets.setdefault(index, []).append(text)

    out: list[dict] = []
    for index in sorted(buckets):
        joined = "".join(buckets[index]).strip()
        if not joined:
            continue
        if len(joined) > MAX_SUMMARY_LINE_CHARS:
            joined = joined[:MAX_SUMMARY_LINE_CHARS] + "…"
        out.append({"start": r3(lo + index * size), "text": joined})
    return out


def render_transcript_lines(
    windows: Sequence[dict],
    *,
    cut_a: float,
    cut_b: float,
    highlight_dur: float,
    main_dur: float,
    to_final: Callable[[float], float] | None = None,
) -> str:
    """要約セグメントを `[秒] テキスト` の行にする。秒は final.mp4 のタイムラインに変換する。

    プロンプト（metadata.md）は「この秒数はすでに final.mp4 に変換済み」と書いてあるので、
    ここで変換しないとチャプターが丸ごとずれる。

    `to_final` を渡すとそれを使う（segments が2本でない config や position="append" 用）。
    渡さなければ SPEC 6-a の式をそのまま使う。
    """
    lines: list[str] = []
    for item in windows:
        raw_start = float(item["start"])
        final_t = (
            to_final(raw_start)
            if to_final is not None
            else to_final_time(
                raw_start,
                cut_a=cut_a,
                cut_b=cut_b,
                highlight_dur=highlight_dur,
                main_dur=main_dur,
            )
        )
        text = str(item["text"]).strip()
        if not text:
            continue
        lines.append(f"[{int(max(0.0, final_t))}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# チャプター（6-a の後処理）
# ---------------------------------------------------------------------------


def ensure_highlight_chapter(
    chapters: Sequence[Chapter],
    *,
    hook_line: str,
    highlight_duration: float,
    final_duration: float,
    min_gap: float = 10.0,
) -> tuple[list[Chapter], list[str]]:
    """`0:00` を必ずハイライトに割り当てる（SPEC 6-a）。(直した結果, 警告文) を返す。

    先頭が 0 でない、または 0 秒のラベルがハイライトを表していない場合は、
    hook_line から作った短いラベルの 0 秒チャプターを先頭に挿入する。
    追い出した元のラベルは本編の開始位置（= ハイライトの尺）へ移す。ただしそこへ置くと
    次のチャプターと min_gap 秒未満で並んでしまうときは、後続の中身のあるチャプターを
    生かすために追い出した方を捨てる。

    **`normalize_chapters` より先に呼ぶこと。** 先に正規化すると「先頭を必ず 0 にする」で
    本編のチャプターが 0 に寄せられ、ハイライトを指していない見出しが 0:00 に居座る。
    ここは並べ替えと 0 秒の確保だけを行い、間隔や尺のチェックは呼び出し側の
    `normalize_chapters` に任せる。
    """
    label = highlight_label(hook_line)
    warnings: list[str] = []
    items = sorted(chapters, key=lambda c: float(c.time_sec))

    if not items:
        return (
            [Chapter(time_sec=0.0, label=label)],
            [f"チャプターが1つも無かったため 0:00「{label}」だけを作りました。"],
        )

    head = items[0]
    if not (float(head.time_sec) <= 0.0 and label_represents_highlight(head.label)):
        if float(head.time_sec) <= 0.0:
            moved_to = max(0.0, float(highlight_duration))
            next_time = next((float(c.time_sec) for c in items[1:] if float(c.time_sec) > 0.0), None)
            can_move = (
                0.0 < moved_to < float(final_duration)
                and (next_time is None or next_time - moved_to >= float(min_gap))
            )
            if can_move:
                warnings.append(
                    f"0:00 のチャプター「{head.label}」は冒頭ハイライトを表していないため、"
                    f"{fmt_timestamp(moved_to)}（本編の頭）に移し、0:00 に「{label}」を入れました。"
                )
                items[0] = Chapter(time_sec=r3(moved_to), label=head.label)
            else:
                warnings.append(
                    f"0:00 のチャプター「{head.label}」は冒頭ハイライトを表しておらず、"
                    f"移動先（{fmt_timestamp(moved_to)}）も次のチャプターと近すぎるため除外し、"
                    f"0:00 に「{label}」を入れました。"
                )
                items.pop(0)
        else:
            warnings.append(
                f"最初のチャプターが {fmt_timestamp(head.time_sec)} だったため、"
                f"0:00 に「{label}」を挿入しました（0:00 は必ずハイライトに割り当てます）。"
            )
        items.insert(0, Chapter(time_sec=0.0, label=label))

    # 0 秒に別のチャプターが残っていると normalize_chapters の「同時刻は後勝ち」で
    # ハイライトのラベルが上書きされるため、ここで落とす。
    for extra in items[1:]:
        if float(extra.time_sec) <= 0.0:
            warnings.append(
                f"0:00 に重複していたチャプター「{extra.label}」を除外しました"
                "（0:00 はハイライトに割り当てます）。"
            )
    kept = [items[0]] + [c for c in items[1:] if float(c.time_sec) > 0.0]
    return (kept, warnings)


# ---------------------------------------------------------------------------
# 成果物の組み立て（純関数）
# ---------------------------------------------------------------------------


def build_description(meta: MetadataResult, youtube: YoutubeConfig) -> str:
    """SPEC 6-a のレイアウトで description.txt の中身を作る。

    LLM には最終フォーマットを作らせない（順序が崩れるため）。
    空の要素は行ごと省き、連続する空行を作らない。
    """
    blocks: list[list[str]] = []

    lead = str(meta.summary_lead).strip()
    if lead:
        blocks.append([lead])

    body = str(meta.body).strip()
    if body:
        blocks.append([body])

    if meta.chapters:
        chapter_lines = [RULE_LINE, CHAPTER_HEADING]
        for chapter in meta.chapters:
            label = str(chapter.label).strip()
            chapter_lines.append(f"{fmt_timestamp(chapter.time_sec)} {label}".rstrip())
        blocks.append(chapter_lines)

    footer = str(youtube.fixed_footer).strip()
    links = [str(link).strip() for link in youtube.channel_links if str(link).strip()]
    hashtags = [str(tag).strip() for tag in youtube.hashtags if str(tag).strip()]

    # 罫線は「フッター・リンク・ハッシュタグのどれかがある」ときだけ引く。
    # 何も無いのに罫線だけ残ると、概要欄の末尾に意味のない線が出るため。
    if footer or links or hashtags:
        tail: list[str] = [RULE_LINE]
        if footer:
            tail.append(footer)
        tail.extend(links)
        blocks.append(tail)

    if hashtags:
        blocks.append([" ".join(hashtags)])

    return "\n\n".join("\n".join(block) for block in blocks).rstrip() + "\n"


def build_titles_markdown(titles: Sequence[TitleCandidate]) -> str:
    """SPEC 6-b の titles.md を作る。方向ごとに H2、通し番号は全体で1〜30の連番。

    各行末に想定文字数「（全角NN字）」を併記する（モバイル一覧で切れないかの判断材料）。
    """
    lines: list[str] = ["# タイトル候補", ""]
    if not titles:
        lines.append("（タイトル候補は生成できませんでした。）")
        return "\n".join(lines) + "\n"

    grouped: dict[str, list[TitleCandidate]] = {}
    for title in titles:
        grouped.setdefault(str(title.direction), []).append(title)

    # 既知の6方向を SPEC の順に並べ、未知の方向は取りこぼさないよう後ろに付ける。
    order = [d for d in TITLE_DIRECTIONS if d in grouped]
    order.extend(d for d in grouped if d not in TITLE_DIRECTIONS)

    number = 1
    for direction in order:
        lines.append(f"## {direction}")
        for title in grouped[direction]:
            text = str(title.text).strip()
            lines.append(f"{number}. {text}（全角{zenkaku_length(text)}字）")
            number += 1
        lines.append("")

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM 入力の組み立て
# ---------------------------------------------------------------------------


def _hook_line(highlight: HighlightResult, transcript: Transcript) -> str:
    """ハイライトの入りの一言。LLM が返していなければ文字起こしから作る。"""
    hook = str(highlight.selected.hook_line).strip()
    if hook:
        return hook
    spoken = transcript.text_between(highlight.selected.start, highlight.selected.end).strip()
    return trim_to_zenkaku(first_sentence(spoken), HOOK_FALLBACK_MAX_ZENKAKU)


def _format_keywords(keywords: Sequence[str]) -> str:
    values = [str(k).strip() for k in keywords if str(k).strip()]
    if not values:
        return "（キーワードは取得できていません。概要本文から補ってください）"
    return "、".join(values)


def _format_directions() -> str:
    return "\n".join(
        f"- {d}（{TITLES_PER_DIRECTION}個）" for d in TITLE_DIRECTIONS
    )


def _record_llm(
    ctx: RunContext,
    step: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    retries: int = 0,
    ok: bool = True,
    error: str | None = None,
) -> None:
    """decisions.json の llm_calls に1件足す。失敗も残す（あとから何が起きたか追えるように）。"""
    ctx.record_llm_call(
        LlmCallRecord(
            step=step,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retries=retries,
            ok=ok,
            error=(error[:MAX_RECORDED_ERROR_CHARS] if error else None),
        )
    )


# ---------------------------------------------------------------------------
# 6-a: 概要欄とチャプター
# ---------------------------------------------------------------------------


def generate_description_parts(
    ctx: RunContext,
    llm: LlmClient,
    *,
    transcript_text: str,
    hook_line: str,
    final_duration: float,
    highlight_duration: float,
    highlight_at_start: bool = True,
) -> MetadataResult:
    """6-a を1回だけ呼び、summary_lead / body / chapters / keywords を得る。

    チャプターの秒数は既に final.mp4 のタイムライン。ここでは検証と整形だけを行う。
    失敗時は LlmError をそのまま投げる（呼び出し側が握って続行する）。
    """
    prompt = render_prompt(
        load_prompt("metadata"),
        {
            "channel": ctx.config.channel or "（番組名の設定なし）",
            "final_duration": int(round(max(0.0, final_duration))),
            "highlight_duration": int(round(max(0.0, highlight_duration))),
            "hook_line": hook_line or "（ハイライトの一言は取得できていません）",
            "transcript": transcript_text or "（この区間の文字起こしがありません）",
        },
    )
    response = llm.complete_json(
        step=STEP_NAME_METADATA, prompt=prompt, schema=METADATA_SCHEMA
    )
    _record_llm(
        ctx,
        STEP_NAME_METADATA,
        llm.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        retries=response.retries,
    )

    data = response.data
    raw_chapters = [Chapter.from_dict(c) for c in data.get("chapters", [])]
    if highlight_at_start:
        # 先に 0:00 をハイライトへ割り当ててから、YouTube の成立条件で整える。
        # 順番を逆にすると normalize_chapters が本編のチャプターを 0 に寄せてしまう。
        staged, warnings = ensure_highlight_chapter(
            raw_chapters,
            hook_line=hook_line,
            highlight_duration=highlight_duration,
            final_duration=final_duration,
        )
    else:
        # position="append" ではハイライトが末尾なので、0:00 は本編の頭。
        # SPEC の「0:00 はハイライト」は既定の prepend を前提にした話。
        staged, warnings = list(raw_chapters), []
    chapters, more = normalize_chapters(staged, final_duration)
    for message in [*warnings, *more]:
        ctx.warn(f"チャプター: {message}")
        logger.warning("チャプターを直しました: %s", message)

    return MetadataResult(
        summary_lead=str(data.get("summary_lead", "")).strip(),
        body=str(data.get("body", "")).strip(),
        chapters=chapters,
        keywords=[str(k).strip() for k in data.get("keywords", []) if str(k).strip()],
    )


# ---------------------------------------------------------------------------
# 6-b: タイトル候補30個
# ---------------------------------------------------------------------------


def generate_titles(
    ctx: RunContext,
    llm: LlmClient,
    *,
    summary_lead: str,
    summary_body: str,
    keywords: Sequence[str],
    hook_line: str,
) -> list[TitleCandidate]:
    """6-b を1回だけ呼び、6方向 × 5個 のタイトル候補を得る。

    件数の偏りは警告にとどめる（30個は揃っているのだから、捨てるより人に見せた方がよい）。
    失敗時は LlmError をそのまま投げる（呼び出し側が握って続行する）。
    """
    prompt = render_prompt(
        load_prompt("titles"),
        {
            "channel": ctx.config.channel or "（番組名の設定なし）",
            "summary_lead": summary_lead or "（リード文は生成できませんでした。下の要約から判断してください）",
            "summary_body": summary_body or "（本文は生成できませんでした）",
            "keywords": _format_keywords(keywords),
            "hook_line": hook_line or "（ハイライトの一言は取得できていません）",
            "directions": _format_directions(),
        },
    )
    response = llm.complete_json(step=STEP_NAME_TITLES, prompt=prompt, schema=TITLES_SCHEMA)
    _record_llm(
        ctx,
        STEP_NAME_TITLES,
        llm.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        retries=response.retries,
    )

    titles = [
        TitleCandidate(direction=str(t.get("direction", "")), text=str(t.get("text", "")).strip())
        for t in response.data.get("titles", [])
    ]
    titles = [t for t in titles if t.text]

    counts: dict[str, int] = {}
    for title in titles:
        counts[title.direction] = counts.get(title.direction, 0) + 1
    for direction in TITLE_DIRECTIONS:
        actual = counts.get(direction, 0)
        if actual != TITLES_PER_DIRECTION:
            message = (
                f"タイトル: 方向「{direction}」が {actual} 個です"
                f"（期待は {TITLES_PER_DIRECTION} 個）。"
            )
            ctx.warn(message)
            logger.warning("%s", message)
    for direction in sorted(set(counts) - set(TITLE_DIRECTIONS)):
        message = f"タイトル: 想定外の方向「{direction}」が {counts[direction]} 個返りました。"
        ctx.warn(message)
        logger.warning("%s", message)

    return titles


# ---------------------------------------------------------------------------
# 出力ファイル
# ---------------------------------------------------------------------------


def write_description(ctx: RunContext, meta: MetadataResult) -> Path:
    """out/<episode_id>/description.txt を書く。"""
    path = description_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_description(meta, ctx.config.youtube), encoding="utf-8")
    logger.info("概要欄を書き出しました: %s（チャプター %d 件）", path, len(meta.chapters))
    return path


def write_titles(ctx: RunContext, titles: Sequence[TitleCandidate]) -> Path:
    """out/<episode_id>/titles.md を書く。"""
    path = titles_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_titles_markdown(titles), encoding="utf-8")
    logger.info("タイトル候補を書き出しました: %s（%d 個）", path, len(titles))
    return path


# ---------------------------------------------------------------------------
# 区間の解決
# ---------------------------------------------------------------------------


def resolve_durations(
    ctx: RunContext,
    cuts: Mapping[str, CutPoint],
    highlight: HighlightResult,
    total_duration: float,
) -> tuple[float, float, float, float, float]:
    """(cut_a, cut_b, Dh, Dm, De) を返す。

    cut_a / cut_b はハイライトの取得元セグメント（config.highlight.source_segment）の
    from / to から解決する。"A" / "B" と決め打ちしないのは、config でアンカーIDも
    セグメント構成も自由に変えられるため（SPEC 5章）。
    """
    source = ctx.config.segment(ctx.config.highlight.source_segment)
    cut_a, cut_b = resolve_segment_bounds(source, cuts, total_duration)
    highlight_dur = max(0.0, float(highlight.selected.duration))
    main_dur = max(0.0, cut_b - cut_a)
    ending_dur = max(0.0, float(total_duration) - cut_b)
    return (cut_a, cut_b, highlight_dur, main_dur, ending_dur)


def build_final_timeline(
    ctx: RunContext,
    cuts: Mapping[str, CutPoint],
    highlight: HighlightResult,
    total_duration: float,
) -> tuple[Callable[[float], float], float, float]:
    """元動画の時刻 → final.mp4 の時刻 に直す関数と、final の総尺・本編の開始位置を返す。

    SPEC 6-a の式は「ハイライトが先頭・その後ろに本編とエンディングの2本」という
    既定の構成を前提にしている。config は segments を自由に組めるし、
    highlight.position は "append" も取れるので、ここでは Step 7 と同じ書き出し順から
    各区間の先頭位置を積み上げて作る。既定の構成では SPEC の式と同じ値になる。

    返り値の3つ目は「本編（ハイライトの取得元セグメント）が final の何秒目から始まるか」。
    prepend ならハイライトの尺、append なら 0 になる。
    """
    spans: list[tuple[float, float]] = []
    for seg in ctx.config.segments:
        spans.append(resolve_segment_bounds(seg, cuts, total_duration))
    highlight_dur = max(0.0, float(highlight.selected.duration))
    source_index = [seg.name for seg in ctx.config.segments].index(
        ctx.config.highlight.source_segment
    )

    prepend = ctx.config.highlight.position != "append"
    offsets: list[float] = []
    cursor = highlight_dur if prepend else 0.0
    for start, end in spans:
        offsets.append(cursor)
        cursor += max(0.0, end - start)
    final_duration = cursor + (0.0 if prepend else highlight_dur)
    main_offset = offsets[source_index] if offsets else 0.0

    def to_final(t: float) -> float:
        """元動画の t を final.mp4 上の秒に直す。どの区間にも入らなければ最寄りに寄せる。"""
        value = float(t)
        for index, (start, end) in enumerate(spans):
            if start <= value < end:
                return offsets[index] + (value - start)
        if spans and value < spans[0][0]:
            return offsets[0]
        if spans:
            last_start, last_end = spans[-1]
            return offsets[-1] + max(0.0, last_end - last_start)
        return 0.0

    return to_final, final_duration, main_offset


def _resolve_total_duration(
    ctx: RunContext, transcript: Transcript, total_duration: float | None
) -> float:
    """総尺を決める。呼び出し側の値 → transcript.duration → 最後の単語の終端 の順。

    総尺が無いとエンディングの尺（De）が出ず、チャプターの時刻が全部ずれるのでここで止める。
    """
    if total_duration is not None and float(total_duration) > 0:
        return float(total_duration)
    if float(transcript.duration) > 0:
        return float(transcript.duration)
    words = transcript.words()
    if words:
        duration = float(words[-1].end)
        logger.warning(
            "総尺が渡されず transcript.duration も 0 だったため、最後の単語の終端 %.3f秒 を使います。",
            duration,
        )
        return duration
    raise MissingArtifactError(
        "動画の総尺が分かりません（transcript.duration が 0 で、単語も1つもありません）。\n"
        f"  先に `radio-cutter run {ctx.input_path} --from-step 1` で"
        "Step 1〜2（音声抽出・文字起こし）をやり直してください。"
    )


# ---------------------------------------------------------------------------
# ステップ本体
# ---------------------------------------------------------------------------


def _previous_metadata(ctx: RunContext) -> MetadataResult | None:
    """前回の metadata.json を読む。無い・壊れていても例外は投げない。

    LLM が落ちた回に空の内容で上書きすると、前回うまくいったときの
    概要欄とタイトル30個がまるごと消える。それを避けるための控え。
    """
    try:
        return load(ctx)
    except (MissingArtifactError, RadioCutterError, OSError, ValueError):
        return None


def run(
    ctx: RunContext,
    transcript: Transcript,
    cuts: Mapping[str, CutPoint],
    highlight: HighlightResult,
    llm: LlmClient,
    total_duration: float | None = None,
) -> MetadataResult:
    """6-a と 6-b を順に呼び、work/metadata.json と out/description.txt / titles.md を書く。

    LLM 呼び出しは必ず2回に分ける。どちらが失敗しても例外は投げず、
    `ctx.warn()` に残して作れたものだけ書く（SPEC 9章）。
    """
    ctx.ensure_dirs()

    total = _resolve_total_duration(ctx, transcript, total_duration)
    cut_a, cut_b, highlight_dur, main_dur, ending_dur = resolve_durations(
        ctx, cuts, highlight, total
    )
    # 書き出し順から組み立てる。segments が2本でない config や position="append" でも
    # チャプターの時刻が合うようにするため（既定の構成なら SPEC 6-a の式と同じ値になる）。
    to_final, final_duration, main_offset = build_final_timeline(ctx, cuts, highlight, total)

    logger.info(
        "final.mp4 の想定尺 %s（ハイライト %.3f秒 + 本編 %.3f秒 + エンディング %.3f秒）",
        fmt_timestamp(final_duration),
        highlight_dur,
        main_dur,
        ending_dur,
    )
    if highlight_dur <= 0:
        ctx.warn("メタデータ: ハイライトの尺が0秒です。チャプターの時刻がずれる可能性があります。")
        logger.warning("ハイライトの尺が0秒です。0:00 のチャプターが実質的に空になります。")

    hook_line = _hook_line(highlight, transcript)

    # 6-a の入力: 本編＋エンディングの文字起こしを30秒単位に丸め、final.mp4 の秒に直す。
    windows = summarize_transcript_window(transcript, cut_a, max(cut_b, total))
    transcript_text = render_transcript_lines(
        windows,
        cut_a=cut_a,
        cut_b=cut_b,
        highlight_dur=highlight_dur,
        main_dur=main_dur,
        to_final=to_final,
    )
    if not windows:
        ctx.warn(
            "メタデータ: 本編〜エンディングの文字起こしが空です。概要欄の品質は期待できません。"
        )
        logger.warning(
            "要約セグメントが0件です（区間 %.3f〜%.3f秒）。文字起こしの範囲を確認してください。",
            cut_a,
            max(cut_b, total),
        )
    else:
        logger.info(
            "6-a に渡す要約セグメント: %d 件（%.0f秒単位・%d文字）",
            len(windows),
            SUMMARY_WINDOW_SEC,
            len(transcript_text),
        )

    result = MetadataResult()
    previous = _previous_metadata(ctx)

    # ---- 6-a: チャプター＋概要欄 ----
    try:
        result = generate_description_parts(
            ctx,
            llm,
            transcript_text=transcript_text,
            hook_line=hook_line,
            final_duration=final_duration,
            highlight_duration=main_offset,
            highlight_at_start=ctx.config.highlight.position != "append",
        )
    except LlmError as exc:
        _record_llm(ctx, STEP_NAME_METADATA, llm.model, ok=False, error=str(exc))
        message = f"Step 6-a（概要欄・チャプター）に失敗したため description.txt は作れませんでした: {exc}"
        ctx.warn(message)
        logger.error("%s", message)
        if previous is not None and (previous.summary_lead or previous.body):
            # 前回ぶんを metadata.json から消さない。ただしチャプターだけは引き継がない。
            # 時刻は今回のカット点に紐づくので、前回の値を残すと嘘の位置を指す。
            result.summary_lead = previous.summary_lead
            result.body = previous.body
            result.keywords = list(previous.keywords)
            note = (
                "メタデータ: 概要欄の本文は前回の metadata.json のものを残しました"
                "（チャプターは今回のカット点と合わないため引き継いでいません）。"
            )
            ctx.warn(note)
            logger.warning("%s", note)
    else:
        write_description(ctx, result)

    # ---- 6-b: タイトル30個（6-a が失敗しても試す） ----
    summary_body = result.body or transcript_text
    try:
        result.titles = generate_titles(
            ctx,
            llm,
            summary_lead=result.summary_lead,
            summary_body=summary_body,
            keywords=result.keywords,
            hook_line=hook_line,
        )
    except LlmError as exc:
        _record_llm(ctx, STEP_NAME_TITLES, llm.model, ok=False, error=str(exc))
        message = f"Step 6-b（タイトル候補）に失敗したため titles.md は作れませんでした: {exc}"
        ctx.warn(message)
        logger.error("%s", message)
        if previous is not None and previous.titles:
            # タイトルはカット点に依存しないので、前回ぶんをそのまま残してよい。
            result.titles = list(previous.titles)
            note = f"メタデータ: タイトル候補は前回の {len(result.titles)} 件を残しました。"
            ctx.warn(note)
            logger.warning("%s", note)
    else:
        write_titles(ctx, result.titles)

    save(ctx, result)
    return result


def regenerate_titles(
    ctx: RunContext,
    meta: MetadataResult,
    highlight: HighlightResult,
    llm: LlmClient,
    transcript: Transcript | None = None,
) -> MetadataResult:
    """6-b だけをやり直して titles.md を上書きする（`radio-cutter titles` 用）。

    失敗しても例外は投げない。既存の metadata.json のタイトルはそのまま残す。
    """
    ctx.ensure_dirs()
    hook_line = str(highlight.selected.hook_line).strip()
    if not hook_line and transcript is not None:
        hook_line = _hook_line(highlight, transcript)
    try:
        meta.titles = generate_titles(
            ctx,
            llm,
            summary_lead=meta.summary_lead,
            summary_body=meta.body,
            keywords=meta.keywords,
            hook_line=hook_line,
        )
    except LlmError as exc:
        _record_llm(ctx, STEP_NAME_TITLES, llm.model, ok=False, error=str(exc))
        message = f"Step 6-b（タイトル候補）の再生成に失敗しました: {exc}"
        ctx.warn(message)
        logger.error("%s", message)
        return meta
    write_titles(ctx, meta.titles)
    save(ctx, meta)
    return meta


# ---------------------------------------------------------------------------
# 中間ファイル
# ---------------------------------------------------------------------------


def save(ctx: RunContext, result: MetadataResult) -> None:
    """work/<episode_id>/metadata.json に書く。run() の中から必ず呼ぶ。"""
    path = metadata_path(ctx)
    write_json(path, result.to_dict())
    logger.debug(
        "メタデータを保存しました: %s（チャプター %d 件 / タイトル %d 個）",
        path,
        len(result.chapters),
        len(result.titles),
    )


def load(ctx: RunContext) -> MetadataResult:
    """work/<episode_id>/metadata.json を読む（--from-step / titles サブコマンド用）。"""
    path = metadata_path(ctx)
    if not path.exists():
        raise MissingArtifactError(
            f"メタデータの中間ファイルがありません: {path}\n"
            f"  先に `radio-cutter run {ctx.input_path} --only-step {STEP}` を実行してください"
            "（Step 6 は Step 2 の transcript.json、Step 4 の cuts.json、"
            "Step 5 の highlight.json を使います）。"
        )
    try:
        data = read_json(path)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissingArtifactError(
            f"メタデータの中間ファイルの JSON が壊れています: {path}\n  {exc}\n"
            f"  このファイルを消して `radio-cutter run {ctx.input_path} --from-step {STEP}` で"
            "作り直してください。"
        ) from exc
    except OSError as exc:
        raise MissingArtifactError(f"メタデータの中間ファイルを読めませんでした: {path}\n  {exc}") from exc

    if not isinstance(data, dict):
        raise MissingArtifactError(
            f"メタデータの中間ファイルが JSON オブジェクトではありません: {path}"
        )
    try:
        return MetadataResult.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise MissingArtifactError(
            f"メタデータの中間ファイルの内容が不正です: {path}\n  {exc}"
        ) from exc
