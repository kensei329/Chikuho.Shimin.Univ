"""パイプライン各ステップが受け渡すデータ型。

方針（SPEC 11章）：
- 秒数は全て float。小数点以下3桁に丸めて保存する（フレーム番号には変換しない）。
- 中間ファイルは JSON。dataclass ⇄ dict の相互変換をここに集約する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 秒数を保存するときの丸め桁数
TIME_PRECISION = 3


def r3(value: float) -> float:
    """秒数を小数点以下3桁に丸める。JSON に書く直前に必ず通す。"""
    return round(float(value), TIME_PRECISION)


# --------------------------------------------------------------------------
# 文字起こし
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    """単語1つ。ASR が返す単語レベルのタイムスタンプ付き。"""

    word: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start": r3(self.start), "end": r3(self.end)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(word=str(d["word"]), start=float(d["start"]), end=float(d["end"]))


@dataclass
class Segment:
    """ASR のセグメント（おおむね1文）。"""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": r3(self.start),
            "end": r3(self.end),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            text=str(d.get("text", "")),
            words=[Word.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class Transcript:
    """work/transcript.json の中身。"""

    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)

    # ----- 変換 -----

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration": r3(self.duration),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return cls(
            language=str(d.get("language", "ja")),
            duration=float(d.get("duration", 0.0)),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ----- 参照 -----

    def words(self) -> list[Word]:
        """全セグメントの単語を時刻順に連結して返す。"""
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def words_in(self, start: float, end: float) -> list[Word]:
        """[start, end) に開始時刻が入る単語だけを返す。"""
        return [w for w in self.words() if start <= w.start < end]

    def segments_in(self, start: float, end: float) -> list[Segment]:
        """区間と少しでも重なるセグメントを返す。"""
        return [s for s in self.segments if s.end > start and s.start < end]

    def text_between(self, start: float, end: float) -> str:
        return "".join(w.word for w in self.words_in(start, end))


# --------------------------------------------------------------------------
# アンカー（Step 3）
# --------------------------------------------------------------------------


@dataclass
class AnchorCandidate:
    """あいまい一致で見つかった候補1つ。絞り込みで落ちた理由も持つ。"""

    score: float
    norm_start: int          # 正規化済み flat 上の開始文字インデックス
    norm_end: int            # 同・終端（排他）
    word_start: int          # 元の単語インデックス
    word_end: int            # 同・終端（排他）
    start_time: float
    end_time: float
    matched_text: str        # 正規化前テキストでの一致部分
    context: str             # 前後30文字程度
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": round(self.score, 2),
            "norm_start": self.norm_start,
            "norm_end": self.norm_end,
            "word_start": self.word_start,
            "word_end": self.word_end,
            "start_time": r3(self.start_time),
            "end_time": r3(self.end_time),
            "matched_text": self.matched_text,
            "context": self.context,
        }
        if self.rejected_reason:
            d["rejected_reason"] = self.rejected_reason
        return d


@dataclass
class AnchorResult:
    """確定したアンカー1つ。work/anchors.json の値。"""

    id: str
    phrase: str
    matched_text: str
    score: float
    raw_cut_time: float
    candidates_found: int
    candidates_rejected: int
    context: str
    word_index: int = -1
    rejected: list[AnchorCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "matched_text": self.matched_text,
            "score": round(self.score, 2),
            "raw_cut_time": r3(self.raw_cut_time),
            "candidates_found": self.candidates_found,
            "candidates_rejected": self.candidates_rejected,
            "context": self.context,
            "word_index": self.word_index,
        }

    @classmethod
    def from_dict(cls, anchor_id: str, d: dict[str, Any]) -> "AnchorResult":
        return cls(
            id=anchor_id,
            phrase=str(d["phrase"]),
            matched_text=str(d.get("matched_text", "")),
            score=float(d.get("score", 0.0)),
            raw_cut_time=float(d["raw_cut_time"]),
            candidates_found=int(d.get("candidates_found", 0)),
            candidates_rejected=int(d.get("candidates_rejected", 0)),
            context=str(d.get("context", "")),
            word_index=int(d.get("word_index", -1)),
        )


# --------------------------------------------------------------------------
# カット点（Step 4）
# --------------------------------------------------------------------------


@dataclass
class CutPoint:
    """無音に寄せたあとの最終カット点。"""

    anchor_id: str
    raw_cut_time: float
    cut_time: float
    silence_found: bool
    score: float = 0.0
    silence_start: float | None = None
    silence_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "raw_cut_time": r3(self.raw_cut_time),
            "cut_time": r3(self.cut_time),
            "silence_found": self.silence_found,
            "score": round(self.score, 2),
        }
        if self.silence_start is not None:
            d["silence_start"] = r3(self.silence_start)
        if self.silence_end is not None:
            d["silence_end"] = r3(self.silence_end)
        return d

    @classmethod
    def from_dict(cls, anchor_id: str, d: dict[str, Any]) -> "CutPoint":
        return cls(
            anchor_id=anchor_id,
            raw_cut_time=float(d["raw_cut_time"]),
            cut_time=float(d["cut_time"]),
            silence_found=bool(d.get("silence_found", False)),
            score=float(d.get("score", 0.0)),
            silence_start=(float(d["silence_start"]) if d.get("silence_start") is not None else None),
            silence_end=(float(d["silence_end"]) if d.get("silence_end") is not None else None),
        )


# --------------------------------------------------------------------------
# ハイライト（Step 5）
# --------------------------------------------------------------------------


@dataclass
class HighlightCandidate:
    start: float
    end: float
    score: float
    hook_line: str = ""
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": r3(self.start),
            "end": r3(self.end),
            "score": round(float(self.score), 2),
            "hook_line": self.hook_line,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HighlightCandidate":
        return cls(
            start=float(d["start"]),
            end=float(d["end"]),
            score=float(d.get("score", 0.0)),
            hook_line=str(d.get("hook_line", "")),
            reason=str(d.get("reason", "")),
        )


@dataclass
class HighlightResult:
    """採用した候補（スナップ後）と、残した次点。"""

    selected: HighlightCandidate
    snapped_from: HighlightCandidate
    alternatives: list[HighlightCandidate] = field(default_factory=list)
    silence_snapped: bool = False
    trimmed_sentences: int = 0

    @property
    def duration(self) -> float:
        return self.selected.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "snapped_from": {"start": r3(self.snapped_from.start), "end": r3(self.snapped_from.end)},
            "alternatives": [c.to_dict() for c in self.alternatives],
            "silence_snapped": self.silence_snapped,
            "trimmed_sentences": self.trimmed_sentences,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HighlightResult":
        selected = HighlightCandidate.from_dict(d["selected"])
        snapped = d.get("snapped_from") or {}
        return cls(
            selected=selected,
            snapped_from=HighlightCandidate(
                start=float(snapped.get("start", selected.start)),
                end=float(snapped.get("end", selected.end)),
                score=selected.score,
                hook_line=selected.hook_line,
                reason=selected.reason,
            ),
            alternatives=[HighlightCandidate.from_dict(c) for c in d.get("alternatives", [])],
            silence_snapped=bool(d.get("silence_snapped", False)),
            trimmed_sentences=int(d.get("trimmed_sentences", 0)),
        )


# --------------------------------------------------------------------------
# メタデータ（Step 6）
# --------------------------------------------------------------------------


@dataclass
class Chapter:
    """final.mp4 のタイムライン上のチャプター。"""

    time_sec: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"time_sec": r3(self.time_sec), "label": self.label}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chapter":
        return cls(time_sec=float(d["time_sec"]), label=str(d["label"]))


@dataclass
class TitleCandidate:
    direction: str
    text: str

    @property
    def length(self) -> int:
        """想定文字数（全角換算）。util.text_normalize.zenkaku_length と同じ数え方。"""
        from .util.text_normalize import zenkaku_length

        return zenkaku_length(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {"direction": self.direction, "text": self.text, "length": self.length}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TitleCandidate":
        return cls(direction=str(d["direction"]), text=str(d["text"]))


@dataclass
class MetadataResult:
    summary_lead: str = ""
    body: str = ""
    chapters: list[Chapter] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    titles: list[TitleCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_lead": self.summary_lead,
            "body": self.body,
            "chapters": [c.to_dict() for c in self.chapters],
            "keywords": list(self.keywords),
            "titles": [t.to_dict() for t in self.titles],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetadataResult":
        return cls(
            summary_lead=str(d.get("summary_lead", "")),
            body=str(d.get("body", "")),
            chapters=[Chapter.from_dict(c) for c in d.get("chapters", [])],
            keywords=[str(k) for k in d.get("keywords", [])],
            titles=[TitleCandidate.from_dict(t) for t in d.get("titles", [])],
        )


# --------------------------------------------------------------------------
# 書き出し（Step 7）
# --------------------------------------------------------------------------


@dataclass
class RenderResult:
    """出力した動画と実尺。"""

    files: dict[str, str] = field(default_factory=dict)          # 論理名 -> 出力パス
    durations: dict[str, float] = field(default_factory=dict)    # 論理名 -> 秒
    used_fallback_codec: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": dict(self.files),
            "durations": {k: r3(v) for k, v in self.durations.items()},
            "used_fallback_codec": self.used_fallback_codec,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderResult":
        return cls(
            files=dict(d.get("files", {})),
            durations={k: float(v) for k, v in d.get("durations", {}).items()},
            used_fallback_codec=bool(d.get("used_fallback_codec", False)),
            warnings=[str(w) for w in d.get("warnings", [])],
        )


# --------------------------------------------------------------------------
# LLM 呼び出し記録（decisions.json 用）
# --------------------------------------------------------------------------


@dataclass
class LlmCallRecord:
    step: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step": self.step,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "ok": self.ok,
        }
        if self.error:
            d["error"] = self.error
        return d


# --------------------------------------------------------------------------
# JSON 入出力
# --------------------------------------------------------------------------


def write_json(path: str | Path, data: Any) -> None:
    """UTF-8・インデント2・非ASCIIそのままで JSON を書く。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
