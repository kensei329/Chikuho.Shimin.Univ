"""SPEC Step 3 の 1「フラット化」と 2「正規化」を担う（アンカー検出の土台）。

方針（SPEC 11章）：
- 文字列比較は必ず正規化後に行う。生の文字起こしテキストで比較しない。
- 過剰正規化は誤検出を招くので、カタカナ長音「ー」や促音には触らない。
- 正規化で消えた・増えた文字があっても、必ず元の文字列インデックスへ戻せるようにする
  （戻せないと、一致位置から単語 index を引けず raw_cut_time が求まらないため）。
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterator, Sequence

from ..models import Word

# ---------------------------------------------------------------------------
# 除去対象の文字
# ---------------------------------------------------------------------------

#: 絶対に除去してはいけない文字。長音符や繰り返し記号は語の一部であり、
#: 落とすと「チャンネル」と「チャネル」の区別が付かなくなる。
_NEVER_STRIP: frozenset[str] = frozenset(
    "ー"  # ー カタカナ・ひらがな長音符
    "ｰ"  # ｰ 半角長音符（NFKC で ー になるが念のため）
    "ゝゞ"  # ゝゞ ひらがな繰り返し記号
    "ヽヾ"  # ヽヾ カタカナ繰り返し記号
    "々"  # 々 同の字点
)

#: 除去対象の候補。読みやすいように用途ごとに分けて持つ。
#: ダッシュ類は長音符と字形が紛らわしいので、必ずコードポイントで書く。
_PUNCT_GROUPS: tuple[str, ...] = (
    # 空白類（半角/全角スペース、タブ、改行、復帰、NBSP、各種スペース）
    " \t\n\r\v\f 　   ​﻿",
    # 句読点（SPEC Step 3-2 が挙げるもの）と、その全角/半角の相方
    "。、,.，．､｡"  # 。、,.，．､｡
    "!?！？"  # !?！？
    ":;：；"  # :;：；
    "・･"  # ・･ 中黒
    "…‥",  # …‥
    # 括弧の類
    "「」『』"  # 「」『』
    "【】〔〕〈〉《》"  # 【】〔〕〈〉《》
    "（）()"  # （）()
    "［］[]"  # ［］[]
    "｛｝{}"  # ｛｝{}
    "｢｣",  # ｢｣
    # 引用符の類
    "\"'`＂＇｀“”‘’„‟«»",
    # ダッシュ・波ダッシュ・罫線（長音符 U+30FC は含めない）
    "-"  # - ハイフンマイナス
    "‐‑‒–—―"  # ‐‑‒–—―
    "−－"  # −－
    "〜～~"  # 〜～~
    "─━┄┅"  # ─━┄┅
    "_＿",  # _＿
    # そのほか、文字起こしやアンカー語に紛れ込みがちな記号
    "/／\\＼|｜"  # /／\＼|｜
    "*＊#＃@＠&＆%％$＄+＋=＝"  # *＊#＃@＠&＆%％$＄+＋=＝
    "<>＜＞"  # <>＜＞
    "、♪※→←↑↓"  # ♪※→←↑↓
    "○●◎△▲□■☆★",  # ○●◎△▲□■☆★
)


def _build_punct_chars() -> str:
    """_PUNCT_GROUPS を重複排除して1本の文字列にする。_NEVER_STRIP は必ず外す。"""
    seen: set[str] = set()
    out: list[str] = []
    for group in _PUNCT_GROUPS:
        for ch in group:
            if ch in _NEVER_STRIP or ch in seen:
                continue
            seen.add(ch)
            out.append(ch)
    return "".join(out)


#: 正規化のときに除去する記号・句読点・空白の一覧。
PUNCT_CHARS: str = _build_punct_chars()

_PUNCT_SET: frozenset[str] = frozenset(PUNCT_CHARS)

#: 濁点・半濁点の「単独で置かれる形」→「結合文字の形」。
#: NFKC を1文字ずつかけると "か"+"゛" が "が" に合成されないので、
#: クラスタを作る前にこちらへ寄せてから NFKC をかける。
_SPACING_MARKS: dict[str, str] = {
    "゛": "゙",  # ゛ -> 結合濁点
    "゜": "゚",  # ゜ -> 結合半濁点
    "ﾞ": "゙",  # ﾞ  -> 結合濁点
    "ﾟ": "゚",  # ﾟ  -> 結合半濁点
}


def _is_combining_mark(ch: str) -> bool:
    """後続の結合文字（濁点・半濁点・アクセント）かどうか。"""
    return unicodedata.combining(ch) != 0 or ch in _SPACING_MARKS


def _iter_clusters(s: str) -> Iterator[tuple[int, str]]:
    """「基底文字＋続く結合文字」を1かたまりにして (元インデックス, かたまり) を返す。

    かたまり単位で NFKC をかけることで、"ﾊ"+"ﾟ" → "パ" のような合成を取りこぼさない。
    """
    n = len(s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and _is_combining_mark(s[j]):
            j += 1
        if j == i + 1:
            cluster = s[i]
        else:
            cluster = s[i] + "".join(_SPACING_MARKS.get(c, c) for c in s[i + 1 : j])
        yield i, cluster
        i = j


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedText:
    """正規化後の文字列と、元文字列への逆引き表。"""

    text: str
    index_map: tuple[int, ...]  # 正規化後 i 文字目 -> 元文字列のインデックス

    def __len__(self) -> int:
        return len(self.text)


def normalize(s: str) -> NormalizedText:
    """SPEC Step 3-2 の正規化。NFKC → PUNCT_CHARS の除去 → casefold。

    1文字（正確には「基底文字＋結合文字」のかたまり）ごとに処理し、
    NFKC で "㍑"→"リットル" のように展開されても、展開後の全文字へ
    同じ元インデックスを割り当てる。長音「ー」や促音「っ」には触らない。
    """
    if not s:
        return NormalizedText(text="", index_map=())

    chars: list[str] = []
    origins: list[int] = []
    for origin, cluster in _iter_clusters(s):
        for ch in unicodedata.normalize("NFKC", cluster):
            if ch in _PUNCT_SET:
                continue
            # casefold は 1 文字が複数文字になることがある（"ß" → "ss"）。
            for folded in ch.casefold():
                if folded in _PUNCT_SET:
                    continue
                chars.append(folded)
                origins.append(origin)
    return NormalizedText(text="".join(chars), index_map=tuple(origins))


def normalize_phrase(s: str) -> str:
    """アンカー語など、逆引きの要らない文字列用。normalize(s).text と同じ。"""
    return normalize(s).text


def zenkaku_length(s: str) -> int:
    """全角換算の想定文字数。タイトルの長さ判定（SPEC 6-b）に使う。

    East Asian Width が F/W/A なら1文字、それ以外は0.5文字として合計し、切り上げる。
    """
    if not s:
        return 0
    total = 0.0
    for ch in s:
        total += 1.0 if unicodedata.east_asian_width(ch) in ("F", "W", "A") else 0.5
    return int(math.ceil(total))


# ---------------------------------------------------------------------------
# フラット化（SPEC Step 3-1）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatText:
    """全単語を連結した1本のテキストと、正規化後 → 単語 index への対応表。"""

    raw: str                      # 全単語の word を連結した生テキスト
    norm: str                     # raw を正規化したもの
    norm_to_raw: tuple[int, ...]  # norm の i 文字目 -> raw のインデックス
    raw_to_word: tuple[int, ...]  # raw の i 文字目 -> words のインデックス
    words: tuple[Word, ...]

    # ----- 内部ヘルパ -----

    def _clamp_norm_index(self, i: int) -> int:
        """norm 上のインデックスを [0, len(norm)-1] に収める。"""
        if not self.norm:
            return 0
        if i < 0:
            return 0
        last = len(self.norm) - 1
        return last if i > last else i

    # ----- 参照 -----

    def word_index_at_norm(self, i: int) -> int:
        """norm 上の文字位置が属する単語の index を返す。単語が無ければ -1。"""
        if not self.norm or not self.raw_to_word:
            return -1
        raw_i = self.norm_to_raw[self._clamp_norm_index(i)]
        if raw_i < 0 or raw_i >= len(self.raw_to_word):
            return -1
        return self.raw_to_word[raw_i]

    def raw_span_for_norm(self, s: int, e: int) -> tuple[int, int]:
        """norm 上の半開区間 [s, e) に対応する raw 上の半開区間を返す。

        e <= s のときは最低1文字ぶんの区間を返す（文脈表示で潰れないようにするため）。
        """
        if not self.norm or not self.raw:
            return (0, 0)
        n = len(self.norm)
        start_idx = self._clamp_norm_index(s)
        end_idx = e if e > start_idx else start_idx + 1
        if end_idx > n:
            end_idx = n
        raw_start = self.norm_to_raw[start_idx]
        raw_end = self.norm_to_raw[end_idx - 1] + 1
        # 濁点などの結合文字は同じかたまりなので、末尾で切り落とさない。
        while raw_end < len(self.raw) and _is_combining_mark(self.raw[raw_end]):
            raw_end += 1
        if raw_end < raw_start:
            raw_end = raw_start
        return (raw_start, raw_end)

    def raw_text_for_norm(self, s: int, e: int) -> str:
        """正規化前の一致テキスト（句読点なども含む見た目どおりの文字列）。"""
        raw_start, raw_end = self.raw_span_for_norm(s, e)
        return self.raw[raw_start:raw_end]

    def context_for_norm(self, s: int, e: int, width: int = 30) -> str:
        """一致箇所の前後 width 文字を raw から取る。端が切れていれば "…" を付ける。"""
        if not self.raw:
            return ""
        raw_start, raw_end = self.raw_span_for_norm(s, e)
        lo = raw_start - width
        hi = raw_end + width
        head = "…" if lo > 0 else ""
        tail = "…" if hi < len(self.raw) else ""
        lo = max(0, lo)
        hi = min(len(self.raw), hi)
        return f"{head}{self.raw[lo:hi]}{tail}"


def build_flat(words: Sequence[Word]) -> FlatText:
    """全 words を連結して FlatText を作る（SPEC Step 3-1）。空の words でも落ちない。"""
    raw_parts: list[str] = []
    raw_to_word: list[int] = []
    for index, word in enumerate(words):
        text = word.word or ""
        if not text:
            continue
        raw_parts.append(text)
        raw_to_word.extend([index] * len(text))
    raw = "".join(raw_parts)
    normalized = normalize(raw)
    return FlatText(
        raw=raw,
        norm=normalized.text,
        norm_to_raw=normalized.index_map,
        raw_to_word=tuple(raw_to_word),
        words=tuple(words),
    )
