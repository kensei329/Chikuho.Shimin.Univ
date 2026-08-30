"""util/text_normalize.py — 正規化とインデックス対応の検証。

SPEC Step 3-2 の「正規化後のインデックスから元インデックスへ戻せること」が
壊れると、アンカーの時刻が丸ごとずれる。ここは厳しめに縛る。
"""

from __future__ import annotations

import pytest

from radio_cutter.models import Word
from radio_cutter.util.text_normalize import (
    NormalizedText,
    build_flat,
    normalize,
    normalize_phrase,
    zenkaku_length,
)


class TestNormalize:
    def test_removes_punctuation_and_space(self):
        got = normalize("この、チャンネル は。").text
        assert got == "このチャンネルは"

    def test_nfkc_unifies_width(self):
        assert normalize_phrase("ＡＩ活用") == normalize_phrase("AI活用")
        assert normalize_phrase("ｶﾀｶﾅ") == normalize_phrase("カタカナ")

    def test_keeps_long_vowel_mark(self):
        # 長音の揺れは触らない（過剰正規化は誤検出を招く）
        assert "ー" in normalize_phrase("メンバーシップ")
        assert normalize_phrase("コンピュータ") != normalize_phrase("コンピューター")

    def test_index_map_points_back_to_source(self):
        src = "この、チャンネルは。"
        n = normalize(src)
        assert len(n.index_map) == len(n.text)
        for i, ch in enumerate(n.text):
            origin = src[n.index_map[i]]
            # 正規化で形が変わらない文字はそのまま一致するはず
            assert normalize_phrase(origin).startswith(ch) or ch in normalize_phrase(origin)

    def test_index_map_is_monotonic(self):
        n = normalize("えー、じゃあ始めます。このチャンネルは")
        assert list(n.index_map) == sorted(n.index_map)

    def test_nfkc_expansion_keeps_mapping(self):
        # NFKC で1文字が複数文字に開く例。どの文字も元の1文字を指す。
        n = normalize("㍑と①")
        assert n.text  # 空にはならない
        assert len(n.index_map) == len(n.text)
        assert max(n.index_map) < len("㍑と①")

    def test_empty_input(self):
        n = normalize("")
        assert n.text == ""
        assert tuple(n.index_map) == ()

    def test_all_punctuation_input(self):
        assert normalize("。、！？ 　").text == ""

    def test_returns_normalized_text_type(self):
        assert isinstance(normalize("あ"), NormalizedText)


class TestZenkakuLength:
    def test_japanese_counts_one_each(self):
        assert zenkaku_length("あいうえお") == 5

    def test_ascii_counts_half(self):
        assert zenkaku_length("abcd") == 2

    def test_mixed_rounds_up(self):
        # 全角2 + 半角1(0.5) = 2.5 -> 3
        assert zenkaku_length("あいa") == 3

    def test_empty(self):
        assert zenkaku_length("") == 0


class TestFlatText:
    @staticmethod
    def words() -> list[Word]:
        return [
            Word("えー、", 0.0, 0.5),
            Word("この", 0.5, 0.9),
            Word("チャンネル", 0.9, 1.5),
            Word("は", 1.5, 1.7),
            Word("AIの", 1.7, 2.2),
        ]

    def test_raw_is_concatenation(self):
        flat = build_flat(self.words())
        assert flat.raw == "えー、このチャンネルはAIの"

    def test_norm_drops_punctuation(self):
        flat = build_flat(self.words())
        assert "、" not in flat.norm

    def test_word_index_at_norm(self):
        flat = build_flat(self.words())
        pos = flat.norm.find(normalize_phrase("このチャンネルは"))
        assert pos >= 0
        assert flat.words[flat.word_index_at_norm(pos)].word == "この"

    def test_raw_text_for_norm_returns_original(self):
        flat = build_flat(self.words())
        phrase = normalize_phrase("このチャンネルは")
        pos = flat.norm.find(phrase)
        assert flat.raw_text_for_norm(pos, pos + len(phrase)) == "このチャンネルは"

    def test_context_includes_surroundings(self):
        flat = build_flat(self.words())
        phrase = normalize_phrase("チャンネル")
        pos = flat.norm.find(phrase)
        ctx = flat.context_for_norm(pos, pos + len(phrase), width=3)
        assert "チャンネル" in ctx

    def test_empty_words(self):
        flat = build_flat([])
        assert flat.raw == ""
        assert flat.norm == ""
        assert flat.words == ()

    def test_word_index_never_out_of_range(self):
        flat = build_flat(self.words())
        for i in range(len(flat.norm)):
            assert 0 <= flat.word_index_at_norm(i) < len(flat.words)
