"""util/timeline.py — 時刻の数え方。

チャプターのずれと語尾の千切れはここの計算で決まる。
"""

from __future__ import annotations

import pytest

from radio_cutter.config import SegmentConfig
from radio_cutter.errors import ConfigError
from radio_cutter.models import Chapter, CutPoint, Word
from radio_cutter.util.timeline import (
    clamp,
    drop_last_sentence,
    expand_to_sentence,
    fmt_timestamp,
    normalize_chapters,
    parse_timestamp,
    resolve_segment_bounds,
    sentence_bounds,
    snap_to_word_boundary,
    to_final_time,
    word_index_at_time,
)


def make_words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(t, s, e) for t, s, e in spec]


SAMPLE = make_words(
    [
        ("これは", 0.0, 0.6),
        ("最初の", 0.6, 1.2),
        ("文です。", 1.2, 1.9),
        ("次の", 1.9, 2.4),
        ("文はここ", 2.4, 3.1),
        ("までです。", 3.1, 3.9),
        ("三つ目", 3.9, 4.5),
        ("の文！", 4.5, 5.2),
    ]
)


class TestFmtTimestamp:
    @pytest.mark.parametrize(
        "sec,want",
        [
            (0, "0:00"),
            (5, "0:05"),
            (32, "0:32"),
            (118, "1:58"),
            (599, "9:59"),
            (600, "10:00"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (7325.9, "2:02:05"),
        ],
    )
    def test_format(self, sec, want):
        assert fmt_timestamp(sec) == want

    def test_negative_is_zero(self):
        assert fmt_timestamp(-3) == "0:00"

    def test_roundtrip(self):
        for sec in (0, 7, 59, 60, 3599, 3600, 4000):
            assert parse_timestamp(fmt_timestamp(sec)) == float(sec)


class TestToFinalTime:
    # SPEC 6-a の式:
    #   本編内       -> Dh + (t - cut_a)
    #   エンディング -> Dh + Dm + (t - cut_b)
    DH = 30.0
    CUT_A = 12.0
    CUT_B = 3218.0
    DM = CUT_B - CUT_A

    def call(self, t: float) -> float:
        return to_final_time(t, cut_a=self.CUT_A, cut_b=self.CUT_B, highlight_dur=self.DH, main_dur=self.DM)

    def test_main_start_maps_to_highlight_end(self):
        assert self.call(self.CUT_A) == pytest.approx(30.0)

    def test_inside_main(self):
        assert self.call(112.0) == pytest.approx(130.0)

    def test_boundary_belongs_to_ending(self):
        assert self.call(self.CUT_B) == pytest.approx(self.DH + self.DM)

    def test_inside_ending(self):
        assert self.call(self.CUT_B + 100.0) == pytest.approx(self.DH + self.DM + 100.0)

    def test_monotonic(self):
        prev = -1.0
        for t in (12.0, 100.0, 3217.9, 3218.0, 3300.0):
            cur = self.call(t)
            assert cur > prev
            prev = cur


class TestWordSnapping:
    def test_word_index_at_time_start_is_next_boundary(self):
        # kind="start" は「t 以上で最も近い単語」
        assert word_index_at_time(SAMPLE, 2.4, kind="start") == 4
        assert word_index_at_time(SAMPLE, 2.5, kind="start") == 5

    def test_word_index_at_time_end_is_previous_boundary(self):
        # kind="end" は「t 以下で最も近い単語」
        assert word_index_at_time(SAMPLE, 2.5, kind="end") == 3

    def test_word_index_clamps_past_the_tail(self):
        assert word_index_at_time(SAMPLE, 999.0, kind="start") == len(SAMPLE) - 1
        assert word_index_at_time(SAMPLE, -5.0, kind="end") == 0

    def test_snap_start_picks_nearest_word_start(self):
        # 2.5 の左右は 2.4 と 3.1。近いほう（かつ語頭が欠けないほう）を採る。
        assert snap_to_word_boundary(SAMPLE, 2.5, kind="start") == pytest.approx(2.4)

    def test_snap_end_picks_nearest_word_end(self):
        # 3.0 の左右の end は 2.4 と 3.1。近いのは 3.1。
        assert snap_to_word_boundary(SAMPLE, 3.0, kind="end") == pytest.approx(3.1)

    def test_snap_is_stable_on_exact_boundary(self):
        assert snap_to_word_boundary(SAMPLE, 1.9, kind="start") == pytest.approx(1.9)
        assert snap_to_word_boundary(SAMPLE, 3.9, kind="end") == pytest.approx(3.9)

    def test_empty_words_returns_input(self):
        assert snap_to_word_boundary([], 4.2, kind="start") == 4.2

    def test_word_index_empty_is_minus_one(self):
        assert word_index_at_time([], 1.0, kind="start") == -1


class TestSentences:
    def test_sentence_bounds_expands_both_ways(self):
        # 4番目の単語（"文はここ"）は2文目の途中
        lo, hi = sentence_bounds(SAMPLE, 4, 4)
        assert SAMPLE[lo].word == "次の"
        assert SAMPLE[hi].word == "までです。"

    def test_expand_to_sentence_widens_range(self):
        start, end = expand_to_sentence(SAMPLE, 2.6, 3.0)
        assert start <= 1.9
        assert end >= 3.9

    def test_expand_is_idempotent(self):
        first = expand_to_sentence(SAMPLE, 2.6, 3.0)
        second = expand_to_sentence(SAMPLE, *first)
        assert second == first

    def test_drop_last_sentence(self):
        start, end = expand_to_sentence(SAMPLE, 0.1, 3.5)
        dropped = drop_last_sentence(SAMPLE, start, end)
        assert dropped is not None
        assert dropped[1] < end

    def test_drop_last_sentence_returns_none_when_single(self):
        start, end = expand_to_sentence(SAMPLE, 0.1, 1.0)
        assert drop_last_sentence(SAMPLE, start, end) is None

    def test_exclamation_terminates(self):
        lo, hi = sentence_bounds(SAMPLE, 7, 7)
        assert SAMPLE[lo].word == "三つ目"


class TestNormalizeChapters:
    def test_first_chapter_forced_to_zero(self):
        chapters = [Chapter(5.0, "冒頭"), Chapter(40.0, "本題"), Chapter(80.0, "まとめ")]
        fixed, warnings = normalize_chapters(chapters, 200.0)
        assert fixed[0].time_sec == 0.0
        assert fixed[0].label == "冒頭"

    def test_sorted_ascending(self):
        chapters = [Chapter(0.0, "a"), Chapter(90.0, "c"), Chapter(40.0, "b")]
        fixed, _ = normalize_chapters(chapters, 200.0)
        assert [c.time_sec for c in fixed] == sorted(c.time_sec for c in fixed)

    def test_drops_chapters_closer_than_min_gap(self):
        chapters = [Chapter(0.0, "a"), Chapter(4.0, "b"), Chapter(40.0, "c"), Chapter(70.0, "d")]
        fixed, _ = normalize_chapters(chapters, 200.0, min_gap=10.0)
        assert 4.0 not in [c.time_sec for c in fixed]
        assert [c.label for c in fixed][:2] == ["a", "c"]

    def test_drops_chapters_past_duration(self):
        chapters = [Chapter(0.0, "a"), Chapter(40.0, "b"), Chapter(500.0, "c")]
        fixed, _ = normalize_chapters(chapters, 200.0)
        assert all(c.time_sec < 200.0 for c in fixed)

    def test_warns_when_fewer_than_three(self):
        chapters = [Chapter(0.0, "a"), Chapter(40.0, "b")]
        fixed, warnings = normalize_chapters(chapters, 200.0)
        assert warnings, "3つ未満なら警告が要る"

    def test_no_exception_on_empty(self):
        fixed, warnings = normalize_chapters([], 200.0)
        assert fixed == []
        assert warnings


class TestResolveSegmentBounds:
    CUTS = {
        "A": CutPoint("A", 12.9, 12.6, True),
        "B": CutPoint("B", 3218.1, 3217.8, True),
    }

    def test_anchor_to_anchor(self):
        seg = SegmentConfig(name="main", file="02_main.mp4", from_="A", to="B")
        assert resolve_segment_bounds(seg, self.CUTS, 3612.4) == (12.6, 3217.8)

    def test_anchor_to_end(self):
        seg = SegmentConfig(name="ending", file="03_ending.mp4", from_="B", to="end")
        assert resolve_segment_bounds(seg, self.CUTS, 3612.4) == (3217.8, 3612.4)

    def test_start_to_anchor(self):
        seg = SegmentConfig(name="intro", file="00_intro.mp4", from_="start", to="A")
        assert resolve_segment_bounds(seg, self.CUTS, 3612.4) == (0.0, 12.6)

    def test_unknown_anchor_raises(self):
        seg = SegmentConfig(name="x", file="x.mp4", from_="Z", to="end")
        with pytest.raises(ConfigError):
            resolve_segment_bounds(seg, self.CUTS, 3612.4)

    def test_inverted_raises(self):
        seg = SegmentConfig(name="x", file="x.mp4", from_="B", to="A")
        with pytest.raises(ConfigError):
            resolve_segment_bounds(seg, self.CUTS, 3612.4)


def test_clamp():
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert clamp(-1.0, 0.0, 10.0) == 0.0
    assert clamp(99.0, 0.0, 10.0) == 10.0
