"""steps/s4_refine_cuts.py — SPEC Step 4「カット点の精密化」。

ここが狂うと Phase 1 の受け入れ基準（SPEC 10章）の
「`02_main.mp4` の冒頭が『このチャンネルは』で始まっている」「語頭が欠けていない」
が丸ごと崩れる。SPEC Step 4 が定める規則はこの4つ。

1. `raw_cut_time` **より前**にある**最後の**無音区間を採用する
2. 採用した区間の **終了時刻 - 50ms** を `cut_time` にする
3. 無音が見つからなければ `cut_time = raw_cut_time - 0.08` に落とし、
   `silence_found: false` を記録する（decisions.json / warnings に残す）
4. しきい値 `-32dB` と `d=0.12` は CLI から上書きできる（`--silence-db` / `--silence-dur`）

`pick_cut_time()` は ffmpeg を触らない純関数なので、規則1〜3はここで直接叩く。
規則4と実音声での挙動は `run()` に `@pytest.mark.ffmpeg` を付けて確かめる。
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import (
    NO_SILENCE_BACKOFF_SEC,
    SILENCE_BACKOFF_SEC,
    SILENCE_LOOKAHEAD_SEC,
    SILENCE_LOOKBACK_SEC,
    SilenceConfig,
)
from radio_cutter.context import RunContext
from radio_cutter.errors import MissingArtifactError
from radio_cutter.models import AnchorResult, CutPoint
from radio_cutter.steps import s1_extract_audio
from radio_cutter.steps import s4_refine_cuts as s4


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------


def make_anchor(
    anchor_id: str,
    raw: float,
    *,
    phrase: str = "このチャンネルは",
    score: float = 100.0,
) -> AnchorResult:
    """Step 3 が返す形の AnchorResult を1つ作る（Step 4 が見るのは id / raw_cut_time / score だけ）。"""
    return AnchorResult(
        id=anchor_id,
        phrase=phrase,
        matched_text=phrase,
        score=score,
        raw_cut_time=raw,
        candidates_found=1,
        candidates_rejected=0,
        context="…テスト用の文脈…",
    )


def install_audio(ctx: RunContext, **kwargs) -> Path:
    """Step 1 が置くはずの work/<ep>/audio.wav を合成音声で用意する。"""
    ctx.ensure_dirs()
    return fixtures.write_tone_wav(ctx.work_path(s1_extract_audio.AUDIO_FILENAME), **kwargs)


def write_quiet_gap_wav(
    path: str | Path,
    *,
    gap: tuple[float, float],
    duration: float = 8.0,
    loud: float = 0.35,
    quiet: float = 0.003,
    sample_rate: int = 16000,
    freq: float = 440.0,
) -> Path:
    """`gap` の区間だけ音量を落とした（無音にはしない）正弦波 WAV。

    `--silence-db` が効いているかを見るための音源。fixtures の合成音声は谷が完全な 0 なので、
    しきい値をどれだけ下げても無音として拾えてしまい、dB の上書きが効いたかどうかを判別できない。
    ここでは谷を約 -50dB の小さな音にしてあるので、
    しきい値 -32dB では「無音」、-60dB では「無音ではない」と判定が割れる。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    gap_start, gap_end = gap
    frames = bytearray()
    two_pi_f = 2.0 * math.pi * freq
    for n in range(int(duration * sample_rate)):
        t = n / sample_rate
        amp = quiet if gap_start <= t < gap_end else loud
        frames += struct.pack("<h", int(amp * 32767 * math.sin(two_pi_f * n / sample_rate)))
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return p


# ---------------------------------------------------------------------------
# 定数（SPEC Step 4 が名指しで決めている値）
# ---------------------------------------------------------------------------


class TestStepConstants:
    """SPEC が数値で決めている定数が、そのままの値で置かれていること。"""

    def test_backoff_values_match_spec(self):
        """採用時は終了 -50ms、無音なしのフォールバックは raw -80ms（SPEC Step 4）。"""
        assert SILENCE_BACKOFF_SEC == 0.05
        assert NO_SILENCE_BACKOFF_SEC == 0.08

    def test_search_window_values_match_spec(self):
        """探索窓は raw-1.5秒 〜 raw+0.5秒（SPEC Step 4 の atrim 式）。"""
        assert SILENCE_LOOKBACK_SEC == 1.5
        assert SILENCE_LOOKAHEAD_SEC == 0.5

    def test_default_silence_thresholds_match_spec(self):
        """既定のしきい値は -32dB / d=0.12（SPEC 7章 CLI の既定値）。"""
        cfg = SilenceConfig()
        assert cfg.noise_db == -32.0
        assert cfg.min_duration_sec == 0.12

    def test_step_identity(self):
        """パイプラインが参照する STEP / 中間ファイル名。"""
        assert s4.STEP == 4
        assert s4.CUTS_FILE == "cuts.json"
        assert s4.CUTS_FILE in s4.OUTPUTS

    def test_cuts_path_is_under_work_dir(self, ctx):
        """cuts.json は work/<episode_id>/ に置く（SPEC 3章「中間ファイルは work に残す」）。"""
        assert s4.cuts_path(ctx) == ctx.work_dir / "cuts.json"


# ---------------------------------------------------------------------------
# search_window
# ---------------------------------------------------------------------------


class TestSearchWindow:
    """SPEC Step 4 の `atrim=start={raw-1.5}:end={raw+0.5}`。"""

    def test_window_is_raw_minus_1_5_to_plus_0_5(self):
        assert s4.search_window(10.0) == (8.5, 10.5)

    def test_window_start_is_clamped_at_zero(self):
        """先頭付近のアンカーで負の開始時刻を ffmpeg に渡さない。"""
        assert s4.search_window(1.0) == (0.0, 1.5)
        assert s4.search_window(0.0) == (0.0, 0.5)

    def test_window_covers_the_raw_time(self):
        """窓は必ず raw を含む（含まなければ谷を探しようがない）。"""
        for raw in (0.0, 0.3, 5.95, 43.95, 3218.06):
            lo, hi = s4.search_window(raw)
            assert lo <= raw <= hi


# ---------------------------------------------------------------------------
# pick_cut_time — 規則1〜3
# ---------------------------------------------------------------------------


class TestPickCutTime:
    """「raw より前にある最後の無音の、終了 -50ms」（SPEC Step 4）。"""

    def test_uses_the_last_silence_before_raw(self):
        """raw より前の無音が複数あるとき、いちばん後ろのものを採る。"""
        silences = [(1.0, 1.4), (3.0, 3.5), (5.60, 5.95)]
        cut, found, span = s4.pick_cut_time(6.30, silences)
        assert found is True
        assert span == (5.60, 5.95)
        assert cut == pytest.approx(5.90)

    def test_silences_after_raw_are_ignored_when_choosing(self):
        """raw より後ろの谷は「最後の無音」の候補にしない。前の谷が残っていればそれを採る。"""
        silences = [(3.0, 3.5), (7.0, 7.4), (9.0, 9.6)]
        cut, found, span = s4.pick_cut_time(5.0, silences)
        assert found is True
        assert span == (3.0, 3.5)
        assert cut == pytest.approx(3.45)

    def test_silence_ending_exactly_at_raw_is_adopted(self):
        """無音の終わりがちょうど raw と等しいケースは採用する（発話の立ち上がり直前で谷が閉じた形）。"""
        cut, found, span = s4.pick_cut_time(5.95, [(5.60, 5.95)])
        assert found is True
        assert span == (5.60, 5.95)
        assert cut == pytest.approx(5.90)

    def test_cut_time_is_silence_end_minus_50ms(self):
        """採用値は必ず「無音の終了 - 0.05秒」。開始時刻や長さには依存しない。"""
        for start, end in [(0.5, 2.0), (1.999, 2.0), (10.0, 12.345)]:
            cut, found, _span = s4.pick_cut_time(20.0, [(start, end)])
            assert found is True
            assert cut == pytest.approx(end - SILENCE_BACKOFF_SEC, abs=1e-9)

    def test_only_silences_after_raw_falls_back(self):
        """raw より後の無音しか無ければ、その谷に寄せず raw - 0.08 にフォールバックする。"""
        cut, found, span = s4.pick_cut_time(5.0, [(5.2, 5.6), (7.0, 7.5)])
        assert found is False
        assert span is None
        assert cut == pytest.approx(4.92)

    def test_empty_silences_falls_back(self):
        """無音が1つも無ければフォールバック。勝手に遠くの谷を探しに行かない。"""
        cut, found, span = s4.pick_cut_time(12.84, [])
        assert found is False
        assert span is None
        assert cut == pytest.approx(12.76)

    def test_fallback_amount_is_exactly_80ms(self):
        """フォールバック量は 0.08秒。raw - 0.08 以外の値を勝手に選ばない。"""
        for raw in (1.0, 5.95, 43.95, 3218.06):
            cut, found, _ = s4.pick_cut_time(raw, [])
            assert found is False
            assert cut == pytest.approx(raw - NO_SILENCE_BACKOFF_SEC, abs=1e-9)

    def test_cut_time_never_goes_negative_on_fallback(self):
        """raw が 0.08 未満でも負の秒数を返さない（ffmpeg の -ss に負値は渡せない）。"""
        cut, found, span = s4.pick_cut_time(0.01, [])
        assert found is False
        assert span is None
        assert cut == 0.0

    def test_cut_time_never_goes_negative_at_zero_raw(self):
        cut, found, _ = s4.pick_cut_time(0.0, [])
        assert found is False
        assert cut == 0.0

    def test_cut_time_never_goes_negative_when_silence_is_at_the_head(self):
        """冒頭の谷（終了 < 0.05秒）に寄せても 0 未満にしない。無音は見つかっている扱いのまま。"""
        cut, found, span = s4.pick_cut_time(1.0, [(0.0, 0.03)])
        assert found is True
        assert span == (0.0, 0.03)
        assert cut == 0.0

    def test_returned_time_is_rounded_to_milliseconds(self):
        """秒数は小数点以下3桁（SPEC 11章）。0.05 / 0.08 の減算で出る二進小数の端数を持ち回らない。

        43.95 - 0.05 は素の float だと 43.900000000000006 になる。
        丸めずに JSON へ書くと、decisions.json の秒数が読めない桁数で並ぶ。
        """
        cut, _found, _span = s4.pick_cut_time(50.0, [(43.60, 43.95)])
        assert cut == 43.9
        assert repr(cut) == "43.9"

        # フォールバック側も同じ（0.29 - 0.08 は 0.20999999999999996）。
        cut2, _f, _s = s4.pick_cut_time(0.29, [])
        assert cut2 == 0.21
        assert repr(cut2) == "0.21"

    def test_unsorted_silences_still_pick_the_last_one(self):
        """silencedetect のログ順に依存しない。並びが乱れていても最後の谷を採る。"""
        silences = [(5.60, 5.95), (1.0, 1.4), (3.0, 3.5)]
        cut, found, span = s4.pick_cut_time(6.0, silences)
        assert found is True
        assert span == (5.60, 5.95)
        assert cut == pytest.approx(5.90)

    def test_zero_length_silence_is_still_a_silence(self):
        """長さ0の区間でも「谷が見つかった」扱い。終了時刻さえあれば寄せ先は決まる。"""
        cut, found, span = s4.pick_cut_time(4.0, [(3.5, 3.5)])
        assert found is True
        assert span == (3.5, 3.5)
        assert cut == pytest.approx(3.45)

    def test_broken_span_with_end_before_start_is_discarded(self):
        """終了が開始より前の壊れた区間は採用しない（採ると尺が逆転する）。"""
        cut, found, span = s4.pick_cut_time(6.0, [(1.0, 1.4), (5.9, 5.2)])
        assert found is True
        assert span == (1.0, 1.4)
        assert cut == pytest.approx(1.35)

    def test_only_broken_spans_falls_back(self):
        cut, found, span = s4.pick_cut_time(6.0, [(5.9, 5.2)])
        assert found is False
        assert span is None
        assert cut == pytest.approx(5.92)

    def test_cut_time_is_never_after_raw(self):
        """語頭を削らないための不変条件: cut_time <= raw_cut_time（SPEC 10章 Phase 1 受け入れ基準）。"""
        cases = [
            (5.95, [(5.60, 5.95)]),
            (5.95, []),
            (5.95, [(6.0, 6.4)]),
            (5.95, [(1.0, 1.2), (5.0, 5.4)]),
            (0.02, []),
        ]
        for raw, silences in cases:
            cut, _found, _span = s4.pick_cut_time(raw, silences)
            assert cut <= raw

    def test_accepts_any_sequence_of_pairs(self):
        """タプルのリストでもタプルのタプルでも、int が混ざっても同じ答えになる。"""
        assert s4.pick_cut_time(6.0, ((1, 2), (3, 4)))[0] == pytest.approx(3.95)

    @pytest.mark.parametrize(
        "raw,expected_cut",
        [
            (fixtures.EXPECTED_ANCHOR_A_RAW, fixtures.EXPECTED_CUT_A),
            (fixtures.EXPECTED_ANCHOR_B_RAW, fixtures.EXPECTED_CUT_B),
        ],
    )
    def test_matches_the_synthetic_episode_expectations(self, raw, expected_cut):
        """合成エピソードの時刻表を丸ごと渡しても、期待どおりの谷に寄る（ffmpeg を通さない検算）。"""
        cut, found, _span = s4.pick_cut_time(raw, fixtures.SILENCES)
        assert found is True
        assert cut == expected_cut

    def test_decoy_silence_does_not_leak_into_anchor_b(self):
        """アンカーB（43.95）の寄せ先は 43.60-43.95 の谷。おとり（19.70-19.95）を選ばない。"""
        _cut, _found, span = s4.pick_cut_time(fixtures.EXPECTED_ANCHOR_B_RAW, fixtures.SILENCES)
        assert span == (43.60, 43.95)


class TestNoSilenceWarning:
    """無音なしの警告文（decisions.json の warnings に載る／SPEC 8章）。"""

    def test_message_names_the_anchor_and_the_fallback_amount(self):
        msg = s4.no_silence_warning("B")
        assert "B" in msg
        assert "0.08" in msg

    def test_messages_differ_per_anchor(self):
        """アンカーごとに別の文言になる（RunContext.warn は同じ文言を1回しか残さないため）。"""
        assert s4.no_silence_warning("A") != s4.no_silence_warning("B")


# ---------------------------------------------------------------------------
# run() — 引数の検証と設定の伝播（ffmpeg 不要）
# ---------------------------------------------------------------------------


class TestRunPreconditions:
    def test_empty_anchors_raises(self, ctx):
        """アンカーが空なら Step 3 を先に回せと言って止まる。無言で空の cuts.json を作らない。"""
        with pytest.raises(MissingArtifactError):
            s4.run(ctx, {})
        assert not s4.cuts_path(ctx).exists()

    def test_missing_audio_raises(self, ctx):
        """audio.wav が無ければ Step 1 を案内して止まる（SPEC 9章「握りつぶさない」）。"""
        ctx.ensure_dirs()
        with pytest.raises(MissingArtifactError) as excinfo:
            s4.run(ctx, {"A": make_anchor("A", 5.95)})
        assert s1_extract_audio.AUDIO_FILENAME in str(excinfo.value)


class TestSilenceConfigPassThrough:
    """`--silence-db` / `--silence-dur` が silencedetect まで届いているか（SPEC Step 4 末尾・7章）。"""

    def _stub_detector(self, monkeypatch, spans):
        calls: list[dict] = []

        def fake_detect(wav_path, *, start, end, noise_db, min_duration):
            calls.append(
                {
                    "path": Path(wav_path),
                    "start": start,
                    "end": end,
                    "noise_db": noise_db,
                    "min_duration": min_duration,
                }
            )
            return list(spans)

        monkeypatch.setattr(s4, "detect_silences", fake_detect)
        return calls

    def test_overridden_thresholds_reach_the_detector(self, ctx, monkeypatch):
        """ctx.silence の値がそのまま silencedetect の n / d に渡る。"""
        calls = self._stub_detector(monkeypatch, [(5.60, 5.95)])
        ctx.silence = SilenceConfig(noise_db=-45.0, min_duration_sec=0.3)
        ctx.ensure_dirs()
        ctx.work_path(s1_extract_audio.AUDIO_FILENAME).write_bytes(b"")

        cuts = s4.run(ctx, {"A": make_anchor("A", 5.95)})

        assert len(calls) == 1
        assert calls[0]["noise_db"] == -45.0
        assert calls[0]["min_duration"] == 0.3
        assert cuts["A"].cut_time == pytest.approx(5.90)

    def test_detector_is_called_on_the_step1_audio_with_the_spec_window(self, ctx, monkeypatch):
        """見る音声は work/audio.wav、窓は [raw-1.5, raw+0.5]（60分を丸ごと走査しない）。"""
        calls = self._stub_detector(monkeypatch, [])
        ctx.ensure_dirs()
        ctx.work_path(s1_extract_audio.AUDIO_FILENAME).write_bytes(b"")

        s4.run(ctx, {"A": make_anchor("A", 10.0)})

        assert calls[0]["path"] == ctx.work_path(s1_extract_audio.AUDIO_FILENAME)
        assert calls[0]["start"] == pytest.approx(8.5)
        assert calls[0]["end"] == pytest.approx(10.5)

    def test_each_anchor_gets_its_own_detection(self, ctx, monkeypatch):
        """アンカーごとに窓を切り直す（1回の検出結果を使い回さない）。"""
        calls = self._stub_detector(monkeypatch, [])
        ctx.ensure_dirs()
        ctx.work_path(s1_extract_audio.AUDIO_FILENAME).write_bytes(b"")

        s4.run(ctx, {"A": make_anchor("A", 5.95), "B": make_anchor("B", 43.95)})

        assert [c["start"] for c in calls] == [pytest.approx(4.45), pytest.approx(42.45)]


class TestRunResultShape:
    """cuts.json の中身と warnings（ffmpeg は使わず検出結果だけ差し替える）。"""

    def _run_with(self, ctx, monkeypatch, spans, anchors):
        monkeypatch.setattr(
            s4,
            "detect_silences",
            lambda wav_path, *, start, end, noise_db, min_duration: list(spans),
        )
        ctx.ensure_dirs()
        ctx.work_path(s1_extract_audio.AUDIO_FILENAME).write_bytes(b"")
        return s4.run(ctx, anchors)

    def test_records_the_adopted_silence_span(self, ctx, monkeypatch):
        """どの谷を採ったかを残す（あとから何が起きたか追えるようにする／SPEC 11章）。"""
        cuts = self._run_with(ctx, monkeypatch, [(5.60, 5.95)], {"A": make_anchor("A", 5.95)})
        cut = cuts["A"]
        assert cut.anchor_id == "A"
        assert cut.raw_cut_time == pytest.approx(5.95)
        assert cut.cut_time == pytest.approx(5.90)
        assert cut.silence_found is True
        assert cut.silence_start == pytest.approx(5.60)
        assert cut.silence_end == pytest.approx(5.95)

    def test_carries_the_anchor_score_into_decisions(self, ctx, monkeypatch):
        """decisions.json の anchors.* は score も持つ（SPEC 8章のスキーマ）。"""
        anchors = {"B": make_anchor("B", 43.95, phrase="ということで", score=96.2)}
        cuts = self._run_with(ctx, monkeypatch, [(43.60, 43.95)], anchors)
        assert cuts["B"].score == pytest.approx(96.2)

    def test_no_silence_sets_flag_and_warning(self, ctx, monkeypatch):
        """無音なしなら silence_found=False と warnings の両方に残す（SPEC Step 4 / 8章）。"""
        cuts = self._run_with(ctx, monkeypatch, [], {"A": make_anchor("A", 5.95)})
        assert cuts["A"].silence_found is False
        assert cuts["A"].cut_time == pytest.approx(5.87)
        assert cuts["A"].silence_start is None
        assert cuts["A"].silence_end is None
        assert any("A" in w for w in ctx.warnings)

    def test_no_silence_warns_once_per_anchor(self, ctx, monkeypatch):
        """アンカーが2つとも外したら警告も2件（1件に潰れない）。"""
        anchors = {"A": make_anchor("A", 5.95), "B": make_anchor("B", 43.95)}
        self._run_with(ctx, monkeypatch, [], anchors)
        assert len(ctx.warnings) == 2

    def test_silence_found_leaves_no_warning(self, ctx, monkeypatch):
        """谷に寄せられたときは警告を出さない（本来の正常系）。"""
        self._run_with(ctx, monkeypatch, [(5.60, 5.95)], {"A": make_anchor("A", 5.95)})
        assert ctx.warnings == []

    def test_result_is_ordered_by_config(self, ctx, monkeypatch):
        """返り値と cuts.json のキー順は config の anchors 順（decisions.json が A→B で並ぶ）。"""
        anchors = {"B": make_anchor("B", 43.95), "A": make_anchor("A", 5.95)}
        cuts = self._run_with(ctx, monkeypatch, [(1.0, 1.4)], anchors)
        assert list(cuts) == ["A", "B"]
        assert list(json.loads(s4.cuts_path(ctx).read_text(encoding="utf-8"))) == ["A", "B"]

    def test_unknown_anchor_id_is_not_dropped(self, ctx, monkeypatch):
        """config に無い ID のアンカーも取りこぼさず処理する。"""
        anchors = {"A": make_anchor("A", 5.95), "Z": make_anchor("Z", 30.0)}
        cuts = self._run_with(ctx, monkeypatch, [(1.0, 1.4)], anchors)
        assert set(cuts) == {"A", "Z"}

    def test_cuts_json_is_written_and_reloadable(self, ctx, monkeypatch):
        """cuts.json を書き、load() で同じ値に戻せる（--from-step 5 以降の再開に必要）。"""
        anchors = {"A": make_anchor("A", 5.95), "B": make_anchor("B", 43.95, score=96.2)}
        cuts = self._run_with(ctx, monkeypatch, [(5.60, 5.95)], anchors)

        path = s4.cuts_path(ctx)
        assert path.exists()

        loaded = s4.load(ctx)
        assert set(loaded) == set(cuts)
        for anchor_id, cut in cuts.items():
            got = loaded[anchor_id]
            assert got.anchor_id == anchor_id
            assert got.raw_cut_time == pytest.approx(cut.raw_cut_time)
            assert got.cut_time == pytest.approx(cut.cut_time)
            assert got.silence_found == cut.silence_found
            assert got.score == pytest.approx(cut.score)
            assert got.silence_start == pytest.approx(cut.silence_start)
            assert got.silence_end == pytest.approx(cut.silence_end)

    def test_cuts_json_times_are_rounded_to_three_decimals(self, ctx, monkeypatch):
        """JSON に書く秒数は小数点以下3桁（SPEC 11章）。"""
        self._run_with(ctx, monkeypatch, [(5.6, 5.951234)], {"A": make_anchor("A", 5.96)})
        payload = json.loads(s4.cuts_path(ctx).read_text(encoding="utf-8"))["A"]
        for key in ("raw_cut_time", "cut_time", "silence_start", "silence_end"):
            assert round(payload[key], 3) == payload[key]

    def test_run_creates_the_work_dir(self, tmp_path, config, monkeypatch):
        """work/<episode_id>/ が無い状態から呼ばれても自分で掘る。"""
        input_path = tmp_path / "ep-test.mp4"
        input_path.write_bytes(b"")
        ctx = RunContext(
            input_path=input_path,
            episode_id="ep-test",
            work_dir=tmp_path / "work" / "ep-test",
            out_dir=tmp_path / "out" / "ep-test",
            config=config,
            silence=SilenceConfig(),
        )
        assert not ctx.work_dir.exists()
        self._run_with(ctx, monkeypatch, [(5.60, 5.95)], {"A": make_anchor("A", 5.95)})
        assert s4.cuts_path(ctx).exists()


class TestLoad:
    """--from-step で再開するときの読み込み。壊れた中間ファイルは黙って通さない。"""

    def test_missing_file_raises_with_a_hint(self, ctx):
        ctx.ensure_dirs()
        with pytest.raises(MissingArtifactError) as excinfo:
            s4.load(ctx)
        assert "cuts.json" in str(excinfo.value)

    def test_broken_json_raises(self, ctx):
        ctx.ensure_dirs()
        s4.cuts_path(ctx).write_text("{ これは JSON ではない", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s4.load(ctx)

    def test_empty_object_raises(self, ctx):
        ctx.ensure_dirs()
        s4.cuts_path(ctx).write_text("{}", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s4.load(ctx)

    def test_non_object_entry_raises(self, ctx):
        ctx.ensure_dirs()
        s4.cuts_path(ctx).write_text('{"A": 5.9}', encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s4.load(ctx)

    def test_entry_missing_required_key_raises(self, ctx):
        """cut_time の無いエントリを 0 秒として通してしまうと、無言で先頭からカットされる。"""
        ctx.ensure_dirs()
        s4.cuts_path(ctx).write_text('{"A": {"raw_cut_time": 5.95}}', encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s4.load(ctx)

    def test_save_then_load_roundtrips(self, ctx):
        """save() → load() が値を保つ（silence_start/end が None のケースも含む）。"""
        ctx.ensure_dirs()
        original = {
            "A": CutPoint(
                anchor_id="A",
                raw_cut_time=5.95,
                cut_time=5.90,
                silence_found=True,
                score=100.0,
                silence_start=5.60,
                silence_end=5.95,
            ),
            "B": CutPoint(
                anchor_id="B",
                raw_cut_time=43.95,
                cut_time=43.87,
                silence_found=False,
                score=96.2,
            ),
        }
        s4.save(ctx, original)
        loaded = s4.load(ctx)
        assert list(loaded) == ["A", "B"]
        assert loaded["A"].silence_end == pytest.approx(5.95)
        assert loaded["B"].silence_found is False
        assert loaded["B"].silence_start is None
        assert loaded["B"].cut_time == pytest.approx(43.87)


# ---------------------------------------------------------------------------
# run() — 本物の ffmpeg / 合成音声
# ---------------------------------------------------------------------------


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestRunWithRealAudio:
    """合成エピソードの音声に silencedetect を実際にかける。ここが Phase 1 の要。"""

    def test_both_anchors_snap_to_the_expected_silence(self, ctx):
        """cut_A / cut_B が fixtures の期待値と 0.05 秒以内で一致する。"""
        install_audio(ctx)
        anchors = {
            "A": make_anchor("A", fixtures.EXPECTED_ANCHOR_A_RAW),
            "B": make_anchor("B", fixtures.EXPECTED_ANCHOR_B_RAW, phrase="ということで", score=96.2),
        }
        cuts = s4.run(ctx, anchors)

        assert cuts["A"].silence_found is True
        assert cuts["B"].silence_found is True
        assert cuts["A"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_A, abs=0.05)
        assert cuts["B"].cut_time == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.05)
        assert ctx.warnings == []

    def test_cut_lands_before_the_word_onset(self, ctx):
        """語頭が欠けない向きに寄る（SPEC 10章 Phase 1 受け入れ基準）。

        raw より後ろに出てはいけないし、無闇に遠くまで戻ってもいけない。
        """
        install_audio(ctx)
        cuts = s4.run(
            ctx,
            {
                "A": make_anchor("A", fixtures.EXPECTED_ANCHOR_A_RAW),
                "B": make_anchor("B", fixtures.EXPECTED_ANCHOR_B_RAW),
            },
        )
        for anchor_id, raw in (("A", fixtures.EXPECTED_ANCHOR_A_RAW), ("B", fixtures.EXPECTED_ANCHOR_B_RAW)):
            cut = cuts[anchor_id].cut_time
            assert cut <= raw
            assert raw - cut <= SILENCE_LOOKBACK_SEC

    def test_adopted_span_is_the_silence_just_before_the_anchor(self, ctx):
        """採用した谷が、時刻表どおり直前の無音区間であること（おとりの谷を掴んでいない）。"""
        install_audio(ctx)
        cuts = s4.run(ctx, {"B": make_anchor("B", fixtures.EXPECTED_ANCHOR_B_RAW)})
        assert cuts["B"].silence_start == pytest.approx(43.60, abs=0.05)
        assert cuts["B"].silence_end == pytest.approx(43.95, abs=0.05)

    def test_cuts_json_is_written_and_reloadable(self, ctx):
        """実音声でも cuts.json が書かれ、load() で復元できる。"""
        install_audio(ctx)
        anchors = {
            "A": make_anchor("A", fixtures.EXPECTED_ANCHOR_A_RAW),
            "B": make_anchor("B", fixtures.EXPECTED_ANCHOR_B_RAW, phrase="ということで", score=96.2),
        }
        cuts = s4.run(ctx, anchors)

        assert s4.cuts_path(ctx).exists()
        loaded = s4.load(ctx)
        assert list(loaded) == ["A", "B"]
        for anchor_id in ("A", "B"):
            assert loaded[anchor_id].cut_time == pytest.approx(cuts[anchor_id].cut_time)
            assert loaded[anchor_id].silence_found == cuts[anchor_id].silence_found
            assert loaded[anchor_id].silence_end == pytest.approx(cuts[anchor_id].silence_end)

    def test_audio_without_silence_falls_back_and_warns(self, ctx):
        """谷の無い音声では silence_found=False・raw-0.08・warnings が揃う（SPEC Step 4）。"""
        install_audio(ctx, silences=())
        anchors = {
            "A": make_anchor("A", fixtures.EXPECTED_ANCHOR_A_RAW),
            "B": make_anchor("B", fixtures.EXPECTED_ANCHOR_B_RAW),
        }
        cuts = s4.run(ctx, anchors)

        for anchor_id, raw in (("A", fixtures.EXPECTED_ANCHOR_A_RAW), ("B", fixtures.EXPECTED_ANCHOR_B_RAW)):
            assert cuts[anchor_id].silence_found is False
            assert cuts[anchor_id].cut_time == pytest.approx(raw - NO_SILENCE_BACKOFF_SEC)
            assert cuts[anchor_id].silence_start is None
            assert cuts[anchor_id].silence_end is None

        assert len(ctx.warnings) == 2
        assert any("A" in w for w in ctx.warnings)
        assert any("B" in w for w in ctx.warnings)

        payload = json.loads(s4.cuts_path(ctx).read_text(encoding="utf-8"))
        assert payload["A"]["silence_found"] is False
        assert "silence_end" not in payload["A"]

    def test_silence_dur_override_changes_the_outcome(self, ctx):
        """`--silence-dur` の上書きが効く。谷（最長0.35秒）より長い d を渡すと何も拾えなくなる。"""
        install_audio(ctx)
        anchors = {"A": make_anchor("A", fixtures.EXPECTED_ANCHOR_A_RAW)}

        ctx.silence = SilenceConfig(min_duration_sec=0.12)
        assert s4.run(ctx, anchors)["A"].silence_found is True

        ctx.warnings.clear()
        ctx.silence = SilenceConfig(min_duration_sec=0.50)
        strict = s4.run(ctx, anchors)["A"]
        assert strict.silence_found is False
        assert strict.cut_time == pytest.approx(fixtures.EXPECTED_ANCHOR_A_RAW - NO_SILENCE_BACKOFF_SEC)
        assert ctx.warnings

    def test_silence_db_override_changes_the_outcome(self, ctx):
        """`--silence-db` の上書きが効く。約 -50dB の谷を持つ音声で判定が割れる。

        既定の -32dB なら谷として拾い、-60dB まで下げると「まだ音がある」と見なす。
        収録環境のノイズフロアに合わせて調整できることが SPEC Step 4 の要求。
        """
        ctx.ensure_dirs()
        raw = 5.95
        write_quiet_gap_wav(
            ctx.work_path(s1_extract_audio.AUDIO_FILENAME), gap=(5.60, raw), duration=8.0
        )
        anchors = {"A": make_anchor("A", raw)}

        ctx.silence = SilenceConfig(noise_db=-32.0)
        loose = s4.run(ctx, anchors)["A"]
        assert loose.silence_found is True
        assert loose.cut_time == pytest.approx(raw - SILENCE_BACKOFF_SEC, abs=0.05)

        ctx.warnings.clear()
        ctx.silence = SilenceConfig(noise_db=-60.0)
        strict = s4.run(ctx, anchors)["A"]
        assert strict.silence_found is False
        assert strict.cut_time == pytest.approx(raw - NO_SILENCE_BACKOFF_SEC)
        assert ctx.warnings

    def test_anchor_near_the_head_does_not_break(self, ctx):
        """探索窓が負にはみ出すアンカー（raw < 1.5秒）でも ffmpeg を落とさない。"""
        install_audio(ctx)
        cuts = s4.run(ctx, {"A": make_anchor("A", 0.30)})
        assert cuts["A"].cut_time >= 0.0
        assert cuts["A"].cut_time <= 0.30
