"""util/ffmpeg.py と util/cache.py — 外部プロセスとの境界。

ここが壊れると、カット点は正しく計算できているのに書き出した動画がずれる、
という「気づきにくい壊れ方」をする。だから

- silencedetect のログ解析（SPEC Step 4 の入口）は ffmpeg 無しで徹底的に、
- 実際に ffmpeg を叩く部分は「指定した区間が本当にその区間か」を尺と中身で、

の二段構えで固める。

`parse_silence_log` の契約（docstring より）:
- ログの相対時刻に `offset`（窓の開始絶対時刻）を足して絶対時刻で返す
- 1行に複数の情報が入る形（"silence_end: 3.5 | silence_duration: 0.4"）に耐える
- silence_end が先に来る崩れたログにも耐える（窓の先頭から始まったものとして残す）
- 閉じていない区間は `window_end` があればそこで閉じ、無ければ捨てる
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import (
    RenderConfig,
    SILENCE_BACKOFF_SEC,
    SILENCE_LOOKAHEAD_SEC,
    SILENCE_LOOKBACK_SEC,
)
from radio_cutter.errors import FfmpegError, RadioCutterError
from radio_cutter.util import ffmpeg as ff
from radio_cutter.util.cache import (
    DEFAULT_TRANSCRIPT_FILE,
    TranscriptCacheEntry,
    load_cache_entry,
    save_cache_entry,
    sha256_file,
    stable_hash,
    transcript_cache_key,
)

# ---------------------------------------------------------------------------
# 実際の ffmpeg 6.1 が吐いた silencedetect ログ（そのまま貼る）
# ---------------------------------------------------------------------------

REAL_LOG = """\
Input #0, wav, from 'audio.wav':
  Duration: 00:01:00.00, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
size=       0kB time=00:00:00.00 bitrate=N/A speed=N/A    \
[silencedetect @ 0x55f1ac4387c0] silence_start: 5.6
[silencedetect @ 0x55f1ac4387c0] silence_end: 5.95006 | silence_duration: 0.350062
[silencedetect @ 0x55f1ac4387c0] silence_start: 12.4
[silencedetect @ 0x55f1ac4387c0] silence_end: 12.7001 | silence_duration: 0.300063
[out#0/null @ 0x55f1ac41c740] video:0kB audio:1875kB subtitle:0kB other streams:0kB
size=N/A time=00:01:00.00 bitrate=N/A speed= 300x
"""


def approx_spans(spans, expected, tol=1e-6):
    """(開始, 終了) のリストを許容誤差つきで比べる。"""
    assert len(spans) == len(expected), f"区間数が違う: {spans} != {expected}"
    for (gs, ge), (es, ee) in zip(spans, expected):
        assert gs == pytest.approx(es, abs=tol)
        assert ge == pytest.approx(ee, abs=tol)


@pytest.fixture
def clean_encoder_cache():
    """`ffmpeg -encoders` のプロセスキャッシュを前後で捨てる（他テストへ漏らさない）。"""
    ff.list_encoders.cache_clear()
    yield
    ff.list_encoders.cache_clear()


# ===========================================================================
# parse_silence_log — ffmpeg 無しで徹底的に
# ===========================================================================


class TestParseSilenceLogReal:
    """実際の silencedetect 出力をそのまま読めること。"""

    def test_実ログから無音区間を取り出す(self):
        """"[silencedetect @ 0x...] silence_start: 5.6" 形式を読めること。"""
        spans = ff.parse_silence_log(REAL_LOG)
        approx_spans(spans, [(5.6, 5.95006), (12.4, 12.7001)])

    def test_1行に複数の情報が入る形でもsilence_durationを拾わない(self):
        """"silence_end: X | silence_duration: Y" の Y を時刻と誤読しないこと。"""
        log = "[silencedetect @ 0x1] silence_start: 1.0\n[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 0.5\n"
        approx_spans(ff.parse_silence_log(log), [(1.0, 1.5)])

    def test_silence_durationだけの行は区間にならない(self):
        """duration 行だけ拾って偽の区間を作らないこと。"""
        assert ff.parse_silence_log("silence_duration: 0.35\nsilence_duration: 1.2\n") == []

    def test_小数点の無い値も読む(self):
        """窓の先頭が無音だと ffmpeg は "silence_start: 0" と整数で出す。"""
        log = "silence_start: 0\nsilence_end: 0.250063 | silence_duration: 0.250063\n"
        approx_spans(ff.parse_silence_log(log), [(0.0, 0.250063)])

    def test_指数表記も読む(self):
        """長尺の録音で ffmpeg が指数表記を出しても落ちないこと。"""
        log = "silence_start: 1.5e+03\nsilence_end: 1.5006e+03\n"
        approx_spans(ff.parse_silence_log(log), [(1500.0, 1500.6)], tol=1e-3)

    def test_負の開始時刻も読む(self):
        """境界で ffmpeg が負値を出すことがある。取りこぼさない。"""
        log = "silence_start: -0.001\nsilence_end: 0.4\n"
        approx_spans(ff.parse_silence_log(log), [(-0.001, 0.4)])


class TestParseSilenceLogOffset:
    """atrim で切り出した窓の相対時刻を絶対時刻に戻すこと（SPEC Step 4）。"""

    def test_offsetが全ての時刻に足される(self):
        """窓 [42.45, 44.45] のログ 1.15/1.50006 は絶対 43.60/43.95006 になる。"""
        log = "silence_start: 1.15\nsilence_end: 1.50006 | silence_duration: 0.350062\n"
        approx_spans(ff.parse_silence_log(log, offset=42.45, window_end=44.45), [(43.60, 43.95006)])

    def test_offset無指定なら相対時刻がそのまま絶対時刻(self):
        """全尺に対して掛けたときは offset=0 で素通しになること。"""
        approx_spans(ff.parse_silence_log("silence_start: 3.0\nsilence_end: 3.4\n"), [(3.0, 3.4)])

    def test_offsetは負の窓でも素直に足す(self):
        log = "silence_start: 1.0\nsilence_end: 2.0\n"
        approx_spans(ff.parse_silence_log(log, offset=-0.5), [(0.5, 1.5)])


class TestParseSilenceLogUnclosed:
    """silence_end が来ないまま終わったログの扱い。"""

    def test_閉じていない区間はwindow_endで閉じる(self):
        """窓の末尾まで無音が続いていた、と解釈するのが正しい（谷を取りこぼさない）。"""
        log = "silence_start: 1.0\nsilence_end: 1.4\nsilence_start: 1.8\n"
        approx_spans(
            ff.parse_silence_log(log, offset=10.0, window_end=12.0),
            [(11.0, 11.4), (11.8, 12.0)],
        )

    def test_window_endが無ければ閉じていない区間は捨てる(self):
        """終端が分からない区間を勝手にでっち上げないこと。"""
        log = "silence_start: 1.0\nsilence_end: 1.4\nsilence_start: 1.8\n"
        approx_spans(ff.parse_silence_log(log, offset=10.0), [(11.0, 11.4)])

    def test_window_endが開始より前なら捨てる(self):
        """窓の外で始まった区間を負の長さで残さないこと。"""
        log = "silence_start: 5.0\n"
        assert ff.parse_silence_log(log, window_end=3.0) == []

    def test_window_endと開始が同時なら長さ0で残る(self):
        log = "silence_start: 3.0\n"
        approx_spans(ff.parse_silence_log(log, window_end=3.0), [(3.0, 3.0)])


class TestParseSilenceLogBroken:
    """順序が崩れたログでも例外を投げず、読めるものは読むこと。"""

    def test_silence_endが先に来たら窓の先頭から始まったものとして残す(self):
        """窓の頭が既に無音だった場合。捨てるとカット点が後ろにずれるので残す。"""
        log = "silence_end: 0.25 | silence_duration: 0.25\nsilence_start: 1.0\nsilence_end: 1.4\n"
        approx_spans(
            ff.parse_silence_log(log, offset=43.7, window_end=45.0),
            [(43.7, 43.95), (44.7, 45.1)],
        )

    def test_silence_endが先に来てoffset0でも残る(self):
        log = "silence_end: 0.25\n"
        approx_spans(ff.parse_silence_log(log), [(0.0, 0.25)])

    def test_silence_startが連続したら先に来たほうを活かす(self):
        """谷の入口は最初の silence_start。あとから来た開始で上書きしない。"""
        log = "silence_start: 1.0\nsilence_start: 2.0\nsilence_end: 3.0\n"
        approx_spans(ff.parse_silence_log(log), [(1.0, 3.0)])

    def test_終了が開始より前の区間は捨てる(self):
        """負の長さの無音区間を下流（Step 4）に渡さないこと。"""
        log = "silence_start: 5.0\nsilence_end: 4.0\n"
        assert ff.parse_silence_log(log) == []

    def test_壊れた区間を捨てても後続は読む(self):
        log = "silence_start: 5.0\nsilence_end: 4.0\nsilence_start: 6.0\nsilence_end: 6.5\n"
        approx_spans(ff.parse_silence_log(log), [(6.0, 6.5)])

    def test_返り値は開始時刻の昇順(self):
        """下流は「raw_cut_time より前にある最後の無音」を取るので順序が要る。"""
        log = (
            "silence_end: 0.5\n"          # 窓頭の無音 → (0.0, 0.5)
            "silence_start: 3.0\nsilence_end: 3.4\n"
            "silence_start: 1.0\nsilence_end: 1.4\n"
        )
        spans = ff.parse_silence_log(log)
        assert [s for s, _ in spans] == sorted(s for s, _ in spans)
        approx_spans(spans, [(0.0, 0.5), (1.0, 1.4), (3.0, 3.4)])


class TestParseSilenceLogNoise:
    """無音イベントが1つも無いログ。"""

    def test_空文字列は空リスト(self):
        assert ff.parse_silence_log("") == []

    def test_無音が無いログは空リスト(self):
        """SPEC Step 4 の「無音が検出されなければフォールバック」の入口。"""
        log = "Input #0, wav, from 'audio.wav':\nsize=N/A time=00:00:02.00 bitrate=N/A speed=89.4x\n"
        assert ff.parse_silence_log(log) == []

    def test_無関係な行が混ざっても無音イベントだけ拾う(self):
        log = (
            "ffmpeg version 6.1.1 Copyright (c) 2000-2023\n"
            "[silencedetect @ 0x1] silence_start: 2.0\n"
            "frame= 100 fps=0.0 q=-1.0 size=1kB time=00:00:03.00 bitrate=2.7kbits/s\n"
            "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 0.5\n"
            "[out#0/null @ 0x2] video:0kB audio:62kB\n"
        )
        approx_spans(ff.parse_silence_log(log), [(2.0, 2.5)])

    def test_似た名前の語に反応しない(self):
        """"silencedetect" や "silence_duration" を時刻イベントと誤認しないこと。"""
        log = "[silencedetect @ 0x1] silencedetect: 1.0\nsilence_threshold: 2.0\n"
        assert ff.parse_silence_log(log) == []

    def test_返り値はfloatのタプル(self):
        spans = ff.parse_silence_log("silence_start: 1\nsilence_end: 2\n")
        assert spans and all(isinstance(s, tuple) and len(s) == 2 for s in spans)
        assert all(isinstance(v, float) for s in spans for v in s)


# ===========================================================================
# MediaInfo — work/probe.json の中身（SPEC Step 1）
# ===========================================================================


class TestMediaInfoDict:
    def test_to_dictは秒を小数3桁に丸める(self):
        """SPEC 11章「秒数は全て float（小数点以下3桁）」。"""
        info = ff.MediaInfo(path="/x.mp4", duration=3612.4444449, fps=29.97002997)
        d = info.to_dict()
        assert d["duration"] == 3612.444
        assert d["fps"] == pytest.approx(29.97003, abs=1e-6)

    def test_to_dictとfrom_dictで往復できる(self):
        """probe.json に書いて読み戻しても同じ素性であること（--from-step の再開に効く）。"""
        info = ff.MediaInfo(
            path="/x.mp4", duration=60.0, fps=15.0, width=160, height=120,
            video_codec="h264", audio_codec="aac", has_video=True, has_audio=True,
            raw={"format": {"duration": "60.000000"}},
        )
        back = ff.MediaInfo.from_dict(json.loads(json.dumps(info.to_dict())))
        assert back == info

    def test_映像が無い場合はNoneのまま往復する(self):
        info = ff.MediaInfo(path="/a.wav", duration=1.5, has_audio=True)
        back = ff.MediaInfo.from_dict(info.to_dict())
        assert back.fps is None and back.width is None and back.height is None
        assert back.video_codec is None and back.has_video is False
        assert back.has_audio is True

    def test_from_dictは欠けた項目を既定値で埋める(self):
        info = ff.MediaInfo.from_dict({"path": "/a.mp4", "duration": "12.5"})
        assert info.duration == 12.5 and info.raw == {} and info.has_audio is False


class TestProbeMediaParsing:
    """ffprobe の JSON をどう解釈するか（ffprobe 自体は差し替えて確かめる）。"""

    def _patch(self, monkeypatch, data):
        monkeypatch.setattr(ff, "ffprobe_json", lambda path: data)

    def test_総尺はformat_durationを優先する(self, monkeypatch):
        self._patch(monkeypatch, {"format": {"duration": "3612.4"}, "streams": []})
        assert ff.probe_media("x.mp4").duration == pytest.approx(3612.4)

    def test_format_durationが無ければ映像ストリームの尺を使う(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"format": {}, "streams": [{"codec_type": "video", "duration": "42.0", "r_frame_rate": "30/1"}]},
        )
        assert ff.probe_media("x.mp4").duration == pytest.approx(42.0)

    def test_durationがNAなら映像ストリームの尺を使う(self, monkeypatch):
        """ffprobe は尺を取れないと文字列 "N/A" を返す。0.0 と誤読しないこと。"""
        self._patch(
            monkeypatch,
            {"format": {"duration": "N/A"}, "streams": [{"codec_type": "video", "duration": "3612.4"}]},
        )
        assert ff.probe_media("x.mp4").duration == pytest.approx(3612.4)

    def test_どこもNAならFfmpegErrorで止まる(self, monkeypatch):
        """尺 0.0 のまま進むと、区間長も final.mp4 の検算も全部おかしくなる。"""
        self._patch(
            monkeypatch,
            {"format": {"duration": "N/A"}, "streams": [{"codec_type": "video", "duration": "N/A"}]},
        )
        with pytest.raises(FfmpegError):
            ff.probe_media("x.mp4")

    def test_尺がどこにも無ければFfmpegErrorで止まる(self, monkeypatch):
        """尺不明のまま先へ進むと最後の検算まで壊れが伝播する。ここで止める。"""
        self._patch(monkeypatch, {"format": {}, "streams": [{"codec_type": "audio"}]})
        with pytest.raises(FfmpegError) as exc:
            ff.probe_media("broken.mp4")
        assert "総尺" in str(exc.value)

    def test_カバーアートは映像トラックとして扱わない(self, monkeypatch):
        """attached_pic を映像とみなすと 1x1 の静止画で fps/解像度が汚れる。"""
        self._patch(
            monkeypatch,
            {
                "format": {"duration": "60.0"},
                "streams": [
                    {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
                     "disposition": {"attached_pic": 1}},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
            },
        )
        info = ff.probe_media("cover.m4a")
        assert info.has_video is False
        assert info.video_codec is None and info.width is None
        assert info.has_audio is True and info.audio_codec == "aac"

    def test_fpsは分数表記を割り算する(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"format": {"duration": "1.0"}, "streams": [{"codec_type": "video", "r_frame_rate": "30000/1001"}]},
        )
        assert ff.probe_media("x.mp4").fps == pytest.approx(29.97003, abs=1e-5)

    def test_r_frame_rateが0除算ならavg_frame_rateに落ちる(self, monkeypatch):
        self._patch(
            monkeypatch,
            {
                "format": {"duration": "1.0"},
                "streams": [{"codec_type": "video", "r_frame_rate": "0/0", "avg_frame_rate": "25/1"}],
            },
        )
        assert ff.probe_media("x.mp4").fps == pytest.approx(25.0)

    def test_fpsが取れなければNone(self, monkeypatch):
        self._patch(
            monkeypatch,
            {
                "format": {"duration": "1.0"},
                "streams": [{"codec_type": "video", "r_frame_rate": "0/0", "avg_frame_rate": "0/0"}],
            },
        )
        assert ff.probe_media("x.mp4").fps is None

    def test_解像度が壊れていてもNoneにするだけで止めない(self, monkeypatch):
        self._patch(
            monkeypatch,
            {"format": {"duration": "1.0"}, "streams": [{"codec_type": "video", "width": "abc", "height": 120}]},
        )
        info = ff.probe_media("x.mp4")
        assert info.width is None and info.height is None and info.has_video is True

    def test_最初の映像と最初の音声だけを採る(self, monkeypatch):
        self._patch(
            monkeypatch,
            {
                "format": {"duration": "1.0"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "video", "codec_name": "hevc"},
                    {"codec_type": "audio", "codec_name": "aac"},
                    {"codec_type": "audio", "codec_name": "mp3"},
                    {"codec_type": "subtitle", "codec_name": "mov_text"},
                ],
            },
        )
        info = ff.probe_media("x.mp4")
        assert info.video_codec == "h264" and info.audio_codec == "aac"

    def test_rawにffprobeの生JSONを残す(self, monkeypatch):
        """SPEC 11章「中間ファイルは消さない。デバッグの起点になる」。"""
        data = {"format": {"duration": "1.0"}, "streams": []}
        self._patch(monkeypatch, data)
        assert ff.probe_media("x.mp4").raw == data


# ===========================================================================
# コーデック選択（SPEC Step 7 / 2章のフォールバック）
# ===========================================================================


class TestChooseVideoCodec:
    def test_使えるならそのまま使いビットレートを添える(self, monkeypatch):
        monkeypatch.setattr(ff, "has_encoder", lambda name: name == "h264_videotoolbox")
        render = RenderConfig()
        codec, extra, fell_back = ff.choose_video_codec(render)
        assert codec == "h264_videotoolbox"
        assert extra == ("-b:v", "12M")
        assert fell_back is False

    def test_使えなければfallback_video_codecに落ちる(self, monkeypatch):
        """SPEC 2章「無ければCPUエンコードにフォールバックする旨を警告表示」。"""
        monkeypatch.setattr(ff, "has_encoder", lambda name: name == "libx264")
        render = RenderConfig()
        codec, extra, fell_back = ff.choose_video_codec(render)
        assert codec == "libx264"
        assert extra == ("-preset", "veryfast", "-crf", "20")
        assert fell_back is True

    def test_フォールバック時はvideo_bitrateではなくfallback_extra_argsを使う(self, monkeypatch):
        """libx264 は -crf 指定。-b:v を混ぜると CRF が効かなくなる。"""
        monkeypatch.setattr(ff, "has_encoder", lambda name: False)
        render = RenderConfig(video_codec="nope", video_bitrate="99M")
        _codec, extra, _ = ff.choose_video_codec(render)
        assert "-b:v" not in extra

    def test_フォールバック先も無い場合でもフォールバック名を返す(self, monkeypatch):
        """ここで例外にすると原因が分からないまま止まる。ffmpeg に投げて stderr を見せる。"""
        monkeypatch.setattr(ff, "has_encoder", lambda name: False)
        codec, _extra, fell_back = ff.choose_video_codec(RenderConfig())
        assert codec == "libx264" and fell_back is True

    def test_fallback_extra_argsは設定を尊重する(self, monkeypatch):
        monkeypatch.setattr(ff, "has_encoder", lambda name: False)
        render = RenderConfig(fallback_video_codec="mpeg4", fallback_extra_args=("-q:v", "3"))
        codec, extra, fell_back = ff.choose_video_codec(render)
        assert (codec, extra, fell_back) == ("mpeg4", ("-q:v", "3"), True)

    def test_空のコーデック名は使えない扱い(self, monkeypatch, clean_encoder_cache):
        monkeypatch.setattr(ff, "_encoder_names", lambda: frozenset({"libx264"}))
        assert ff.has_encoder("") is False


# ===========================================================================
# バイナリの解決とエラーの出し方（SPEC 9章「握りつぶさない」）
# ===========================================================================


class TestBinaries:
    def test_環境変数でffmpegのパスを差し替えられる(self, monkeypatch):
        monkeypatch.setenv(ff.FFMPEG_ENV, "/opt/custom/ffmpeg")
        monkeypatch.setenv(ff.FFPROBE_ENV, "/opt/custom/ffprobe")
        assert ff.ffmpeg_bin() == "/opt/custom/ffmpeg"
        assert ff.ffprobe_bin() == "/opt/custom/ffprobe"

    def test_未設定なら素のffmpeg(self, monkeypatch):
        monkeypatch.delenv(ff.FFMPEG_ENV, raising=False)
        monkeypatch.delenv(ff.FFPROBE_ENV, raising=False)
        assert ff.ffmpeg_bin() == "ffmpeg" and ff.ffprobe_bin() == "ffprobe"

    def test_バイナリが無ければ入れ方を添えて止める(self, monkeypatch):
        """doctor が「無い」と言えるだけでなく、次の一手を出すこと（SPEC 9章）。"""
        monkeypatch.setenv(ff.FFMPEG_ENV, "radio-cutter-no-such-ffmpeg")
        monkeypatch.setenv(ff.FFPROBE_ENV, "radio-cutter-no-such-ffprobe")
        with pytest.raises(FfmpegError) as exc:
            ff.require_binaries()
        msg = str(exc.value)
        assert "radio-cutter-no-such-ffmpeg" in msg
        assert "radio-cutter-no-such-ffprobe" in msg
        assert ff.FFMPEG_ENV in msg

    def test_起動できないコマンドはFfmpegErrorになる(self, monkeypatch):
        monkeypatch.setenv(ff.FFMPEG_ENV, "radio-cutter-no-such-ffmpeg")
        with pytest.raises(FfmpegError) as exc:
            ff.run_ffmpeg(["-version"])
        assert "見つかりません" in str(exc.value)

    def test_ffmpegが無ければversionはNone(self, monkeypatch):
        """doctor は例外ではなく None を見て「無い」と表示する。"""
        monkeypatch.setenv(ff.FFMPEG_ENV, "radio-cutter-no-such-ffmpeg")
        assert ff.ffmpeg_version() is None

    def test_ffmpegが無ければエンコーダ一覧はFfmpegError(self, monkeypatch, clean_encoder_cache):
        """空集合を返すと「videotoolbox が無い」と誤診してフォールバックを黙って選んでしまう。"""
        monkeypatch.setenv(ff.FFMPEG_ENV, "radio-cutter-no-such-ffmpeg")
        with pytest.raises(FfmpegError):
            ff.list_encoders()

    def test_FfmpegErrorはstderrとコマンドと終了コードを載せる(self):
        """SPEC 9章「stderrをそのまま表示して停止。握りつぶさない」。"""
        exc = FfmpegError("失敗", cmd=["ffmpeg", "-i", "x.mp4"], stderr="No such file or directory", returncode=1)
        text = str(exc)
        assert "失敗" in text
        assert "ffmpeg -i x.mp4" in text
        assert "終了コード: 1" in text
        assert "No such file or directory" in text


# ===========================================================================
# 引数チェック（ffmpeg を起動する前に落ちるべきもの）
# ===========================================================================


class TestGuardsBeforeSpawning:
    def test_窓が空ならffmpegを起動せず空リスト(self, monkeypatch, tmp_path):
        def boom(*a, **k):  # pragma: no cover - 呼ばれたら失敗
            raise AssertionError("ffmpeg を起動してはいけない")

        monkeypatch.setattr(ff, "run_ffmpeg", boom)
        assert ff.detect_silences(tmp_path / "nope.wav", start=5.0, end=5.0, noise_db=-32, min_duration=0.12) == []
        assert ff.detect_silences(tmp_path / "nope.wav", start=5.0, end=1.0, noise_db=-32, min_duration=0.12) == []

    def test_長さ0以下の区間は書き出さない(self, tmp_path, monkeypatch):
        """0秒の mp4 を作って後段の連結を壊さないこと。"""
        def boom(*a, **k):  # pragma: no cover
            raise AssertionError("ffmpeg を起動してはいけない")

        monkeypatch.setattr(ff, "run_ffmpeg", boom)
        with pytest.raises(FfmpegError) as exc:
            ff.encode_segment("in.mp4", 10.0, 10.0, tmp_path / "o.mp4", RenderConfig())
        assert "0以下" in str(exc.value)
        with pytest.raises(FfmpegError):
            ff.encode_segment("in.mp4", 10.0, 9.0, tmp_path / "o.mp4", RenderConfig())

    def test_連結対象が空ならFfmpegError(self, tmp_path):
        with pytest.raises(FfmpegError) as exc:
            ff.concat_files([], tmp_path / "final.mp4", tmp_path)
        assert "1本もありません" in str(exc.value)

    def test_連結対象が存在しなければ名前を挙げて止める(self, tmp_path):
        real = tmp_path / "a.mp4"
        real.write_bytes(b"x")
        with pytest.raises(FfmpegError) as exc:
            ff.concat_files([real, tmp_path / "missing.mp4"], tmp_path / "final.mp4", tmp_path)
        assert "missing.mp4" in str(exc.value)


class TestConcatEscape:
    """concat demuxer のリストは `file '...'` 形式。クォートの扱いを間違えると連結が壊れる。"""

    def test_シングルクォートを閉じて埋め直す(self, tmp_path):
        p = tmp_path / "it's a file.mp4"
        p.write_bytes(b"x")
        escaped = ff._concat_escape(p)
        assert "'\\''" in escaped
        assert escaped.count("'") == 3  # ' \' ' の3つ
        assert "a file.mp4" in escaped

    def test_クォートが無ければ絶対パスのまま(self, tmp_path):
        p = tmp_path / "plain.mp4"
        p.write_bytes(b"x")
        assert ff._concat_escape(p) == str(p.resolve())

    def test_相対パスは絶対パスになる(self, tmp_path, monkeypatch):
        """concat.txt は work/ に置くので、相対のままだと基準がずれる。"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.mp4").write_bytes(b"x")
        assert Path(ff._concat_escape(Path("a.mp4"))).is_absolute()


class _RecordingFfmpeg:
    """run_ffmpeg を差し替えて引数だけ記録する。出力ファイルは空で作っておく。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        argv = [str(a) for a in args]
        self.calls.append(argv)
        out = Path(argv[-1])
        if out.suffix:  # "-" のような出力先は作らない
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")
        return subprocess.CompletedProcess(argv, 0, "", "")

    @property
    def last(self) -> list[str]:
        return self.calls[-1]


class TestCommandShape:
    """SPEC が明示している ffmpeg コマンドの形を守ること（実バイナリは要らない）。"""

    @pytest.fixture
    def rec(self, monkeypatch) -> _RecordingFfmpeg:
        r = _RecordingFfmpeg()
        monkeypatch.setattr(ff, "run_ffmpeg", r)
        return r

    def test_音声抽出はSPEC_Step1のコマンドそのもの(self, rec, tmp_path: Path):
        """`-vn -ac 1 -ar 16000 -c:a pcm_s16le`。ASR も silencedetect もこの前提で動く。"""
        src = tmp_path / "ep.mp4"
        dst = tmp_path / "work" / "audio.wav"
        ff.extract_audio(src, dst)
        assert rec.last == [
            "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(dst),
        ]

    def test_無音検出のフィルタはSPEC_Step4のもの(self, rec, tmp_path: Path):
        """atrim で窓を切り、asetpts で0起点に戻してから silencedetect にかける。"""
        ff.detect_silences(tmp_path / "audio.wav", start=4.45, end=6.45, noise_db=-32.0, min_duration=0.12)
        argv = rec.last
        af = argv[argv.index("-af") + 1]
        assert af == "atrim=start=4.450:end=6.450,asetpts=PTS-STARTPTS,silencedetect=n=-32dB:d=0.12"
        assert argv[-3:] == ["-f", "null", "-"]

    def test_無音検出のしきい値はCLIオプションを反映する(self, rec, tmp_path: Path):
        """SPEC Step 4「-32dB と d=0.12 はCLIオプションで上書き可能」。"""
        ff.detect_silences(tmp_path / "a.wav", start=0.0, end=1.0, noise_db=-27.5, min_duration=0.2)
        af = rec.last[rec.last.index("-af") + 1]
        assert "silencedetect=n=-27.5dB:d=0.2" in af

    def test_無音検出の窓の開始は0未満に落とさない(self, rec, tmp_path: Path):
        ff.detect_silences(tmp_path / "a.wav", start=-3.0, end=1.0, noise_db=-32.0, min_duration=0.12)
        af = rec.last[rec.last.index("-af") + 1]
        assert af.startswith("atrim=start=0.000:end=1.000")

    def test_書き出しはcopyせず必ず再エンコードする(self, rec, tmp_path: Path):
        """SPEC Step 7「`-c copy` は最寄りのキーフレームまでずれるため使えない」。"""
        render = RenderConfig(video_codec="libx264", fallback_video_codec="libx264")
        ff.encode_segment(tmp_path / "ep.mp4", 5.9, 43.9, tmp_path / "02_main.mp4", render, use_fallback=False)
        argv = rec.last
        assert "copy" not in argv
        assert argv[argv.index("-c:v") + 1] == "libx264"
        assert argv[argv.index("-c:a") + 1] == render.audio_codec
        assert argv[argv.index("-b:a") + 1] == render.audio_bitrate
        assert argv[argv.index("-movflags") + 1] == "+faststart"

    def test_ssは入力の前に置き長さはtで渡す(self, rec, tmp_path: Path):
        """`-ss` を `-i` の前に置くのが SPEC Step 7。長さは区間長で渡す。"""
        render = RenderConfig(video_codec="libx264")
        ff.encode_segment(tmp_path / "ep.mp4", 5.9, 43.9, tmp_path / "o.mp4", render, use_fallback=False)
        argv = rec.last
        assert argv.index("-ss") < argv.index("-i")
        assert argv[argv.index("-ss") + 1] == "5.900"
        assert argv[argv.index("-t") + 1] == "38.000"
        assert "-to" not in argv, "-ss を -i の前に置いた場合 -to は区間長と解釈されるので使わない"

    def test_秒は小数3桁で渡す(self, rec, tmp_path: Path):
        """SPEC 11章「秒数は全て float（小数点以下3桁）」。"""
        ff.encode_segment(
            tmp_path / "ep.mp4", 12.3456789, 15.6789012, tmp_path / "o.mp4",
            RenderConfig(), use_fallback=True,
        )
        argv = rec.last
        assert argv[argv.index("-ss") + 1] == "12.346"
        assert argv[argv.index("-t") + 1] == "3.333"

    def test_フォールバック時はcrf引数が付く(self, rec, tmp_path: Path):
        ff.encode_segment(tmp_path / "ep.mp4", 0.0, 1.0, tmp_path / "o.mp4", RenderConfig(), use_fallback=True)
        argv = rec.last
        assert argv[argv.index("-c:v") + 1] == "libx264"
        assert "-crf" in argv and "-b:v" not in argv

    def test_連結はconcat_demuxerでcopyする(self, rec, tmp_path: Path):
        """3本とも同一パラメータで書き出した直後なので、ここだけは copy でよい。"""
        parts = []
        for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4"):
            p = tmp_path / name
            p.write_bytes(b"x")
            parts.append(p)
        out = tmp_path / "out" / "final.mp4"
        work = tmp_path / "work"
        ff.concat_files(parts, out, work)
        assert rec.last == [
            "-y", "-f", "concat", "-safe", "0", "-i", str(work / "concat.txt"),
            "-c", "copy", "-movflags", "+faststart", str(out),
        ]
        listing = (work / "concat.txt").read_text(encoding="utf-8")
        assert listing.splitlines() == [f"file '{p.resolve()}'" for p in parts]

    def test_連結リストは毎回書き直す(self, rec, tmp_path: Path):
        """前回の残骸に追記して4本連結、が起きないこと。"""
        work = tmp_path / "work"
        work.mkdir()
        (work / "concat.txt").write_text("file '/stale/old.mp4'\n", encoding="utf-8")
        p = tmp_path / "a.mp4"
        p.write_bytes(b"x")
        ff.concat_files([p], tmp_path / "final.mp4", work)
        listing = (work / "concat.txt").read_text(encoding="utf-8")
        assert "stale" not in listing
        assert len(listing.splitlines()) == 1


class TestStderrTail:
    """FfmpegError に載せる stderr の切り詰め方（SPEC 9章）。"""

    def test_短いstderrはそのまま(self):
        assert ff._tail("No such file or directory") == "No such file or directory"

    def test_長いstderrは末尾を残す(self):
        """ffmpeg は本当のエラーを最後に出す。頭を残したら原因が見えない。"""
        text = ("x" * 10000) + "\nError opening output file"
        tail = ff._tail(text, limit=200)
        assert tail.endswith("Error opening output file")
        assert "省略" in tail
        assert len(tail) < len(text)

    def test_境界ちょうどなら削らない(self):
        text = "y" * ff.STDERR_TAIL_CHARS
        assert ff._tail(text) == text


class TestRunProcess:
    def test_タイムアウトはFfmpegErrorになる(self):
        """固まった ffmpeg をいつまでも待たないこと。"""
        with pytest.raises(FfmpegError) as exc:
            ff._run([sys.executable, "-c", "import time; time.sleep(30)"], check=True, timeout=0.5)
        assert "タイムアウト" in str(exc.value)

    def test_check_Falseなら非ゼロでも例外にしない(self):
        """doctor はエンコーダ一覧の失敗を自分で判定する。"""
        proc = ff._run([sys.executable, "-c", "import sys; sys.exit(3)"], check=False, timeout=30)
        assert proc.returncode == 3

    def test_ffprobeの出力がJSONでなければFfmpegError(self, monkeypatch):
        monkeypatch.setattr(
            ff, "run_ffprobe",
            lambda args, **kw: subprocess.CompletedProcess(list(args), 0, "not json at all", ""),
        )
        with pytest.raises(FfmpegError) as exc:
            ff.ffprobe_json("x.mp4")
        assert "JSON" in str(exc.value)

    def test_ffprobeがJSON配列を返してもFfmpegError(self, monkeypatch):
        monkeypatch.setattr(
            ff, "run_ffprobe",
            lambda args, **kw: subprocess.CompletedProcess(list(args), 0, "[1, 2]", ""),
        )
        with pytest.raises(FfmpegError) as exc:
            ff.ffprobe_json("x.mp4")
        assert "オブジェクト" in str(exc.value)

    def test_ffprobeが空出力なら空dict(self, monkeypatch):
        monkeypatch.setattr(
            ff, "run_ffprobe",
            lambda args, **kw: subprocess.CompletedProcess(list(args), 0, "", ""),
        )
        assert ff.ffprobe_json("x.mp4") == {}


# ===========================================================================
# ここから ffmpeg 実バイナリを使う
# ===========================================================================


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestProbeMediaReal:
    def test_合成エピソードの素性を取る(self, episode_video: Path):
        """SPEC Step 1「ffprobe で総尺・fps・解像度も取得」。"""
        info = ff.probe_media(episode_video)
        assert info.duration == pytest.approx(fixtures.EPISODE_DURATION, abs=0.2)
        assert info.fps == pytest.approx(15.0, abs=0.01)
        assert (info.width, info.height) == (160, 120)
        assert info.has_video is True and info.has_audio is True
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.path == str(episode_video)

    def test_probe_jsonにそのまま書ける(self, episode_video: Path, tmp_path: Path):
        """work/probe.json は JSON 化できて読み戻せること。"""
        info = ff.probe_media(episode_video)
        p = tmp_path / "probe.json"
        p.write_text(json.dumps(info.to_dict(), ensure_ascii=False), encoding="utf-8")
        back = ff.MediaInfo.from_dict(json.loads(p.read_text(encoding="utf-8")))
        assert back.duration == pytest.approx(info.duration, abs=1e-3)
        assert back.width == info.width and back.audio_codec == info.audio_codec

    def test_media_durationはprobeと同じ尺を返す(self, episode_video: Path):
        assert ff.media_duration(episode_video) == pytest.approx(ff.probe_media(episode_video).duration)

    def test_音声トラックが無い動画はhas_audioがFalse(self, tmp_path: Path):
        """probe 自体は成功する。落ちるのは音声抽出（Step 1）のほう。"""
        mute = make_mute_video(tmp_path / "mute.mp4")
        info = ff.probe_media(mute)
        assert info.has_video is True
        assert info.has_audio is False and info.audio_codec is None

    def test_存在しない入力はstderr付きのFfmpegError(self, tmp_path: Path):
        """SPEC 9章「ffmpegが非ゼロ終了 → stderrをそのまま表示して停止」。"""
        with pytest.raises(FfmpegError) as exc:
            ff.probe_media(tmp_path / "no-such-episode.mp4")
        err = exc.value
        assert err.returncode not in (0, None)
        assert err.stderr, "stderr が載っていない"
        assert "no-such-episode.mp4" in err.stderr
        assert "--- ffmpeg stderr ---" in str(err)

    def test_中身が動画でないファイルもFfmpegError(self, tmp_path: Path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"this is not a video" * 100)
        with pytest.raises(FfmpegError):
            ff.probe_media(junk)


def make_mute_video(path: Path, *, duration: float = 3.0) -> Path:
    """音声トラックを持たない小さな mp4 を作る（テスト用の道具）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ff.run_ffmpeg(
        [
            "-y",
            "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=15:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path),
        ]
    )
    return path


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestExtractAudio:
    def test_16kHzモノラルのPCMになる(self, episode_video: Path, tmp_path: Path):
        """SPEC Step 1 の `-vn -ac 1 -ar 16000 -c:a pcm_s16le`。ASR と silencedetect の前提。"""
        out = ff.extract_audio(episode_video, tmp_path / "work" / "ep" / "audio.wav")
        assert out.exists()
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() / 16000 == pytest.approx(fixtures.EPISODE_DURATION, abs=0.2)
        info = ff.probe_media(out)
        assert info.audio_codec == "pcm_s16le"
        assert info.has_video is False

    def test_出力先の親ディレクトリを作る(self, episode_video: Path, tmp_path: Path):
        out = ff.extract_audio(episode_video, tmp_path / "a" / "b" / "c" / "audio.wav")
        assert out.exists() and out.parent.is_dir()

    def test_音声トラックが無い動画はFfmpegErrorで止まる(self, tmp_path: Path):
        """握りつぶして空の wav を残すと、Step 2 以降が謎の失敗をする。"""
        mute = make_mute_video(tmp_path / "mute.mp4")
        with pytest.raises(FfmpegError) as exc:
            ff.extract_audio(mute, tmp_path / "audio.wav")
        assert exc.value.stderr, "stderr が載っていない"
        assert not (tmp_path / "audio.wav").exists()

    def test_存在しない入力はFfmpegError(self, tmp_path: Path):
        with pytest.raises(FfmpegError) as exc:
            ff.extract_audio(tmp_path / "nope.mp4", tmp_path / "audio.wav")
        assert exc.value.stderr


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestDetectSilencesReal:
    """SPEC Step 4。合成エピソードの無音区間表と一致すること。"""

    def test_全尺で合成エピソードの無音区間と一致する(self, episode_wav: Path):
        spans = ff.detect_silences(
            episode_wav, start=0.0, end=fixtures.EPISODE_DURATION, noise_db=-32.0, min_duration=0.12
        )
        assert len(spans) == len(fixtures.SILENCES)
        for (gs, ge), (es, ee) in zip(spans, fixtures.SILENCES):
            assert gs == pytest.approx(es, abs=0.05), f"開始がずれた {gs} != {es}"
            assert ge == pytest.approx(ee, abs=0.05), f"終了がずれた {ge} != {ee}"

    @pytest.mark.parametrize("expected", fixtures.SILENCES)
    def test_窓を変えても同じ絶対時刻を返す(self, episode_wav: Path, expected):
        """窓の相対時刻を絶対時刻に戻せていること（offset の加算）。"""
        es, ee = expected
        spans = ff.detect_silences(
            episode_wav, start=es - 0.4, end=ee + 0.4, noise_db=-32.0, min_duration=0.12
        )
        assert len(spans) == 1, f"窓 [{es - 0.4}, {ee + 0.4}] で {spans}"
        gs, ge = spans[0]
        assert gs == pytest.approx(es, abs=0.05)
        assert ge == pytest.approx(ee, abs=0.05)

    @pytest.mark.parametrize(
        "raw,expected_cut",
        [
            (fixtures.EXPECTED_ANCHOR_A_RAW, fixtures.EXPECTED_CUT_A),
            (fixtures.EXPECTED_ANCHOR_B_RAW, fixtures.EXPECTED_CUT_B),
        ],
    )
    def test_Step4の探索窓でアンカー直前の谷が見つかる(self, episode_wav: Path, raw, expected_cut):
        """`raw_cut_time` の [-1.5, +0.5] を見て、谷の終わり -50ms がカット点になること。"""
        spans = ff.detect_silences(
            episode_wav,
            start=raw - SILENCE_LOOKBACK_SEC,
            end=raw + SILENCE_LOOKAHEAD_SEC,
            noise_db=-32.0,
            min_duration=0.12,
        )
        before = [s for s in spans if s[0] < raw]
        assert before, "raw_cut_time より前の無音が見つからない"
        valley_start, valley_end = before[-1]
        cut = valley_end - SILENCE_BACKOFF_SEC
        assert cut == pytest.approx(expected_cut, abs=0.05)
        # カット点は谷の内側に落ちる＝語頭も直前の語尾も削らない（SPEC Phase 1 の受け入れ基準）
        assert valley_start <= cut < raw

    def test_返り値は絶対時刻の昇順で重ならない(self, episode_wav: Path):
        spans = ff.detect_silences(
            episode_wav, start=0.0, end=fixtures.EPISODE_DURATION, noise_db=-32.0, min_duration=0.12
        )
        for (s, e) in spans:
            assert e > s
        for prev, nxt in zip(spans, spans[1:]):
            assert prev[1] <= nxt[0]

    def test_min_durationを長くすると何も見つからない(self, episode_wav: Path):
        """CLI の --silence-dur が効くこと。合成エピソードの谷は最長0.35秒。"""
        spans = ff.detect_silences(
            episode_wav, start=0.0, end=fixtures.EPISODE_DURATION, noise_db=-32.0, min_duration=1.0
        )
        assert spans == []

    def test_noise_dbが実際に効く(self, episode_wav: Path):
        """--silence-db が ffmpeg まで届いていること。

        合成エピソードの発話部分は 440Hz・振幅0.35（≒ -9dB）なので、
        しきい値を -5dB まで緩めると窓全体が無音扱いになる。
        """
        loose = ff.detect_silences(
            episode_wav, start=0.0, end=fixtures.EPISODE_DURATION, noise_db=-5.0, min_duration=0.12
        )
        assert len(loose) == 1
        assert loose[0][0] == pytest.approx(0.0, abs=0.05)
        assert loose[0][1] == pytest.approx(fixtures.EPISODE_DURATION, abs=0.05)

    def test_しきい値を下げても真の無音は拾える(self, episode_wav: Path):
        """谷は完全な0なので、-90dB まで厳しくしても8区間そのまま見つかる。"""
        spans = ff.detect_silences(
            episode_wav, start=0.0, end=fixtures.EPISODE_DURATION, noise_db=-90.0, min_duration=0.12
        )
        assert len(spans) == len(fixtures.SILENCES)

    def test_発話の途中だけを窓にすれば無音は無い(self, episode_wav: Path):
        """SPEC Step 4 の「無音が検出されなければフォールバック」に入る条件。"""
        assert ff.detect_silences(
            episode_wav, start=6.5, end=11.0, noise_db=-32.0, min_duration=0.12
        ) == []

    def test_窓が音声の外でも落ちない(self, episode_wav: Path):
        spans = ff.detect_silences(
            episode_wav, start=fixtures.EPISODE_DURATION + 5, end=fixtures.EPISODE_DURATION + 10,
            noise_db=-32.0, min_duration=0.12,
        )
        assert spans == []

    def test_窓の開始が負でも絶対時刻はずれない(self, episode_wav: Path):
        """アンカーAが冒頭近くだと raw-1.5 が負になる。窓は0に丸めるが offset も0にすること。"""
        spans = ff.detect_silences(episode_wav, start=-3.0, end=6.5, noise_db=-32.0, min_duration=0.12)
        assert len(spans) == 1
        assert spans[0][0] == pytest.approx(5.60, abs=0.05)
        assert spans[0][1] == pytest.approx(5.95, abs=0.05)

    def test_窓が音声の末尾を越えても落ちない(self, episode_wav: Path):
        spans = ff.detect_silences(
            episode_wav, start=50.0, end=fixtures.EPISODE_DURATION + 10, noise_db=-32.0, min_duration=0.12
        )
        assert len(spans) == 1
        assert spans[0][0] == pytest.approx(51.20, abs=0.05)

    def test_パスに記号が含まれても無音検出できる(self, episode_wav: Path, tmp_path: Path):
        """episode_id は入力ファイル名の stem。フィルタ式にパスを埋め込んでいたら
        カンマやクォートでフィルタが壊れる。ここで気づけるようにしておく。
        """
        weird = tmp_path / "it's, a [test] dir" / "audio.wav"
        weird.parent.mkdir(parents=True, exist_ok=True)
        weird.write_bytes(episode_wav.read_bytes())
        spans = ff.detect_silences(weird, start=42.45, end=44.45, noise_db=-32.0, min_duration=0.12)
        assert len(spans) == 1
        assert spans[0][1] == pytest.approx(43.95, abs=0.05)

    def test_おとりの谷とアンカーBの谷は別物として出る(self, episode_wav: Path):
        """19.95秒の「ということで」（おとり）と 43.95秒の本命が、それぞれ自分の谷を持つこと。"""
        decoy = ff.detect_silences(episode_wav, start=18.5, end=20.5, noise_db=-32.0, min_duration=0.12)
        real = ff.detect_silences(episode_wav, start=42.45, end=44.45, noise_db=-32.0, min_duration=0.12)
        assert decoy[-1][1] - SILENCE_BACKOFF_SEC == pytest.approx(19.90, abs=0.05)
        assert real[-1][1] - SILENCE_BACKOFF_SEC == pytest.approx(fixtures.EXPECTED_CUT_B, abs=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestEncodeSegmentReal:
    """SPEC Step 7。`-ss` / `-t` の扱いが正しいこと＝指定した区間がそのまま出ること。"""

    def test_実尺が指定区間と一致する(self, episode_video: Path, tmp_path: Path):
        out = ff.encode_segment(episode_video, 10.0, 20.0, tmp_path / "seg.mp4", RenderConfig())
        assert ff.media_duration(out) == pytest.approx(10.0, abs=0.2)

    @pytest.mark.parametrize("start,end", [(0.0, 5.0), (5.9, 43.9), (43.9, 60.0), (12.345, 15.678)])
    def test_いろいろな区間で実尺が合う(self, episode_video: Path, tmp_path: Path, start, end):
        out = ff.encode_segment(
            episode_video, start, end, tmp_path / f"seg-{start}-{end}.mp4", RenderConfig()
        )
        assert ff.media_duration(out) == pytest.approx(end - start, abs=0.2)

    def test_切り出した中身が指定した区間そのものである(self, episode_video: Path, tmp_path: Path):
        """尺だけ合っていても中身がずれていたら意味がない。
        アンカーB直前の谷（43.60〜43.95）が、区間 [43.0, 46.0] の 0.60〜0.95 に来ること。
        """
        out = ff.encode_segment(episode_video, 43.0, 46.0, tmp_path / "seg.mp4", RenderConfig())
        wav = ff.extract_audio(out, tmp_path / "seg.wav")
        spans = ff.detect_silences(wav, start=0.0, end=3.0, noise_db=-32.0, min_duration=0.12)
        assert len(spans) == 1, f"想定外の無音区間: {spans}"
        assert spans[0][0] == pytest.approx(43.60 - 43.0, abs=0.15)
        assert spans[0][1] == pytest.approx(43.95 - 43.0, abs=0.15)

    def test_出力先の親ディレクトリを作る(self, episode_video: Path, tmp_path: Path):
        out = ff.encode_segment(
            episode_video, 1.0, 3.0, tmp_path / "out" / "ep" / "02_main.mp4", RenderConfig()
        )
        assert out.exists()

    def test_終端が総尺を越えても書き出せる(self, episode_video: Path, tmp_path: Path):
        """エンディングは「アンカーB〜終端」。終端の丸めで尺を少し超えても止まらないこと。"""
        out = ff.encode_segment(
            episode_video, 55.0, fixtures.EPISODE_DURATION + 5.0, tmp_path / "ending.mp4", RenderConfig()
        )
        assert ff.media_duration(out) == pytest.approx(5.0, abs=0.3)

    def test_既存ファイルを上書きする(self, episode_video: Path, tmp_path: Path):
        """--from-step 7 で焼き直すときに「上書きしますか」で固まらないこと。"""
        dst = tmp_path / "seg.mp4"
        dst.write_bytes(b"stale")
        out = ff.encode_segment(episode_video, 1.0, 3.0, dst, RenderConfig())
        assert ff.media_duration(out) == pytest.approx(2.0, abs=0.2)

    def test_本編とエンディングを繋ぐと元の尺になる(self, episode_video: Path, tmp_path: Path):
        """SPEC Phase 1。A〜B と B〜終端で元動画の A 以降を過不足なく覆うこと。"""
        a, b = fixtures.EXPECTED_CUT_A, fixtures.EXPECTED_CUT_B
        main = ff.encode_segment(episode_video, a, b, tmp_path / "02_main.mp4", RenderConfig())
        ending = ff.encode_segment(
            episode_video, b, fixtures.EPISODE_DURATION, tmp_path / "03_ending.mp4", RenderConfig()
        )
        total = ff.media_duration(main) + ff.media_duration(ending)
        assert total == pytest.approx(fixtures.EPISODE_DURATION - a, abs=0.3)

    def test_use_fallbackでCPUエンコードを強制できる(self, episode_video: Path, tmp_path: Path):
        out = ff.encode_segment(
            episode_video, 1.0, 3.0, tmp_path / "fb.mp4", RenderConfig(), use_fallback=True
        )
        assert ff.probe_media(out).video_codec == "h264"

    def test_使えないコーデックを強制するとstderr付きで止まる(self, episode_video: Path, tmp_path: Path):
        render = RenderConfig(video_codec="radio_cutter_bogus_encoder")
        with pytest.raises(FfmpegError) as exc:
            ff.encode_segment(episode_video, 1.0, 3.0, tmp_path / "x.mp4", render, use_fallback=False)
        assert exc.value.stderr

    def test_存在しない入力はstderr付きのFfmpegError(self, tmp_path: Path):
        with pytest.raises(FfmpegError) as exc:
            ff.encode_segment(tmp_path / "nope.mp4", 0.0, 1.0, tmp_path / "o.mp4", RenderConfig())
        assert exc.value.stderr
        assert "nope.mp4" in exc.value.stderr

    def test_シングルクォートを含むパスでも書き出せる(self, episode_video: Path, tmp_path: Path):
        d = tmp_path / "it's out"
        out = ff.encode_segment(episode_video, 1.0, 3.0, d / "a'b c.mp4", RenderConfig())
        assert out.exists()
        assert ff.media_duration(out) == pytest.approx(2.0, abs=0.2)


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestConcatFilesReal:
    """SPEC Step 7 の連結。`final.mp4` の尺が3本の合計と合うこと。"""

    def _segments(self, episode_video: Path, tmp_path: Path, spans):
        return [
            ff.encode_segment(episode_video, s, e, tmp_path / f"p{i}.mp4", RenderConfig())
            for i, (s, e) in enumerate(spans)
        ]

    def test_合計尺が3本の合計と一致する(self, episode_video: Path, tmp_path: Path):
        spans = [(0.0, 3.0), (3.0, 8.0), (8.0, 10.0)]
        parts = self._segments(episode_video, tmp_path, spans)
        out = ff.concat_files(parts, tmp_path / "out" / "final.mp4", tmp_path / "work")
        total = sum(e - s for s, e in spans)
        assert ff.media_duration(out) == pytest.approx(total, abs=RenderConfig().duration_tolerance_sec)

    def test_concat_txtがwork配下にfile形式で残る(self, episode_video: Path, tmp_path: Path):
        """SPEC 11章「中間ファイルは消さない」。"""
        parts = self._segments(episode_video, tmp_path, [(0.0, 2.0), (2.0, 4.0)])
        ff.concat_files(parts, tmp_path / "final.mp4", tmp_path / "work")
        listing = (tmp_path / "work" / "concat.txt").read_text(encoding="utf-8")
        lines = listing.splitlines()
        assert len(lines) == 2
        for line, part in zip(lines, parts):
            assert line.startswith("file '") and line.endswith("'")
            assert str(part.resolve()) in line
        assert listing.endswith("\n")

    def test_リスト名を変えられる(self, episode_video: Path, tmp_path: Path):
        parts = self._segments(episode_video, tmp_path, [(0.0, 2.0)])
        ff.concat_files(parts, tmp_path / "final.mp4", tmp_path / "work", list_name="preview.txt")
        assert (tmp_path / "work" / "preview.txt").exists()

    def test_シングルクォートを含むパスでも連結できる(self, episode_video: Path, tmp_path: Path):
        """`file '...'` の中のクォートを閉じて埋め直せていること。"""
        d = tmp_path / "it's work"
        parts = [
            ff.encode_segment(episode_video, 0.0, 2.0, d / "a'b.mp4", RenderConfig()),
            ff.encode_segment(episode_video, 2.0, 4.0, d / "don't stop.mp4", RenderConfig()),
        ]
        out = ff.concat_files(parts, d / "final'.mp4", d / "work dir")
        assert out.exists()
        assert ff.media_duration(out) == pytest.approx(4.0, abs=0.5)

    def test_1本だけでも連結できる(self, episode_video: Path, tmp_path: Path):
        parts = self._segments(episode_video, tmp_path, [(0.0, 3.0)])
        out = ff.concat_files(parts, tmp_path / "final.mp4", tmp_path / "work")
        assert ff.media_duration(out) == pytest.approx(3.0, abs=0.3)

    def test_連結の順序が保たれる(self, episode_video: Path, tmp_path: Path):
        """ハイライトを先頭に足す（SPEC Step 6 のチャプター計算の前提）。
        [43.0, 46.0] を先頭に、[10.0, 12.0] を後ろに繋いだら、谷は先頭側にだけ来る。
        """
        head = ff.encode_segment(episode_video, 43.0, 46.0, tmp_path / "head.mp4", RenderConfig())
        tail = ff.encode_segment(episode_video, 10.0, 12.0, tmp_path / "tail.mp4", RenderConfig())
        out = ff.concat_files([head, tail], tmp_path / "final.mp4", tmp_path / "work")
        wav = ff.extract_audio(out, tmp_path / "final.wav")
        spans = ff.detect_silences(wav, start=0.0, end=5.0, noise_db=-32.0, min_duration=0.12)
        assert len(spans) == 1, f"想定外の無音区間: {spans}"
        assert spans[0][0] == pytest.approx(0.60, abs=0.2)


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestEncoderListReal:
    def test_使えるエンコーダの一覧が取れる(self, clean_encoder_cache):
        names = ff.list_encoders()
        assert isinstance(names, set) and names
        assert "aac" in names, "AAC すら無い ffmpeg ビルドは想定していない"

    def test_一覧はコピーを返す(self, clean_encoder_cache):
        """呼び先が書き換えてもキャッシュを壊さないこと。"""
        first = ff.list_encoders()
        first.add("radio_cutter_bogus_encoder")
        assert "radio_cutter_bogus_encoder" not in ff.list_encoders()

    def test_存在しないエンコーダはFalse(self, clean_encoder_cache):
        assert ff.has_encoder("radio_cutter_bogus_encoder") is False
        assert ff.has_encoder("") is False

    def test_実ffmpegでフォールバックが決まる(self, clean_encoder_cache):
        """SPEC 2章。VideoToolbox が無い環境では libx264 に落ちること。"""
        render = RenderConfig(video_codec="radio_cutter_bogus_encoder", fallback_video_codec="libx264")
        codec, extra, fell_back = ff.choose_video_codec(render)
        assert codec == "libx264"
        assert extra == ("-preset", "veryfast", "-crf", "20")
        assert fell_back is True

    def test_ffmpeg_versionが文字列を返す(self):
        assert isinstance(ff.ffmpeg_version(), str)

    def test_require_binariesが通る(self):
        ff.require_binaries()


# ===========================================================================
# util/cache.py（SPEC Step 2 のキャッシュ）
# ===========================================================================

#: 既知の入力 → 既知の SHA-256
KNOWN_SHA = {
    b"": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    b"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    b"hello": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
}


class TestSha256File:
    @pytest.mark.parametrize("content,digest", sorted(KNOWN_SHA.items()))
    def test_既知の内容は既知のハッシュになる(self, tmp_path: Path, content, digest):
        """decisions.json の input_sha256 が本物の SHA-256 であること。"""
        p = tmp_path / "f.bin"
        p.write_bytes(content)
        assert sha256_file(p) == digest

    def test_チャンクサイズを変えても同じハッシュ(self, tmp_path: Path):
        """60分の mp4 をチャンク読みしても結果が変わらないこと。"""
        p = tmp_path / "big.bin"
        p.write_bytes(bytes(range(256)) * 400)  # 102400 バイト
        base = sha256_file(p)
        for chunk in (1, 7, 1024, 1 << 20):
            assert sha256_file(p, chunk_size=chunk) == base

    def test_1バイト違えばハッシュが変わる(self, tmp_path: Path):
        """入力が差し替わったのにキャッシュを使う、が起きないこと。"""
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"episode-content")
        b.write_bytes(b"episode-contenu")
        assert sha256_file(a) != sha256_file(b)

    def test_文字列のパスでも受ける(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        assert sha256_file(str(p)) == KNOWN_SHA[b"abc"]

    def test_chunk_sizeが0以下ならValueError(self, tmp_path: Path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abc")
        for bad in (0, -1):
            with pytest.raises(ValueError):
                sha256_file(p, chunk_size=bad)

    def test_ファイルが無ければRadioCutterError(self, tmp_path: Path):
        with pytest.raises(RadioCutterError) as exc:
            sha256_file(tmp_path / "missing.mp4")
        assert "missing.mp4" in str(exc.value)

    def test_ディレクトリを渡したらRadioCutterError(self, tmp_path: Path):
        with pytest.raises(RadioCutterError):
            sha256_file(tmp_path)


@dataclass(frozen=True)
class _Sample:
    a: int
    b: str


class TestStableHash:
    def test_キーの順序に依存しない(self):
        """dict のリテラル順が変わっただけでキャッシュが無効化されないこと。"""
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_入れ子のキー順にも依存しない(self):
        left = {"asr": {"model": "m", "language": "ja", "beam_size": 5}, "input_sha256": "x"}
        right = {"input_sha256": "x", "asr": {"beam_size": 5, "language": "ja", "model": "m"}}
        assert stable_hash(left) == stable_hash(right)

    def test_値が違えばハッシュも違う(self):
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})

    def test_配列の順序は意味を持つ(self):
        """順序に意味のある値（プロンプト列など）まで無視しないこと。"""
        assert stable_hash([1, 2]) != stable_hash([2, 1])

    def test_集合は並べ替えて安定させる(self):
        """set の反復順は実行ごとに変わりうる。キャッシュキーが揺れないこと。"""
        assert stable_hash({"x": {"b", "a"}}) == stable_hash({"x": {"a", "b"}})

    def test_Pathは文字列として扱う(self, tmp_path: Path):
        assert stable_hash({"p": Path("/tmp/a.mp4")}) == stable_hash({"p": "/tmp/a.mp4"})

    def test_dataclassも扱える(self):
        assert stable_hash(_Sample(1, "x")) == stable_hash({"a": 1, "b": "x"})

    def test_日本語を含んでも安定する(self):
        payload = {"initial_prompt": "AI活用法実験ラジオ。このチャンネルは。"}
        assert stable_hash(payload) == stable_hash(dict(payload))

    def test_同じ呼び出しは何度でも同じ値(self):
        payload = {"asr": {"model": "m"}, "input_sha256": "abc"}
        assert len({stable_hash(payload) for _ in range(5)}) == 1

    def test_SHA256の16進64文字を返す(self):
        h = stable_hash({"a": 1})
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    def test_JSONにできない値はRadioCutterError(self):
        """握りつぶして常に同じキーを返すと、別設定でキャッシュが当たってしまう。"""
        with pytest.raises(RadioCutterError):
            stable_hash({"x": object()})


class TestTranscriptCacheKey:
    def test_同じ入力と同じASR設定なら同じキー(self, config):
        """SPEC Step 2「同一ならこのステップを丸ごとスキップ」。"""
        payload = config.asr.cache_key_payload()
        assert transcript_cache_key("abc", payload) == transcript_cache_key("abc", dict(payload))

    def test_入力が変わればキーも変わる(self, config):
        payload = config.asr.cache_key_payload()
        assert transcript_cache_key("abc", payload) != transcript_cache_key("abd", payload)

    def test_ASR設定が変わればキーも変わる(self, config):
        """initial_prompt を変えたのに古い文字起こしを使う、が起きないこと。"""
        base = config.asr.cache_key_payload()
        changed = dict(base, initial_prompt=base["initial_prompt"] + "！")
        assert transcript_cache_key("abc", base) != transcript_cache_key("abc", changed)

    def test_モデルが変わればキーも変わる(self, config):
        base = config.asr.cache_key_payload()
        assert transcript_cache_key("abc", base) != transcript_cache_key(
            "abc", dict(base, model="tiny")
        )

    def test_ASR設定のキー順は影響しない(self, config):
        base = config.asr.cache_key_payload()
        shuffled = {k: base[k] for k in reversed(list(base))}
        assert transcript_cache_key("abc", base) == transcript_cache_key("abc", shuffled)


class TestCacheEntryIo:
    def _entry(self, key="k" * 64) -> TranscriptCacheEntry:
        return TranscriptCacheEntry(
            input_sha256="a" * 64,
            asr_hash="b" * 64,
            key=key,
            created_at="2026-08-30T14:20:11+09:00",
        )

    def test_書いて読み戻せる(self, tmp_path: Path):
        p = tmp_path / "work" / "ep" / "transcript_cache.json"
        entry = self._entry()
        save_cache_entry(p, entry)
        assert load_cache_entry(p) == entry

    def test_保存後に一時ファイルが残らない(self, tmp_path: Path):
        p = tmp_path / "transcript_cache.json"
        save_cache_entry(p, self._entry())
        assert not (tmp_path / "transcript_cache.json.tmp").exists()

    def test_上書きできる(self, tmp_path: Path):
        p = tmp_path / "transcript_cache.json"
        save_cache_entry(p, self._entry(key="old"))
        save_cache_entry(p, self._entry(key="new"))
        loaded = load_cache_entry(p)
        assert loaded is not None and loaded.key == "new"

    def test_transcript_fileの既定値(self, tmp_path: Path):
        p = tmp_path / "c.json"
        save_cache_entry(p, self._entry())
        loaded = load_cache_entry(p)
        assert loaded is not None and loaded.transcript_file == DEFAULT_TRANSCRIPT_FILE

    def test_ファイルが無ければNone(self, tmp_path: Path):
        """初回実行。例外ではなく None（文字起こしをやり直す）。"""
        assert load_cache_entry(tmp_path / "nope.json") is None

    def test_壊れたJSONならNone(self, tmp_path: Path):
        """SPEC の方針「壊れたキャッシュで実行を止めない」。"""
        p = tmp_path / "c.json"
        p.write_text("{ this is not json", encoding="utf-8")
        assert load_cache_entry(p) is None

    def test_途中で切れたJSONならNone(self, tmp_path: Path):
        p = tmp_path / "c.json"
        p.write_text('{"input_sha256": "a", "asr_hash":', encoding="utf-8")
        assert load_cache_entry(p) is None

    def test_空ファイルならNone(self, tmp_path: Path):
        p = tmp_path / "c.json"
        p.write_text("", encoding="utf-8")
        assert load_cache_entry(p) is None

    @pytest.mark.parametrize("body", ["[1, 2, 3]", '"just a string"', "null", "123"])
    def test_JSONオブジェクトでなければNone(self, tmp_path: Path, body):
        p = tmp_path / "c.json"
        p.write_text(body, encoding="utf-8")
        assert load_cache_entry(p) is None

    @pytest.mark.parametrize("missing", ["input_sha256", "asr_hash", "key", "created_at"])
    def test_必須項目が欠けていればNone(self, tmp_path: Path, missing):
        d = self._entry().to_dict()
        d.pop(missing)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert load_cache_entry(p) is None

    @pytest.mark.parametrize("bad", ["", None, 123, []])
    def test_必須項目が空や型違いならNone(self, tmp_path: Path, bad):
        d = self._entry().to_dict()
        d["key"] = bad
        p = tmp_path / "c.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert load_cache_entry(p) is None

    def test_transcript_fileが不正ならNone(self, tmp_path: Path):
        d = self._entry().to_dict()
        d["transcript_file"] = ""
        p = tmp_path / "c.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert load_cache_entry(p) is None

    def test_ディレクトリを渡してもNone(self, tmp_path: Path):
        assert load_cache_entry(tmp_path) is None

    def test_未知の項目があっても読める(self, tmp_path: Path):
        """将来キャッシュに項目が増えても、古い実装が落ちないこと。"""
        d = self._entry().to_dict()
        d["future_field"] = {"anything": 1}
        p = tmp_path / "c.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        loaded = load_cache_entry(p)
        assert loaded is not None and loaded.key == "k" * 64

    def test_matchesは同じキーだけTrue(self):
        entry = self._entry(key="abc")
        assert entry.matches("abc") is True
        assert entry.matches("abd") is False

    def test_matchesは空キーをTrueにしない(self):
        """キーの計算に失敗した空文字で「一致」と判定しないこと。"""
        assert self._entry(key="").matches("") is False

    def test_実際のキーで往復する(self, tmp_path: Path, config):
        """Step 2 が実際にやる流れ: 入力ハッシュ＋ASR設定 → キー → 保存 → 一致判定。"""
        src = tmp_path / "ep.mp4"
        src.write_bytes(b"pretend this is an episode")
        sha = sha256_file(src)
        payload = config.asr.cache_key_payload()
        key = transcript_cache_key(sha, payload)
        p = tmp_path / "transcript_cache.json"
        save_cache_entry(
            p,
            TranscriptCacheEntry(
                input_sha256=sha, asr_hash=stable_hash(payload), key=key, created_at="2026-08-30T00:00:00+09:00"
            ),
        )
        loaded = load_cache_entry(p)
        assert loaded is not None
        assert loaded.matches(transcript_cache_key(sha, payload)) is True
        assert loaded.matches(transcript_cache_key("other-sha", payload)) is False
