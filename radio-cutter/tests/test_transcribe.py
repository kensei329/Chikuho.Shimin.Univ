"""steps/s1_extract_audio.py と steps/s2_transcribe.py — パイプラインの入口。

ここで作られる `work/audio.wav` と `work/transcript.json` が
Step 3 以降のすべての土台になる。特に文字起こしは

- 単語の start が1つでも欠けると Step 3 がカット時刻を引けない
- 単語列が時刻順に並んでいないと util/timeline.py の二分探索が黙って誤答する
- 全工程の8割の時間を占めるのでキャッシュが効かないと運用にならない（SPEC 6章 Step2）

という壊れ方をする。どれも「動くけれど間違っている」ので、
バックエンドを一切呼ばずに検証できる `normalize_asr_result`（純関数）を厚く固め、
キャッシュはバックエンド呼び出し回数を数えて確かめる。

SPEC 由来の契約:
- Step 1: `ffmpeg -vn -ac 1 -ar 16000 -c:a pcm_s16le` で `work/audio.wav`、
  `ffprobe` の結果を `work/probe.json`（SPEC 6章 Step1）
- Step 2: 出力は language / duration / segments[].{start,end,text,words[].{word,start,end}}
  （SPEC 6章 Step2 の JSON 例）
- Step 2: `mlx-whisper` を使い、利用できない場合は `whisperx` にフォールバックする
- Step 2: `config.asr.initial_prompt` を必ず渡す
- Step 2: 入力の SHA-256 と ASR 設定のハッシュをキーにしてステップを丸ごとスキップする
- SPEC 9章: ffmpeg が非ゼロ終了したら握りつぶさず止める
- SPEC 11章: 秒数は float、保存時は小数点以下3桁
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import AsrConfig, Config, SilenceConfig
from radio_cutter.context import RunContext
from radio_cutter.errors import (
    FfmpegError,
    MissingArtifactError,
    RadioCutterError,
    TranscriptionError,
)
from radio_cutter.models import Transcript
from radio_cutter.steps import s1_extract_audio as s1
from radio_cutter.steps import s2_transcribe as s2
from radio_cutter.util.ffmpeg import MediaInfo

# ---------------------------------------------------------------------------
# 共通のヘルパ
# ---------------------------------------------------------------------------

#: バックエンドを差し替えたテストで run() に渡す入力の素性（ffprobe を呼ばせないため）
FAKE_MEDIA = MediaInfo(
    path="ep-test.mp4",
    duration=fixtures.EPISODE_DURATION,
    fps=15.0,
    width=160,
    height=120,
    video_codec="h264",
    audio_codec="aac",
    has_video=True,
    has_audio=True,
)

#: whisperx が返す形（segments[].words[].{word,start,end}）
WHISPERX_RAW: dict = {
    "language": "ja",
    "segments": [
        {
            "start": 0.42,
            "end": 1.30,
            "text": "このチャンネルは",
            "words": [
                {"word": "この", "start": 0.42, "end": 0.66},
                {"word": "チャンネル", "start": 0.66, "end": 1.18},
                {"word": "は", "start": 1.18, "end": 1.30},
            ],
        },
        {
            "start": 2.00,
            "end": 3.40,
            "text": "AIの活用法を",
            "words": [
                {"word": "AI", "start": 2.00, "end": 2.40},
                {"word": "の活用法を", "start": 2.40, "end": 3.40},
            ],
        },
    ],
}

#: mlx-whisper が返す形（words[].{text,start,end}）
MLX_RAW: dict = {
    "language": "ja",
    "text": "このチャンネルはAIの活用法を",
    "segments": [
        {
            "id": 0,
            "start": 0.42,
            "end": 1.30,
            "text": "このチャンネルは",
            "words": [
                {"text": "この", "start": 0.42, "end": 0.66},
                {"text": "チャンネル", "start": 0.66, "end": 1.18},
                {"text": "は", "start": 1.18, "end": 1.30},
            ],
        },
        {
            "id": 1,
            "start": 2.00,
            "end": 3.40,
            "text": "AIの活用法を",
            "words": [
                {"text": "AI", "start": 2.00, "end": 2.40},
                {"text": "の活用法を", "start": 2.40, "end": 3.40},
            ],
        },
    ],
}


def seg(start: float, end: float, text: str = "", words=None) -> dict:
    """テスト用のセグメント1件を組み立てる（words は生 dict のまま渡す）。"""
    d: dict = {"start": start, "end": end, "text": text}
    if words is not None:
        d["words"] = words
    return d


def starts_of(transcript: Transcript) -> list[float]:
    return [w.start for w in transcript.words()]


def texts_of(transcript: Transcript) -> list[str]:
    return [w.word for w in transcript.words()]


def prepare_audio(ctx: RunContext) -> Path:
    """work/audio.wav をでっち上げる。

    バックエンドを差し替えるテストでは中身は読まれない。
    Step 2 が「音声が無ければ止まる」ことだけが効いてくる。
    """
    ctx.ensure_dirs()
    path = ctx.work_path(s1.AUDIO_FILENAME)
    path.write_bytes(b"RIFF----WAVEfmt ")
    return path


def install_backends(monkeypatch, raw: dict | None = None) -> list[dict]:
    """全バックエンドを差し替え、呼び出しを記録するリストを返す。

    実際に呼ばれるのは `_backend_order()` が返した最初の1つだけなので、
    リストの長さがそのまま「文字起こしを何回走らせたか」になる。
    """
    calls: list[dict] = []
    payload = WHISPERX_RAW if raw is None else raw

    def make(name: str):
        def fake(audio_path, asr):
            calls.append({"backend": name, "audio": Path(audio_path), "asr": asr})
            return json.loads(json.dumps(payload))  # 呼び出しごとに独立した dict

        return fake

    for name in list(s2._BACKENDS):
        monkeypatch.setitem(s2._BACKENDS, name, make(name))
    return calls


def hide_real_backends(monkeypatch) -> None:
    """mlx-whisper / whisperx が「入っていない」環境を作る。

    sys.modules に None を置くと `import x` が ImportError になる。
    この環境に本当に入っているかどうかに関係なくテストを成立させるため。
    """
    for name in ("mlx_whisper", "whisperx"):
        monkeypatch.setitem(sys.modules, name, None)


def with_asr(ctx: RunContext, **overrides) -> AsrConfig:
    """ctx の ASR 設定だけを差し替える（session スコープの Config は壊さない）。"""
    new_asr = dataclasses.replace(ctx.config.asr, **overrides)
    ctx.config = dataclasses.replace(ctx.config, asr=new_asr)
    return new_asr


def fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ===========================================================================
# Step 2 — normalize_asr_result（純関数）
# ===========================================================================


class TestNormalizeWhisperxShape:
    """whisperx 形式（segments[].words[].{word,start,end}）をそのまま読めること。"""

    def test_単語と時刻がそのまま保たれる(self):
        t = s2.normalize_asr_result(WHISPERX_RAW, 60.0)

        assert isinstance(t, Transcript)
        assert texts_of(t) == ["この", "チャンネル", "は", "AI", "の活用法を"]
        assert starts_of(t) == [0.42, 0.66, 1.18, 2.00, 2.40]
        assert [w.end for w in t.words()] == [0.66, 1.18, 1.30, 2.40, 3.40]

    def test_セグメントの区切りが保たれる(self):
        """Step 5 の文単位スナップがセグメントを見るので、1本に潰してはいけない。"""
        t = s2.normalize_asr_result(WHISPERX_RAW, 60.0)

        assert len(t.segments) == 2
        assert t.segments[0].text == "このチャンネルは"
        assert t.segments[1].start == pytest.approx(2.00)

    def test_言語は結果のものを優先する(self):
        t = s2.normalize_asr_result({"language": "en", "segments": []}, 10.0, language="ja")
        assert t.language == "en"

    def test_結果に言語が無ければ設定の言語を使う(self):
        t = s2.normalize_asr_result({"segments": []}, 10.0, language="ja")
        assert t.language == "ja"


class TestNormalizeMlxShape:
    """mlx-whisper 形式（words[].{text,start,end}）も同じ Transcript になること。

    キーが "word" か "text" かはバックエンドの都合であって、
    Step 3 以降がそれを気にしてはいけない。
    """

    def test_textキーの単語を読める(self):
        t = s2.normalize_asr_result(MLX_RAW, 60.0)

        assert texts_of(t) == ["この", "チャンネル", "は", "AI", "の活用法を"]
        assert starts_of(t) == [0.42, 0.66, 1.18, 2.00, 2.40]

    def test_whisperx形式と同じ結果になる(self):
        mlx = s2.normalize_asr_result(MLX_RAW, 60.0)
        wx = s2.normalize_asr_result(WHISPERX_RAW, 60.0)
        assert mlx.to_dict() == wx.to_dict()

    def test_wordキーとtextキーが混在しても読める(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    2.0,
                    "あいう",
                    [
                        {"word": "あ", "start": 0.0, "end": 0.5},
                        {"text": "い", "start": 0.5, "end": 1.0},
                        {"word": "う", "start": 1.0, "end": 2.0},
                    ],
                )
            ]
        }
        t = s2.normalize_asr_result(raw, 10.0)
        assert texts_of(t) == ["あ", "い", "う"]


class TestNormalizeEmptyWords:
    """words が空のセグメントは、セグメント全体を1単語として補完すること。

    ここで捨ててしまうと、その区間のテキストが Step 3 の flat から消えて
    アンカーが「無い」ことになる。
    """

    def test_wordsが空配列ならセグメント全体が1単語になる(self):
        raw = {"segments": [seg(2.0, 5.0, "まるごと1単語になるはず", [])]}
        t = s2.normalize_asr_result(raw, 60.0)

        words = t.words()
        assert len(words) == 1
        assert words[0].word == "まるごと1単語になるはず"
        assert words[0].start == pytest.approx(2.0)
        assert words[0].end == pytest.approx(5.0)

    def test_wordsキーが無くてもセグメント全体が1単語になる(self):
        raw = {"segments": [{"start": 2.0, "end": 5.0, "text": "wordsキーが無い"}]}
        t = s2.normalize_asr_result(raw, 60.0)

        words = t.words()
        assert len(words) == 1
        assert words[0].word == "wordsキーが無い"
        assert (words[0].start, words[0].end) == (pytest.approx(2.0), pytest.approx(5.0))

    def test_wordsもtextも無いセグメントは捨てる(self):
        """時刻しか無いセグメントを残すと、flat に空文字が混ざって索引がずれる。"""
        raw = {"segments": [seg(2.0, 5.0, "", []), seg(6.0, 7.0, "残るほう", [])]}
        t = s2.normalize_asr_result(raw, 60.0)

        assert [s.text for s in t.segments] == ["残るほう"]

    def test_wordsが空でも他のセグメントと時刻順に並ぶ(self):
        raw = {
            "segments": [
                seg(6.0, 8.0, "あとの文", []),
                seg(1.0, 3.0, "さきの文", [{"word": "さきの文", "start": 1.0, "end": 3.0}]),
            ]
        }
        t = s2.normalize_asr_result(raw, 60.0)
        assert texts_of(t) == ["さきの文", "あとの文"]


class TestNormalizeMissingWordTimes:
    """start/end が欠けた単語を前後から補間すること。

    whisperx のアライメントは数字・記号・フィラーでよく外れて時刻が落ちる。
    落ちたまま通すと Step 3 が「一致位置の単語の start」を引けない。
    """

    def test_両方欠けた単語は前後の間に収まる(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    4.0,
                    "",
                    [
                        {"word": "あ", "start": 0.0, "end": 1.0},
                        {"word": "い"},
                        {"word": "う", "start": 3.0, "end": 4.0},
                    ],
                )
            ]
        }
        t = s2.normalize_asr_result(raw, 10.0)
        a, i, u = t.words()

        assert (a.start, a.end) == (pytest.approx(0.0), pytest.approx(1.0))
        assert (u.start, u.end) == (pytest.approx(3.0), pytest.approx(4.0))
        # 前の単語の終わりから次の単語の始まりまでの間に入っていること
        assert a.end <= i.start <= i.end <= u.start

    def test_連続して欠けても順番を保って埋まる(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    6.0,
                    "",
                    [
                        {"word": "あ", "start": 0.0, "end": 2.0},
                        {"word": "い"},
                        {"word": "う"},
                        {"word": "え", "start": 5.0, "end": 6.0},
                    ],
                )
            ]
        }
        words = s2.normalize_asr_result(raw, 10.0).words()

        assert [w.word for w in words] == ["あ", "い", "う", "え"]
        assert 2.0 <= words[1].start <= words[1].end <= words[2].start
        assert words[2].start <= words[2].end <= 5.0

    def test_startだけ分かっている単語はその値を捨てない(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    4.0,
                    "",
                    [
                        {"word": "あ", "start": 0.0, "end": 1.0},
                        {"word": "い", "start": 1.5},
                        {"word": "う", "start": 3.0, "end": 4.0},
                    ],
                )
            ]
        }
        i = s2.normalize_asr_result(raw, 10.0).words()[1]

        assert i.start == pytest.approx(1.5), "分かっている start を補間で上書きしてはいけない"
        assert i.start <= i.end <= 3.0

    def test_endだけ分かっている単語はその値を捨てない(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    4.0,
                    "",
                    [
                        {"word": "あ", "start": 0.0, "end": 1.0},
                        {"word": "い", "end": 2.5},
                        {"word": "う", "start": 3.0, "end": 4.0},
                    ],
                )
            ]
        }
        i = s2.normalize_asr_result(raw, 10.0).words()[1]

        assert i.end == pytest.approx(2.5), "分かっている end を補間で上書きしてはいけない"
        assert 1.0 <= i.start <= i.end

    def test_手がかりが無ければセグメントの時刻で挟む(self):
        """前にも後ろにも既知の時刻が無いときはセグメントの start/end に落とす。"""
        raw = {"segments": [seg(10.0, 12.0, "", [{"word": "あ"}, {"word": "い"}])]}
        words = s2.normalize_asr_result(raw, 20.0).words()

        assert words[0].start == pytest.approx(10.0)
        assert words[-1].end == pytest.approx(12.0)
        assert words[0].end <= words[1].start

    def test_先頭の単語だけ欠けてもセグメント開始から埋まる(self):
        raw = {
            "segments": [
                seg(5.0, 8.0, "", [{"word": "あ"}, {"word": "い", "start": 6.0, "end": 8.0}])
            ]
        }
        words = s2.normalize_asr_result(raw, 20.0).words()

        assert words[0].start == pytest.approx(5.0)
        assert words[0].end <= 6.0

    def test_末尾の単語だけ欠けてもセグメント終了まで埋まる(self):
        raw = {
            "segments": [
                seg(5.0, 8.0, "", [{"word": "あ", "start": 5.0, "end": 6.0}, {"word": "い"}])
            ]
        }
        words = s2.normalize_asr_result(raw, 20.0).words()

        assert words[1].start >= 6.0
        assert words[1].end == pytest.approx(8.0)

    def test_時刻の無い単語だけのセグメントでも時刻は必ず埋まる(self):
        """SPEC 11章「秒数は全て float」。None のまま Word を作ってはいけない。"""
        raw = {"segments": [seg(1.0, 2.0, "", [{"word": "あ"}, {"word": "い"}, {"word": "う"}])]}
        for w in s2.normalize_asr_result(raw, 20.0).words():
            assert isinstance(w.start, float) and isinstance(w.end, float)
            assert w.start <= w.end


class TestNormalizeSegmentTimes:
    """セグメントの start/end が欠けている・矛盾している入力。"""

    def test_セグメントのstartが無ければ単語から求める(self):
        raw = {
            "segments": [
                {
                    "text": "あい",
                    "words": [
                        {"word": "あ", "start": 3.0, "end": 3.5},
                        {"word": "い", "start": 3.5, "end": 4.0},
                    ],
                }
            ]
        }
        s = s2.normalize_asr_result(raw, 20.0).segments[0]

        assert s.start == pytest.approx(3.0)
        assert s.end == pytest.approx(4.0)

    def test_セグメントのテキストが無ければ単語から組み立てる(self):
        """Step 3 は flat を単語から作るが、Step 5/6 はセグメントの text を読む。"""
        raw = {
            "segments": [
                seg(
                    0.0,
                    2.0,
                    "",
                    [
                        {"word": "この", "start": 0.0, "end": 1.0},
                        {"word": "チャンネルは", "start": 1.0, "end": 2.0},
                    ],
                )
            ]
        }
        assert s2.normalize_asr_result(raw, 20.0).segments[0].text == "このチャンネルは"

    def test_セグメントの範囲は単語をはみ出さない(self):
        """単語がセグメント外にはみ出していたらセグメント側を広げる（区間検索が取りこぼすため）。"""
        raw = {
            "segments": [
                seg(
                    1.0,
                    2.0,
                    "あい",
                    [
                        {"word": "あ", "start": 0.5, "end": 1.5},
                        {"word": "い", "start": 1.5, "end": 3.0},
                    ],
                )
            ]
        }
        s = s2.normalize_asr_result(raw, 20.0).segments[0]

        assert s.start <= 0.5
        assert s.end >= 3.0

    def test_辞書でないセグメントは捨てる(self):
        raw = {"segments": ["こわれている", None, 42, seg(1.0, 2.0, "生き残る", [])]}
        t = s2.normalize_asr_result(raw, 20.0)
        assert [s.text for s in t.segments] == ["生き残る"]


class TestNormalizeOrdering:
    """`words()` は必ず時刻の昇順であること。

    util/timeline.py の `word_index_at_time` は starts に bisect をかける。
    並びが崩れていても例外は出ず、静かに違う単語を指す。
    Step 3 のカット時刻・Step 5 のスナップがまとめてずれるので、ここは絶対に守る。
    """

    def test_セグメントが順不同でも時刻順に並べ直す(self):
        raw = {
            "segments": [
                seg(10.0, 12.0, "あと", [{"word": "あと", "start": 10.0, "end": 12.0}]),
                seg(1.0, 2.0, "さき", [{"word": "さき", "start": 1.0, "end": 2.0}]),
            ]
        }
        t = s2.normalize_asr_result(raw, 20.0)

        assert [s.text for s in t.segments] == ["さき", "あと"]
        assert starts_of(t) == sorted(starts_of(t))

    def test_セグメント内で時刻が逆行しても昇順に均す(self):
        raw = {
            "segments": [
                seg(
                    0.0,
                    6.0,
                    "",
                    [
                        {"word": "a", "start": 5.0, "end": 6.0},
                        {"word": "b", "start": 2.0, "end": 3.0},
                    ],
                )
            ]
        }
        words = s2.normalize_asr_result(raw, 10.0).words()

        assert [w.word for w in words] == ["a", "b"]
        assert [w.start for w in words] == sorted(w.start for w in words)
        assert all(w.start <= w.end for w in words)

    def test_セグメントが重なっていても単語列は昇順になる(self):
        """whisperx は VAD のチャンク境界でセグメントを重ねて返すことがある。

        セグメント単位で並べ替えるだけでは、後ろのセグメントの単語が
        前のセグメントの終盤より前に来て `words()` が昇順でなくなる。
        """
        raw = {
            "segments": [
                seg(
                    0.0,
                    9.0,
                    "",
                    [
                        {"word": "あ", "start": 0.0, "end": 1.0},
                        {"word": "い", "start": 8.0, "end": 9.0},
                    ],
                ),
                seg(4.0, 5.0, "", [{"word": "う", "start": 4.0, "end": 5.0}]),
            ]
        }
        t = s2.normalize_asr_result(raw, 20.0)

        assert starts_of(t) == sorted(starts_of(t)), (
            "words() が時刻順でないと Step 3 の二分探索が違う単語を指す"
        )

    def test_負の時刻は0に寄せる(self):
        raw = {
            "segments": [
                seg(-1.0, 2.0, "", [{"word": "あ", "start": -0.5, "end": 1.0}]),
            ]
        }
        w = s2.normalize_asr_result(raw, 10.0).words()[0]
        assert w.start >= 0.0


class TestNormalizeDuration:
    """duration の反映（decisions.json と Step 6 のチャプター計算が読む）。"""

    def test_渡した総尺がそのまま入る(self):
        t = s2.normalize_asr_result(WHISPERX_RAW, 3612.4)
        assert t.duration == pytest.approx(3612.4)

    @pytest.mark.parametrize("bad", [0, 0.0, -5.0, None])
    def test_総尺が取れないときは最後のセグメント終端で代用する(self, bad):
        raw = {"segments": [seg(0.0, 7.5, "x", [{"word": "x", "start": 0.0, "end": 7.5}])]}
        t = s2.normalize_asr_result(raw, bad)
        assert t.duration == pytest.approx(7.5)

    def test_セグメントも総尺も無ければ0になる(self):
        t = s2.normalize_asr_result({"segments": []}, 0.0)
        assert t.duration == pytest.approx(0.0)


class TestNormalizeEmptyAndBroken:
    """空・壊れた入力。ここは「静かに間違える」より「止まる」が正しい。"""

    def test_セグメントが空でもTranscriptを返す(self):
        t = s2.normalize_asr_result({"language": "ja", "segments": []}, 60.0)

        assert isinstance(t, Transcript)
        assert t.segments == []
        assert t.words() == []
        assert t.duration == pytest.approx(60.0)

    def test_segmentsキーが無くてもTranscriptを返す(self):
        t = s2.normalize_asr_result({"language": "ja"}, 60.0)
        assert isinstance(t, Transcript)
        assert t.segments == []

    def test_segmentsが配列でなければ止まる(self):
        with pytest.raises(TranscriptionError):
            s2.normalize_asr_result({"segments": {"start": 0}}, 60.0)

    @pytest.mark.parametrize("bad", ["文字列", ["配列"], 42, None])
    def test_辞書でない結果は止まる(self, bad):
        with pytest.raises(TranscriptionError):
            s2.normalize_asr_result(bad, 60.0)

    def test_空白だけの単語は捨てる(self):
        """空文字の単語を残すと flat の文字index → 単語index の対応がずれる。"""
        raw = {
            "segments": [
                seg(
                    0.0,
                    3.0,
                    "あい",
                    [
                        {"word": "あ", "start": 0.0, "end": 1.0},
                        {"word": "   ", "start": 1.0, "end": 2.0},
                        {"word": "い", "start": 2.0, "end": 3.0},
                    ],
                )
            ]
        }
        assert texts_of(s2.normalize_asr_result(raw, 10.0)) == ["あ", "い"]


class TestTranscriptJsonShape:
    """SPEC 6章 Step2 が示す work/transcript.json の形をそのまま守ること。"""

    def test_保存したJSONがSPECのキー構成になる(self, tmp_path):
        t = s2.normalize_asr_result(WHISPERX_RAW, 3612.4)
        path = tmp_path / "transcript.json"
        t.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"language", "duration", "segments"}
        assert data["language"] == "ja"
        assert data["duration"] == pytest.approx(3612.4)

        first = data["segments"][0]
        assert set(first) == {"start", "end", "text", "words"}
        assert set(first["words"][0]) == {"word", "start", "end"}

    def test_秒数は小数点以下3桁で保存される(self, tmp_path):
        """SPEC 11章「秒数は全て float（小数点以下3桁）」。"""
        raw = {
            "segments": [
                seg(
                    0.1234567,
                    1.9876543,
                    "あ",
                    [{"word": "あ", "start": 0.1234567, "end": 1.9876543}],
                )
            ]
        }
        path = tmp_path / "transcript.json"
        s2.normalize_asr_result(raw, 60.0).save(path)

        word = json.loads(path.read_text(encoding="utf-8"))["segments"][0]["words"][0]
        assert word["start"] == 0.123
        assert word["end"] == 1.988

    def test_保存して読み直しても同じ内容になる(self, tmp_path):
        t = s2.normalize_asr_result(WHISPERX_RAW, 60.0)
        path = tmp_path / "transcript.json"
        t.save(path)
        assert Transcript.load(path).to_dict() == t.to_dict()


# ===========================================================================
# Step 2 — バックエンドの選択とフォールバック
# ===========================================================================


class TestBackendUnavailable:
    """バックエンドがどれも無い環境では止まり、入れ方を案内すること（SPEC 6章 Step2）。"""

    def test_バックエンドが無ければTranscriptionErrorになる(self, ctx, monkeypatch):
        hide_real_backends(monkeypatch)
        prepare_audio(ctx)

        with pytest.raises(TranscriptionError) as exc:
            s2.run(ctx, FAKE_MEDIA)

        message = str(exc.value)
        assert "mlx-whisper" in message
        assert "whisperx" in message

    def test_メッセージに両方の入れ方が書いてある(self, ctx, monkeypatch):
        hide_real_backends(monkeypatch)
        prepare_audio(ctx)

        with pytest.raises(TranscriptionError) as exc:
            s2.run(ctx, FAKE_MEDIA)

        message = str(exc.value)
        assert "pip install mlx-whisper" in message
        assert "pip install whisperx" in message

    def test_バックエンドが無いときtranscript_jsonを作らない(self, ctx, monkeypatch):
        """中途半端な成果物を残すと --from-step 3 が壊れた入力で走ってしまう。"""
        hide_real_backends(monkeypatch)
        prepare_audio(ctx)

        with pytest.raises(TranscriptionError):
            s2.run(ctx, FAKE_MEDIA)

        assert not s2.transcript_path(ctx).exists()
        assert not s2.cache_path(ctx).exists()

    def test_未知のbackend名は止まる(self, ctx, monkeypatch):
        prepare_audio(ctx)
        with_asr(ctx, backend="mlx-whisper-turbo-9000")

        with pytest.raises(TranscriptionError) as exc:
            s2.run(ctx, FAKE_MEDIA)
        assert "backend" in str(exc.value)


class TestBackendSelection:
    """SPEC 6章 Step2:「mlx-whisper を使い、利用できない場合は whisperx にフォールバックする」。"""

    def test_autoならmlx_whisperを先に試す(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)
        with_asr(ctx, backend="auto")

        s2.run(ctx, FAKE_MEDIA)

        assert calls[0]["backend"] == "mlx_whisper", (
            "SPEC は mlx-whisper が主で whisperx がフォールバック。順番が逆だと "
            "config の mlx 用モデル（mlx-community/...）を whisperx に渡すことになる"
        )

    def test_mlxが使えなければwhisperxにフォールバックする(self, ctx, monkeypatch):
        calls: list[str] = []

        def unavailable(audio_path, asr):
            calls.append("mlx_whisper")
            raise s2._BackendUnavailable("mlx-whisper が入っていません。")

        def ok(audio_path, asr):
            calls.append("whisperx")
            return json.loads(json.dumps(WHISPERX_RAW))

        monkeypatch.setitem(s2._BACKENDS, "mlx_whisper", unavailable)
        monkeypatch.setitem(s2._BACKENDS, "whisperx", ok)
        prepare_audio(ctx)
        with_asr(ctx, backend="auto")

        t = s2.run(ctx, FAKE_MEDIA)

        assert "whisperx" in calls
        assert texts_of(t) == ["この", "チャンネル", "は", "AI", "の活用法を"]

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("whisperx", "whisperx"),
            ("whisper-x", "whisperx"),
            ("mlx-whisper", "mlx_whisper"),
            ("mlx_whisper", "mlx_whisper"),
            ("mlx", "mlx_whisper"),
            ("whispermlx", "mlx_whisper"),
        ],
    )
    def test_backend名を固定するとそれだけを使う(self, ctx, monkeypatch, name, expected):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)
        with_asr(ctx, backend=name)

        s2.run(ctx, FAKE_MEDIA)

        assert [c["backend"] for c in calls] == [expected]

    def test_単語タイムスタンプが無い結果は受け付けない(self, ctx, monkeypatch):
        """SPEC Step2「単語レベルのタイムスタンプを取得する」。無ければ Step 3 が成立しない。"""
        no_words = {
            "language": "ja",
            "segments": [{"start": 0.0, "end": 3.0, "text": "単語が無い"}],
        }
        install_backends(monkeypatch, no_words)
        prepare_audio(ctx)

        with pytest.raises(TranscriptionError) as exc:
            s2.run(ctx, FAKE_MEDIA)
        assert "単語" in str(exc.value)

    def test_音声が無ければMissingArtifactErrorで案内する(self, ctx, monkeypatch):
        install_backends(monkeypatch)
        ctx.ensure_dirs()

        with pytest.raises(MissingArtifactError) as exc:
            s2.run(ctx, FAKE_MEDIA)
        assert s1.AUDIO_FILENAME in str(exc.value)


class TestInitialPromptIsPassed:
    """SPEC 6章 Step2:「config.asr.initial_prompt を必ず渡す」。

    アンカー語をモデルにバイアスさせるのが目的なので、
    これが落ちると Step 3 の検出率が静かに下がる。
    """

    def test_mlx_whisperにinitial_promptとword_timestampsを渡す(self, ctx, monkeypatch):
        captured: dict = {}

        def transcribe(path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return json.loads(json.dumps(MLX_RAW))

        monkeypatch.setitem(
            sys.modules, "mlx_whisper", fake_module("mlx_whisper", transcribe=transcribe)
        )
        monkeypatch.setitem(sys.modules, "whisperx", None)
        prepare_audio(ctx)
        asr = with_asr(ctx, backend="mlx_whisper")

        s2.run(ctx, FAKE_MEDIA)

        assert captured["initial_prompt"] == asr.initial_prompt
        assert captured["initial_prompt"], "config に initial_prompt があるのに空で渡している"
        assert captured["word_timestamps"] is True
        assert captured["path_or_hf_repo"] == asr.model
        assert captured.get("language") == asr.language

    def test_whisperxにinitial_promptを渡しalignまで通す(self, ctx, monkeypatch):
        captured: dict = {}

        class FakeModel:
            def transcribe(self, audio):
                return {"language": "ja", "segments": [{"start": 0.0, "end": 1.3, "text": "このチャンネルは"}]}

        def load_model(model, device, compute_type=None, language=None, asr_options=None):
            captured["model"] = model
            captured["asr_options"] = asr_options or {}
            return FakeModel()

        def align(segments, align_model, metadata, audio, device, return_char_alignments=False):
            captured["aligned"] = True
            return {"segments": WHISPERX_RAW["segments"]}

        monkeypatch.setitem(
            sys.modules,
            "whisperx",
            fake_module(
                "whisperx",
                load_model=load_model,
                load_audio=lambda path: [0.0],
                load_align_model=lambda language_code, device: ("align-model", {}),
                align=align,
            ),
        )
        prepare_audio(ctx)
        asr = with_asr(ctx, backend="whisperx")

        t = s2.run(ctx, FAKE_MEDIA)

        assert captured["asr_options"].get("initial_prompt") == asr.initial_prompt
        assert captured["aligned"] is True, "align を通さないと単語タイムスタンプが得られない"
        assert texts_of(t) == ["この", "チャンネル", "は", "AI", "の活用法を"]

    def test_バックエンドにはwork配下のaudio_wavが渡る(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        audio = prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)

        assert calls[0]["audio"] == audio


# ===========================================================================
# Step 2 — キャッシュ（SPEC 6章 Step2 / SPEC 3章）
# ===========================================================================


class TestTranscriptCache:
    """「同じ入力・同じASR設定なら丸ごとスキップ」。全工程の8割を占めるので必須。"""

    def test_一度目は文字起こしを実行し成果物を残す(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        t = s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 1
        assert s2.transcript_path(ctx).exists()
        assert s2.cache_path(ctx).exists()
        assert texts_of(t) == ["この", "チャンネル", "は", "AI", "の活用法を"]

    def test_二度目はバックエンドを呼ばない(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        first = s2.run(ctx, FAKE_MEDIA)
        second = s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 1, "同じ入力・同じASR設定なら2回目は丸ごとスキップする"
        assert second.to_dict() == first.to_dict()

    def test_force_transcribeならキャッシュを無視する(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        ctx.force_transcribe = True
        s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2

    def test_ASRモデルを変えるとキャッシュが効かない(self, ctx, monkeypatch):
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        with_asr(ctx, model="mlx-community/whisper-small-mlx")
        s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2

    @pytest.mark.parametrize(
        "field,value",
        [
            ("language", "en"),
            ("initial_prompt", "まったく別のプロンプト"),
            ("backend", "whisperx"),
            ("compute_type", "int8"),
            ("beam_size", 1),
        ],
    )
    def test_ASR設定のどれを変えてもキャッシュが効かない(self, ctx, monkeypatch, field, value):
        """model 以外の設定も結果を変えるので、すべてキーに混ざっていること。"""
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        with_asr(ctx, **{field: value})
        s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2, f"asr.{field} がキャッシュキーに入っていない"

    def test_入力ファイルの中身が変わるとキャッシュが効かない(self, ctx, monkeypatch):
        """SPEC:「入力ファイルの SHA-256 と ASR 設定のハッシュをキーにする」。

        ファイル名が同じでも録り直したら別物。ここを取り違えると
        前回のエピソードの文字起こしで今回のカット点を決めてしまう。
        """
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        ctx.input_path.write_bytes(b"\x00differently recorded episode")
        s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2

    def test_キャッシュキーは入力とASR設定の両方から作られる(self, ctx, monkeypatch):
        install_backends(monkeypatch)
        prepare_audio(ctx)
        s2.run(ctx, FAKE_MEDIA)

        entry = json.loads(s2.cache_path(ctx).read_text(encoding="utf-8"))
        assert set(entry) >= {"input_sha256", "asr_hash", "key", "transcript_file"}
        assert entry["transcript_file"] == s2.TRANSCRIPT_FILENAME
        assert len(entry["input_sha256"]) == 64

    def test_transcript_jsonが消えていたら文字起こしし直す(self, ctx, monkeypatch):
        """キャッシュだけ残って本体が無い状態で「成功」を返してはいけない。"""
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        s2.transcript_path(ctx).unlink()
        t = s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2
        assert t.words()

    def test_キャッシュが壊れていても止まらずやり直す(self, ctx, monkeypatch):
        """キャッシュは「使えたら得」なだけ。壊れていても実行は止めない。"""
        calls = install_backends(monkeypatch)
        prepare_audio(ctx)

        s2.run(ctx, FAKE_MEDIA)
        s2.cache_path(ctx).write_text("{壊れたJSON", encoding="utf-8")
        s2.run(ctx, FAKE_MEDIA)

        assert len(calls) == 2

    def test_キャッシュが指す先はwork配下に閉じる(self, ctx, monkeypatch):
        """transcript_file に外のパスが書かれていても work/ の外を読ませない。"""
        install_backends(monkeypatch)
        prepare_audio(ctx)
        s2.run(ctx, FAKE_MEDIA)

        entry = json.loads(s2.cache_path(ctx).read_text(encoding="utf-8"))
        entry["transcript_file"] = "../../../etc/passwd"
        s2.cache_path(ctx).write_text(json.dumps(entry), encoding="utf-8")

        t = s2.run(ctx, FAKE_MEDIA)
        assert t.words(), "work/ 内の transcript.json として解決されるはず"


class TestTranscriptSaveLoad:
    """--from-step で再開するための save/load。"""

    def test_保存した文字起こしをloadで読み直せる(self, ctx, monkeypatch):
        install_backends(monkeypatch)
        prepare_audio(ctx)

        saved = s2.run(ctx, FAKE_MEDIA)
        assert s2.load(ctx).to_dict() == saved.to_dict()

    def test_transcript_jsonが無ければ案内付きで止まる(self, ctx):
        ctx.ensure_dirs()
        with pytest.raises(MissingArtifactError) as exc:
            s2.load(ctx)

        message = str(exc.value)
        assert s2.TRANSCRIPT_FILENAME in message
        assert "Step 2" in message

    def test_transcript_jsonが壊れていれば止まる(self, ctx):
        ctx.ensure_dirs()
        s2.transcript_path(ctx).write_text("{これはJSONではない", encoding="utf-8")

        with pytest.raises(TranscriptionError):
            s2.load(ctx)


# ===========================================================================
# Step 1 — 音声抽出（ffmpeg を実際に叩く）
# ===========================================================================


def build_video_without_audio(path: Path, *, duration: float = 2.0) -> Path:
    """音声トラックを持たない mp4 を作る（Step 1 が止まるべき入力）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=15:duration={duration}",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"無音声動画の生成に失敗しました:\n{proc.stderr[-2000:]}")
    return path


def make_ctx(tmp_path: Path, config: Config, input_path: Path) -> RunContext:
    ctx = RunContext(
        input_path=input_path,
        episode_id="ep-test",
        work_dir=tmp_path / "work" / "ep-test",
        out_dir=tmp_path / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    return ctx


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestExtractAudio:
    """SPEC 6章 Step1: 16kHz モノラル WAV と work/probe.json を作る。"""

    def test_audio_wavとprobe_jsonが出る(self, video_ctx):
        media = s1.run(video_ctx)

        wav = s1.audio_path(video_ctx)
        probe = s1.probe_path(video_ctx)
        assert wav.exists() and wav.stat().st_size > 0
        assert probe.exists()
        assert wav.name == "audio.wav" and probe.name == "probe.json"
        assert isinstance(media, MediaInfo)

    def test_抽出した音声は16kHzモノラル16bitになる(self, video_ctx):
        """SPEC の ffmpeg コマンド（-ac 1 -ar 16000 -c:a pcm_s16le）どおりであること。

        ASR も silencedetect もこの前提で動く。
        """
        import wave

        s1.run(video_ctx)
        with wave.open(str(s1.audio_path(video_ctx)), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() / wf.getframerate() == pytest.approx(
                fixtures.EPISODE_DURATION, abs=0.3
            )

    def test_probe_jsonに総尺とfpsと解像度が入る(self, video_ctx):
        """SPEC:「ffprobe で元動画の総尺・fps・解像度も取得し、work/probe.json に保存する」。"""
        media = s1.run(video_ctx)

        assert media.duration == pytest.approx(fixtures.EPISODE_DURATION, abs=0.5)
        assert media.fps == pytest.approx(15.0, abs=0.1)
        assert (media.width, media.height) == (160, 120)
        assert media.has_audio is True
        assert media.has_video is True

        data = json.loads(s1.probe_path(video_ctx).read_text(encoding="utf-8"))
        assert set(data) >= {
            "path", "duration", "fps", "width", "height",
            "video_codec", "audio_codec", "has_video", "has_audio",
        }

    def test_probe_jsonからMediaInfoを復元できる(self, video_ctx):
        """--from-step で再開したとき、run() の返り値と同じものが読めること。"""
        media = s1.run(video_ctx)
        loaded = s1.load(video_ctx)

        assert isinstance(loaded, MediaInfo)
        assert loaded.to_dict() == media.to_dict()
        assert loaded.duration == pytest.approx(media.duration)

    def test_probe_jsonが無ければMissingArtifactErrorになる(self, ctx):
        ctx.ensure_dirs()
        with pytest.raises(MissingArtifactError) as exc:
            s1.load(ctx)

        message = str(exc.value)
        assert s1.PROBE_FILENAME in message
        assert "Step 1" in message

    def test_probe_jsonが壊れていればMissingArtifactErrorになる(self, ctx):
        ctx.ensure_dirs()
        s1.probe_path(ctx).write_text("{これはJSONではない", encoding="utf-8")

        with pytest.raises(MissingArtifactError):
            s1.load(ctx)

    def test_probe_jsonがオブジェクトでなければ止まる(self, ctx):
        ctx.ensure_dirs()
        s1.probe_path(ctx).write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(MissingArtifactError):
            s1.load(ctx)

    def test_音声トラックが無い動画は止まる(self, tmp_path, config):
        """この先の文字起こしも無音検出も成立しないので、ここで止めるのが正しい。"""
        video = build_video_without_audio(tmp_path / "silent.mp4")
        ctx = make_ctx(tmp_path, config, video)

        with pytest.raises(FfmpegError) as exc:
            s1.run(ctx)

        assert "音声" in str(exc.value)
        assert not s1.audio_path(ctx).exists()

    def test_映像トラックが無い入力は警告を残して続行する(self, tmp_path, config, episode_wav):
        """音声だけでもカット点までは求められる。ただし Step 7 の書き出しに響くので警告する。"""
        ctx = make_ctx(tmp_path, config, episode_wav)

        media = s1.run(ctx)

        assert media.has_audio is True
        assert media.has_video is False
        assert ctx.warnings, "映像が無いことを decisions.json に残すべき"
        assert s1.audio_path(ctx).exists()

    def test_ffprobeが読めない入力は握りつぶさず止まる(self, ctx):
        """SPEC 9章:「ffmpeg が非ゼロ終了 → stderr をそのまま表示して停止」。"""
        with pytest.raises(FfmpegError) as exc:
            s1.run(ctx)  # ctx の入力は中身が空のファイル

        assert exc.value.stderr or "ffprobe" in str(exc.value)

    def test_入力が無ければ止まる(self, tmp_path, config):
        ctx = make_ctx(tmp_path, config, tmp_path / "ないファイル.mp4")

        with pytest.raises(RadioCutterError) as exc:
            s1.run(ctx)
        assert "見つかりません" in str(exc.value)

    def test_入力がディレクトリなら止まる(self, tmp_path, config):
        directory = tmp_path / "これはフォルダ"
        directory.mkdir()
        ctx = make_ctx(tmp_path, config, directory)

        with pytest.raises(RadioCutterError):
            s1.run(ctx)

    def test_抽出済みの音声は作り直さない(self, video_ctx):
        """60分の抽出をやり直さないための既存ファイル再利用（モジュールの契約）。"""
        s1.run(video_ctx)
        wav = s1.audio_path(video_ctx)
        marker = b"MARKER" + b"\x00" * 64
        wav.write_bytes(marker)

        s1.run(video_ctx)

        assert wav.read_bytes() == marker, "入力より新しい audio.wav は再抽出しない"

    def test_入力が更新されていれば抽出し直す(self, video_ctx, episode_video):
        import os
        import shutil
        import time

        s1.run(video_ctx)
        wav = s1.audio_path(video_ctx)
        wav.write_bytes(b"stale")

        # 入力を audio.wav より新しくする
        shutil.copyfile(episode_video, video_ctx.input_path)
        now = time.time() + 10
        os.utime(video_ctx.input_path, (now, now))

        s1.run(video_ctx)

        assert wav.read_bytes() != b"stale"
        assert wav.stat().st_size > 1000


@pytest.mark.ffmpeg
@requires_ffmpeg
class TestExtractAudioThenTranscribe:
    """Step 1 → Step 2 のつなぎ目。Step 2 は Step 1 が置いた audio.wav を読む。"""

    def test_step1の出力をstep2がそのまま読む(self, video_ctx, monkeypatch):
        calls = install_backends(monkeypatch)

        media = s1.run(video_ctx)
        transcript = s2.run(video_ctx, media)

        assert calls[0]["audio"] == s1.audio_path(video_ctx)
        assert transcript.duration == pytest.approx(media.duration)

    def test_probe_jsonがあれば総尺はそこから決まる(self, video_ctx, monkeypatch):
        """media を渡さずに再開した場合でも duration がゼロにならないこと。"""
        install_backends(monkeypatch)
        s1.run(video_ctx)

        transcript = s2.run(video_ctx)

        assert transcript.duration == pytest.approx(fixtures.EPISODE_DURATION, abs=0.5)
