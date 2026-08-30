"""config.py — 設定ファイルの読み込みと検証。

このファイルが守らせたいこと（SPEC 5章・9章・11章）:

- 同梱の `config/ai-radio.json` が SPEC 5章の例そのままの値として読めること。
  アンカー語はコードにハードコードしない、という方針の土台がここ。
- 設定の誤りは **必ず ConfigError** で止まること。パイプラインの奥（Step 3 のアンカー
  未検出、Step 7 の ffmpeg 失敗）まで持ち越すと、原因の切り分けが極端に難しくなる。
  「JSON を読んで型の付いた値にし、おかしければ ConfigError で止める」のがこの層の全責務。
- エラーメッセージは、どの項目が悪いのかを名指しすること（SPEC 9章「握りつぶさない」）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from radio_cutter.config import (
    DEFAULT_SILENCE_MIN_DUR,
    DEFAULT_SILENCE_NOISE_DB,
    AnchorConfig,
    AsrConfig,
    Config,
    MustFollow,
    SegmentConfig,
    load_config,
)
from radio_cutter.errors import ConfigError, RadioCutterError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "ai-radio.json"


# ---------------------------------------------------------------------------
# ヘルパ
# ---------------------------------------------------------------------------


def base_dict() -> dict[str, Any]:
    """同梱 config の生 JSON を毎回まっさらな dict で返す（テスト間で汚さない）。"""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def build(mutate=None) -> Config:
    """同梱 config を土台に、一箇所だけ壊した設定を読み込む。"""
    d = base_dict()
    if mutate is not None:
        mutate(d)
    return Config.from_dict(d)


def expect_error(mutate) -> ConfigError:
    """壊した設定が ConfigError になることを確かめ、その例外を返す。"""
    with pytest.raises(ConfigError) as excinfo:
        build(mutate)
    return excinfo.value


def write_config(tmp_path: Path, data: Any, name: str = "cfg.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 同梱 config が SPEC 5章の例そのままに読めること
# ---------------------------------------------------------------------------


class TestBundledConfig:
    """`config/ai-radio.json` は SPEC 5章の例と1対1で対応していなければならない。

    ここが崩れると「アンカー語をコードにハードコードしない」という前提ごと崩れる。
    """

    def test_チャンネル名(self, config: Config) -> None:
        assert config.channel == "AI活用法実験ラジオ"

    def test_アンカーは_A_と_B_の2つ(self, config: Config) -> None:
        assert config.anchor_ids() == ["A", "B"]

    def test_アンカーA(self, config: Config) -> None:
        a = config.anchor("A")
        assert a.phrase == "このチャンネルは"
        assert a.occurrence == "first"
        assert a.search_window_sec == (0.0, 600.0)
        assert a.cut == "before"
        assert a.fuzzy_threshold == pytest.approx(0.82)
        assert a.must_follow is None
        assert a.nth is None

    def test_アンカーB(self, config: Config) -> None:
        b = config.anchor("B")
        assert b.phrase == "ということで"
        assert b.occurrence == "last"
        assert b.cut == "before"
        assert b.fuzzy_threshold == pytest.approx(0.85)
        assert b.search_window_sec is None
        assert b.must_follow == MustFollow(phrase="木原", within_sec=4.0, fuzzy_threshold=None)

    def test_しきい値は_rapidfuzz_の0_100尺度に直せる(self, config: Config) -> None:
        """fuzzy_threshold は 0〜1 で書く。Step 3 は 0〜100 のスコアで比べる。"""
        assert config.anchor("A").threshold_score == pytest.approx(82.0)
        assert config.anchor("B").threshold_score == pytest.approx(85.0)

    def test_セグメント(self, config: Config) -> None:
        assert [s.name for s in config.segments] == ["main", "ending"]
        main = config.segment("main")
        assert (main.file, main.from_, main.to) == ("02_main.mp4", "A", "B")
        ending = config.segment("ending")
        assert (ending.file, ending.from_, ending.to) == ("03_ending.mp4", "B", "end")

    def test_ハイライト(self, config: Config) -> None:
        h = config.highlight
        assert h.file == "01_highlight.mp4"
        assert h.source_segment == "main"
        assert h.target_duration_sec == pytest.approx(30.0)
        assert h.min_duration_sec == pytest.approx(20.0)
        assert h.max_duration_sec == pytest.approx(45.0)
        assert h.position == "prepend"
        assert h.allow_multi_cut is False

    def test_ハイライト候補は3つ(self, config: Config) -> None:
        """SPEC Step 5「LLMには候補を3つ返させ、スコア最上位を採用する」。"""
        assert config.highlight.num_candidates == 3

    def test_ASR設定(self, config: Config) -> None:
        assert config.asr.model == "mlx-community/whisper-large-v3-mlx"
        assert config.asr.language == "ja"
        # initial_prompt はアンカー語をモデルにバイアスさせるためのもの（SPEC Step 2）。
        assert config.asr.initial_prompt == "AI活用法実験ラジオ。このチャンネルは。ということで、木原さん。"
        for phrase in ("このチャンネルは", "ということで", "木原"):
            assert phrase in config.asr.initial_prompt

    def test_LLM設定(self, config: Config) -> None:
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-sonnet-4-6"
        assert config.llm.max_retries == 3

    def test_レンダリング設定(self, config: Config) -> None:
        r = config.render
        assert r.video_codec == "h264_videotoolbox"
        assert r.video_bitrate == "12M"
        assert r.audio_codec == "aac"
        assert r.audio_bitrate == "192k"
        # VideoToolbox が無い環境向けのフォールバック（SPEC Step 7）。
        assert r.fallback_video_codec == "libx264"

    def test_YouTube設定(self, config: Config) -> None:
        assert config.youtube.channel_links == ()
        assert config.youtube.fixed_footer == ""
        assert config.youtube.hashtags == ("#AI", "#AI活用", "#生成AI")

    def test_無音検出の既定値はSPEC_Step4のとおり(self, config: Config) -> None:
        """同梱 config に silence 節は無い。既定は -32dB / 0.12 秒。"""
        assert config.silence.noise_db == pytest.approx(-32.0)
        assert config.silence.min_duration_sec == pytest.approx(0.12)
        assert DEFAULT_SILENCE_NOISE_DB == pytest.approx(-32.0)
        assert DEFAULT_SILENCE_MIN_DUR == pytest.approx(0.12)

    def test_読み込み元のパスと生JSONを保持する(self) -> None:
        """デバッグの起点になるので、どのファイルを読んだかを持ち続ける。"""
        cfg = load_config(CONFIG_PATH)
        assert cfg.path == CONFIG_PATH
        assert cfg.raw["channel"] == "AI活用法実験ラジオ"

    def test_文字列パスでも読める(self) -> None:
        assert load_config(str(CONFIG_PATH)).channel == "AI活用法実験ラジオ"

    def test_アンカーは書き換えられない(self, config: Config) -> None:
        """設定は読み込み後に不変。途中のステップが黙って書き換えないための保険。"""
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.anchor("A").phrase = "べつのフレーズ"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# occurrence
# ---------------------------------------------------------------------------


class TestOccurrence:
    """SPEC 5章「`occurrence` は `first` / `last` / `nth` を受け付ける」。"""

    @pytest.mark.parametrize("value", ["first", "last"])
    def test_firstとlastは追加項目なしで通る(self, value: str) -> None:
        cfg = build(lambda d: d["anchors"][0].update(occurrence=value))
        assert cfg.anchor("A").occurrence == value

    def test_nthは順位とセットで通る(self) -> None:
        cfg = build(lambda d: d["anchors"][0].update(occurrence="nth", nth=2))
        a = cfg.anchor("A")
        assert (a.occurrence, a.nth) == ("nth", 2)

    def test_nth未指定はエラー(self) -> None:
        """`nth` を選んだのに順位が無ければ、何番目を採るか決められない。"""
        exc = expect_error(lambda d: d["anchors"][0].update(occurrence="nth"))
        assert "nth" in str(exc)

    def test_nthがnullでもエラー(self) -> None:
        expect_error(lambda d: d["anchors"][0].update(occurrence="nth", nth=None))

    @pytest.mark.parametrize("bad", [0, -1])
    def test_nthは1始まり(self, bad: int) -> None:
        expect_error(lambda d: d["anchors"][0].update(occurrence="nth", nth=bad))

    def test_省略時はfirst(self) -> None:
        cfg = build(lambda d: d["anchors"][0].pop("occurrence"))
        assert cfg.anchor("A").occurrence == "first"

    @pytest.mark.parametrize("bad", ["second", "FIRST", "all", "", 1])
    def test_未知のoccurrenceはエラー(self, bad: Any) -> None:
        exc = expect_error(lambda d: d["anchors"][0].update(occurrence=bad))
        assert "occurrence" in str(exc)

    def test_first_last_のときnthは効かない(self) -> None:
        """`nth` は occurrence='nth' のときだけ意味を持つ。"""
        cfg = build(lambda d: d["anchors"][0].update(occurrence="first", nth=3))
        assert cfg.anchor("A").nth is None


# ---------------------------------------------------------------------------
# must_follow
# ---------------------------------------------------------------------------


class TestMustFollow:
    """SPEC 5章「`must_follow` は指定フレーズが `within_sec` 以内に続く候補だけを残すフィルタ」。

    これが正しくパースされないと、アンカーBのおとり（本編途中の「ということで」）を
    落とせず、本編とエンディングの境目が丸ごとずれる。
    """

    def test_フレーズと秒数をパースする(self, config: Config) -> None:
        mf = config.anchor("B").must_follow
        assert mf is not None
        assert mf.phrase == "木原"
        assert mf.within_sec == pytest.approx(4.0)

    def test_独自のしきい値を持てる(self) -> None:
        cfg = build(lambda d: d["anchors"][1]["must_follow"].update(fuzzy_threshold=0.7))
        mf = cfg.anchor("B").must_follow
        assert mf is not None
        assert mf.fuzzy_threshold == pytest.approx(0.7)

    def test_しきい値未指定はNone_アンカー本体のしきい値に委ねる(self, config: Config) -> None:
        mf = config.anchor("B").must_follow
        assert mf is not None
        assert mf.fuzzy_threshold is None

    def test_省略できる(self, config: Config) -> None:
        assert config.anchor("A").must_follow is None

    def test_phrase欠落はエラー(self) -> None:
        exc = expect_error(lambda d: d["anchors"][1]["must_follow"].pop("phrase"))
        assert "phrase" in str(exc)

    def test_within_sec欠落はエラー(self) -> None:
        exc = expect_error(lambda d: d["anchors"][1]["must_follow"].pop("within_sec"))
        assert "within_sec" in str(exc)

    @pytest.mark.parametrize("bad", ["木原", ["木原", 4.0], 4.0])
    def test_オブジェクト以外はエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["anchors"][1].update(must_follow=bad))

    @pytest.mark.parametrize("bad", ["soon", None, "4秒"])
    def test_within_secが数値でなければエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["anchors"][1]["must_follow"].update(within_sec=bad))

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_空のフレーズはエラー(self, bad: str) -> None:
        """空フレーズはどの候補にも一致せず、全候補が黙って落ちて
        「アンカー候補が0件」という無関係なエラーに化ける。設定段階で止める。"""
        expect_error(lambda d: d["anchors"][1]["must_follow"].update(phrase=bad))

    @pytest.mark.parametrize("bad", [-4.0, -0.1])
    def test_within_secが負ならエラー(self, bad: float) -> None:
        """負の「以内」は満たしようがない。全候補が落ちるだけなので設定段階で止める。"""
        expect_error(lambda d: d["anchors"][1]["must_follow"].update(within_sec=bad))


# ---------------------------------------------------------------------------
# search_window_sec
# ---------------------------------------------------------------------------


class TestSearchWindow:
    """SPEC Step 3-4「`search_window_sec` の範囲外の候補を除外」。

    形式が壊れていると窓の意味が変わり、除外すべき候補を通してしまう。
    """

    def test_2要素の配列をfloatのタプルにする(self, config: Config) -> None:
        window = config.anchor("A").search_window_sec
        assert window == (0.0, 600.0)
        assert all(isinstance(x, float) for x in window)

    def test_省略時はNone_窓なし(self, config: Config) -> None:
        assert config.anchor("B").search_window_sec is None

    @pytest.mark.parametrize(
        "bad",
        [
            600,                        # スカラー
            [600],                      # 1要素
            [0, 300, 600],              # 3要素
            {"from": 0, "to": 600},     # オブジェクト
            "0,600",                    # 文字列
            [],
        ],
        ids=["scalar", "one", "three", "object", "string", "empty"],
    )
    def test_2要素の配列でなければエラー(self, bad: Any) -> None:
        exc = expect_error(lambda d: d["anchors"][0].update(search_window_sec=bad))
        assert "search_window_sec" in str(exc)

    @pytest.mark.parametrize("bad", [["a", "b"], [0, None], [None, 600]])
    def test_数値でない要素はエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["anchors"][0].update(search_window_sec=bad))

    @pytest.mark.parametrize("bad", [[600, 0], [10, 10], [5, 4.9]])
    def test_開始が終了以上ならエラー(self, bad: Any) -> None:
        """開始 >= 終了 の窓は何も通さない。設定ミスとして止める。"""
        expect_error(lambda d: d["anchors"][0].update(search_window_sec=bad))


# ---------------------------------------------------------------------------
# fuzzy_threshold / cut
# ---------------------------------------------------------------------------


class TestFuzzyThreshold:
    """しきい値は 0 < x <= 1。0〜100 のスコアと混同した設定を素通りさせない。"""

    @pytest.mark.parametrize("ok", [0.5, 0.82, 1.0, 0.999])
    def test_0より大きく1以下なら通る(self, ok: float) -> None:
        cfg = build(lambda d: d["anchors"][0].update(fuzzy_threshold=ok))
        assert cfg.anchor("A").fuzzy_threshold == pytest.approx(ok)

    @pytest.mark.parametrize("bad", [0, 0.0, -0.5, 1.01, 82, 100])
    def test_範囲外はエラー(self, bad: float) -> None:
        """82（0〜100 の尺度で書いてしまった）も弾く。素通りすると全候補が落ちる。"""
        exc = expect_error(lambda d: d["anchors"][0].update(fuzzy_threshold=bad))
        assert "fuzzy_threshold" in str(exc)

    @pytest.mark.parametrize("bad", ["high", None, [0.8]])
    def test_数値でなければエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["anchors"][0].update(fuzzy_threshold=bad))

    def test_省略時の既定は0_85(self) -> None:
        cfg = build(lambda d: d["anchors"][0].pop("fuzzy_threshold"))
        assert cfg.anchor("A").fuzzy_threshold == pytest.approx(0.85)


class TestCut:
    """`cut` は before / after のみ。ここが不正だとカット点が逆側に付く。"""

    @pytest.mark.parametrize("ok", ["before", "after"])
    def test_beforeとafterは通る(self, ok: str) -> None:
        cfg = build(lambda d: d["anchors"][0].update(cut=ok))
        assert cfg.anchor("A").cut == ok

    @pytest.mark.parametrize("bad", ["middle", "BEFORE", "", "front", 0])
    def test_それ以外はエラー(self, bad: Any) -> None:
        exc = expect_error(lambda d: d["anchors"][0].update(cut=bad))
        assert "cut" in str(exc)

    def test_省略時はbefore(self) -> None:
        cfg = build(lambda d: d["anchors"][0].pop("cut"))
        assert cfg.anchor("A").cut == "before"


# ---------------------------------------------------------------------------
# anchors 全体
# ---------------------------------------------------------------------------


class TestAnchorsSection:
    def test_id重複はエラー(self) -> None:
        """ID が重なると `anchor()` がどちらを返すか決まらず、区間が壊れる。"""
        exc = expect_error(lambda d: d["anchors"][1].update(id="A"))
        assert "A" in str(exc)

    def test_phrase欠落はエラー(self) -> None:
        expect_error(lambda d: d["anchors"][0].pop("phrase"))

    @pytest.mark.parametrize("bad", ["", "   ", "\n"])
    def test_空のphraseはエラー(self, bad: str) -> None:
        """空フレーズは全文に一致してしまう。コードにハードコードした値で補わない。"""
        expect_error(lambda d: d["anchors"][0].update(phrase=bad))

    def test_id欠落はエラー(self) -> None:
        expect_error(lambda d: d["anchors"][0].pop("id"))

    def test_anchors節が無ければエラー(self) -> None:
        expect_error(lambda d: d.pop("anchors"))

    def test_anchorsが空配列ならエラー(self) -> None:
        expect_error(lambda d: d.update(anchors=[]))

    @pytest.mark.parametrize("bad", [{"A": {}}, "A", None])
    def test_anchorsが配列でなければエラー(self, bad: Any) -> None:
        expect_error(lambda d: d.update(anchors=bad))

    def test_要素がオブジェクトでなければエラー(self) -> None:
        expect_error(lambda d: d["anchors"].append("C"))


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------


class TestSegments:
    """segments の from/to はアンカーID か start / end しか指さない。

    未知の参照を通すと Step 7 の直前まで気付けない。
    """

    @pytest.mark.parametrize("key", ["name", "file", "from", "to"])
    def test_必須項目の欠落はエラー(self, key: str) -> None:
        exc = expect_error(lambda d: d["segments"][0].pop(key))
        assert key in str(exc)

    def test_名前の重複はエラー(self) -> None:
        exc = expect_error(lambda d: d["segments"][1].update(name="main"))
        assert "main" in str(exc)

    def test_出力ファイル名の重複はエラー(self) -> None:
        """同じファイル名だと後から書いた方が前のを黙って上書きする。"""
        exc = expect_error(lambda d: d["segments"][1].update(file="02_main.mp4"))
        assert "02_main.mp4" in str(exc)

    @pytest.mark.parametrize("bad", ["Z", "A2", "begin", "finish", "main", ""])
    def test_fromが未知の参照ならエラー(self, bad: str) -> None:
        exc = expect_error(lambda d: d["segments"][0].update({"from": bad}))
        assert "from" in str(exc)

    @pytest.mark.parametrize("bad", ["Z", "終端", "END", "last"])
    def test_toが未知の参照ならエラー(self, bad: str) -> None:
        exc = expect_error(lambda d: d["segments"][1].update(to=bad))
        assert "to" in str(exc)

    def test_startとendは予約語として使える(self) -> None:
        def mutate(d: dict[str, Any]) -> None:
            d["segments"][0]["from"] = "start"
            d["segments"][1]["to"] = "end"

        cfg = build(mutate)
        assert cfg.segment("main").from_ == "start"
        assert cfg.segment("ending").to == "end"

    def test_アンカーIDを参照できる(self, config: Config) -> None:
        for seg in config.segments:
            for ref in (seg.from_, seg.to):
                assert ref in set(config.anchor_ids()) | {"start", "end"}

    def test_segments節が無ければエラー(self) -> None:
        expect_error(lambda d: d.pop("segments"))

    def test_segmentsが空配列ならエラー(self) -> None:
        expect_error(lambda d: d.update(segments=[]))

    def test_要素がオブジェクトでなければエラー(self) -> None:
        expect_error(lambda d: d["segments"].append("main"))


# ---------------------------------------------------------------------------
# highlight
# ---------------------------------------------------------------------------


class TestHighlight:
    def test_source_segmentがsegmentsに無ければエラー(self) -> None:
        """存在しない区間からハイライトは切り出せない。"""
        exc = expect_error(lambda d: d["highlight"].update(source_segment="ending2"))
        assert "source_segment" in str(exc)

    def test_endingからも切り出せる(self) -> None:
        cfg = build(lambda d: d["highlight"].update(source_segment="ending"))
        assert cfg.highlight.source_segment == "ending"

    @pytest.mark.parametrize(
        "patch",
        [
            {"min_duration_sec": 35},                        # min > target
            {"max_duration_sec": 25},                        # target > max
            {"min_duration_sec": 50, "max_duration_sec": 10},  # min > max
            {"target_duration_sec": 5},                      # target < min
            {"target_duration_sec": 60},                     # target > max
        ],
        ids=["min>target", "target>max", "min>max", "target<min", "target>max2"],
    )
    def test_min_target_maxの大小関係違反はエラー(self, patch: dict[str, Any]) -> None:
        expect_error(lambda d: d["highlight"].update(patch))

    def test_境界は許す(self) -> None:
        """min == target == max（尺を固定したい）は正当な設定。"""
        cfg = build(
            lambda d: d["highlight"].update(
                min_duration_sec=30, target_duration_sec=30, max_duration_sec=30
            )
        )
        assert cfg.highlight.target_duration_sec == pytest.approx(30.0)

    @pytest.mark.parametrize("bad", ["middle", "PREPEND", "", "front"])
    def test_positionはprepend_appendのみ(self, bad: str) -> None:
        exc = expect_error(lambda d: d["highlight"].update(position=bad))
        assert "position" in str(exc)

    def test_appendも使える(self) -> None:
        cfg = build(lambda d: d["highlight"].update(position="append"))
        assert cfg.highlight.position == "append"

    @pytest.mark.parametrize("bad", ["ten", None, [30]])
    def test_尺が数値でなければエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["highlight"].update(target_duration_sec=bad))

    def test_出力ファイル名がセグメントと衝突したらエラー(self) -> None:
        """ハイライトとセグメントが同じファイル名だと、Step 7 で片方が黙って消える。"""
        expect_error(lambda d: d["highlight"].update(file="02_main.mp4"))


# ---------------------------------------------------------------------------
# llm / asr / render
# ---------------------------------------------------------------------------


class TestLlmConfig:
    """SPEC 9章「LLMがJSON以外を返す → 3回までリトライ」。リトライ回数は1以上。"""

    @pytest.mark.parametrize("bad", [0, -1, -3])
    def test_max_retriesが0以下ならエラー(self, bad: int) -> None:
        exc = expect_error(lambda d: d["llm"].update(max_retries=bad))
        assert "max_retries" in str(exc)

    @pytest.mark.parametrize("ok", [1, 3, 5])
    def test_1以上なら通る(self, ok: int) -> None:
        cfg = build(lambda d: d["llm"].update(max_retries=ok))
        assert cfg.llm.max_retries == ok

    def test_llm節が無くても既定値で読める(self) -> None:
        cfg = build(lambda d: d.pop("llm"))
        assert cfg.llm.max_retries == 3
        assert cfg.llm.provider == "anthropic"

    def test_APIキーは環境変数名で持つ(self, config: Config) -> None:
        """SPEC 2章「LLM APIキー（環境変数）」。キー本体を config に書かせない。"""
        assert config.llm.api_key_env == "ANTHROPIC_API_KEY"
        assert "api_key" not in config.raw.get("llm", {})


class TestRenderConfig:
    def test_フォールバックはlibx264_veryfast_crf20(self, config: Config) -> None:
        """SPEC Step 7「h264_videotoolbox が使えない環境では libx264 -preset veryfast -crf 20」。"""
        assert config.render.fallback_video_codec == "libx264"
        assert config.render.fallback_extra_args == ("-preset", "veryfast", "-crf", "20")

    def test_尺の検算許容差は0_5秒(self, config: Config) -> None:
        """SPEC Step 7「Dh + Dm + De との差が0.5秒を超えたら警告」。"""
        assert config.render.duration_tolerance_sec == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", ["-preset veryfast", 20, {"preset": "veryfast"}])
    def test_fallback_extra_argsが配列でなければエラー(self, bad: Any) -> None:
        expect_error(lambda d: d["render"].update(fallback_extra_args=bad))


# ---------------------------------------------------------------------------
# 数値項目は必ず ConfigError に落とす
# ---------------------------------------------------------------------------


class TestNumericFieldsRejectNonNumeric:
    """数値を書くべき場所に数値以外が来たら ConfigError。

    ConfigError は「設定ファイルが読めない・スキーマに合わない」ための例外で、
    CLI はこれを捕まえて終了コード1で止まる（errors.py）。素の ValueError が
    漏れると、設定ミスがトレースバックとして出て「どの項目が悪いか」が伝わらない。
    """

    @pytest.mark.parametrize(
        "section,key",
        [
            ("anchors", "nth"),
            ("llm", "max_retries"),
            ("llm", "max_tokens"),
            ("llm", "temperature"),
            ("asr", "beam_size"),
            ("highlight", "num_candidates"),
            ("render", "duration_tolerance_sec"),
        ],
    )
    def test_数値でなければConfigError(self, section: str, key: str) -> None:
        def mutate(d: dict[str, Any]) -> None:
            if section == "anchors":
                d["anchors"][0]["occurrence"] = "nth"
                d["anchors"][0][key] = "たくさん"
            else:
                d[section][key] = "たくさん"

        exc = expect_error(mutate)
        assert key in str(exc)


# ---------------------------------------------------------------------------
# エラーメッセージの質
# ---------------------------------------------------------------------------


class TestErrorMessagesNameTheField:
    """SPEC 9章。どの項目が悪いのかを名指しできないエラーは直しようがない。"""

    @pytest.mark.parametrize(
        "mutate,needle",
        [
            (lambda d: d["anchors"][0].update(occurrence="second"), "occurrence"),
            (lambda d: d["anchors"][0].update(cut="middle"), "cut"),
            (lambda d: d["anchors"][0].update(fuzzy_threshold=2.0), "fuzzy_threshold"),
            (lambda d: d["anchors"][0].update(search_window_sec=[600, 0]), "search_window_sec"),
            (lambda d: d["highlight"].update(source_segment="nope"), "source_segment"),
            (lambda d: d["llm"].update(max_retries=0), "max_retries"),
        ],
        ids=["occurrence", "cut", "fuzzy", "window", "source_segment", "retries"],
    )
    def test_項目名がメッセージに出る(self, mutate, needle: str) -> None:
        assert needle in str(expect_error(mutate))

    def test_アンカーのエラーはどのアンカーかを示す(self) -> None:
        """アンカーが増えたとき、どれが悪いのか分からないと直せない。"""
        assert "B" in str(expect_error(lambda d: d["anchors"][1].update(cut="middle")))


# ---------------------------------------------------------------------------
# load_config（ファイル入出力）
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_存在しないパスはConfigError(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as exc:
            load_config(tmp_path / "no-such-config.json")
        assert "no-such-config.json" in str(exc.value)

    def test_壊れたJSONはConfigError(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.json"
        p.write_text('{ "channel": "x", "anchors": [ ', encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)

    def test_空ファイルはConfigError(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)

    @pytest.mark.parametrize("payload", [[], "hello", 42, None], ids=["list", "str", "int", "null"])
    def test_トップレベルがオブジェクトでなければConfigError(self, tmp_path: Path, payload: Any) -> None:
        with pytest.raises(ConfigError):
            load_config(write_config(tmp_path, payload))

    def test_ディレクトリを渡してもConfigError(self, tmp_path: Path) -> None:
        """`--config config/` のような取り違えでトレースバックを出さない。"""
        d = tmp_path / "config"
        d.mkdir()
        with pytest.raises(ConfigError):
            load_config(d)

    def test_ConfigErrorはRadioCutterErrorの一種(self) -> None:
        """CLI は RadioCutterError を捕まえて終了コード1で止まる。"""
        assert issubclass(ConfigError, RadioCutterError)

    def test_書き出して読み直しても同じ設定になる(self, tmp_path: Path, config: Config) -> None:
        again = load_config(write_config(tmp_path, base_dict()))
        assert again.anchors == config.anchors
        assert again.segments == config.segments
        assert again.highlight == config.highlight
        assert again.asr == config.asr


# ---------------------------------------------------------------------------
# 参照ヘルパ
# ---------------------------------------------------------------------------


class TestLookups:
    def test_anchorは該当するアンカーを返す(self, config: Config) -> None:
        assert config.anchor("B").phrase == "ということで"

    def test_未知のアンカーIDはConfigError(self, config: Config) -> None:
        with pytest.raises(ConfigError) as exc:
            config.anchor("Z")
        assert "Z" in str(exc.value)

    @pytest.mark.parametrize("bad", ["a", "", " A", "end"])
    def test_アンカーIDは完全一致(self, config: Config, bad: str) -> None:
        with pytest.raises(ConfigError):
            config.anchor(bad)

    def test_segmentは該当するセグメントを返す(self, config: Config) -> None:
        assert config.segment("ending").file == "03_ending.mp4"

    def test_未知のセグメント名はConfigError(self, config: Config) -> None:
        with pytest.raises(ConfigError) as exc:
            config.segment("intro")
        assert "intro" in str(exc.value)

    def test_anchor_idsは定義順(self, config: Config) -> None:
        """並び順は Step 3 のエラーメッセージや区間の解決に効く。"""
        assert config.anchor_ids() == ["A", "B"]


# ---------------------------------------------------------------------------
# ASR キャッシュキー
# ---------------------------------------------------------------------------


class TestAsrCacheKeyPayload:
    """SPEC Step 2「入力ファイルのSHA-256とASR設定のハッシュをキーにして、
    同一ならこのステップを丸ごとスキップする」。

    ペイロードから漏れた項目は、変えてもキャッシュが当たってしまう。
    古い設定の文字起こしを使い続け、しかもそれに気付けない。
    """

    def test_AsrConfigの全項目を含む(self) -> None:
        fields = {f.name for f in dataclasses.fields(AsrConfig)}
        assert set(AsrConfig().cache_key_payload()) == fields

    def test_initial_promptを含む(self, config: Config) -> None:
        """initial_prompt を変えると文字起こしが変わる。キャッシュキーに必須。"""
        payload = config.asr.cache_key_payload()
        assert payload["initial_prompt"] == config.asr.initial_prompt
        assert payload["model"] == config.asr.model
        assert payload["language"] == config.asr.language

    @pytest.mark.parametrize(
        "field,changed",
        [
            ("model", "mlx-community/whisper-small"),
            ("language", "en"),
            ("initial_prompt", "べつのプロンプト"),
            ("backend", "whisperx"),
            ("compute_type", "int8"),
            ("beam_size", 9),
        ],
    )
    def test_どの項目を変えてもペイロードが変わる(self, field: str, changed: Any) -> None:
        base = AsrConfig()
        other = dataclasses.replace(base, **{field: changed})
        assert base.cache_key_payload() != other.cache_key_payload()

    def test_同じ設定なら同じペイロード(self) -> None:
        assert AsrConfig.from_dict(base_dict()["asr"]).cache_key_payload() == AsrConfig.from_dict(
            base_dict()["asr"]
        ).cache_key_payload()

    def test_JSONにできる(self, config: Config) -> None:
        """キャッシュキーは JSON 経由でハッシュする（util/cache.py）。"""
        dumped = json.dumps(config.asr.cache_key_payload(), sort_keys=True, ensure_ascii=False)
        assert json.loads(dumped) == config.asr.cache_key_payload()

    def test_ASR以外の設定は混ざらない(self, config: Config) -> None:
        """レンダリング設定やLLM設定を変えても文字起こしは変わらない。
        混ぜるとキャッシュが無駄に外れ、一番重い工程を毎回やり直すことになる。"""
        payload = config.asr.cache_key_payload()
        assert "video_codec" not in payload
        assert "provider" not in payload


# ---------------------------------------------------------------------------
# from_dict は入力を壊さない
# ---------------------------------------------------------------------------


class TestFromDictPurity:
    def test_渡したdictを書き換えない(self) -> None:
        """raw を保持する都合で同じ dict を参照するが、読み込みで中身を変えないこと。"""
        d = base_dict()
        before = copy.deepcopy(d)
        Config.from_dict(d)
        assert d == before

    def test_型付きの値になっている(self) -> None:
        cfg = Config.from_dict(base_dict())
        assert isinstance(cfg.anchors[0], AnchorConfig)
        assert isinstance(cfg.segments[0], SegmentConfig)
        assert isinstance(cfg.asr, AsrConfig)
