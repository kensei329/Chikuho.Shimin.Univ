"""steps/s7_render.py と s8_preview.py — 書き出しと確認用プレビュー。

ここは「実際に ffmpeg を回して出てきた mp4 を測る」テスト。
SPEC Step 7・Step 8 と、Phase 1/2 の受け入れ基準（SPEC 10章）を根拠にしている。

守らせたいこと:
- 3本＋final が出て、実尺が指定した区間と合っていること（尺がずれる＝カット位置がずれている）
- `02_main.mp4` がカット点Aちょうどから始まっていて語頭が欠けていないこと（Phase 1 の要）
- `final.mp4` が3本の合計と一致すること（Phase 2 の要）
- 検算がちゃんと効くこと。ずれたら**警告**を出す。**例外は投げない**（書き出し自体は成功しているため）
- プレビューはカット点の前後2秒＝計4秒。動画の端でも、はみ出さずクランプされること
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import fixtures
from conftest import requires_ffmpeg
from radio_cutter.config import Config, SilenceConfig
from radio_cutter.context import RunContext
from radio_cutter.errors import MissingArtifactError, RenderError
from radio_cutter.models import CutPoint, HighlightCandidate, HighlightResult
from radio_cutter.steps import s7_render as s7
from radio_cutter.steps import s8_preview as s8
from radio_cutter.util.ffmpeg import (
    MediaInfo,
    detect_silences,
    extract_audio,
    media_duration,
    probe_media,
)

pytestmark = pytest.mark.ffmpeg

# ---------------------------------------------------------------------------
# 合成エピソードから引いてくる期待値（tests/fixtures.py と同じ時刻表を共有する）
# ---------------------------------------------------------------------------

CUT_A = fixtures.EXPECTED_CUT_A          # 5.90
CUT_B = fixtures.EXPECTED_CUT_B          # 43.90
EPISODE_DURATION = fixtures.EPISODE_DURATION

#: ハイライトに使う発話「実はAIに議事録を書かせるのは…」。前後がちょうど無音の谷。
HL_START, HL_END = next(
    (start, end) for start, end, text in fixtures.UTTERANCES if text.startswith("実は")
)

#: 各セグメントの実尺が指定区間と一致すべき許容（再エンコードのフレーム丸め分だけ見込む）
SEGMENT_TOLERANCE = 0.2

#: 無音検出のしきい値。fixtures の無音区間は完全な 0 なので -32dB で確実に拾える。
SILENCE_DB = -32.0


def _cuts() -> dict[str, CutPoint]:
    """Step 4 が返すはずのカット点（合成エピソードの正解値）。"""
    return {
        "A": CutPoint(
            anchor_id="A",
            raw_cut_time=fixtures.EXPECTED_ANCHOR_A_RAW,
            cut_time=CUT_A,
            silence_found=True,
            score=100.0,
        ),
        "B": CutPoint(
            anchor_id="B",
            raw_cut_time=fixtures.EXPECTED_ANCHOR_B_RAW,
            cut_time=CUT_B,
            silence_found=True,
            score=96.2,
        ),
    }


def _highlight() -> HighlightResult:
    """Step 5 が返すはずのハイライト（3段スナップ済みの区間）。"""
    return HighlightResult(
        selected=HighlightCandidate(
            start=HL_START,
            end=HL_END,
            score=92.0,
            hook_line="実はAIに議事録を書かせるのは一番もったいない使い方なんです",
            reason="結論が先に来ていて単体で意味が通る。",
        ),
        snapped_from=HighlightCandidate(start=26.5, end=33.0, score=92.0),
        silence_snapped=True,
    )


def make_ctx(base: Path, config: Config, video: Path) -> RunContext:
    """独立した work/out を持つ RunContext を作る（conftest の video_ctx の派生版）。

    設定を差し替えた書き出しを何通りも試すので、テストごとに置き場を分ける。
    """
    ctx = RunContext(
        input_path=video,
        episode_id="ep-test",
        work_dir=base / "work" / "ep-test",
        out_dir=base / "out" / "ep-test",
        config=config,
        silence=SilenceConfig(),
    )
    ctx.ensure_dirs()
    return ctx


# ---------------------------------------------------------------------------
# 書き出した mp4 の中身を測る（尺だけでは「どこを切ったか」が分からないため）
# ---------------------------------------------------------------------------

_SILENCE_CACHE: dict[tuple[str, float], list[tuple[float, float]]] = {}


def silences_of(mp4: Path, scratch: Path, *, min_duration: float = 0.12) -> list[tuple[float, float]]:
    """書き出した動画の中の無音区間を、その動画の先頭からの相対時刻で返す。

    合成エピソードは発話の切れ目が完全な無音なので、無音の位置が分かれば
    「元動画のどこを切り出したか」が逆算できる。実尺だけでは
    「長さは合っているが位置がずれている」書き出しを見逃してしまう。
    """
    key = (str(mp4), float(min_duration))
    if key not in _SILENCE_CACHE:
        scratch.mkdir(parents=True, exist_ok=True)
        wav = scratch / f"{Path(mp4).stem}-{min_duration:g}.wav"
        extract_audio(mp4, wav)
        _SILENCE_CACHE[key] = detect_silences(
            wav,
            start=0.0,
            end=media_duration(mp4),
            noise_db=SILENCE_DB,
            min_duration=min_duration,
        )
    return _SILENCE_CACHE[key]


def first_silence_start(mp4: Path, scratch: Path, *, min_duration: float = 0.12) -> float:
    """最初に現れる無音区間の開始時刻（無ければ AssertionError）。"""
    spans = silences_of(mp4, scratch, min_duration=min_duration)
    assert spans, f"{mp4.name} に無音区間が1つも見つかりません（切り出し位置がおかしい可能性）。"
    return spans[0][0]


def top_level_boxes(mp4: Path) -> list[str]:
    """mp4 のトップレベル box 名を出現順に返す（`-movflags +faststart` の検証用）。"""
    names: list[str] = []
    with open(mp4, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size = int.from_bytes(header[:4], "big")
            names.append(header[4:8].decode("ascii", "replace"))
            if size == 1:  # 64bit サイズ
                ext = f.read(8)
                if len(ext) < 8:
                    break
                size = int.from_bytes(ext, "big")
                f.seek(size - 16, 1)
            elif size == 0:  # 「ファイル末尾まで」
                break
            else:
                f.seek(size - 8, 1)
    return names


# ---------------------------------------------------------------------------
# Step 7 を1回だけ走らせて使い回すフィクスチャ
# ---------------------------------------------------------------------------


class Rendered:
    """既定の config で走らせた Step 7 の結果一式。"""

    def __init__(self, ctx: RunContext, result, media: MediaInfo, scratch: Path) -> None:
        self.ctx = ctx
        self.result = result
        self.media = media
        self.scratch = scratch

    def out(self, name: str) -> Path:
        return self.ctx.out_path(name)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, config: Config, episode_video: Path) -> Rendered:
    """同梱の ai-radio.json のまま Step 7 を1回走らせる（重いので module 内で共有）。"""
    base = tmp_path_factory.mktemp("render-default")
    media = probe_media(episode_video)
    ctx = make_ctx(base, config, episode_video)
    result = s7.run(ctx, _cuts(), _highlight(), media)
    return Rendered(ctx, result, media, base / "analysis")


@pytest.fixture(scope="module")
def previewed(tmp_path_factory, config: Config, episode_video: Path):
    """既定の config で Step 8 を1回走らせる（module 内で共有）。"""
    base = tmp_path_factory.mktemp("preview-default")
    media = probe_media(episode_video)
    ctx = make_ctx(base, config, episode_video)
    outputs = s8.run(ctx, _cuts(), _highlight(), media)
    return (ctx, outputs, media, base / "analysis")


# ===========================================================================
# Step 7 — 成果物が揃うこと
# ===========================================================================


@requires_ffmpeg
class TestRenderOutputs:
    """SPEC 1章の成果物表と Step 7。3本＋final が out/<episode_id>/ に出ること。"""

    def test_四本の動画が出力される(self, rendered: Rendered) -> None:
        """01_highlight / 02_main / 03_ending / final の4本が実ファイルとして出る。"""
        for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4", "final.mp4"):
            path = rendered.out(name)
            assert path.exists(), f"{name} が書き出されていません: {path}"
            assert path.stat().st_size > 0, f"{name} が空ファイルです。"

    def test_filesのキーは論理名(self, rendered: Rendered) -> None:
        """RenderResult.files のキーは highlight / main / ending / final。

        decisions.json の durations がこのキーで書かれる（SPEC 8章）ので、
        ファイル名ではなく論理名で引けなければならない。
        """
        assert set(rendered.result.files) == {"highlight", "main", "ending", "final"}
        assert set(rendered.result.durations) == {"highlight", "main", "ending", "final"}

    def test_filesの値は実在するパス(self, rendered: Rendered) -> None:
        """files に載っているパスがそのまま開けること（相対名やダミーではない）。"""
        expected = {
            "highlight": rendered.out("01_highlight.mp4"),
            "main": rendered.out("02_main.mp4"),
            "ending": rendered.out("03_ending.mp4"),
            "final": rendered.out("final.mp4"),
        }
        for key, path in expected.items():
            assert Path(rendered.result.files[key]) == path
            assert Path(rendered.result.files[key]).exists()

    def test_finalはout直下のfinal_mp4(self, rendered: Rendered) -> None:
        """連結済み動画の置き場は out/<episode_id>/final.mp4（SPEC 1章）。"""
        assert s7.final_path(rendered.ctx) == rendered.ctx.out_path("final.mp4")
        assert s7.final_path(rendered.ctx).exists()

    def test_finalに映像と音声が両方入っている(self, rendered: Rendered) -> None:
        """concat で音声が落ちていないこと（-c copy でストリームを取りこぼす事故が起きやすい）。"""
        info = probe_media(rendered.out("final.mp4"))
        assert info.has_video, "final.mp4 に映像ストリームがありません。"
        assert info.has_audio, "final.mp4 に音声ストリームがありません。"

    def test_faststartが効いている(self, rendered: Rendered) -> None:
        """SPEC Step 7 の `-movflags +faststart`。moov が mdat より前に来る。

        YouTube にアップする成果物なので、インデックスが末尾にあると
        アップロード後の処理と手元での頭出しの両方で待たされる。
        """
        for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4", "final.mp4"):
            boxes = top_level_boxes(rendered.out(name))
            assert "moov" in boxes and "mdat" in boxes, f"{name} の box 構成が異常です: {boxes}"
            assert boxes.index("moov") < boxes.index("mdat"), (
                f"{name} は moov が mdat より後ろにあります（faststart が効いていません）: {boxes}"
            )


# ===========================================================================
# Step 7 — 尺が指定した区間と一致すること
# ===========================================================================


@requires_ffmpeg
class TestRenderDurations:
    """SPEC Step 7。実尺がずれる＝カット位置がずれている、なので必ず測る。"""

    def test_各セグメントの実尺が指定区間と一致する(self, rendered: Rendered) -> None:
        """highlight / main / ending の実尺が、指定した区間長と 0.2 秒以内で一致する。"""
        total = rendered.media.duration
        expected = {
            "highlight": HL_END - HL_START,
            "main": CUT_B - CUT_A,
            "ending": total - CUT_B,
        }
        for key, want in expected.items():
            got = rendered.result.durations[key]
            assert got == pytest.approx(want, abs=SEGMENT_TOLERANCE), (
                f"{key} の実尺が {got:.3f}秒 で、指定区間 {want:.3f}秒 と "
                f"{abs(got - want):.3f}秒 ずれています。"
            )

    def test_finalの実尺が三本の合計と一致する(self, rendered: Rendered) -> None:
        """SPEC Step 7 の検算。Dh + Dm + De と final.mp4 の実尺が許容内で一致する。"""
        parts = sum(rendered.result.durations[k] for k in ("highlight", "main", "ending"))
        final = rendered.result.durations["final"]
        tolerance = rendered.ctx.config.render.duration_tolerance_sec
        assert final == pytest.approx(parts, abs=tolerance), (
            f"final.mp4 の実尺 {final:.3f}秒 が3本の合計 {parts:.3f}秒 と "
            f"{abs(final - parts):.3f}秒 ずれています（許容 {tolerance}秒）。"
        )

    def test_許容内なら尺の警告は出ない(self, rendered: Rendered) -> None:
        """既定の許容（0.5秒）に収まっているのだから、尺由来の警告は1つも出てはいけない。

        ここが鳴ると decisions.json の warnings がノイズで埋まり、本物の警告が埋もれる。
        """
        assert rendered.result.warnings == []

    def test_連結の前提である同一パラメータで書き出されている(self, rendered: Rendered) -> None:
        """3本は同じコーデック・同じ解像度で書き出される（SPEC Step 7「同一パラメータ」）。

        concat demuxer は `-c copy` なので、ここがずれると連結が壊れる。
        """
        infos = [
            probe_media(rendered.out(name))
            for name in ("01_highlight.mp4", "02_main.mp4", "03_ending.mp4")
        ]
        shapes = {(i.video_codec, i.width, i.height, i.audio_codec) for i in infos}
        assert len(shapes) == 1, f"3本のパラメータが揃っていません: {shapes}"


# ===========================================================================
# Step 7 — どこを切ったか（Phase 1 の受け入れ基準）
# ===========================================================================


@requires_ffmpeg
class TestRenderCutPositions:
    """SPEC 10章 Phase 1。「02_main の冒頭が『このチャンネルは』」「語頭が欠けていない」。

    文字起こしはテストから回せないので、合成エピソードの無音の位置で代用する。
    無音の位置が正しければ、切り出した区間が元動画のどこだったかが一意に決まる。
    """

    def test_mainはカット点Aちょうどから始まる(self, rendered: Rendered) -> None:
        """02_main.mp4 の最初の無音が 12.40-5.90=6.50 秒に来る（＝5.90 から切り出した証拠）。"""
        got = first_silence_start(rendered.out("02_main.mp4"), rendered.scratch)
        want = fixtures.SILENCES[1][0] - CUT_A
        assert got == pytest.approx(want, abs=0.15), (
            f"02_main.mp4 の最初の無音が {got:.3f}秒 にあります（想定 {want:.3f}秒）。"
            " 切り出しの開始位置がカット点Aとずれています。"
        )

    def test_mainの語頭が欠けていない(self, rendered: Rendered) -> None:
        """先頭に無音の残り（5.90〜5.95 の 0.05 秒）があり、発話 5.95 が丸ごと入っている。

        カット点が発話側にずれると、この頭の無音が消えて語頭が千切れる。
        """
        spans = silences_of(rendered.out("02_main.mp4"), rendered.scratch, min_duration=0.04)
        assert spans and spans[0][0] == pytest.approx(0.0, abs=0.02), (
            "02_main.mp4 の先頭に無音がありません。カット点が発話に食い込んでいる可能性があります"
            f"（検出した無音: {spans[:3]}）。"
        )
        head_end = spans[0][1]
        want = fixtures.EXPECTED_ANCHOR_A_RAW - CUT_A  # 0.05
        assert head_end == pytest.approx(want, abs=0.05), (
            f"先頭の無音が {head_end:.3f}秒 まで続いています（想定 {want:.3f}秒）。"
        )

    def test_endingはカット点Bちょうどから始まる(self, rendered: Rendered) -> None:
        """03_ending.mp4 の最初の無音が 51.20-43.90=7.30 秒に来る。"""
        got = first_silence_start(rendered.out("03_ending.mp4"), rendered.scratch)
        want = fixtures.SILENCES[7][0] - CUT_B
        assert got == pytest.approx(want, abs=0.15), (
            f"03_ending.mp4 の最初の無音が {got:.3f}秒 にあります（想定 {want:.3f}秒）。"
        )

    def test_endingは動画の終端まで入っている(self, rendered: Rendered) -> None:
        """segments[ending].to == "end" なので、終端までまるごと入る（SPEC 5章）。"""
        got = rendered.result.durations["ending"]
        want = rendered.media.duration - CUT_B
        assert got == pytest.approx(want, abs=SEGMENT_TOLERANCE)


# ===========================================================================
# Step 7 — 連結順（highlight.position）
# ===========================================================================


@requires_ffmpeg
class TestConcatOrder:
    """SPEC 5章 highlight.position。prepend は先頭、append は末尾。"""

    def test_concat_orderはprependで先頭にハイライトを置く(self) -> None:
        """純関数。既定（prepend）はハイライトが先頭。"""
        hl = Path("01_highlight.mp4")
        segs = [Path("02_main.mp4"), Path("03_ending.mp4")]
        assert s7.concat_order("prepend", hl, segs) == [hl, *segs]

    def test_concat_orderはappendで末尾にハイライトを置く(self) -> None:
        """純関数。append はハイライトが末尾。"""
        hl = Path("01_highlight.mp4")
        segs = [Path("02_main.mp4"), Path("03_ending.mp4")]
        assert s7.concat_order("append", hl, segs) == [*segs, hl]

    def test_prependではconcatリストの先頭がハイライト(self, rendered: Rendered) -> None:
        """work/concat.txt の並びがそのまま連結順（SPEC Step 7 の concat demuxer）。"""
        lines = _concat_lines(rendered.ctx)
        assert lines[0].endswith("01_highlight.mp4"), f"連結順が想定と違います: {lines}"
        assert [Path(x).name for x in lines] == [
            "01_highlight.mp4",
            "02_main.mp4",
            "03_ending.mp4",
        ]

    def test_prependではfinalの冒頭がハイライト(self, rendered: Rendered) -> None:
        """ハイライト（25.70〜33.60）は無音を含まないので、final の先頭 7.9 秒に無音が無い。"""
        got = first_silence_start(rendered.out("final.mp4"), rendered.scratch)
        hl_dur = rendered.result.durations["highlight"]
        assert got > hl_dur - 0.3, (
            f"final.mp4 の最初の無音が {got:.3f}秒 にあります。"
            f" 先頭 {hl_dur:.3f}秒 はハイライト（無音なし）のはずです。"
        )

    def test_appendにすると連結順が変わる(self, tmp_path, config: Config, episode_video: Path) -> None:
        """position を append にすると、ハイライトが末尾に回る。

        concat リストの並びだけでなく、書き出した final.mp4 の中身でも確かめる。
        append なら本編が先頭なので、最初の無音が 12.40-5.90=6.50 秒に現れる。
        """
        cfg = replace(config, highlight=replace(config.highlight, position="append"))
        ctx = make_ctx(tmp_path, cfg, episode_video)
        media = probe_media(episode_video)
        result = s7.run(ctx, _cuts(), _highlight(), media)

        assert [Path(x).name for x in _concat_lines(ctx)] == [
            "02_main.mp4",
            "03_ending.mp4",
            "01_highlight.mp4",
        ]

        first = first_silence_start(ctx.out_path("final.mp4"), tmp_path / "analysis")
        want = fixtures.SILENCES[1][0] - CUT_A  # 6.50
        assert first == pytest.approx(want, abs=0.3), (
            f"append なのに final.mp4 の最初の無音が {first:.3f}秒 にあります（想定 {want:.3f}秒）。"
            " 本編が先頭に来ていません。"
        )

        # 順番が変わっても合計尺は変わらない。
        parts = sum(result.durations[k] for k in ("highlight", "main", "ending"))
        assert result.durations["final"] == pytest.approx(
            parts, abs=cfg.render.duration_tolerance_sec
        )


def _concat_lines(ctx: RunContext) -> list[str]:
    """work/<episode_id>/concat.txt の `file '...'` 行からパスだけ取り出す。"""
    path = ctx.work_path(s7.CONCAT_LIST_NAME)
    assert path.exists(), f"concat リストが書かれていません: {path}"
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        assert raw.startswith("file '") and raw.endswith("'"), f"concat.txt の行が不正です: {raw!r}"
        lines.append(raw[len("file '") : -1])
    return lines


# ===========================================================================
# Step 7 — 検算とフォールバック
# ===========================================================================


@requires_ffmpeg
class TestRenderVerification:
    """SPEC Step 7「差が0.5秒を超えたら警告を出す」。警告であって停止ではない。"""

    def test_尺がずれたら警告を出すが例外は投げない(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """許容を 0.001 秒にすると必ず検算に引っかかる。それでも書き出しは完走する。

        SPEC Step 7 は「警告を出す」としか言っていない。書き出し自体は成功しているので、
        ここで例外を投げると `--from-step 7` の運用が回らなくなる。
        """
        cfg = replace(config, render=replace(config.render, duration_tolerance_sec=0.001))
        ctx = make_ctx(tmp_path, cfg, episode_video)

        result = s7.run(ctx, _cuts(), _highlight(), probe_media(episode_video))  # 例外が出たら失敗

        final = ctx.out_path("final.mp4")
        assert final.exists(), "警告を出すだけで、final.mp4 は作られていなければならない。"

        final_warnings = [w for w in ctx.warnings if "final.mp4" in w]
        assert final_warnings, (
            f"final.mp4 の尺の検算警告が ctx.warnings に入っていません: {ctx.warnings}"
        )
        assert any("final.mp4" in w for w in result.warnings), (
            f"RenderResult.warnings にも残っていません: {result.warnings}"
        )
        # decisions.json に残す警告なので、何秒ずれたかが読み取れること。
        assert "ずれ" in final_warnings[0]

    def test_duration_gap_warningは許容内なら黙る(self) -> None:
        """純関数。差が許容以内なら None（警告を出さない）。"""
        assert s7.duration_gap_warning(100.0, 100.4, 0.5) is None
        assert s7.duration_gap_warning(100.0, 99.6, 0.5) is None
        assert s7.duration_gap_warning(100.0, 100.5, 0.5) is None  # 境界は許容に含める

    def test_duration_gap_warningは超えたら文言を返す(self) -> None:
        """純関数。許容を超えたら、実尺・想定・差が分かる文言を返す。"""
        msg = s7.duration_gap_warning(100.0, 101.2, 0.5)
        assert msg is not None
        assert "101.2" in msg and "100.0" in msg and "1.2" in msg

    def test_フォールバックコーデックが記録される(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """SPEC 2章・Step 7。使えないコーデックを指定したら CPU エンコードに落ち、
        その事実が RenderResult.used_fallback_codec と ctx.warnings の両方に残る。
        """
        cfg = replace(config, render=replace(config.render, video_codec="no_such_encoder_xyz"))
        ctx = make_ctx(tmp_path, cfg, episode_video)

        result = s7.run(ctx, _cuts(), _highlight(), probe_media(episode_video))

        assert result.used_fallback_codec is True
        assert any("フォールバック" in w for w in ctx.warnings), (
            f"フォールバックした旨が警告に残っていません: {ctx.warnings}"
        )
        assert ctx.out_path("final.mp4").exists(), "フォールバックしても書き出しは完走する。"

    def test_使えるコーデックならフォールバックしない(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """このビルドで使えるコーデックを指定したときに、余計な警告を出さないこと。"""
        cfg = replace(
            config,
            render=replace(config.render, video_codec="libx264", video_bitrate="800k"),
        )
        ctx = make_ctx(tmp_path, cfg, episode_video)

        result = s7.run(ctx, _cuts(), _highlight(), probe_media(episode_video))

        assert result.used_fallback_codec is False
        assert not [w for w in ctx.warnings if "フォールバック" in w], (
            f"フォールバックしていないのに警告が出ています: {ctx.warnings}"
        )


# ===========================================================================
# Step 7 — 中間ファイル（--from-step 用）
# ===========================================================================


@requires_ffmpeg
class TestRenderArtifact:
    """SPEC 3章「中間ファイルを work/ に残し --from-step N で再開できること」。"""

    def test_render_jsonが書かれる(self, rendered: Rendered) -> None:
        """work/<episode_id>/render.json が出る。"""
        path = s7.render_path(rendered.ctx)
        assert path == rendered.ctx.work_path("render.json")
        assert path.exists(), f"render.json が書かれていません: {path}"

    def test_loadで書き出し結果を復元できる(self, rendered: Rendered) -> None:
        """load() が files / durations / used_fallback_codec をそのまま復元する。"""
        restored = s7.load(rendered.ctx)
        assert restored.files == rendered.result.files
        assert restored.used_fallback_codec == rendered.result.used_fallback_codec
        for key, value in rendered.result.durations.items():
            assert restored.durations[key] == pytest.approx(value, abs=0.001)

    def test_render_jsonが無ければ次の一手を添えて止まる(self, tmp_path, config: Config) -> None:
        """中間ファイルが無いまま load() したら MissingArtifactError（勝手に作り直さない）。"""
        ctx = RunContext(
            input_path=tmp_path / "ep-test.mp4",
            episode_id="ep-test",
            work_dir=tmp_path / "work",
            out_dir=tmp_path / "out",
            config=config,
            silence=SilenceConfig(),
        )
        with pytest.raises(MissingArtifactError) as exc:
            s7.load(ctx)
        assert "render.json" in str(exc.value)

    def test_render_jsonが壊れていたら止まる(self, tmp_path, config: Config) -> None:
        """壊れた JSON を黙って無視しない（SPEC 9章「握りつぶさない」）。"""
        ctx = RunContext(
            input_path=tmp_path / "ep-test.mp4",
            episode_id="ep-test",
            work_dir=tmp_path / "work",
            out_dir=tmp_path / "out",
            config=config,
            silence=SilenceConfig(),
        )
        ctx.ensure_dirs()
        s7.render_path(ctx).write_text("{ 壊れている", encoding="utf-8")
        with pytest.raises(MissingArtifactError):
            s7.load(ctx)


# ===========================================================================
# Step 7 — 止まるべきところで止まる
# ===========================================================================


@requires_ffmpeg
class TestRenderGuards:
    """SPEC 9章。おかしな入力で 0 秒の動画を黙って吐かないこと。"""

    def test_カット点が無ければ止まる(self, tmp_path, config: Config, episode_video: Path) -> None:
        """Step 4 を飛ばした状態で書き出そうとしたら MissingArtifactError。"""
        ctx = make_ctx(tmp_path, config, episode_video)
        with pytest.raises(MissingArtifactError):
            s7.run(ctx, {}, _highlight(), probe_media(episode_video))

    def test_セグメント名が予約語と衝突したら止まる(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """durations は highlight / final を予約キーとして使う（SPEC 8章）。

        同名のセグメントを許すと decisions.json の尺が静かに上書きされてしまう。
        """
        collided = replace(config.segments[0], name="highlight")
        cfg = replace(
            config,
            segments=[collided, config.segments[1]],
            highlight=replace(config.highlight, source_segment="highlight"),
        )
        ctx = make_ctx(tmp_path, cfg, episode_video)
        with pytest.raises(RenderError) as exc:
            s7.run(ctx, _cuts(), _highlight(), probe_media(episode_video))
        assert "highlight" in str(exc.value)

    def test_clamp_spanは動画の外に出た区間を切り詰める(self) -> None:
        """純関数。総尺を超える終端は総尺へ、負の開始は 0 へ。"""
        assert s7.clamp_span("テスト", -3.0, 70.0, 60.0) == (0.0, 60.0)
        assert s7.clamp_span("テスト", 5.9, 43.9, 60.0) == (5.9, 43.9)

    def test_clamp_spanは空区間を拒む(self) -> None:
        """切り詰めた結果が空なら RenderError。0秒の mp4 を黙って吐かせない。"""
        with pytest.raises(RenderError):
            s7.clamp_span("テスト", 61.0, 70.0, 60.0)
        with pytest.raises(RenderError):
            s7.clamp_span("テスト", 10.0, 10.0, 60.0)

    def test_clamp_spanは総尺が取れていなければ止まる(self) -> None:
        """総尺 0 は probe の失敗。そのまま書き出しても0秒になるのでここで止める。"""
        with pytest.raises(RenderError):
            s7.clamp_span("テスト", 0.0, 10.0, 0.0)


# ===========================================================================
# Step 8 — 確認用プレビュー
# ===========================================================================


@requires_ffmpeg
class TestPreviewOutputs:
    """SPEC Step 8。カット点2箇所とハイライトの始点・終点、前後2秒＝計4秒。"""

    def test_四本のプレビューが出る(self, previewed) -> None:
        """preview/cut_A.mp4 cut_B.mp4 highlight_in.mp4 highlight_out.mp4（SPEC Step 8）。"""
        ctx, outputs, _media, _scratch = previewed
        directory = s8.preview_dir(ctx)
        assert directory == ctx.out_path("preview")
        names = {"cut_A.mp4", "cut_B.mp4", "highlight_in.mp4", "highlight_out.mp4"}
        for name in names:
            path = directory / name
            assert path.exists(), f"{name} が書き出されていません: {path}"
            assert path.stat().st_size > 0
        assert {p.name for p in outputs} == names

    def test_それぞれ約四秒(self, previewed) -> None:
        """前後2秒なので計4秒。中央のカット点の前後が同じだけ見えないと確認にならない。"""
        ctx, outputs, _media, _scratch = previewed
        for path in outputs:
            got = media_duration(path)
            assert got == pytest.approx(4.0, abs=SEGMENT_TOLERANCE), (
                f"{path.name} が {got:.3f}秒 です（想定 4.0秒）。"
            )

    def test_カット点は窓の中央にある(self, previewed) -> None:
        """cut_A / cut_B の 2.0 秒地点にカット点（無音の谷の終わり）が来る。

        窓が [center, center+4] のようにずれていると、
        「カット直前に何が入っているか」が確認できずプレビューの意味が無くなる。
        """
        ctx, _outputs, _media, scratch = previewed
        for name in ("cut_A.mp4", "cut_B.mp4"):
            spans = silences_of(s8.preview_dir(ctx) / name, scratch)
            assert any(s - 0.1 <= 2.0 <= e + 0.1 for s, e in spans), (
                f"{name} の 2.0秒 地点に無音の谷がありません（検出: {spans}）。"
                " カット点が窓の中央に来ていません。"
            )

    def test_loadでプレビュー一覧を読み戻せる(self, previewed) -> None:
        """--preview-only の再実行で既存のプレビューを拾えること。"""
        ctx, outputs, _media, _scratch = previewed
        assert [p.name for p in s8.load(ctx)] == sorted(p.name for p in outputs)


@pytest.fixture(scope="module")
def edges(tmp_path_factory, config: Config, episode_video: Path):
    """先頭近く（1秒）と末尾近く（総尺-1秒）にカット点を置いて Step 8 を走らせる。"""
    base = tmp_path_factory.mktemp("preview-edges")
    media = probe_media(episode_video)
    ctx = make_ctx(base, config, episode_video)
    cuts = {
        "A": CutPoint(anchor_id="A", raw_cut_time=1.0, cut_time=1.0, silence_found=False),
        "B": CutPoint(
            anchor_id="B",
            raw_cut_time=media.duration - 1.0,
            cut_time=media.duration - 1.0,
            silence_found=False,
        ),
    }
    outputs = s8.run(ctx, cuts, _highlight(), media)
    return (ctx, outputs, media)


@requires_ffmpeg
class TestPreviewClamp:
    """動画の端に寄ったカット点でも、窓が範囲外に出ないこと（SPEC Step 8）。"""

    def test_先頭近くのカット点でも頭がはみ出さない(self, edges) -> None:
        """center=1.0 なら窓は [0.0, 3.0]。負の開始で ffmpeg に無効な -ss を渡さない。"""
        ctx, _outputs, _media = edges
        path = s8.preview_dir(ctx) / "cut_A.mp4"
        assert path.exists()
        assert media_duration(path) == pytest.approx(3.0, abs=SEGMENT_TOLERANCE)

    def test_末尾近くのカット点でも尻がはみ出さない(self, edges) -> None:
        """center=総尺-1 なら窓は [総尺-3, 総尺]。総尺を超える -t を渡さない。"""
        ctx, _outputs, media = edges
        path = s8.preview_dir(ctx) / "cut_B.mp4"
        assert path.exists()
        got = media_duration(path)
        assert got == pytest.approx(3.0, abs=SEGMENT_TOLERANCE)
        assert got <= media.duration

    def test_端に寄せても四本とも出る(self, edges) -> None:
        """クランプされても本数は減らない。"""
        _ctx, outputs, _media = edges
        assert {p.name for p in outputs} == {
            "cut_A.mp4",
            "cut_B.mp4",
            "highlight_in.mp4",
            "highlight_out.mp4",
        }


class TestPreviewWindow:
    """preview_window の純関数としての約束（合成エピソードと同じ総尺 60 秒で確かめる）。"""

    def test_中央に置いた窓は前後2秒(self) -> None:
        """余裕があるときは [center-2, center+2]。"""
        assert s8.preview_window(30.0, EPISODE_DURATION) == (28.0, 32.0)

    def test_先頭ではゼロで止める(self) -> None:
        """負の時刻を作らない（ffmpeg に負の -ss を渡さない）。"""
        assert s8.preview_window(1.0, EPISODE_DURATION) == (0.0, 3.0)
        assert s8.preview_window(0.0, EPISODE_DURATION) == (0.0, 2.0)

    def test_末尾では総尺で止める(self) -> None:
        """総尺を超えない。"""
        assert s8.preview_window(EPISODE_DURATION - 1.0, EPISODE_DURATION) == (57.0, 60.0)
        assert s8.preview_window(EPISODE_DURATION, EPISODE_DURATION) == (58.0, 60.0)

    def test_範囲外の中心はNone(self) -> None:
        """窓が空になるなら None を返し、呼び出し側がスキップできるようにする。"""
        assert s8.preview_window(EPISODE_DURATION * 2, EPISODE_DURATION) is None

    def test_marginは指定できる(self) -> None:
        """前後の幅は引数で変えられる（既定は SPEC の2秒）。"""
        assert s8.PREVIEW_MARGIN_SEC == 2.0
        assert s8.preview_window(30.0, EPISODE_DURATION, margin=0.5) == (29.5, 30.5)

    def test_cut_preview_nameはSPECのファイル名(self) -> None:
        """アンカーID A / B から cut_A.mp4 / cut_B.mp4（SPEC Step 8）。"""
        assert s8.cut_preview_name("A") == "cut_A.mp4"
        assert s8.cut_preview_name("B") == "cut_B.mp4"


@requires_ffmpeg
class TestPreviewPartial:
    """前段が欠けていてもプレビューは止まらない（確認用なので、出せる分だけ出す）。"""

    def test_ハイライトが無くてもカット点は出る(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """Step 5 未実行（--dry-run 前）でも cut_A / cut_B は確認できる。"""
        ctx = make_ctx(tmp_path, config, episode_video)
        outputs = s8.run(ctx, _cuts(), None, probe_media(episode_video))
        assert {p.name for p in outputs} == {"cut_A.mp4", "cut_B.mp4"}
        assert not (s8.preview_dir(ctx) / "highlight_in.mp4").exists()
        assert any("ハイライト" in w for w in ctx.warnings), (
            f"ハイライトを作らなかった旨が警告に残っていません: {ctx.warnings}"
        )

    def test_カット点が無くてもハイライトは出る(
        self, tmp_path, config: Config, episode_video: Path
    ) -> None:
        """逆も同じ。落とした分は警告として decisions.json に残す。"""
        ctx = make_ctx(tmp_path, config, episode_video)
        outputs = s8.run(ctx, {}, _highlight(), probe_media(episode_video))
        assert {p.name for p in outputs} == {"highlight_in.mp4", "highlight_out.mp4"}
        assert any("カット点" in w for w in ctx.warnings), (
            f"カット点が無い旨が警告に残っていません: {ctx.warnings}"
        )

    def test_プレビューが無くてもloadは空リスト(self, tmp_path, config: Config) -> None:
        """後段が依存しないので、無いこと自体はエラーにしない（s7.load とはここが違う）。"""
        ctx = RunContext(
            input_path=tmp_path / "ep-test.mp4",
            episode_id="ep-test",
            work_dir=tmp_path / "work",
            out_dir=tmp_path / "out",
            config=config,
            silence=SilenceConfig(),
        )
        assert s8.load(ctx) == []
