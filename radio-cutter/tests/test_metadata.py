"""steps/s6_metadata.py — 概要欄・チャプター・タイトル候補（SPEC Step 6）。

ここで守らせたいのは3つ。

1. `description.txt` のレイアウトは SPEC 6-a の並びそのもの。
   LLM に最終フォーマットを作らせない以上、コード側の組み立てが唯一の保証になる。
2. チャプターの秒数は `final.mp4` のタイムライン。
   先頭にハイライトを足しているので、ここがずれると概要欄のチャプターが全部ずれる。
   `0:00` は必ずハイライトに割り当てる。
3. LLM が落ちても動画の書き出しは止めない（SPEC 9章）。
   6-a / 6-b のどちらが落ちても例外は投げず、警告に残して作れたものだけ書く。
"""

from __future__ import annotations

import json
import re

import pytest

import fixtures
from radio_cutter.config import YoutubeConfig
from radio_cutter.llm.client import StubLlmClient
from radio_cutter.models import (
    Chapter,
    CutPoint,
    HighlightCandidate,
    HighlightResult,
    MetadataResult,
    Segment,
    TitleCandidate,
    Transcript,
    Word,
)
from radio_cutter.steps import s6_metadata as s6
from radio_cutter.util.text_normalize import zenkaku_length
from radio_cutter.util.timeline import parse_timestamp

# ---------------------------------------------------------------------------
# SPEC から直に写した期待値（実装の定数は参照しない）
# ---------------------------------------------------------------------------

#: SPEC 6-a のレイアウトに出てくる罫線（全角罫線15本）
SPEC_RULE_LINE = "━━━━━━━━━━━━━━━"

#: SPEC 6-a のチャプター見出し
SPEC_CHAPTER_HEADING = "■ チャプター"

#: SPEC 6-b の表の並び。titles.md の見出しはこの順に出る。
SPEC_DIRECTIONS: tuple[str, ...] = (
    "結論直球型",
    "逆説・否定型",
    "数字型",
    "疑問型",
    "実験・検証型",
    "ターゲット明示型",
)

#: YouTube チャプターの成立条件（SPEC 6-a）
SPEC_MIN_CHAPTER_GAP = 10.0

# ---------------------------------------------------------------------------
# 合成エピソードから決まる数値
# ---------------------------------------------------------------------------

CUT_A = fixtures.EXPECTED_CUT_A          # 5.90
CUT_B = fixtures.EXPECTED_CUT_B          # 43.90
TOTAL = fixtures.EPISODE_DURATION        # 60.0

#: ハイライトに使う本編内の一文（25.70〜33.60）
HL_START = 25.70
HL_END = 33.60
HOOK = "実はAIに議事録を書かせるのは一番もったいない使い方なんです"


def make_cuts() -> dict[str, CutPoint]:
    """Step 4 が出すカット点（合成エピソードの期待値）。"""
    return {
        "A": CutPoint(anchor_id="A", raw_cut_time=fixtures.EXPECTED_ANCHOR_A_RAW,
                      cut_time=CUT_A, silence_found=True, score=100.0),
        "B": CutPoint(anchor_id="B", raw_cut_time=fixtures.EXPECTED_ANCHOR_B_RAW,
                      cut_time=CUT_B, silence_found=True, score=96.2),
    }


def make_highlight(
    start: float = HL_START, end: float = HL_END, hook_line: str = HOOK
) -> HighlightResult:
    """Step 5 が出すハイライト（スナップ済み）。"""
    selected = HighlightCandidate(
        start=start, end=end, score=92.0, hook_line=hook_line, reason="逆説を含む結論。"
    )
    return HighlightResult(
        selected=selected,
        snapped_from=HighlightCandidate(start=start + 0.8, end=end - 0.6, score=92.0),
        alternatives=[],
        silence_snapped=True,
    )


def make_stub(metadata: dict | None = None, titles: dict | None = None) -> StubLlmClient:
    """step 名を持たせなければ、その呼び出しは LlmError になる（= LLM 失敗の再現）。"""
    responses: dict[str, dict] = {}
    if metadata is not None:
        responses["metadata"] = metadata
    if titles is not None:
        responses["titles"] = titles
    return StubLlmClient(responses, model="stub-model")


def metadata_response(chapters: list[dict]) -> dict:
    """METADATA_SCHEMA を満たす 6-a の応答。チャプターだけ差し替える。"""
    base = fixtures.stub_metadata_response()
    base["chapters"] = chapters
    return base


def chapter_lines(text: str) -> list[str]:
    """description.txt から「M:SS ラベル」の行だけを取り出す。"""
    lines = text.splitlines()
    head = lines.index(SPEC_CHAPTER_HEADING)
    out: list[str] = []
    for line in lines[head + 1 :]:
        if not line.strip() or line.startswith(SPEC_RULE_LINE):
            break
        out.append(line)
    return out


def titles_from(response: dict) -> list[TitleCandidate]:
    """6-b のスタブ応答を TitleCandidate のリストにする。"""
    return [TitleCandidate.from_dict(t) for t in response["titles"]]


# ---------------------------------------------------------------------------
# build_description（純関数）
# ---------------------------------------------------------------------------


class TestBuildDescription:
    """SPEC 6-a のレイアウトをコード側で固定する。"""

    def test_SPEC_6a_のレイアウトどおりの順で並ぶ(self) -> None:
        """lead → body → 罫線 → チャプター → 罫線 → footer → links → hashtags の順。"""
        meta = MetadataResult(
            summary_lead="AIに議事録を書かせるのは、実はいちばんもったいない使い方でした。",
            body="本文の1段落目。\n\n本文の2段落目。",
            chapters=[
                Chapter(time_sec=0.0, label="今回の結論"),
                Chapter(time_sec=32.0, label="オープニング"),
                Chapter(time_sec=118.0, label="AI議事録の落とし穴"),
            ],
            keywords=["AI議事録"],
        )
        youtube = YoutubeConfig(
            channel_links=("https://example.com/ch", "https://example.com/x"),
            fixed_footer="ご視聴ありがとうございます。",
            hashtags=("#AI", "#AI活用", "#生成AI"),
        )
        expected = "\n".join(
            [
                "AIに議事録を書かせるのは、実はいちばんもったいない使い方でした。",
                "",
                "本文の1段落目。",
                "",
                "本文の2段落目。",
                "",
                SPEC_RULE_LINE,
                SPEC_CHAPTER_HEADING,
                "0:00 今回の結論",
                "0:32 オープニング",
                "1:58 AI議事録の落とし穴",
                "",
                SPEC_RULE_LINE,
                "ご視聴ありがとうございます。",
                "https://example.com/ch",
                "https://example.com/x",
                "",
                "#AI #AI活用 #生成AI",
                "",
            ]
        )
        assert s6.build_description(meta, youtube) == expected

    @pytest.mark.parametrize(
        "sec,want",
        [
            (0.0, "0:00"),
            (5.0, "0:05"),
            (32.0, "0:32"),
            (118.0, "1:58"),
            (599.0, "9:59"),
            (600.0, "10:00"),
            (3599.0, "59:59"),
            (3600.0, "1:00:00"),
            (3661.0, "1:01:01"),
            (7325.9, "2:02:05"),
        ],
    )
    def test_チャプター時刻は1時間未満が_M_SS_1時間以上が_H_MM_SS(self, sec, want) -> None:
        """SPEC 6-a「時刻の書式は1時間未満なら M:SS、1時間以上なら H:MM:SS」。"""
        meta = MetadataResult(chapters=[Chapter(time_sec=sec, label="ラベル")])
        line = chapter_lines(s6.build_description(meta, YoutubeConfig()))[0]
        assert line == f"{want} ラベル"

    def test_youtube_設定が空でも余計な空行や罫線が残らない(self) -> None:
        """fixed_footer / channel_links / hashtags が空なら、その行も末尾の罫線も出ない。"""
        meta = MetadataResult(
            summary_lead="リード。",
            body="本文。",
            chapters=[Chapter(time_sec=0.0, label="今回の結論")],
        )
        text = s6.build_description(meta, YoutubeConfig())

        assert text.count(SPEC_RULE_LINE) == 1, "空の末尾ブロックに罫線だけ残ってはいけない"
        assert "\n\n\n" not in text, "空要素の分の空行が残ってはいけない"
        assert text.endswith("今回の結論\n")

    def test_hashtags_だけあるときフッター行は出ないが罫線は引かれる(self) -> None:
        """ハッシュタグは罫線より下（SPEC のレイアウト）。空の footer / links は行ごと消える。"""
        meta = MetadataResult(
            summary_lead="リード。", body="本文。",
            chapters=[Chapter(time_sec=0.0, label="今回の結論")],
        )
        youtube = YoutubeConfig(hashtags=("#AI", "#生成AI"))
        text = s6.build_description(meta, youtube)

        assert text.count(SPEC_RULE_LINE) == 2
        assert "\n\n\n" not in text
        assert text.rstrip().endswith("#AI #生成AI")

    def test_空文字のリンクやハッシュタグは落とす(self) -> None:
        """設定に空文字が混ざっても空行にしない。"""
        meta = MetadataResult(summary_lead="リード。", chapters=[Chapter(0.0, "今回の結論")])
        youtube = YoutubeConfig(
            channel_links=("", "https://example.com/ch", "   "),
            fixed_footer="  \n ",
            hashtags=("#AI", "", "  "),
        )
        text = s6.build_description(meta, youtube)

        assert text == "\n".join(
            [
                "リード。",
                "",
                SPEC_RULE_LINE,
                SPEC_CHAPTER_HEADING,
                "0:00 今回の結論",
                "",
                SPEC_RULE_LINE,
                "https://example.com/ch",
                "",
                "#AI",
                "",
            ]
        )

    def test_チャプターが無ければ見出しごと出さない(self) -> None:
        """0件のときに「■ チャプター」だけ残ると概要欄として壊れる。"""
        meta = MetadataResult(summary_lead="リード。", body="本文。")
        text = s6.build_description(meta, YoutubeConfig(hashtags=("#AI",)))

        assert SPEC_CHAPTER_HEADING not in text
        assert "\n\n\n" not in text

    def test_最初のチャプター行は_0_00_で始まる(self) -> None:
        """SPEC 6-a「最初のチャプターは必ず 0:00」。"""
        meta = MetadataResult(
            chapters=[Chapter(0.0, "今回の結論"), Chapter(45.0, "本編")],
        )
        lines = chapter_lines(s6.build_description(meta, YoutubeConfig()))
        assert lines[0].startswith("0:00 ")

    def test_リードや本文が空でも空行から始まらない(self) -> None:
        """6-a が部分的にしか返らなかったときに、先頭が空行になると貼り付け事故になる。"""
        meta = MetadataResult(chapters=[Chapter(0.0, "今回の結論")])
        text = s6.build_description(meta, YoutubeConfig(hashtags=("#AI",)))

        assert not text.startswith("\n")
        assert text.splitlines()[0] == SPEC_RULE_LINE
        assert "\n\n\n" not in text

    def test_ラベルの前後の空白は落として1スペース区切りにする(self) -> None:
        """チャプター行は「M:SS ラベル」。余分な空白で貼り付け時にずれないようにする。"""
        meta = MetadataResult(chapters=[Chapter(0.0, "  今回の結論  ")])
        assert chapter_lines(s6.build_description(meta, YoutubeConfig()))[0] == "0:00 今回の結論"


# ---------------------------------------------------------------------------
# build_titles_markdown（純関数）
# ---------------------------------------------------------------------------


class TestBuildTitlesMarkdown:
    """SPEC 6-b の titles.md。30個・6方向・通し番号・全角字数。"""

    @pytest.fixture
    def titles(self) -> list[TitleCandidate]:
        return titles_from(fixtures.stub_titles_response())

    def test_見出しと方向の並びが_SPEC_の表の順になる(self, titles) -> None:
        """入力の並びに関係なく、SPEC 6-b の表の順で H2 が出る。"""
        shuffled = list(reversed(titles))
        text = s6.build_titles_markdown(shuffled)
        lines = text.splitlines()

        assert lines[0] == "# タイトル候補"
        headings = [line[3:] for line in lines if line.startswith("## ")]
        assert headings == list(SPEC_DIRECTIONS)

    def test_通し番号が全体で1から30まで連番になる(self, titles) -> None:
        """方向ごとに1に戻さない（SPEC の例では逆説型が 6. から始まる）。"""
        text = s6.build_titles_markdown(titles)
        numbers = [
            int(line.split(".", 1)[0])
            for line in text.splitlines()
            if line[:1].isdigit()
        ]
        assert numbers == list(range(1, 31))

    def test_各行末に全角字数が付く(self, titles) -> None:
        """SPEC 6-b「各行に想定文字数を併記する」。"""
        text = s6.build_titles_markdown(titles)
        body_lines = [line for line in text.splitlines() if line[:1].isdigit()]
        assert len(body_lines) == 30

        for line in body_lines:
            assert line.endswith("字）"), line
            head, _, tail = line.partition("（全角")
            assert tail.endswith("字）"), line
            length = int(tail[: -len("字）")])
            title_text = head.split(". ", 1)[1]
            assert length == zenkaku_length(title_text), line

    def test_方向ごとに5個ずつ並ぶ(self, titles) -> None:
        """6方向 × 5個（SPEC 6-b）。"""
        text = s6.build_titles_markdown(titles)
        counts: dict[str, int] = {}
        current = ""
        for line in text.splitlines():
            if line.startswith("## "):
                current = line[3:]
                counts[current] = 0
            elif line[:1].isdigit():
                counts[current] += 1
        assert counts == {d: 5 for d in SPEC_DIRECTIONS}

    def test_タイトルが0個でも見出しだけは書ける(self) -> None:
        """6-b が空を返しても titles.md の体裁は壊さない。"""
        text = s6.build_titles_markdown([])
        assert text.startswith("# タイトル候補")
        assert text.endswith("\n")

    def test_未知の方向も取りこぼさない(self, titles) -> None:
        """想定外の方向が混ざっても捨てない（人が見て判断できるように残す）。"""
        extra = TitleCandidate(direction="謎の方向", text="よく分からないタイトル")
        text = s6.build_titles_markdown([*titles, extra])
        headings = [line[3:] for line in text.splitlines() if line.startswith("## ")]

        assert headings[: len(SPEC_DIRECTIONS)] == list(SPEC_DIRECTIONS)
        assert "謎の方向" in headings
        assert "31. よく分からないタイトル" in text


# ---------------------------------------------------------------------------
# summarize_transcript_window（6-a の入力）
# ---------------------------------------------------------------------------


def tiny_transcript(spans: list[tuple[float, float, str]], duration: float = 100.0) -> Transcript:
    segments = [
        Segment(start=s, end=e, text=t, words=[Word(word=t, start=s, end=e)])
        for s, e, t in spans
    ]
    return Transcript(language="ja", duration=duration, segments=segments)


class TestSummarizeTranscriptWindow:
    """SPEC 6-a「トークン節約のため30秒単位に丸めたセグメント要約でよい」。"""

    def test_30秒単位のバケツに丸められる(self, transcript) -> None:
        """バケツの開始秒は必ず窓幅の倍数。ここが崩れると LLM に渡す時刻表が信用できなくなる。"""
        windows = s6.summarize_transcript_window(transcript, 0.0, fixtures.EPISODE_DURATION)
        starts = [w["start"] for w in windows]

        assert starts == [0.0, 30.0]
        assert all(w["start"] % 30.0 == 0 for w in windows)

    def test_バケツの中身は開始時刻でその区間のものだけになる(self, transcript) -> None:
        """セグメントは開始時刻の属するバケツに入る。隣のバケツへ漏れてはいけない。"""
        windows = s6.summarize_transcript_window(transcript, 0.0, fixtures.EPISODE_DURATION)
        first, second = windows

        # 25.70 開始の発話は前半のバケツ、33.90 開始の発話は後半のバケツ。
        assert "一番もったいない" in first["text"]
        assert "一番もったいない" not in second["text"]
        assert "理由は三つ" in second["text"]
        assert "理由は三つ" not in first["text"]

    def test_区間外のセグメントは入らない(self, transcript) -> None:
        """アンカーBより後（エンディング）だけを取ると、本編の話は混ざらない。"""
        windows = s6.summarize_transcript_window(transcript, CUT_B, fixtures.EPISODE_DURATION)
        joined = "".join(w["text"] for w in windows)

        assert "木原" in joined
        assert "議題の設計" not in joined, "アンカーBより前の本編が混ざっている"
        assert "始めていきます" not in joined

    def test_境界のセグメントの扱い(self) -> None:
        """区間の外側で終わる／始まるセグメントは入れない。少しでも重なるものは入れる。"""
        t = tiny_transcript(
            [
                (0.0, 10.0, "前"),      # end == lo → 入らない
                (10.0, 20.0, "重なり"),  # 区間の頭に接する → 入る
                (35.0, 40.0, "中"),
                (40.0, 50.0, "後"),      # start == hi → 入らない
            ]
        )
        joined = "".join(w["text"] for w in s6.summarize_transcript_window(t, 10.0, 40.0))

        assert "前" not in joined
        assert "重なり" in joined
        assert "中" in joined
        assert "後" not in joined

    def test_バケツの開始時刻は元動画の絶対秒のまま返す(self) -> None:
        """final.mp4 への変換は呼び出し側の仕事（ここで二重に変換されると全部ずれる）。"""
        t = tiny_transcript([(12.0, 13.0, "あ"), (70.0, 71.0, "い")])
        windows = s6.summarize_transcript_window(t, 10.0, 80.0)

        assert [w["start"] for w in windows] == [10.0, 70.0]

    def test_window_は変えられる(self) -> None:
        """丸め幅は引数で変えられる（既定の30秒はSPECの目安）。"""
        t = tiny_transcript([(0.0, 1.0, "あ"), (10.0, 11.0, "い"), (20.0, 21.0, "う")])
        windows = s6.summarize_transcript_window(t, 0.0, 30.0, window=10.0)

        assert [w["start"] for w in windows] == [0.0, 10.0, 20.0]

    def test_空区間は空リスト(self, transcript) -> None:
        """start >= end の指定は空。例外にせず空リストで返す（呼び出し側が警告を出す）。"""
        assert s6.summarize_transcript_window(transcript, 30.0, 30.0) == []
        assert s6.summarize_transcript_window(transcript, 40.0, 30.0) == []

    def test_window_が0以下なら_ValueError(self, transcript) -> None:
        """0秒幅のバケツは作れない。黙って無限ループするより落とす。"""
        with pytest.raises(ValueError):
            s6.summarize_transcript_window(transcript, 0.0, 60.0, window=0.0)


class TestRenderTranscriptLines:
    """SPEC 6-a の時刻変換式（`Dh + (t - cut_A)` / `Dh + Dm + (t - cut_B)`）。"""

    def test_本編とエンディングで別々の式が使われる(self) -> None:
        """本編は Dh + (t - cut_A)、エンディングは Dh + Dm + (t - cut_B)。"""
        windows = [
            {"start": 10.0, "text": "本編の話"},
            {"start": 50.0, "text": "エンディングの話"},
        ]
        text = s6.render_transcript_lines(
            windows, cut_a=CUT_A, cut_b=CUT_B, highlight_dur=7.9, main_dur=38.0
        )
        seconds = [int(line[1 : line.index("]")]) for line in text.splitlines()]

        # 本編:      7.9 + (10.0 - 5.90) = 12.0
        # エンディング: 7.9 + 38.0 + (50.0 - 43.90) = 52.0
        assert seconds == [12, 52]
        assert "本編の話" in text and "エンディングの話" in text


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def run_step(ctx, transcript, llm, *, highlight: HighlightResult | None = None):
    return s6.run(
        ctx,
        transcript,
        make_cuts(),
        highlight or make_highlight(),
        llm,
        total_duration=TOTAL,
    )


class TestRunHappyPath:
    """スタブ応答で一通り通ったときに、何がどこに書かれるか。"""

    @pytest.fixture
    def result(self, ctx, transcript, stub_llm):
        return run_step(ctx, transcript, stub_llm), ctx

    def test_成果物と中間ファイルが揃う(self, result) -> None:
        """out に description.txt と titles.md、work に metadata.json。"""
        _, ctx = result
        assert (ctx.out_dir / "description.txt").exists()
        assert (ctx.out_dir / "titles.md").exists()
        assert (ctx.work_dir / "metadata.json").exists()

    def test_チャプターは_final_mp4_のタイムラインで昇順になる(self, result) -> None:
        """スタブの 0 / 12 / 40 / 44 秒のうち、44 秒は直前と 4 秒差なので落ちる。"""
        meta, ctx = result
        times = [c.time_sec for c in meta.chapters]

        assert times == sorted(times)
        assert times[0] == 0.0
        assert times == [0.0, 12.0, 40.0]
        assert all(
            b - a >= SPEC_MIN_CHAPTER_GAP for a, b in zip(times, times[1:])
        ), "10秒未満の間隔のチャプターが残っている"

    def test_間隔が詰まったチャプターを落としたことが警告に残る(self, result) -> None:
        """黙って捨てるとあとから何が起きたか追えない（decisions.json の warnings に載る）。"""
        _, ctx = result
        assert any("議題の設計に使う" in w for w in ctx.warnings), ctx.warnings

    def test_0_00_はハイライトに割り当てられる(self, result) -> None:
        """SPEC 6-a「0:00 は必ずハイライト部分に割り当てる」。"""
        meta, _ = result
        assert meta.chapters[0].time_sec == 0.0
        assert meta.chapters[0].label == "今回の結論"

    def test_チャプターは3つ以上ある(self, result) -> None:
        """YouTube のチャプター成立条件（SPEC 6-a）。"""
        meta, _ = result
        assert len(meta.chapters) >= 3

    def test_description_txt_の中身は_build_description_と一致する(self, result, config) -> None:
        """書き出しは組み立て結果そのまま。ファイル側で余計な加工をしない。"""
        meta, ctx = result
        text = (ctx.out_dir / "description.txt").read_text(encoding="utf-8")
        assert text == s6.build_description(meta, config.youtube)

    def test_description_txt_のチャプター行が読み戻せる(self, result) -> None:
        """概要欄はそのまま貼れる状態でなければならない（SPEC 1章）。"""
        meta, ctx = result
        text = (ctx.out_dir / "description.txt").read_text(encoding="utf-8")
        lines = chapter_lines(text)

        assert len(lines) == len(meta.chapters)
        assert lines[0].startswith("0:00 ")
        for line, chapter in zip(lines, meta.chapters):
            stamp, _, label = line.partition(" ")
            assert parse_timestamp(stamp) == pytest.approx(int(chapter.time_sec))
            assert label == chapter.label

    def test_titles_md_に30個書かれる(self, result) -> None:
        """SPEC 6-b の30個が out/titles.md にそのまま出る。"""
        meta, ctx = result
        text = (ctx.out_dir / "titles.md").read_text(encoding="utf-8")

        assert len(meta.titles) == 30
        assert text == s6.build_titles_markdown(meta.titles)
        assert text.splitlines()[0] == "# タイトル候補"

    def test_llm_呼び出しは2回で_step_名が_metadata_と_titles(self, result) -> None:
        """SPEC 6章「LLM呼び出しは2回に分ける。同時に投げると片方の品質が落ちる」。"""
        _, ctx = result
        assert [c.step for c in ctx.llm_calls] == ["metadata", "titles"]
        assert all(c.ok for c in ctx.llm_calls)
        assert all(c.model == "stub-model" for c in ctx.llm_calls)

    def test_metadata_json_を読み戻せる(self, result) -> None:
        """`--from-step` と `titles` サブコマンドが読み直すので、往復して壊れないこと。"""
        meta, ctx = result
        loaded = s6.load(ctx)

        assert loaded.summary_lead == meta.summary_lead
        assert [c.to_dict() for c in loaded.chapters] == [c.to_dict() for c in meta.chapters]
        assert len(loaded.titles) == len(meta.titles)

    def test_6a_の入力は_final_mp4_の秒に直してから渡される(self, ctx, transcript, stub_llm) -> None:
        """SPEC 6-a「チャプターの時刻は必ず final.mp4 のタイムラインに変換すること」。

        本編の頭（cut_A = 5.90秒）は final 上ではハイライトの尺（7.9秒）ちょうど。
        変換せずに渡すと LLM は元動画の秒でチャプターを作り、全部ずれる。
        """
        run_step(ctx, transcript, stub_llm)
        prompt = next(c["prompt"] for c in stub_llm.calls if c["step"] == "metadata")
        lines = re.findall(r"^\[(\d+)\] (.+)$", prompt, re.MULTILINE)

        # 30秒バケツの先頭は元動画の 5.90 / 35.90 秒。final では 7.9 / 37.9 秒。
        assert [int(sec) for sec, _ in lines] == [7, 37], lines
        assert "このチャンネルは" in lines[0][1]
        assert "始めていきます" not in prompt, "アンカーAより前が 6-a に渡っている"


class TestRunChapterRepair:
    """LLM の返すチャプターは素直ではない。ここを直せないと概要欄が壊れる。"""

    def test_0_00_がオープニングでもハイライトに差し替える(self, ctx, transcript) -> None:
        """SPEC 6-a の悪い例。0:00 は冒頭のハイライトを指していないといけない。"""
        highlight = make_highlight(start=19.95, end=33.60)   # 尺 13.65 秒
        llm = make_stub(
            metadata=metadata_response(
                [
                    {"time_sec": 0, "label": "オープニング"},
                    {"time_sec": 30, "label": "AI議事録の落とし穴"},
                    {"time_sec": 50, "label": "まとめ"},
                ]
            ),
            titles=fixtures.stub_titles_response(),
        )
        meta = run_step(ctx, transcript, llm, highlight=highlight)
        times = [c.time_sec for c in meta.chapters]
        labels = [c.label for c in meta.chapters]

        assert times[0] == 0.0
        assert labels[0] != "オープニング", "0:00 が冒頭ハイライトを指していない"
        assert "オープニング" in labels
        # 追い出された「オープニング」は本編の頭（= ハイライトの尺）へ移る。
        assert meta.chapters[labels.index("オープニング")].time_sec == pytest.approx(13.65)
        assert any("0:00" in w for w in ctx.warnings), ctx.warnings

    def test_順番が崩れていても昇順に直す(self, ctx, transcript) -> None:
        """SPEC 6-a「動画内の時刻の昇順」。LLM の返す順は当てにしない。"""
        llm = make_stub(
            metadata=metadata_response(
                [
                    {"time_sec": 40, "label": "三つ目"},
                    {"time_sec": 0, "label": "今回の結論"},
                    {"time_sec": 12, "label": "二つ目"},
                ]
            ),
            titles=fixtures.stub_titles_response(),
        )
        meta = run_step(ctx, transcript, llm)

        assert [c.time_sec for c in meta.chapters] == [0.0, 12.0, 40.0]
        assert [c.label for c in meta.chapters] == ["今回の結論", "二つ目", "三つ目"]

    def test_0_00_が無ければ挿入する(self, ctx, transcript) -> None:
        """SPEC 6-a「最初のチャプターは必ず 0:00」。"""
        llm = make_stub(
            metadata=metadata_response(
                [
                    {"time_sec": 20, "label": "本編の話"},
                    {"time_sec": 40, "label": "まとめ"},
                ]
            ),
            titles=fixtures.stub_titles_response(),
        )
        meta = run_step(ctx, transcript, llm)

        assert meta.chapters[0].time_sec == 0.0
        assert [c.label for c in meta.chapters][1:] == ["本編の話", "まとめ"]

    def test_final_mp4_の尺を超えるチャプターは落とす(self, ctx, transcript) -> None:
        """final.mp4 は 62.0 秒（7.9 + 38.0 + 16.1）。その先を指すチャプターは成立しない。"""
        llm = make_stub(
            metadata=metadata_response(
                [
                    {"time_sec": 0, "label": "今回の結論"},
                    {"time_sec": 20, "label": "本編の話"},
                    {"time_sec": 40, "label": "まとめ"},
                    {"time_sec": 900, "label": "存在しない区間"},
                ]
            ),
            titles=fixtures.stub_titles_response(),
        )
        meta = run_step(ctx, transcript, llm)

        assert [c.label for c in meta.chapters] == ["今回の結論", "本編の話", "まとめ"]
        assert any("存在しない区間" in w for w in ctx.warnings), ctx.warnings

    def test_チャプターが3つ未満なら警告に残す(self, ctx, transcript) -> None:
        """SPEC 6-a「3つ以上」。黙って出すと YouTube にチャプターとして認識されない。"""
        llm = make_stub(
            metadata=metadata_response(
                [
                    {"time_sec": 0, "label": "今回の結論"},
                    {"time_sec": 20, "label": "本編の話"},
                ]
            ),
            titles=fixtures.stub_titles_response(),
        )
        meta = run_step(ctx, transcript, llm)

        assert len(meta.chapters) == 2
        assert any("3" in w and "チャプター" in w for w in ctx.warnings), ctx.warnings


class TestRunTitleDirections:
    """SPEC 6-b「30個を6方向 × 5個で生成させる。指定しないと同じ言い回しの30変奏になる」。"""

    def test_6b_のプロンプトに6方向が全部載る(self, ctx, transcript, stub_llm) -> None:
        """方向を指定しないと同じ言い回しの30変奏になる（SPEC 6-b）。"""
        run_step(ctx, transcript, stub_llm)
        prompt = next(c["prompt"] for c in stub_llm.calls if c["step"] == "titles")

        for direction in SPEC_DIRECTIONS:
            assert direction in prompt, direction

    def test_方向の本数が偏っていたら警告するが捨てはしない(self, ctx, transcript) -> None:
        """30個は揃っているのだから、こちらで間引くより人に見せた方がよい。"""
        response = fixtures.stub_titles_response()
        # 「数字型」の1個を「疑問型」に付け替える（合計30個のまま 4個 / 6個 にする）。
        moved = next(t for t in response["titles"] if t["direction"] == "数字型")
        moved["direction"] = "疑問型"
        llm = make_stub(metadata=fixtures.stub_metadata_response(), titles=response)

        meta = run_step(ctx, transcript, llm)

        assert len(meta.titles) == 30
        assert any("数字型" in w and "4" in w for w in ctx.warnings), ctx.warnings
        text = (ctx.out_dir / "titles.md").read_text(encoding="utf-8")
        numbers = [int(line.split(".", 1)[0]) for line in text.splitlines() if line[:1].isdigit()]
        assert numbers == list(range(1, 31))


class TestRunLlmFailure:
    """SPEC 9章「そのステップだけ落とし、動画の書き出しは続行する」。"""

    def test_6a_が落ちても例外を投げずに続行する(self, ctx, transcript) -> None:
        """description.txt が無くても動画は出す（SPEC 9章）。titles.md は作る。"""
        llm = make_stub(metadata=None, titles=fixtures.stub_titles_response())
        meta = run_step(ctx, transcript, llm)

        assert not (ctx.out_dir / "description.txt").exists()
        assert (ctx.out_dir / "titles.md").exists()
        assert (ctx.work_dir / "metadata.json").exists()
        assert meta.chapters == []
        assert len(meta.titles) == 30
        assert any("6-a" in w for w in ctx.warnings), ctx.warnings

    def test_6a_が落ちても_llm_calls_に失敗が残る(self, ctx, transcript) -> None:
        """失敗も decisions.json に残す。あとから何が起きたか追えるようにする。"""
        llm = make_stub(metadata=None, titles=fixtures.stub_titles_response())
        run_step(ctx, transcript, llm)
        by_step = {c.step: c for c in ctx.llm_calls}

        assert [c.step for c in ctx.llm_calls] == ["metadata", "titles"]
        assert by_step["metadata"].ok is False
        assert by_step["metadata"].error
        assert by_step["titles"].ok is True

    def test_6b_が落ちても例外を投げずに続行する(self, ctx, transcript) -> None:
        """titles.md を書かずに続行し、description.txt は残す（SPEC 9章）。"""
        llm = make_stub(metadata=fixtures.stub_metadata_response(), titles=None)
        meta = run_step(ctx, transcript, llm)

        assert (ctx.out_dir / "description.txt").exists()
        assert not (ctx.out_dir / "titles.md").exists()
        assert (ctx.work_dir / "metadata.json").exists()
        assert meta.titles == []
        assert meta.chapters, "6-a は成功しているのでチャプターは残る"
        assert any("6-b" in w for w in ctx.warnings), ctx.warnings

    def test_6b_が落ちても_llm_calls_に失敗が残る(self, ctx, transcript) -> None:
        """失敗した呼び出しも step 名つきで記録する。"""
        llm = make_stub(metadata=fixtures.stub_metadata_response(), titles=None)
        run_step(ctx, transcript, llm)
        by_step = {c.step: c for c in ctx.llm_calls}

        assert [c.step for c in ctx.llm_calls] == ["metadata", "titles"]
        assert by_step["metadata"].ok is True
        assert by_step["titles"].ok is False

    def test_両方落ちても_metadata_json_は書かれる(self, ctx, transcript) -> None:
        """あとから何が起きたか追えるように、空でも中間ファイルは残す。"""
        llm = make_stub(metadata=None, titles=None)
        meta = run_step(ctx, transcript, llm)

        assert not (ctx.out_dir / "description.txt").exists()
        assert not (ctx.out_dir / "titles.md").exists()

        path = ctx.work_dir / "metadata.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["chapters"] == []
        assert data["titles"] == []
        assert meta.summary_lead == ""
        assert len(ctx.warnings) >= 2

    def test_6a_が落ちても_6b_は試す(self, ctx, transcript) -> None:
        """タイトルだけでも出れば手作業の起点になる。"""
        llm = make_stub(metadata=None, titles=fixtures.stub_titles_response())
        run_step(ctx, transcript, llm)

        assert [c["step"] for c in llm.calls] == ["metadata", "titles"]


class TestRegenerateTitles:
    """`radio-cutter titles <ep-id>` はタイトルだけを作り直す（SPEC 7章）。"""

    def test_titles_md_を上書きし_6a_は呼ばない(self, ctx, transcript, stub_llm) -> None:
        """タイトルだけ作り直す。概要欄の LLM 呼び出しは走らせない。"""
        meta = run_step(ctx, transcript, stub_llm)
        (ctx.out_dir / "titles.md").write_text("古い内容", encoding="utf-8")

        again = make_stub(titles=fixtures.stub_titles_response())
        s6.regenerate_titles(ctx, meta, make_highlight(), again)

        assert [c["step"] for c in again.calls] == ["titles"]
        text = (ctx.out_dir / "titles.md").read_text(encoding="utf-8")
        assert text.startswith("# タイトル候補")
        assert "古い内容" not in text

    def test_失敗しても例外を投げない(self, ctx, transcript, stub_llm) -> None:
        """再生成に失敗したら既存の titles.md をそのまま残す（消さない）。"""
        meta = run_step(ctx, transcript, stub_llm)
        before = (ctx.out_dir / "titles.md").read_text(encoding="utf-8")

        broken = make_stub()  # titles の応答が無い → LlmError
        s6.regenerate_titles(ctx, meta, make_highlight(), broken)

        assert (ctx.out_dir / "titles.md").read_text(encoding="utf-8") == before
        assert any("6-b" in w for w in ctx.warnings), ctx.warnings
