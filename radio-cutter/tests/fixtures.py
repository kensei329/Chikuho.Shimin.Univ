"""テスト用の合成エピソードを作る。

実物の収録動画をリポジトリに置けないので、
「決まり文句が仕様どおりの位置に入っていて、その直前に無音がある」
60秒のエピソードを合成する。音声・動画・文字起こしの3つが同じ時刻表を共有する。
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from radio_cutter.models import Segment, Transcript, Word

SAMPLE_RATE = 16000
TONE_FREQ = 440.0
TONE_AMPLITUDE = 0.35

#: 合成エピソードの総尺
EPISODE_DURATION = 60.0

#: 無音区間（発話の切れ目）。カット点はこの終わりに寄る。
SILENCES: tuple[tuple[float, float], ...] = (
    (5.60, 5.95),    # アンカーA「このチャンネルは」の直前
    (12.40, 12.70),
    (19.70, 19.95),  # おとりの「ということで」の直前
    (25.40, 25.70),
    (33.60, 33.90),
    (39.80, 40.10),
    (43.60, 43.95),  # アンカーB「ということで、木原さん」の直前
    (51.20, 51.50),
)

#: (開始秒, 終了秒, 文) — 文字起こしの素。単語はこの中で等分される。
UTTERANCES: tuple[tuple[float, float, str], ...] = (
    (0.20, 5.60, "えーっと、じゃあ今日も始めていきます。"),
    (5.95, 12.40, "このチャンネルはAIの活用法を実験する番組です。"),
    (12.70, 19.70, "今日のテーマは議事録の自動化について話していきます。"),
    (19.95, 25.40, "ということで、まず前回のおさらいからいきましょう。"),
    (25.70, 33.60, "実はAIに議事録を書かせるのは一番もったいない使い方なんです。"),
    (33.90, 39.80, "理由は三つあって、要約の質が会議の質を超えられないからです。"),
    (40.10, 43.60, "だから議事録ではなく議題の設計に使うほうがいいんですね。"),
    (43.95, 51.20, "ということで、木原さん、今日はありがとうございました。"),
    (51.50, 58.60, "また次回もよろしくお願いします。"),
)

#: 期待値（テストが参照する）
EXPECTED_ANCHOR_A_RAW = 5.95
EXPECTED_ANCHOR_B_RAW = 43.95
EXPECTED_CUT_A = 5.90     # 無音の終わり - 50ms
EXPECTED_CUT_B = 43.90


# ---------------------------------------------------------------------------
# 文字起こし
# ---------------------------------------------------------------------------


def split_into_words(text: str, start: float, end: float, *, chunk: int = 3) -> list[Word]:
    """文を chunk 文字ずつの「単語」に割り、区間内で等分に時刻を振る。

    実際の ASR も単語境界は当てにならないので、文字数で等分するだけで十分に近い。
    """
    pieces = [text[i : i + chunk] for i in range(0, len(text), chunk)] or [text]
    span = (end - start) / len(pieces)
    words: list[Word] = []
    for i, piece in enumerate(pieces):
        w_start = round(start + i * span, 3)
        w_end = round(start + (i + 1) * span, 3)
        words.append(Word(word=piece, start=w_start, end=w_end))
    return words


def build_transcript(
    utterances: tuple[tuple[float, float, str], ...] = UTTERANCES,
    *,
    duration: float = EPISODE_DURATION,
    language: str = "ja",
) -> Transcript:
    """合成エピソードの文字起こしを作る。"""
    segments: list[Segment] = []
    for start, end, text in utterances:
        words = split_into_words(text, start, end)
        segments.append(Segment(start=start, end=end, text=text, words=words))
    return Transcript(language=language, duration=duration, segments=segments)


def word_start_of(transcript: Transcript, phrase: str, *, occurrence: int = 0) -> float:
    """生テキスト上で phrase が occurrence 番目に出る位置の単語 start を返す（期待値の検算用）。"""
    words = transcript.words()
    flat = "".join(w.word for w in words)
    # 文字index -> 単語index
    owner: list[int] = []
    for i, w in enumerate(words):
        owner.extend([i] * len(w.word))
    pos = -1
    for _ in range(occurrence + 1):
        pos = flat.find(phrase, pos + 1)
        if pos < 0:
            raise AssertionError(f"phrase が見つかりません: {phrase!r}")
    return words[owner[pos]].start


# ---------------------------------------------------------------------------
# 音声・動画
# ---------------------------------------------------------------------------


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def write_tone_wav(
    path: str | Path,
    *,
    duration: float = EPISODE_DURATION,
    silences: tuple[tuple[float, float], ...] = SILENCES,
    sample_rate: int = SAMPLE_RATE,
    freq: float = TONE_FREQ,
    amplitude: float = TONE_AMPLITUDE,
) -> Path:
    """指定区間だけ無音になる正弦波の 16bit モノラル WAV を書く。

    ffmpeg のフィルタ式で作るより、こちらのほうが境界が正確で読みやすい。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    total = int(duration * sample_rate)
    silence_samples = [(int(s * sample_rate), int(e * sample_rate)) for s, e in silences]

    frames = bytearray()
    two_pi_f = 2.0 * math.pi * freq
    cursor = 0
    for s, e in silence_samples:
        for n in range(cursor, min(s, total)):
            value = int(amplitude * 32767 * math.sin(two_pi_f * n / sample_rate))
            frames += struct.pack("<h", value)
        for _ in range(max(0, min(e, total) - min(s, total))):
            frames += struct.pack("<h", 0)
        cursor = max(cursor, min(e, total))
    for n in range(cursor, total):
        value = int(amplitude * 32767 * math.sin(two_pi_f * n / sample_rate))
        frames += struct.pack("<h", value)

    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return p


def build_test_video(
    path: str | Path,
    *,
    duration: float = EPISODE_DURATION,
    silences: tuple[tuple[float, float], ...] = SILENCES,
    size: str = "160x120",
    fps: int = 15,
) -> Path:
    """合成音声を持つ小さな mp4 を作る。ffmpeg が要る。

    映像はフレーム番号が焼かれた単色（カット位置の目視確認用に testsrc を使う）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wav = p.with_suffix(".source.wav")
    write_tone_wav(wav, duration=duration, silences=silences)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={duration}",
        "-i", str(wav),
        "-shortest",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"テスト動画の生成に失敗しました:\n{proc.stderr[-2000:]}")
    wav.unlink(missing_ok=True)
    return p


# ---------------------------------------------------------------------------
# LLM スタブの応答
# ---------------------------------------------------------------------------


def stub_highlight_response(*, start: float = 26.5, end: float = 33.0) -> dict:
    """Step 5 用のスタブ応答。文の途中で切れた秒数をわざと返す。"""
    return {
        "candidates": [
            {
                "start": start,
                "end": end,
                "score": 92,
                "hook_line": "実はAIに議事録を書かせるのは一番もったいない使い方なんです",
                "reason": "結論が先に来ていて、単体で意味が通る。逆説を含む。",
            },
            {
                "start": 34.5,
                "end": 39.0,
                "score": 80,
                "hook_line": "要約の質は会議の質を超えられない",
                "reason": "理由の説明として強い。",
            },
            {
                "start": 999.0,
                "end": 1020.0,
                "score": 95,
                "hook_line": "範囲外の候補",
                "reason": "本編の外にあるので破棄されるはず。",
            },
        ]
    }


def stub_metadata_response() -> dict:
    return {
        "summary_lead": "AIに議事録を書かせるのは、実はいちばんもったいない使い方でした。",
        "body": "今回は議事録の自動化をテーマに実験しました。\n\n結論から言うと、要約の質は会議の質を超えられません。\n\nだから使うべきは議題の設計のほうです。",
        "chapters": [
            {"time_sec": 0, "label": "今回の結論"},
            {"time_sec": 12, "label": "オープニング"},
            {"time_sec": 40, "label": "AI議事録の落とし穴"},
            {"time_sec": 44, "label": "議題の設計に使う"},
        ],
        "keywords": ["AI議事録", "文字起こし", "業務自動化", "会議の設計"],
    }


def stub_titles_response() -> dict:
    directions = (
        "結論直球型",
        "逆説・否定型",
        "数字型",
        "疑問型",
        "実験・検証型",
        "ターゲット明示型",
    )
    samples = {
        "結論直球型": [
            "AIに議事録を書かせるのは一番もったいない使い方でした",
            "議事録より議題の設計にAIを使うべき理由",
            "AI活用の正解は要約ではなく設計にありました",
            "会議の質がAIの要約の上限を決めていました",
            "議事録自動化より先にやるべきことがあります",
        ],
        "逆説・否定型": [
            "AI議事録はもうやめました。半年試した結論です",
            "実は逆効果。AIに議事録を任せて失われたもの",
            "文字起こしを増やすほど会議が雑になる話",
            "自動化してはいけない業務がひとつだけあります",
            "便利すぎるAI議事録が会議を壊していきました",
        ],
        "数字型": [
            "AI議事録で失われる3つのもの、実験で分かりました",
            "90%の人が知らないAI議事録の本当の使いどころ",
            "1ヶ月AI議事録を使って分かった2つの限界",
            "会議1回あたり40分を取り戻した設計の話",
            "3つの手順でAIを議題設計に回す方法",
        ],
        "疑問型": [
            "AIに議事録を書かせて、本当に楽になりましたか",
            "その議事録、後から誰か読み返していますか",
            "なぜAIの要約は会議の質を超えられないのか",
            "議事録の自動化、どこまでやれば十分なのか",
            "AIに任せる前に決めるべきことは何なのか",
        ],
        "実験・検証型": [
            "AI議事録を1ヶ月使い倒して分かったことを話します",
            "議事録の自動化を試してみた結果、方針を変えました",
            "会議20回分でAI要約の限界を検証してみました",
            "AIに議題設計をやらせる実験をしてみました",
            "文字起こしと要約、どちらが効いたか比べました",
        ],
        "ターゲット明示型": [
            "非エンジニアのためのAI議事録の使いどころ",
            "中小企業の会議でAIを使うならここから始める",
            "会議が多いチームリーダーのためのAI活用法",
            "情報共有に困っている現場のためのAI議事録入門",
            "はじめてAIを業務に入れる人のための実験ノート",
        ],
    }
    titles = []
    for direction in directions:
        for text in samples[direction]:
            titles.append({"direction": direction, "text": text})
    return {"titles": titles}


def stub_responses() -> dict:
    """StubLlmClient にそのまま渡せる応答表。"""
    return {
        "highlight": stub_highlight_response(),
        "metadata": stub_metadata_response(),
        "titles": stub_titles_response(),
    }
