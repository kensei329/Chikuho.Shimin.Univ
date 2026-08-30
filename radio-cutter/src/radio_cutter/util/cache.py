"""SPEC Step 2 の「キャッシュ」を担う（入力SHA-256とASR設定ハッシュでの再実行判定）。

方針（SPEC 6章 Step2）：
- 文字起こしは全工程の8割の時間を占めるため、同じ入力・同じASR設定なら丸ごとスキップする。
- 入力は60分の mp4（数GB）になりうるので、ハッシュは必ずチャンク読みで計算する。
- キャッシュは「壊れていたら使わない」だけでよい。壊れたキャッシュで実行を止めない。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any

from ..errors import RadioCutterError
from ..logging_util import get_logger

logger = get_logger(__name__)

#: 既定の読み込み単位（1MiB）。大きな mp4 を丸ごとメモリに載せないための上限。
DEFAULT_CHUNK_SIZE = 1 << 20

#: キャッシュが指す文字起こしファイルの既定名（work/<episode_id>/ からの相対）
DEFAULT_TRANSCRIPT_FILE = "transcript.json"


# ---------------------------------------------------------------------------
# ハッシュ
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """ファイルの SHA-256 を16進文字列で返す。

    60分の mp4 を丸ごとメモリに載せないよう `chunk_size`（既定1MiB）ずつ読む。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size は1以上にしてください（実際: {chunk_size}）。")

    p = Path(path)
    digest = hashlib.sha256()
    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    try:
        with p.open("rb") as fh:
            while True:
                read = fh.readinto(view)
                if not read:
                    break
                digest.update(view[:read])
    except FileNotFoundError as exc:
        raise RadioCutterError(f"ハッシュを計算するファイルが見つかりません: {p}") from exc
    except IsADirectoryError as exc:
        raise RadioCutterError(f"ハッシュを計算する対象がディレクトリです: {p}") from exc
    except OSError as exc:
        raise RadioCutterError(f"ファイルを読めませんでした: {p}\n{exc}") from exc

    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """JSON にできない値を安定した形に落とす（Path・集合・dataclass のみ）。

    集合はそのままだと順序が実行ごとに変わりキャッシュキーが不安定になるため、
    必ずソートしてから配列にする。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"JSON にできない値です: {type(value).__name__}")


def stable_hash(payload: Any) -> str:
    """任意の JSON 化できる値を、キーの並び順に依存しない SHA-256 にする。

    `sort_keys=True` と区切り文字の固定により、同じ内容なら常に同じキーになる。
    """
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise RadioCutterError(f"ハッシュ対象を JSON にできませんでした: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transcript_cache_key(input_sha: str, asr_payload: dict) -> str:
    """入力ファイルのSHA-256とASR設定から、文字起こしキャッシュのキーを作る。

    ASR のモデルや initial_prompt が変われば結果も変わるので、両方をキーに混ぜる。
    """
    return stable_hash({"input_sha256": str(input_sha), "asr": asr_payload})


# ---------------------------------------------------------------------------
# キャッシュファイル（work/<episode_id>/transcript_cache.json）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptCacheEntry:
    """文字起こしキャッシュの1件。どの入力・どのASR設定で作った transcript かを示す。"""

    input_sha256: str
    asr_hash: str
    key: str
    created_at: str                                     # ISO8601。呼び出し側から渡す
    transcript_file: str = DEFAULT_TRANSCRIPT_FILE

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_sha256": self.input_sha256,
            "asr_hash": self.asr_hash,
            "key": self.key,
            "created_at": self.created_at,
            "transcript_file": self.transcript_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptCacheEntry":
        """dict から復元する。必須項目が欠けていれば ValueError（呼び出し側が握る）。"""
        if not isinstance(d, dict):
            raise ValueError("キャッシュの中身が JSON オブジェクトではありません。")
        values: dict[str, str] = {}
        for field_name in ("input_sha256", "asr_hash", "key", "created_at"):
            raw = d.get(field_name)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"キャッシュに必須項目 '{field_name}' がありません（または空です）。")
            values[field_name] = raw
        transcript_file = d.get("transcript_file", DEFAULT_TRANSCRIPT_FILE)
        if not isinstance(transcript_file, str) or not transcript_file:
            raise ValueError("キャッシュの 'transcript_file' が不正です。")
        return cls(
            input_sha256=values["input_sha256"],
            asr_hash=values["asr_hash"],
            key=values["key"],
            created_at=values["created_at"],
            transcript_file=transcript_file,
        )

    def matches(self, key: str) -> bool:
        """このキャッシュが指定のキーと一致するか（Step 2 のスキップ判定用）。"""
        return bool(key) and self.key == key


def load_cache_entry(path: str | Path) -> TranscriptCacheEntry | None:
    """キャッシュファイルを読む。無い・壊れている場合は例外を投げず None を返す。

    キャッシュは「使えたら得をする」だけのものなので、読めないときは
    黙って文字起こしをやり直すのが正しい（実行を止めない）。
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("文字起こしキャッシュがありません: %s", p)
        return None
    except OSError as exc:
        logger.warning("文字起こしキャッシュを読めませんでした（無視します）: %s（%s）", p, exc)
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("文字起こしキャッシュの JSON が壊れています（無視します）: %s（%s）", p, exc)
        return None

    try:
        return TranscriptCacheEntry.from_dict(data)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("文字起こしキャッシュの内容が不正です（無視します）: %s（%s）", p, exc)
        return None


def save_cache_entry(path: str | Path, entry: TranscriptCacheEntry) -> None:
    """キャッシュファイルを書く。

    途中で落ちて中途半端なファイルが残らないよう、同じディレクトリの一時ファイルに
    書いてから置き換える（壊れたキャッシュを次回読ませないため）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entry.to_dict(), ensure_ascii=False, indent=2) + "\n"
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RadioCutterError(f"文字起こしキャッシュを書けませんでした: {p}\n{exc}") from exc
    logger.debug("文字起こしキャッシュを更新しました: %s（key=%s…）", p, entry.key[:12])
