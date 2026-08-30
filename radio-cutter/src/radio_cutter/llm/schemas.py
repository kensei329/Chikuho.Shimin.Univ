"""LLM に返させる JSON の形（SPEC Step 5 / Step 6-a / Step 6-b の「期待するJSON」）と検証。

SPEC 11章「LLMには必ず JSON Schema かそれに準ずる出力形式指定を渡し、
返り値をスキーマ検証してからパースする」をここで担保する。
スキーマは Anthropic の tool の input_schema にそのまま渡すため、
`additionalProperties: false` と `required` を必ず明示する（余計なキーを混ぜさせない）。
"""

from __future__ import annotations

from typing import Any

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from ..errors import LlmError

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

#: タイトルの方向性（SPEC 6-b の表）。この6方向 × 5個 = 30個を生成させる。
TITLE_DIRECTIONS: tuple[str, ...] = (
    "結論直球型",
    "逆説・否定型",
    "数字型",
    "疑問型",
    "実験・検証型",
    "ターゲット明示型",
)

#: 1方向あたりの本数と合計本数（SPEC 6-b）。
TITLES_PER_DIRECTION = 5
TITLE_COUNT = len(TITLE_DIRECTIONS) * TITLES_PER_DIRECTION  # 30

#: エラーメッセージに並べる違反の最大件数（全部出すと読めなくなるため）
MAX_REPORTED_ERRORS = 8


# ---------------------------------------------------------------------------
# スキーマ
# ---------------------------------------------------------------------------

#: Step 5 ハイライト候補。候補は最低1件（SPEC では3件返させる）。
HIGHLIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": 1,
            "description": "ハイライト候補。スコアの高い順に並べる。",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "score", "hook_line", "reason"],
                "properties": {
                    "start": {
                        "type": "number",
                        "minimum": 0,
                        "description": "元動画の絶対秒での開始時刻。",
                    },
                    "end": {
                        "type": "number",
                        "minimum": 0,
                        "description": "元動画の絶対秒での終了時刻。start より後にすること。",
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "フックとしての強さ（0〜100）。",
                    },
                    "hook_line": {
                        "type": "string",
                        "description": "この区間の中で最も強い一文。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "なぜこの区間を選んだかの理由。",
                    },
                },
            },
        }
    },
}

#: Step 6-a 概要欄とチャプター。time_sec は final.mp4 のタイムライン上の秒。
METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary_lead", "body", "chapters", "keywords"],
    "properties": {
        "summary_lead": {
            "type": "string",
            "description": "「もっと見る」の前に出る2〜3行のリード文。",
        },
        "body": {
            "type": "string",
            "description": "概要欄の本文。3〜5段落。",
        },
        "chapters": {
            "type": "array",
            "minItems": 1,
            "description": "final.mp4 のタイムライン上のチャプター。昇順、先頭は 0。",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time_sec", "label"],
                "properties": {
                    "time_sec": {
                        "type": "number",
                        "minimum": 0,
                        "description": "final.mp4 の先頭からの秒数。",
                    },
                    "label": {
                        "type": "string",
                        "minLength": 1,
                        "description": "チャプター名。",
                    },
                },
            },
        },
        "keywords": {
            "type": "array",
            "description": "検索キーワード。",
            "items": {"type": "string"},
        },
    },
}

#: Step 6-b タイトル候補。6方向 × 5個 = ちょうど30個。
TITLES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["titles"],
    "properties": {
        "titles": {
            "type": "array",
            "minItems": TITLE_COUNT,
            "maxItems": TITLE_COUNT,
            "description": f"タイトル候補ちょうど{TITLE_COUNT}件。各方向 {TITLES_PER_DIRECTION} 件ずつ。",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["direction", "text"],
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": list(TITLE_DIRECTIONS),
                        "description": "タイトルの方向性。",
                    },
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "タイトル本文。絵文字は使わない。",
                    },
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# 検証
# ---------------------------------------------------------------------------


def _describe(error: ValidationError) -> str:
    """1件の違反を「どのパスがどう違うか」の1行にする。"""
    path = error.json_path or "$"
    return f"{path}: {error.message}"


def validate_payload(data: Any, schema: dict, *, where: str) -> None:
    """LLM が返した値をスキーマ検証する。合わなければ LlmError で止める。

    リトライ時にそのままモデルへ差し戻せるよう、違反箇所（json_path）と
    理由を全部メッセージに書き出す。
    """
    try:
        jsonschema.validate(instance=data, schema=schema, cls=Draft202012Validator)
    except SchemaError as exc:
        # こちらのスキーマ定義が壊れている。モデルのせいではないので文言を分ける。
        raise LlmError(
            f"JSON スキーマの定義そのものが不正です（{where}）: {exc.message}"
        ) from exc
    except ValidationError as first:
        try:
            errors = sorted(
                Draft202012Validator(schema).iter_errors(data),
                key=lambda e: (str(list(e.absolute_path)), e.message),
            )
        except Exception:  # noqa: BLE001 - 列挙に失敗しても最初の1件は必ず報告する
            errors = []
        lines = [_describe(e) for e in errors] or [_describe(first)]
        shown = lines[:MAX_REPORTED_ERRORS]
        if len(lines) > MAX_REPORTED_ERRORS:
            shown.append(f"（ほか {len(lines) - MAX_REPORTED_ERRORS} 件）")
        detail = "\n".join(f"  - {line}" for line in shown)
        raise LlmError(
            f"LLM の出力が期待する JSON スキーマに合っていません（{where}）。\n"
            f"合っていない箇所:\n{detail}"
        ) from first
