"""LLM 層（`llm/client.py` と `llm/schemas.py`）の契約テスト。

守らせたいこと（SPEC 11章「プロンプトは .md に外出し」「返り値をスキーマ検証してからパースする」、
SPEC 9章「LLM が JSON 以外を返したら 3 回までリトライ」）:

- プロンプトは `.md` に外出しされていて、コードが埋める変数と過不足なく一致していること
- `{{key}}` の置換で JSON 例の `{ }` を壊さないこと。埋め残しは黙って通さないこと
- LLM の返り値は必ずスキーマ検証を通ること。落ちたときは「どのパスが悪いか」が分かること
- リトライは失敗理由を差し戻して行い、上限まで失敗したら LlmError で止まること
- スタブ応答も本番と同じスキーマ検証を通ること（スタブの取り違えをテストで検出するため）
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import fixtures
from radio_cutter.config import LlmConfig
from radio_cutter.errors import LlmError, RadioCutterError
from radio_cutter.llm import client as client_mod
from radio_cutter.llm.client import (
    AnthropicClient,
    LlmResponse,
    StubLlmClient,
    build_client,
    extract_json,
    load_prompt,
    load_stub_responses,
    render_prompt,
)
from radio_cutter.llm.schemas import (
    HIGHLIGHT_SCHEMA,
    METADATA_SCHEMA,
    TITLE_COUNT,
    TITLE_DIRECTIONS,
    TITLES_PER_DIRECTION,
    TITLES_SCHEMA,
    validate_payload,
)

# ---------------------------------------------------------------------------
# 契約（SPEC / 各ステップが埋める変数）
# ---------------------------------------------------------------------------

#: プロンプト名 → コード側が必ず埋める変数の集合。
#: s5_pick_highlight.build_prompt / s6_metadata の render_prompt 呼び出しと対になる。
PROMPT_CONTRACT: dict[str, set[str]] = {
    "highlight": {
        "channel",
        "num_candidates",
        "target_duration_sec",
        "min_duration_sec",
        "max_duration_sec",
        "main_start",
        "main_end",
        "transcript",
    },
    "metadata": {
        "channel",
        "final_duration",
        "highlight_duration",
        "hook_line",
        "transcript",
    },
    "titles": {
        "channel",
        "summary_lead",
        "keywords",
        "hook_line",
        "summary_body",
        "directions",
    },
}


def placeholders_in(template: str) -> set[str]:
    """テンプレート中の `{{name}}` を集める。"""
    return {m.group(1) for m in client_mod._PLACEHOLDER_RE.finditer(template)}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """リトライのバックオフでテストを待たせない。"""
    monkeypatch.setenv(client_mod.NO_SLEEP_ENV, "1")


# ---------------------------------------------------------------------------
# 正しいペイロード（各スキーマを通るもの）
# ---------------------------------------------------------------------------


def good_highlight() -> dict:
    return fixtures.stub_highlight_response()


def good_metadata() -> dict:
    return fixtures.stub_metadata_response()


def good_titles() -> dict:
    return fixtures.stub_titles_response()


# ---------------------------------------------------------------------------
# 偽の anthropic SDK（未インストール環境でも本番経路を通すため）
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    """anthropic の Message 相当。content はブロックの列。"""

    def __init__(
        self,
        content: list[Any],
        *,
        usage: FakeUsage | None = None,
        stop_reason: str = "tool_use",
    ) -> None:
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason


class FakeApiError(Exception):
    """HTTP ステータスを持つ SDK 例外の代役。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def tool_use_block(data: Any, *, name: str = "emit_result") -> dict:
    return {"type": "tool_use", "name": name, "input": data}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_message(data: Any, *, name: str = "emit_result", usage: FakeUsage | None = None) -> FakeMessage:
    return FakeMessage([tool_use_block(data, name=name)], usage=usage)


class FakeMessages:
    """`client.messages.create(**request)` の記録役。台本を頭から消費する。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError(
                f"想定より多く LLM が呼ばれました（{len(self.requests)} 回目）。"
            )
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeSdk:
    """差し込んだ偽 SDK の観測点。"""

    def __init__(self, messages: FakeMessages, created: list[Any]) -> None:
        self.messages = messages
        self.created = created

    @property
    def requests(self) -> list[dict]:
        return self.messages.requests


def install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
    *,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> FakeSdk:
    """`import anthropic` が偽モジュールを拾うようにする（この環境に本物は無い）。"""
    module = types.ModuleType("anthropic")
    messages = FakeMessages(script)
    created: list[Any] = []

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.init_kwargs = kwargs
            self.messages = messages
            created.append(self)

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv(api_key_env, "test-key")
    return FakeSdk(messages, created)


# ===========================================================================
# render_prompt
# ===========================================================================


class TestRenderPrompt:
    def test_プレースホルダを置換する(self) -> None:
        """`{{key}}` が値に置き換わり、他の文字は触られない。"""
        out = render_prompt("番組「{{channel}}」の{{n}}件です。", {"channel": "実験ラジオ", "n": 3})
        assert out == "番組「実験ラジオ」の3件です。"

    def test_前後の空白を許す(self) -> None:
        """`{{ key }}` のように空白が入っていても同じ変数として扱う。"""
        assert render_prompt("A{{ channel }}B", {"channel": "X"}) == "AXB"

    def test_JSON例の波括弧を壊さない(self) -> None:
        """str.format ではないので、プロンプト中の JSON 例の `{ }` はそのまま残る。

        ここが壊れると「出力形式の指示」がプロンプトから消えて品質が落ちる。
        """
        template = '例:\n```json\n{\n  "candidates": [\n    { "start": 1.0, "end": 2.0 }\n  ]\n}\n```\n対象: {{channel}}'
        out = render_prompt(template, {"channel": "ラジオ"})
        assert '{\n  "candidates": [' in out
        assert '{ "start": 1.0, "end": 2.0 }' in out
        assert out.endswith("対象: ラジオ")

    def test_単独の波括弧はプレースホルダではない(self) -> None:
        """`{name}` は置換対象ではない（二重括弧だけが変数）。"""
        assert render_prompt("{channel} と {{channel}}", {"channel": "X"}) == "{channel} と X"

    def test_数値も文字列化される(self) -> None:
        """int / float / bool を渡してもそのまま埋まる（コード側が str 化を強いられない）。"""
        out = render_prompt(
            "{{n}} / {{sec}} / {{flag}}", {"n": 3, "sec": 30.5, "flag": True}
        )
        assert out == "3 / 30.5 / True"

    def test_値が空文字でも通る(self) -> None:
        """空文字は「値が無い」ではない。埋め残し扱いにしてはいけない。"""
        assert render_prompt("[{{x}}]", {"x": ""}) == "[]"

    def test_同じ変数を何度でも使える(self) -> None:
        assert render_prompt("{{a}}-{{a}}-{{a}}", {"a": "z"}) == "z-z-z"

    def test_値が渡されていない変数はLlmError(self) -> None:
        """埋め残しを黙って投げない（SPEC 11章の「空欄のまま投げていた」事故防止）。"""
        with pytest.raises(LlmError) as ei:
            render_prompt("{{channel}} と {{missing}}", {"channel": "X"})
        msg = str(ei.value)
        assert "{{missing}}" in msg
        assert "channel" in msg  # 渡された変数も示して原因を切り分けられること

    def test_複数の埋め残しを全部報告する(self) -> None:
        with pytest.raises(LlmError) as ei:
            render_prompt("{{a}}{{b}}{{c}}", {"b": 1})
        msg = str(ei.value)
        assert "{{a}}" in msg and "{{c}}" in msg

    def test_変数が空でも埋め残しなら止まる(self) -> None:
        with pytest.raises(LlmError):
            render_prompt("{{a}}", {})

    def test_プレースホルダが無ければそのまま返す(self) -> None:
        assert render_prompt("ただの文章。", {}) == "ただの文章。"

    def test_壊れた二重波括弧を検出する(self) -> None:
        """`{{}}` のように変数名として解釈できない `{{` は素通ししない。"""
        with pytest.raises(LlmError) as ei:
            render_prompt("これは {{}} です", {})
        assert "{{" in str(ei.value)

    def test_値に二重波括弧が含まれても誤検知しない(self) -> None:
        """判定はテンプレート側の位置で行うので、値の中身は影響しない。"""
        out = render_prompt("[{{x}}]", {"x": "{{not_a_placeholder}}"})
        assert out == "[{{not_a_placeholder}}]"

    def test_置換結果は再帰的に展開されない(self) -> None:
        """値の中の `{{y}}` を二次展開しない（プロンプトインジェクション的な事故を防ぐ）。"""
        out = render_prompt("{{x}}", {"x": "{{y}}", "y": "展開されるべきでない"})
        assert out == "{{y}}"

    def test_使われない変数があっても例外にはしない(self) -> None:
        """余った変数は警告どまり。プロンプト側を先に直せる余地を残す。"""
        assert render_prompt("{{a}}", {"a": "1", "unused": "2"}) == "1"


# ===========================================================================
# load_prompt
# ===========================================================================


class TestLoadPrompt:
    @pytest.mark.parametrize("name", sorted(PROMPT_CONTRACT))
    def test_拡張子の有無どちらでも読める(self, name: str) -> None:
        """"highlight" でも "highlight.md" でも同じ中身が返る。"""
        without = load_prompt(name)
        with_ext = load_prompt(f"{name}.md")
        assert without == with_ext
        assert without.strip(), f"{name}.md が空です"

    @pytest.mark.parametrize("name,expected", sorted(PROMPT_CONTRACT.items()))
    def test_契約どおりのプレースホルダを持つ(self, name: str, expected: set[str]) -> None:
        """プロンプトの変数とコードが埋める変数が過不足なく一致していること。

        片方だけ増えると「空欄のまま投げる」か「LlmError で止まる」かのどちらかになる。
        """
        assert placeholders_in(load_prompt(name)) == expected

    @pytest.mark.parametrize("name,expected", sorted(PROMPT_CONTRACT.items()))
    def test_契約の変数だけで描画できる(self, name: str, expected: set[str]) -> None:
        """契約どおりの変数を渡せば埋め残しゼロで描画できる。"""
        rendered = render_prompt(load_prompt(name), {k: f"<{k}>" for k in expected})
        assert "{{" not in rendered
        for key in expected:
            assert f"<{key}>" in rendered

    def test_プロンプトはコードに埋め込まれずmdにある(self) -> None:
        """SPEC 11章。3枚とも `llm/prompts/` から読めること。"""
        pkg_dir = Path(client_mod.__file__).resolve().parent
        for name in PROMPT_CONTRACT:
            assert (pkg_dir / client_mod.PROMPT_DIR / f"{name}.md").is_file()

    def test_出力形式のJSON例が残っている(self) -> None:
        """スキーマを tool で強制していても、プロンプト側の JSON 例は生きていること。"""
        for name in PROMPT_CONTRACT:
            assert "```json" in load_prompt(name), f"{name}.md に JSON 例がありません"

    def test_存在しない名前はLlmError(self) -> None:
        with pytest.raises(LlmError) as ei:
            load_prompt("no_such_prompt")
        msg = str(ei.value)
        assert "no_such_prompt.md" in msg

    @pytest.mark.parametrize("bad", ["../secrets", "sub/highlight", "a\\b", "", "   "])
    def test_ファイル名以外は受け付けない(self, bad: str) -> None:
        """パスを渡してパッケージ外を読ませない。"""
        with pytest.raises(LlmError):
            load_prompt(bad)

    def test_titlesプロンプトは6方向を一字一句持つ(self) -> None:
        """SPEC 6-b の6方向。ここがずれると enum 違反でリトライを消費する。"""
        text = load_prompt("titles")
        for direction in TITLE_DIRECTIONS:
            assert direction in text, f"titles.md に {direction} がありません"


# ===========================================================================
# schemas: 定数
# ===========================================================================


class TestSchemaConstants:
    def test_6方向5個で30個(self) -> None:
        """SPEC 6-b「30個を6方向 × 5個で生成させる」。"""
        assert len(TITLE_DIRECTIONS) == 6
        assert TITLES_PER_DIRECTION == 5
        assert TITLE_COUNT == 30

    def test_方向の表記はSPECの表どおり(self) -> None:
        assert TITLE_DIRECTIONS == (
            "結論直球型",
            "逆説・否定型",
            "数字型",
            "疑問型",
            "実験・検証型",
            "ターゲット明示型",
        )

    @pytest.mark.parametrize(
        "schema", [HIGHLIGHT_SCHEMA, METADATA_SCHEMA, TITLES_SCHEMA], ids=["highlight", "metadata", "titles"]
    )
    def test_余計なキーを許さない形になっている(self, schema: dict) -> None:
        """schemas.py の宣言どおり additionalProperties:false と required を明示すること。

        tool の input_schema にそのまま渡すため、ここが緩いとゴミが混ざったまま通る。
        """
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]

    @pytest.mark.parametrize(
        "schema", [HIGHLIGHT_SCHEMA, METADATA_SCHEMA, TITLES_SCHEMA], ids=["highlight", "metadata", "titles"]
    )
    def test_スキーマ自体がDraft202012として妥当(self, schema: dict) -> None:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)


# ===========================================================================
# validate_payload: highlight
# ===========================================================================


class TestValidateHighlight:
    def test_正しい応答は通る(self) -> None:
        validate_payload(good_highlight(), HIGHLIGHT_SCHEMA, where="t")

    def test_候補1件でも通る(self) -> None:
        payload = {"candidates": [good_highlight()["candidates"][0]]}
        validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")

    def test_candidatesが無いと落ちる(self) -> None:
        with pytest.raises(LlmError) as ei:
            validate_payload({}, HIGHLIGHT_SCHEMA, where="t")
        assert "candidates" in str(ei.value)

    def test_候補が0件だと落ちる(self) -> None:
        """1件も返さないなら Step 5 は成立しない。"""
        with pytest.raises(LlmError):
            validate_payload({"candidates": []}, HIGHLIGHT_SCHEMA, where="t")

    def test_余分なキーは落ちる(self) -> None:
        payload = good_highlight()
        payload["note"] = "余計な説明"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")
        assert "note" in str(ei.value)

    def test_候補の中の余分なキーも落ちる(self) -> None:
        payload = good_highlight()
        payload["candidates"][0]["confidence"] = 0.9
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")
        assert "confidence" in str(ei.value)

    @pytest.mark.parametrize("key", ["start", "end", "score", "hook_line", "reason"])
    def test_必須キーが欠けると落ちる(self, key: str) -> None:
        payload = good_highlight()
        del payload["candidates"][0][key]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")
        assert key in str(ei.value)

    def test_秒数が文字列だと落ちる(self) -> None:
        """SPEC 11章「秒数は全て float」。"842.5" は数値ではない。"""
        payload = good_highlight()
        payload["candidates"][0]["start"] = "26.5"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")
        msg = str(ei.value)
        assert "$.candidates[0].start" in msg
        assert "number" in msg

    def test_負の秒数は落ちる(self) -> None:
        payload = good_highlight()
        payload["candidates"][0]["start"] = -1.0
        with pytest.raises(LlmError):
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")

    @pytest.mark.parametrize("score", [-1, 101, 1000])
    def test_scoreが0から100の外だと落ちる(self, score: float) -> None:
        payload = good_highlight()
        payload["candidates"][0]["score"] = score
        with pytest.raises(LlmError):
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")

    def test_hook_lineが文字列でないと落ちる(self) -> None:
        payload = good_highlight()
        payload["candidates"][0]["hook_line"] = ["a", "b"]
        with pytest.raises(LlmError):
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")

    def test_candidatesが配列でないと落ちる(self) -> None:
        with pytest.raises(LlmError):
            validate_payload({"candidates": {"start": 1.0}}, HIGHLIGHT_SCHEMA, where="t")

    def test_トップレベルがオブジェクトでないと落ちる(self) -> None:
        with pytest.raises(LlmError):
            validate_payload([{"start": 1.0}], HIGHLIGHT_SCHEMA, where="t")


# ===========================================================================
# validate_payload: metadata
# ===========================================================================


class TestValidateMetadata:
    def test_正しい応答は通る(self) -> None:
        validate_payload(good_metadata(), METADATA_SCHEMA, where="t")

    @pytest.mark.parametrize("key", ["summary_lead", "body", "chapters", "keywords"])
    def test_必須キーが欠けると落ちる(self, key: str) -> None:
        payload = good_metadata()
        del payload[key]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert key in str(ei.value)

    def test_余分なキーは落ちる(self) -> None:
        """概要欄の最終フォーマットはコード側で組む（SPEC Step 6-a）。

        LLM が description ごと返してきたら受け取らない。
        """
        payload = good_metadata()
        payload["description"] = "全部作ってみました"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert "description" in str(ei.value)

    def test_チャプターの余分なキーも落ちる(self) -> None:
        payload = good_metadata()
        payload["chapters"][0]["time"] = "0:00"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert "time" in str(ei.value)

    def test_time_secが文字列だと落ちる(self) -> None:
        """`"0:00"` のような表示用文字列を受け取らない（変換はコード側の仕事）。"""
        payload = good_metadata()
        payload["chapters"][0]["time_sec"] = "0:00"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert "$.chapters[0].time_sec" in str(ei.value)

    def test_time_secが負だと落ちる(self) -> None:
        payload = good_metadata()
        payload["chapters"][1]["time_sec"] = -5
        with pytest.raises(LlmError):
            validate_payload(payload, METADATA_SCHEMA, where="t")

    def test_ラベルが空文字だと落ちる(self) -> None:
        payload = good_metadata()
        payload["chapters"][0]["label"] = ""
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert "$.chapters[0].label" in str(ei.value)

    def test_チャプターが0件だと落ちる(self) -> None:
        payload = good_metadata()
        payload["chapters"] = []
        with pytest.raises(LlmError):
            validate_payload(payload, METADATA_SCHEMA, where="t")

    def test_キーワードが文字列の配列でないと落ちる(self) -> None:
        payload = good_metadata()
        payload["keywords"] = ["AI議事録", 42]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, METADATA_SCHEMA, where="t")
        assert "$.keywords[1]" in str(ei.value)


# ===========================================================================
# validate_payload: titles
# ===========================================================================


class TestValidateTitles:
    def test_正しい応答は通る(self) -> None:
        validate_payload(good_titles(), TITLES_SCHEMA, where="t")

    def test_ちょうど30件でないと落ちる_足りない(self) -> None:
        """SPEC 6-b「30個を6方向 × 5個」。29個は不可。"""
        payload = good_titles()
        payload["titles"] = payload["titles"][:29]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        assert "titles" in str(ei.value)

    def test_ちょうど30件でないと落ちる_多い(self) -> None:
        payload = good_titles()
        payload["titles"] = payload["titles"] + [{"direction": "数字型", "text": "31本目"}]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        assert "titles" in str(ei.value)

    def test_方向が6語のどれでもないと落ちる(self) -> None:
        """`逆説型` のような表記ゆれを通さない（titles.md も一字一句と指示している）。"""
        payload = good_titles()
        payload["titles"][5]["direction"] = "逆説型"
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        msg = str(ei.value)
        assert "$.titles[5].direction" in msg

    def test_タイトル本文が空だと落ちる(self) -> None:
        payload = good_titles()
        payload["titles"][0]["text"] = ""
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        assert "$.titles[0].text" in str(ei.value)

    @pytest.mark.parametrize("key", ["direction", "text"])
    def test_必須キーが欠けると落ちる(self, key: str) -> None:
        payload = good_titles()
        del payload["titles"][3][key]
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        assert key in str(ei.value)

    def test_余分なキーは落ちる(self) -> None:
        """文字数の併記はコード側で数える（SPEC 6-b）。LLM に持たせない。"""
        payload = good_titles()
        payload["titles"][0]["length"] = 26
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        assert "length" in str(ei.value)

    def test_titlesが無いと落ちる(self) -> None:
        with pytest.raises(LlmError) as ei:
            validate_payload({}, TITLES_SCHEMA, where="t")
        assert "titles" in str(ei.value)


# ===========================================================================
# validate_payload: エラーメッセージの質
# ===========================================================================


class TestValidateMessages:
    def test_whereがメッセージに載る(self) -> None:
        """どのステップの何回目で落ちたかを人が追えること。"""
        with pytest.raises(LlmError) as ei:
            validate_payload({}, HIGHLIGHT_SCHEMA, where="highlight / 2回目")
        assert "highlight / 2回目" in str(ei.value)

    def test_複数の違反をまとめて報告する(self) -> None:
        """リトライ時にそのまま差し戻せるよう、違反は1件で打ち切らない。"""
        payload = {
            "candidates": [
                {"start": "x", "end": "y", "score": 500, "hook_line": 1, "reason": None}
            ]
        }
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, HIGHLIGHT_SCHEMA, where="t")
        msg = str(ei.value)
        for path in ("$.candidates[0].start", "$.candidates[0].end", "$.candidates[0].score"):
            assert path in msg, f"{path} が報告されていません:\n{msg}"

    def test_違反が多いときは件数を添えて打ち切る(self) -> None:
        """全部並べるとプロンプトが膨らむので上限を設けつつ、隠さず件数を出す。"""
        payload = {"titles": [{"direction": "x", "text": ""} for _ in range(TITLE_COUNT)]}
        with pytest.raises(LlmError) as ei:
            validate_payload(payload, TITLES_SCHEMA, where="t")
        msg = str(ei.value)
        assert "ほか" in msg
        assert len([ln for ln in msg.splitlines() if ln.startswith("  - ")]) <= (
            client_mod_max_reported() + 1
        )

    def test_スキーマ定義が壊れている場合は文言を分ける(self) -> None:
        """モデルのせいではないので、リトライさせずに開発者に向けて言う。"""
        with pytest.raises(LlmError) as ei:
            validate_payload({}, {"type": "nonsense"}, where="t")
        assert "スキーマの定義そのもの" in str(ei.value)

    def test_LlmErrorはRadioCutterErrorである(self) -> None:
        """CLI が一括で捕まえて終了コード1にできること。"""
        assert issubclass(LlmError, RadioCutterError)


def client_mod_max_reported() -> int:
    from radio_cutter.llm.schemas import MAX_REPORTED_ERRORS

    return MAX_REPORTED_ERRORS


# ===========================================================================
# extract_json
# ===========================================================================


class TestExtractJson:
    def test_jsonフェンスから取り出す(self) -> None:
        text = 'はい、こちらです。\n```json\n{"a": 1}\n```\n以上です。'
        assert extract_json(text) == {"a": 1}

    def test_フェンス無しでも波括弧から取り出す(self) -> None:
        assert extract_json('前置き {"a": [1, 2]} 後置き') == {"a": [1, 2]}

    def test_入れ子のオブジェクトを最後まで取る(self) -> None:
        data = {"candidates": [{"start": 1.0, "end": 2.0}]}
        assert extract_json("説明\n" + json.dumps(data) + "\n終わり") == data

    def test_JSONが無ければLlmError(self) -> None:
        with pytest.raises(LlmError) as ei:
            extract_json("申し訳ありませんが、お答えできません。")
        assert "申し訳ありません" in str(ei.value)  # 応答の冒頭を見せて原因を追えること

    def test_配列だけの応答はLlmError(self) -> None:
        """トップレベルはオブジェクトであることを前提にしている。"""
        with pytest.raises(LlmError):
            extract_json("[1, 2, 3]")


# ===========================================================================
# StubLlmClient
# ===========================================================================


class TestStubLlmClient:
    def test_用意した応答を返す(self) -> None:
        stub = StubLlmClient({"highlight": good_highlight()})
        res = stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert isinstance(res, LlmResponse)
        assert res.data == good_highlight()
        assert res.retries == 0
        assert res.input_tokens == 0 and res.output_tokens == 0

    def test_呼び出しを記録する(self) -> None:
        """どのプロンプトが投げられたかをテストから覗けること。"""
        stub = StubLlmClient({"highlight": good_highlight()})
        stub.complete_json(step="highlight", prompt="こんにちは", schema=HIGHLIGHT_SCHEMA)
        assert len(stub.calls) == 1
        assert stub.calls[0]["step"] == "highlight"
        assert stub.calls[0]["prompt"] == "こんにちは"

    def test_返り値を書き換えても元の応答は汚れない(self) -> None:
        """同じスタブを2ステップで使い回しても互いに干渉しないこと。"""
        stub = StubLlmClient({"highlight": good_highlight()})
        first = stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        first.data["candidates"][0]["score"] = 0
        second = stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert second.data["candidates"][0]["score"] == 92

    def test_未知のstepはLlmError(self) -> None:
        stub = StubLlmClient({"highlight": good_highlight()})
        with pytest.raises(LlmError) as ei:
            stub.complete_json(step="metadata", prompt="P", schema=METADATA_SCHEMA)
        msg = str(ei.value)
        assert "metadata" in msg
        assert "highlight" in msg  # 用意されている step を案内すること

    def test_応答が空でも未知のstepとして案内する(self) -> None:
        stub = StubLlmClient({})
        with pytest.raises(LlmError) as ei:
            stub.complete_json(step="titles", prompt="P", schema=TITLES_SCHEMA)
        assert "titles" in str(ei.value)

    def test_callableな応答を受け付ける(self) -> None:
        """プロンプトの中身で応答を変えたいテスト用。"""
        seen: list[str] = []

        def respond(prompt: str) -> dict:
            seen.append(prompt)
            return good_metadata()

        stub = StubLlmClient({"metadata": respond})
        res = stub.complete_json(step="metadata", prompt="本文...", schema=METADATA_SCHEMA)
        assert seen == ["本文..."]
        assert res.data == good_metadata()

    def test_callableがdictでなければLlmError(self) -> None:
        stub = StubLlmClient({"metadata": lambda prompt: "文字列を返してしまった"})
        with pytest.raises(LlmError) as ei:
            stub.complete_json(step="metadata", prompt="P", schema=METADATA_SCHEMA)
        assert "str" in str(ei.value)

    def test_スタブの返り値もスキーマ検証される(self) -> None:
        """本番と同じ検証を通す。スタブの取り違えをここで落とす。"""
        broken = {"candidates": [{"start": 1.0}]}
        stub = StubLlmClient({"highlight": broken})
        with pytest.raises(LlmError) as ei:
            stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert "スタブ応答" in str(ei.value)

    def test_スキーマ違いのスタブを渡すと落ちる(self) -> None:
        """metadata の応答を highlight のスキーマで検証したら通ってはいけない。"""
        stub = StubLlmClient({"highlight": good_metadata()})
        with pytest.raises(LlmError):
            stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)

    def test_渡した応答表を後から変えても影響しない(self) -> None:
        """コンストラクタで dict をコピーしていること。"""
        table: dict = {"highlight": good_highlight()}
        stub = StubLlmClient(table)
        table.clear()
        assert stub.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA).data

    def test_modelを持つ(self) -> None:
        """decisions.json の llm_calls に載るので空にしない。"""
        stub = StubLlmClient({}, model="stub-model")
        assert stub.model == "stub-model"

    def test_fixturesのスタブ応答は3つとも本物のスキーマを通る(self) -> None:
        """tests/fixtures.py の合成応答が SPEC の形からずれていないこと。"""
        stub = StubLlmClient(fixtures.stub_responses())
        for step, schema in (
            ("highlight", HIGHLIGHT_SCHEMA),
            ("metadata", METADATA_SCHEMA),
            ("titles", TITLES_SCHEMA),
        ):
            res = stub.complete_json(step=step, prompt="P", schema=schema)
            assert res.data

    def test_conftestのstub_llmフィクスチャも同じく通る(self, stub_llm: StubLlmClient) -> None:
        res = stub_llm.complete_json(step="titles", prompt="P", schema=TITLES_SCHEMA)
        assert len(res.data["titles"]) == TITLE_COUNT


# ===========================================================================
# build_client
# ===========================================================================


class TestBuildClient:
    def test_stub_responsesがあればスタブを返す(self) -> None:
        cfg = LlmConfig()
        c = build_client(cfg, stub_responses=fixtures.stub_responses())
        assert isinstance(c, StubLlmClient)
        assert cfg.model in c.model  # 本来使うはずのモデル名が追える形であること

    def test_空のstub_responsesでもスタブを返す(self) -> None:
        """`--stub-llm` に空の応答表を渡したときに本番 API を叩いてしまわないこと。"""
        assert isinstance(build_client(LlmConfig(), stub_responses={}), StubLlmClient)

    def test_anthropicならAnthropicClient(self) -> None:
        cfg = LlmConfig(provider="anthropic", model="claude-sonnet-4-6")
        c = build_client(cfg)
        assert isinstance(c, AnthropicClient)
        assert c.model == "claude-sonnet-4-6"

    @pytest.mark.parametrize("provider", ["Anthropic", " ANTHROPIC ", "anthropic"])
    def test_プロバイダ名は大小文字と空白を吸収する(self, provider: str) -> None:
        assert isinstance(build_client(LlmConfig(provider=provider)), AnthropicClient)

    @pytest.mark.parametrize("provider", ["openai", "", "  ", "claude"])
    def test_未知のプロバイダはLlmError(self, provider: str) -> None:
        with pytest.raises(LlmError) as ei:
            build_client(LlmConfig(provider=provider))
        msg = str(ei.value)
        assert "anthropic" in msg  # 直し方を案内すること

    def test_未知のプロバイダ名がメッセージに出る(self) -> None:
        with pytest.raises(LlmError) as ei:
            build_client(LlmConfig(provider="openai"))
        assert "openai" in str(ei.value)

    def test_configのllm設定からそのまま作れる(self, config) -> None:
        """同梱 config/ai-radio.json の llm 設定で作れること。"""
        assert isinstance(build_client(config.llm), AnthropicClient)


# ===========================================================================
# load_stub_responses
# ===========================================================================


class TestLoadStubResponses:
    def test_conftestのファイルを読める(self, stub_llm_file: Path) -> None:
        data = load_stub_responses(stub_llm_file)
        assert set(data) == {"highlight", "metadata", "titles"}

    def test_読んだ応答はそのままスタブに渡せる(self, stub_llm_file: Path) -> None:
        client = build_client(LlmConfig(), stub_responses=load_stub_responses(stub_llm_file))
        res = client.complete_json(step="metadata", prompt="P", schema=METADATA_SCHEMA)
        assert res.data["chapters"][0]["time_sec"] == 0

    def test_ファイルが無ければLlmError(self, tmp_path: Path) -> None:
        with pytest.raises(LlmError) as ei:
            load_stub_responses(tmp_path / "no-such.json")
        assert "no-such.json" in str(ei.value)

    def test_JSONが壊れていればLlmError(self, tmp_path: Path) -> None:
        p = tmp_path / "broken.json"
        p.write_text("{ではない", encoding="utf-8")
        with pytest.raises(LlmError) as ei:
            load_stub_responses(p)
        assert str(p) in str(ei.value)

    def test_オブジェクトでなければLlmError(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(LlmError):
            load_stub_responses(p)


# ===========================================================================
# AnthropicClient: 準備段階
# ===========================================================================


class TestAnthropicSetup:
    def test_SDKが無ければ次の一手を書いて止まる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """anthropic 未インストール環境で「何をすればいいか」が分かること。"""
        monkeypatch.setitem(sys.modules, "anthropic", None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        c = AnthropicClient(LlmConfig())
        with pytest.raises(LlmError) as ei:
            c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        msg = str(ei.value)
        assert "anthropic" in msg
        assert "pip install" in msg

    def test_APIキーが無ければ環境変数名を出して止まる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_anthropic(monkeypatch, [], api_key_env="MY_KEY")
        monkeypatch.delenv("MY_KEY", raising=False)
        c = AnthropicClient(LlmConfig(api_key_env="MY_KEY"))
        with pytest.raises(LlmError) as ei:
            c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert "MY_KEY" in str(ei.value)

    def test_SDKクライアントは使い回す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """呼び出しごとに接続を作り直さないこと。"""
        fake = install_fake_anthropic(
            monkeypatch, [tool_message(good_highlight()), tool_message(good_highlight())]
        )
        c = AnthropicClient(LlmConfig())
        c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert len(fake.created) == 1

    def test_タイムアウトとAPIキーをSDKに渡す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        c = AnthropicClient(LlmConfig(timeout_sec=12.5))
        c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA)
        assert fake.created[0].init_kwargs["api_key"] == "test-key"
        assert fake.created[0].init_kwargs["timeout"] == 12.5


# ===========================================================================
# AnthropicClient: リクエストの形
# ===========================================================================


class TestAnthropicRequest:
    def test_スキーマをtoolのinput_schemaで渡す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEC 11章「必ず JSON Schema かそれに準ずる出力形式指定を渡す」。"""
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        req = fake.requests[0]
        assert req["tools"][0]["input_schema"] == HIGHLIGHT_SCHEMA
        assert req["tools"][0]["name"] == "emit_result"
        assert req["tool_choice"] == {"type": "tool", "name": "emit_result"}

    def test_モデルとプロンプトを渡す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig(model="claude-test")).complete_json(
            step="highlight", prompt="本編の文字起こし", schema=HIGHLIGHT_SCHEMA
        )
        req = fake.requests[0]
        assert req["model"] == "claude-test"
        assert req["messages"] == [{"role": "user", "content": "本編の文字起こし"}]

    def test_systemを渡せる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA, system="あなたは編集者です"
        )
        assert fake.requests[0]["system"] == "あなたは編集者です"

    def test_systemを渡さなければキー自体を送らない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert "system" not in fake.requests[0]

    def test_max_tokensは呼び出し側で上書きできる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig(max_tokens=8000)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA, max_tokens=1234
        )
        assert fake.requests[0]["max_tokens"] == 1234

    def test_temperatureは既定値なら送らない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """新しいモデルが受け付けないため、1.0 のままなら省く。"""
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig(temperature=1.0)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert "temperature" not in fake.requests[0]

    def test_temperatureを変えたら送る(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig(temperature=0.2)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert fake.requests[0]["temperature"] == 0.2

    def test_tool_nameを変えられる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight(), name="emit_x")])
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA, tool_name="emit_x"
        )
        assert fake.requests[0]["tool_choice"]["name"] == "emit_x"
        assert res.data == good_highlight()


# ===========================================================================
# AnthropicClient: 応答の取り出し
# ===========================================================================


class TestAnthropicPayload:
    def test_tool_useのinputを使う(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_anthropic(monkeypatch, [tool_message(good_metadata())])
        res = AnthropicClient(LlmConfig()).complete_json(
            step="metadata", prompt="P", schema=METADATA_SCHEMA
        )
        assert res.data == good_metadata()

    def test_inputがJSON文字列でもパースする(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_anthropic(
            monkeypatch,
            [FakeMessage([tool_use_block(json.dumps(good_highlight(), ensure_ascii=False))])],
        )
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.data == good_highlight()

    def test_名前の一致するtool_useを優先する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """複数の tool_use が返っても、指定した名前のものを使うこと。"""
        other = {"candidates": [{"start": 0.0, "end": 1.0, "score": 1, "hook_line": "x", "reason": "y"}]}
        install_fake_anthropic(
            monkeypatch,
            [
                FakeMessage(
                    [
                        tool_use_block(other, name="something_else"),
                        tool_use_block(good_highlight(), name="emit_result"),
                    ]
                )
            ],
        )
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.data == good_highlight()

    def test_tool_useが無ければテキストから拾う(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEC 9章「LLM が JSON 以外を返す」への保険。フェンス付きでも拾えること。"""
        body = "承知しました。\n```json\n" + json.dumps(good_highlight(), ensure_ascii=False) + "\n```"
        install_fake_anthropic(monkeypatch, [FakeMessage([text_block(body)], stop_reason="end_turn")])
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.data == good_highlight()

    def test_中身が空ならstop_reasonを添えて報告する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_anthropic(
            monkeypatch,
            [
                FakeMessage([], stop_reason="max_tokens"),
                FakeMessage([], stop_reason="max_tokens"),
                FakeMessage([], stop_reason="max_tokens"),
            ],
        )
        with pytest.raises(LlmError) as ei:
            AnthropicClient(LlmConfig(max_retries=3)).complete_json(
                step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
            )
        assert "max_tokens" in str(ei.value)

    def test_トークン数をLlmResponseに載せる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """decisions.json の llm_calls に input_tokens を書くため（SPEC 8章）。"""
        install_fake_anthropic(
            monkeypatch,
            [tool_message(good_highlight(), usage=FakeUsage(input_tokens=24810, output_tokens=512))],
        )
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.input_tokens == 24810
        assert res.output_tokens == 512
        assert res.retries == 0

    def test_usageが無くても0で埋める(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        res = AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.input_tokens == 0 and res.output_tokens == 0


# ===========================================================================
# AnthropicClient: リトライ（SPEC 9章）
# ===========================================================================


class TestAnthropicRetry:
    def test_スキーマ違反なら1回やり直して通る(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """1回目がスキーマ違反、2回目が正しいなら retries=1 で成功すること。"""
        bad = {"candidates": [{"start": "x", "end": 2.0, "score": 90, "hook_line": "a", "reason": "b"}]}
        fake = install_fake_anthropic(
            monkeypatch, [tool_message(bad), tool_message(good_highlight())]
        )
        res = AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="元のプロンプト", schema=HIGHLIGHT_SCHEMA
        )
        assert res.retries == 1
        assert res.data == good_highlight()
        assert len(fake.requests) == 2

    def test_やり直しには失敗理由を添える(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """同じ失敗を繰り返させないため、直前の違反箇所を差し戻すこと。"""
        bad = {"candidates": [{"start": "x", "end": 2.0, "score": 90, "hook_line": "a", "reason": "b"}]}
        fake = install_fake_anthropic(
            monkeypatch, [tool_message(bad), tool_message(good_highlight())]
        )
        AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="元のプロンプト", schema=HIGHLIGHT_SCHEMA
        )
        retry_prompt = fake.requests[1]["messages"][0]["content"]
        assert "元のプロンプト" in retry_prompt
        assert client_mod.RETRY_HEADER in retry_prompt
        assert "$.candidates[0].start" in retry_prompt

    def test_成功したら余計に呼ばない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert len(fake.requests) == 1

    def test_max_retries回すべて失敗したらLlmError(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEC 9章「3回までリトライ。失敗したらそのステップだけ落とす」。"""
        bad = {"candidates": []}
        fake = install_fake_anthropic(monkeypatch, [tool_message(bad) for _ in range(3)])
        with pytest.raises(LlmError) as ei:
            AnthropicClient(LlmConfig(max_retries=3)).complete_json(
                step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
            )
        msg = str(ei.value)
        assert len(fake.requests) == 3
        assert "3" in msg
        assert "highlight" in msg
        assert "$.candidates" in msg  # 最後のエラーの中身も残す

    def test_max_retriesが1なら1回だけ呼ぶ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = install_fake_anthropic(monkeypatch, [tool_message({"candidates": []})])
        with pytest.raises(LlmError):
            AnthropicClient(LlmConfig(max_retries=1)).complete_json(
                step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
            )
        assert len(fake.requests) == 1

    def test_JSONでない応答もリトライ対象(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEC 9章「LLM が JSON 以外を返す → 3回までリトライ」。"""
        fake = install_fake_anthropic(
            monkeypatch,
            [
                FakeMessage([text_block("すみません、JSONは出せません。")], stop_reason="end_turn"),
                tool_message(good_highlight()),
            ],
        )
        res = AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.retries == 1
        assert len(fake.requests) == 2

    def test_API例外はリトライする(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """一時的な障害（5xx / タイムアウト）は握らずに再試行する。"""
        fake = install_fake_anthropic(
            monkeypatch,
            [FakeApiError("overloaded", status_code=529), tool_message(good_highlight())],
        )
        res = AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert res.retries == 1
        assert len(fake.requests) == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
    def test_回復不能なHTTPエラーはリトライしない(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        """設定ミス・リクエスト不正は何度やっても同じ。無駄に課金しない。"""
        fake = install_fake_anthropic(
            monkeypatch, [FakeApiError("bad request", status_code=status)]
        )
        with pytest.raises(LlmError) as ei:
            AnthropicClient(LlmConfig(max_retries=3)).complete_json(
                step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
            )
        assert len(fake.requests) == 1
        assert str(status) in str(ei.value)

    def test_リトライ中はバックオフを挟む(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """待ち時間は指数的に増える（テストでは環境変数で実際には眠らない）。"""
        slept: list[float] = []
        monkeypatch.delenv(client_mod.NO_SLEEP_ENV, raising=False)
        monkeypatch.setattr(client_mod.time, "sleep", lambda s: slept.append(s))
        install_fake_anthropic(
            monkeypatch,
            [tool_message({"candidates": []}), tool_message({"candidates": []}), tool_message(good_highlight())],
        )
        AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert slept == [
            client_mod.RETRY_BACKOFF_BASE_SEC,
            client_mod.RETRY_BACKOFF_BASE_SEC * 2,
        ]

    def test_NO_SLEEP環境変数で待たない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr(client_mod.time, "sleep", lambda s: slept.append(s))
        install_fake_anthropic(
            monkeypatch, [tool_message({"candidates": []}), tool_message(good_highlight())]
        )
        AnthropicClient(LlmConfig(max_retries=3)).complete_json(
            step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA
        )
        assert slept == []

    def test_3ステップとも同じ経路で通る(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """highlight / metadata / titles の3スキーマが本番経路で受理されること。"""
        install_fake_anthropic(
            monkeypatch,
            [
                tool_message(good_highlight()),
                tool_message(good_metadata()),
                tool_message(good_titles()),
            ],
        )
        c = AnthropicClient(LlmConfig())
        assert c.complete_json(step="highlight", prompt="P", schema=HIGHLIGHT_SCHEMA).data
        assert c.complete_json(step="metadata", prompt="P", schema=METADATA_SCHEMA).data
        titles = c.complete_json(step="titles", prompt="P", schema=TITLES_SCHEMA).data
        assert len(titles["titles"]) == TITLE_COUNT


# ===========================================================================
# エンドツーエンド寄り: プロンプト → 呼び出し
# ===========================================================================


class TestPromptToCall:
    def test_描画したプロンプトをそのまま投げられる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`.md` から読んだプロンプトが、埋め残しなく API に届くこと。"""
        fake = install_fake_anthropic(monkeypatch, [tool_message(good_highlight())])
        prompt = render_prompt(
            load_prompt("highlight"),
            {
                "channel": "AI活用法実験ラジオ",
                "num_candidates": 3,
                "target_duration_sec": 30,
                "min_duration_sec": 20,
                "max_duration_sec": 45,
                "main_start": "5.9",
                "main_end": "43.9",
                "transcript": "[5.9] このチャンネルはAIの活用法を実験する番組です。",
            },
        )
        AnthropicClient(LlmConfig()).complete_json(
            step="highlight", prompt=prompt, schema=HIGHLIGHT_SCHEMA
        )
        sent = fake.requests[0]["messages"][0]["content"]
        assert "{{" not in sent
        assert "AI活用法実験ラジオ" in sent
        assert "[5.9] このチャンネルは" in sent
        assert re.search(r'"candidates"', sent), "JSON 例が壊れています"
