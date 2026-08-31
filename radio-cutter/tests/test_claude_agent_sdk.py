"""このパソコンの Claude Code を呼ぶ側（ClaudeAgentSdkClient）の検証。

本物の `claude` コマンドは呼ばない。偽の claude_agent_sdk を差し込んで、
「JSON の取り出し」「投げ直し」「落ちたときの案内」を確かめる。

API 版と違ってツール呼び出しで JSON を強制できないので、
本文から JSON を拾ってスキーマ検証する経路がここの生命線になる。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from radio_cutter.config import LlmConfig
from radio_cutter.errors import LlmError
from radio_cutter.llm.client import ClaudeAgentSdkClient
from radio_cutter.llm.schemas import HIGHLIGHT_SCHEMA

VALID = {
    "candidates": [
        {
            "start": 10.0,
            "end": 40.0,
            "score": 90,
            "hook_line": "結論から言うと、議事録は要らない",
            "reason": "単体で意味が通る。",
        }
    ]
}


# ---------------------------------------------------------------------------
# 偽の claude_agent_sdk
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list[Any]


@dataclass
class FakeResultMessage:
    usage: dict = field(default_factory=dict)
    is_error: bool = False
    result: str = ""
    errors: Any = None


class FakeCLINotFoundError(Exception):
    pass


class FakeCLIConnectionError(Exception):
    pass


def install_fake_sdk(monkeypatch: pytest.MonkeyPatch, turns: list[Any]) -> dict:
    """`turns` の各要素を1回の呼び出しの応答として返す偽 SDK を入れる。

    各要素は文字列（本文）か、送出する例外インスタンス。
    """
    calls: dict[str, Any] = {"prompts": [], "options": [], "count": 0}
    module = types.ModuleType("claude_agent_sdk")

    class FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    async def fake_query(*, prompt: str, options: Any = None, **_: Any):
        index = calls["count"]
        calls["count"] += 1
        calls["prompts"].append(prompt)
        calls["options"].append(options)
        turn = turns[min(index, len(turns) - 1)]
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, FakeResultMessage):
            yield turn
            return
        yield FakeAssistantMessage(content=[FakeTextBlock(text=str(turn))])
        yield FakeResultMessage(usage={"input_tokens": 11, "output_tokens": 22})

    module.query = fake_query
    module.ClaudeAgentOptions = FakeOptions
    module.AssistantMessage = FakeAssistantMessage
    module.TextBlock = FakeTextBlock
    module.ResultMessage = FakeResultMessage
    module.CLINotFoundError = FakeCLINotFoundError
    module.CLIConnectionError = FakeCLIConnectionError
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    monkeypatch.setenv("RADIO_CUTTER_NO_SLEEP", "1")
    return calls


def make_client(**kwargs: Any) -> ClaudeAgentSdkClient:
    cfg = LlmConfig(provider="claude_agent_sdk", model=kwargs.pop("model", "opus"), **kwargs)
    return ClaudeAgentSdkClient(cfg)


def ask(client: ClaudeAgentSdkClient):
    return client.complete_json(step="highlight", prompt="候補を出して", schema=HIGHLIGHT_SCHEMA)


# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_素のJSONを受け取れる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        install_fake_sdk(monkeypatch, [json.dumps(VALID, ensure_ascii=False)])
        res = ask(make_client())
        assert res.data == VALID
        assert res.retries == 0

    def test_コードフェンス付きでも取り出せる(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        body = "はい、こちらです。\n```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```\n"
        install_fake_sdk(monkeypatch, [body])
        assert ask(make_client()).data == VALID

    def test_トークン数を記録する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """decisions.json の llm_calls に載る値。"""
        import json

        install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        res = ask(make_client())
        assert (res.input_tokens, res.output_tokens) == (11, 22)

    def test_APIキーは見ない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Claude Code のログインを使うので、環境変数が無くても動くこと。"""
        import json

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        assert ask(make_client()).data == VALID


class TestOptions:
    def test_道具を一切使わせない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """文章を書かせるだけなので、ファイル読み書きなどはさせない。"""
        import json

        calls = install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        ask(make_client())
        opts = calls["options"][0]
        assert opts.allowed_tools == []
        assert opts.max_turns == 1

    def test_使う人の設定を読み込まない(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLAUDE.md や個人設定を読むと、同じ入力でも出力が変わってしまう。"""
        import json

        calls = install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        ask(make_client())
        assert calls["options"][0].setting_sources == []

    def test_systemにスキーマを渡す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ツールで形を強制できないぶん、スキーマ本体を見せて縛る。"""
        import json

        calls = install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        ask(make_client())
        system = calls["options"][0].system_prompt
        assert "candidates" in system
        assert "JSON" in system

    def test_モデル名がそのまま渡る(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        calls = install_fake_sdk(monkeypatch, [json.dumps(VALID)])
        ask(make_client(model="haiku"))
        assert calls["options"][0].model == "haiku"


class TestRetry:
    def test_JSONでなければ投げ直す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        calls = install_fake_sdk(
            monkeypatch, ["すみません、よく分かりません。", json.dumps(VALID)]
        )
        res = ask(make_client(max_retries=3))
        assert res.retries == 1
        assert calls["count"] == 2

    def test_スキーマに合わなければ投げ直す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        calls = install_fake_sdk(
            monkeypatch, [json.dumps({"candidates": []}), json.dumps(VALID)]
        )
        assert ask(make_client(max_retries=3)).retries == 1
        assert calls["count"] == 2

    def test_投げ直すときは理由を添える(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """同じ失敗を繰り返させないため。"""
        import json

        calls = install_fake_sdk(monkeypatch, ["だめな返事", json.dumps(VALID)])
        ask(make_client(max_retries=3))
        assert "受理できませんでした" in calls["prompts"][1]

    def test_回数を使い切ったらLlmError(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = install_fake_sdk(monkeypatch, ["だめ", "だめ", "だめ", "だめ"])
        with pytest.raises(LlmError) as ei:
            ask(make_client(max_retries=3))
        assert calls["count"] == 3
        assert "3 回" in str(ei.value)

    def test_max_retriesが1なら1回だけ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = install_fake_sdk(monkeypatch, ["だめ", "だめ"])
        with pytest.raises(LlmError):
            ask(make_client(max_retries=1))
        assert calls["count"] == 1


class TestErrors:
    def test_ClaudeCodeが無いときは入れ方を案内する(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_sdk(monkeypatch, [FakeCLINotFoundError("not found")])
        with pytest.raises(LlmError) as ei:
            ask(make_client(max_retries=1))
        message = str(ei.value)
        assert "claude" in message.lower()
        assert "ログイン" in message

    def test_接続できないときも次の一手を出す(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_sdk(monkeypatch, [FakeCLIConnectionError("boom")])
        with pytest.raises(LlmError) as ei:
            ask(make_client(max_retries=1))
        assert "確認" in str(ei.value)

    def test_ClaudeCodeがエラーを返したらLlmError(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_sdk(
            monkeypatch, [FakeResultMessage(is_error=True, result="利用上限に達しました")]
        )
        with pytest.raises(LlmError) as ei:
            ask(make_client(max_retries=1))
        assert "利用上限" in str(ei.value)

    def test_SDKが入っていなければ入れ方を案内する(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any):
            if name == "claude_agent_sdk":
                raise ImportError("No module named 'claude_agent_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(LlmError) as ei:
            ask(make_client(max_retries=1))
        message = str(ei.value)
        assert "claude-agent-sdk" in message
        assert "Claude Code" in message
