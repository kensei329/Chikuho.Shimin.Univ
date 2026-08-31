"""LLM クライアント（SPEC Step 5 / Step 6、および 11章「プロンプトは .md に外出し」）。

役割は3つだけ:
- `llm/prompts/*.md` を読んで `{{name}}` を埋める
- スキーマを tool として渡し、JSON を強制して受け取る
- スキーマに合わなければリトライし、それでも駄目なら LlmError で止める（SPEC 9章）
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import LlmConfig
from ..errors import LlmError
from ..logging_util import get_logger
from .schemas import validate_payload

logger = get_logger("llm.client")

#: プロンプトを置くパッケージとサブディレクトリ
PROMPT_PACKAGE = "radio_cutter.llm"
PROMPT_DIR = "prompts"

#: リトライ時の待ち時間（0.5s, 1s, 2s, ...）
RETRY_BACKOFF_BASE_SEC = 0.5

#: `time.sleep` を飛ばすための環境変数（テスト用）
NO_SLEEP_ENV = "RADIO_CUTTER_NO_SLEEP"

#: リトライ時にプロンプトへ足す前置き
RETRY_HEADER = "前回の出力は次の理由で受理できませんでした:"

#: 差し戻すエラー本文の最大長（プロンプトを膨らませないため）
MAX_RETRY_ERROR_CHARS = 2000

#: tool の説明文（JSON 以外を書かせないための念押し）
TOOL_DESCRIPTION = (
    "指定された JSON スキーマちょうどの形で結果を返す。"
    "前置きや説明文は書かず、必ずこのツールを1回だけ呼び出すこと。"
)

#: HTTP ステータスのうちリトライしても直らないもの（設定ミス・リクエスト不正）
FATAL_STATUS_CODES = (400, 401, 403, 404, 413)

#: config の llm.provider に書ける別名 → 実際のプロバイダ名
PROVIDER_ALIASES = {
    "claude_agent_sdk": "claude_agent_sdk",
    "claude-agent-sdk": "claude_agent_sdk",
    "claude_code": "claude_agent_sdk",
    "claude-code": "claude_agent_sdk",
    "local": "claude_agent_sdk",
    "sdk": "claude_agent_sdk",
    "anthropic": "anthropic",
    "anthropic_api": "anthropic",
    "api": "anthropic",
}

#: Claude Agent SDK に渡す既定のモデル名（"opus" / "sonnet" / "haiku" の別名が使える）
DEFAULT_SDK_MODEL = "opus"

#: Claude Agent SDK に「JSON だけ返せ」と言うための前置き
SDK_SYSTEM_HEADER = (
    "あなたは JSON だけを返します。前置き・説明文・コードフェンス・"
    "「はい」などの返事を一切書かないでください。"
    "返すのは次の JSON Schema にちょうど合うオブジェクト1つだけです。"
)

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_][A-Za-z0-9_.\-]*)\s*\}\}")
_FENCE_RE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# 返り値
# ---------------------------------------------------------------------------


@dataclass
class LlmResponse:
    """1回の LLM 呼び出しの結果。decisions.json の llm_calls に載る値も持つ。"""

    data: dict
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    """`llm/prompts/<name>.md` を読む。zip 配布でも動くよう importlib.resources を使う。

    name は "highlight" でも "highlight.md" でも受け付ける。
    """
    filename = name if name.endswith(".md") else f"{name}.md"
    if not filename.strip() or "/" in filename or "\\" in filename or ".." in filename:
        raise LlmError(f"プロンプト名が不正です: {name!r}（ファイル名だけを指定してください）。")
    try:
        resource = resources.files(PROMPT_PACKAGE).joinpath(PROMPT_DIR, filename)
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise LlmError(
            f"プロンプトファイルが見つかりません: {PROMPT_DIR}/{filename}\n"
            f"（{PROMPT_PACKAGE} パッケージの {PROMPT_DIR}/ に置いてください）"
        ) from exc
    except OSError as exc:
        raise LlmError(f"プロンプトファイルを読めませんでした: {PROMPT_DIR}/{filename}\n{exc}") from exc


def render_prompt(template: str, variables: Mapping[str, Any]) -> str:
    """`{{key}}` を str(value) に置き換える。

    str.format は使わない（プロンプト中の JSON 例の `{ }` が壊れるため）。
    埋め残しは黙って通さず LlmError にする。プロンプトの変数名を変えたときに
    「なぜか空欄のまま投げていた」を防ぐのが目的。
    """
    used: set[str] = set()
    missing: list[str] = []
    matched_starts: set[int] = set()

    def _replace(match: re.Match[str]) -> str:
        matched_starts.add(match.start())
        key = match.group(1)
        if key not in variables:
            if key not in missing:
                missing.append(key)
            return match.group(0)
        used.add(key)
        return str(variables[key])

    rendered = _PLACEHOLDER_RE.sub(_replace, template)

    # 置換できなかった `{{` を検出する。値そのものに `{{` が含まれていても
    # 誤検知しないよう、判定はテンプレート側の位置で行う。
    malformed = [
        template[pos : pos + 24]
        for pos in (m.start() for m in re.finditer(r"\{\{", template))
        if pos not in matched_starts
    ]

    if missing or malformed:
        parts = ["プロンプトのプレースホルダを埋められませんでした。"]
        if missing:
            parts.append("値が渡されていない変数: " + ", ".join(f"{{{{{k}}}}}" for k in missing))
        if malformed:
            parts.append("プレースホルダとして解釈できない `{{` があります: " + " / ".join(malformed))
        parts.append("渡された変数: " + (", ".join(sorted(variables)) if variables else "（なし）"))
        raise LlmError("\n".join(parts))

    unused = sorted(set(variables) - used)
    if unused:
        logger.warning("プロンプトで使われていない変数があります: %s", ", ".join(unused))
    return rendered


# ---------------------------------------------------------------------------
# 応答から JSON を取り出す
# ---------------------------------------------------------------------------


def _block_get(block: Any, key: str) -> Any:
    """SDK のオブジェクトでも dict でも同じように値を取る（テストしやすさのため）。"""
    if isinstance(block, Mapping):
        return block.get(key)
    return getattr(block, key, None)


def extract_json(text: str) -> dict:
    """テキストから JSON オブジェクトを抜き出す。tool_use が無かったときの保険。

    ```json フェンス → 最初の `{` から最後の `}` の順に試す。
    """
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(text)]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = text.strip()[:300]
    raise LlmError(f"応答から JSON を取り出せませんでした。応答の冒頭:\n{preview}")


def _sleep(seconds: float) -> None:
    """指数バックオフの待ち。テストでは環境変数で飛ばす。"""
    if os.environ.get(NO_SLEEP_ENV):
        return
    time.sleep(seconds)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _run_async(coro: Any) -> Any:
    """非同期の呼び出しを同期のコードから使う。

    パイプラインは同期なので普段は `asyncio.run` で足りるが、
    すでにイベントループが回っている場所（ノートブック等）から呼ばれても
    落ちないよう、その場合は別スレッドで新しいループを回す。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - 呼び出し元に投げ直す
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


# ---------------------------------------------------------------------------
# クライアント
# ---------------------------------------------------------------------------


class LlmClient:
    """LLM クライアントの抽象基底。ステップ側はこの型だけに依存する。"""

    model: str = ""

    def complete_json(
        self,
        *,
        step: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "emit_result",
        max_tokens: int | None = None,
    ) -> LlmResponse:
        """プロンプトを投げ、schema に合う JSON を返す。合わなければ LlmError。"""
        raise NotImplementedError


class AnthropicClient(LlmClient):
    """Anthropic API を叩く実装。tool_choice で JSON を強制する。

    `anthropic` は任意依存なので import はメソッド内（遅延 import）で行う。
    """

    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self._sdk: Any = None
        self._client: Any = None

    # ----- 準備 -----

    def _ensure_client(self) -> Any:
        """SDK と API キーを用意する。足りなければ次の一手を書いて止める。"""
        if self._client is not None:
            return self._client
        try:
            import anthropic  # 遅延 import（LLM を使わない実行では不要）
        except ImportError as exc:
            raise LlmError(
                "anthropic SDK が見つかりません。"
                "pip install 'radio-cutter[llm]' を実行してください。"
            ) from exc

        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise LlmError(
                f"環境変数 {self.cfg.api_key_env} に API キーが設定されていません。"
                f"export {self.cfg.api_key_env}=... を実行してから再実行してください。"
            )

        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self.cfg.timeout_sec)
        return self._client

    # ----- 呼び出し -----

    def complete_json(
        self,
        *,
        step: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "emit_result",
        max_tokens: int | None = None,
    ) -> LlmResponse:
        """schema を tool の input_schema にして JSON を強制し、検証して返す。"""
        client = self._ensure_client()
        attempts = max(1, int(self.cfg.max_retries))
        tools = [
            {
                "name": tool_name,
                "description": TOOL_DESCRIPTION,
                "input_schema": schema,
            }
        ]
        request: dict[str, Any] = {
            "model": self.cfg.model,
            "max_tokens": int(max_tokens or self.cfg.max_tokens),
            "tools": tools,
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if system:
            request["system"] = system
        # temperature は新しいモデルでは受け付けないため、既定値のままなら送らない。
        if self.cfg.temperature != 1.0:
            request["temperature"] = self.cfg.temperature

        last_error = ""
        for attempt in range(attempts):
            current_prompt = prompt
            if attempt > 0:
                _sleep(RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
                current_prompt = self._retry_prompt(prompt, last_error, tool_name)
                logger.warning(
                    "LLM（%s）を再試行します（%d/%d 回目）。直前の失敗: %s",
                    step,
                    attempt + 1,
                    attempts,
                    last_error.splitlines()[0] if last_error else "不明",
                )

            request["messages"] = [{"role": "user", "content": current_prompt}]
            logger.debug(
                "LLM 呼び出し step=%s model=%s prompt=%d文字 attempt=%d",
                step,
                self.cfg.model,
                len(current_prompt),
                attempt + 1,
            )

            try:
                message = client.messages.create(**request)
            except Exception as exc:  # noqa: BLE001 - SDK の例外型に依存しない
                status = getattr(exc, "status_code", None)
                if isinstance(status, int) and status in FATAL_STATUS_CODES:
                    raise LlmError(
                        f"LLM（{step}）の呼び出しが回復不能なエラーで失敗しました"
                        f"（HTTP {status}）: {exc}"
                    ) from exc
                last_error = f"API 呼び出しが失敗しました: {exc}"
                logger.warning("LLM（%s）の呼び出しに失敗しました: %s", step, exc)
                continue

            try:
                payload = self._extract_payload(message, tool_name=tool_name)
                validate_payload(payload, schema, where=f"{step} / {attempt + 1}回目")
            except LlmError as exc:
                last_error = str(exc)
                continue

            usage = getattr(message, "usage", None)
            return LlmResponse(
                data=payload,
                input_tokens=_as_int(_block_get(usage, "input_tokens")),
                output_tokens=_as_int(_block_get(usage, "output_tokens")),
                retries=attempt,
            )

        raise LlmError(
            f"LLM（{step}）が {attempts} 回の試行すべてで受理できる JSON を返しませんでした。\n"
            f"最後のエラー:\n{last_error}"
        )

    # ----- 補助 -----

    @staticmethod
    def _retry_prompt(prompt: str, last_error: str, tool_name: str) -> str:
        """前回の失敗理由を足したプロンプトを作る。同じ失敗を繰り返させないため。"""
        detail = (last_error or "不明なエラー")[:MAX_RETRY_ERROR_CHARS]
        return (
            f"{prompt}\n\n"
            "---\n"
            f"{RETRY_HEADER}\n{detail}\n\n"
            f"指定された JSON スキーマちょうどの形にして、{tool_name} ツールを1回だけ呼び出してください。"
        )

    @staticmethod
    def _extract_payload(message: Any, *, tool_name: str) -> dict:
        """応答から JSON を取り出す。tool_use を優先し、無ければテキストから拾う。"""
        blocks = list(getattr(message, "content", None) or [])

        fallback_block: dict | None = None
        for block in blocks:
            if _block_get(block, "type") != "tool_use":
                continue
            data = _block_get(block, "input")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, dict):
                continue
            if _block_get(block, "name") == tool_name:
                return data
            if fallback_block is None:
                fallback_block = data
        if fallback_block is not None:
            logger.warning("想定と違う名前の tool_use が返りました。中身をそのまま使います。")
            return fallback_block

        texts = [
            str(_block_get(block, "text") or "")
            for block in blocks
            if _block_get(block, "type") == "text"
        ]
        joined = "\n".join(t for t in texts if t)
        if not joined.strip():
            stop_reason = getattr(message, "stop_reason", None)
            raise LlmError(
                "応答に tool_use ブロックもテキストもありませんでした"
                f"（stop_reason={stop_reason!r}）。"
            )
        logger.warning("tool_use が返らなかったため、テキストから JSON を抜き出します。")
        return extract_json(joined)


class ClaudeAgentSdkClient(LlmClient):
    """このパソコンに入っている Claude Code を呼ぶ（APIキー不要）。

    Claude Agent SDK は Claude Code 本体を部品として呼び出す作りなので、
    認証は Claude Code のログイン（サブスクリプション）をそのまま使う。
    `ANTHROPIC_API_KEY` は要らないし、見ない。

    API 版と違って「ツール呼び出しで JSON を強制する」口が無いので、
    プロンプトと system で JSON だけを書かせ、返ってきた本文から
    `extract_json` で取り出して `validate_payload` にかける。
    形が合わなければ理由を添えて投げ直す（回数は API 版と同じ）。
    """

    def __init__(self, cfg: LlmConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model or DEFAULT_SDK_MODEL

    # ----- 呼び出し -----

    def complete_json(
        self,
        *,
        step: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "emit_result",
        max_tokens: int | None = None,
    ) -> LlmResponse:
        attempts = max(1, int(self.cfg.max_retries))
        current = prompt
        last_error = "不明なエラー"
        usage: dict[str, Any] = {}

        for attempt in range(attempts):
            if attempt > 0:
                _sleep(RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
                current = self._retry_prompt(prompt, last_error)
                logger.warning(
                    "LLM（%s）を投げ直します（%d/%d回目）: %s",
                    step, attempt + 1, attempts, last_error[:200],
                )
            logger.info(
                "Claude Code に問い合わせます step=%s model=%s prompt=%d文字（%d/%d回目）",
                step, self.model, len(current), attempt + 1, attempts,
            )
            try:
                text, usage = self._ask(current, system=system, schema=schema)
                payload = extract_json(text)
                validate_payload(payload, schema, where=f"{step} / {attempt + 1}回目")
            except LlmError as exc:
                last_error = str(exc)
                continue

            return LlmResponse(
                data=payload,
                input_tokens=_as_int(usage.get("input_tokens")),
                output_tokens=_as_int(usage.get("output_tokens")),
                retries=attempt,
            )

        raise LlmError(
            f"LLM（{step}）が {attempts} 回の試行すべてで受理できる JSON を返しませんでした。\n"
            f"最後のエラー: {last_error}"
        )

    # ----- 内部 -----

    @staticmethod
    def _retry_prompt(prompt: str, last_error: str) -> str:
        """前回の失敗理由を足したプロンプトを作る。同じ失敗を繰り返させないため。"""
        detail = (last_error or "不明なエラー")[:MAX_RETRY_ERROR_CHARS]
        return (
            f"{prompt}\n\n"
            "---\n"
            f"{RETRY_HEADER}\n{detail}\n\n"
            "スキーマちょうどの JSON オブジェクトだけを返してください。"
            "前置きも説明文もコードフェンスも書かないでください。"
        )

    def _system_prompt(self, schema: dict, system: str | None) -> str:
        """JSON だけを書かせるための system。スキーマ本体も渡す。"""
        parts = [SDK_SYSTEM_HEADER, json.dumps(schema, ensure_ascii=False, indent=2)]
        if system:
            parts.append(system)
        return "\n\n".join(parts)

    def _options(self, schema: dict, system: str | None) -> Any:
        """Claude Agent SDK に渡す設定。道具は一切使わせない。

        `setting_sources=[]` にしているのは、使う人の CLAUDE.md や設定を読み込むと
        同じ入力でも出力が変わってしまうため。ここは純粋な文章生成として使う。
        """
        sdk = self._import_sdk()
        return sdk.ClaudeAgentOptions(
            system_prompt=self._system_prompt(schema, system),
            model=self.model,
            max_turns=1,
            allowed_tools=[],
            setting_sources=[],
            permission_mode="dontAsk",
        )

    @staticmethod
    def _import_sdk() -> Any:
        """遅延 import。入っていなければ次の一手を書いて止める。"""
        try:
            import claude_agent_sdk
        except ImportError as exc:
            raise LlmError(
                "claude-agent-sdk が見つかりません。\n"
                "  pip install 'radio-cutter[llm]' を実行してください。\n"
                "  あわせて Claude Code 本体（https://claude.com/claude-code）を入れ、"
                "`claude` コマンドでログインしておく必要があります。"
            ) from exc
        return claude_agent_sdk

    def _ask(self, prompt: str, *, system: str | None, schema: dict) -> tuple[str, dict]:
        """Claude Code に1回投げて、本文と usage を返す。"""
        sdk = self._import_sdk()
        options = self._options(schema, system)

        async def run() -> tuple[str, dict]:
            chunks: list[str] = []
            usage: dict[str, Any] = {}
            async for message in sdk.query(prompt=prompt, options=options):
                if isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            chunks.append(block.text)
                elif isinstance(message, sdk.ResultMessage):
                    raw = getattr(message, "usage", None)
                    if isinstance(raw, dict):
                        usage = raw
                    if getattr(message, "is_error", False):
                        detail = getattr(message, "result", None) or getattr(message, "errors", "")
                        raise LlmError(f"Claude Code がエラーを返しました: {detail}")
            return ("".join(chunks), usage)

        try:
            return _run_async(run())
        except LlmError:
            raise
        except Exception as exc:  # SDK 側の例外はここで日本語にして返す
            raise LlmError(self._friendly_error(sdk, exc)) from exc

    @staticmethod
    def _friendly_error(sdk: Any, exc: Exception) -> str:
        """SDK の例外を、次に何をすればいいか分かる文言にする。"""
        not_found = getattr(sdk, "CLINotFoundError", None)
        if not_found is not None and isinstance(exc, not_found):
            return (
                "Claude Code（`claude` コマンド）が見つかりません。\n"
                "  https://claude.com/claude-code から入れて、一度 `claude` を起動して"
                "ログインしてください。"
            )
        connection = getattr(sdk, "CLIConnectionError", None)
        if connection is not None and isinstance(exc, connection):
            return (
                f"Claude Code に接続できませんでした: {exc}\n"
                "  ターミナルで `claude` が起動するか、ログイン済みかを確認してください。"
            )
        return f"Claude Code の呼び出しに失敗しました: {type(exc).__name__}: {exc}"


class StubLlmClient(LlmClient):
    """テストと `--stub-llm` 用。step 名で決め打ちの JSON を返す。

    返り値も本番と同じスキーマ検証に通す（スタブの取り違えをテストで検出するため）。
    """

    def __init__(
        self,
        responses: dict[str, dict | Callable[[str], dict]],
        *,
        model: str = "stub",
    ) -> None:
        self.responses: dict[str, dict | Callable[[str], dict]] = dict(responses)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        step: str,
        prompt: str,
        schema: dict,
        system: str | None = None,
        tool_name: str = "emit_result",
        max_tokens: int | None = None,
    ) -> LlmResponse:
        """用意した応答を返す。step が無ければ LlmError。"""
        self.calls.append({"step": step, "prompt": prompt, "tool_name": tool_name})
        if step not in self.responses:
            available = ", ".join(sorted(self.responses)) or "（なし）"
            raise LlmError(
                f"スタブ応答に step '{step}' がありません。用意されているのは: {available}"
            )
        entry = self.responses[step]
        data = entry(prompt) if callable(entry) else entry
        if not isinstance(data, dict):
            raise LlmError(
                f"スタブ応答（{step}）が JSON オブジェクトではありません（実際: {type(data).__name__}）。"
            )
        validate_payload(data, schema, where=f"スタブ応答 / {step}")
        logger.info("LLM（%s）はスタブ応答を使いました。", step)
        return LlmResponse(data=deepcopy(data), input_tokens=0, output_tokens=0, retries=0)


# ---------------------------------------------------------------------------
# 組み立て
# ---------------------------------------------------------------------------


def build_client(cfg: LlmConfig, *, stub_responses: dict | None = None) -> LlmClient:
    """設定から LLM クライアントを作る。stub_responses があればスタブを優先する。"""
    if stub_responses is not None:
        return StubLlmClient(stub_responses, model=f"stub:{cfg.model}")
    raw = (cfg.provider or "").strip().lower()
    provider = PROVIDER_ALIASES.get(raw)
    if provider == "claude_agent_sdk":
        return ClaudeAgentSdkClient(cfg)
    if provider == "anthropic":
        return AnthropicClient(cfg)
    known = sorted(set(PROVIDER_ALIASES.values()))
    raise LlmError(
        f"未対応の LLM プロバイダです: {cfg.provider!r}"
        f"（対応しているのは {known}。config の llm.provider を直してください）。\n"
        "  'claude_agent_sdk' … このパソコンの Claude Code を使う（APIキー不要・既定）\n"
        "  'anthropic'        … Anthropic API を使う（APIキーが要る）"
    )


def load_stub_responses(path: str | Path) -> dict[str, dict]:
    """`--stub-llm PATH` 用。{"step名": {...}} 形式の JSON を読む。"""
    p = Path(path)
    if not p.exists():
        raise LlmError(f"スタブ応答ファイルが見つかりません: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LlmError(f"スタブ応答ファイルの JSON が壊れています: {p}\n{exc}") from exc
    if not isinstance(data, dict):
        raise LlmError(f"スタブ応答ファイルは {{\"step名\": {{...}}}} 形式にしてください: {p}")
    return data
