"""
llm_provider.py — LLM provider abstraction layer.

Supports Claude (Anthropic), OpenAI, and Google Gemini.
Select via LLM_PROVIDER env var (default: claude).

Agent code interacts only with LLMProvider — never with provider SDKs.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

# ── Canonical Types ────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    input: dict


@dataclass
class LLMUsage:
    """Token usage for one API call."""
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """Canonical response from any LLM provider."""
    stop_reason: str          # "tool_use" | "end_turn"
    tool_calls: List[ToolCall]
    text: str                 # Final text (non-empty on end_turn)
    usage: LLMUsage
    raw: object               # Original provider response (for debugging)


# ── Abstract Base ──────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """
    Abstract base for LLM providers.

    Each provider translates between canonical types and its SDK format.
    Implement all four methods to add a new provider.
    """

    @abstractmethod
    def chat(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        """Send messages and return a canonical LLMResponse."""

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Convert a provider response into an assistant message for history."""

    @abstractmethod
    def format_tool_messages(self, tool_calls: List[ToolCall], results: list) -> list:
        """
        Build the follow-up message(s) carrying tool results.
        results: list of JSON strings, one per tool_call (same order).
        Returns a list of message dicts to extend history with.
        """

    @abstractmethod
    def convert_tools(self, tools: list) -> object:
        """
        Convert canonical TOOLS list (Claude input_schema format)
        to provider-specific format.
        """


# ── Claude Provider ────────────────────────────────────────────────────────────


class ClaudeProvider(LLMProvider):
    """Anthropic Claude — input_schema format is a passthrough."""

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._max_tokens = int(
            os.environ.get("LLM_MAX_TOKENS",
                           os.environ.get("ANTHROPIC_MAX_TOKENS", "512"))
        )
        self._max_retries = 3

    def chat(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        import anthropic
        native_tools = self.convert_tools(tools)
        for attempt in range(self._max_retries):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    tools=native_tools,
                    messages=messages,
                )
                tool_calls = []
                text = ""
                for block in resp.content:
                    if block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            id=block.id,
                            name=block.name,
                            input=block.input,
                        ))
                    elif hasattr(block, "text") and block.text:
                        text = block.text
                stop_reason = "tool_use" if resp.stop_reason == "tool_use" else "end_turn"
                return LLMResponse(
                    stop_reason=stop_reason,
                    tool_calls=tool_calls,
                    text=text,
                    usage=LLMUsage(
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                    ),
                    raw=resp,
                )
            except anthropic.RateLimitError as e:
                if attempt < self._max_retries - 1:
                    wait = int(
                        getattr(e.response, "headers", {}).get("retry-after", 2 ** attempt)
                    )
                    time.sleep(min(wait, 30))
                else:
                    raise
            except anthropic.InternalServerError:
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("Claude API unavailable after retries")

    def format_assistant_message(self, response: LLMResponse) -> dict:
        # Claude requires the raw content list (preserves tool_use blocks)
        return {"role": "assistant", "content": response.raw.content}

    def format_tool_messages(self, tool_calls: List[ToolCall], results: list) -> list:
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            }
            for tc, result in zip(tool_calls, results)
        ]
        return [{"role": "user", "content": tool_results}]

    def convert_tools(self, tools: list) -> list:
        # Claude uses input_schema natively — passthrough
        return tools


# ── OpenAI Provider ────────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    """OpenAI — converts input_schema to function/parameters format."""

    def __init__(self):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))
        self._max_retries = 3

    def chat(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        import json
        import openai
        native_tools = self.convert_tools(tools)
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    tools=native_tools,
                    messages=oai_messages,
                )
                choice = resp.choices[0]
                msg = choice.message
                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            input=json.loads(tc.function.arguments),
                        ))
                text = msg.content or ""
                stop_reason = "tool_use" if tool_calls else "end_turn"
                return LLMResponse(
                    stop_reason=stop_reason,
                    tool_calls=tool_calls,
                    text=text,
                    usage=LLMUsage(
                        input_tokens=resp.usage.prompt_tokens,
                        output_tokens=resp.usage.completion_tokens,
                    ),
                    raw=resp,
                )
            except openai.RateLimitError:
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except openai.InternalServerError:
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("OpenAI API unavailable after retries")

    def format_assistant_message(self, response: LLMResponse) -> dict:
        msg = response.raw.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return assistant_msg

    def format_tool_messages(self, tool_calls: List[ToolCall], results: list) -> list:
        # OpenAI: one "tool" role message per call
        return [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
            for tc, result in zip(tool_calls, results)
        ]

    def convert_tools(self, tools: list) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]


# ── Gemini Provider ────────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """
    Google Gemini — converts to FunctionDeclaration format.
    Beta: function-calling history is approximated as text.
    Requires: pip install google-generativeai
    """

    def __init__(self):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self._max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))

    def chat(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        native_tools = self.convert_tools(tools)
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            tools=native_tools,
            system_instruction=system or None,
            generation_config={"max_output_tokens": self._max_tokens},
        )
        history = self._build_history(messages)
        if len(history) > 1:
            chat = model.start_chat(history=history[:-1])
            last_parts = history[-1]["parts"]
        else:
            chat = model.start_chat()
            last_parts = history[0]["parts"] if history else [""]

        resp = chat.send_message(last_parts)
        tool_calls = []
        text = ""
        for i, part in enumerate(resp.parts):
            if hasattr(part, "function_call") and part.function_call.name:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=f"{fc.name}_{i}",
                    name=fc.name,
                    input=dict(fc.args),
                ))
            elif hasattr(part, "text") and part.text:
                text = part.text

        stop_reason = "tool_use" if tool_calls else "end_turn"
        usage_meta = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
        out_tok = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
        return LLMResponse(
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text=text,
            usage=LLMUsage(input_tokens=in_tok, output_tokens=out_tok),
            raw=resp,
        )

    def _build_history(self, messages: list) -> list:
        gemini_msgs = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg.get("content", "")
            if isinstance(content, str):
                gemini_msgs.append({"role": role, "parts": [content]})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or str(item)
                        parts.append(str(text))
                    else:
                        parts.append(str(item))
                gemini_msgs.append({"role": role, "parts": parts})
        return gemini_msgs

    def format_assistant_message(self, response: LLMResponse) -> dict:
        if response.tool_calls:
            parts = [f"[tool_call:{tc.name}]" for tc in response.tool_calls]
            return {"role": "assistant", "content": " ".join(parts)}
        return {"role": "assistant", "content": response.text}

    def format_tool_messages(self, tool_calls: List[ToolCall], results: list) -> list:
        parts = [
            f"[tool_result:{tc.name}] {result}"
            for tc, result in zip(tool_calls, results)
        ]
        return [{"role": "user", "content": " ".join(parts)}]

    def convert_tools(self, tools: list) -> list:
        from google.generativeai import protos
        TYPE_MAP = {
            "string":  protos.Type.STRING,
            "number":  protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "array":   protos.Type.ARRAY,
            "object":  protos.Type.OBJECT,
        }
        declarations = []
        for t in tools:
            schema = t["input_schema"]
            params = protos.Schema(
                type=protos.Type.OBJECT,
                properties={
                    k: protos.Schema(type=TYPE_MAP.get(v.get("type", "string"), protos.Type.STRING))
                    for k, v in schema.get("properties", {}).items()
                },
                required=schema.get("required", []),
            )
            declarations.append(protos.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=params,
            ))
        return [protos.Tool(function_declarations=declarations)]


# ── Factory ────────────────────────────────────────────────────────────────────


def create_provider() -> LLMProvider:
    """
    Instantiate the LLM provider selected by LLM_PROVIDER env var.
    Defaults to 'claude'.
    """
    provider_name = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider_name == "claude":
        return ClaudeProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name!r}. Choose: claude, openai, gemini"
        )
