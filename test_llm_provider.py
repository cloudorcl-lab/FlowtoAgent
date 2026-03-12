"""
test_llm_provider.py — Tests for llm_provider.py

Covers: canonical type construction, schema conversion per provider,
message formatting, and factory function.
No live API calls — all SDK clients are patched.

Run: python -m unittest test_llm_provider.py -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from llm_provider import (
    ToolCall,
    LLMUsage,
    LLMResponse,
    ClaudeProvider,
    OpenAIProvider,
    create_provider,
)

# ── Sample tools in canonical (Claude input_schema) format ────────────────────

SAMPLE_TOOLS = [
    {
        "name": "validate_zip",
        "description": "Check if a string is a valid 5-digit US ZIP code.",
        "input_schema": {
            "type": "object",
            "properties": {"zip_code": {"type": "string"}},
            "required": ["zip_code"],
        },
    },
    {
        "name": "grid_finder",
        "description": "Get NWS weather forecast for a lat/lon coordinate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude":  {"type": "string"},
                "longitude": {"type": "string"},
            },
            "required": ["latitude", "longitude"],
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_claude_provider():
    with patch("anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        return ClaudeProvider()


def _make_openai_provider():
    import sys
    mock_openai = MagicMock()
    mock_openai.OpenAI = MagicMock()
    mock_openai.RateLimitError = Exception
    mock_openai.InternalServerError = Exception
    with patch.dict("sys.modules", {"openai": mock_openai}), \
         patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return OpenAIProvider()


# ── 1. Canonical Type Tests ───────────────────────────────────────────────────

class TestToolCall(unittest.TestCase):

    def test_fields_accessible(self):
        tc = ToolCall(id="tu_1", name="validate_zip", input={"zip_code": "90210"})
        self.assertEqual(tc.id, "tu_1")
        self.assertEqual(tc.name, "validate_zip")
        self.assertEqual(tc.input["zip_code"], "90210")


class TestLLMUsage(unittest.TestCase):

    def test_fields_accessible(self):
        u = LLMUsage(input_tokens=100, output_tokens=30)
        self.assertEqual(u.input_tokens, 100)
        self.assertEqual(u.output_tokens, 30)


class TestLLMResponse(unittest.TestCase):

    def test_tool_use_response(self):
        tc = ToolCall(id="tu_1", name="foo", input={})
        r = LLMResponse(
            stop_reason="tool_use",
            tool_calls=[tc],
            text="",
            usage=LLMUsage(50, 20),
            raw=None,
        )
        self.assertEqual(r.stop_reason, "tool_use")
        self.assertEqual(len(r.tool_calls), 1)
        self.assertEqual(r.tool_calls[0].name, "foo")

    def test_end_turn_response(self):
        r = LLMResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text="All done.",
            usage=LLMUsage(50, 20),
            raw=None,
        )
        self.assertEqual(r.stop_reason, "end_turn")
        self.assertEqual(r.text, "All done.")
        self.assertEqual(r.tool_calls, [])


# ── 2. ClaudeProvider Tests ───────────────────────────────────────────────────

class TestClaudeProviderConvertTools(unittest.TestCase):
    """convert_tools is a passthrough — returns TOOLS unchanged."""

    def test_passthrough_returns_tools_unchanged(self):
        p = _make_claude_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertEqual(result, SAMPLE_TOOLS)

    def test_input_schema_key_preserved(self):
        p = _make_claude_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertIn("input_schema", result[0])

    def test_does_not_wrap_in_function(self):
        p = _make_claude_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertNotIn("function", result[0])

    def test_all_tools_returned(self):
        p = _make_claude_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertEqual(len(result), len(SAMPLE_TOOLS))


class TestClaudeProviderFormatMessages(unittest.TestCase):

    def test_tool_messages_single_user_message(self):
        p = _make_claude_provider()
        tcs = [ToolCall(id="tu_1", name="validate_zip", input={"zip_code": "90210"})]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}'])
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_tool_result_type_in_content(self):
        p = _make_claude_provider()
        tcs = [ToolCall(id="tu_1", name="validate_zip", input={})]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}'])
        content = msgs[0]["content"]
        self.assertEqual(content[0]["type"], "tool_result")
        self.assertEqual(content[0]["tool_use_id"], "tu_1")

    def test_multiple_tool_results_in_one_message(self):
        p = _make_claude_provider()
        tcs = [
            ToolCall(id="tu_1", name="tool_a", input={}),
            ToolCall(id="tu_2", name="tool_b", input={}),
        ]
        msgs = p.format_tool_messages(tcs, ['{"a": 1}', '{"b": 2}'])
        # Claude packs all results into one user message
        self.assertEqual(len(msgs), 1)
        self.assertEqual(len(msgs[0]["content"]), 2)


# ── 3. OpenAIProvider Tests ───────────────────────────────────────────────────

class TestOpenAIProviderConvertTools(unittest.TestCase):
    """convert_tools wraps in function/parameters format."""

    def test_wraps_in_function_type(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertEqual(result[0]["type"], "function")

    def test_input_schema_becomes_parameters(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        fn = result[0]["function"]
        self.assertIn("parameters", fn)
        self.assertNotIn("input_schema", fn)

    def test_name_preserved(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertEqual(result[0]["function"]["name"], "validate_zip")

    def test_description_preserved(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertIn("valid 5-digit", result[0]["function"]["description"])

    def test_required_fields_preserved(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        params = result[0]["function"]["parameters"]
        self.assertIn("required", params)
        self.assertEqual(params["required"], ["zip_code"])

    def test_all_tools_converted(self):
        p = _make_openai_provider()
        result = p.convert_tools(SAMPLE_TOOLS)
        self.assertEqual(len(result), len(SAMPLE_TOOLS))


class TestOpenAIProviderFormatMessages(unittest.TestCase):

    def test_tool_messages_one_per_call(self):
        p = _make_openai_provider()
        tcs = [
            ToolCall(id="tc_1", name="validate_zip", input={"zip_code": "90210"}),
            ToolCall(id="tc_2", name="grid_finder",  input={"latitude": "34.09", "longitude": "-118.40"}),
        ]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}', '{"location": "Beverly Hills, CA"}'])
        # OpenAI sends one "tool" message per result
        self.assertEqual(len(msgs), 2)

    def test_tool_message_role(self):
        p = _make_openai_provider()
        tcs = [ToolCall(id="tc_1", name="validate_zip", input={})]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}'])
        self.assertEqual(msgs[0]["role"], "tool")

    def test_tool_call_id_matches(self):
        p = _make_openai_provider()
        tcs = [ToolCall(id="tc_abc", name="validate_zip", input={})]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}'])
        self.assertEqual(msgs[0]["tool_call_id"], "tc_abc")

    def test_content_is_result_string(self):
        p = _make_openai_provider()
        tcs = [ToolCall(id="tc_1", name="validate_zip", input={})]
        msgs = p.format_tool_messages(tcs, ['{"is_valid": true}'])
        self.assertEqual(msgs[0]["content"], '{"is_valid": true}')


# ── 4. Factory Tests ──────────────────────────────────────────────────────────

class TestCreateProvider(unittest.TestCase):

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"LLM_PROVIDER": "claude", "ANTHROPIC_API_KEY": "test-key"})
    def test_creates_claude_provider(self, mock_anthropic):
        p = create_provider()
        self.assertIsInstance(p, ClaudeProvider)

    def test_creates_openai_provider(self):
        import sys
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock()
        mock_openai.RateLimitError = Exception
        mock_openai.InternalServerError = Exception
        with patch.dict("sys.modules", {"openai": mock_openai}), \
             patch.dict("os.environ", {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
            p = create_provider()
        self.assertIsInstance(p, OpenAIProvider)

    @patch.dict("os.environ", {"LLM_PROVIDER": "unknown_xyz"})
    def test_unknown_provider_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            create_provider()
        self.assertIn("unknown_xyz", str(ctx.exception))

    @patch("anthropic.Anthropic")
    @patch.dict("os.environ", {"LLM_PROVIDER": "claude", "ANTHROPIC_API_KEY": "test-key"})
    def test_default_is_claude(self, mock_anthropic):
        # Explicitly setting claude is equivalent to the default behaviour
        p = create_provider()
        self.assertIsInstance(p, ClaudeProvider)

    @patch.dict("os.environ", {"LLM_PROVIDER": "claude"}, clear=False)
    def test_missing_api_key_raises_value_error(self):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        env["LLM_PROVIDER"] = "claude"
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                create_provider()
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
