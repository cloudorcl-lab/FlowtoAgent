# SOP: Building Production Anthropic Tool-Use Agents

**Version:** 1.0
**Reference Project:** FlowtoAgent — ZIP-to-Weather Multi-Stage Agent
**Skill:** `/anthropic-tool-use-agent`

---

## Overview

This SOP defines the standard approach for building multi-stage Claude agents that call external tools through a manual agentic loop using the Anthropic Python SDK. It is optimized for token efficiency, self-healing reliability, and observable performance.

**Use this pattern when:**
- Claude needs to call external APIs or local functions in a defined sequence
- You need full control over the loop (retries, logging, approval gates)
- Token cost matters (production workloads, high-volume queries)
- The agent must be testable with no live dependencies

**Do NOT use this pattern when:**
- You need built-in file/web/terminal access → use Claude Agent SDK instead
- You need MCP server integration out of the box

---

## Step 1: Design the Workflow

Before writing code, define on paper (or in Excalidraw):

1. **Stages** — what ordered steps must Claude execute?
2. **Gate conditions** — when does the workflow stop early?
3. **Tools** — one tool per stage, minimum fields in/out
4. **Failure modes** — what can go wrong at each stage?

**Example (ZIP-to-Weather):**

```
User Input
    ↓
Stage 1: validate_zip (local, regex)
    ├── invalid → re-prompt user
    └── valid ↓
Stage 2: zipgeocode (HTTP: zippopotam.us)
    ├── error → Claude reports, stops
    └── success ↓
Stage 3: grid_finder (HTTP: api.weather.gov × 2)
    └── Claude formats final forecast
```

---

## Step 2: Choose the Model

| Task Complexity | Model | Cost (in/out) |
|---|---|---|
| Simple routing / validation / structured extraction | `claude-haiku-4-5` | $1 / $5 per 1M |
| Moderate reasoning with analysis | `claude-sonnet-4-6` | $3 / $15 per 1M |
| Complex multi-turn reasoning | `claude-opus-4-6` | $5 / $25 per 1M |

**Rule:** Start with `claude-haiku-4-5`. Only upgrade if Claude makes wrong tool choices.
**Never enable thinking/adaptive thinking** for tool-routing agents — it wastes tokens on decisions that don't require deep reasoning.
**Set `max_tokens=512`** — tool routing + summary fits in 512 tokens.

---

## Step 3: Write Constants

```python
MODEL = "claude-haiku-4-5"
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a [domain] agent. Call tools in this exact order:\n"
    "1. stage_one_tool — if [condition]=false, tell user and stop.\n"
    "2. stage_two_tool — show [key output fields].\n"
    "3. stage_three_tool — present [final output] clearly.\n"
    "If any tool returns an error field, report it and stop. Never skip steps."
)

TOOLS = [
    {
        "name": "stage_one_tool",
        "description": "One sentence description.",
        "input_schema": {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        },
    },
    # ... one entry per stage
]
```

**System prompt rules:**
- State the exact call order — prevents Claude skipping steps
- State what to do on failure — prevents hallucination
- One sentence per tool
- Under 80 words total

---

## Step 4: Implement Metrics

Add this dataclass before any tool implementations:

```python
from dataclasses import dataclass, field
import time

@dataclass
class Metrics:
    start_time: float = field(default_factory=time.time)
    stage_times: dict = field(default_factory=dict)
    api_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record_stage(self, tool_name, elapsed_ms):
        self.stage_times[tool_name] = elapsed_ms

    def record_response(self, response):
        self.api_calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

    def total_elapsed_ms(self):
        return (time.time() - self.start_time) * 1000

    def estimated_cost_usd(self):
        # Haiku: $1/$5 per 1M tokens
        return (self.input_tokens * 1e-6) + (self.output_tokens * 5e-6)

    def print_summary(self):
        bar = "━" * 38
        print(f"\n{bar}\n PERFORMANCE SUMMARY\n{bar}")
        for key, ms in self.stage_times.items():
            print(f"   {key:<22} : {ms:>8.1f} ms")
        print(f"\n Anthropic API calls : {self.api_calls}")
        print(f" Input tokens        : {self.input_tokens:,}")
        print(f" Output tokens       : {self.output_tokens:,}")
        print(f" Estimated cost      : ${self.estimated_cost_usd():.4f}")
        print(f" Total elapsed       : {self.total_elapsed_ms():,.0f} ms")
        print(f" Retries             : {self.retries}")
        print(bar)
```

**Call `metrics.print_summary()` after every run — success or failure.**

---

## Step 5: Implement Self-Healing Helpers

### HTTP Retry (for external APIs)

```python
import requests
import time

def http_get_with_retry(url, headers=None, max_retries=3, backoff_base=1.0, metrics=None):
    """GET with exponential backoff on transient errors."""
    retryable_codes = {429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code in retryable_codes and attempt < max_retries - 1:
                wait = min(backoff_base * (2 ** attempt), 10)
                if metrics: metrics.retries += 1
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException:
            if attempt < max_retries - 1:
                wait = min(backoff_base * (2 ** attempt), 10)
                if metrics: metrics.retries += 1
                time.sleep(wait)
            else:
                raise
```

### Anthropic API Retry

```python
import anthropic

def claude_create_with_retry(client, max_retries=3, metrics=None, **kwargs):
    """Call client.messages.create with retry on rate-limit and server errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            wait = int(getattr(e.response, "headers", {}).get("retry-after", 2 ** attempt))
            if metrics: metrics.retries += 1
            time.sleep(min(wait, 30))
        except anthropic.InternalServerError:
            if metrics: metrics.retries += 1
            time.sleep(2 ** attempt)
    raise RuntimeError("Claude API unavailable after retries")
```

---

## Step 6: Implement Tools

Each tool function:
- **Never raises** — catches all exceptions and returns `{"error": "..."}`
- Returns **only essential fields** — minimum output tokens
- Accepts `metrics=None` as last param — pass through to HTTP helpers

```python
def my_tool(param: str, metrics: Metrics = None) -> dict:
    url = f"https://api.example.com/{param}"
    try:
        resp = http_get_with_retry(url, metrics=metrics)
    except requests.RequestException:
        return {"error": "Service unavailable. Try again later."}

    if resp.status_code == 404:
        return {"error": f"{param} not found."}
    if resp.status_code != 200:
        return {"error": f"Service error (HTTP {resp.status_code})."}

    try:
        data = resp.json()
        return {"field1": data["key1"], "field2": data["key2"]}  # minimal
    except (KeyError, ValueError):
        return {"error": "Unexpected response format."}
```

### Tool Dispatch Map

```python
import json

TOOL_MAP = {
    "tool_name": lambda inp, m: my_tool(inp["param"], metrics=m),
}

def dispatch_tool(name: str, tool_input: dict, metrics: Metrics) -> str:
    """Call the named tool and return a JSON string. Never raises."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(tool_input, metrics))
    except Exception as e:
        return json.dumps({"error": f"Tool execution error: {str(e)}"})
```

---

## Step 7: Implement the Agentic Loop

```python
def run_agent(client: anthropic.Anthropic, user_input: str, metrics: Metrics) -> bool:
    messages = [{"role": "user", "content": f"[Task description] {user_input}"}]

    while True:
        response = claude_create_with_retry(
            client,
            metrics=metrics,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        metrics.record_response(response)
        messages.append({"role": "assistant", "content": response.content})  # full object!

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    t0 = time.time()
                    result_str = dispatch_tool(block.name, block.input, metrics)
                    metrics.record_stage(block.name, (time.time() - t0) * 1000)

                    result_dict = json.loads(result_str)

                    # Stage gate — fail-fast on Stage 1 validation failure
                    if block.name == "stage_one_tool" and not result_dict.get("is_valid"):
                        return False  # caller will re-prompt user

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
            messages.append({"role": "user", "content": tool_results})

        else:  # end_turn
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(f"\n{block.text}")
            return True
```

---

## Step 8: Implement Main Loop

```python
import os
import sys

def main():
    # Windows UTF-8 fix for emoji in Claude responses
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    print("Agent Name")
    print("=" * 38)

    while True:
        user_input = input("\nEnter input (or 'quit' to exit): ").strip()
        if user_input.lower() in ("quit", "q", "exit"):
            print("Goodbye.")
            break

        metrics = Metrics()
        success = run_agent(client, user_input, metrics)
        metrics.print_summary()

        if success:
            another = input("\nAnother? (y/n): ").strip().lower()
            if another != "y":
                break
        else:
            print("Invalid input. Please try again.")


if __name__ == "__main__":
    main()
```

---

## Step 9: Write the Test Suite

**File:** `test_agent.py`
**Framework:** `unittest` + `unittest.mock` (stdlib — no new dependencies)

### Run command
```bash
python -m unittest test_agent.py -v
```

### Test class structure

```python
import json
import unittest
from unittest.mock import MagicMock, patch
from agent import (
    validate_xxx, tool_one, tool_two,
    http_get_with_retry, dispatch_tool, run_agent,
    Metrics,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
TOOL_ONE_OK = {"key": "value"}  # minimal happy-path API fixture

def make_mock_response(status_code, json_data=None):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    return mock

class MockUsage:
    input_tokens = 100
    output_tokens = 30

class MockToolUseBlock:
    def __init__(self, name, input_dict, block_id="tu_001"):
        self.type = "tool_use"
        self.name = name
        self.input = input_dict
        self.id = block_id

class MockTextBlock:
    type = "text"
    def __init__(self, text="Done."):
        self.text = text

class MockResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content
        self.usage = MockUsage()

# ── Test Classes ──────────────────────────────────────────────────────────────

class TestValidateXxx(unittest.TestCase):
    """Pure Python — no mocks needed."""
    def test_valid(self): ...
    def test_invalid(self): ...

class TestToolOne(unittest.TestCase):
    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_success(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(200, TOOL_ONE_OK)
        result = tool_one("input")
        self.assertNotIn("error", result)

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_not_found(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(404)
        result = tool_one("input")
        self.assertIn("error", result)

class TestHttpGetWithRetry(unittest.TestCase):
    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_retries_500_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            make_mock_response(500),
            make_mock_response(500),
            make_mock_response(200, {"ok": True}),
        ]
        resp = http_get_with_retry("https://example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 3)

class TestRunAgent(unittest.TestCase):
    def _client(self):
        return MagicMock()

    @patch("agent.dispatch_tool")
    @patch("agent.claude_create_with_retry")
    def test_happy_path_returns_true(self, mock_claude, mock_dispatch):
        mock_dispatch.return_value = json.dumps({"is_valid": True, "key": "value"})
        mock_claude.side_effect = [
            MockResponse("tool_use", [MockToolUseBlock("stage_one_tool", {"param": "x"}, "tu_1")]),
            MockResponse("end_turn", [MockTextBlock("Result here.")]),
        ]
        self.assertTrue(run_agent(self._client(), "test input", Metrics()))

    @patch("agent.dispatch_tool")
    @patch("agent.claude_create_with_retry")
    def test_invalid_input_returns_false(self, mock_claude, mock_dispatch):
        mock_dispatch.return_value = json.dumps({"is_valid": False, "reason": "Bad input"})
        mock_claude.return_value = MockResponse(
            "tool_use", [MockToolUseBlock("stage_one_tool", {"param": "bad"}, "tu_1")]
        )
        self.assertFalse(run_agent(self._client(), "bad", Metrics()))


if __name__ == "__main__":
    unittest.main()
```

### Coverage checklist (per tool)

- [ ] Happy path (200, valid JSON)
- [ ] Not found (404)
- [ ] Service error (500, all retries exhausted)
- [ ] Network error (`ConnectionError`)
- [ ] Malformed response (200, missing keys)

### Coverage checklist (`http_get_with_retry`)

- [ ] Success first try
- [ ] 500 → success on 3rd attempt
- [ ] ConnectionError → success on 3rd attempt
- [ ] ConnectionError × 3 → raises
- [ ] 400 → returns immediately (no retry)
- [ ] `metrics.retries` increments correctly

### Coverage checklist (`run_agent`)

- [ ] Happy path returns `True`
- [ ] Stage 1 invalid returns `False`
- [ ] Tool error in later stage returns `True` (Claude handles it)
- [ ] `metrics.api_calls` and `metrics.input_tokens` > 0

---

## Step 10: Run Tests, Then Release

```bash
python -m unittest test_agent.py -v
```

**All tests must pass.** Fix any failures in `agent.py` and re-run. Do not release until green.

---

## Standard File Layout

```
my_agent/
├── agent.py                         # All logic
├── test_agent.py                    # Full test suite
├── requirements.txt                 # Runtime deps only
└── SOP_Anthropic_Tool_Use_Agent.md  # This document
```

**requirements.txt:**
```
anthropic>=0.25.0
requests>=2.31.0
```

---

## Token Budget Reference (Haiku)

| Component | Approx tokens |
|---|---|
| System prompt (80 words) | ~100 |
| Tool schemas (3 tools) | ~200 |
| User message | ~20 |
| Tool call responses × 3 | ~200 |
| Conversation history growth | ~400 |
| Claude output (routing + summary) | ~190 |
| **Total per run** | **~1,100–1,400** |
| **Cost at Haiku rates** | **~$0.002–$0.006** |

---

## Quick Reference Checklist

```
Planning
  [ ] Workflow diagram with stages, gates, and failure modes
  [ ] One tool per stage minimum

Implementation
  [ ] MODEL = "claude-haiku-4-5", MAX_TOKENS = 512
  [ ] System prompt < 80 words with explicit call order
  [ ] Tool schemas: lean, 1-sentence descriptions
  [ ] Metrics dataclass wired through all API-calling functions
  [ ] http_get_with_retry for all external HTTP calls
  [ ] claude_create_with_retry wrapping all Anthropic calls
  [ ] Tools never raise — return {"error": "..."} dicts
  [ ] Dispatch map + dispatch_tool wrapper
  [ ] Agentic loop appends full response.content, not just text
  [ ] Windows UTF-8 reconfigure in main()

Testing
  [ ] test_agent.py with unittest + unittest.mock only
  [ ] All tool functions covered (happy + 4 failure modes each)
  [ ] http_get_with_retry fully tested (retry count, exhaustion, no-retry codes)
  [ ] dispatch_tool: valid, unknown, exception-caught
  [ ] run_agent: happy, invalid stage 1, later error, metrics populated
  [ ] time.sleep patched in all retry tests
  [ ] All tests pass before release
```

---

## Reference Implementation

- **agent.py**: `f:\GoogleDrive\Oracle\1. Oracle EA 2\Content\AI ML\Agents\FlowtoAgent\agent.py`
- **test_agent.py**: `f:\GoogleDrive\Oracle\1. Oracle EA 2\Content\AI ML\Agents\FlowtoAgent\test_agent.py`
- **Skill**: `/anthropic-tool-use-agent` (invoke in any Claude Code session)
