# FlowtoAgent — ZIP-to-Weather Multi-Stage Agent

A production-ready Python agent built with the Anthropic Claude API that converts a US ZIP code into a live weather forecast through a three-stage tool-use pipeline.

---

## How This Agent Was Built

This project was built entirely through a Claude Code session — from a hand-drawn workflow diagram to a tested, documented, and published agent. Here's how the process worked.

### Step 1 — Design First, Code Second

The build started with a workflow diagram created in [Excalidraw](https://excalidraw.com). Both the diagram image and its underlying JSON were provided to Claude Code as the design specification.

```
Excalidraw JSON + Image
        │
        ▼
  Claude Code reads the diagram
  and scopes out the implementation
```

The diagram defined:
- Three sequential stages (validate → geocode → forecast)
- Gate conditions at each stage (what stops the pipeline early)
- The decision logic for re-prompting vs reporting errors

No written spec was required — the visual diagram was sufficient input to generate the full implementation plan.

### Step 2 — Planning with Constraints

Before any code was written, the plan went through two rounds of feedback:

**Round 1 — Token optimization:**
> *"Plan to be effective in the API calls. Objective is to minimize token consumption while being functionally complete."*

This drove the key architectural decisions:
- Switch from `claude-opus-4-6` to `claude-haiku-4-5` (5× cheaper)
- Disable adaptive thinking entirely (not needed for tool routing)
- Cap `max_tokens` at 512
- Write lean tool schemas (1-sentence descriptions, minimum required fields)
- Return only essential fields from each tool implementation

**Round 2 — Reliability and observability:**
> *"Include performance metrics as timing and resource consumption. The agents must be self-healing."*

This added:
- A `Metrics` dataclass tracking per-stage timing, token counts, retry counts, and estimated cost
- HTTP retry with exponential backoff on transient errors
- Anthropic API retry respecting `retry-after` headers
- Tools that never raise — returning `{"error": "..."}` dicts instead so Claude handles failures gracefully

### Step 3 — Implementation

With the plan approved, `agent.py` was generated in a single pass (~375 lines). The final architecture:

| Layer | What it does |
|---|---|
| Constants | Model, max_tokens, system prompt, tool schemas |
| Metrics | Per-run observability dataclass |
| Self-healing | HTTP retry + Anthropic API retry helpers |
| Tools | `validate_zip`, `zipgeocode`, `grid_finder` |
| Dispatch | Tool router that never raises |
| Agentic loop | Manual `tool_use` loop with stage gates |
| Main | Input loop, metrics print, re-prompt logic |

### Step 4 — Test Before Release

A replan was requested before the code was considered released:

> *"Replan. Include testing of the valid path and other failure scenarios. Perform the testing before releasing."*

`test_agent.py` was written with **32 unit tests** across 6 test classes using only `unittest` and `unittest.mock` — no additional dependencies. All external I/O (HTTP calls, Anthropic API) is mocked so tests run in under 0.1 seconds with no API key required.

```bash
python -m unittest test_agent.py -v
# Ran 32 tests in 0.058s
# OK
```

All 32 tests passed on the first run with no changes needed to `agent.py`. The agent was then considered released.

### Step 5 — Live Validation

The agent was then run against real network calls to validate end-to-end behavior. A Windows console encoding issue surfaced (Claude's response contained emoji that `cp1252` couldn't encode) and was fixed by adding a stdout reconfigure at startup.

```
Stage 1 — validate_zip  :      0.1 ms  (local regex)
Stage 2 — zipgeocode    :     41.3 ms  (zippopotam.us)
Stage 3 — grid_finder   :  1,217.8 ms  (2 × api.weather.gov)
Total cost per run      :    ~$0.006
```

### Step 6 — Documentation and SOP

Two documents were generated to make this pattern reusable:

- **`SOP_Anthropic_Tool_Use_Agent.md`** — A 10-step standard operating procedure for building future agents with the same architecture
- **Claude Code skill** — A persistent skill file (`/anthropic-tool-use-agent`) saved to the local Claude Code environment so the pattern auto-loads in future sessions

---

## What the Agent Does

Converts a US ZIP code into a live NWS weather forecast through a three-stage Claude tool-use pipeline.

```
User enters ZIP code
        │
        ▼
┌─────────────────────┐
│  STAGE 1            │  validate_zip  (local regex)
│  ZIP Validation     │──── invalid ──▶ re-prompt user
└────────┬────────────┘
         │ valid
         ▼
┌─────────────────────┐
│  STAGE 2            │  zipgeocode → zippopotam.us
│  Geocoding          │──── error ───▶ Claude reports, stops
└────────┬────────────┘
         │ city, state, lat/lon
         ▼
┌─────────────────────┐
│  STAGE 3            │  grid_finder → api.weather.gov (2 calls)
│  Weather Forecast   │  points endpoint + forecast endpoint
└────────┬────────────┘
         │
         ▼
  Claude formats and
  presents the forecast
```

---

## Sample Output

```
ZIP-to-Weather Agent
======================================

Enter US ZIP code (or 'quit' to exit): 90210

=== STAGE 1: ZIP VALIDATION ===

=== STAGE 2: GEOCODING ===
  Beverly Hills, CA (34.0901, -118.4065)

=== STAGE 3: WEATHER GRID & FORECAST ===
  Location: Beverly Hills, CA

**Weather for 90210 (Beverly Hills, CA)**

📍 **Location:** Beverly Hills, California
Coordinates: 34.0901° N, -118.4065° W

**🌞 Today**
- Temperature: 92°F
- Conditions: Sunny
- Wind: 5 to 10 mph W

**🌙 Tonight**
- Temperature: 70°F
- Conditions: Mostly Clear
- Wind: 5 to 10 mph WNW

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Stage timings:
   validate_zip  (local) :      0.1 ms
   zipgeocode          :     41.3 ms
   grid_finder (2 NWS) :   1217.8 ms

 Anthropic API calls : 4
 Input tokens        : 3,889
 Output tokens       : 361
 Estimated cost      : $0.0057
 Total elapsed       : 5,483 ms
 Retries             : 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Architecture

### Model and settings

| Setting | Value | Reason |
|---|---|---|
| Model | `claude-haiku-4-5` | $1/$5 per 1M tokens — 5× cheaper than Sonnet |
| `max_tokens` | 512 | Tool routing + summary fits comfortably |
| Thinking | Disabled | Not needed for structured tool orchestration |

### External APIs (free, no key required)

| API | Purpose | Endpoint |
|---|---|---|
| [zippopotam.us](http://api.zippopotam.us) | ZIP → lat/lon, city, state | `GET /us/{zip}` |
| [api.weather.gov](https://www.weather.gov/documentation/services-web-api) | Grid lookup | `GET /points/{lat},{lon}` |
| [api.weather.gov](https://www.weather.gov/documentation/services-web-api) | Forecast | `GET /gridpoints/{office}/{x},{y}/forecast` |

### Self-healing (3 layers)

| Layer | Mechanism |
|---|---|
| HTTP | Exponential backoff (1s → 2s → 4s) on `429, 500, 502, 503, 504` and `ConnectionError` |
| Anthropic API | Respects `retry-after` header on rate limits; backs off on server errors |
| Claude | Tools return `{"error": "..."}` dicts — Claude reads and reports, never silently fails |

---

## Project Structure

```
FlowtoAgent/
├── agent.py                         # Agent implementation (~375 lines)
├── test_agent.py                    # 32 unit tests, stdlib only
├── requirements.txt                 # Runtime dependencies
└── SOP_Anthropic_Tool_Use_Agent.md  # 10-step build guide for reuse
```

---

## Setup

**Prerequisites:** Python 3.10+, an [Anthropic API key](https://console.anthropic.com/settings/keys)

```bash
git clone https://github.com/cloudorcl-lab/FlowtoAgent.git
cd FlowtoAgent
pip install -r requirements.txt
```

**Run:**

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py

# Windows
set ANTHROPIC_API_KEY=sk-ant-...
python agent.py
```

---

## Tests

No API key required — all external I/O is mocked.

```bash
python -m unittest test_agent.py -v
```

| Test class | Coverage |
|---|---|
| `TestValidateZip` | Valid, invalid letters/length/empty/spaces (7 tests) |
| `TestZipgeocode` | Success, 404, 500, network error, malformed JSON (5 tests) |
| `TestGridFinder` | Success, 422 offshore, 404, points/forecast errors, 500, truncation (7 tests) |
| `TestHttpGetWithRetry` | First-try, retry→success, exhausted retries, no-retry 400, metrics (6 tests) |
| `TestDispatchTool` | Valid tool, unknown tool, exception containment (3 tests) |
| `TestRunAgent` | Happy path, invalid ZIP, tool error, metrics populated (4 tests) |

---

## Reusing This Pattern

`SOP_Anthropic_Tool_Use_Agent.md` is a 10-step build guide for creating new agents with this same architecture — from workflow diagram to tested release. The pattern is designed to work for any multi-stage tool-use agent that calls external APIs through Claude.

---

## License

MIT
