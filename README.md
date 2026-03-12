# FlowtoAgent — ZIP-to-Weather Multi-Stage Agent

A production-ready Python agent built with the Anthropic Claude API that converts a US ZIP code into a live weather forecast through a three-stage tool-use pipeline. Designed as a reference implementation for token-optimized, self-healing, observable Claude agents.

---

## How It Works

```
User enters ZIP code
        │
        ▼
┌─────────────────────┐
│  STAGE 1            │  validate_zip (local regex)
│  ZIP Validation     │──── invalid ──▶ re-prompt user
└────────┬────────────┘
         │ valid
         ▼
┌─────────────────────┐
│  STAGE 2            │  zipgeocode → zippopotam.us API
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

Claude orchestrates the tool calls in sequence. Each stage is a gate — failure stops the pipeline and surfaces a clear error message rather than continuing with bad data.

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

### Model Choice
Uses `claude-haiku-4-5` — the fastest and most cost-efficient Claude model. Tool routing does not require deep reasoning, so Opus/Sonnet would be wasteful. No thinking/extended reasoning enabled.

| Setting | Value | Reason |
|---|---|---|
| Model | `claude-haiku-4-5` | $1/$5 per 1M tokens — 5× cheaper than Sonnet |
| `max_tokens` | 512 | Caps output; tool routing + summary fits comfortably |
| Thinking | Disabled | Not needed for structured tool orchestration |

### External APIs (all free, no key required)

| API | Purpose | Endpoint |
|---|---|---|
| [zippopotam.us](http://api.zippopotam.us) | ZIP → lat/lon, city, state | `GET /us/{zip}` |
| [api.weather.gov](https://www.weather.gov/documentation/services-web-api) | Grid lookup + forecast | `GET /points/{lat},{lon}` + `GET /gridpoints/{office}/{x},{y}/forecast` |

### Self-Healing (3 layers)

1. **HTTP retry** — exponential backoff (1s → 2s → 4s, max 10s) on `429, 500, 502, 503, 504` and `ConnectionError`
2. **Anthropic API retry** — respects `retry-after` header on rate limits; backs off on server errors
3. **Claude-driven error propagation** — tools never raise; they return `{"error": "..."}` dicts that Claude reads and reports to the user

### Observability

A `Metrics` dataclass tracks per-run:
- Per-stage wall-clock timing (ms)
- Anthropic API call count
- Input/output token counts
- Estimated cost (USD)
- Total elapsed time
- Retry count

Printed after every run, success or failure.

---

## Project Structure

```
FlowtoAgent/
├── agent.py                         # Agent implementation (~375 lines)
├── test_agent.py                    # 32 unit tests, stdlib only
├── requirements.txt                 # Runtime dependencies
└── SOP_Anthropic_Tool_Use_Agent.md  # 10-step build guide / reuse SOP
```

---

## Setup

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/settings/keys)

### Install

```bash
git clone https://github.com/cloudorcl-lab/FlowtoAgent.git
cd FlowtoAgent
pip install -r requirements.txt
```

### Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py
```

On Windows:

```cmd
set ANTHROPIC_API_KEY=sk-ant-...
python agent.py
```

---

## Tests

No API key required — all external I/O is mocked.

```bash
python -m unittest test_agent.py -v
```

Expected output:

```
TestDispatchTool.test_tool_exception_caught ... ok
TestDispatchTool.test_unknown_tool ... ok
TestDispatchTool.test_valid_tool_validate_zip ... ok
TestGridFinder.test_forecast_500 ... ok
TestGridFinder.test_forecast_network_error ... ok
TestGridFinder.test_forecast_truncated_to_2 ... ok
TestGridFinder.test_nws_404 ... ok
TestGridFinder.test_nws_422_offshore ... ok
TestGridFinder.test_points_network_error ... ok
TestGridFinder.test_success ... ok
TestHttpGetWithRetry.test_exhausts_retries_raises ... ok
TestHttpGetWithRetry.test_increments_metrics_retries ... ok
TestHttpGetWithRetry.test_non_retryable_400_no_retry ... ok
TestHttpGetWithRetry.test_retries_on_500_then_succeeds ... ok
TestHttpGetWithRetry.test_retries_on_connection_error_then_succeeds ... ok
TestHttpGetWithRetry.test_success_first_try ... ok
TestRunAgent.test_geocode_error_returns_true ... ok
TestRunAgent.test_happy_path_returns_true ... ok
TestRunAgent.test_invalid_zip_returns_false ... ok
TestRunAgent.test_metrics_populated ... ok
TestValidateZip.test_invalid_empty ... ok
TestValidateZip.test_invalid_letters ... ok
TestValidateZip.test_invalid_too_long ... ok
TestValidateZip.test_invalid_too_short ... ok
TestValidateZip.test_invalid_with_spaces ... ok
TestValidateZip.test_valid_5digit ... ok
TestValidateZip.test_valid_nyc ... ok
TestZipgeocode.test_malformed_response ... ok
TestZipgeocode.test_network_timeout ... ok
TestZipgeocode.test_service_error ... ok
TestZipgeocode.test_success ... ok
TestZipgeocode.test_zip_not_found ... ok

Ran 32 tests in 0.058s

OK
```

### Test coverage

| Class | Scenarios |
|---|---|
| `TestValidateZip` | Valid, invalid (letters, too short, too long, empty, spaces) |
| `TestZipgeocode` | Success, 404, 500, network error, malformed JSON |
| `TestGridFinder` | Success, 422 offshore, 404, points error, forecast error, 500, 2-period truncation |
| `TestHttpGetWithRetry` | First try, retry→success, ConnectionError→success, exhausted, no-retry 400, metrics counter |
| `TestDispatchTool` | Valid tool, unknown tool, exception containment |
| `TestRunAgent` | Happy path, invalid ZIP, geocode error, metrics populated |

---

## Reuse as a Template

`SOP_Anthropic_Tool_Use_Agent.md` is a 10-step build guide for creating new agents with this same pattern:

1. Design the workflow (stages + gates + failure modes)
2. Choose model
3. Write constants (system prompt + tool schemas)
4. Implement `Metrics`
5. Implement self-healing HTTP + Anthropic helpers
6. Implement tools (never raise — return error dicts)
7. Implement the agentic loop
8. Implement `main()`
9. Write the test suite
10. Run tests → release

---

## Cost Estimate

| Component | Tokens |
|---|---|
| System prompt + tool schemas | ~300 |
| Conversation history per run | ~400 |
| Tool call responses × 3 | ~200 |
| Claude output (routing + summary) | ~190 |
| **Total per run** | **~1,100–1,400** |
| **Cost at Haiku rates** | **~$0.002–$0.006** |

---

## License

MIT
