#!/usr/bin/env python3
"""
ZIP-to-Weather Multi-Stage Agent
Stage 1: Validate ZIP via LLM
Stage 2: Geocode ZIP to lat/lon (zippopotam.us)
Stage 3: Fetch NWS weather forecast (api.weather.gov)

Self-healing: HTTP retries with exponential backoff, Anthropic API retries.
Metrics: per-stage timing, token usage, estimated cost.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import anthropic
import requests

# ── Constants ────────────────────────────────────────────────────────────────

MODEL      = os.environ.get("ANTHROPIC_MODEL",      "claude-haiku-4-5")
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "512"))
NWS_HEADERS = {"User-Agent": "ZIPWeather/1.0 (weather-agent)"}

SYSTEM_PROMPT = (
    "You are a weather lookup agent. Call tools in this exact order:\n"
    "1. validate_zip — if is_valid=false, tell user and stop.\n"
    "2. zipgeocode — show city, state, coordinates.\n"
    "3. grid_finder — present the 2-period forecast clearly.\n"
    "If any tool returns an error field, report it and stop. Never skip steps."
)

STAGE_HEADERS = {
    "validate_zip": "\n=== STAGE 1: ZIP VALIDATION ===",
    "zipgeocode":   "\n=== STAGE 2: GEOCODING ===",
    "grid_finder":  "\n=== STAGE 3: WEATHER GRID & FORECAST ===",
}

TOOLS = [
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
        "name": "zipgeocode",
        "description": "Convert a US ZIP code to latitude, longitude, city, and state.",
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


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    start_time: float = field(default_factory=time.time)
    stage_times: dict = field(default_factory=dict)
    api_calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record_stage(self, tool_name: str, elapsed_ms: float):
        self.stage_times[tool_name] = elapsed_ms

    def record_response(self, response):
        self.api_calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

    def total_elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def estimated_cost_usd(self) -> float:
        return (self.input_tokens * 1e-6) + (self.output_tokens * 5e-6)

    def print_summary(self):
        bar = "━" * 38
        print(f"\n{bar}")
        print(" PERFORMANCE SUMMARY")
        print(bar)
        if self.stage_times:
            print(" Stage timings:")
            labels = {
                "validate_zip": "validate_zip  (local)",
                "zipgeocode":   "zipgeocode         ",
                "grid_finder":  "grid_finder (2 NWS)",
            }
            for key, ms in self.stage_times.items():
                label = labels.get(key, key)
                print(f"   {label} : {ms:>8.1f} ms")
        print(f"\n Anthropic API calls : {self.api_calls}")
        print(f" Input tokens        : {self.input_tokens:,}")
        print(f" Output tokens       : {self.output_tokens:,}")
        print(f" Estimated cost      : ${self.estimated_cost_usd():.4f}")
        print(f" Total elapsed       : {self.total_elapsed_ms():,.0f} ms")
        print(f" Retries             : {self.retries}")
        print(bar)


# ── Self-Healing Helpers ──────────────────────────────────────────────────────

def http_get_with_retry(
    url: str,
    headers: dict = None,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    metrics: Metrics = None,
) -> requests.Response:
    """GET with exponential backoff on transient errors."""
    retryable_codes = {429, 500, 502, 503, 504}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code in retryable_codes and attempt < max_retries - 1:
                wait = min(backoff_base * (2 ** attempt), 10)
                if metrics:
                    metrics.retries += 1
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException:
            if attempt < max_retries - 1:
                wait = min(backoff_base * (2 ** attempt), 10)
                if metrics:
                    metrics.retries += 1
                time.sleep(wait)
            else:
                raise


def claude_create_with_retry(client: anthropic.Anthropic, max_retries: int = 3, metrics: Metrics = None, **kwargs):
    """Call client.messages.create with retry on rate-limit and server errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt < max_retries - 1:
                wait = int(getattr(e.response, "headers", {}).get("retry-after", 2 ** attempt))
                if metrics:
                    metrics.retries += 1
                time.sleep(min(wait, 30))
            else:
                raise
        except anthropic.InternalServerError:
            if attempt < max_retries - 1:
                if metrics:
                    metrics.retries += 1
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("Claude API unavailable after retries")


# ── Tool Implementations ──────────────────────────────────────────────────────

def validate_zip(zip_code: str) -> dict:
    if re.fullmatch(r"\d{5}", zip_code.strip()):
        return {"is_valid": True}
    return {"is_valid": False, "reason": "Must be exactly 5 digits (e.g. 90210)"}


def zipgeocode(zip_code: str, metrics: Metrics = None) -> dict:
    url = f"http://api.zippopotam.us/us/{zip_code.strip()}"
    try:
        resp = http_get_with_retry(url, metrics=metrics)
    except requests.RequestException:
        return {"error": "Geocoding service unavailable. Try again later."}

    if resp.status_code == 404:
        return {"error": f"ZIP {zip_code} not found in database."}
    if resp.status_code != 200:
        return {"error": f"Geocoding service error (HTTP {resp.status_code}). Try again later."}

    try:
        data = resp.json()
        place = data["places"][0]
        return {
            "city":  place["place name"],
            "state": place["state abbreviation"],
            "lat":   place["latitude"],
            "lon":   place["longitude"],
        }
    except (KeyError, IndexError, ValueError):
        return {"error": "Unexpected geocoding response format."}


def grid_finder(latitude: str, longitude: str, metrics: Metrics = None) -> dict:
    points_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    try:
        resp = http_get_with_retry(points_url, headers=NWS_HEADERS, metrics=metrics)
    except requests.RequestException:
        return {"error": "NWS weather service unavailable. Try again later."}

    if resp.status_code == 404 or resp.status_code == 422:
        return {"error": "Location not supported by NWS (offshore or non-US?)."}
    if resp.status_code != 200:
        return {"error": f"NWS points API error (HTTP {resp.status_code}). Try again later."}

    try:
        pts = resp.json()["properties"]
        office = pts["gridId"]
        gx = pts["gridX"]
        gy = pts["gridY"]
        city  = pts["relativeLocation"]["properties"]["city"]
        state = pts["relativeLocation"]["properties"]["state"]
    except (KeyError, TypeError, ValueError):
        return {"error": "Unexpected NWS points response format."}

    forecast_url = f"https://api.weather.gov/gridpoints/{office}/{gx},{gy}/forecast"
    try:
        resp2 = http_get_with_retry(forecast_url, headers=NWS_HEADERS, metrics=metrics)
    except requests.RequestException:
        return {"error": "NWS forecast service unavailable. Try again later."}

    if resp2.status_code != 200:
        return {"error": f"NWS forecast error (HTTP {resp2.status_code}). Try again later."}

    try:
        periods = resp2.json()["properties"]["periods"][:2]
        forecast = [
            {
                "period":   p["name"],
                "temp":     f"{p['temperature']}{p['temperatureUnit']}",
                "wind":     f"{p['windSpeed']} {p['windDirection']}",
                "summary":  p["shortForecast"],
            }
            for p in periods
        ]
        return {"location": f"{city}, {state}", "forecast": forecast}
    except (KeyError, IndexError, TypeError, ValueError):
        return {"error": "Unexpected NWS forecast response format."}


# ── Dispatch ──────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "validate_zip": lambda inp, m: validate_zip(inp["zip_code"]),
    "zipgeocode":   lambda inp, m: zipgeocode(inp["zip_code"], metrics=m),
    "grid_finder":  lambda inp, m: grid_finder(inp["latitude"], inp["longitude"], metrics=m),
}


def dispatch_tool(name: str, tool_input: dict, metrics: Metrics) -> str:
    """Call the named tool and return a JSON string. Never raises."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(tool_input, metrics)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Tool execution error: {str(e)}"})


# ── Agentic Loop ──────────────────────────────────────────────────────────────

def run_agent(client: anthropic.Anthropic, zip_input: str, metrics: Metrics) -> bool:
    """
    Run one complete lookup attempt.
    Returns True on success (forecast shown), False if ZIP invalid (re-prompt).
    """
    messages = [{"role": "user", "content": f"Weather for ZIP {zip_input}"}]

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
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(STAGE_HEADERS.get(block.name, f"\n=== {block.name.upper()} ==="))
                    t0 = time.time()
                    result_str = dispatch_tool(block.name, block.input, metrics)
                    elapsed_ms = (time.time() - t0) * 1000
                    metrics.record_stage(block.name, elapsed_ms)

                    result_dict = json.loads(result_str)

                    # Stage 1 gate — fail-fast on invalid ZIP
                    if block.name == "validate_zip" and not result_dict.get("is_valid"):
                        print(f"  Invalid ZIP: {result_dict.get('reason', 'Unknown reason')}")
                        return False

                    # Show tool output summary
                    if "error" not in result_dict:
                        if block.name == "zipgeocode":
                            print(f"  {result_dict.get('city')}, {result_dict.get('state')} "
                                  f"({result_dict.get('lat')}, {result_dict.get('lon')})")
                        elif block.name == "grid_finder":
                            print(f"  Location: {result_dict.get('location')}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            # end_turn — print Claude's final summary
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(f"\n{block.text}")
            return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("       Copy .env.example to .env and fill in your key.")
        return

    print(f"Model      : {MODEL}")
    print(f"Max tokens : {MAX_TOKENS}")
    client = anthropic.Anthropic(api_key=api_key)

    print("ZIP-to-Weather Agent")
    print("=" * 38)

    while True:
        zip_input = input("\nEnter US ZIP code (or 'quit' to exit): ").strip()
        if zip_input.lower() in ("quit", "q", "exit"):
            print("Goodbye.")
            break

        metrics = Metrics()
        success = run_agent(client, zip_input, metrics)
        metrics.print_summary()

        if success:
            another = input("\nLook up another ZIP? (y/n): ").strip().lower()
            if another != "y":
                break
        else:
            print("Please enter a valid 5-digit US ZIP code.")


if __name__ == "__main__":
    main()
