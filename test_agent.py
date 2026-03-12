"""
test_agent.py — Full test suite for agent.py

Covers: valid path (happy path) and all failure scenarios.
No real HTTP or LLM API calls — all external I/O is mocked.

Run: python -m unittest test_agent.py -v
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from llm_provider import LLMResponse, LLMUsage, ToolCall
from agent import (
    validate_zip,
    zipgeocode,
    grid_finder,
    http_get_with_retry,
    dispatch_tool,
    run_agent,
    Metrics,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ZIPPOPOTAM_OK = {
    "post code": "90210",
    "places": [{
        "place name": "Beverly Hills",
        "state abbreviation": "CA",
        "latitude": "34.0901",
        "longitude": "-118.4065",
    }]
}

NWS_POINTS_OK = {
    "properties": {
        "gridId": "LOX",
        "gridX": 149,
        "gridY": 48,
        "relativeLocation": {
            "properties": {"city": "Beverly Hills", "state": "CA"}
        }
    }
}

NWS_FORECAST_OK = {
    "properties": {
        "periods": [
            {
                "name": "Tonight",
                "temperature": 62,
                "temperatureUnit": "F",
                "windSpeed": "5 mph",
                "windDirection": "W",
                "shortForecast": "Clear",
            },
            {
                "name": "Wednesday",
                "temperature": 72,
                "temperatureUnit": "F",
                "windSpeed": "10 mph",
                "windDirection": "SW",
                "shortForecast": "Sunny",
            },
            {
                "name": "Wednesday Night",
                "temperature": 58,
                "temperatureUnit": "F",
                "windSpeed": "5 mph",
                "windDirection": "W",
                "shortForecast": "Partly Cloudy",
            },
        ]
    }
}


def make_mock_response(status_code, json_data=None):
    """Build a mock requests.Response with a given status code and JSON body."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    return mock


# ── MockProvider ──────────────────────────────────────────────────────────────

class MockProvider:
    """
    Test double for LLMProvider.
    Feed pre-built LLMResponse objects via the responses list.
    Implements the same interface as LLMProvider without ABC overhead.
    """

    def __init__(self, responses):
        self._responses = iter(responses)

    def chat(self, messages, tools, system=""):
        return next(self._responses)

    def format_assistant_message(self, response):
        return {"role": "assistant", "content": ""}

    def format_tool_messages(self, tool_calls, results):
        return [{"role": "user", "content": results}]

    def convert_tools(self, tools):
        return tools


def make_tool_response(tool_calls_data, in_tok=100, out_tok=30):
    """Build an LLMResponse with stop_reason='tool_use'."""
    tcs = [
        ToolCall(id=d["id"], name=d["name"], input=d["input"])
        for d in tool_calls_data
    ]
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=tcs,
        text="",
        usage=LLMUsage(input_tokens=in_tok, output_tokens=out_tok),
        raw=None,
    )


def make_text_response(text="Done.", in_tok=100, out_tok=30):
    """Build an LLMResponse with stop_reason='end_turn'."""
    return LLMResponse(
        stop_reason="end_turn",
        tool_calls=[],
        text=text,
        usage=LLMUsage(input_tokens=in_tok, output_tokens=out_tok),
        raw=None,
    )


# ── Happy path response sequence ──────────────────────────────────────────────

_FORECAST_DATA = [
    {"period": "Tonight", "temp": "62F", "wind": "5 mph W", "summary": "Clear"},
    {"period": "Wednesday", "temp": "72F", "wind": "10 mph SW", "summary": "Sunny"},
]

_HAPPY_RESPONSES = [
    make_tool_response([{"id": "tu_1", "name": "validate_zip", "input": {"zip_code": "90210"}}]),
    make_tool_response([{"id": "tu_2", "name": "zipgeocode",   "input": {"zip_code": "90210"}}]),
    make_tool_response([{"id": "tu_3", "name": "grid_finder",  "input": {"latitude": "34.09", "longitude": "-118.40"}}]),
    make_text_response("Tonight: Clear 62F. Wednesday: Sunny 72F."),
]

_DISPATCH_HAPPY = [
    json.dumps({"is_valid": True}),
    json.dumps({"city": "Beverly Hills", "state": "CA", "lat": "34.09", "lon": "-118.40"}),
    json.dumps({"location": "Beverly Hills, CA", "forecast": _FORECAST_DATA}),
]


# ── 1. TestValidateZip ────────────────────────────────────────────────────────

class TestValidateZip(unittest.TestCase):
    """Pure unit tests — no mocks needed."""

    def test_valid_5digit(self):
        self.assertTrue(validate_zip("90210")["is_valid"])

    def test_valid_nyc(self):
        self.assertTrue(validate_zip("10001")["is_valid"])

    def test_invalid_letters(self):
        result = validate_zip("abc12")
        self.assertFalse(result["is_valid"])
        self.assertIn("reason", result)

    def test_invalid_too_short(self):
        self.assertFalse(validate_zip("1234")["is_valid"])

    def test_invalid_too_long(self):
        self.assertFalse(validate_zip("123456")["is_valid"])

    def test_invalid_empty(self):
        self.assertFalse(validate_zip("")["is_valid"])

    def test_invalid_with_leading_trailing_spaces(self):
        # validate_zip strips input, so " 90210 " must pass
        self.assertTrue(validate_zip(" 90210 ")["is_valid"])


# ── 2. TestZipgeocode ─────────────────────────────────────────────────────────

class TestZipgeocode(unittest.TestCase):

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_success(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(200, ZIPPOPOTAM_OK)
        result = zipgeocode("90210")
        self.assertEqual(result["city"], "Beverly Hills")
        self.assertEqual(result["state"], "CA")
        self.assertEqual(result["lat"], "34.0901")
        self.assertEqual(result["lon"], "-118.4065")

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_zip_not_found(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(404)
        result = zipgeocode("00000")
        self.assertIn("error", result)
        self.assertIn("not found", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_service_error_500(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(500)
        result = zipgeocode("90210")
        self.assertIn("error", result)
        self.assertIn("500", result["error"])

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_network_timeout(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        result = zipgeocode("90210")
        self.assertIn("error", result)
        self.assertIn("unavailable", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_malformed_response(self, mock_get, mock_sleep):
        # Valid 200 but JSON missing the "places" key
        mock_get.return_value = make_mock_response(200, {"post code": "90210"})
        result = zipgeocode("90210")
        self.assertIn("error", result)
        self.assertIn("Unexpected", result["error"])


# ── 3. TestGridFinder ─────────────────────────────────────────────────────────

class TestGridFinder(unittest.TestCase):

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_success(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            make_mock_response(200, NWS_POINTS_OK),
            make_mock_response(200, NWS_FORECAST_OK),
        ]
        result = grid_finder("34.09", "-118.40")
        self.assertEqual(result["location"], "Beverly Hills, CA")
        self.assertEqual(len(result["forecast"]), 2)
        self.assertEqual(result["forecast"][0]["period"], "Tonight")
        self.assertEqual(result["forecast"][1]["period"], "Wednesday")

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_nws_422_offshore(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(422)
        result = grid_finder("0.00", "0.00")
        self.assertIn("error", result)
        self.assertIn("not supported", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_nws_404(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(404)
        result = grid_finder("0.00", "0.00")
        self.assertIn("error", result)
        self.assertIn("not supported", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_points_network_error(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.ConnectionError("Network down")
        result = grid_finder("34.09", "-118.40")
        self.assertIn("error", result)
        self.assertIn("unavailable", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_forecast_network_error(self, mock_get, mock_sleep):
        # Points succeeds; all 3 forecast retries fail
        mock_get.side_effect = [
            make_mock_response(200, NWS_POINTS_OK),
            requests.ConnectionError("Network down"),
            requests.ConnectionError("Network down"),
            requests.ConnectionError("Network down"),
        ]
        result = grid_finder("34.09", "-118.40")
        self.assertIn("error", result)
        self.assertIn("unavailable", result["error"].lower())

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_forecast_500(self, mock_get, mock_sleep):
        # Points succeeds; all 3 forecast retries return 500
        mock_get.side_effect = [
            make_mock_response(200, NWS_POINTS_OK),
            make_mock_response(500),
            make_mock_response(500),
            make_mock_response(500),
        ]
        result = grid_finder("34.09", "-118.40")
        self.assertIn("error", result)
        self.assertIn("500", result["error"])

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_forecast_truncated_to_2(self, mock_get, mock_sleep):
        """NWS returns 3 periods; only first 2 must be included in result."""
        mock_get.side_effect = [
            make_mock_response(200, NWS_POINTS_OK),
            make_mock_response(200, NWS_FORECAST_OK),  # fixture has 3 periods
        ]
        result = grid_finder("34.09", "-118.40")
        self.assertNotIn("error", result)
        self.assertEqual(len(result["forecast"]), 2)


# ── 4. TestHttpGetWithRetry ───────────────────────────────────────────────────

class TestHttpGetWithRetry(unittest.TestCase):

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_success_first_try(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(200)
        resp = http_get_with_retry("http://example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_retries_on_500_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            make_mock_response(500),
            make_mock_response(500),
            make_mock_response(200),
        ]
        resp = http_get_with_retry("http://example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_retries_on_connection_error_then_succeeds(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            requests.ConnectionError(),
            requests.ConnectionError(),
            make_mock_response(200),
        ]
        resp = http_get_with_retry("http://example.com")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_get.call_count, 3)

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_exhausts_retries_raises(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.ConnectionError("Permanent failure")
        with self.assertRaises(requests.RequestException):
            http_get_with_retry("http://example.com")
        self.assertEqual(mock_get.call_count, 3)

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_non_retryable_400_no_retry(self, mock_get, mock_sleep):
        mock_get.return_value = make_mock_response(400)
        resp = http_get_with_retry("http://example.com")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("agent.time.sleep")
    @patch("agent.requests.get")
    def test_increments_metrics_retries(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            make_mock_response(500),
            make_mock_response(200),
        ]
        m = Metrics()
        http_get_with_retry("http://example.com", metrics=m)
        self.assertEqual(m.retries, 1)


# ── 5. TestDispatchTool ───────────────────────────────────────────────────────

class TestDispatchTool(unittest.TestCase):

    def test_valid_tool_validate_zip(self):
        result = json.loads(dispatch_tool("validate_zip", {"zip_code": "90210"}, Metrics()))
        self.assertTrue(result["is_valid"])

    def test_unknown_tool(self):
        result = json.loads(dispatch_tool("nonexistent_tool", {}, Metrics()))
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    @patch("agent.validate_zip", side_effect=RuntimeError("Simulated crash"))
    def test_tool_exception_caught(self, mock_validate):
        result = json.loads(dispatch_tool("validate_zip", {"zip_code": "90210"}, Metrics()))
        self.assertIn("error", result)
        self.assertIn("Tool execution error", result["error"])


# ── 6. TestRunAgent ───────────────────────────────────────────────────────────

class TestRunAgent(unittest.TestCase):

    @patch("agent.dispatch_tool")
    def test_happy_path_returns_true(self, mock_dispatch):
        mock_dispatch.side_effect = list(_DISPATCH_HAPPY)
        provider = MockProvider(list(_HAPPY_RESPONSES))
        self.assertTrue(run_agent(provider, "90210", Metrics()))

    @patch("agent.dispatch_tool")
    def test_invalid_zip_returns_false(self, mock_dispatch):
        mock_dispatch.return_value = json.dumps({"is_valid": False, "reason": "Must be exactly 5 digits"})
        provider = MockProvider([
            make_tool_response([{"id": "tu_1", "name": "validate_zip", "input": {"zip_code": "abc"}}]),
        ])
        self.assertFalse(run_agent(provider, "abc", Metrics()))

    @patch("agent.dispatch_tool")
    def test_geocode_error_returns_true(self, mock_dispatch):
        """When geocoding fails, model reports the error at end_turn; run_agent returns True."""
        mock_dispatch.side_effect = [
            json.dumps({"is_valid": True}),
            json.dumps({"error": "ZIP 00000 not found in database."}),
        ]
        provider = MockProvider([
            make_tool_response([{"id": "tu_1", "name": "validate_zip", "input": {"zip_code": "00000"}}]),
            make_tool_response([{"id": "tu_2", "name": "zipgeocode",   "input": {"zip_code": "00000"}}]),
            make_text_response("Sorry, ZIP 00000 was not found."),
        ])
        self.assertTrue(run_agent(provider, "00000", Metrics()))

    @patch("agent.dispatch_tool")
    def test_metrics_populated(self, mock_dispatch):
        """API call count and token totals must be accumulated across all 4 turns."""
        mock_dispatch.side_effect = list(_DISPATCH_HAPPY)
        provider = MockProvider(list(_HAPPY_RESPONSES))
        m = Metrics()
        run_agent(provider, "90210", m)
        self.assertEqual(m.api_calls, 4)
        self.assertEqual(m.input_tokens,  4 * 100)   # 100 per MockResponse
        self.assertEqual(m.output_tokens, 4 * 30)    # 30 per MockResponse


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
