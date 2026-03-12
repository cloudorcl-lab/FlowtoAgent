# FlowtoAgent — Project Standards

## This project is the reference implementation for the Anthropic Tool-Use Agent SOP.

Skill: `/anthropic-tool-use-agent`
SOP: `SOP_Anthropic_Tool_Use_Agent.md`

## Environment configuration

All LLM config lives in the environment — never hardcoded in source:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | No | `claude` | Provider: `claude`, `openai`, `gemini` |
| `LLM_MAX_TOKENS` | No | `512` | Output token cap (all providers) |
| `ANTHROPIC_API_KEY` | When `LLM_PROVIDER=claude` | — | API authentication |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | Claude model selection |
| `OPENAI_API_KEY` | When `LLM_PROVIDER=openai` | — | API authentication |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model selection |
| `GEMINI_API_KEY` | When `LLM_PROVIDER=gemini` | — | API authentication |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | Gemini model selection |

- `.env.example` is committed — copy to `.env` and fill in values
- `.env` is gitignored — never commit real keys
- To swap provider: set `LLM_PROVIDER=openai` in `.env`, no code change needed

## Architecture constraints (do not change without justification)

- All LLM interaction goes through `LLMProvider` in `llm_provider.py` — never import provider SDKs directly in `agent.py`
- Default provider: `claude` with `claude-haiku-4-5` — cheapest capable model for tool routing
- Default `max_tokens`: 512 — sufficient for tool routing + summary
- Thinking: disabled — not needed for tool orchestration
- Tools: never raise exceptions — return `{"error": "..."}` dicts
- Metrics: wire `Metrics` through every function that calls an external API

## Test requirements

- `test_agent.py` — 32 tests covering all agent logic (no LLM/HTTP mocks needed via MockProvider)
- `test_llm_provider.py` — provider schema conversion and factory tests
- No new test dependencies — use `unittest` + `unittest.mock` only
- Always patch `agent.time.sleep` in retry tests
- Run both suites: `python -m unittest test_agent.py test_llm_provider.py -v`

## Release checklist

- [ ] All tests pass (`test_agent.py` + `test_llm_provider.py`)
- [ ] Live end-to-end run completed (at least with `LLM_PROVIDER=claude`)
- [ ] `metrics.print_summary()` shows 0 unexpected retries
- [ ] README updated if behavior changed
- [ ] Committed and pushed to GitHub
