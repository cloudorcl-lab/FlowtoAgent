# FlowtoAgent — Project Standards

## This project is the reference implementation for the Anthropic Tool-Use Agent SOP.

Skill: `/anthropic-tool-use-agent`
SOP: `SOP_Anthropic_Tool_Use_Agent.md`

## Environment configuration

All LLM config lives in the environment — never hardcoded in source:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | API authentication |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5` | Model selection |
| `ANTHROPIC_MAX_TOKENS` | No | `512` | Output token cap |

- `.env.example` is committed — copy to `.env` and fill in values
- `.env` is gitignored — never commit real keys
- To swap model: set `ANTHROPIC_MODEL=claude-sonnet-4-6` in `.env`, no code change needed

## Architecture constraints (do not change without justification)

- Default model: `claude-haiku-4-5` — override via env, document why if upgrading
- Default `max_tokens`: 512 — sufficient for tool routing + summary
- Thinking: disabled — not needed for tool orchestration
- Tools: never raise exceptions — return `{"error": "..."}` dicts
- Metrics: wire `Metrics` through every function that calls an external API

## Test requirements

- All tests in `test_agent.py` must pass before any release
- No new test dependencies — use `unittest` + `unittest.mock` only
- Always patch `agent.time.sleep` in retry tests
- Run: `python -m unittest test_agent.py -v`

## Release checklist

- [ ] All 32 tests pass
- [ ] Live end-to-end run completed
- [ ] `metrics.print_summary()` shows 0 unexpected retries
- [ ] README updated if behavior changed
- [ ] Committed and pushed to GitHub
