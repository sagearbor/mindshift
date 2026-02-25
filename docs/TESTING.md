# MindShift Testing Guide

## Running Tests

```bash
# All tests (mocked — no API key needed)
pytest server/tests/ tests/

# Only server unit tests
pytest server/tests/

# Only integration / tone baseline tests
pytest tests/
```

## LLM Response Cache

The `LLMResponseCache` class (`server/test_cache.py`) prevents re-spending API credits on unchanged prompts during development and CI.

### How It Works

1. **Cache key** = SHA-256 of `model + system_prompt + user_prompt`
2. **On miss**: calls the real LLM API, stores response in `tests/fixtures/llm_cache/{key}.json`
3. **On hit**: loads from disk, skips the API call entirely
4. **Forced refresh**: set `REFRESH_LLM_CACHE=1` to re-run even on cache hit

### Cache Invalidation

The cache auto-invalidates when any of these change:
- The **model name** (e.g. switching from `claude-3-haiku` to `gpt-4o`)
- The **system prompt** (any wording change)
- The **user prompt** (any input change)

To manually invalidate:
```bash
# Refresh all cached responses
REFRESH_LLM_CACHE=1 ANTHROPIC_API_KEY=sk-... pytest tests/test_tone_baseline.py

# Delete a specific cached response
rm tests/fixtures/llm_cache/<sha256>.json

# Clear entire cache
rm tests/fixtures/llm_cache/*.json
```

### When to Refresh

- After changing scoring prompt wording in `test_tone_baseline.py`
- After switching the default model
- When you suspect cached responses are stale or incorrect
- Before publishing benchmark results

### File Layout

```
tests/fixtures/llm_cache/
├── .gitkeep          # Tracked — keeps directory in git
├── .gitignore        # Ignores *.json so real API responses aren't committed
└── <sha256>.json     # Cached response files (local only)
```

### Using the Cache in Tests

Tests in `test_tone_baseline.py` use `LLMResponseCache` when `ANTHROPIC_API_KEY` is set:

```bash
# Without API key — all tests mock the LLM (default, fast)
pytest tests/test_tone_baseline.py

# With API key — live-cached tests also run
ANTHROPIC_API_KEY=sk-... pytest tests/test_tone_baseline.py
```

## Auth / Patient Model

The `POST /auth/session` endpoint creates session tokens linking a therapist to a patient. Pass `X-Session-Token` header on `POST /session` to associate data sessions with a patient. Query endpoints:

- `GET /therapist/{id}/patients` — list all patients with session counts
- `GET /therapist/{id}/patient/{pid}/sessions` — list sessions for a patient
