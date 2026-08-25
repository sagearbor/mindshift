"""Tests for the multi-vendor LLM abstraction layer."""

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

# `llm_client` is imported as a module (not just its names) because the hedged
# streaming tests below resolve attributes at use time — an earlier test in this
# file reloads the module.
import llm_client
from llm_client import LLMClient, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_claude_models(self):
        for model in ["claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-5-sonnet-20241022"]:
            assert LLMClient._detect_provider(model) == "anthropic"

    def test_openai_chat_models(self):
        for model in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
            assert LLMClient._detect_provider(model) == "openai"

    def test_openai_responses_models(self):
        for model in ["gpt-5", "gpt-5-reasoning", "o1-preview", "o1-mini", "o3-mini", "o4-mini"]:
            assert LLMClient._detect_provider(model) == "openai"

    def test_gemini_models(self):
        for model in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-pro"]:
            assert LLMClient._detect_provider(model) == "google"

    def test_mistral_models(self):
        for model in ["mistral-large", "mistral-small", "mistral-7b"]:
            assert LLMClient._detect_provider(model) == "mistral"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model provider"):
            LLMClient._detect_provider("llama-3-70b")


# ---------------------------------------------------------------------------
# Static helper methods
# ---------------------------------------------------------------------------

class TestIsReasoningModel:
    def test_o1_models(self):
        assert LLMClient.is_reasoning_model("o1-preview") is True
        assert LLMClient.is_reasoning_model("o1-mini") is True

    def test_o3_models(self):
        assert LLMClient.is_reasoning_model("o3-mini") is True

    def test_o4_models(self):
        assert LLMClient.is_reasoning_model("o4-mini") is True

    def test_gpt5_reasoning(self):
        assert LLMClient.is_reasoning_model("gpt-5-reasoning") is True

    def test_gpt5_non_reasoning(self):
        assert LLMClient.is_reasoning_model("gpt-5") is False

    def test_normal_models(self):
        assert LLMClient.is_reasoning_model("gpt-4o") is False
        assert LLMClient.is_reasoning_model("claude-3-haiku-20240307") is False
        assert LLMClient.is_reasoning_model("gemini-2.0-flash") is False


class TestUsesResponsesApi:
    def test_gpt5(self):
        assert LLMClient.uses_responses_api("gpt-5") is True
        assert LLMClient.uses_responses_api("gpt-5-reasoning") is True

    def test_o_series(self):
        assert LLMClient.uses_responses_api("o1-preview") is True
        assert LLMClient.uses_responses_api("o3-mini") is True
        assert LLMClient.uses_responses_api("o4-mini") is True

    def test_chat_completions_models(self):
        assert LLMClient.uses_responses_api("gpt-4o") is False
        assert LLMClient.uses_responses_api("gpt-4o-mini") is False
        assert LLMClient.uses_responses_api("gpt-4-turbo") is False
        assert LLMClient.uses_responses_api("gpt-3.5-turbo") is False

    def test_non_openai_models(self):
        assert LLMClient.uses_responses_api("claude-3-haiku-20240307") is False
        assert LLMClient.uses_responses_api("gemini-2.0-flash") is False


# ---------------------------------------------------------------------------
# Client construction — production timeouts/retries (P0-1)
# ---------------------------------------------------------------------------

class TestBuildClientConfig:
    def test_anthropic_gets_timeout_and_retries(self):
        with patch("anthropic.Anthropic") as mock_cls:
            LLMClient(model="claude-3-haiku-20240307", api_key="test-key")
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS
        assert kwargs["max_retries"] == MAX_RETRIES

    def test_openai_gets_timeout_and_retries(self, monkeypatch):
        fake_openai = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        LLMClient(model="gpt-4o-mini", api_key="test-key")
        kwargs = fake_openai.OpenAI.call_args.kwargs
        assert kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS
        assert kwargs["max_retries"] == MAX_RETRIES

    def test_mistral_gets_timeout_ms(self, monkeypatch):
        fake_mistralai = MagicMock()
        monkeypatch.setitem(sys.modules, "mistralai", fake_mistralai)
        LLMClient(model="mistral-large", api_key="test-key")
        kwargs = fake_mistralai.Mistral.call_args.kwargs
        assert kwargs["timeout_ms"] == REQUEST_TIMEOUT_SECONDS * 1000


# ---------------------------------------------------------------------------
# Shutdown — close() releases the provider client (P1-8)
# ---------------------------------------------------------------------------

class TestClose:
    def _make_client(self, underlying) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client.model = "claude-3-haiku-20240307"
        client._provider = "anthropic"
        client._api_key = None
        client._client = underlying
        return client

    def test_close_calls_underlying_close(self):
        underlying = MagicMock()
        self._make_client(underlying).close()
        underlying.close.assert_called_once()

    def test_close_without_close_method_is_noop(self):
        # e.g. the google genai module has no close() — must not raise.
        self._make_client(object()).close()

    def test_close_swallows_errors(self):
        underlying = MagicMock()
        underlying.close.side_effect = RuntimeError("pool already closed")
        self._make_client(underlying).close()  # must not raise


# ---------------------------------------------------------------------------
# Temperature rules
# ---------------------------------------------------------------------------

class TestTemperatureRules:
    def _make_client(self, model: str) -> LLMClient:
        """Create an LLMClient without calling real SDK constructors."""
        client = LLMClient.__new__(LLMClient)
        client.model = model
        client._provider = LLMClient._detect_provider(model)
        client._api_key = None
        client._client = MagicMock()
        return client

    def test_claude_passes_temperature(self):
        c = self._make_client("claude-3-haiku-20240307")
        assert c._resolve_temperature(0.7) == 0.7
        assert self._make_client("claude-haiku-4-5-20251001")._resolve_temperature(0.7) == 0.7
        assert self._make_client("claude-sonnet-4-6")._resolve_temperature(0.7) == 0.7
        assert self._make_client("claude-opus-4-6")._resolve_temperature(0.7) == 0.7

    def test_claude_5_family_omits_temperature(self):
        # The API rejects `temperature` on these with a 400 ("deprecated for
        # this model") — found by scripts/bench_suggestions.py --model claude-sonnet-5.
        for model in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5",
                      "claude-mythos-5", "claude-opus-4-7", "claude-opus-4-8"):
            assert self._make_client(model)._resolve_temperature(0.7) is None, model

    def test_anthropic_omitted_temperature_is_not_sent(self):
        underlying = MagicMock()
        underlying.messages.create.return_value.content = [MagicMock(text="{}")]
        c = self._make_client("claude-sonnet-5")
        c._client = underlying
        c.complete(system="s", user="u")
        assert "temperature" not in underlying.messages.create.call_args.kwargs

    def test_gpt4o_passes_temperature(self):
        c = self._make_client("gpt-4o-mini")
        assert c._resolve_temperature(0.5) == 0.5

    def test_o1_omits_temperature(self):
        c = self._make_client("o1-preview")
        assert c._resolve_temperature(0.7) is None

    def test_o3_omits_temperature(self):
        c = self._make_client("o3-mini")
        assert c._resolve_temperature(0.7) is None

    def test_o4_omits_temperature(self):
        c = self._make_client("o4-mini")
        assert c._resolve_temperature(0.5) is None

    def test_gpt5_reasoning_forces_1(self):
        c = self._make_client("gpt-5-reasoning")
        assert c._resolve_temperature(0.3) == 1.0

    def test_gpt5_non_reasoning_passes_temperature(self):
        c = self._make_client("gpt-5")
        assert c._resolve_temperature(0.8) == 0.8

    def test_gemini_passes_temperature(self):
        c = self._make_client("gemini-2.0-flash")
        assert c._resolve_temperature(0.9) == 0.9


# ---------------------------------------------------------------------------
# Completion routing (mocked SDK calls)
# ---------------------------------------------------------------------------

class TestCompleteAnthropic:
    def test_anthropic_complete(self):
        mock_sdk = MagicMock()
        block = MagicMock()
        block.text = "Hello from Claude"
        msg = MagicMock()
        msg.content = [block]
        mock_sdk.messages.create.return_value = msg

        client = LLMClient.__new__(LLMClient)
        client.model = "claude-3-haiku-20240307"
        client._provider = "anthropic"
        client._api_key = "test-key"
        client._client = mock_sdk

        result = client.complete(system="Be helpful", user="Hi")

        assert result == "Hello from Claude"
        call_kwargs = mock_sdk.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku-20240307"
        # Prompt caching is OFF by default (see llm_client.PROMPT_CACHE_ENABLED):
        # the request is byte-identical to before — a plain system string.
        assert call_kwargs["system"] == "Be helpful"
        assert "temperature" in call_kwargs
        assert "output_config" not in call_kwargs


# ---------------------------------------------------------------------------
# Prompt caching flags + usage accounting (perf/cloud-suggestion-latency)
# ---------------------------------------------------------------------------

class _Usage:
    def __init__(self, **kw):
        self.input_tokens = kw.get("input_tokens", 0)
        self.output_tokens = kw.get("output_tokens", 0)
        self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)
        self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 0)


class TestPromptCaching:
    def _make_client(self, underlying, *, cache_system_prompt: bool = True) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client.model = "claude-haiku-4-5-20251001"
        client._provider = "anthropic"
        client._api_key = None
        client._client = underlying
        client.cache_system_prompt = cache_system_prompt
        return client

    def test_default_is_off_unless_env_opts_in(self, monkeypatch):
        import importlib

        import llm_client

        monkeypatch.delenv("MINDSHIFT_PROMPT_CACHE", raising=False)
        assert importlib.reload(llm_client).PROMPT_CACHE_ENABLED is False
        monkeypatch.setenv("MINDSHIFT_PROMPT_CACHE", "1")
        assert importlib.reload(llm_client).PROMPT_CACHE_ENABLED is True
        monkeypatch.delenv("MINDSHIFT_PROMPT_CACHE", raising=False)
        importlib.reload(llm_client)

    def test_anthropic_system_blocks_helper(self):
        from llm_client import anthropic_system_blocks

        assert anthropic_system_blocks("sys", cache=False) == "sys"
        assert anthropic_system_blocks("sys", cache=True) == [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
        ]

    def test_cache_flag_off_sends_plain_string(self):
        underlying = MagicMock()
        underlying.messages.create.return_value.content = [MagicMock(text="{}")]
        c = self._make_client(underlying, cache_system_prompt=False)
        c.complete(system="sys", user="usr")
        assert underlying.messages.create.call_args.kwargs["system"] == "sys"

    def test_constructor_flag_overrides_env_default(self):
        from llm_client import PROMPT_CACHE_ENABLED

        with patch("anthropic.Anthropic"):
            default = LLMClient(model="claude-haiku-4-5-20251001", api_key="k")
            on = LLMClient(
                model="claude-haiku-4-5-20251001", api_key="k", cache_system_prompt=True,
            )
            off = LLMClient(
                model="claude-haiku-4-5-20251001", api_key="k", cache_system_prompt=False,
            )
        assert default.cache_system_prompt is PROMPT_CACHE_ENABLED
        assert on.cache_system_prompt is True
        assert off.cache_system_prompt is False

    def test_stream_puts_cache_marker_on_system_block(self):
        underlying = MagicMock()
        stream = underlying.messages.stream.return_value.__enter__.return_value
        stream.text_stream = iter(["a", "b"])
        stream.get_final_message.return_value.usage = _Usage(
            input_tokens=12, output_tokens=2, cache_read_input_tokens=100,
        )
        c = self._make_client(underlying)
        assert "".join(c.stream_complete(system="sys", user="usr")) == "ab"
        kwargs = underlying.messages.stream.call_args.kwargs
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["system"][0]["text"] == "sys"

    def test_usage_is_accumulated_from_streamed_and_plain_calls(self):
        underlying = MagicMock()
        stream = underlying.messages.stream.return_value.__enter__.return_value
        stream.text_stream = iter(["x"])
        stream.get_final_message.return_value.usage = _Usage(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=1500, cache_read_input_tokens=0,
        )
        msg = MagicMock()
        msg.content = [MagicMock(text="{}")]
        msg.usage = _Usage(
            input_tokens=10, output_tokens=7,
            cache_creation_input_tokens=0, cache_read_input_tokens=1500,
        )
        underlying.messages.create.return_value = msg
        c = self._make_client(underlying)

        list(c.stream_complete(system="sys", user="usr"))
        c.complete(system="sys", user="usr")

        assert c.usage_totals == {
            "calls": 2, "input_tokens": 20, "output_tokens": 12,
            "cache_creation_input_tokens": 1500, "cache_read_input_tokens": 1500,
        }
        assert c.last_usage["cache_read_input_tokens"] == 1500

    def test_usage_read_failure_never_breaks_a_reply(self):
        underlying = MagicMock()
        stream = underlying.messages.stream.return_value.__enter__.return_value
        stream.text_stream = iter(["ok"])
        stream.get_final_message.side_effect = RuntimeError("no final message")
        c = self._make_client(underlying)
        assert list(c.stream_complete(system="sys", user="usr")) == ["ok"]
        assert c.usage_totals["calls"] == 0

    def test_response_schema_becomes_output_config(self):
        underlying = MagicMock()
        underlying.messages.create.return_value.content = [MagicMock(text="{}")]
        c = self._make_client(underlying)
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        c.complete(system="sys", user="usr", response_schema=schema)
        assert underlying.messages.create.call_args.kwargs["output_config"] == {
            "format": {"type": "json_schema", "schema": schema},
        }

    def test_response_schema_is_ignored_by_other_providers(self):
        underlying = MagicMock()
        underlying.chat.completions.create.return_value.choices[0].message.content = "{}"
        c = LLMClient.__new__(LLMClient)
        c.model, c._provider, c._api_key, c._client = "gpt-4o-mini", "openai", None, underlying
        c.complete(system="sys", user="usr", response_schema={"type": "object"})
        assert "output_config" not in underlying.chat.completions.create.call_args.kwargs


class TestCompleteOpenAIChat:
    def test_openai_chat_complete(self):
        mock_sdk = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "Hello from GPT-4o"
        mock_sdk.chat.completions.create.return_value = response

        client = LLMClient.__new__(LLMClient)
        client.model = "gpt-4o-mini"
        client._provider = "openai"
        client._api_key = "test-key"
        client._client = mock_sdk

        result = client.complete(system="Be helpful", user="Hi", temperature=0.5)

        assert result == "Hello from GPT-4o"
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.5
        assert len(call_kwargs["messages"]) == 2


class TestCompleteOpenAIResponses:
    def test_responses_api_complete(self):
        mock_sdk = MagicMock()
        response = MagicMock()
        response.output_text = "Hello from GPT-5"
        mock_sdk.responses.create.return_value = response

        client = LLMClient.__new__(LLMClient)
        client.model = "gpt-5"
        client._provider = "openai"
        client._api_key = "test-key"
        client._client = mock_sdk

        result = client.complete(system="Be helpful", user="Hi")

        assert result == "Hello from GPT-5"
        call_kwargs = mock_sdk.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5"
        assert "input" in call_kwargs

    def test_o1_omits_temperature(self):
        mock_sdk = MagicMock()
        response = MagicMock()
        response.output_text = "Reasoned response"
        mock_sdk.responses.create.return_value = response

        client = LLMClient.__new__(LLMClient)
        client.model = "o1-preview"
        client._provider = "openai"
        client._api_key = "test-key"
        client._client = mock_sdk

        client.complete(system="Be helpful", user="Hi", temperature=0.7)

        call_kwargs = mock_sdk.responses.create.call_args.kwargs
        assert "temperature" not in call_kwargs


class TestCompleteGoogle:
    def test_google_complete(self):
        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = LLMClient.__new__(LLMClient)
        client.model = "gemini-2.0-flash"
        client._provider = "google"
        client._api_key = "test-key"
        client._client = mock_genai

        result = client.complete(system="Be helpful", user="Hi")
        assert result == "Hello from Gemini"


class TestCompleteMistral:
    def test_mistral_complete(self):
        mock_sdk = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello from Mistral"
        mock_sdk.chat.complete.return_value = mock_response

        client = LLMClient.__new__(LLMClient)
        client.model = "mistral-large"
        client._provider = "mistral"
        client._api_key = "test-key"
        client._client = mock_sdk

        result = client.complete(system="Be helpful", user="Hi")
        assert result == "Hello from Mistral"
        call_kwargs = mock_sdk.chat.complete.call_args.kwargs
        assert call_kwargs["model"] == "mistral-large"
        assert "temperature" in call_kwargs


# ---------------------------------------------------------------------------
# stream_complete — token streaming on Anthropic, one-chunk fallback elsewhere
# ---------------------------------------------------------------------------

class TestStreamComplete:
    def _make_client(self, model: str, underlying) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client.model = model
        client._provider = LLMClient._detect_provider(model)
        client._api_key = None
        client._client = underlying
        return client

    def test_anthropic_yields_text_deltas_from_the_stream_helper(self):
        underlying = MagicMock()
        stream = underlying.messages.stream.return_value.__enter__.return_value
        stream.text_stream = iter(['{"sugg', 'estions": ["Hi."]}'])
        c = self._make_client("claude-3-haiku-20240307", underlying)

        deltas = list(c.stream_complete(system="sys", user="usr"))

        assert deltas == ['{"sugg', 'estions": ["Hi."]}']
        kwargs = underlying.messages.stream.call_args.kwargs
        assert kwargs["system"] == "sys"  # caching off by default — see TestPromptCaching
        assert kwargs["messages"] == [{"role": "user", "content": "usr"}]
        assert kwargs["temperature"] == 0.7 and kwargs["max_tokens"] == 512
        # The context manager is exited so the SSE connection is released.
        underlying.messages.stream.return_value.__exit__.assert_called_once()

    def test_non_anthropic_falls_back_to_one_chunk_of_complete(self):
        underlying = MagicMock()
        underlying.chat.completions.create.return_value.choices[0].message.content = "whole"
        c = self._make_client("gpt-4o-mini", underlying)
        assert list(c.stream_complete(system="sys", user="usr")) == ["whole"]


# ---------------------------------------------------------------------------
# Hedged streaming (perf/llm-hedging)
# ---------------------------------------------------------------------------

class FakeSSE:
    """One scripted streaming attempt, shaped like the SDK's MessageStream
    context manager. ``stall=True`` blocks the first byte until ``release``
    is set — by the test, or by ``close()`` (what a cancelled attempt sees).
    Deterministic: nothing here sleeps; a stalled attempt waits forever
    until something happens to it."""

    def __init__(self, deltas=("a", "b"), *, stall=False, error=None, usage=None):
        self.deltas = list(deltas)
        self.release = threading.Event()
        if not stall:
            self.release.set()
        self.error = error
        self.usage = usage
        self.closed = False
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True
        self.release.set()

    @property
    def text_stream(self):
        def gen():
            self.release.wait()
            if self.closed:
                return  # cancelled before its first byte
            if self.error is not None:
                raise self.error
            yield from self.deltas
        return gen()

    def get_final_message(self):
        return SimpleNamespace(usage=self.usage)


def scripted(*attempts):
    """``open_stream`` handing out the scripted attempts in order."""
    pending = list(attempts)
    lock = threading.Lock()

    def open_stream():
        with lock:
            assert pending, "more attempts opened than scripted"
            return pending.pop(0)
    return open_stream


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    return predicate()


class TestHedgedStream:
    def test_healthy_stream_never_hedges(self):
        usage = SimpleNamespace(input_tokens=10, output_tokens=2)
        first = FakeSSE(["he", "llo"], usage=usage)
        seen_usage, events = [], []
        hs = llm_client.HedgedStream(scripted(first), hedge_after_ms=10, deadline_ms=200,
                          on_usage=seen_usage.append, on_event=lambda e, s: events.append(e))
        assert "".join(hs) == "hello"
        assert hs.hedged is False and hs.hedge_won is False and hs.attempts == 1
        assert hs.first_token_ms is not None and hs.first_token_ms < 200
        assert seen_usage == [usage] and hs.usage is usage
        assert events == ["start", "first_token", "done"]
        assert first.closed  # drained → context manager exited

    def test_stalled_first_attempt_is_hedged_and_the_hedge_wins(self):
        first = FakeSSE(["never"], stall=True)
        second = FakeSSE(["hed", "ged"], usage=SimpleNamespace(input_tokens=7))
        events = []
        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=10, deadline_ms=2000,
                          on_event=lambda e, s: events.append(e))
        assert "".join(hs) == "hedged"
        assert hs.hedged is True and hs.hedge_won is True and hs.attempts == 2
        assert events == ["start", "hedged", "first_token", "done"]
        # The loser was cancelled: its SSE closed so it stops generating.
        assert first.closed and wait_for(lambda: first.release.is_set())
        assert second.closed  # drained

    def test_hedge_fires_but_the_original_answers_first(self):
        first = FakeSSE(["orig", "inal"], stall=True)
        second = FakeSSE(["late"], stall=True)

        def on_event(event, _stream):
            if event == "hedged":
                first.release.set()  # the original wakes up right after the hedge fired

        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=10, deadline_ms=2000,
                          on_event=on_event)
        assert "".join(hs) == "original"
        assert hs.hedged is True and hs.hedge_won is False and hs.attempts == 2
        assert wait_for(lambda: second.closed)  # the hedge was cancelled

    def test_deadline_abandons_the_call(self):
        first, second = FakeSSE(stall=True), FakeSSE(stall=True)
        events = []
        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=5, deadline_ms=40,
                          on_event=lambda e, s: events.append(e))
        with pytest.raises(llm_client.LLMFirstTokenTimeout) as info:
            list(hs)
        assert info.value.attempts == 2 and info.value.deadline_ms == 40
        assert hs.hedged is True and hs.hedge_won is False and hs.first_token_ms is None
        assert events == ["start", "hedged", "slow_llm"]
        assert wait_for(lambda: first.closed and second.closed)

    def test_error_from_the_only_live_attempt_propagates(self):
        boom = RuntimeError("provider down")
        hs = llm_client.HedgedStream(scripted(FakeSSE(error=boom)), hedge_after_ms=500, deadline_ms=2000)
        with pytest.raises(RuntimeError, match="provider down"):
            list(hs)
        assert hs.hedged is False

    def test_error_from_one_attempt_falls_through_to_the_other(self):
        first = FakeSSE(error=RuntimeError("529 overloaded"), stall=True)
        second = FakeSSE(["ok"], stall=True)

        def on_event(event, _stream):
            if event == "hedged":
                first.release.set()   # the original fails...
                second.release.set()  # ...and the hedge answers

        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=10, deadline_ms=2000,
                          on_event=on_event)
        assert "".join(hs) == "ok"
        assert hs.hedged is True and hs.hedge_won is True

    def test_consumer_stopping_early_closes_the_winner(self):
        first = FakeSSE(["one", "two", "three"])
        hs = llm_client.HedgedStream(scripted(first), hedge_after_ms=10, deadline_ms=200)
        it = iter(hs)
        assert next(it) == "one"
        it.close()
        assert first.closed

    def test_empty_completion_counts_as_an_answer(self):
        first = FakeSSE([], usage=SimpleNamespace(input_tokens=3))
        seen = []
        hs = llm_client.HedgedStream(scripted(first), hedge_after_ms=10, deadline_ms=200, on_usage=seen.append)
        assert list(hs) == []
        assert hs.hedged is False and len(seen) == 1

    def test_zero_hedge_after_disables_hedging(self):
        first = FakeSSE(stall=True)
        hs = llm_client.HedgedStream(scripted(first), hedge_after_ms=0, deadline_ms=30)
        with pytest.raises(llm_client.LLMFirstTokenTimeout) as info:
            list(hs)
        assert info.value.attempts == 1 and hs.hedged is False

    def test_zero_deadline_disables_the_cap(self):
        first, second = FakeSSE(stall=True), FakeSSE(["fine"])
        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=10, deadline_ms=0)
        assert "".join(hs) == "fine" and hs.hedge_won is True

    def test_straggler_from_a_cancelled_attempt_is_ignored(self):
        """A loser that wakes up AFTER the winner is chosen must not leak
        its deltas into the answer."""
        first = FakeSSE(["LEAK"], stall=True)
        second = FakeSSE(["good"], stall=True)

        def on_event(event, stream):
            if event == "hedged":
                second.release.set()
            if event == "first_token":
                # The loser is closed by now; even if it produced a delta it
                # would be a straggler — simulate a late wake-up anyway.
                first.closed = False
                first.release.set()

        hs = llm_client.HedgedStream(scripted(first, second), hedge_after_ms=10, deadline_ms=2000,
                          on_event=on_event)
        assert "".join(hs) == "good"


class TestClientHedgeWiring:
    def _make_client(self, underlying) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client.model = "claude-3-haiku-20240307"
        client._provider = "anthropic"
        client._api_key = None
        client._client = underlying
        return client

    def test_defaults_come_from_env_knobs(self):
        assert llm_client.LLM_HEDGE_AFTER_MS == 1500
        assert llm_client.LLM_FIRST_TOKEN_DEADLINE_MS == 6000

    def test_stream_complete_returns_the_hedged_stream_with_accounting(self, monkeypatch):
        monkeypatch.setattr(llm_client, "LLM_HEDGE_AFTER_MS", 10)
        monkeypatch.setattr(llm_client, "LLM_FIRST_TOKEN_DEADLINE_MS", 2000)
        first = FakeSSE(stall=True)
        second = FakeSSE(['{"ok"', ": 1}"], usage=SimpleNamespace(
            input_tokens=42, output_tokens=5, cache_creation_input_tokens=0,
            cache_read_input_tokens=0))
        underlying = MagicMock()
        opener = scripted(first, second)
        underlying.messages.stream.side_effect = lambda **kw: opener()
        c = self._make_client(underlying)

        stream = c.stream_complete(system="sys", user="usr", max_tokens=99)
        assert isinstance(stream, llm_client.HedgedStream)
        assert "".join(stream) == '{"ok": 1}'
        assert stream.hedged and stream.hedge_won
        # Both attempts sent the identical request.
        assert underlying.messages.stream.call_count == 2
        for call in underlying.messages.stream.call_args_list:
            assert call.kwargs["max_tokens"] == 99 and call.kwargs["system"] == "sys"
        # Usage: the winner's, once. Hedge cost: the loser's (identical) prompt.
        assert c.usage_totals["calls"] == 1 and c.usage_totals["input_tokens"] == 42
        assert c.hedge_totals == {
            "streams": 1, "hedged": 1, "hedge_won": 1, "slow_llm": 0,
            "hedge_extra_input_tokens": 42,
        }

    def test_slow_llm_is_counted_and_raised(self, monkeypatch):
        monkeypatch.setattr(llm_client, "LLM_HEDGE_AFTER_MS", 5)
        monkeypatch.setattr(llm_client, "LLM_FIRST_TOKEN_DEADLINE_MS", 40)
        underlying = MagicMock()
        underlying.messages.stream.side_effect = lambda **kw: FakeSSE(stall=True)
        c = self._make_client(underlying)
        with pytest.raises(llm_client.LLMFirstTokenTimeout):
            list(c.stream_complete(system="sys", user="usr"))
        assert c.hedge_totals["slow_llm"] == 1 and c.hedge_totals["hedged"] == 1
        assert c.usage_totals["calls"] == 0  # nothing was answered, nothing recorded

    def test_healthy_call_costs_nothing_extra(self):
        underlying = MagicMock()
        underlying.messages.stream.side_effect = lambda **kw: FakeSSE(
            ["x"], usage=SimpleNamespace(input_tokens=9, output_tokens=1))
        c = self._make_client(underlying)
        assert list(c.stream_complete(system="sys", user="usr")) == ["x"]
        assert underlying.messages.stream.call_count == 1
        assert c.hedge_totals == {
            "streams": 1, "hedged": 0, "hedge_won": 0, "slow_llm": 0,
            "hedge_extra_input_tokens": 0,
        }
