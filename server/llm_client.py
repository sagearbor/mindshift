"""Multi-vendor LLM abstraction layer.

Auto-detects provider from model name and routes to the correct SDK/API.
Handles temperature rules per PRD Section 12.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Iterator

logger = logging.getLogger(__name__)

# Production guardrails: without an explicit timeout the Anthropic/OpenAI SDKs
# default to a 600s request timeout with 2 retries — a single hung call could
# occupy a worker thread for 30 minutes. Fail fast instead.
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1

# Anthropic prompt caching: with ``MINDSHIFT_PROMPT_CACHE=1`` the system
# prompt is sent as a content block carrying ``cache_control`` so a stable
# prefix (the coaching stance + output contract, identical for every turn of
# a session) is served from the prompt cache instead of re-processed. The
# per-call ``usage`` fields say whether it engaged (:attr:`LLMClient.usage_totals`).
#
# OFF by default, by measurement (scripts/bench_suggestions.py, 2026-08-24):
# the live coaching prompts are ~200 tokens, far below Haiku 4.5's 4096-token
# cacheable minimum, so the marker never produced a cache read
# (cache_read_input_tokens stayed 0 over 150+ calls) — and requests carrying
# it showed a much worse time-to-first-token tail (p95 2.3–10.5 s vs
# 0.9–1.6 s without, in three interleaved A/B blocks; p50 identical). Turn it
# on only for a deployment whose system prompt is long enough to cache on
# its model (Opus 5 / Sonnet 5 minimums are 512 / 1024 tokens).
PROMPT_CACHE_ENABLED = os.getenv("MINDSHIFT_PROMPT_CACHE", "0") == "1"

# The usage fields copied off every Anthropic response (streaming or not).
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
_usage_lock = threading.Lock()

# Anthropic models that reject `temperature` (400): Fable 5 / Mythos 5 /
# Opus 5 / Sonnet 5 and Opus 4.7 / 4.8. Haiku 4.5, Sonnet 4.6, Opus 4.6 and
# older still accept it.
_ANTHROPIC_NO_SAMPLING_RE = re.compile(
    r"^claude-(fable-5|mythos-5|opus-5|sonnet-5|opus-4-[78])(-|$)"
)


def anthropic_system_blocks(system: str, cache: bool) -> str | list[dict]:
    """The ``system`` argument for the Anthropic SDK: a plain string when
    caching is off (byte-identical to the pre-caching request), else one text
    block with an ephemeral ``cache_control`` marker on it."""
    if not cache:
        return system
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

# Single source of truth for the app-wide default model (final review M6):
# both server/main.py's MINDSHIFT_MODEL env-var default and
# server/watch/services.py's build_llm() fallback import this constant
# instead of each hardcoding their own copy of the literal.
DEFAULT_MODEL = "claude-3-haiku-20240307"


class LLMClient:
    """Unified interface for Anthropic, OpenAI, Google, and Mistral models."""

    # Class-level defaults so a client built via ``__new__`` (tests) behaves
    # like a constructed one.
    cache_system_prompt: bool = PROMPT_CACHE_ENABLED
    last_usage: dict[str, int] | None = None

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        cache_system_prompt: bool | None = None,
    ):
        self.model = model
        self._provider = self._detect_provider(model)
        self._api_key = api_key
        if cache_system_prompt is not None:
            self.cache_system_prompt = cache_system_prompt
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Usage accounting (Anthropic) — how we verify prompt-cache hits
    # ------------------------------------------------------------------

    @property
    def usage_totals(self) -> dict[str, int]:
        """Running per-client sums of the Anthropic ``usage`` fields plus
        ``calls`` — ``cache_read_input_tokens`` > 0 across repeated calls is
        the proof that prompt caching engaged. Lazily created so clients
        built without ``__init__`` still work; guarded by a lock because
        calls run on worker threads."""
        totals = self.__dict__.get("_usage_totals")
        if totals is None:
            totals = {"calls": 0, **{k: 0 for k in _USAGE_FIELDS}}
            self.__dict__["_usage_totals"] = totals
        return totals

    def _record_usage(self, usage) -> None:
        if usage is None:
            return
        snap: dict[str, int] = {}
        for key in _USAGE_FIELDS:
            value = getattr(usage, key, None)
            snap[key] = value if isinstance(value, int) else 0
        with _usage_lock:
            totals = self.usage_totals
            totals["calls"] += 1
            for key, value in snap.items():
                totals[key] += value
            self.last_usage = snap
        logger.debug("LLM usage model=%s %s", self.model, snap)

    # ------------------------------------------------------------------
    # Provider detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_provider(model: str) -> str:
        if model.startswith("claude-"):
            return "anthropic"
        if re.match(r"^(gpt-|o1|o3|o4)", model):
            return "openai"
        if model.startswith("gemini-"):
            return "google"
        if model.startswith("mistral-"):
            return "mistral"
        raise ValueError(f"Unknown model provider for: {model}")

    # ------------------------------------------------------------------
    # Static helpers (PRD spec)
    # ------------------------------------------------------------------

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        return model.startswith(("o1", "o3", "o4")) or \
               ("gpt-5" in model and "reasoning" in model)

    @staticmethod
    def uses_responses_api(model: str) -> bool:
        return model.startswith(("gpt-5", "o1", "o3", "o4"))

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _build_client(self):
        if self._provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(
                api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )

        if self._provider == "openai":
            import openai
            return openai.OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )

        if self._provider == "google":
            import google.generativeai as genai
            genai.configure(
                api_key=self._api_key or os.environ.get("GOOGLE_API_KEY"),
            )
            return genai

        if self._provider == "mistral":
            from mistralai import Mistral
            # Speakeasy-generated SDK takes a client-wide timeout in ms.
            return Mistral(
                api_key=self._api_key or os.environ.get("MISTRAL_API_KEY"),
                timeout_ms=REQUEST_TIMEOUT_SECONDS * 1000,
            )

        raise ValueError(f"No client builder for provider: {self._provider}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying provider client's connection pool.

        Anthropic/OpenAI SDK clients expose ``close()`` (httpx pool); the
        Google ``genai`` module does not — quietly skip where unsupported.
        """
        close_fn = getattr(self._client, "close", None)
        if not callable(close_fn):
            return
        try:
            close_fn()
        except Exception:  # noqa: BLE001 — shutdown must never raise
            logger.warning(
                "Error closing %s LLM client", self._provider, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Temperature rules (PRD Section 12)
    # ------------------------------------------------------------------

    def _resolve_temperature(self, temperature: float) -> float | None:
        """Apply per-model temperature rules. Returns None to omit."""
        # o1/o3/o4 reject temperature entirely
        if self.model.startswith(("o1", "o3", "o4")):
            return None
        # Claude 5-family (and Opus 4.7/4.8) removed sampling parameters —
        # the API answers 400 "`temperature` is deprecated for this model".
        if _ANTHROPIC_NO_SAMPLING_RE.match(self.model):
            return None
        # gpt-5 with reasoning must be 1.0
        if "gpt-5" in self.model and "reasoning" in self.model:
            return 1.0
        return temperature

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_schema: dict | None = None,
    ) -> str:
        """Send a prompt and return plain text. Provider is auto-detected.

        ``response_schema`` (optional JSON schema) asks the provider to
        constrain the output to that shape — Anthropic structured outputs
        (``output_config.format``). Other providers ignore it: the prompt's
        own "return ONLY JSON" instruction is all they get, exactly as before.
        """
        temp = self._resolve_temperature(temperature)

        if self._provider == "anthropic":
            return self._complete_anthropic(
                system, user, temp, max_tokens, response_schema,
            )
        if self._provider == "openai":
            if self.uses_responses_api(self.model):
                return self._complete_openai_responses(system, user, temp, max_tokens)
            return self._complete_openai_chat(system, user, temp, max_tokens)
        if self._provider == "google":
            return self._complete_google(system, user, temp, max_tokens)
        if self._provider == "mistral":
            return self._complete_mistral(system, user, temp, max_tokens)

        raise ValueError(f"No completion handler for provider: {self._provider}")

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_schema: dict | None = None,
    ) -> Iterator[str]:
        """Yield the completion as text deltas as they arrive (blocking iterator).

        Anthropic: genuine token streaming through the SDK's ``messages.stream``
        helper, so a caller can act on the first complete sentence of a
        suggestion while the rest is still being generated (the realtime
        pipeline sends a ``partial`` SuggestionEvent from it). Every other
        provider: a NON-streaming fallback that yields the whole
        :meth:`complete` result as one chunk — honest (nothing is fabricated,
        just delivered all at once) and it lets callers always iterate
        without a per-provider branch. Concatenating every yielded chunk is
        always exactly what ``complete()`` would have returned.

        Blocking, like ``complete()``: callers on an event loop run the
        iteration in a thread (``asyncio.to_thread``) the same way.
        """
        temp = self._resolve_temperature(temperature)
        if self._provider == "anthropic":
            yield from self._stream_anthropic(
                system, user, temp, max_tokens, response_schema,
            )
            return
        yield self.complete(system, user, temperature, max_tokens, response_schema)

    # --- Anthropic Messages API ---

    def _anthropic_kwargs(
        self,
        system: str,
        user: str,
        temp: float | None,
        max_tokens: int,
        response_schema: dict | None,
    ) -> dict:
        """The shared request shape for ``messages.create`` / ``.stream``.

        System prompt first (cacheable prefix), the per-turn user content
        after it, so the cache marker sits exactly at the stable/volatile
        boundary (see :func:`anthropic_system_blocks`).
        """
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=anthropic_system_blocks(system, self.cache_system_prompt),
            messages=[{"role": "user", "content": user}],
        )
        if temp is not None:
            kwargs["temperature"] = temp
        if response_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": response_schema},
            }
        return kwargs

    def _stream_anthropic(
        self,
        system: str,
        user: str,
        temp: float | None,
        max_tokens: int,
        response_schema: dict | None = None,
    ) -> Iterator[str]:
        kwargs = self._anthropic_kwargs(system, user, temp, max_tokens, response_schema)
        # The context manager closes the SSE connection even if the consumer
        # stops iterating early (e.g. the pipeline's worker is cancelled on
        # a mid-generation stop) — a bare `stream=True` iterator would not.
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
            # Only reached when the consumer drained the stream: the final
            # message (and its usage — cache hits live there) is complete.
            try:
                self._record_usage(stream.get_final_message().usage)
            except Exception:  # noqa: BLE001 — accounting must never break a reply
                logger.debug("Could not read streamed usage", exc_info=True)

    def _complete_anthropic(
        self,
        system: str,
        user: str,
        temp: float | None,
        max_tokens: int,
        response_schema: dict | None = None,
    ) -> str:
        kwargs = self._anthropic_kwargs(system, user, temp, max_tokens, response_schema)
        message = self._client.messages.create(**kwargs)
        self._record_usage(getattr(message, "usage", None))
        return message.content[0].text

    # --- OpenAI Chat Completions ---

    def _complete_openai_chat(
        self, system: str, user: str, temp: float | None, max_tokens: int,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        if temp is not None:
            kwargs["temperature"] = temp
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # --- OpenAI Responses API (gpt-5+, o1, o3, o4) ---

    def _complete_openai_responses(
        self, system: str, user: str, temp: float | None, max_tokens: int,
    ) -> str:
        prompt = f"{system}\n\n{user}"
        kwargs: dict = dict(
            model=self.model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        if temp is not None:
            kwargs["temperature"] = temp
        response = self._client.responses.create(**kwargs)
        return response.output_text

    # --- Google Generative AI ---

    def _complete_google(
        self, system: str, user: str, temp: float | None, max_tokens: int,
    ) -> str:
        generation_config: dict = {"max_output_tokens": max_tokens}
        if temp is not None:
            generation_config["temperature"] = temp
        model = self._client.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=generation_config,
        )
        response = model.generate_content(user)
        return response.text

    # --- Mistral Chat Completions ---

    def _complete_mistral(
        self, system: str, user: str, temp: float | None, max_tokens: int,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        if temp is not None:
            kwargs["temperature"] = temp
        response = self._client.chat.complete(**kwargs)
        return response.choices[0].message.content
