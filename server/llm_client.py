"""Multi-vendor LLM abstraction layer.

Auto-detects provider from model name and routes to the correct SDK/API.
Handles temperature rules per PRD Section 12.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from typing import Callable, ContextManager, Iterator

logger = logging.getLogger(__name__)

# Production guardrails: without an explicit timeout the Anthropic/OpenAI SDKs
# default to a 600s request timeout with 2 retries — a single hung call could
# occupy a worker thread for 30 minutes. Fail fast instead.
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1

# Hedged streaming (perf/llm-hedging). Measured 2026-08-24 against the real
# API (scripts/bench_suggestions.py): the cloud-suggestion time-to-first-token
# is ~0.5–0.9 s at p50 but ~1 in 30 calls stalls for 5–11 s before the first
# byte, in every prompt/request variant — a tail that is the provider's, not
# ours. In a live conversation a 7 s stall is worse than a slightly higher
# median, so a streaming call that has produced NO token after
# ``MINDSHIFT_LLM_HEDGE_AFTER_MS`` fires a second identical request and keeps
# whichever streams first (the loser is cancelled: its SSE connection is
# closed, so it stops generating). Threshold well above p95 of the healthy
# distribution so the hedge fires only on the stall tail (≈3 % of calls at
# 1500 ms; the bench reports the actual rate). Cost of a hedge: the loser's
# prompt tokens (identical to the winner's) plus whatever output it managed
# before the close — ~+3 % input tokens overall at that rate.
#
# ``MINDSHIFT_LLM_FIRST_TOKEN_DEADLINE_MS`` is the hard cap: no attempt has
# produced a first token by then → the call is abandoned with
# :class:`LLMFirstTokenTimeout` (the realtime pipeline reports it as a
# ``suggestion_error`` reason ``slow_llm`` and moves on to the next turn)
# instead of holding the worker for the SDK's 30 s read timeout. Either knob
# ``0`` disables that half.
LLM_HEDGE_AFTER_MS = int(os.getenv("MINDSHIFT_LLM_HEDGE_AFTER_MS", "1500"))
LLM_FIRST_TOKEN_DEADLINE_MS = int(os.getenv("MINDSHIFT_LLM_FIRST_TOKEN_DEADLINE_MS", "6000"))
# Bound on how long a winner may go between deltas before the consumer gives
# up on its thread — well past the SDK read timeout (plus its one retry) so it
# can only fire if the producer thread died without reporting, never first.
_STREAM_STALL_LIMIT_S = REQUEST_TIMEOUT_SECONDS * (MAX_RETRIES + 1) + 5


class LLMFirstTokenTimeout(TimeoutError):
    """No attempt of a hedged streaming call produced its first token within
    ``deadline_ms`` — the call was abandoned (every attempt cancelled)."""

    def __init__(self, deadline_ms: int, attempts: int) -> None:
        super().__init__(
            f"no first token from {attempts} attempt(s) within {deadline_ms} ms"
        )
        self.deadline_ms = deadline_ms
        self.attempts = attempts


# Queue message kinds from an attempt's producer thread to the consumer.
_DELTA, _DONE, _ERROR = "delta", "done", "error"


class _Attempt:
    """One in-flight streaming request of a :class:`HedgedStream`."""

    __slots__ = ("index", "cancel", "stream", "started_at")

    def __init__(self, index: int, started_at: float) -> None:
        self.index = index
        self.cancel = threading.Event()
        self.stream = None  # the SDK MessageStream once the connection is up
        self.started_at = started_at


class HedgedStream:
    """Text-delta iterator over ONE logical streaming request, hedged.

    ``open_stream()`` returns the SDK's ``messages.stream(...)`` context
    manager. Each attempt runs on its own daemon thread and forwards deltas
    to a queue; the consumer (whoever iterates this object) starts attempt 0,
    fires attempt 1 if no delta has arrived after ``hedge_after_ms``, adopts
    the first attempt to deliver a delta as the winner, cancels the rest, and
    raises :class:`LLMFirstTokenTimeout` if nothing has arrived by
    ``deadline_ms``. Once a winner exists its deltas are yielded until the
    stream ends (usage handed to ``on_usage``) or fails (the error is raised
    here, on the consumer's thread).

    Attributes readable after iteration: ``hedged`` (a second request was
    fired), ``hedge_won`` (the second request was the one used),
    ``attempts``, ``first_token_ms`` (from the first request's start to the
    winner's first delta; None when abandoned).

    Cancellation is best-effort by design. A cancelled attempt's flag is
    checked between deltas and its SSE response is closed from the consumer
    thread (the SDK's ``MessageStream.close()``, which releases the
    connection); a loser still stalled before its first byte may keep its
    thread blocked until that byte or the SDK read timeout arrives, then
    exits — it never reaches the consumer. Dedicated threads (not the
    asyncio default executor) so a lingering loser cannot starve the event
    loop's ``to_thread`` pool.
    """

    def __init__(
        self,
        open_stream: Callable[[], ContextManager],
        *,
        hedge_after_ms: int,
        deadline_ms: int,
        on_usage: Callable[[object], None] | None = None,
        on_event: Callable[[str, "HedgedStream"], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        label: str = "",
    ) -> None:
        self._open = open_stream
        self._hedge_after_s = max(0, hedge_after_ms) / 1000.0
        self._deadline_s = max(0, deadline_ms) / 1000.0
        self._on_usage = on_usage
        self._on_event = on_event
        self._clock = clock
        self._label = label
        self._q: "queue.Queue[tuple[_Attempt, str, object]]" = queue.Queue()
        self._attempts: list[_Attempt] = []
        self._winner: _Attempt | None = None
        self._closed = False
        self.hedged = False
        self.hedge_won = False
        self.attempts = 0
        self.first_token_ms: float | None = None
        self.usage: object = None  # the winner's final-message usage, once drained

    # -- producer side --------------------------------------------------------

    def _start_attempt(self) -> _Attempt:
        attempt = _Attempt(len(self._attempts), self._clock())
        self._attempts.append(attempt)
        self.attempts = len(self._attempts)
        threading.Thread(
            target=self._run, args=(attempt,), daemon=True,
            name=f"llm-stream-{attempt.index}",
        ).start()
        return attempt

    def _run(self, attempt: _Attempt) -> None:
        try:
            # The context manager closes the SSE connection on every exit —
            # drained, cancelled, or failed — so a loser never keeps
            # generating (and billing) once it is abandoned.
            with self._open() as stream:
                attempt.stream = stream
                if attempt.cancel.is_set():
                    return
                for text in stream.text_stream:
                    if attempt.cancel.is_set():
                        return
                    self._q.put((attempt, _DELTA, text))
                if attempt.cancel.is_set():
                    return
                usage = None
                try:
                    usage = stream.get_final_message().usage
                except Exception:  # noqa: BLE001 — accounting must never break a reply
                    logger.debug("Could not read streamed usage", exc_info=True)
                self._q.put((attempt, _DONE, usage))
        except BaseException as exc:  # noqa: BLE001 — forwarded to the consumer
            if not attempt.cancel.is_set():
                self._q.put((attempt, _ERROR, exc))

    def _cancel(self, attempt: _Attempt) -> None:
        attempt.cancel.set()
        stream = attempt.stream
        if stream is not None:
            # Best-effort unblock of a thread waiting on the next SSE byte.
            # The response object is the SDK's; closing it from another
            # thread is tolerated (the reader sees a closed stream and its
            # own exception path swallows it as cancelled).
            with_close = getattr(stream, "close", None)
            if callable(with_close):
                try:
                    with_close()
                except Exception:  # noqa: BLE001 — cancellation is best-effort
                    logger.debug("Closing a cancelled LLM stream failed", exc_info=True)

    def close(self) -> None:
        """Cancel every attempt (winner included) — the consumer is done."""
        self._closed = True
        for attempt in self._attempts:
            if not attempt.cancel.is_set():
                self._cancel(attempt)

    def _emit(self, event: str) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event, self)
            except Exception:  # noqa: BLE001 — accounting must never break a reply
                logger.debug("HedgedStream on_event failed", exc_info=True)

    # -- consumer side --------------------------------------------------------

    def __iter__(self) -> Iterator[str]:
        try:
            yield from self._iterate()
        finally:
            self.close()

    def _iterate(self) -> Iterator[str]:
        t0 = self._clock()
        self._start_attempt()
        self._emit("start")
        live = 1
        last_error: BaseException | None = None
        hedge_at = t0 + self._hedge_after_s if self._hedge_after_s > 0 else None
        deadline_at = t0 + self._deadline_s if self._deadline_s > 0 else None

        # Phase 1: wait for SOMEONE's first delta, hedging and bounding it.
        while self._winner is None:
            now = self._clock()
            if hedge_at is not None and now >= hedge_at and not self.hedged:
                self.hedged = True
                live += 1
                self._start_attempt()
                logger.info(
                    "LLM stream %s: no first token after %.0f ms — hedging with a "
                    "second request", self._label, (now - t0) * 1000.0,
                )
                self._emit("hedged")
            if deadline_at is not None and now >= deadline_at:
                self.close()
                self._emit("slow_llm")
                logger.warning(
                    "LLM stream %s: no first token from %d attempt(s) within "
                    "%.0f ms — abandoning the call", self._label, self.attempts,
                    self._deadline_s * 1000.0,
                )
                raise LLMFirstTokenTimeout(int(self._deadline_s * 1000), self.attempts)
            pending = [deadline_at] + ([hedge_at] if not self.hedged else [])
            pending = [x for x in pending if x is not None]
            timeout = max(0.0, min(pending) - now) if pending else None
            try:
                attempt, kind, payload = self._q.get(timeout=timeout)
            except queue.Empty:
                continue
            if attempt.cancel.is_set():
                continue  # a straggler from an abandoned attempt
            if kind == _ERROR:
                live -= 1
                last_error = payload  # type: ignore[assignment]
                if live <= 0:
                    self.close()
                    raise last_error  # type: ignore[misc]
                continue
            # First delta (or an empty completion) — this attempt wins.
            self._winner = attempt
            self.first_token_ms = (self._clock() - t0) * 1000.0
            self.hedge_won = self.hedged and attempt.index > 0
            for other in self._attempts:
                if other is not attempt:
                    self._cancel(other)
            if self.hedged:
                logger.info(
                    "LLM stream %s: hedge %s (first token at %.0f ms from attempt %d)",
                    self._label, "won" if self.hedge_won else "lost",
                    self.first_token_ms, attempt.index,
                )
            self._emit("first_token")
            if kind == _DONE:
                self._finish(payload)
                return
            yield payload  # type: ignore[misc]

        # Phase 2: the winner's stream, to the end.
        while True:
            try:
                attempt, kind, payload = self._q.get(timeout=_STREAM_STALL_LIMIT_S)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(
                    f"LLM stream {self._label}: winner produced nothing for "
                    f"{_STREAM_STALL_LIMIT_S:.0f} s"
                ) from exc
            if attempt is not self._winner:
                continue
            if kind == _DELTA:
                yield payload  # type: ignore[misc]
            elif kind == _DONE:
                self._finish(payload)
                return
            else:
                raise payload  # type: ignore[misc]

    def _finish(self, usage: object) -> None:
        self.usage = usage
        if self._on_usage is not None:
            self._on_usage(usage)
        self._emit("done")

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
        # Cost guardrails: attribute this call's tokens to the uid + call site
        # bound by usage_meter.attribute()/usage_scope(). Imported lazily and
        # failure-swallowing so a client built outside the server process (the
        # bench scripts, the watch worker) keeps working untouched.
        try:
            import usage_meter

            usage_meter.note_llm_usage(snap)
        except Exception:  # noqa: BLE001 — accounting never breaks a call
            logger.debug("usage_meter unavailable", exc_info=True)

    @property
    def hedge_totals(self) -> dict[str, int]:
        """Running per-client hedged-streaming counters: ``streams`` (hedge-
        capable streaming calls started), ``hedged`` (a second request was
        fired), ``hedge_won`` (the second request was the one used),
        ``slow_llm`` (abandoned at the first-token deadline) and
        ``hedge_extra_input_tokens`` — the cost side: every hedged call
        bills its loser's prompt too, which is byte-identical to the
        winner's, so the winner's ``input_tokens`` is added here per hedged
        call (a lower bound; the loser's few output tokens before its
        close are not counted)."""
        totals = self.__dict__.get("_hedge_totals")
        if totals is None:
            totals = {
                "streams": 0, "hedged": 0, "hedge_won": 0, "slow_llm": 0,
                "hedge_extra_input_tokens": 0,
            }
            self.__dict__["_hedge_totals"] = totals
        return totals

    def _on_hedge_event(self, event: str, stream: HedgedStream) -> None:
        with _usage_lock:
            totals = self.hedge_totals
            if event == "start":
                totals["streams"] += 1
            elif event == "hedged":
                totals["hedged"] += 1
            elif event == "slow_llm":
                totals["slow_llm"] += 1
            elif event == "first_token" and stream.hedge_won:
                totals["hedge_won"] += 1
            elif event == "done" and stream.hedged:
                extra = getattr(stream.usage, "input_tokens", 0)
                extra = extra if isinstance(extra, int) else 0
                totals["hedge_extra_input_tokens"] += extra
                # Same surcharge, attributed per uid/site (cost guardrails).
                try:
                    import usage_meter

                    usage_meter.note_hedge_extra(extra)
                except Exception:  # noqa: BLE001
                    logger.debug("usage_meter unavailable", exc_info=True)

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

        Anthropic calls are HEDGED (see :class:`HedgedStream` and the
        ``MINDSHIFT_LLM_HEDGE_AFTER_MS`` / ``MINDSHIFT_LLM_FIRST_TOKEN_DEADLINE_MS``
        knobs): the returned iterator is the :class:`HedgedStream` itself,
        so a caller can read ``hedged`` / ``hedge_won`` / ``first_token_ms``
        off it after draining, and iteration raises
        :class:`LLMFirstTokenTimeout` when no attempt produced a first token
        by the deadline. Nothing is sent until iteration starts.
        """
        temp = self._resolve_temperature(temperature)
        if self._provider == "anthropic":
            return self._stream_anthropic(
                system, user, temp, max_tokens, response_schema,
            )

        def _one_chunk() -> Iterator[str]:
            yield self.complete(system, user, temperature, max_tokens, response_schema)

        return _one_chunk()

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
    ) -> HedgedStream:
        kwargs = self._anthropic_kwargs(system, user, temp, max_tokens, response_schema)
        # Each attempt runs `with messages.stream(**kwargs)` on its own thread
        # (HedgedStream._run): the context manager closes the SSE connection
        # on every exit — drained, cancelled as the hedge's loser, or when the
        # consumer stops iterating early (a mid-generation stop) — so an
        # abandoned request never keeps generating. Usage (cache hits live in
        # the final message) is recorded only for the winner, once drained.
        return HedgedStream(
            lambda: self._client.messages.stream(**kwargs),
            hedge_after_ms=LLM_HEDGE_AFTER_MS,
            deadline_ms=LLM_FIRST_TOKEN_DEADLINE_MS,
            on_usage=self._record_usage,
            on_event=self._on_hedge_event,
            label=self.model,
        )

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
