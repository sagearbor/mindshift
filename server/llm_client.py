"""Multi-vendor LLM abstraction layer.

Auto-detects provider from model name and routes to the correct SDK/API.
Handles temperature rules per PRD Section 12.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Production guardrails: without an explicit timeout the Anthropic/OpenAI SDKs
# default to a 600s request timeout with 2 retries — a single hung call could
# occupy a worker thread for 30 minutes. Fail fast instead.
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1


class LLMClient:
    """Unified interface for Anthropic, OpenAI, Google, and Mistral models."""

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._provider = self._detect_provider(model)
        self._api_key = api_key
        self._client = self._build_client()

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
    ) -> str:
        """Send a prompt and return plain text. Provider is auto-detected."""
        temp = self._resolve_temperature(temperature)

        if self._provider == "anthropic":
            return self._complete_anthropic(system, user, temp, max_tokens)
        if self._provider == "openai":
            if self.uses_responses_api(self.model):
                return self._complete_openai_responses(system, user, temp, max_tokens)
            return self._complete_openai_chat(system, user, temp, max_tokens)
        if self._provider == "google":
            return self._complete_google(system, user, temp, max_tokens)
        if self._provider == "mistral":
            return self._complete_mistral(system, user, temp, max_tokens)

        raise ValueError(f"No completion handler for provider: {self._provider}")

    # --- Anthropic Messages API ---

    def _complete_anthropic(
        self, system: str, user: str, temp: float | None, max_tokens: int,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if temp is not None:
            kwargs["temperature"] = temp
        message = self._client.messages.create(**kwargs)
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
