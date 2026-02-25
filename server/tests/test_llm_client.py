"""Tests for the multi-vendor LLM abstraction layer."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from llm_client import LLMClient


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_claude_models(self):
        for model in ["claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-5-sonnet-20241022"]:
            client = LLMClient.__new__(LLMClient)
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
        assert call_kwargs["system"] == "Be helpful"
        assert "temperature" in call_kwargs


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
