"""
Unit tests for LLMClient error handling and resilience (I-13).
"""

from unittest.mock import MagicMock, patch
import pytest
import httpx
import openai

from config.settings import Settings
from modules.llm_client import LLMClient, LLMServiceError


def _make_dummy_response(status_code: int = 400) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


class TestLLMClientErrorHandling:
    @pytest.fixture
    def client(self):
        settings = Settings(
            llm_provider="groq",
            groq_api_key="test_groq_key",
        )
        return LLMClient(settings)

    def test_is_configured_behavior(self):
        s_unconfigured = Settings(llm_provider="groq", groq_api_key="placeholder_groq_key")
        assert not LLMClient(s_unconfigured).is_configured()

        s_configured = Settings(llm_provider="groq", groq_api_key="gsk_actual_key_123")
        assert LLMClient(s_configured).is_configured()

    def test_auth_error_maps_to_503(self, client):
        dummy_resp = _make_dummy_response(401)
        client.client.chat.completions.create = MagicMock(
            side_effect=openai.AuthenticationError("Invalid API key", response=dummy_resp, body=None)
        )
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("hello", "simple")
        assert exc_info.value.status_code == 503
        assert "authentication failed" in exc_info.value.message.lower()

    def test_rate_limit_error_maps_to_429(self, client):
        dummy_resp = _make_dummy_response(429)
        client.client.chat.completions.create = MagicMock(
            side_effect=openai.RateLimitError("Rate limit exceeded", response=dummy_resp, body=None)
        )
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("hello", "simple")
        assert exc_info.value.status_code == 429
        assert "rate limit" in exc_info.value.message.lower()

    def test_timeout_error_maps_to_503(self, client):
        client.client.chat.completions.create = MagicMock(
            side_effect=openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
        )
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("hello", "simple")
        assert exc_info.value.status_code == 503
        assert "timed out" in exc_info.value.message.lower()

    def test_connection_error_maps_to_503(self, client):
        client.client.chat.completions.create = MagicMock(
            side_effect=openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
        )
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("hello", "simple")
        assert exc_info.value.status_code == 503
        assert "connect" in exc_info.value.message.lower()

    def test_content_filter_maps_to_503(self, client):
        mock_choice = MagicMock()
        mock_choice.finish_reason = "content_filter"
        mock_choice.message.content = ""
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        client.client.chat.completions.create = MagicMock(return_value=mock_resp)
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("unsafe query", "simple")
        assert exc_info.value.status_code == 503
        assert "safety/content filter" in exc_info.value.message.lower()

    def test_empty_content_maps_to_503(self, client):
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "   "
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        client.client.chat.completions.create = MagicMock(return_value=mock_resp)
        with pytest.raises(LLMServiceError) as exc_info:
            client.generate("query", "simple")
        assert exc_info.value.status_code == 503
        assert "empty content" in exc_info.value.message.lower()

    def test_successful_generation(self, client):
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Valid answer text"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        client.client.chat.completions.create = MagicMock(return_value=mock_resp)
        ans = client.generate("query", "simple")
        assert ans == "Valid answer text"
