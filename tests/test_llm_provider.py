"""Tests for the LLM provider layer (mocked LiteLLM)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.config import ModelProviderConfig, ModelsConfig
from agent.llm.provider import LLMProvider


def _make_mock_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Create a mock LiteLLM response object."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


class TestLLMProvider:
    """Test LLMProvider functionality."""

    @pytest.fixture
    def config(self) -> ModelsConfig:
        return ModelsConfig(
            default="gpt-4o-mini",
            fallback="gpt-3.5-turbo",
            providers={
                "openai": ModelProviderConfig(api_key="sk-test-key-123"),
            },
        )

    @pytest.fixture
    def provider(self, config: ModelsConfig) -> LLMProvider:
        return LLMProvider(config)

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_completion_default_model(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()

        messages = [{"role": "user", "content": "Hi"}]
        response = await provider.completion(messages=messages)

        assert response.content == "Hello!"
        assert response.model == "gpt-4o-mini"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "stop"

        mock_acompletion.assert_called_once()

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_completion_custom_model(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()

        messages = [{"role": "user", "content": "Hi"}]
        response = await provider.completion(messages=messages, model="custom-model")

        assert response.model == "custom-model"
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs[1]["model"] == "custom-model"

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_failover_to_fallback(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        # First call fails, second (fallback) succeeds
        mock_acompletion.side_effect = [
            Exception("Primary model error"),
            _make_mock_response(content="Fallback response"),
        ]

        messages = [{"role": "user", "content": "Hi"}]
        response = await provider.completion(messages=messages)

        assert response.content == "Fallback response"
        assert response.model == "gpt-3.5-turbo"
        assert mock_acompletion.call_count == 2

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_no_fallback_raises(self, mock_acompletion: AsyncMock) -> None:
        config = ModelsConfig(default="gpt-4o-mini", fallback=None)
        provider = LLMProvider(config)

        mock_acompletion.side_effect = Exception("Model error")

        with pytest.raises(Exception, match="Model error"):
            await provider.completion(messages=[{"role": "user", "content": "Hi"}])

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_both_models_fail(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.side_effect = RuntimeError("All models down")

        with pytest.raises(RuntimeError):
            await provider.completion(messages=[{"role": "user", "content": "Hi"}])

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_token_usage_parsed(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.return_value = _make_mock_response(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

        response = await provider.completion(messages=[{"role": "user", "content": "Hi"}])

        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 50
        assert response.usage.total_tokens == 150

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_test_connection_success(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.return_value = _make_mock_response()

        result = await provider.test_connection("gpt-4o-mini")
        assert result is True

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_test_connection_failure(
        self, mock_acompletion: AsyncMock, provider: LLMProvider
    ) -> None:
        mock_acompletion.side_effect = Exception("Connection failed")

        result = await provider.test_connection("gpt-4o-mini")
        assert result is False

    def test_api_keys_set_in_environment(self) -> None:
        config = ModelsConfig(
            providers={
                "openai": ModelProviderConfig(api_key="sk-test-openai"),
                "anthropic": ModelProviderConfig(api_key="sk-ant-test"),
            }
        )
        LLMProvider(config)

        assert os.environ.get("OPENAI_API_KEY") == "sk-test-openai"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test"
