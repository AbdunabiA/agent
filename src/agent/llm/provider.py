"""LLM provider layer — unified interface via LiteLLM.

Wraps LiteLLM to provide async completion with automatic failover
from the primary model to a configured fallback.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import litellm
import structlog

from agent.config import ModelsConfig
from agent.core.session import TokenUsage, ToolCall

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM completion call."""

    content: str
    model: str
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    finish_reason: str = "stop"
    raw_response: Any = None


@dataclass
class LLMStreamChunk:
    """A single chunk from a streaming LLM response."""

    content: str = ""
    done: bool = False
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    model: str = ""
    finish_reason: str | None = None


class LLMProvider:
    """Unified interface for all LLM providers via LiteLLM."""

    def __init__(self, config: ModelsConfig) -> None:
        self.config = config
        self._setup_providers()

    def _setup_providers(self) -> None:
        """Set API keys and base URLs from config into environment for LiteLLM."""
        env_key_map: dict[str, str] = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }

        for provider_name, provider_config in self.config.providers.items():
            if provider_config.api_key:
                env_var = env_key_map.get(provider_name)
                if env_var:
                    os.environ[env_var] = provider_config.api_key

            if provider_config.base_url and provider_name == "ollama":
                os.environ["OLLAMA_API_BASE"] = provider_config.base_url

        # Suppress LiteLLM debug noise
        litellm.suppress_debug_info = True

    async def completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a completion request to the configured LLM.

        Uses the default model if none specified.
        Falls back to fallback model on failure.

        Args:
            messages: Conversation messages for the LLM.
            model: Optional model override.
            tools: Tool definitions for function calling (Phase 2).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            stream: Whether to stream the response.

        Returns:
            Structured LLMResponse with content, usage, etc.

        Raises:
            Exception: If both primary and fallback models fail.
        """
        target_model = model or self.config.default

        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await litellm.acompletion(
                model=target_model,
                messages=messages,
                **kwargs,
            )
            return self._parse_response(response, target_model)
        except Exception as e:
            logger.warning(
                "primary_model_failed",
                model=target_model,
                error=str(e),
            )

            if self.config.fallback and target_model != self.config.fallback:
                logger.info("falling_back", fallback_model=self.config.fallback)
                try:
                    response = await litellm.acompletion(
                        model=self.config.fallback,
                        messages=messages,
                        **kwargs,
                    )
                    return self._parse_response(response, self.config.fallback)
                except Exception as fallback_error:
                    logger.error(
                        "fallback_model_failed",
                        model=self.config.fallback,
                        error=str(fallback_error),
                    )
                    raise fallback_error from e

            raise

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream a completion response, yielding chunks as they arrive.

        Yields LLMStreamChunk objects with text deltas. The final chunk
        has done=True and includes accumulated tool_calls and usage.

        Falls back to non-streaming on the fallback model if primary fails.

        Args:
            messages: Conversation messages for the LLM.
            model: Optional model override.
            tools: Tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Yields:
            LLMStreamChunk with content deltas and final metadata.
        """
        target_model = model or self.config.default

        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await litellm.acompletion(
                model=target_model,
                messages=messages,
                **kwargs,
            )

            accumulated_tool_calls: dict[int, dict[str, str]] = {}
            usage: TokenUsage | None = None
            finish_reason: str | None = None

            async for chunk in response:
                if not chunk.choices:
                    # Usage-only chunk (some providers send usage separately)
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = TokenUsage(
                            input_tokens=getattr(
                                chunk.usage, "prompt_tokens", 0
                            ) or 0,
                            output_tokens=getattr(
                                chunk.usage, "completion_tokens", 0
                            ) or 0,
                            total_tokens=getattr(
                                chunk.usage, "total_tokens", 0
                            ) or 0,
                        )
                    continue

                delta = chunk.choices[0].delta

                # Text content
                if delta and delta.content:
                    yield LLMStreamChunk(
                        content=delta.content,
                        model=target_model,
                    )

                # Tool call deltas
                if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            accumulated_tool_calls[idx]["id"] = tc_delta.id
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            if tc_delta.function.name:
                                accumulated_tool_calls[idx][
                                    "name"
                                ] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                accumulated_tool_calls[idx][
                                    "arguments"
                                ] += tc_delta.function.arguments

                # Finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage (usually in the last chunk)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = TokenUsage(
                        input_tokens=getattr(
                            chunk.usage, "prompt_tokens", 0
                        ) or 0,
                        output_tokens=getattr(
                            chunk.usage, "completion_tokens", 0
                        ) or 0,
                        total_tokens=getattr(
                            chunk.usage, "total_tokens", 0
                        ) or 0,
                    )

            # Parse accumulated tool calls
            tool_calls: list[ToolCall] | None = None
            if accumulated_tool_calls:
                tool_calls = []
                for tc_data in accumulated_tool_calls.values():
                    args: dict[str, object] | str = tc_data["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    tool_calls.append(
                        ToolCall(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=args if isinstance(args, dict) else {},
                        )
                    )

            # Final chunk with completion info
            yield LLMStreamChunk(
                content="",
                done=True,
                tool_calls=tool_calls,
                usage=usage or TokenUsage(0, 0, 0),
                model=target_model,
                finish_reason=finish_reason or "stop",
            )

        except Exception as e:
            logger.warning(
                "primary_model_stream_failed",
                model=target_model,
                error=str(e),
            )

            if self.config.fallback and target_model != self.config.fallback:
                logger.info(
                    "stream_falling_back",
                    fallback_model=self.config.fallback,
                )
                try:
                    # Fallback to non-streaming for simplicity
                    fallback_kwargs = {
                        k: v for k, v in kwargs.items()
                        if k not in ("stream", "stream_options")
                    }
                    fb_response = await litellm.acompletion(
                        model=self.config.fallback,
                        messages=messages,
                        **fallback_kwargs,
                    )
                    parsed = self._parse_response(
                        fb_response, self.config.fallback
                    )
                    if parsed.content:
                        yield LLMStreamChunk(
                            content=parsed.content,
                            model=self.config.fallback,
                        )
                    yield LLMStreamChunk(
                        content="",
                        done=True,
                        tool_calls=parsed.tool_calls,
                        usage=parsed.usage,
                        model=self.config.fallback,
                        finish_reason=parsed.finish_reason,
                    )
                    return
                except Exception as fallback_error:
                    logger.error(
                        "fallback_model_stream_failed",
                        error=str(fallback_error),
                    )
                    raise fallback_error from e

            raise

    def _parse_response(self, response: Any, model: str) -> LLMResponse:
        """Parse a LiteLLM response into our structured format.

        Args:
            response: Raw response from litellm.acompletion.
            model: The model that was used.

        Returns:
            Parsed LLMResponse.
        """
        if not response.choices:
            return LLMResponse(
                content="[No response from model]",
                model=model,
                usage=TokenUsage(0, 0, 0),
            )

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls if present
        tool_calls: list[ToolCall] | None = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                elif not isinstance(args, dict):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Parse usage
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return LLMResponse(
            content=message.content or "",
            model=model,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

    async def test_connection(self, model: str) -> bool:
        """Test if a model is accessible.

        Sends a tiny test message and checks for success.
        Used by the `agent doctor` command.

        Args:
            model: Model identifier to test.

        Returns:
            True if the model responds, False otherwise.
        """
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.debug("connection_test_failed", model=model, error=str(e))
            return False
