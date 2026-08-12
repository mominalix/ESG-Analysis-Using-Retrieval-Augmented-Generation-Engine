"""Lazy, provider-neutral language model service."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langsmith import traceable

from ..core.config import reveal, settings
from ..core.exceptions import ConfigurationError, LLMProviderError
from ..core.logging import LoggingMixin


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    tokens_used: int | None = None
    latency_ms: int | None = None


class BaseLLMProvider(ABC, LoggingMixin):
    @abstractmethod
    async def generate(self, messages: list[BaseMessage], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError


def _message_text(message: Any) -> str:
    """Normalize text returned by Chat Completions or the Responses API."""
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    return str(content or "")


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        from langchain_openai import ChatOpenAI

        client_options: dict[str, Any] = {
            "api_key": api_key,
            "model": model,
            "use_responses_api": settings.openai_use_responses_api,
        }
        if settings.max_output_tokens:
            client_options["max_tokens"] = settings.max_output_tokens
        if settings.openai_reasoning_effort != "none":
            client_options["reasoning"] = {
                "effort": settings.openai_reasoning_effort,
                "summary": "auto",
            }
        else:
            client_options["temperature"] = settings.temperature

        self.model = model
        self.client = ChatOpenAI(**client_options)

    @traceable(name="openai_generate")
    async def generate(self, messages: list[BaseMessage], **kwargs: Any) -> LLMResponse:
        try:
            response = await self.client.ainvoke(messages, **kwargs)
            usage = response.response_metadata.get("token_usage", {})
            return LLMResponse(
                content=_message_text(response),
                model=self.model,
                provider="openai",
                tokens_used=usage.get("total_tokens"),
            )
        except Exception as exc:
            self.logger.error("OpenAI generation failed", error=str(exc))
            raise LLMProviderError("OpenAI generation failed") from exc

    async def stream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                content = _message_text(chunk)
                if content:
                    yield content
        except Exception as exc:
            self.logger.error("OpenAI streaming failed", error=str(exc))
            raise LLMProviderError("OpenAI streaming failed") from exc


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        from langchain_anthropic import ChatAnthropic

        self.model = model
        self.client = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens or 4096,
        )

    @traceable(name="anthropic_generate")
    async def generate(self, messages: list[BaseMessage], **kwargs: Any) -> LLMResponse:
        try:
            response = await self.client.ainvoke(messages, **kwargs)
            usage = response.response_metadata.get("usage", {})
            return LLMResponse(
                content=_message_text(response),
                model=self.model,
                provider="anthropic",
                tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )
        except Exception as exc:
            self.logger.error("Anthropic generation failed", error=str(exc))
            raise LLMProviderError("Anthropic generation failed") from exc

    async def stream(self, messages: list[BaseMessage], **kwargs: Any) -> AsyncIterator[str]:
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                content = _message_text(chunk)
                if content:
                    yield content
        except Exception as exc:
            self.logger.error("Anthropic streaming failed", error=str(exc))
            raise LLMProviderError("Anthropic streaming failed") from exc


ProviderFactory = Callable[[], BaseLLMProvider]


class LLMService(LoggingMixin):
    """Provider registry that avoids network/client work during module import."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._factories: dict[str, ProviderFactory] = {}
        self._register_configured_providers()

    def _register_configured_providers(self) -> None:
        openai_key = reveal(settings.openai_api_key)
        if openai_key:
            self.register_provider(
                "openai", lambda: OpenAIProvider(openai_key, settings.openai_model)
            )

        anthropic_key = reveal(settings.anthropic_api_key)
        if anthropic_key:
            self.register_provider(
                "anthropic", lambda: AnthropicProvider(anthropic_key, settings.anthropic_model)
            )

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        self._factories[name.lower()] = factory

    def get_provider(self, provider_name: str | None = None) -> BaseLLMProvider:
        name = (provider_name or settings.default_llm_provider).lower()
        if name not in self._factories:
            configured = ", ".join(self.list_providers()) or "none"
            raise ConfigurationError(
                f"LLM provider '{name}' is not configured (available: {configured})",
                error_code="LLM_PROVIDER_NOT_CONFIGURED",
            )
        if name not in self._providers:
            self._providers[name] = self._factories[name]()
        return self._providers[name]

    @traceable(name="llm_generate")
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return await self.get_provider(provider).generate(messages, **kwargs)

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        async for chunk in self.get_provider(provider).stream(messages, **kwargs):
            yield chunk

    def list_providers(self) -> list[str]:
        return sorted(self._factories)


llm_service = LLMService()
