"""
LLM Service for handling various language model providers
"""
from typing import Optional, Dict, Any, List, AsyncIterator
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks import CallbackManager
from langsmith import traceable

from ..core.config import settings
from ..core.logging import LoggingMixin
from ..core.exceptions import LLMProviderError, ConfigurationError


@dataclass
class LLMResponse:
    """Response from LLM with metadata"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None


class BaseLLMProvider(ABC, LoggingMixin):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    async def stream(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response from messages"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        self.model = model
        self.client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    
    @traceable(name="openai_generate")
    async def generate(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response using OpenAI"""
        try:
            self.logger.info("Generating response with OpenAI", model=self.model)
            
            response = await self.client.ainvoke(messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                provider="openai",
                tokens_used=response.response_metadata.get("token_usage", {}).get("total_tokens"),
            )
        except Exception as e:
            self.logger.error("OpenAI generation failed", error=str(e))
            raise LLMProviderError(f"OpenAI generation failed: {str(e)}")
    
    async def stream(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response using OpenAI"""
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            self.logger.error("OpenAI streaming failed", error=str(e))
            raise LLMProviderError(f"OpenAI streaming failed: {str(e)}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        if not api_key:
            raise ConfigurationError("Anthropic API key is required")
        
        self.model = model
        self.client = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    
    @traceable(name="anthropic_generate")
    async def generate(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response using Anthropic"""
        try:
            self.logger.info("Generating response with Anthropic", model=self.model)
            
            response = await self.client.ainvoke(messages, **kwargs)
            
            return LLMResponse(
                content=response.content,
                model=self.model,
                provider="anthropic",
                tokens_used=response.response_metadata.get("usage", {}).get("total_tokens"),
            )
        except Exception as e:
            self.logger.error("Anthropic generation failed", error=str(e))
            raise LLMProviderError(f"Anthropic generation failed: {str(e)}")
    
    async def stream(
        self, 
        messages: List[BaseMessage], 
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response using Anthropic"""
        try:
            async for chunk in self.client.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            self.logger.error("Anthropic streaming failed", error=str(e))
            raise LLMProviderError(f"Anthropic streaming failed: {str(e)}")


class LLMService(LoggingMixin):
    """Main LLM service that manages multiple providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize available LLM providers"""
        if settings.openai_api_key:
            self.providers["openai"] = OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.default_model
            )
            self.logger.info("OpenAI provider initialized")
        
        # Temporarily disabled due to version compatibility issue
        # if settings.anthropic_api_key:
        #     self.providers["anthropic"] = AnthropicProvider(
        #         api_key=settings.anthropic_api_key
        #     )
        #     self.logger.info("Anthropic provider initialized")
        
        if not self.providers:
            raise ConfigurationError("No LLM providers configured")
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get LLM provider by name"""
        provider_name = provider_name or settings.default_llm_provider
        
        if provider_name not in self.providers:
            raise LLMProviderError(f"Provider '{provider_name}' not available")
        
        return self.providers[provider_name]
    
    @traceable(name="llm_generate")
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate response from prompt"""
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        llm_provider = self.get_provider(provider)
        return await llm_provider.generate(messages, **kwargs)
    
    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream response from prompt"""
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        llm_provider = self.get_provider(provider)
        async for chunk in llm_provider.stream(messages, **kwargs):
            yield chunk
    
    def list_providers(self) -> List[str]:
        """List available providers"""
        return list(self.providers.keys())


# Global LLM service instance
llm_service = LLMService()