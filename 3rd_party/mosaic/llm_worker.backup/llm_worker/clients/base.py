"""Base LLM Client Interface for MOSAIC LLM Worker.

This module defines the abstract base class and common utilities
for LLM API clients.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import MosaicLLMWorkerConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: The generated text content.
        model: Model identifier used.
        usage: Token usage statistics.
        finish_reason: Reason for stopping generation.
        latency_ms: Request latency in milliseconds.
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients.

    All LLM clients should implement this interface to ensure
    interchangeability in the worker runtime.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        timeout: float = 30.0,
    ):
        """Initialize LLM client.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3.5-sonnet").
            api_key: API key for authentication.
            base_url: Optional custom API base URL.
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt/message.
            system: Optional system message.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse with generated content.

        Raises:
            Exception: On API errors.
        """
        pass

    def generate_with_retry(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate with automatic retry on failure.

        Args:
            prompt: User prompt/message.
            system: Optional system message.
            max_retries: Maximum retry attempts.
            retry_delay: Base delay between retries (doubles each retry).
            **kwargs: Additional arguments passed to generate().

        Returns:
            LLMResponse with generated content.

        Raises:
            Exception: If all retries fail.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM request failed after {max_retries} attempts: {e}")

        raise last_error if last_error else RuntimeError("Unknown error in generate_with_retry")

    @classmethod
    def from_config(cls, config: "MosaicLLMWorkerConfig") -> "BaseLLMClient":
        """Create client from worker config.

        Args:
            config: Worker configuration.

        Returns:
            Configured LLM client.
        """
        return cls(
            model_id=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            base_url=config.api_base_url,
        )


def create_client(
    client_name: str,
    model_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    timeout: float = 30.0,
) -> BaseLLMClient:
    """Factory function to create LLM client by name.

    Args:
        client_name: Client type ("openrouter", "openai", "anthropic", "vllm").
        model_id: Model identifier.
        api_key: API key (uses environment variable if not provided).
        base_url: Optional custom API base URL.
        temperature: Generation temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout.

    Returns:
        Configured LLM client.

    Raises:
        ValueError: If client_name is not supported.
        ImportError: If required package is not installed.
    """
    client_name = client_name.lower()

    if client_name == "openrouter":
        from .openrouter import OpenRouterClient
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        return OpenRouterClient(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif client_name == "openai":
        try:
            from .openai_client import OpenAIClient
        except ImportError:
            raise ImportError(
                "OpenAI client requires 'openai' package. "
                "Install with: pip install openai"
            )
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIClient(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif client_name == "anthropic":
        try:
            from .anthropic_client import AnthropicClient
        except ImportError:
            raise ImportError(
                "Anthropic client requires 'anthropic' package. "
                "Install with: pip install anthropic"
            )
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        return AnthropicClient(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif client_name == "vllm":
        # vLLM uses OpenAI-compatible API
        try:
            from .openai_client import OpenAIClient
        except ImportError:
            raise ImportError(
                "vLLM client requires 'openai' package. "
                "Install with: pip install openai"
            )
        if base_url is None:
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return OpenAIClient(
            model_id=model_id,
            api_key=api_key or "EMPTY",  # vLLM doesn't require real key
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif client_name in ("google", "gemini"):
        try:
            from .google_client import GoogleClient
        except ImportError:
            raise ImportError(
                "Google client requires 'google-genai' package. "
                "Install with: pip install google-genai"
            )
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        return GoogleClient(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    else:
        raise ValueError(
            f"Unknown client_name: {client_name}. "
            f"Supported: openrouter, openai, anthropic, vllm, google, gemini"
        )


__all__ = ["BaseLLMClient", "LLMResponse", "create_client"]
