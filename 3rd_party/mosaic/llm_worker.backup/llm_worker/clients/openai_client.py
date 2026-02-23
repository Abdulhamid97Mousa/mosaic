"""OpenAI LLM Client for MOSAIC LLM Worker.

This client works with OpenAI API and any OpenAI-compatible endpoints
(e.g., vLLM, llama.cpp, Ollama).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Import openai - this module requires the openai package
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    openai = None


class OpenAIClient(BaseLLMClient):
    """OpenAI API client.

    Works with:
    - OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
    - Azure OpenAI
    - vLLM (OpenAI-compatible endpoint)
    - llama.cpp server
    - Ollama
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
        """Initialize OpenAI client.

        Args:
            model_id: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
            api_key: OpenAI API key. Required unless using local endpoint.
            base_url: Custom API URL for vLLM/llama.cpp/Ollama.
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.

        Raises:
            ImportError: If openai package is not installed.
        """
        if not _HAS_OPENAI:
            raise ImportError(
                "OpenAI client requires 'openai' package. "
                "Install with: pip install openai"
            )

        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Create OpenAI client
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenAI API.

        Args:
            prompt: User message/prompt.
            system: Optional system message.
            **kwargs: Additional arguments (stop, n, etc.).

        Returns:
            LLMResponse with generated content.

        Raises:
            openai.APIError: On API request errors.
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stop=kwargs.get("stop"),
            )
        finally:
            latency_ms = (time.time() - start_time) * 1000

        # Parse response
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason if choice else None,
            latency_ms=latency_ms,
        )


__all__ = ["OpenAIClient"]
