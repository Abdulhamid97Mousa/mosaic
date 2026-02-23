"""OpenRouter LLM Client for MOSAIC LLM Worker.

OpenRouter provides access to multiple LLM providers through a unified API.
This is the default client for MOSAIC LLM Worker.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client.

    OpenRouter provides access to multiple LLM providers (OpenAI, Anthropic,
    Google, Meta, etc.) through a unified API with competitive pricing.
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
        """Initialize OpenRouter client.

        Args:
            model_id: Model identifier (e.g., "anthropic/claude-3.5-sonnet").
            api_key: OpenRouter API key. Required for API calls.
            base_url: Custom API URL (default: OpenRouter API).
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url or OPENROUTER_API_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using OpenRouter API.

        Args:
            prompt: User message/prompt.
            system: Optional system message.
            **kwargs: Additional arguments (stop, n, etc.).

        Returns:
            LLMResponse with generated content.

        Raises:
            ValueError: If API key is not set.
            httpx.HTTPError: On API request errors.
        """
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add optional parameters
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/MOSAIC-RL/GUI_BDI_RL",
            "X-Title": "MOSAIC LLM Worker",
        }

        start_time = time.time()
        try:
            response = self._client.post(
                self.base_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000

        # Parse response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=data.get("model", self.model_id),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
        )

    def __del__(self):
        """Close HTTP client on cleanup."""
        if hasattr(self, "_client"):
            self._client.close()


__all__ = ["OpenRouterClient"]
