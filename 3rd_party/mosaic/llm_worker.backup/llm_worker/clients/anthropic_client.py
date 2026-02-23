"""Anthropic LLM Client for MOSAIC LLM Worker.

Direct integration with Anthropic's Claude API.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Import anthropic - this module requires the anthropic package
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False
    anthropic = None


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client.

    Direct access to Claude models (claude-3-opus, claude-3-sonnet, etc.).
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
        """Initialize Anthropic client.

        Args:
            model_id: Model identifier (e.g., "claude-3-sonnet-20240229").
            api_key: Anthropic API key. Required.
            base_url: Custom API URL (rarely needed).
            temperature: Generation temperature (0.0-1.0 for Anthropic).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.

        Raises:
            ImportError: If anthropic package is not installed.
        """
        if not _HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic client requires 'anthropic' package. "
                "Install with: pip install anthropic"
            )

        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=min(temperature, 1.0),  # Anthropic max is 1.0
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Create Anthropic client
        self._client = anthropic.Anthropic(
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
        """Generate a response using Anthropic API.

        Args:
            prompt: User message/prompt.
            system: Optional system message.
            **kwargs: Additional arguments (stop_sequences, etc.).

        Returns:
            LLMResponse with generated content.

        Raises:
            anthropic.APIError: On API request errors.
        """
        # Build messages (Anthropic doesn't use system in messages array)
        messages = [{"role": "user", "content": prompt}]

        create_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add system message if provided
        if system:
            create_kwargs["system"] = system

        # Add stop sequences if provided
        if "stop" in kwargs:
            create_kwargs["stop_sequences"] = kwargs["stop"]

        start_time = time.time()
        try:
            response = self._client.messages.create(**create_kwargs)
        finally:
            latency_ms = (time.time() - start_time) * 1000

        # Parse response
        content = ""
        if response.content:
            # Anthropic returns list of content blocks
            text_blocks = [
                block.text for block in response.content
                if hasattr(block, 'text')
            ]
            content = "".join(text_blocks)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
        )


__all__ = ["AnthropicClient"]
