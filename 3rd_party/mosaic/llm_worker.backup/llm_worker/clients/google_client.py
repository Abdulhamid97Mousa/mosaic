"""Google Generative AI (Gemini) Client for LLM Worker.

Provides integration with Google's Generative AI API.
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


def process_image_google(image) -> Dict[str, Any]:
    """Process an image for Google GenAI API by converting it to base64.

    Args:
        image: PIL Image or similar to process.

    Returns:
        Dict with image data formatted for Google GenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {
        "mime_type": "image/png",
        "data": base64_image,
    }


class GoogleClient(BaseLLMClient):
    """Client for interacting with Google's Generative AI (Gemini) API."""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        timeout: float = 60.0,
        thinking_budget: int = -1,
    ):
        """Initialize the Google GenAI client.

        Args:
            model_id: Model identifier (e.g., 'gemini-1.5-pro').
            api_key: API key for authentication (uses GOOGLE_API_KEY env var if not provided).
            base_url: Not used for Google GenAI, kept for interface compatibility.
            temperature: Generation temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            thinking_budget: Budget for thinking/reasoning (-1 for default).
        """
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self.thinking_budget = thinking_budget
        self._initialized = False
        self._client = None
        self._generation_config = None

    def _initialize_client(self):
        """Initialize the Google GenAI client if not already initialized."""
        if self._initialized:
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Google GenAI client requires 'google-genai' package. "
                "Install with: pip install google-genai"
            )

        self._client = genai.Client()

        # Build generation config
        config_kwargs = {
            "max_output_tokens": self.max_tokens,
        }

        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        # Add thinking config if budget is specified
        if self.thinking_budget >= 0:
            self._generation_config = genai.types.GenerateContentConfig(
                **config_kwargs,
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
            )
        else:
            self._generation_config = genai.types.GenerateContentConfig(**config_kwargs)

        self._initialized = True

    def _convert_messages(self, messages: List) -> List:
        """Convert messages to Google GenAI format.

        Args:
            messages: List of message objects with role and content.

        Returns:
            List of Content objects for Google GenAI API.
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Google GenAI client requires 'google-genai' package. "
                "Install with: pip install google-genai"
            )

        converted_messages = []

        for msg in messages:
            parts = []

            # Map roles: assistant -> model, system -> user
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"

            if msg.content:
                parts.append(types.Part(text=msg.content))

            if hasattr(msg, "attachment") and msg.attachment is not None:
                parts.append(types.Part(image=msg.attachment))

            converted_messages.append(
                types.Content(role=role, parts=parts)
            )

        return converted_messages

    def _extract_completion(self, response) -> str:
        """Extract completion text from API response.

        Args:
            response: Response object from Google GenAI API.

        Returns:
            Extracted completion text.

        Raises:
            Exception: If response is invalid or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")

        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        text = getattr(content_parts[0], "text", None)
        if text is None:
            raise Exception("No text found in the content parts.")

        return text.strip()

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from Google GenAI.

        Args:
            prompt: User prompt/message.
            system: Optional system message (prepended to prompt).
            messages: Optional list of message objects (overrides prompt if provided).
            **kwargs: Additional arguments (ignored).

        Returns:
            LLMResponse with generated content.

        Raises:
            Exception: On API errors.
        """
        self._initialize_client()

        start_time = time.time()

        # Convert messages if provided, otherwise create from prompt
        if messages is not None:
            converted_messages = self._convert_messages(messages)
        else:
            from google.genai import types

            # Build simple message list
            converted_messages = []
            if system:
                converted_messages.append(
                    types.Content(role="user", parts=[types.Part(text=system)])
                )
            converted_messages.append(
                types.Content(role="user", parts=[types.Part(text=prompt)])
            )

        # Make API call
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=converted_messages,
            config=self._generation_config,
        )

        # Extract completion
        completion = self._extract_completion(response)

        latency_ms = (time.time() - start_time) * 1000

        # Extract token usage if available
        usage = None
        if response and hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": (
                    getattr(response.usage_metadata, "prompt_token_count", 0) +
                    getattr(response.usage_metadata, "candidates_token_count", 0)
                ),
            }

        # Extract finish reason
        finish_reason = None
        if response and hasattr(response, "candidates") and response.candidates:
            finish_reason = str(getattr(response.candidates[0], "finish_reason", "unknown"))

        return LLMResponse(
            content=completion,
            model=self.model_id,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )


__all__ = ["GoogleClient", "process_image_google"]
