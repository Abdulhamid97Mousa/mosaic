"""MOSAIC LLM Worker Clients - LLM API Integrations.

This module provides LLM client implementations for various providers.
All clients implement the BaseLLMClient interface for interchangeability.
"""

from .base import BaseLLMClient, LLMResponse, create_client
from .openrouter import OpenRouterClient

# Conditional imports for optional providers
try:
    from .openai_client import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .anthropic_client import AnthropicClient
except ImportError:
    AnthropicClient = None

try:
    from .google_client import GoogleClient
except ImportError:
    GoogleClient = None

__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "create_client",
    "OpenRouterClient",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
]
