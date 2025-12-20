"""LLM service layer for Chat with Mosaic feature.

Provides unified access to LLM providers (OpenRouter, vLLM) with
provider-namespaced model identities.

This module is optional - it requires the `requests` library.
Install with: pip install -e ".[chat]"

Usage:
    >>> from gym_gui.services.llm import LLM_CHAT_AVAILABLE
    >>> if LLM_CHAT_AVAILABLE:
    ...     from gym_gui.services.llm import LLMService, ChatMessage
    ...     # Use the service
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_LOGGER = logging.getLogger(__name__)

# Always import types for static type checking (pyright/mypy)
# These imports are only used by type checkers, not at runtime
if TYPE_CHECKING:
    from gym_gui.services.llm.models import LLMProvider, ModelIdentity
    from gym_gui.services.llm.config import LLMConfig
    from gym_gui.services.llm.service import (
        LLMService,
        ChatMessage,
        CompletionResult,
        LLMServiceError,
    )
    from gym_gui.services.llm.model_manager import (
        ModelStatus,
        ModelInfo,
        HuggingFaceAuth,
        ProxyConfig,
        ModelDownloader,
        VLLMServerManager,
        ModelManager,
        MODEL_SIZES_GB,
    )
    from gym_gui.services.llm.gpu_detector import (
        GPUInfo,
        GPUDetectionResult,
        GPUDetector,
    )

# Check if requests is available (the only external dependency)
try:
    import requests  # noqa: F401

    LLM_CHAT_AVAILABLE = True
except ImportError:
    LLM_CHAT_AVAILABLE = False
    _LOGGER.info(
        "LLM Chat feature not available: 'requests' library not installed. "
        "Install with: pip install -e \".[chat]\""
    )

# Runtime imports - only if dependencies are available
if LLM_CHAT_AVAILABLE:
    from gym_gui.services.llm.models import LLMProvider, ModelIdentity
    from gym_gui.services.llm.config import LLMConfig
    from gym_gui.services.llm.service import (
        LLMService,
        ChatMessage,
        CompletionResult,
        LLMServiceError,
    )
    from gym_gui.services.llm.model_manager import (
        ModelStatus,
        ModelInfo,
        HuggingFaceAuth,
        ProxyConfig,
        ModelDownloader,
        VLLMServerManager,
        ModelManager,
        MODEL_SIZES_GB,
    )
    from gym_gui.services.llm.gpu_detector import (
        GPUInfo,
        GPUDetectionResult,
        GPUDetector,
    )

__all__ = [
    "LLM_CHAT_AVAILABLE",
    "LLMProvider",
    "ModelIdentity",
    "LLMConfig",
    "LLMService",
    "ChatMessage",
    "CompletionResult",
    "LLMServiceError",
    "ModelStatus",
    "ModelInfo",
    "HuggingFaceAuth",
    "ProxyConfig",
    "ModelDownloader",
    "VLLMServerManager",
    "ModelManager",
    "MODEL_SIZES_GB",
    "GPUInfo",
    "GPUDetectionResult",
    "GPUDetector",
]
