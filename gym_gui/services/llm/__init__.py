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

_LOGGER = logging.getLogger(__name__)

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

# Only export the real classes if dependencies are available
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
else:
    # Placeholder exports when dependencies not available
    LLMProvider = None  # type: ignore[misc, assignment]
    ModelIdentity = None  # type: ignore[misc, assignment]
    LLMConfig = None  # type: ignore[misc, assignment]
    LLMService = None  # type: ignore[misc, assignment]
    ChatMessage = None  # type: ignore[misc, assignment]
    CompletionResult = None  # type: ignore[misc, assignment]
    LLMServiceError = None  # type: ignore[misc, assignment]
    ModelStatus = None  # type: ignore[misc, assignment]
    ModelInfo = None  # type: ignore[misc, assignment]
    HuggingFaceAuth = None  # type: ignore[misc, assignment]
    ProxyConfig = None  # type: ignore[misc, assignment]
    ModelDownloader = None  # type: ignore[misc, assignment]
    VLLMServerManager = None  # type: ignore[misc, assignment]
    ModelManager = None  # type: ignore[misc, assignment]
    MODEL_SIZES_GB = None  # type: ignore[misc, assignment]
    GPUInfo = None  # type: ignore[misc, assignment]
    GPUDetectionResult = None  # type: ignore[misc, assignment]
    GPUDetector = None  # type: ignore[misc, assignment]

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
