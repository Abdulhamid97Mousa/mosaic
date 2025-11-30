"""Unified evaluation system for all CleanRL algorithms.

This package provides a single evaluation loop that works with all CleanRL algorithms
using algorithm-specific ActionSelector adapters.
"""

from __future__ import annotations

from .base import ActionSelector
from .evaluator import evaluate, EvalResult
from .registry import get_adapter, ADAPTER_REGISTRY

__all__ = [
    "ActionSelector",
    "evaluate",
    "EvalResult",
    "get_adapter",
    "ADAPTER_REGISTRY",
]
