"""Shared constants for vectorised environment metadata."""

from __future__ import annotations

# Canonical autoreset modes exposed by Gymnasium's vector environments.
DEFAULT_AUTORESET_MODE = "NextStep"
SUPPORTED_AUTORESET_MODES: frozenset[str] = frozenset({"NextStep", "SameStep", "Disabled"})

# Telemetry metadata keys related to vectorised environments.
RESET_MASK_KEY = "reset_mask"
VECTOR_ENV_INDEX_KEY = "vector_index"
VECTOR_ENV_BATCH_SIZE_KEY = "batch_size"
VECTOR_SEED_KEY = "seeds"

__all__ = [
    "DEFAULT_AUTORESET_MODE",
    "SUPPORTED_AUTORESET_MODES",
    "RESET_MASK_KEY",
    "VECTOR_ENV_INDEX_KEY",
    "VECTOR_ENV_BATCH_SIZE_KEY",
    "VECTOR_SEED_KEY",
]
