"""Constants for the safe, bounded episode counter system.

Consolidated from:
- Original: gym_gui/core/constants_episode_counter.py

Defines episode ID formatting, counter bounds, persistence parameters,
and worker configuration for distributed/parallel rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

# ================================================================
# Episode Counter Bounds & Formatting
# ================================================================

# Maximum episodes per run (hard limit for safety)
# Note: 6-digit counter can represent 0-999,999, so max is 999,999
DEFAULT_MAX_EPISODES_PER_RUN = 999_999

# Counter width (number of digits used in episode ID padding)
# 6 digits = 0–999,999 capacity
EPISODE_COUNTER_WIDTH = 6

# Maximum value representable with COUNTER_WIDTH digits
MAX_COUNTER_VALUE = 10 ** EPISODE_COUNTER_WIDTH - 1  # 999,999

# Episode ID format components
EPISODE_ID_SEPARATOR = "-"
EPISODE_ID_PREFIX = "ep"

# ================================================================
# Worker ID Configuration (for distributed rollouts)
# ================================================================

# Worker ID is generated as ULID, same as run_id
# Format: "{run_id}-w{worker_id}-ep{ep_index:06d}"
# If worker_id is None (single-process), use simpler format:
# Format: "{run_id}-ep{ep_index:06d}"

WORKER_ID_PREFIX = "w"  # Prefix for worker ID in episode_id
WORKER_ID_WIDTH = 6  # Worker ID field width (matches counter width for consistency)

# ================================================================
# Database Constraints & Persistence
# ================================================================

# SQLite UNIQUE constraint on episodes table:
# - Single-process: UNIQUE(run_id, ep_index)
# - Distributed: UNIQUE(run_id, worker_id, ep_index)
# Both prevent accidental reuse of episode indices within a run/worker.

# Column names for episode tracking
EPISODE_ID_COLUMN = "episode_id"
EP_INDEX_COLUMN = "ep_index"
WORKER_ID_COLUMN = "worker_id"

# Run table max_episodes column (stores per-run limit)
MAX_EPISODES_COLUMN = "max_episodes_per_run"

# ================================================================
# Thread Safety & Concurrency
# ================================================================

# Lock timeout for thread-safe counter access (seconds)
COUNTER_LOCK_TIMEOUT_S = 5.0

# ================================================================
# Resume & Validation
# ================================================================

# Sanity check: refuse to resume if stored counter > this fraction of max
# (e.g., 0.95 means "warn if using >95% of capacity")
RESUME_CAPACITY_WARNING_THRESHOLD = 0.95

# Error message patterns (for validation on load)
COUNTER_NOT_INITIALIZED_ERROR = "RunCounterManager not initialized; call initialize() first"
MAX_EPISODES_REACHED_ERROR = (
    "Max episodes ({max_eps}) reached for run {run_id}. "
    "Consider rotating to a new run or increasing max_episodes."
)
COUNTER_EXCEEDS_MAX_ERROR = (
    "Stored episode index ({idx}) exceeds max_episodes ({max_eps}). "
    "Refusing to resume; create a new run or increase max_episodes."
)
INVALID_MAX_EPISODES_ERROR = "max_episodes must be positive, got {value}"
COUNTER_CAPACITY_EXCEEDED_ERROR = (
    "max_episodes ({max_eps}) exceeds counter capacity ({capacity}). "
    "Use a wider counter width or lower max_episodes."
)

# ================================================================
# Dataclass for centralized configuration
# ================================================================


@dataclass(frozen=True)
class EpisodeCounterConfig:
    """Centralized episode counter configuration."""

    max_episodes_per_run: int = DEFAULT_MAX_EPISODES_PER_RUN
    counter_width: int = EPISODE_COUNTER_WIDTH
    worker_id_width: int = WORKER_ID_WIDTH
    lock_timeout_s: float = COUNTER_LOCK_TIMEOUT_S
    resume_capacity_warning_threshold: float = RESUME_CAPACITY_WARNING_THRESHOLD

    @property
    def max_counter_value(self) -> int:
        """Maximum value representable with configured counter width."""
        return 10 ** self.counter_width - 1


# ================================================================
# Utility Functions
# ================================================================


def format_episode_id(run_id: str, ep_index: int, worker_id: str | None = None) -> str:
    """Format a complete episode ID.

    Args:
        run_id: Run identifier (ULID)
        ep_index: Episode index within the run (0–max_counter_value)
        worker_id: Optional worker ID for distributed scenarios

    Returns:
        Formatted episode ID string.
        - With worker_id: "{run_id}-w{worker_id}-ep{ep_index:06d}"
        - Without worker_id: "{run_id}-ep{ep_index:06d}"
    """
    if worker_id is not None:
        return f"{run_id}{EPISODE_ID_SEPARATOR}{WORKER_ID_PREFIX}{worker_id}{EPISODE_ID_SEPARATOR}{EPISODE_ID_PREFIX}{ep_index:0{EPISODE_COUNTER_WIDTH}d}"
    else:
        return f"{run_id}{EPISODE_ID_SEPARATOR}{EPISODE_ID_PREFIX}{ep_index:0{EPISODE_COUNTER_WIDTH}d}"


def parse_episode_id(episode_id: str) -> dict[str, str | int | None]:
    """Parse a formatted episode ID back into components.

    Args:
        episode_id: Formatted episode ID string

    Returns:
        Dict with keys: "run_id", "worker_id" (or None), "ep_index"

    Raises:
        ValueError: If episode_id format is invalid
    """
    parts = episode_id.split(EPISODE_ID_SEPARATOR)

    if len(parts) == 2:
        # Format: "{run_id}-ep{ep_index:06d}"
        run_id, ep_part = parts
        if not ep_part.startswith(EPISODE_ID_PREFIX):
            raise ValueError(f"Invalid episode ID format: {episode_id}")
        try:
            ep_index = int(ep_part[len(EPISODE_ID_PREFIX) :])
        except ValueError:
            raise ValueError(f"Invalid episode index in ID: {episode_id}")
        return {"run_id": run_id, "worker_id": None, "ep_index": ep_index}

    elif len(parts) == 3:
        # Format: "{run_id}-w{worker_id}-ep{ep_index:06d}"
        run_id, worker_part, ep_part = parts
        if not worker_part.startswith(WORKER_ID_PREFIX):
            raise ValueError(f"Invalid worker ID format in: {episode_id}")
        if not ep_part.startswith(EPISODE_ID_PREFIX):
            raise ValueError(f"Invalid episode part format in: {episode_id}")
        try:
            worker_id = worker_part[len(WORKER_ID_PREFIX) :]
            ep_index = int(ep_part[len(EPISODE_ID_PREFIX) :])
        except ValueError:
            raise ValueError(f"Invalid episode ID format: {episode_id}")
        return {"run_id": run_id, "worker_id": worker_id, "ep_index": ep_index}

    else:
        raise ValueError(f"Invalid episode ID format (wrong number of separators): {episode_id}")


__all__ = [
    "DEFAULT_MAX_EPISODES_PER_RUN",
    "EPISODE_COUNTER_WIDTH",
    "MAX_COUNTER_VALUE",
    "EPISODE_ID_SEPARATOR",
    "EPISODE_ID_PREFIX",
    "WORKER_ID_PREFIX",
    "WORKER_ID_WIDTH",
    "EPISODE_ID_COLUMN",
    "EP_INDEX_COLUMN",
    "WORKER_ID_COLUMN",
    "MAX_EPISODES_COLUMN",
    "COUNTER_LOCK_TIMEOUT_S",
    "RESUME_CAPACITY_WARNING_THRESHOLD",
    "COUNTER_NOT_INITIALIZED_ERROR",
    "MAX_EPISODES_REACHED_ERROR",
    "COUNTER_EXCEEDS_MAX_ERROR",
    "INVALID_MAX_EPISODES_ERROR",
    "COUNTER_CAPACITY_EXCEEDED_ERROR",
    "EpisodeCounterConfig",
    "format_episode_id",
    "parse_episode_id",
]
