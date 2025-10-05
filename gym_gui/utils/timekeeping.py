from __future__ import annotations

"""Utility helpers for tracking timing milestones inside the Qt shell."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

__all__ = [
    "format_timedelta",
    "format_timestamp",
    "ElapsedClock",
    "SessionTimers",
]


def format_timedelta(delta: timedelta) -> str:
    """Format a :class:`datetime.timedelta` as ``HH:MM:SS`` with zero padding."""

    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamp(timestamp: Optional[datetime]) -> str:
    """Return a human-friendly ``HH:MM:SS`` representation or an em dash."""

    if timestamp is None:
        return "—"
    return timestamp.strftime("%H:%M:%S")


@dataclass(slots=True)
class ElapsedClock:
    """Monitors elapsed time relative to a starting point."""

    started_at: datetime = field(default_factory=datetime.now)

    def reset(self, *, start_at: Optional[datetime] = None) -> None:
        self.started_at = start_at or datetime.now()

    def elapsed(self) -> timedelta:
        return datetime.now() - self.started_at

    def formatted(self) -> str:
        return format_timedelta(self.elapsed())


@dataclass(slots=True)
class SessionTimers:
    """Tracks the key timing milestones for a play session."""

    launch_at: datetime = field(default_factory=datetime.now)
    first_move_at: datetime | None = None
    outcome_at: datetime | None = None

    def reset_episode(self) -> None:
        """Clear per-episode markers while preserving the launch timestamp."""

        self.first_move_at = None
        self.outcome_at = None

    def mark_first_move(self, when: Optional[datetime] = None) -> None:
        if self.first_move_at is None:
            self.first_move_at = when or datetime.now()

    def mark_outcome(self, when: Optional[datetime] = None) -> None:
        self.outcome_at = when or datetime.now()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def elapsed_since_launch(self) -> timedelta:
        return datetime.now() - self.launch_at

    def elapsed_since_first_move(self) -> timedelta | None:
        if self.first_move_at is None:
            return None
        return datetime.now() - self.first_move_at

    def launch_elapsed_formatted(self) -> str:
        return format_timedelta(self.elapsed_since_launch())

    def first_move_elapsed_formatted(self) -> str:
        delta = self.elapsed_since_first_move()
        return format_timedelta(delta) if delta is not None else "—"

    def outcome_timestamp_formatted(self) -> str:
        return format_timestamp(self.outcome_at)
