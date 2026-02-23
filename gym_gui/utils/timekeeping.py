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
        # If episode has ended, calculate time from first move to outcome
        # Otherwise, calculate time from first move to now
        end_time = self.outcome_at if self.outcome_at is not None else datetime.now()
        return end_time - self.first_move_at

    def launch_elapsed_formatted(self) -> str:
        return format_timedelta(self.elapsed_since_launch())

    def first_move_elapsed_formatted(self) -> str:
        delta = self.elapsed_since_first_move()
        return format_timedelta(delta) if delta is not None else "—"

    def outcome_elapsed_formatted(self) -> str:
        """Return elapsed time between the first move and the outcome.

        The control panel labels this as "Outcome Time" to emphasise how long
        it took to finish the episode after the first interaction. When an
        outcome has not been recorded yet the em dash placeholder is returned.
        """

        delta = self.episode_duration()
        return format_timedelta(delta) if delta is not None else "—"

    def outcome_timestamp_formatted(self) -> str:
        """Backward-compatible alias for :meth:`outcome_elapsed_formatted`."""

        return self.outcome_elapsed_formatted()

    def outcome_wall_clock_formatted(self) -> str:
        """Return the wall-clock timestamp for the recorded outcome."""

        return format_timestamp(self.outcome_at)

    def episode_duration(self) -> timedelta | None:
        """Calculate the total episode duration (first move to outcome)."""
        if self.first_move_at is None or self.outcome_at is None:
            return None
        return self.outcome_at - self.first_move_at

    def episode_duration_formatted(self) -> str:
        """Return formatted episode duration."""
        delta = self.episode_duration()
        return format_timedelta(delta) if delta is not None else "—"