from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any


class InteractionController(ABC):
    """Family-specific stepping/idle/passive-action policy."""

    @abstractmethod
    def idle_interval_ms(self) -> Optional[int]:
        """Desired idle tick interval in milliseconds, or None to disable idle ticking."""
        raise NotImplementedError

    @abstractmethod
    def should_idle_tick(self) -> bool:
        """Whether an idle tick should be performed now (timer gating)."""
        raise NotImplementedError

    @abstractmethod
    def maybe_passive_action(self) -> Optional[Any]:
        """Return a passive action to apply on idle, or None if not applicable."""
        raise NotImplementedError

    @abstractmethod
    def step_dt(self) -> float:
        """Fixed simulation dt if applicable (physics), else 0.0."""
        raise NotImplementedError


class Box2DInteractionController(InteractionController):
    def __init__(self, owner, target_hz: int = 50):
        # owner is SessionController; used to reuse existing passive action logic safely
        self._owner = owner
        self._dt = 1.0 / float(target_hz)
        self._interval_ms = int(1000 / float(target_hz))

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        # Delegate to owner's existing gating to keep parity
        return self._owner._control_mode.name == "HUMAN_ONLY" and not self._owner._game_paused and self._owner._passive_action is not None and self._owner._adapter is not None and self._owner._game_id is not None

    def maybe_passive_action(self) -> Optional[Any]:
        # Reuse owner's resolved passive action (kept in sync by owner)
        return self._owner._passive_action

    def step_dt(self) -> float:
        return self._dt


class TurnBasedInteractionController(InteractionController):
    def idle_interval_ms(self) -> Optional[int]:
        return None

    def should_idle_tick(self) -> bool:
        return False

    def maybe_passive_action(self) -> Optional[Any]:
        return None

    def step_dt(self) -> float:
        return 0.0


class AleInteractionController(InteractionController):
    """Idle controller for Atari/ALE: step continuously with NOOP when idle."""

    def __init__(self, owner, target_hz: int = 60):
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~16ms

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        # Human-only, not paused, adapter and game present, episode not finished
        o = self._owner
        if o._adapter is None or o._game_id is None or o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        # ALE NOOP is action 0 in minimal action set
        return 0

    def step_dt(self) -> float:
        return 0.0
