"""Qt signals for trainer lifecycle events.

This module provides a centralized signal bus for training lifecycle events:
- training_started: Emitted when a training run begins
- episode_finalized: Emitted after each episode completes
- training_finished: Emitted when training completes or fails
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

_LOGGER = logging.getLogger(__name__)


class TrainerSignals(QtCore.QObject):
    """Qt signals for trainer lifecycle events."""

    training_started = pyqtSignal(str, dict)
    episode_finalized = pyqtSignal(str, str, int, dict)
    training_finished = pyqtSignal(str, str, str)
    run_heartbeat = pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        _LOGGER.debug("TrainerSignals initialized")

    def emit_training_started(self, run_id: str, metadata: Dict[str, Any]) -> None:
        """Emit training_started signal.
        
        Args:
            run_id: Unique identifier for the training run
            metadata: Dictionary containing run metadata (game_id, agent_id, etc.)
        """
        _LOGGER.info(
            "Emitting training_started",
            extra={"run_id": run_id, "agent_id": metadata.get("agent_id")},
        )
        self.training_started.emit(run_id, metadata)

    def emit_episode_finalized(
        self,
        run_id: str,
        agent_id: str,
        episode_index: int,
        rollup: Dict[str, Any],
    ) -> None:
        """Emit episode_finalized signal.
        
        Args:
            run_id: Unique identifier for the training run
            agent_id: Identifier for the agent
            episode_index: Zero-based episode number
            rollup: Dictionary containing episode statistics (reward, steps, success, etc.)
        """
        _LOGGER.debug(
            "Emitting episode_finalized",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "episode_index": episode_index,
                "reward": rollup.get("reward"),
            },
        )
        self.episode_finalized.emit(run_id, agent_id, episode_index, rollup)

    def emit_training_finished(
        self,
        run_id: str,
        outcome: str,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Emit training_finished signal.
        
        Args:
            run_id: Unique identifier for the training run
            outcome: Terminal status ("succeeded", "failed", "canceled")
            failure_reason: Optional error message if outcome is "failed"
        """
        _LOGGER.info(
            "Emitting training_finished",
            extra={
                "run_id": run_id,
                "outcome": outcome,
                "failure_reason": failure_reason,
            },
        )
        self.training_finished.emit(run_id, outcome, failure_reason or "")

    def emit_run_heartbeat(self, run_id: str, timestamp: str) -> None:
        """Emit run_heartbeat signal.
        
        Args:
            run_id: Unique identifier for the training run
            timestamp: ISO-8601 timestamp of the heartbeat
        """
        self.run_heartbeat.emit(run_id, timestamp)


_TRAINER_SIGNALS: Optional[TrainerSignals] = None


def get_trainer_signals() -> TrainerSignals:
    """Return the shared TrainerSignals instance."""

    global _TRAINER_SIGNALS
    if _TRAINER_SIGNALS is None:
        _TRAINER_SIGNALS = TrainerSignals()
    return _TRAINER_SIGNALS


__all__ = ["TrainerSignals", "get_trainer_signals"]
