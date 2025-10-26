"""Qt signals for trainer lifecycle events.

This module provides a centralized signal bus for training lifecycle events:
- training_started: Emitted when a training run begins
- episode_finalized: Emitted after each episode completes
- training_finished: Emitted when training completes or fails
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, QMutex
from PyQt6.QtCore import pyqtSignal as Signal

_LOGGER = logging.getLogger(__name__)


class TrainerSignals(QObject):
    """Qt signals for trainer lifecycle events.

    This class provides a centralized signal bus for training lifecycle events.
    All signals are emitted on the Qt main thread for safe UI updates.
    """

    # Signal: training_started(run_id: str, metadata: dict)
    # Emitted when a training run is submitted and begins execution
    training_started = Signal(str, dict)

    # Signal: episode_finalized(run_id: str, agent_id: str, episode_index: int, rollup: dict)
    # Emitted after each episode completes with episode statistics
    episode_finalized = Signal(str, str, int, dict)

    # Signal: training_finished(run_id: str, outcome: str, failure_reason: Optional[str])
    # Emitted when training completes (outcome: "succeeded", "failed", "canceled")
    training_finished = Signal(str, str, str)

    # Signal: run_heartbeat(run_id: str, timestamp: str)
    # Emitted periodically to indicate the run is still alive
    run_heartbeat = Signal(str, str)

    _instance: Optional[TrainerSignals] = None
    _lock = QMutex()
    # No class-wide init flags; guard per-instance to avoid double QObject init

    def __new__(cls) -> TrainerSignals:
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._lock.lock()
            try:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
            finally:
                cls._lock.unlock()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the signals object (called only once)."""
        if not hasattr(self, "_initialized"):
            try:
                super().__init__()
                self._initialized = True
                _LOGGER.debug("TrainerSignals initialized")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize TrainerSignals: {e}", exc_info=True)
                # Mark as initialized anyway to prevent repeated attempts
                self._initialized = True
                raise

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


def get_trainer_signals() -> TrainerSignals:
    """Get the singleton TrainerSignals instance.
    
    Returns:
        The global TrainerSignals instance
    """
    return TrainerSignals()


__all__ = ["TrainerSignals", "get_trainer_signals"]

