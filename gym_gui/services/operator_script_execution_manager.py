"""Script Execution Manager for automated baseline operator experiments.

This module provides the OperatorScriptExecutionManager class that orchestrates
automated execution of baseline experiments defined in Python scripts.

Key responsibilities:
- Episode state machine (launch → reset → step → episode_end → next_episode)
- Seed progression across episodes
- Operator subprocess lifecycle management
- Progress tracking and reporting

Architecture:
    ScriptExperimentWidget (UI)
        → OperatorScriptExecutionManager (Logic)
        → MainWindow (Communication)
        → Operator Subprocesses

Separation from Manual Mode:
    Manual Mode uses OperatorsTab with shared seed execution.
    Script Mode uses this manager with predefined seed sequences.
    They operate completely independently without shared state.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal, QTimer

from gym_gui.services.operator import OperatorConfig


_LOGGER = logging.getLogger(__name__)


class OperatorScriptExecutionManager(QtCore.QObject):
    """Manages automated execution of baseline operator experiments from scripts.

    This class implements an event-driven state machine for running multiple
    episodes with different seeds automatically. It handles:
    - Launching operators with initial seeds
    - Stepping operators until episode completion
    - Advancing to next episode with new seed
    - Tracking progress and emitting updates

    Signals:
        launch_operator: Request to launch operator subprocess
        reset_operator: Request to reset operator with new seed
        step_operator: Request to step operator
        stop_operator: Request to stop operator
        progress_updated: Progress notification for UI
        experiment_completed: Experiment finished notification
    """

    # Operator control signals (sent to MainWindow)
    launch_operator = pyqtSignal(str, object, int)  # operator_id, config, seed
    reset_operator = pyqtSignal(str, int)  # operator_id, seed
    step_operator = pyqtSignal(str)  # operator_id
    stop_operator = pyqtSignal(str)  # operator_id

    # Progress signals (sent to ScriptExperimentWidget)
    progress_updated = pyqtSignal(int, int, int)  # episode_num, total_episodes, seed
    experiment_completed = pyqtSignal(int)  # num_episodes

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        """Initialize script execution manager.

        Args:
            parent: Optional parent QObject for Qt ownership.
        """
        super().__init__(parent)

        # Execution state
        self._is_running: bool = False
        self._operator_configs: List[OperatorConfig] = []
        self._seeds: List[int] = []
        self._current_episode_idx: int = 0
        self._waiting_for_response: bool = False

        # Step pacing: delay between steps to allow Qt paint events to render each frame.
        # Without this, steps fire faster than the UI can repaint, causing visual jitter.
        self._step_delay_ms: int = 50  # default 50ms between steps

        # Environment mode: "procedural" = different seed per episode (ProcGen-style),
        # "fixed" = same seed every episode (isolate agent variance).
        self._environment_mode: str = "procedural"

        # Track which operators are running
        self._operator_states: Dict[str, str] = {}  # operator_id -> "stopped", "running"

    def start_experiment(
        self,
        operator_configs: List[OperatorConfig],
        execution_config: Dict[str, Any]
    ) -> None:
        """Start automated experiment execution.

        Args:
            operator_configs: List of operator configurations from script.
            execution_config: Execution settings (num_episodes, seeds).
        """
        if self._is_running:
            _LOGGER.warning("Experiment already running, ignoring start request")
            return

        # Extract execution parameters
        num_episodes = execution_config.get("num_episodes", 10)
        seeds = list(execution_config.get("seeds", range(1000, 1000 + num_episodes)))
        self._step_delay_ms = execution_config.get("step_delay_ms", 50)
        self._environment_mode = execution_config.get("environment_mode", "procedural")

        # Extend seeds if user requested more episodes than script defined
        if len(seeds) < num_episodes:
            last_seed = seeds[-1] if seeds else 1000
            seeds.extend(range(last_seed + 1, last_seed + 1 + (num_episodes - len(seeds))))

        # Initialize state
        self._is_running = True
        self._operator_configs = operator_configs
        self._seeds = seeds[:num_episodes]
        self._current_episode_idx = 0
        self._waiting_for_response = False

        # Initialize operator states
        self._operator_states = {
            cfg.operator_id: "stopped" for cfg in operator_configs
        }

        _LOGGER.info(
            f"Starting script experiment: {len(operator_configs)} operators, "
            f"{len(self._seeds)} episodes, seeds {self._seeds[0]}-{self._seeds[-1]}, "
            f"environment_mode={self._environment_mode}"
        )

        # Check if operators are already running (from previous experiment)
        operators_already_running = all(
            self._operator_states.get(cfg.operator_id) == "running"
            for cfg in operator_configs
        )

        # Emit progress for first episode
        first_seed = self._seeds[0]
        self.progress_updated.emit(1, len(self._seeds), first_seed)

        if operators_already_running:
            # Operators exist - reset with first seed
            _LOGGER.info("Operators already running, resetting with first seed")
            self._waiting_for_response = True
            for config in operator_configs:
                self.reset_operator.emit(config.operator_id, first_seed)
        else:
            # Launch new operators with first seed
            _LOGGER.info(f"Launching {len(operator_configs)} operators with seed {first_seed}")
            self._waiting_for_response = True
            for config in operator_configs:
                self.launch_operator.emit(config.operator_id, config, first_seed)

    def stop_experiment(self) -> None:
        """Stop running experiment."""
        if not self._is_running:
            return

        _LOGGER.info("Stopping script experiment")
        self._is_running = False
        self._waiting_for_response = False

        # Stop all running operators
        for operator_id in list(self._operator_states.keys()):
            if self._operator_states.get(operator_id) == "running":
                self.stop_operator.emit(operator_id)
                self._operator_states[operator_id] = "stopped"

    def on_ready_received(self, operator_id: str) -> None:
        """Handle ready response from operator (after reset/launch).

        This triggers automatic stepping.

        Args:
            operator_id: ID of the operator that is ready.
        """
        # Mark operator as running
        self._operator_states[operator_id] = "running"

        _LOGGER.info(
            f"on_ready_received: operator_id={operator_id}, "
            f"is_running={self._is_running}, waiting={self._waiting_for_response}"
        )

        if not self._is_running or not self._waiting_for_response:
            _LOGGER.info("Not in automatic execution mode, ignoring ready")
            return

        # Start automatic stepping for this operator
        _LOGGER.info(f"Starting automatic stepping for {operator_id}")
        self.step_operator.emit(operator_id)

    def on_step_received(self, operator_id: str) -> None:
        """Handle step response from operator.

        Triggers the next step after a pacing delay, giving Qt's event loop
        time to process paint events so the render view updates each frame.

        Args:
            operator_id: ID of the operator that completed a step.
        """
        _LOGGER.debug(
            f"on_step_received: operator_id={operator_id}, "
            f"is_running={self._is_running}, waiting={self._waiting_for_response}"
        )

        if not self._is_running or not self._waiting_for_response:
            return

        # Pace the step loop: delay before sending next step so Qt can paint
        # the current frame. Without this, paint events pile up and the render
        # view shows visual jumps (e.g., step 26 → 37).
        def _emit_next_step():
            if self._is_running and self._waiting_for_response:
                self.step_operator.emit(operator_id)

        QTimer.singleShot(self._step_delay_ms, _emit_next_step)

    def on_episode_ended(
        self,
        operator_id: str,
        terminated: bool,
        truncated: bool
    ) -> None:
        """Handle episode end from operator.

        This advances to the next episode or completes the experiment.

        Args:
            operator_id: ID of the operator that finished an episode.
            terminated: Whether episode terminated naturally.
            truncated: Whether episode was truncated.
        """
        _LOGGER.info(
            f"on_episode_ended: operator_id={operator_id}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"current_episode={self._current_episode_idx + 1}"
        )

        if not self._is_running:
            _LOGGER.info("Not running, ignoring episode end")
            return

        # Move to next episode
        self._current_episode_idx += 1
        self._waiting_for_response = False

        # Check if all episodes complete
        if self._current_episode_idx >= len(self._seeds):
            _LOGGER.info(f"All {len(self._seeds)} episodes completed")
            self._is_running = False
            self.experiment_completed.emit(len(self._seeds))
            return

        # Start next episode
        self._start_next_episode()

    def _start_next_episode(self) -> None:
        """Start the next episode with the next seed."""
        if not self._is_running:
            return

        if self._current_episode_idx >= len(self._seeds):
            # Should not happen, but safety check
            _LOGGER.warning("Attempted to start episode beyond seed list")
            return

        # Fixed mode: reuse seeds[0] every episode (identical layout for credit assignment)
        # Procedural mode: advance seed per episode (generalization across layouts)
        if self._environment_mode == "fixed":
            seed = self._seeds[0]
        else:
            seed = self._seeds[self._current_episode_idx]
        episode_num = self._current_episode_idx + 1

        _LOGGER.info(f"Starting episode {episode_num}/{len(self._seeds)} with seed {seed}")

        # Emit progress update
        self.progress_updated.emit(episode_num, len(self._seeds), seed)

        # Send reset to all operators
        self._waiting_for_response = True
        for config in self._operator_configs:
            self.reset_operator.emit(config.operator_id, seed)

    @property
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self._is_running

    @property
    def current_episode(self) -> int:
        """Get current episode number (1-indexed)."""
        return self._current_episode_idx + 1 if self._is_running else 0

    @property
    def total_episodes(self) -> int:
        """Get total number of episodes."""
        return len(self._seeds)

    @property
    def step_delay_ms(self) -> int:
        """Get step pacing delay in milliseconds."""
        return self._step_delay_ms

    @step_delay_ms.setter
    def step_delay_ms(self, value: int) -> None:
        """Set step pacing delay (can be adjusted during execution)."""
        self._step_delay_ms = max(0, value)
