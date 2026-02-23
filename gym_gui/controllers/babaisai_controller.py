"""BabaIsAI game controller for Human Control gameplay.

This controller manages the turn-based flow for human players in BabaIsAI,
coordinating the adapter with the RGB renderer.

BabaIsAI is a rule manipulation puzzle game where players push word blocks
to change the rules of the game world.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from qtpy import QtCore
from qtpy.QtCore import Signal as pyqtSignal

from gym_gui.core.adapters.babaisai import BabaIsAIAdapter, BabaIsAIConfig, BABAISAI_ACTIONS

_LOG = logging.getLogger(__name__)


@dataclass
class BabaIsAIState:
    """State snapshot for BabaIsAI environment.

    Attributes:
        observation: Current observation dict with text, image, rules, prompt
        active_rules: List of currently active rules (e.g., "baba is you")
        step_count: Number of steps taken
        is_game_over: Whether the puzzle is solved or failed
        reward: Cumulative reward
    """

    observation: Dict[str, Any]
    active_rules: List[str]
    step_count: int
    is_game_over: bool
    reward: float


class BabaIsAIGameController(QtCore.QObject):
    """Controller for BabaIsAI Human Control gameplay.

    This controller manages:
    - Single-player turn-based flow
    - Action execution via keyboard input
    - Game state synchronization with the UI
    - Rule change notifications

    Signals:
        state_changed(BabaIsAIState): Emitted when game state changes
        game_started(): Emitted when a new game starts
        game_over(bool): Emitted when game ends (won: True/False)
        action_executed(str): Emitted when an action is taken
        rules_changed(list): Emitted when active rules change
        error_occurred(str): Emitted on errors
        status_message(str): Emitted for status updates

    Usage:
        controller = BabaIsAIGameController()
        controller.state_changed.connect(renderer.update_frame)
        controller.start_game(env_id="two_room-break_stop-make_win")

        # From keyboard handler:
        controller.submit_action("UP")  # or 0 for action index
    """

    # Signals
    state_changed = pyqtSignal(object)  # BabaIsAIState
    game_started = pyqtSignal()
    game_over = pyqtSignal(bool)  # won
    action_executed = pyqtSignal(str)  # action name
    rules_changed = pyqtSignal(list)  # new rules list
    error_occurred = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        """Initialize the BabaIsAI game controller.

        Args:
            parent: Parent QObject for memory management
        """
        super().__init__(parent)

        self._adapter: Optional[BabaIsAIAdapter] = None
        self._current_state: Optional[BabaIsAIState] = None
        self._game_active: bool = False
        self._cumulative_reward: float = 0.0
        self._previous_rules: List[str] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_game(
        self,
        env_id: str = "two_room-break_stop-make_win",
        seed: Optional[int] = None,
    ) -> None:
        """Start a new BabaIsAI game.

        Args:
            env_id: Specific puzzle environment ID
            seed: Random seed for environment reset
        """
        _LOG.info(f"Starting BabaIsAI game: env_id={env_id}, seed={seed}")

        # Create and load adapter
        config = BabaIsAIConfig(env_id=env_id, seed=seed)
        self._adapter = BabaIsAIAdapter(config=config)

        try:
            self._adapter.load()
            step_result = self._adapter.reset(seed=seed)
        except Exception as e:
            _LOG.error(f"Failed to load BabaIsAI environment: {e}")
            self.error_occurred.emit(f"Failed to load BabaIsAI: {e}")
            return

        self._game_active = True
        self._cumulative_reward = 0.0

        # Build initial state
        observation = step_result.observation
        rules = observation.get("rules", [])
        self._previous_rules = rules.copy()

        self._current_state = BabaIsAIState(
            observation=observation,
            active_rules=rules,
            step_count=0,
            is_game_over=False,
            reward=0.0,
        )

        self.game_started.emit()
        self.state_changed.emit(self._current_state)
        self.status_message.emit(f"BabaIsAI puzzle '{env_id}' started. Use arrow keys to move.")

    def reset_game(self, seed: Optional[int] = None) -> None:
        """Reset the current game.

        Args:
            seed: New random seed
        """
        if self._adapter is None:
            self.error_occurred.emit("No game to reset")
            return

        env_id = self._adapter._env_id
        self.start_game(env_id=env_id, seed=seed)

    def submit_action(self, action: str | int) -> bool:
        """Submit an action from the human player.

        Args:
            action: Action to execute (string name like "UP" or integer index)

        Returns:
            True if action was valid and executed
        """
        if not self._game_active or self._adapter is None:
            self.error_occurred.emit("No active game")
            return False

        # Convert action index to string if needed
        if isinstance(action, int):
            if 0 <= action < len(BABAISAI_ACTIONS):
                action_str = BABAISAI_ACTIONS[action]
            else:
                self.error_occurred.emit(f"Invalid action index: {action}")
                return False
        else:
            action_str = str(action).upper()

        # Execute action
        try:
            step_result = self._adapter.step(action_str)
        except Exception as e:
            self.error_occurred.emit(f"Action failed: {e}")
            return False

        self._cumulative_reward += step_result.reward
        observation = step_result.observation
        rules = observation.get("rules", [])

        # Check for rule changes
        if rules != self._previous_rules:
            self.rules_changed.emit(rules)
            self._previous_rules = rules.copy()

        # Update state
        self._current_state = BabaIsAIState(
            observation=observation,
            active_rules=rules,
            step_count=self._current_state.step_count + 1 if self._current_state else 1,
            is_game_over=step_result.terminated or step_result.truncated,
            reward=self._cumulative_reward,
        )

        self.action_executed.emit(action_str)
        self.state_changed.emit(self._current_state)

        # Check for game over
        if step_result.terminated or step_result.truncated:
            won = step_result.reward > 0
            self._handle_game_over(won)

        return True

    def get_state(self) -> Optional[BabaIsAIState]:
        """Get the current game state."""
        return self._current_state

    def get_legal_actions(self) -> List[str]:
        """Get available actions.

        Returns:
            List of action names
        """
        return BABAISAI_ACTIONS.copy()

    def get_current_rules(self) -> List[str]:
        """Get currently active rules.

        Returns:
            List of rule strings
        """
        if self._current_state is None:
            return []
        return self._current_state.active_rules.copy()

    def get_text_observation(self) -> str:
        """Get current text observation/prompt.

        Returns:
            Text description of game state
        """
        if self._current_state is None or self._current_state.observation is None:
            return ""
        return self._current_state.observation.get("prompt", "")

    def is_game_active(self) -> bool:
        """Check if a game is currently active."""
        return self._game_active

    def close(self) -> None:
        """Clean up resources."""
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        self._game_active = False
        self._current_state = None

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _handle_game_over(self, won: bool) -> None:
        """Handle end of game.

        Args:
            won: Whether the player won (solved puzzle)
        """
        self._game_active = False

        if won:
            result_msg = "Puzzle Solved! You win!"
        else:
            result_msg = "Game Over: Puzzle failed or time limit reached."

        _LOG.info(result_msg)
        self.status_message.emit(result_msg)
        self.game_over.emit(won)


__all__ = ["BabaIsAIGameController", "BabaIsAIState"]
