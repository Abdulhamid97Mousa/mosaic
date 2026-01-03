"""Presenter for Human Worker.

This presenter handles:
1. Building configuration for human player
2. No special tabs needed (board click happens in render view)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class HumanWorkerPresenter:
    """Presenter for Human Worker.

    Human workers don't need complex configuration - just player name
    and optional settings for move confirmation.
    """

    @property
    def id(self) -> str:
        """Return unique identifier for this presenter."""
        return "human_worker"

    def build_train_request(
        self,
        policy_path: Any,
        current_game: Optional[Any],
    ) -> dict:
        """Build a configuration for human player.

        Note: Human worker doesn't train - this just sets up the player.

        Args:
            policy_path: Not used for human player
            current_game: Currently selected game

        Returns:
            dict: Configuration dictionary
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"human-{timestamp}"

        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "human_worker.cli"],
            "metadata": {
                "ui": {
                    "worker_id": self.id,
                    "mode": "human_input",
                },
                "worker": {
                    "module": "human_worker.cli",
                    "config": {
                        "player_name": "Human",
                    },
                },
            },
        }

        _LOGGER.info("Built Human player config: run=%s", run_name)
        return config

    def build_human_config(
        self,
        player_name: str = "Human",
        show_legal_moves: bool = True,
        confirm_moves: bool = False,
        timeout_seconds: float = 0.0,
        **kwargs: Any,
    ) -> dict:
        """Build configuration for the human worker.

        Args:
            player_name: Display name for the human player
            show_legal_moves: Whether to highlight legal moves
            confirm_moves: Whether to require move confirmation
            timeout_seconds: Timeout for input (0 = no timeout)
            **kwargs: Additional options

        Returns:
            dict: Human worker configuration
        """
        return {
            "player_name": player_name,
            "show_legal_moves": show_legal_moves,
            "confirm_moves": confirm_moves,
            "timeout_seconds": timeout_seconds,
            **kwargs,
        }

    def create_tabs(
        self,
        run_id: str,
        agent_id: str,
        first_payload: dict,
        parent: Any,
    ) -> List[Any]:
        """Create worker-specific UI tabs.

        Human workers don't need special tabs - interaction happens
        directly on the board in the render view.

        Returns:
            list: Empty list (no special tabs)
        """
        return []


__all__ = ["HumanWorkerPresenter"]
