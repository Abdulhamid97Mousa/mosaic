"""Presenter for MCTX GPU-accelerated MCTS training worker."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class MctxWorkerPresenter:
    """Presenter for MCTX GPU-accelerated MCTS training worker.

    Handles:
    1. Building training configurations for mctx (AlphaZero/MuZero) runs
    2. Creating visualization tabs for training progress
    3. Loading trained policies for evaluation
    """

    @property
    def id(self) -> str:
        return "mctx_worker"

    def build_train_request(
        self,
        policy_path: Any,
        current_game: Optional[Any],
        *,
        env_id: Optional[str] = None,
        algorithm: str = "gumbel_muzero",
        max_steps: int = 100_000,
        num_simulations: int = 800,
        batch_size: int = 256,
        learning_rate: float = 2e-4,
        device: str = "gpu",
    ) -> dict:
        """Build an MCTX training request.

        Args:
            policy_path: Path to load policy from (for evaluation) or None (for training).
            current_game: Currently selected game (used if env_id not specified).
            env_id: Pgx environment ID (e.g., "chess", "go_9x9").
            algorithm: MCTS algorithm (alphazero, muzero, gumbel_muzero).
            max_steps: Maximum training steps.
            num_simulations: MCTS simulations per move.
            batch_size: Training batch size.
            learning_rate: Learning rate for optimizer.
            device: Device to use (gpu, tpu, cpu).

        Returns:
            Configuration dict suitable for mctx_worker CLI.
        """
        # Generate run ID
        run_id = f"mctx_{env_id or 'game'}_{uuid.uuid4().hex[:8]}"

        # Infer environment from current_game if not provided
        if env_id is None and current_game is not None:
            env_id = self._game_to_pgx_env(current_game)

        config = {
            "run_id": run_id,
            "env_id": env_id or "chess",
            "algorithm": algorithm,
            "seed": 42,
            "max_steps": max_steps,
            "device": device,
            "mcts": {
                "num_simulations": num_simulations,
                "dirichlet_alpha": 0.3,
                "dirichlet_fraction": 0.25,
                "temperature": 1.0,
            },
            "network": {
                "num_res_blocks": 8,
                "channels": 128,
            },
            "training": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "replay_buffer_size": 100_000,
            },
        }

        # Add policy path for evaluation mode
        if policy_path is not None:
            config["checkpoint_path"] = str(policy_path)
            config["mode"] = "eval"
        else:
            config["mode"] = "train"

        return config

    def build_train_config(
        self,
        env_id: str,
        algorithm: str = "gumbel_muzero",
        **kwargs,
    ) -> dict:
        """Build a training configuration for the MCTX worker.

        Args:
            env_id: Pgx environment ID.
            algorithm: MCTS algorithm to use.
            **kwargs: Additional training parameters.

        Returns:
            Configuration dict for mctx_worker.
        """
        return self.build_train_request(
            policy_path=None,
            current_game=None,
            env_id=env_id,
            algorithm=algorithm,
            **kwargs,
        )

    def create_tabs(
        self,
        run_id: str,
        agent_id: str,
        first_payload: dict,
        parent: Any,
    ) -> List[Any]:
        """Create worker-specific UI tabs for a running MCTX training.

        For MCTX, this creates:
        1. Board game visualization tab for interactive display
        2. Metrics tab for training progress (loss curves, etc.)

        Args:
            run_id: Unique run identifier.
            agent_id: Agent identifier.
            first_payload: First telemetry payload containing metadata.
            parent: Parent Qt widget.

        Returns:
            List of QWidget tab instances.
        """
        tabs = []

        # Extract environment info from payload
        env_id = first_payload.get("env_id", "chess")

        try:
            # MCTX is primarily for board games, use BoardGameFastLaneTab
            from gym_gui.ui.widgets.board_game_fastlane_tab import BoardGameFastLaneTab
            from gym_gui.core.enums import GameId

            # Map Pgx env_id to GameId
            game_id = self._pgx_to_game_id(env_id)

            if game_id is not None:
                tab_name = f"MCTX-{env_id}-{run_id[:8]}"
                board_tab = BoardGameFastLaneTab(
                    run_id=run_id,
                    agent_id=agent_id,
                    game_id=game_id,
                    mode_label=tab_name,
                    parent=parent,
                )
                tabs.append(board_tab)
                _LOGGER.info(f"Created BoardGameFastLane tab for MCTX: {tab_name}")

        except ImportError as e:
            _LOGGER.warning(f"Board game tab not available: {e}")

        return tabs

    def _game_to_pgx_env(self, game: Any) -> str:
        """Map GameId enum to Pgx environment ID.

        Args:
            game: GameId enum value.

        Returns:
            Pgx environment identifier.
        """
        from gym_gui.core.enums import GameId

        mapping = {
            GameId.CHESS: "chess",
            GameId.GO: "go_19x19",
            GameId.CONNECT_FOUR: "connect_four",
            GameId.TIC_TAC_TOE: "tic_tac_toe",
        }
        return mapping.get(game, "chess")

    def _pgx_to_game_id(self, env_id: str) -> Any:
        """Map Pgx environment ID to GameId enum.

        Args:
            env_id: Pgx environment identifier.

        Returns:
            GameId enum value or None.
        """
        from gym_gui.core.enums import GameId

        mapping = {
            "chess": GameId.CHESS,
            "go_9x9": GameId.GO,
            "go_19x19": GameId.GO,
            "connect_four": GameId.CONNECT_FOUR,
            "tic_tac_toe": GameId.TIC_TAC_TOE,
        }
        return mapping.get(env_id)

    def load_policy(self, checkpoint_path: str) -> Any:
        """Load a trained policy from an MCTX checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            Policy actor instance ready for inference.
        """
        try:
            # Import and load MCTX policy
            import pickle
            from pathlib import Path

            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            with open(ckpt_path, "rb") as f:
                checkpoint = pickle.load(f)

            _LOGGER.info(f"Loaded MCTX policy from: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            _LOGGER.error(f"Failed to load MCTX policy: {e}")
            raise

    def get_available_algorithms(self) -> List[Dict[str, str]]:
        """Get list of available MCTS algorithms.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        return [
            {"id": "alphazero", "name": "AlphaZero"},
            {"id": "muzero", "name": "MuZero"},
            {"id": "gumbel_muzero", "name": "Gumbel MuZero"},
            {"id": "stochastic_muzero", "name": "Stochastic MuZero"},
        ]

    def get_supported_games(self) -> List[Dict[str, str]]:
        """Get list of supported Pgx games.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        return [
            {"id": "chess", "name": "Chess"},
            {"id": "go_9x9", "name": "Go (9x9)"},
            {"id": "go_19x19", "name": "Go (19x19)"},
            {"id": "shogi", "name": "Shogi"},
            {"id": "connect_four", "name": "Connect Four"},
            {"id": "tic_tac_toe", "name": "Tic-Tac-Toe"},
            {"id": "othello", "name": "Othello"},
            {"id": "hex", "name": "Hex"},
            {"id": "backgammon", "name": "Backgammon"},
        ]


__all__ = ["MctxWorkerPresenter"]
