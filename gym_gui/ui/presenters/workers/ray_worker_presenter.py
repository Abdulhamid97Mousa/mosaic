"""Presenter for Ray RLlib multi-agent training worker."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class RayWorkerPresenter:
    """Presenter for Ray RLlib multi-agent training worker.

    Handles:
    1. Building training configurations for Ray RLlib runs
    2. Creating FastLane visualization tabs for live training
    3. Loading trained policies for evaluation
    """

    @property
    def id(self) -> str:
        return "ray_worker"

    def build_train_request(
        self,
        policy_path: Any,
        current_game: Optional[Any],
        *,
        env_id: Optional[str] = None,
        env_family: Optional[str] = None,
        paradigm: str = "parameter_sharing",
        total_timesteps: int = 100_000,
        train_batch_size: int = 4000,
        num_workers: int = 0,
        num_gpus: float = 0.0,
    ) -> dict:
        """Build a Ray RLlib training request.

        Args:
            policy_path: Path to load policy from (for evaluation) or None (for training).
            current_game: Currently selected game (used if env_id not specified).
            env_id: PettingZoo environment ID (e.g., "waterworld_v4").
            env_family: Environment family (e.g., "sisl", "classic").
            paradigm: Training paradigm (parameter_sharing, independent, self_play, shared_value).
            total_timesteps: Total training timesteps.
            train_batch_size: Batch size for training.
            num_workers: Number of Ray rollout workers.
            num_gpus: Number of GPUs to use.

        Returns:
            Configuration dict suitable for ray_worker CLI.
        """
        # Generate run ID
        run_id = f"ray_{env_id or 'env'}_{uuid.uuid4().hex[:8]}"

        # Infer environment info from current_game if not provided
        if env_id is None and current_game is not None:
            env_id = str(current_game)

        if env_family is None:
            # Try to infer family from env_id
            env_family = self._infer_family(env_id) if env_id else "sisl"

        config = {
            "run_id": run_id,
            "environment": {
                "family": env_family,
                "env_id": env_id or "waterworld_v4",
                "api_type": "aec",  # Ray RLlib uses AEC API
            },
            "paradigm": paradigm,
            "training": {
                "algorithm": "PPO",
                "total_timesteps": total_timesteps,
                "train_batch_size": train_batch_size,
            },
            "resources": {
                "num_workers": num_workers,
                "num_gpus": num_gpus,
            },
            "checkpoint": {
                "checkpoint_freq": 10,
                "checkpoint_at_end": True,
            },
        }

        # Add policy path for evaluation mode
        if policy_path is not None:
            config["policy_path"] = str(policy_path)
            config["mode"] = "evaluate"
        else:
            config["mode"] = "train"

        return config

    def build_train_config(
        self,
        env_id: str,
        env_family: str,
        paradigm: str = "parameter_sharing",
        **kwargs,
    ) -> dict:
        """Build a training configuration for the Ray worker.

        Args:
            env_id: PettingZoo environment ID.
            env_family: Environment family.
            paradigm: Training paradigm.
            **kwargs: Additional training parameters.

        Returns:
            Configuration dict for ray_worker.
        """
        return self.build_train_request(
            policy_path=None,
            current_game=None,
            env_id=env_id,
            env_family=env_family,
            paradigm=paradigm,
            **kwargs,
        )

    def create_tabs(
        self,
        run_id: str,
        agent_id: str,
        first_payload: dict,
        parent: Any,
    ) -> List[Any]:
        """Create worker-specific UI tabs for a running Ray training.

        For Ray RLlib, this creates:
        1. FastLane tab for live visualization (composite agent view)
        2. Metrics tab for training progress

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
        env_name = first_payload.get("env_name", "Multi-Agent")

        try:
            from gym_gui.ui.widgets.fastlane_tab import FastLaneTab

            # Create FastLane tab with Ray-specific naming
            # Tab name format: Ray-Live-{Env}-{run_id}
            tab_name = f"Ray-Live-{env_name}-{run_id[:8]}"
            fastlane_tab = FastLaneTab(
                run_id=run_id,
                tab_name=tab_name,
                parent=parent,
            )
            tabs.append(fastlane_tab)
            _LOGGER.info(f"Created FastLane tab: {tab_name}")

        except ImportError as e:
            _LOGGER.warning(f"FastLane tab not available: {e}")

        return tabs

    def load_policy(self, checkpoint_path: str, policy_id: str = "shared") -> Any:
        """Load a trained policy from a Ray RLlib checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            policy_id: Policy ID to load.

        Returns:
            RayPolicyActor instance ready for inference.
        """
        try:
            from ray_worker.policy_actor import RayPolicyActor

            actor = RayPolicyActor.from_checkpoint(
                checkpoint_path,
                policy_id=policy_id,
                actor_id=f"ray_policy_{policy_id}",
                deterministic=True,
            )
            _LOGGER.info(f"Loaded Ray policy from: {checkpoint_path}")
            return actor

        except Exception as e:
            _LOGGER.error(f"Failed to load Ray policy: {e}")
            raise

    def _infer_family(self, env_id: str) -> str:
        """Infer environment family from env_id.

        Args:
            env_id: Environment identifier.

        Returns:
            Inferred family name.
        """
        # SISL environments
        if env_id in ("waterworld_v4", "multiwalker_v9", "pursuit_v4"):
            return "sisl"

        # Classic board games
        if env_id in ("chess_v6", "go_v5", "connect_four_v3", "tictactoe_v3"):
            return "classic"

        # Butterfly environments
        if env_id in ("knights_archers_zombies_v10", "cooperative_pong_v5", "pistonball_v6"):
            return "butterfly"

        # MPE environments
        if env_id in ("simple_spread_v3", "simple_adversary_v3", "simple_tag_v3"):
            return "mpe"

        # Default to sisl
        return "sisl"

    def get_available_paradigms(self) -> List[Dict[str, str]]:
        """Get list of available training paradigms.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        return [
            {"id": "parameter_sharing", "name": "Parameter Sharing"},
            {"id": "independent", "name": "Independent Learning"},
            {"id": "self_play", "name": "Self-Play"},
            {"id": "shared_value", "name": "Shared Value Function (CTDE)"},
        ]


__all__ = ["RayWorkerPresenter"]
