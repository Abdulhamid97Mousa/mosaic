"""Presenter for PettingZoo multi-agent worker.

This presenter handles:
1. Building training configurations for PettingZoo environments
2. Creating worker-specific UI tabs for multi-agent visualization
3. Policy evaluation for trained multi-agent policies
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from gym_gui.core.pettingzoo_enums import (
    PETTINGZOO_ENV_METADATA,
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    is_aec_env,
)

_LOGGER = logging.getLogger(__name__)


class PettingZooWorkerPresenter:
    """Presenter for PettingZoo multi-agent training and evaluation.

    Handles configuration building for multi-agent environments including:
    - AEC (turn-based) environments like Chess, Go, Tic-Tac-Toe
    - Parallel (simultaneous) environments like MPE, SISL, Butterfly

    The presenter extracts environment metadata, agent configurations,
    and builds training requests compatible with the trainer daemon.
    """

    @property
    def id(self) -> str:
        """Return unique identifier for this presenter."""
        return "pettingzoo_worker"

    def build_train_request(
        self,
        policy_path: Any,
        current_game: Optional[Any],
    ) -> dict:
        """Build a training request for policy evaluation.

        Args:
            policy_path: Path to the saved policy/model file
            current_game: Currently selected game (may be PettingZooEnvId or None)

        Returns:
            dict: Configuration dictionary for TrainerClient submission
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Determine environment info
        env_id = None
        family = PettingZooFamily.CLASSIC
        is_parallel = False

        if isinstance(current_game, PettingZooEnvId):
            env_id = current_game.value
            if current_game in PETTINGZOO_ENV_METADATA:
                family = PETTINGZOO_ENV_METADATA[current_game][0]
            is_parallel = not is_aec_env(current_game)
        elif current_game is not None:
            env_id = str(current_game)

        run_name = f"pettingzoo-eval-{timestamp}"

        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "pettingzoo_worker.cli", "evaluate"],
            "environment": {
                "PETTINGZOO_ENV_ID": env_id or "tictactoe_v3",
                "PETTINGZOO_FAMILY": family.value if isinstance(family, PettingZooFamily) else str(family),
                "PETTINGZOO_API_TYPE": "parallel" if is_parallel else "aec",
                "POLICY_PATH": str(policy_path) if policy_path else "",
                "RENDER_MODE": "rgb_array",
            },
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
            "metadata": {
                "ui": {
                    "worker_id": self.id,
                    "env_id": env_id,
                    "family": family.value if isinstance(family, PettingZooFamily) else str(family),
                    "is_parallel": is_parallel,
                    "mode": "evaluation",
                },
                "worker": {
                    "module": "pettingzoo_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "policy_path": str(policy_path) if policy_path else None,
                        "render": True,
                        "num_episodes": 10,
                    },
                },
            },
        }

        _LOGGER.info(
            "Built PettingZoo evaluation request: env=%s, policy=%s",
            env_id,
            policy_path,
        )

        return config

    def build_training_config(
        self,
        env_id: str,
        family: str,
        algorithm: str = "PPO",
        num_agents: int = 2,
        total_timesteps: int = 100000,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict:
        """Build a complete training configuration from form data.

        This method is called by the training form to construct the
        configuration dictionary for multi-agent training.

        Args:
            env_id: PettingZoo environment ID (e.g., "tictactoe_v3")
            family: Environment family (classic, mpe, sisl, butterfly, atari)
            algorithm: Training algorithm (PPO, DQN, etc.)
            num_agents: Number of agents in the environment
            total_timesteps: Total training timesteps
            seed: Random seed for reproducibility
            **kwargs: Additional configuration options

        Returns:
            dict: Complete training configuration
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"pettingzoo-{env_id.replace('_', '-')}-{timestamp}"

        # Determine API type from env_id
        is_parallel = False
        try:
            pz_env_id = PettingZooEnvId(env_id)
            is_parallel = not is_aec_env(pz_env_id)
        except ValueError:
            pass

        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "pettingzoo_worker.cli", "train"],
            "environment": {
                "PETTINGZOO_ENV_ID": env_id,
                "PETTINGZOO_FAMILY": family,
                "PETTINGZOO_API_TYPE": "parallel" if is_parallel else "aec",
                "ALGORITHM": algorithm,
                "NUM_AGENTS": str(num_agents),
                "TOTAL_TIMESTEPS": str(total_timesteps),
                "SEED": str(seed) if seed is not None else "",
                "RENDER_MODE": kwargs.get("render_mode", "rgb_array"),
            },
            "resources": {
                "cpus": kwargs.get("cpus", 4),
                "memory_mb": kwargs.get("memory_mb", 4096),
                "gpus": {
                    "requested": kwargs.get("gpus", 0),
                    "mandatory": kwargs.get("gpu_mandatory", False),
                },
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": kwargs.get("keep_checkpoints", True),
            },
            "metadata": {
                "ui": {
                    "worker_id": self.id,
                    "env_id": env_id,
                    "family": family,
                    "algorithm": algorithm,
                    "num_agents": num_agents,
                    "is_parallel": is_parallel,
                    "mode": "training",
                },
                "worker": {
                    "module": "pettingzoo_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "algorithm": algorithm,
                        "total_timesteps": total_timesteps,
                        "seed": seed,
                        "render": kwargs.get("render", True),
                        "save_model": kwargs.get("save_model", True),
                        "log_interval": kwargs.get("log_interval", 100),
                        "eval_episodes": kwargs.get("eval_episodes", 10),
                        **{k: v for k, v in kwargs.items() if k not in [
                            "render_mode", "cpus", "memory_mb", "gpus",
                            "gpu_mandatory", "keep_checkpoints", "render",
                            "save_model", "log_interval", "eval_episodes",
                        ]},
                    },
                },
                "artifacts": {
                    "tensorboard": {
                        "enabled": kwargs.get("tensorboard_enabled", True),
                        "log_dir": f"runs/{run_name}/tensorboard",
                    },
                    "wandb": {
                        "enabled": kwargs.get("wandb_enabled", False),
                        "project": kwargs.get("wandb_project", "pettingzoo-training"),
                        "entity": kwargs.get("wandb_entity", None),
                    },
                },
            },
        }

        _LOGGER.info(
            "Built PettingZoo training config: env=%s, algo=%s, timesteps=%d",
            env_id,
            algorithm,
            total_timesteps,
        )

        return config

    def create_tabs(
        self,
        run_id: str,
        agent_id: str,
        first_payload: dict,
        parent: Any,
    ) -> List[Any]:
        """Create worker-specific UI tabs for a running multi-agent session.

        This method is called when a training/evaluation run starts to
        create visualization tabs specific to multi-agent environments.

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier within the run
            first_payload: First telemetry payload containing metadata
            parent: Parent Qt widget

        Returns:
            list: List of QWidget tab instances (empty for now, can be extended)
        """
        # For now, return empty list - tabs can be added in future iterations
        # Potential tabs:
        # - Multi-agent performance comparison
        # - Agent coordination metrics
        # - Environment state visualization
        # - Turn-by-turn game replay (for AEC envs)

        _LOGGER.debug(
            "Creating PettingZoo tabs for run=%s, agent=%s",
            run_id,
            agent_id,
        )

        return []

    def extract_env_metadata(self, env_id: str) -> Dict[str, Any]:
        """Extract metadata for a PettingZoo environment.

        Args:
            env_id: Environment identifier string

        Returns:
            dict: Environment metadata including family, API type, description
        """
        try:
            pz_env_id = PettingZooEnvId(env_id)
            if pz_env_id in PETTINGZOO_ENV_METADATA:
                family, api_type, display_name, description = PETTINGZOO_ENV_METADATA[pz_env_id]
                return {
                    "env_id": env_id,
                    "family": family.value,
                    "api_type": api_type.value,
                    "display_name": display_name,
                    "description": description,
                    "is_aec": api_type.value == "aec",
                }
        except ValueError:
            pass

        return {
            "env_id": env_id,
            "family": "unknown",
            "api_type": "unknown",
            "display_name": env_id,
            "description": "",
            "is_aec": False,
        }


__all__ = ["PettingZooWorkerPresenter"]
