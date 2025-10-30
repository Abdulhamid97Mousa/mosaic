"""SPADE-BDI worker presenter for UI orchestration.

This module encapsulates SPADE-BDI specific logic for:
1. Building training configurations from user form data
2. Creating SPADE-BDI specific UI tabs for live/replay visualization
3. Mapping metadata to DTOs for API contracts

The presenter abstracts worker-specific details from main_window.py,
allowing the UI to remain agnostic about orchestration logic.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

from gym_gui.ui.widgets.spade_bdi_worker_tabs import TabFactory


class SpadeBdiWorkerPresenter:
    """Presenter for SPADE-BDI worker orchestration.

    Responsible for:
    - Composing worker configuration and metadata payloads
    - Creating SPADE-BDI specific UI tabs (online, replay, grid, raw, video)
    - Supporting metadata extraction for DTO/API contracts

    This presenter is stateless and can be instantiated once at application startup.
    """

    @property
    def id(self) -> str:
        """Unique identifier for this worker."""
        return "spade_bdi_rl"

    def build_train_request(self, policy_path: Path, current_game: Optional[object]) -> dict:
        """Build a training request from policy path and game selection.

        Composes the full configuration dictionary including:
        - Worker configuration (run_id, game_id, seed, episodes, etc.)
        - Metadata payload (ui, worker, environment, resources, artifacts)
        - Environment variables for the worker process

        Args:
            policy_path: Path object pointing to the policy file
            current_game: GameId enum value or None (from control panel selection)

        Returns:
            dict: Configuration dictionary suitable for submission to TrainerClient

        Raises:
            FileNotFoundError: If policy file does not exist
            ValueError: If game_id cannot be determined
            json.JSONDecodeError: If policy file metadata is invalid JSON
        """
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        # Extract metadata from policy file
        metadata = self._extract_policy_metadata(policy_path)

        # Determine game_id: prefer metadata, then current_game, else error
        game_id = metadata.get("game_id")
        if not game_id and current_game is not None:
            # current_game is a GameId enum, extract its value
            game_id = getattr(current_game, "value", None)
        if game_id is None:
            raise ValueError(
                "Game environment could not be determined from policy metadata or current selection. "
                "Either embed metadata.game_id in the policy file or select a game before loading."
            )

        # Extract configuration parameters
        agent_id = metadata.get("agent_id") or policy_path.stem
        seed = int(metadata.get("seed", 0))
        max_episodes = int(metadata.get("eval_episodes", 5))
        max_steps = int(metadata.get("max_steps_per_episode", metadata.get("max_steps", 200)))
        algorithm_label = metadata.get("algorithm", "Loaded Policy")

        # Generate run name with timestamp
        run_name = f"eval-{agent_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # Build worker configuration (passed to worker process)
        worker_config = {
            "run_id": run_name,
            "game_id": game_id,
            "seed": seed,
            "max_episodes": max_episodes,
            "max_steps_per_episode": max_steps,
            "policy_strategy": "eval",
            "policy_path": str(policy_path),
            "agent_id": agent_id,
            "capture_video": False,
            "headless": True,
            "extra": {
                "policy_path": str(policy_path),
                "algorithm": algorithm_label,
            },
        }

        # Build metadata payload (for telemetry and UI context)
        metadata_payload = {
            "ui": {
                "algorithm": algorithm_label,
                "source_policy": str(policy_path),
                "eval": {
                    "max_episodes": max_episodes,
                    "max_steps_per_episode": max_steps,
                    "seed": seed,
                },
            },
            "worker": {
                "module": "spade_bdi_rl.worker",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "agent_id": agent_id,
                "config": worker_config,
            },
        }

        # Set environment variables for worker process
        environment = {
            "GYM_ENV_ID": game_id,
            "TRAIN_SEED": str(seed),
            "EVAL_POLICY_PATH": str(policy_path),
        }

        # Compose complete training request
        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "spade_bdi_rl.worker"],
            "environment": environment,
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
            "metadata": metadata_payload,
        }

        return config

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent) -> list:
        """Create worker-specific UI tabs for a running agent.

        Creates a standard set of SPADE-BDI tabs:
        - AgentOnlineTab: Primary real-time statistics view
        - AgentReplayTab: Historical episode browser
        - AgentOnlineGridTab: Live grid/environment visualization (ToyText only)
        - AgentOnlineRawTab: Debug stream view (hidden by default)
        - AgentOnlineVideoTab: Live RGB frames (visual environments only)

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier within the run
            first_payload: First telemetry step payload containing metadata
            parent: Parent Qt widget

        Returns:
            list: List of QWidget tab instances (without registration)

        Note:
            Tabs are not registered with _render_tabs here; main_window.py
            is responsible for registration and metadata binding.
        """
        factory = TabFactory()
        tabs = factory.create_tabs(run_id, agent_id, first_payload, parent)
        return tabs

    def extract_metadata(self, config: dict) -> dict:
        """Extract and normalize metadata from training config for DTO/API use.

        Transforms the nested config structure into a flat DTO-friendly format.
        Useful for serialization, API contracts, and future multi-agent support.

        Args:
            config: Training configuration dictionary

        Returns:
            dict: Normalized metadata with 'agent_id', 'game_id', 'run_id', etc.
        """
        metadata = config.get("metadata", {})
        worker_meta = metadata.get("worker", {})
        worker_cfg = worker_meta.get("config", {})

        return {
            "agent_id": worker_meta.get("agent_id") or worker_cfg.get("agent_id"),
            "game_id": worker_cfg.get("game_id"),
            "run_id": worker_cfg.get("run_id") or config.get("run_name"),
            "algorithm": metadata.get("ui", {}).get("algorithm"),
            "source_policy": metadata.get("ui", {}).get("source_policy"),
            "worker_module": worker_meta.get("module"),
        }

    @staticmethod
    def _extract_policy_metadata(policy_path: Path) -> dict:
        """Load and parse metadata from a policy file.

        Attempts to parse the policy file as JSON and extract metadata.
        Falls back to an empty dict if the file is not valid JSON.

        Args:
            policy_path: Path to policy file

        Returns:
            dict: Metadata dictionary from file, or empty dict if not found/invalid

        Note:
            This is a simple fallback; real policy files should include
            metadata in their JSON structure (e.g., under a "metadata" key).
        """
        try:
            with open(policy_path, "r") as f:
                payload = json.load(f)
                if isinstance(payload, dict):
                    return payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def extract_agent_id(config: dict) -> Optional[str]:
        """Extract agent_id from a training configuration.

        Utility method to retrieve agent_id from nested config structure.
        Used in main_window.py for tab identification.

        Args:
            config: Training configuration dictionary

        Returns:
            str: Agent identifier, or None if not found
        """
        try:
            metadata = config.get("metadata", {})
            worker_meta = metadata.get("worker", {})
            agent_id = worker_meta.get("agent_id")
            if not agent_id:
                worker_cfg = worker_meta.get("config", {})
                agent_id = worker_cfg.get("agent_id")
            if agent_id:
                return str(agent_id)
        except Exception:  # pragma: no cover - defensive
            pass
        return None
