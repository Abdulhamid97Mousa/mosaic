"""Evaluation results storage.

Saves evaluation metrics to JSON and CSV files for later analysis.

Directory structure:
    var/trainer/runs/{run_id}/eval/{eval_id}/
        ├── config.json      # Evaluation configuration
        ├── summary.json     # Summary statistics
        ├── episodes.csv     # Per-episode metrics
        └── agents.csv       # Per-agent reward breakdown
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationResultsConfig:
    """Configuration for evaluation results storage.

    Attributes:
        training_run_id: ID of the training run being evaluated.
        checkpoint_path: Path to the checkpoint used.
        env_id: Environment ID.
        env_family: Environment family.
        num_episodes: Number of evaluation episodes.
        max_steps_per_episode: Max steps per episode.
        deterministic: Whether deterministic actions were used.
        seed: Random seed.
    """

    training_run_id: str
    checkpoint_path: str
    env_id: str
    env_family: str
    num_episodes: int
    max_steps_per_episode: int
    deterministic: bool
    seed: Optional[int]


class EvaluationResultsWriter:
    """Writes evaluation results to JSON and CSV files.

    Creates a timestamped directory under the training run's eval/ folder
    and saves results in multiple formats for analysis.
    """

    def __init__(
        self,
        training_run_id: str,
        base_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the results writer.

        Args:
            training_run_id: ID of the training run (e.g., "01KCGZ49GH0ASHJ069ZBWDAX2A").
            base_dir: Base directory for trainer runs. Defaults to var/trainer/runs.
        """
        self.training_run_id = training_run_id

        # Default to var/trainer/runs
        if base_dir is None:
            # Find project root by looking for var/trainer/runs
            base_dir = self._find_trainer_runs_dir()

        self.base_dir = base_dir
        self.eval_id = datetime.now().strftime("%y%m%d_%H%M%S")
        self.eval_dir = base_dir / training_run_id / "eval" / self.eval_id

        self._config: Optional[EvaluationResultsConfig] = None
        self._episodes: List[Dict[str, Any]] = []
        self._summary: Dict[str, Any] = {}

    def _find_trainer_runs_dir(self) -> Path:
        """Find the var/trainer/runs directory.

        Returns:
            Path to trainer runs directory.
        """
        # Try common locations
        candidates = [
            Path.cwd() / "var" / "trainer" / "runs",
            Path.home() / "Desktop" / "Projects" / "GUI_BDI_RL" / "var" / "trainer" / "runs",
            Path(__file__).parent.parent.parent.parent / "var" / "trainer" / "runs",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Default to current working directory
        default = Path.cwd() / "var" / "trainer" / "runs"
        default.mkdir(parents=True, exist_ok=True)
        return default

    def setup(self) -> None:
        """Create the evaluation directory."""
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info("Evaluation results directory: %s", self.eval_dir)

    def set_config(self, config: EvaluationResultsConfig) -> None:
        """Set the evaluation configuration.

        Args:
            config: Evaluation configuration.
        """
        self._config = config

    def add_episode(
        self,
        episode_id: int,
        total_reward: float,
        episode_length: int,
        agent_rewards: Dict[str, float],
        duration_seconds: float,
        terminated: bool,
    ) -> None:
        """Record metrics for a single episode.

        Args:
            episode_id: Episode index.
            total_reward: Total reward for the episode.
            episode_length: Number of steps in the episode.
            agent_rewards: Rewards per agent.
            duration_seconds: Episode duration.
            terminated: Whether episode terminated naturally.
        """
        self._episodes.append({
            "episode_id": episode_id,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "agent_rewards": agent_rewards,
            "duration_seconds": duration_seconds,
            "terminated": terminated,
        })

    def set_summary(self, summary: Dict[str, Any]) -> None:
        """Set the evaluation summary statistics.

        Args:
            summary: Summary dictionary with mean_reward, std_reward, etc.
        """
        self._summary = summary

    def save(self) -> Path:
        """Save all results to files.

        Returns:
            Path to the evaluation directory.
        """
        self.setup()

        # Save config
        self._save_config()

        # Save summary
        self._save_summary()

        # Save episodes CSV
        self._save_episodes_csv()

        # Save agents CSV
        self._save_agents_csv()

        _LOGGER.info(
            "Evaluation results saved to %s: %d episodes",
            self.eval_dir,
            len(self._episodes),
        )

        return self.eval_dir

    def _save_config(self) -> None:
        """Save evaluation configuration to JSON."""
        config_path = self.eval_dir / "config.json"

        config_data = {
            "eval_id": self.eval_id,
            "training_run_id": self.training_run_id,
            "timestamp": datetime.now().isoformat(),
        }

        if self._config:
            config_data.update(asdict(self._config))

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

    def _save_summary(self) -> None:
        """Save summary statistics to JSON."""
        summary_path = self.eval_dir / "summary.json"

        summary_data = {
            "eval_id": self.eval_id,
            "training_run_id": self.training_run_id,
            "timestamp": datetime.now().isoformat(),
            **self._summary,
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

    def _save_episodes_csv(self) -> None:
        """Save per-episode metrics to CSV."""
        if not self._episodes:
            return

        episodes_path = self.eval_dir / "episodes.csv"

        # Flatten agent_rewards into separate columns
        fieldnames = [
            "episode_id",
            "total_reward",
            "episode_length",
            "duration_seconds",
            "terminated",
        ]

        # Get all agent IDs from first episode
        agent_ids: list[str] = []
        if self._episodes and "agent_rewards" in self._episodes[0]:
            agent_ids = sorted(self._episodes[0]["agent_rewards"].keys())
            fieldnames.extend([f"reward_{agent_id}" for agent_id in agent_ids])

        with open(episodes_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ep in self._episodes:
                row = {
                    "episode_id": ep["episode_id"],
                    "total_reward": ep["total_reward"],
                    "episode_length": ep["episode_length"],
                    "duration_seconds": ep["duration_seconds"],
                    "terminated": ep["terminated"],
                }
                # Add per-agent rewards
                agent_rewards = ep.get("agent_rewards", {})
                for agent_id in agent_ids:
                    row[f"reward_{agent_id}"] = agent_rewards.get(agent_id, 0.0)

                writer.writerow(row)

    def _save_agents_csv(self) -> None:
        """Save per-agent aggregated metrics to CSV."""
        if not self._episodes:
            return

        agents_path = self.eval_dir / "agents.csv"

        # Aggregate rewards per agent across all episodes
        agent_totals: Dict[str, List[float]] = {}

        for ep in self._episodes:
            agent_rewards = ep.get("agent_rewards", {})
            for agent_id, reward in agent_rewards.items():
                if agent_id not in agent_totals:
                    agent_totals[agent_id] = []
                agent_totals[agent_id].append(reward)

        if not agent_totals:
            return

        # Calculate statistics per agent
        import numpy as np

        with open(agents_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "agent_id",
                    "num_episodes",
                    "mean_reward",
                    "std_reward",
                    "min_reward",
                    "max_reward",
                    "total_reward",
                ],
            )
            writer.writeheader()

            for agent_id, rewards in sorted(agent_totals.items()):
                writer.writerow({
                    "agent_id": agent_id,
                    "num_episodes": len(rewards),
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "min_reward": np.min(rewards),
                    "max_reward": np.max(rewards),
                    "total_reward": np.sum(rewards),
                })


def extract_run_id_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Extract the training run ID from a checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint (e.g., var/trainer/runs/01KC.../checkpoints/...)

    Returns:
        Run ID if found, None otherwise.
    """
    path = Path(checkpoint_path)

    # Walk up the path looking for the runs directory structure
    for parent in path.parents:
        if parent.parent.name == "runs" and parent.parent.parent.name == "trainer":
            return parent.name

    # Try to extract ULID-like pattern from path
    import re
    ulid_pattern = r"[0-9A-HJKMNP-TV-Z]{26}"

    for part in path.parts:
        if re.match(ulid_pattern, part):
            return part

    return None


__all__ = [
    "EvaluationResultsConfig",
    "EvaluationResultsWriter",
    "extract_run_id_from_checkpoint",
]
