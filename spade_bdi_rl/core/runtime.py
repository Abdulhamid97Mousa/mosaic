"""Headless training loop that powers the refactored worker process."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Union
import time

if TYPE_CHECKING:
    from ..adapters import AdapterType


from ..algorithms import create_agent, create_runtime
from ..policies import PolicyStorage
from .config import PolicyStrategy, RunConfig
from .worker_telemetry import TelemetryEmitter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool


class HeadlessTrainer:
    """Drive the learning/evaluation loop and emit JSONL telemetry."""

    def __init__(
        self,
        adapter: AdapterType,
        config: RunConfig,
        emitter: TelemetryEmitter,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.emitter = emitter

        # Ensure canonical policy path exists before any IO
        policy_path = self.config.ensure_policy_path()
        self.policy_store = PolicyStorage(policy_path)

        self.agent = create_agent(adapter)
        self.runtime = create_runtime(adapter, self.agent)

        # Strategy flags
        self._is_training = config.policy_strategy in (
            PolicyStrategy.TRAIN,
            PolicyStrategy.TRAIN_AND_SAVE,
            PolicyStrategy.LOAD,
        )
        self._should_save = config.policy_strategy in (
            PolicyStrategy.TRAIN,
            PolicyStrategy.TRAIN_AND_SAVE,
        )
        self._require_existing = config.policy_strategy in (
            PolicyStrategy.LOAD,
            PolicyStrategy.EVAL,
        )

        self._maybe_load_policy()
        if config.policy_strategy is PolicyStrategy.EVAL:
            self.agent.epsilon = 0.0
        
        logger.info(f"HEADLESS_TRAINER_INIT | type={self.__class__.__name__} | adapter={adapter.__class__.__name__} | env={config.game_id} | policy_strategy={config.policy_strategy.value} | max_episodes={config.max_episodes} | seed={config.seed} | policy_path={policy_path}")

    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute training/evaluation and emit JSONL telemetry to stdout."""
        config_payload = _config_payload(self.config)
        self.emitter.run_started(self.config.run_id, config_payload)

        try:
            summaries: list[EpisodeMetrics] = []
            for episode_index in range(self.config.max_episodes):
                # CRITICAL: Separation of concerns for reproducibility and environment variation
                #
                # seed (config.seed): Base seed for reproducible experiment (e.g., seed=1)
                #   - Constant throughout the run
                #   - Used for reproducibility across runs
                #   - Does NOT change per episode
                #
                # episode_number: Display value for user (seed + episode_index)
                #   - For seed=1: episodes 1, 2, 3, 4, ...
                #   - For seed=39: episodes 39, 40, 41, 42, ...
                #   - Used in telemetry and UI display
                #
                # episode_seed: Unique seed for environment variation per episode
                #   - Derived from episode_index (0, 1, 2, 3, ...)
                #   - Each episode gets different environment starting state
                #   - Allows agent to learn from diverse experiences
                #   - Still reproducible (deterministically derived from episode_index)
                #
                episode_number = self.config.seed + episode_index
                episode_seed = episode_index  # Unique seed per episode (0, 1, 2, 3, ...)

                summary = self._run_episode(episode_index, episode_number, episode_seed)
                summaries.append(summary)
                episode_metadata = {
                    "control_mode": "agent_only",
                    "run_id": self.config.run_id,
                    "agent_id": self.config.agent_id,
                    "game_id": self.config.game_id,
                    "seed": self.config.seed,  # Base seed (constant, for reproducibility)
                    "episode_seed": episode_seed,  # Unique seed per episode (for variation)
                    "episode_index": episode_index,
                    "episode_number": episode_number,  # Display value (seed + episode_index)
                    "policy_strategy": self.config.policy_strategy.value,
                    "success": summary.success,
                }
                self.emitter.episode(
                    self.config.run_id,
                    episode_number,  # Pass episode_number (display value = seed + episode_index)
                    agent_id=self.config.agent_id,
                    reward=summary.total_reward,
                    steps=summary.steps,
                    success=summary.success,
                    metadata=episode_metadata,
                )

            if self._should_save:
                metadata = {
                    "run_id": self.config.run_id,
                    "game_id": self.config.game_id,
                    "agent_id": self.config.agent_id,
                    "episodes": self.config.max_episodes,
                    "strategy": self.config.policy_strategy.value,
                }
                path = self.policy_store.save(self.agent.q_table, metadata)
                self.emitter.artifact(self.config.run_id, "policy", str(path))

            self.emitter.run_completed(self.config.run_id, status="completed")
            return 0
        except Exception as exc:  # noqa: BLE001
            self.emitter.run_completed(
                self.config.run_id,
                status="failed",
                error=str(exc),
            )
            return 1

    # ------------------------------------------------------------------
    def _run_episode(self, episode_index: int, episode_number: int, episode_seed: int) -> EpisodeMetrics:
        """Run single episode with JSONL telemetry.

        Args:
            episode_index: 0-based loop counter (0, 1, 2, 3, ...)
            episode_number: Display value for telemetry (seed + episode_index)
            episode_seed: Unique seed for environment variation (derived from episode_index)
        """
        # CRITICAL: Use episode_seed to reset the environment
        # Each episode gets a unique seed (0, 1, 2, 3, ...) for environment variation
        # This allows the agent to learn from diverse starting states
        state, obs = self.adapter.reset(seed=episode_seed)
        total_reward = 0.0
        success = False
        steps_taken = 0

        max_steps = self.config.max_steps_per_episode
        for step_index in range(max_steps):
            action = self.runtime.get_action(state, training=self._is_training)
            q_before = float(self.agent.q_table[state, action])

            next_state, reward, terminated, truncated, next_obs = self.adapter.step(action)
            if not terminated and not truncated and (step_index + 1) >= max_steps:
                truncated = True
            done = terminated or truncated

            if self._is_training:
                self.runtime.update_q_online(state, action, reward, next_state, done)

            # Generate render payload for grid visualization from the NEW observation
            # CRITICAL: Use next_obs (after step), not obs (before step)
            render_payload = self._generate_render_payload(next_state, next_obs)

            self.emitter.step(
                self.config.run_id,
                episode_number,  # Pass episode_number (display value = seed + episode_index)
                step_index,
                agent_id=self.config.agent_id,
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                state=int(state),
                next_state=int(next_state),
                observation=obs,
                next_observation=next_obs,
                q_before=q_before,
                q_after=float(self.agent.q_table[state, action]),
                epsilon=float(self.agent.epsilon),
                render_payload=render_payload,
                episode_seed=int(episode_seed),  # NEW FIELD: Unique seed per episode for environment variation
            )

            # Apply step delay for real-time observation (if configured)
            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)

            total_reward += float(reward)
            steps_taken = step_index + 1
            state, obs = next_state, next_obs

            if terminated and reward > 0:
                success = True
            if done:
                break

        return EpisodeMetrics(
            episode=episode_index,
            total_reward=total_reward,
            steps=steps_taken,
            success=success,
        )

    def _generate_render_payload(self, state: int, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate render payload for grid visualization from observation.

        Args:
            state: Current state (flat index)
            obs: Observation dict from adapter

        Returns:
            Render payload dict with grid, agent position, and game metadata
        """
        if not obs:
            return {}

        # Extract grid information from observation
        grid_size = obs.get("grid_size")
        if isinstance(grid_size, dict):
            nrow = grid_size.get("height", 4)
            ncol = grid_size.get("width", 4)
        else:
            nrow = ncol = grid_size if isinstance(grid_size, int) else 4

        # Get agent position
        position = obs.get("position", {})
        agent_row = position.get("row", 0)
        agent_col = position.get("col", 0)

        # Get goal position
        goal = obs.get("goal", {})
        goal_row = goal.get("row", nrow - 1)
        goal_col = goal.get("col", ncol - 1)

        # Get holes
        holes_list = obs.get("holes", [])
        holes = set()
        for h in holes_list:
            if isinstance(h, dict):
                holes.add((h.get("row", 0), h.get("col", 0)))

        # Create grid representation
        grid = [['F' for _ in range(ncol)] for _ in range(nrow)]

        # Mark holes
        for h_row, h_col in holes:
            if 0 <= h_row < nrow and 0 <= h_col < ncol:
                grid[h_row][h_col] = 'H'

        # Mark goal
        if 0 <= goal_row < nrow and 0 <= goal_col < ncol:
            grid[goal_row][goal_col] = 'G'

        # Mark agent position
        if 0 <= agent_row < nrow and 0 <= agent_col < ncol:
            grid[agent_row][agent_col] = 'A'

        payload = {
            "mode": "grid",
            "grid": grid,
            "agent_position": (int(agent_row), int(agent_col)),
            "game_id": self.config.game_id,
        }

        # Include holes and goal for UI rendering
        if holes_list:
            payload["holes"] = holes_list
        if goal:
            payload["goal"] = goal

        return payload

    def _maybe_load_policy(self) -> None:
        snapshot = self.policy_store.load()
        if snapshot is None:
            if self._require_existing:
                raise FileNotFoundError(
                    f"Required policy not found at {self.policy_store.path}"
                )
            return

        if snapshot.q_table.shape != self.agent.q_table.shape:
            raise ValueError(
                "Policy snapshot shape does not match agent Q-table: "
                f"expected {self.agent.q_table.shape}, got {snapshot.q_table.shape}"
            )

        self.agent.q_table[:] = snapshot.q_table

    # ------------------------------------------------------------------


def _config_payload(config: RunConfig) -> Dict[str, Any]:
    return {
        "game_id": config.game_id,
        "seed": config.seed,
        "max_episodes": config.max_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "policy_strategy": config.policy_strategy.value,
        "policy_path": str(config.ensure_policy_path()),
        "agent_id": config.agent_id,
        "capture_video": config.capture_video,
        "headless": config.headless,
        "extra": config.extra,
    }
