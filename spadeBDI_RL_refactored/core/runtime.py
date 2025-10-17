"""Headless training loop that powers the refactored worker process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..adapters.frozenlake import FrozenLakeAdapter
from ..algorithms import create_agent, create_runtime
from ..policies import PolicyStorage
from .config import PolicyStrategy, RunConfig
from .telemetry import TelemetryEmitter


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
        adapter: FrozenLakeAdapter,
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

    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute training/evaluation and emit JSONL telemetry to stdout."""
        config_payload = _config_payload(self.config)
        self.emitter.run_started(self.config.run_id, config_payload)

        try:
            summaries: list[EpisodeMetrics] = []
            for episode in range(self.config.max_episodes):
                summary = self._run_episode(episode)
                summaries.append(summary)
                self.emitter.episode(
                    self.config.run_id,
                    episode,
                    reward=summary.total_reward,
                    steps=summary.steps,
                    success=summary.success,
                )

            if self._should_save:
                metadata = {
                    "run_id": self.config.run_id,
                    "env_id": self.config.env_id,
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
    def _run_episode(self, episode_index: int) -> EpisodeMetrics:
        """Run single episode with JSONL telemetry."""
        state, obs = self.adapter.reset(seed=self.config.seed + episode_index)
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

            self.emitter.step(
                self.config.run_id,
                episode_index,
                step_index,
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
            )

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
        "env_id": config.env_id,
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
