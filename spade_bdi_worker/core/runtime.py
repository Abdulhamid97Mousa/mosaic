"""Headless training loop that powers the refactored worker process."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union
import time

if TYPE_CHECKING:
    from ..adapters import AdapterType


from ..algorithms import create_agent, create_runtime
from ..policies import PolicyStorage
from .config import PolicyStrategy, RunConfig
from .telemetry_worker import TelemetryEmitter
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_RUNTIME_EVENT,
    LOG_WORKER_RUNTIME_JSON_SANITIZED,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    success: bool


class HeadlessTrainer(LogConstantMixin):
    """Drive the learning/evaluation loop and emit JSONL telemetry."""

    def __init__(
        self,
        adapter: AdapterType,
        config: RunConfig,
        emitter: TelemetryEmitter,
    ) -> None:
        self._logger = _LOGGER
        self.adapter = adapter
        self.config = config
        self.emitter = emitter

        # Ensure canonical policy path exists before any IO
        policy_path = self.config.ensure_policy_path()
        self.policy_store = PolicyStorage(policy_path)

        # Type ignore: GUI adapters return AdapterStep, but create_agent handles this
        self.agent = create_agent(adapter)  # type: ignore[arg-type]
        self.runtime = create_runtime(adapter, self.agent)  # type: ignore[arg-type]

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
        
        self.log_constant(
            LOG_WORKER_RUNTIME_EVENT,
            message="HEADLESS_TRAINER_INIT",
            extra={
                "trainer": self.__class__.__name__,
                "adapter": adapter.__class__.__name__,
                "game_id": config.game_id,
                "policy_strategy": config.policy_strategy.value,
                "max_episodes": config.max_episodes,
                "seed": config.seed,
                "policy_path": str(policy_path),
            },
        )

    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute training/evaluation and emit JSONL telemetry to stdout."""
        config_payload = _config_payload(self.config)
        self.emitter.run_started(
            self.config.run_id,
            config_payload,
            worker_id=self.config.worker_id,  # type: ignore[call-arg]
        )

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
                    "worker_id": self.config.worker_id,
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
                    worker_id=self.config.worker_id,  # type: ignore[call-arg]
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
                self.emitter.artifact(
                    self.config.run_id,
                    "policy",
                    str(path),
                    worker_id=self.config.worker_id,  # type: ignore[call-arg]
                )

            self.emitter.run_completed(
                self.config.run_id,
                status="completed",
                worker_id=self.config.worker_id,  # type: ignore[call-arg]
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            self.emitter.run_completed(
                self.config.run_id,
                status="failed",
                error=str(exc),
                worker_id=self.config.worker_id,  # type: ignore[call-arg]
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
        reset_result = self.adapter.reset(seed=episode_seed)
        state = int(reset_result.observation)  # GUI adapters return AdapterStep objects
        obs = reset_result.info
        total_reward = 0.0
        success = False
        steps_taken = 0

        max_steps = self.config.max_steps_per_episode
        for step_index in range(max_steps):
            action = self.runtime.get_action(state, training=self._is_training)
            q_before = float(self.agent.q_table[state, action])

            step_result = self.adapter.step(action)
            next_state = int(step_result.observation)
            reward = float(step_result.reward)
            terminated = bool(step_result.terminated)
            truncated = bool(step_result.truncated)
            next_obs = step_result.info
            if not terminated and not truncated and (step_index + 1) >= max_steps:
                truncated = True
            done = terminated or truncated

            if self._is_training:
                self.runtime.update_q_online(state, action, reward, next_state, done)

            # Generate render payload for grid visualization from the NEW observation
            # CRITICAL: Use next_obs (after step), not obs (before step)
            # Convert Mapping to Dict for render payload generation
            render_payload = self._generate_render_payload(next_state, dict(next_obs))

            # Build observation dict for telemetry: include state and grid position
            # This provides meaningful context for replays and analysis
            obs_for_telemetry = self._build_observation_dict(state, dict(obs))
            next_obs_for_telemetry = self._build_observation_dict(next_state, dict(next_obs))

            self.emitter.step(
                self.config.run_id,
                episode_number,  # Pass episode_number (display value = seed + episode_index)
                step_index,
                agent_id=self.config.agent_id,
                worker_id=self.config.worker_id,  # type: ignore[call-arg]
                action=int(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                state=int(state),
                next_state=int(next_state),
                observation=obs_for_telemetry,
                next_observation=next_obs_for_telemetry,
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

    def _build_observation_dict(self, state: int, info: Dict[str, Any]) -> Dict[str, Any]:
        """Build a meaningful observation dictionary for telemetry.
        
        For toy-text environments like FrozenLake, converts the state integer
        and info dict into a structured observation with position and metadata.
        
        Args:
            state: Current state (flat index)
            info: Observation info dict from adapter
            
        Returns:
            Dict with state, position, and environment metadata
        """
        observation_dict: Dict[str, Any] = {
            "state": int(state),
        }
        
        # Try to compute grid position from state
        try:
            if hasattr(self.adapter, "state_to_pos"):
                row, col = self.adapter.state_to_pos(state)
                observation_dict["position"] = {"row": int(row), "col": int(col)}
        except Exception:
            pass
        
        # Include environment-specific info (like probability for FrozenLake)
        if info:
            observation_dict.update(self._make_json_safe(info))

        return observation_dict

    def _make_json_safe(self, value: Any, path: str = "root") -> Any:
        """Recursively convert telemetry payloads into JSON-serialisable structures."""

        # Handle mappings
        if isinstance(value, dict):
            return {
                key: self._make_json_safe(item, f"{path}.{key}")
                for key, item in value.items()
            }

        # Handle sequences (lists/tuples)
        if isinstance(value, (list, tuple)):
            return [self._make_json_safe(item, f"{path}[{idx}]") for idx, item in enumerate(value)]

        # Handle sets by converting to lists to preserve JSON compatibility
        if isinstance(value, set):
            self.log_constant(
                LOG_WORKER_RUNTIME_JSON_SANITIZED,
                message="Converted set to list for telemetry",
                extra={"field": path, "strategy": "set_to_list"},
            )
            return [self._make_json_safe(item, f"{path}[{idx}]") for idx, item in enumerate(sorted(value, key=str))]

        # NumPy arrays/scalars expose ``tolist``/``item`` helpers; use them when available
        if hasattr(value, "tolist"):
            try:
                safe_value = value.tolist()
                self.log_constant(
                    LOG_WORKER_RUNTIME_JSON_SANITIZED,
                    message="Converted ndarray to list for telemetry",
                    extra={"field": path, "strategy": "tolist"},
                )
                return self._make_json_safe(safe_value, path)
            except Exception:
                pass

        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                scalar_value = value.item()
                self.log_constant(
                    LOG_WORKER_RUNTIME_JSON_SANITIZED,
                    message="Converted numpy scalar to Python scalar",
                    extra={"field": path, "strategy": "item"},
                )
                return self._make_json_safe(scalar_value, path)
            except Exception:
                pass

        # Primitives are already JSON friendly
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        # Fallback: stringify any remaining unsupported object
        self.log_constant(
            LOG_WORKER_RUNTIME_JSON_SANITIZED,
            message="Stringified unsupported telemetry value",
            extra={"field": path, "strategy": "str"},
        )
        return str(value)

    def _generate_render_payload(self, state: int, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate render payload for grid visualization.
        
        GUI adapters don't populate observation info dicts with game metadata,
        so we need to query the adapter directly for holes, goal positions, etc.

        Args:
            state: Current state (flat index)
            obs: Observation info dict from adapter (minimal Gymnasium info)

        Returns:
            Render payload dict with grid, agent position, and game metadata
        """
        # Call the adapter's render() method to get the full payload
        # GUI adapters have a render() method that produces complete visualization data
        if hasattr(self.adapter, "render") and callable(self.adapter.render):
            try:
                payload = self.adapter.render()
                self.log_constant(
                    LOG_WORKER_RUNTIME_EVENT,
                    message="Render payload from adapter.render()",
                    extra={
                        "state": state,
                        "payload_keys": list(payload.keys()) if isinstance(payload, dict) else "not_dict",
                        "has_holes": "holes" in payload if isinstance(payload, dict) else False,
                        "has_goal": "goal" in payload if isinstance(payload, dict) else False,
                        "has_grid": "grid" in payload if isinstance(payload, dict) else False,
                    },
                )
                return payload
            except Exception as e:
                self.log_constant(
                    LOG_WORKER_RUNTIME_EVENT,
                    message=f"Adapter render() failed, falling back to manual construction: {e}",
                )
        
        # Fallback: manual construction (for adapters without render method)
        # Get grid dimensions from adapter
        if hasattr(self.adapter, "_get_grid_width"):
            ncol = self.adapter._get_grid_width()
            if hasattr(self.adapter, "defaults"):
                nrow = self.adapter.defaults.grid_height
            else:
                nrow = ncol  # Assume square
        else:
            nrow = ncol = 8  # Default fallback
        
        # Get agent position from state
        if hasattr(self.adapter, "state_to_pos"):
            agent_row, agent_col = self.adapter.state_to_pos(state)
        else:
            agent_row = state // ncol
            agent_col = state % ncol
        
        # Get goal from adapter defaults
        if hasattr(self.adapter, "defaults"):
            goal_row, goal_col = self.adapter.defaults.goal
        else:
            goal_row, goal_col = nrow - 1, ncol - 1
        
        # Basic grid (will be missing holes in fallback mode)
        grid = [['F' for _ in range(ncol)] for _ in range(nrow)]
        grid[goal_row][goal_col] = 'G'
        grid[agent_row][agent_col] = 'A'
        
        return {
            "mode": "grid",
            "grid": grid,
            "agent_position": (int(agent_row), int(agent_col)),
            "game_id": self.config.game_id,
            "goal": {"row": int(goal_row), "col": int(goal_col)},
        }

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
        "worker_id": config.worker_id,
        "capture_video": config.capture_video,
        "headless": config.headless,
        "extra": config.extra,
    }
