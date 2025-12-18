"""Policy Evaluator for visualizing trained Ray RLlib policies.

This module provides evaluation functionality that:
1. Loads trained policies from checkpoints
2. Runs evaluation episodes on PettingZoo environments
3. Streams frames to FastLane for real-time visualization
4. Collects metrics (rewards, episode lengths)

Example:
    >>> from ray_worker.policy_evaluator import PolicyEvaluator
    >>>
    >>> evaluator = PolicyEvaluator(
    ...     env_id="pursuit_v4",
    ...     env_family="sisl",
    ...     checkpoint_path="/path/to/checkpoint",
    ...     run_id="eval_01",
    ... )
    >>> evaluator.run(num_episodes=10)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ray_worker.evaluation_results import (
    EvaluationResultsConfig,
    EvaluationResultsWriter,
    extract_run_id_from_checkpoint,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation.

    Attributes:
        env_id: PettingZoo environment ID (e.g., "pursuit_v4").
        env_family: Environment family (e.g., "sisl", "mpe").
        checkpoint_path: Path to Ray RLlib checkpoint.
        run_id: Unique run identifier for FastLane.
        policy_id: Policy ID to use (default: "shared").
        num_episodes: Number of evaluation episodes to run.
        max_steps_per_episode: Maximum steps per episode.
        render_mode: Render mode ("rgb_array" for FastLane).
        fastlane_enabled: Whether to stream to FastLane.
        deterministic: Use deterministic policy actions.
        seed: Random seed for reproducibility.
        agent_policies: Optional per-agent policy mappings.
        save_results: Whether to save results to disk.
        training_run_id: Training run ID for results storage (auto-extracted if None).
    """
    env_id: str
    env_family: str
    checkpoint_path: str
    run_id: str
    policy_id: str = "shared"
    num_episodes: int = 10
    max_steps_per_episode: int = 1000
    render_mode: str = "rgb_array"
    fastlane_enabled: bool = True
    deterministic: bool = True
    seed: Optional[int] = 42
    agent_policies: Dict[str, str] = field(default_factory=dict)
    save_results: bool = True
    training_run_id: Optional[str] = None


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""
    episode_id: int
    total_reward: float
    episode_length: int
    agent_rewards: Dict[str, float]
    duration_seconds: float
    terminated: bool


class PolicyEvaluator:
    """Evaluator for trained Ray RLlib policies with FastLane visualization.

    Runs evaluation episodes on PettingZoo environments using trained policies
    and streams visualization frames to FastLane.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self._env = None
        self._policy_actor = None
        self._metrics: List[EpisodeMetrics] = []
        self._running = False
        self._results_writer: Optional[EvaluationResultsWriter] = None
        self._results_dir: Optional[Path] = None

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> "PolicyEvaluator":
        """Create evaluator from configuration."""
        return cls(config)

    def setup(self) -> None:
        """Set up environment (with FastLane wrapper) and policy."""
        self._setup_environment()
        self._setup_policy()
        self._setup_results_writer()
        # FastLane is now integrated into the environment wrapper
        # - ParallelFastLaneWrapper publishes frames on each step()
        # - No separate FastLane producer needed

    def _setup_results_writer(self) -> None:
        """Set up the results writer for saving evaluation metrics."""
        if not self.config.save_results:
            return

        # Extract or use provided training run ID
        training_run_id = self.config.training_run_id
        if not training_run_id:
            training_run_id = extract_run_id_from_checkpoint(self.config.checkpoint_path)

        if not training_run_id:
            _LOGGER.warning(
                "Could not extract training run ID from checkpoint path. "
                "Results will not be saved."
            )
            return

        self._results_writer = EvaluationResultsWriter(training_run_id)

        # Set evaluation config
        results_config = EvaluationResultsConfig(
            training_run_id=training_run_id,
            checkpoint_path=self.config.checkpoint_path,
            env_id=self.config.env_id,
            env_family=self.config.env_family,
            num_episodes=self.config.num_episodes,
            max_steps_per_episode=self.config.max_steps_per_episode,
            deterministic=self.config.deterministic,
            seed=self.config.seed,
        )
        self._results_writer.set_config(results_config)

        _LOGGER.info(
            "Results writer initialized for training run: %s",
            training_run_id,
        )

    def _setup_environment(self) -> None:
        """Create the PettingZoo environment with FastLane wrapper if enabled."""
        import os

        # Try to disable pygame display window - we only need rgb_array for FastLane
        # Set multiple env vars to maximize chances of suppressing the window
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        os.environ["SDL_VIDEO_WINDOW_POS"] = "-10000,-10000"  # Position offscreen
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

        _LOGGER.info(
            "Setting up environment: %s (family=%s)",
            self.config.env_id,
            self.config.env_family,
        )

        # Import environment creator based on family
        env_family = self.config.env_family.lower()
        env_id = self.config.env_id

        try:
            if env_family == "sisl":
                from pettingzoo.sisl import (
                    pursuit_v4,
                    multiwalker_v9,
                    waterworld_v4,
                )
                env_creators = {
                    "pursuit_v4": pursuit_v4,
                    "multiwalker_v9": multiwalker_v9,
                    "waterworld_v4": waterworld_v4,
                }
            elif env_family == "mpe":
                from pettingzoo.mpe import (
                    simple_spread_v3,
                    simple_adversary_v3,
                    simple_tag_v3,
                )
                env_creators = {
                    "simple_spread_v3": simple_spread_v3,
                    "simple_adversary_v3": simple_adversary_v3,
                    "simple_tag_v3": simple_tag_v3,
                }
            elif env_family == "butterfly":
                from pettingzoo.butterfly import (
                    cooperative_pong_v5,
                    pistonball_v6,
                )
                env_creators = {
                    "cooperative_pong_v5": cooperative_pong_v5,
                    "pistonball_v6": pistonball_v6,
                }
            elif env_family == "classic":
                from pettingzoo.classic import (
                    chess_v6,
                    connect_four_v3,
                    go_v5,
                )
                env_creators = {
                    "chess_v6": chess_v6,
                    "connect_four_v3": connect_four_v3,
                    "go_v5": go_v5,
                }
            else:
                raise ValueError(f"Unknown environment family: {env_family}")

            if env_id not in env_creators:
                raise ValueError(
                    f"Unknown environment '{env_id}' in family '{env_family}'. "
                    f"Available: {list(env_creators.keys())}"
                )

            # Create environment with render mode
            creator = env_creators[env_id]
            env = creator.parallel_env(render_mode=self.config.render_mode)

            # Wrap with FastLane for live visualization if enabled
            if self.config.fastlane_enabled:
                from ray_worker.fastlane import (
                    ParallelFastLaneWrapper,
                    FastLaneRayConfig,
                    set_fastlane_env_vars,
                )

                # Set env vars for FastLane
                set_fastlane_env_vars(
                    run_id=self.config.run_id,
                    env_name=f"{env_family}/{env_id}",
                    enabled=True,
                    throttle_ms=33,  # ~30 FPS
                )

                # Create FastLane config with worker_index=0 for evaluation
                fl_config = FastLaneRayConfig(
                    enabled=True,
                    run_id=self.config.run_id,
                    env_name=f"{env_family}/{env_id}",
                    worker_index=0,
                    throttle_interval_ms=33,
                )

                env = ParallelFastLaneWrapper(env, fl_config)
                _LOGGER.info(
                    "Environment wrapped with FastLane: stream_id=%s",
                    fl_config.stream_id,
                )

            self._env = env

            # Close pygame display window if it was created
            # Some PettingZoo envs open a window even with render_mode="rgb_array"
            self._close_pygame_display()

            _LOGGER.info(
                "Environment created: %s, agents=%s",
                env_id,
                self._env.possible_agents,
            )

        except ImportError as e:
            _LOGGER.error("Failed to import environment: %s", e)
            raise
        except Exception as e:
            _LOGGER.error("Failed to create environment: %s", e)
            raise

    def _close_pygame_display(self) -> None:
        """Close/hide any pygame display window.

        PettingZoo environments may open a pygame window even with
        render_mode="rgb_array". We close it since we use FastLane for display.
        """
        try:
            import pygame
            if pygame.display.get_init():
                # Try to close the display
                pygame.display.quit()
                _LOGGER.debug("Closed pygame display")
                # Reinitialize without a window for render() to work
                pygame.display.init()
        except Exception as e:
            _LOGGER.debug("Could not close pygame display: %s", e)

    def _setup_policy(self) -> None:
        """Load the trained policy from checkpoint."""
        _LOGGER.info(
            "Loading policy from checkpoint: %s",
            self.config.checkpoint_path,
        )

        from ray_worker.policy_actor import RayPolicyActor

        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._policy_actor = RayPolicyActor.from_checkpoint(
            checkpoint_path,
            policy_id=self.config.policy_id,
            actor_id=f"eval_{self.config.run_id}",
            env_family=self.config.env_family,
            env_id=self.config.env_id,
            deterministic=self.config.deterministic,
        )

        # Set up per-agent policy mapping if provided
        if self.config.agent_policies:
            self._policy_actor.set_policy_mapping(self.config.agent_policies)

        _LOGGER.info(
            "Policy loaded: %s, available_policies=%s",
            self.config.policy_id,
            self._policy_actor.available_policies,
        )

    def run(
        self,
        num_episodes: Optional[int] = None,
        callback: Optional[Callable[[EpisodeMetrics], None]] = None,
    ) -> List[EpisodeMetrics]:
        """Run evaluation episodes.

        Args:
            num_episodes: Number of episodes (overrides config if provided).
            callback: Optional callback called after each episode.

        Returns:
            List of episode metrics.
        """
        num_episodes = num_episodes or self.config.num_episodes
        self._metrics = []
        self._running = True

        _LOGGER.info("Starting evaluation: %d episodes", num_episodes)

        for episode_idx in range(num_episodes):
            if not self._running:
                _LOGGER.info("Evaluation stopped early")
                break

            metrics = self._run_episode(episode_idx)
            self._metrics.append(metrics)

            # Add to results writer
            if self._results_writer:
                self._results_writer.add_episode(
                    episode_id=metrics.episode_id,
                    total_reward=metrics.total_reward,
                    episode_length=metrics.episode_length,
                    agent_rewards=metrics.agent_rewards,
                    duration_seconds=metrics.duration_seconds,
                    terminated=metrics.terminated,
                )

            _LOGGER.info(
                "Episode %d/%d: reward=%.2f, length=%d, time=%.2fs",
                episode_idx + 1,
                num_episodes,
                metrics.total_reward,
                metrics.episode_length,
                metrics.duration_seconds,
            )

            if callback:
                callback(metrics)

        self._running = False

        # Save results to disk
        self._save_results()

        return self._metrics

    def _save_results(self) -> None:
        """Save evaluation results to disk."""
        if not self._results_writer:
            return

        try:
            # Set summary statistics
            summary = self.get_summary()
            self._results_writer.set_summary(summary)

            # Save all results
            self._results_dir = self._results_writer.save()

            _LOGGER.info(
                "Evaluation results saved to: %s",
                self._results_dir,
            )
        except Exception as e:
            _LOGGER.error("Failed to save evaluation results: %s", e)

    def _run_episode(self, episode_idx: int) -> EpisodeMetrics:
        """Run a single evaluation episode.

        Args:
            episode_idx: Episode index.

        Returns:
            Episode metrics.
        """
        # Type narrowing: these are set by setup() before run() is called
        assert self._env is not None, "Environment not initialized, call setup() first"
        assert self._policy_actor is not None, "Policy not initialized, call setup() first"

        start_time = time.time()

        # Reset environment
        seed = self.config.seed + episode_idx if self.config.seed else None
        observations, infos = self._env.reset(seed=seed)

        agent_rewards = {agent: 0.0 for agent in self._env.possible_agents}
        total_reward = 0.0
        step_count = 0
        terminated = False

        while step_count < self.config.max_steps_per_episode:
            # Get actions from policy
            actions = self._policy_actor.select_actions(observations, infos)

            # Fill in random actions for agents without policy
            for agent in observations:
                if agent not in actions:
                    actions[agent] = self._env.action_space(agent).sample()

            # Step environment
            observations, rewards, terminations, truncations, infos = self._env.step(actions)

            # Accumulate rewards
            for agent, reward in rewards.items():
                agent_rewards[agent] = agent_rewards.get(agent, 0.0) + reward
                total_reward += reward

            # FastLane frames are published by ParallelFastLaneWrapper on each step()
            step_count += 1

            # Check if all agents are done
            if all(terminations.values()) or all(truncations.values()):
                terminated = True
                break

            if not observations:  # No agents left
                break

        duration = time.time() - start_time

        return EpisodeMetrics(
            episode_id=episode_idx,
            total_reward=total_reward,
            episode_length=step_count,
            agent_rewards=agent_rewards,
            duration_seconds=duration,
            terminated=terminated,
        )

    def stop(self) -> None:
        """Stop the evaluation."""
        self._running = False

    def cleanup(self) -> None:
        """Clean up resources."""
        _LOGGER.info("Cleaning up evaluator resources")

        # FastLane is integrated into the env wrapper - closing env handles cleanup

        if self._policy_actor is not None:
            try:
                self._policy_actor.cleanup()
            except Exception as e:
                _LOGGER.warning("Error cleaning up policy actor: %s", e)
            self._policy_actor = None

        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                _LOGGER.warning("Error closing environment: %s", e)
            self._env = None

    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics.

        Returns:
            Dictionary with summary statistics.
        """
        if not self._metrics:
            return {}

        rewards = [m.total_reward for m in self._metrics]
        lengths = [m.episode_length for m in self._metrics]
        durations = [m.duration_seconds for m in self._metrics]

        summary = {
            "num_episodes": len(self._metrics),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "mean_duration": float(np.mean(durations)),
            "total_duration": float(sum(durations)),
        }

        # Add results directory if available
        if self._results_dir:
            summary["results_dir"] = str(self._results_dir)

        return summary

    def __enter__(self) -> "PolicyEvaluator":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()


def run_evaluation(
    env_id: str,
    env_family: str,
    checkpoint_path: str,
    run_id: str,
    *,
    policy_id: str = "shared",
    num_episodes: int = 10,
    deterministic: bool = True,
    fastlane_enabled: bool = True,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Convenience function to run policy evaluation.

    Args:
        env_id: PettingZoo environment ID.
        env_family: Environment family.
        checkpoint_path: Path to checkpoint.
        run_id: Unique run identifier.
        policy_id: Policy ID to use.
        num_episodes: Number of episodes.
        deterministic: Use deterministic actions.
        fastlane_enabled: Enable FastLane streaming.
        seed: Random seed.

    Returns:
        Evaluation summary dictionary.
    """
    config = EvaluationConfig(
        env_id=env_id,
        env_family=env_family,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        policy_id=policy_id,
        num_episodes=num_episodes,
        deterministic=deterministic,
        fastlane_enabled=fastlane_enabled,
        seed=seed,
    )

    with PolicyEvaluator(config) as evaluator:
        evaluator.run()
        return evaluator.get_summary()


__all__ = [
    "EvaluationConfig",
    "EpisodeMetrics",
    "PolicyEvaluator",
    "run_evaluation",
]
