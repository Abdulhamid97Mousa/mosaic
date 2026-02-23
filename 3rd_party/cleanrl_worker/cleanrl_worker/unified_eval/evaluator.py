"""Unified evaluation loop for all CleanRL algorithms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from .base import ActionSelector

LOGGER = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of an evaluation run."""

    returns: list[float]
    lengths: list[int]
    avg_return: float
    avg_length: float
    min_return: float
    max_return: float
    std_return: float
    episodes: int

    @classmethod
    def from_episodes(
        cls,
        returns: Sequence[float],
        lengths: Sequence[int],
    ) -> "EvalResult":
        """Create result from episode data."""
        returns_list = list(returns)
        lengths_list = list(lengths)
        n = len(returns_list)

        if n == 0:
            return cls(
                returns=[],
                lengths=[],
                avg_return=0.0,
                avg_length=0.0,
                min_return=0.0,
                max_return=0.0,
                std_return=0.0,
                episodes=0,
            )

        avg_ret = sum(returns_list) / n
        avg_len = sum(lengths_list) / n
        variance = sum((r - avg_ret) ** 2 for r in returns_list) / n

        return cls(
            returns=returns_list,
            lengths=lengths_list,
            avg_return=avg_ret,
            avg_length=avg_len,
            min_return=min(returns_list),
            max_return=max(returns_list),
            std_return=variance ** 0.5,
            episodes=n,
        )


def evaluate(
    selector: ActionSelector,
    envs: Any,
    eval_episodes: int,
    *,
    writer: Optional["SummaryWriter"] = None,
    on_episode: Optional[Callable[[int, float, int], None]] = None,
) -> EvalResult:
    """Run evaluation using the unified loop.

    This single function handles evaluation for ALL CleanRL algorithms.
    The algorithm-specific logic is encapsulated in the ActionSelector.

    Args:
        selector: Algorithm-specific action selector (already loaded with model)
        envs: Gymnasium vector environment
        eval_episodes: Number of episodes to evaluate
        writer: Optional TensorBoard SummaryWriter for logging
        on_episode: Optional callback called after each episode (idx, return, length)

    Returns:
        EvalResult with episode statistics
    """
    obs, _ = envs.reset()
    episodic_returns: list[float] = []
    episode_lengths: list[int] = []

    while len(episodic_returns) < eval_episodes:
        actions = selector.select_action(obs)
        next_obs, _, _, _, infos = envs.step(actions)

        # Handle both old Gymnasium API (final_info) and new API (episode in info)
        episodes_to_process = []

        if "final_info" in infos:
            # Old Gymnasium API: final_info is a list of per-env info dicts
            for info in infos["final_info"]:
                if info is None:
                    continue
                if "episode" not in info:
                    continue
                episodes_to_process.append((
                    float(info["episode"]["r"]),
                    int(info["episode"]["l"]),
                ))
        elif "episode" in infos and "_episode" in infos:
            # New Gymnasium 1.0+ API: episode info is a dict with arrays
            # _episode is a boolean array indicating which envs completed
            episode_done = infos["_episode"]
            for idx, done in enumerate(episode_done):
                if done:
                    episodes_to_process.append((
                        float(infos["episode"]["r"][idx]),
                        int(infos["episode"]["l"][idx]),
                    ))

        for ep_return, ep_length in episodes_to_process:
            episodic_returns.append(ep_return)
            episode_lengths.append(ep_length)

            episode_idx = len(episodic_returns) - 1
            LOGGER.info(
                "eval_episode=%d episodic_return=%.2f episode_length=%d",
                episode_idx,
                ep_return,
                ep_length,
            )

            # Log to TensorBoard if writer provided
            if writer is not None:
                writer.add_scalar("eval/episodic_return", ep_return, episode_idx)
                writer.add_scalar("eval/episode_length", ep_length, episode_idx)

            # Call callback if provided
            if on_episode is not None:
                on_episode(episode_idx, ep_return, ep_length)

            # Stop if we have enough episodes
            if len(episodic_returns) >= eval_episodes:
                break

        obs = next_obs

    result = EvalResult.from_episodes(episodic_returns, episode_lengths)

    # Log summary statistics
    if writer is not None and result.episodes > 0:
        writer.add_scalar("eval/avg_return", result.avg_return, result.episodes)
        writer.add_scalar("eval/avg_length", result.avg_length, result.episodes)
        writer.add_scalar("eval/min_return", result.min_return, result.episodes)
        writer.add_scalar("eval/max_return", result.max_return, result.episodes)
        writer.add_scalar("eval/std_return", result.std_return, result.episodes)
        writer.flush()

    LOGGER.info(
        "Evaluation complete: episodes=%d avg_return=%.2f std=%.2f",
        result.episodes,
        result.avg_return,
        result.std_return,
    )

    return result


def evaluate_from_checkpoint(
    model_path: str,
    make_env: Callable[..., Any],
    env_id: str,
    algo: str,
    eval_episodes: int,
    *,
    device: str = "cpu",
    capture_video: bool = False,
    run_name: str = "eval",
    gamma: float = 0.99,
    tensorboard_dir: Optional[str] = None,
    **adapter_kwargs: Any,
) -> EvalResult:
    """Convenience function to evaluate a checkpoint file.

    This handles the full flow: create env, load model, run evaluation.

    Args:
        model_path: Path to checkpoint file
        make_env: Environment factory function
        env_id: Environment ID (e.g., "CartPole-v1")
        algo: Algorithm name (e.g., "ppo", "dqn")
        eval_episodes: Number of episodes to evaluate
        device: Device for model ("cpu" or "cuda")
        capture_video: Whether to capture video
        run_name: Name for the evaluation run
        gamma: Discount factor (for environments that need it)
        tensorboard_dir: Optional directory for TensorBoard logs
        **adapter_kwargs: Additional adapter-specific parameters

    Returns:
        EvalResult with episode statistics
    """
    import gymnasium as gym

    from .registry import get_adapter

    # Get adapter for this algorithm
    adapter = get_adapter(algo)
    if adapter is None:
        raise ValueError(f"No adapter registered for algorithm: {algo}")

    # Create vectorized environment
    def env_factory():
        return make_env(env_id, 0, capture_video, run_name, gamma)

    envs = gym.vector.SyncVectorEnv([env_factory])

    # Load model
    adapter.load(model_path, envs, device, **adapter_kwargs)

    # Create TensorBoard writer if requested
    writer = None
    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
        except Exception as e:
            LOGGER.warning("Failed to create TensorBoard writer: %s", e)

    try:
        result = evaluate(adapter, envs, eval_episodes, writer=writer)
    finally:
        if writer is not None:
            writer.close()
        envs.close()
        adapter.close()

    return result
