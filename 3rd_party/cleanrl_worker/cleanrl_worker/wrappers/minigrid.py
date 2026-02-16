"""
MiniGrid/BabyAI environment wrappers and factory functions.

This module provides algorithm-agnostic wrappers for MiniGrid and BabyAI environments,
including observation preprocessing, episode limits, and reward scaling.

Usage:
    from cleanrl_worker.wrappers.minigrid import make_minigrid_env, is_minigrid_env

    # Create a single environment
    env = make_minigrid_env("BabyAI-GoToRedBallNoDists-v0", max_episode_steps=256)

    # Create vectorized environments for training
    envs = make_minigrid_envs("BabyAI-GoToRedBallNoDists-v0", num_envs=4, seed=42)
"""

import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def _has_wrapper(env: gym.Env, wrapper_cls: type) -> bool:
    """Walk the wrapper chain and return True if *wrapper_cls* is present."""
    current = env
    while current is not None:
        if isinstance(current, wrapper_cls):
            return True
        current = getattr(current, "env", None)
    return False


def is_minigrid_env(env_id: str) -> bool:
    """Check if the environment is a MiniGrid or BabyAI environment.

    Args:
        env_id: The environment ID string

    Returns:
        True if the environment is MiniGrid or BabyAI based
    """
    return "MiniGrid" in env_id or "BabyAI" in env_id


def make_minigrid_env(
    env_id: str,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str = "",
    max_episode_steps: int = 256,
    reward_scale: float = 1.0,
    procedural_generation: bool = True,
    seed: Optional[int] = None,
) -> Callable[[], gym.Env]:
    """Create a MiniGrid/BabyAI environment factory function.

    This function returns a thunk (factory function) that creates and wraps
    the environment with appropriate wrappers for MiniGrid/BabyAI training.

    Wrappers applied (in order):
        1. RecordVideo (optional, for first env only)
        2. ImgObsWrapper - Converts Dict obs to image-only (7, 7, 3)
        3. TimeLimit - Ensures episodes terminate
        4. TransformReward - Optional reward scaling (BabyAI paper uses 20x)
        5. RecordEpisodeStatistics - Track episode returns/lengths
        6. ProceduralGenerationWrapper - Control level randomization

    Args:
        env_id: MiniGrid/BabyAI environment ID
        idx: Environment index (for parallel envs)
        capture_video: Whether to record video (only env 0)
        run_name: Name for video directory
        max_episode_steps: Maximum steps per episode (default 256)
        reward_scale: Reward multiplier (BabyAI paper uses 20.0)
        procedural_generation: If True, new random layout each episode
        seed: Random seed for reproducibility

    Returns:
        A thunk function that creates the wrapped environment

    Example:
        >>> env_fn = make_minigrid_env("BabyAI-GoToRedBallNoDists-v0", seed=42)
        >>> env = env_fn()
        >>> obs, info = env.reset()
        >>> print(obs.shape)  # (7, 7, 3)
    """
    def thunk():
        # Create base environment
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # Apply MiniGrid-specific wrappers
        # 1. Convert Dict observation to image-only (skip if already Box,
        #    e.g. when sitecustomize.py already applied ImgObsWrapper)
        if isinstance(env.observation_space, gym.spaces.Dict):
            from minigrid.wrappers import ImgObsWrapper
            env = ImgObsWrapper(env)

        # 2. Add time limit (MiniGrid default max_steps=0 means no limit)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        # 3. Optional reward scaling (BabyAI paper uses 20x)
        if reward_scale != 1.0:
            env = gym.wrappers.TransformReward(env, lambda r: r * reward_scale)

        # 4. Track episode statistics
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # 5. Control procedural generation (skip if sitecustomize.py already
        #    applied ProceduralGenerationWrapper inside gym.make)
        from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
        if not _has_wrapper(env, ProceduralGenerationWrapper):
            env = ProceduralGenerationWrapper(
                env,
                procedural=procedural_generation,
                fixed_seed=seed if not procedural_generation else (seed + idx if seed is not None else None)
            )

        return env

    return thunk


def make_minigrid_envs(
    env_id: str,
    num_envs: int = 4,
    seed: Optional[int] = None,
    capture_video: bool = False,
    run_name: str = "",
    max_episode_steps: int = 256,
    reward_scale: float = 1.0,
    procedural_generation: bool = True,
    async_envs: bool = False,
) -> gym.vector.VectorEnv:
    """Create vectorized MiniGrid/BabyAI environments.

    Args:
        env_id: MiniGrid/BabyAI environment ID
        num_envs: Number of parallel environments
        seed: Random seed for reproducibility
        capture_video: Whether to record video (only env 0)
        run_name: Name for video/run directory
        max_episode_steps: Maximum steps per episode
        reward_scale: Reward multiplier
        procedural_generation: If True, new random layout each episode
        async_envs: If True, use AsyncVectorEnv (not recommended for MiniGrid)

    Returns:
        Vectorized environment

    Example:
        >>> envs = make_minigrid_envs("BabyAI-GoToRedBallNoDists-v0", num_envs=8)
        >>> obs, _ = envs.reset()
        >>> print(obs.shape)  # (8, 7, 7, 3)
    """
    env_fns = [
        make_minigrid_env(
            env_id=env_id,
            idx=i,
            capture_video=capture_video,
            run_name=run_name,
            max_episode_steps=max_episode_steps,
            reward_scale=reward_scale,
            procedural_generation=procedural_generation,
            seed=seed,
        )
        for i in range(num_envs)
    ]

    if async_envs:
        return gym.vector.AsyncVectorEnv(env_fns)
    else:
        return gym.vector.SyncVectorEnv(env_fns)


def make_env(
    env_id: str,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str = "",
    max_episode_steps: int = 256,
    reward_scale: float = 1.0,
    procedural_generation: bool = True,
    seed: Optional[int] = None,
    **kwargs,
) -> Callable[[], gym.Env]:
    """Universal environment factory that auto-detects MiniGrid environments.

    This is the main entry point for creating environments. It automatically
    detects if the environment is MiniGrid/BabyAI and applies appropriate wrappers.

    Args:
        env_id: Environment ID
        idx: Environment index
        capture_video: Whether to record video
        run_name: Name for video directory
        max_episode_steps: Maximum steps (only for MiniGrid)
        reward_scale: Reward multiplier (only for MiniGrid)
        procedural_generation: Control randomization (only for MiniGrid)
        seed: Random seed
        **kwargs: Additional arguments passed to gym.make

    Returns:
        A thunk function that creates the wrapped environment
    """
    if is_minigrid_env(env_id):
        logger.debug("Detected MiniGrid environment: %s", env_id)
        return make_minigrid_env(
            env_id=env_id,
            idx=idx,
            capture_video=capture_video,
            run_name=run_name,
            max_episode_steps=max_episode_steps,
            reward_scale=reward_scale,
            procedural_generation=procedural_generation,
            seed=seed,
        )
    else:
        # Generic environment factory for non-MiniGrid envs
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", **kwargs)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **kwargs)

            env = gym.wrappers.RecordEpisodeStatistics(env)

            # Add procedural generation wrapper if not already applied
            try:
                from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
                if not _has_wrapper(env, ProceduralGenerationWrapper):
                    env = ProceduralGenerationWrapper(
                        env,
                        procedural=procedural_generation,
                        fixed_seed=seed if not procedural_generation else (seed + idx if seed is not None else None)
                    )
            except ImportError:
                pass

            return env

        return thunk
