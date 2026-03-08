"""
MOSAIC MultiGrid environment wrappers and factory functions.

This module provides algorithm-agnostic wrappers for MOSAIC MultiGrid environments,
including observation preprocessing, episode limits, and reward scaling.

Usage:
    from cleanrl_worker.wrappers.mosaic_multigrid import make_mosaic_env, is_mosaic_env

    # Create a single environment
    env = make_mosaic_env("MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0", view_size=7)

    # Create vectorized environments for training
    envs = gym.vector.SyncVectorEnv([
        make_mosaic_env(env_id, i, view_size=7) for i in range(num_envs)
    ])
"""

import logging
import os
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from cleanrl_worker.fastlane import maybe_wrap_env

logger = logging.getLogger(__name__)


class SingleAgentWrapper(gym.Wrapper):
    """Wrapper to convert multi-agent MOSAIC environments to single-agent interface.

    MOSAIC MultiGrid environments return nested Dict spaces:
    - Action space: Dict(0: Discrete(n))
    - Observation space: Dict(0: Dict('direction': Discrete(4), 'image': Box(...), 'mission': ...))

    This wrapper extracts agent 0's image observation and action space.
    """

    def __init__(self, env: gym.Env, agent_id: int = 0):
        super().__init__(env)
        self.agent_id = agent_id

        # Extract single agent's action space from Dict
        if isinstance(env.action_space, gym.spaces.Dict):
            self.action_space = env.action_space[agent_id]
        else:
            self.action_space = env.action_space

        # Extract image observation space from nested Dict
        if isinstance(env.observation_space, gym.spaces.Dict):
            agent_obs_space = env.observation_space[agent_id]
            if isinstance(agent_obs_space, gym.spaces.Dict) and 'image' in agent_obs_space.spaces:
                # Extract just the image component
                self.observation_space = agent_obs_space['image']
            else:
                self.observation_space = agent_obs_space
        else:
            self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Extract single agent's image observation from nested dict
        if isinstance(obs, dict):
            obs = obs[self.agent_id]
            if isinstance(obs, dict) and 'image' in obs:
                obs = obs['image']
        return obs, info

    def step(self, action):
        # Wrap action in dict for multi-agent env
        if isinstance(self.env.action_space, gym.spaces.Dict):
            action_dict = {self.agent_id: action}
        else:
            action_dict = action

        obs, reward, terminated, truncated, info = self.env.step(action_dict)

        # Extract single agent's data from dicts
        if isinstance(obs, dict):
            obs = obs[self.agent_id]
            if isinstance(obs, dict) and 'image' in obs:
                obs = obs['image']
        if isinstance(reward, dict):
            reward = reward[self.agent_id]
        if isinstance(terminated, dict):
            terminated = terminated[self.agent_id]
        if isinstance(truncated, dict):
            truncated = truncated[self.agent_id]

        return obs, reward, terminated, truncated, info


def is_mosaic_env(env_id: str) -> bool:
    """Check if the environment is a MOSAIC MultiGrid environment.

    Args:
        env_id: The environment ID string

    Returns:
        True if the environment is MOSAIC MultiGrid based
    """
    return "MosaicMultiGrid" in env_id or "Mosaic" in env_id


def make_mosaic_env(
    env_id: str,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str = "",
    max_episode_steps: int = 256,
    reward_scale: float = 1.0,
    view_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Callable[[], gym.Env]:
    """Create a MOSAIC MultiGrid environment factory function.

    This function returns a thunk (factory function) that creates and wraps
    the environment with appropriate wrappers for MOSAIC MultiGrid training.

    Wrappers applied (in order):
        1. RecordVideo (optional, for first env only)
        2. TimeLimit - Ensures episodes terminate
        3. TransformReward - Optional reward scaling
        4. RecordEpisodeStatistics - Track episode returns/lengths

    Args:
        env_id: MOSAIC MultiGrid environment ID (e.g., "MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0")
        idx: Environment index (for seeding and video recording)
        capture_video: Whether to record videos (only for idx=0)
        run_name: Name for video directory
        max_episode_steps: Maximum steps per episode
        reward_scale: Reward scaling factor (default: 1.0)
        view_size: Agent view size (3, 5, 7, etc.). If provided, sets MOSAIC_VIEW_SIZE env var
        seed: Random seed for environment

    Returns:
        A thunk (callable) that creates the wrapped environment

    Example:
        >>> env_fn = make_mosaic_env("MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0", view_size=7)
        >>> env = env_fn()
        >>> obs, info = env.reset()
    """
    def thunk():
        # Import MOSAIC MultiGrid environments to register them
        try:
            import mosaic_multigrid.envs  # noqa: F401 - registers environments
        except ImportError:
            logger.error(f"mosaic_multigrid not installed, cannot create {env_id}")
            raise ImportError(
                "mosaic_multigrid package is required for MOSAIC MultiGrid environments. "
                "Install it with: pip install mosaic_multigrid"
            )

        # Create base environment with view_size parameter
        make_kwargs = {}
        if view_size is not None:
            make_kwargs['view_size'] = view_size
            logger.debug(f"Creating {env_id} with view_size={view_size}")

        # Check if fastlane is enabled (requires rendering)
        fastlane_enabled = os.getenv("GYM_GUI_FASTLANE_ONLY") in {"1", "true", "True", "yes"}
        needs_rendering = capture_video or fastlane_enabled

        # Disable passive env checker since SingleAgentWrapper handles multi-agent conversion
        make_kwargs['disable_env_checker'] = True

        if needs_rendering:
            env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **make_kwargs)

        # Convert multi-agent Dict spaces to single-agent for solo environments
        env = SingleAgentWrapper(env, agent_id=0)

        # Add time limit
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        # Optional reward scaling
        if reward_scale != 1.0:
            env = gym.wrappers.TransformReward(env, lambda r: r * reward_scale)

        # Track episode statistics
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Wrap with FastLane telemetry if enabled
        env = maybe_wrap_env(env)

        return env

    return thunk


def make_mosaic_envs(
    env_id: str,
    num_envs: int = 4,
    seed: Optional[int] = None,
    capture_video: bool = False,
    run_name: str = "",
    max_episode_steps: int = 256,
    reward_scale: float = 1.0,
    view_size: Optional[int] = None,
) -> gym.vector.SyncVectorEnv:
    """Create vectorized MOSAIC MultiGrid environments.

    Args:
        env_id: MOSAIC MultiGrid environment ID
        num_envs: Number of parallel environments
        seed: Base random seed (each env gets seed + idx)
        capture_video: Whether to record videos (only for first env)
        run_name: Name for video directory
        max_episode_steps: Maximum steps per episode
        reward_scale: Reward scaling factor
        view_size: Agent view size (3, 5, 7, etc.)

    Returns:
        Vectorized environment

    Example:
        >>> envs = make_mosaic_envs(
        ...     "MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0",
        ...     num_envs=8,
        ...     view_size=7,
        ...     seed=42
        ... )
        >>> obs, info = envs.reset()
    """
    return gym.vector.SyncVectorEnv([
        make_mosaic_env(
            env_id,
            idx=i,
            capture_video=capture_video,
            run_name=run_name,
            max_episode_steps=max_episode_steps,
            reward_scale=reward_scale,
            view_size=view_size,
            seed=seed + i if seed is not None else None,
        )
        for i in range(num_envs)
    ])


# Alias for compatibility with make_env naming convention
make_env = make_mosaic_env

__all__ = [
    "is_mosaic_env",
    "make_mosaic_env",
    "make_mosaic_envs",
    "make_env",
]
