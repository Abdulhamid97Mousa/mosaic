"""Reproducibility wrapper for legacy gym-multigrid environments.

The original gym-multigrid (ArnaudFickinger) has a reproducibility bug in step():
    order = np.random.permutation(len(actions))  # Uses global RNG!

This ignores the seeded self.np_random, making trajectories non-reproducible
even when env.seed() is called before reset().

This wrapper fixes the issue by seeding the global np.random from the
environment's seeded np_random before each step(), ensuring deterministic
action ordering without modifying the 3rd_party source code.

Usage:
    from gym_multigrid.envs import SoccerGame4HEnv10x15N2
    from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper

    env = SoccerGame4HEnv10x15N2()
    env = ReproducibleMultiGridWrapper(env)
    env.seed(42)  # Now fully reproducible!
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class ReproducibleMultiGridWrapper:
    """Wrapper that fixes reproducibility in legacy gym-multigrid environments.

    The underlying gym-multigrid step() uses np.random.permutation() for action
    ordering, which ignores the seeded np_random. This wrapper seeds the global
    np.random from env.np_random before each step, making trajectories reproducible.

    This wrapper is transparent to Operators (human play) since it only affects
    the random number generation used for action ordering - the game mechanics
    remain identical.

    Attributes:
        env: The wrapped gym-multigrid environment
    """

    def __init__(self, env: Any) -> None:
        """Initialize the wrapper.

        Args:
            env: A gym-multigrid environment instance (SoccerGame, CollectGame, etc.)
        """
        self._env = env
        # Track if we've been seeded (for reproducibility verification)
        self._seeded = False

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        return getattr(self._env, name)

    def seed(self, seed: int | None = None) -> List[int]:
        """Seed the environment for reproducibility.

        Args:
            seed: Random seed value

        Returns:
            List containing the seed used
        """
        self._seeded = True
        return self._env.seed(seed)

    def reset(self) -> Any:
        """Reset the environment.

        Returns:
            Initial observations (list of observations per agent)
        """
        return self._env.reset()

    def step(self, actions: List[int]) -> Tuple[Any, Any, bool, dict]:
        """Execute actions with reproducible action ordering.

        This method fixes the reproducibility bug by seeding np.random
        from the environment's seeded np_random before calling step().

        Args:
            actions: List of actions, one per agent

        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Fix reproducibility: Seed global np.random from env's seeded RNG.
        # This ensures np.random.permutation() in the underlying env.step()
        # produces deterministic results based on env.np_random state.
        if hasattr(self._env, 'np_random') and self._env.np_random is not None:
            # Generate a deterministic seed from the env's seeded RNG
            # Use a large range to minimize collision probability
            deterministic_seed = int(self._env.np_random.integers(0, 2**31 - 1))
            np.random.seed(deterministic_seed)

        return self._env.step(actions)

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """Render the environment."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

    @property
    def unwrapped(self) -> Any:
        """Return the unwrapped environment."""
        return self._env

    @property
    def agents(self) -> List[Any]:
        """Return the list of agents."""
        return self._env.agents

    @property
    def action_space(self) -> Any:
        """Return the action space."""
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """Return the observation space."""
        return self._env.observation_space

    @property
    def np_random(self) -> Any:
        """Return the seeded random number generator."""
        return self._env.np_random

    def __repr__(self) -> str:
        return f"ReproducibleMultiGridWrapper({self._env})"


__all__ = ["ReproducibleMultiGridWrapper"]
