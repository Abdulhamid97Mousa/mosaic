"""
Wrapper to control procedural generation in environments.

This wrapper allows toggling between:
- Procedural generation: Each episode gets a different level layout (standard RL training)
- Fixed generation: Each episode uses the same level layout (for debugging/memorization)
"""

import gymnasium as gym
import numpy as np


class ProceduralGenerationWrapper(gym.Wrapper):
    """
    Controls whether environment uses procedural generation or fixed layouts.

    Args:
        env: The environment to wrap
        procedural: If True, each episode gets a new random layout.
                   If False, all episodes use the same layout (fixed seed).
        fixed_seed: The seed to use for fixed generation mode.
    """

    def __init__(self, env, procedural=True, fixed_seed=None):
        super().__init__(env)
        self.procedural = procedural
        self.fixed_seed = fixed_seed
        self.episode_count = 0

        # For procedural generation, we need an RNG
        if self.procedural:
            self.rng = np.random.default_rng(fixed_seed)

    def reset(self, **kwargs):
        """Reset the environment with appropriate seeding strategy."""

        if self.procedural:
            # Procedural generation: use a new random seed each episode
            # Remove any user-provided seed to avoid conflicts
            kwargs.pop('seed', None)

            # Generate a unique seed for this episode
            # Convert to Python int (Gymnasium requires Python int, not numpy.int64)
            new_seed = int(self.rng.integers(0, 2**31 - 1))

            obs, info = self.env.reset(seed=new_seed, **kwargs)

            # Add metadata about procedural generation
            info['procedural_generation'] = True
            info['episode_seed'] = new_seed
            info['episode_number'] = self.episode_count

        else:
            # Fixed generation: always use the same seed
            kwargs['seed'] = self.fixed_seed

            obs, info = self.env.reset(**kwargs)

            # Add metadata about fixed generation
            info['procedural_generation'] = False
            info['fixed_seed'] = self.fixed_seed
            info['episode_number'] = self.episode_count

        self.episode_count += 1
        return obs, info
