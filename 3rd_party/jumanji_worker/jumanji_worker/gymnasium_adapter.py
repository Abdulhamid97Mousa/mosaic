"""Gymnasium adapter for exposing Jumanji environments to other MOSAIC workers.

This module bridges Jumanji's JAX-based environments to the standard Gymnasium API,
enabling CleanRL, Ray, and XuanCe workers to use Jumanji environments.

Implementation Strategy:
    Rather than reinventing the wheel, this module leverages existing ecosystem tools:

    1. **Jumanji's JumanjiToGymWrapper** - Built-in Gymnasium compatibility wrapper
       that handles state management, PRNG key splitting, JIT compilation, and
       pytree-to-numpy conversion.

    2. **Gymnasium's FlattenObservation** - Standard wrapper for flattening
       Dict/nested observation spaces to Box spaces.

    3. **SuperSuit** (optional) - Additional preprocessing wrappers like
       frame stacking, color reduction, etc.

Usage by other workers:
    # After importing this module, environments are registered:
    import gymnasium
    env = gymnasium.make("jumanji/Game2048-v1")

    # Or use the factory function directly:
    from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
    env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)

See Also:
    - jumanji.wrappers.JumanjiToGymWrapper: The upstream Gymnasium bridge
    - shimmy: Farama's compatibility wrapper library (similar pattern)
    - supersuit: Preprocessing wrappers for Gymnasium/PettingZoo
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

# CRITICAL: Configure JAX BEFORE any JAX imports
# Disable autotuning to prevent XLA compilation errors
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
# Force CPU backend by default to avoid GPU/XLA issues
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
# Disable JAX GPU memory preallocation
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# CRITICAL: Set matplotlib to non-interactive backend BEFORE any imports
# This prevents popup windows when rendering Jumanji environments
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
try:
    matplotlib.use('Agg', force=True)
except Exception:
    pass  # Backend may already be set

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Disable interactive mode and patch show() to prevent popups
plt.ioff()

# Patch plt.show() - global show function
_original_plt_show = plt.show
def _no_plt_show(*args, **kwargs):
    """No-op replacement for plt.show() to prevent popup windows."""
    pass
plt.show = _no_plt_show

# Patch Figure.show() - instance method called by Jumanji viewer
_original_fig_show = Figure.show
def _no_fig_show(self, *args, **kwargs):
    """No-op replacement for fig.show() to prevent popup windows."""
    pass
Figure.show = _no_fig_show

import gymnasium as gym
import numpy as np

from jumanji_worker.config import SUPPORTED_ENVIRONMENTS

LOGGER = logging.getLogger(__name__)

# Lazy imports to handle missing JAX gracefully
_HAS_JAX = False
_JumanjiToGymWrapper = None


def _ensure_jumanji_imported():
    """Lazily import Jumanji and its wrapper."""
    global _HAS_JAX, _JumanjiToGymWrapper

    if _JumanjiToGymWrapper is not None:
        return True

    try:
        import jax  # noqa: F401
        import jumanji  # noqa: F401
        from jumanji.wrappers import JumanjiToGymWrapper

        _HAS_JAX = True
        _JumanjiToGymWrapper = JumanjiToGymWrapper
        return True
    except ImportError:
        _HAS_JAX = False
        return False


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten nested Dict observations to a single Box space.

    This is a thin wrapper that flattens structured observations from
    Jumanji environments (which often return chex.dataclass observations)
    into a single flat numpy array suitable for standard RL algorithms.

    Note:
        Gymnasium provides `FlattenObservation` but it requires the observation
        space to already be a Dict. Jumanji's observations may be dataclass-like,
        so we handle the conversion here.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Compute flattened observation space
        self._flat_size = self._compute_flat_size(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._flat_size,),
            dtype=np.float32,
        )

    def _compute_flat_size(self, space: gym.spaces.Space) -> int:
        """Recursively compute the flattened size of a space."""
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Discrete):
            return 1
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return len(space.nvec)
        elif isinstance(space, gym.spaces.Dict):
            return sum(self._compute_flat_size(s) for s in space.spaces.values())
        else:
            return 1

    def observation(self, observation: Any) -> np.ndarray:
        """Flatten the observation to a 1D numpy array."""
        return self._flatten(observation).astype(np.float32)

    def _flatten(self, obs: Any) -> np.ndarray:
        """Recursively flatten observation."""
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, dict):
            parts = [self._flatten(obs[k]) for k in sorted(obs.keys())]
            return np.concatenate(parts) if parts else np.array([])
        elif isinstance(obs, (int, float)):
            return np.array([obs])
        else:
            return np.asarray(obs).flatten()


def _create_jumanji_env_with_viewer(env_id: str, render_mode: str = "rgb_array"):
    """Create Jumanji environment with the specified viewer render mode.

    Jumanji viewers are created at environment init time and default to 'human' mode.
    To get RGB arrays for GUI embedding, we need to recreate the viewer with 'rgb_array'.
    """
    import matplotlib.pyplot as plt

    # CRITICAL: Disable interactive mode to prevent popup windows
    # This must be done before creating the environment
    plt.ioff()

    import jumanji

    # Create environment first
    jumanji_env = jumanji.make(env_id)

    # Replace viewer with one configured for rgb_array mode
    if render_mode == "rgb_array" and hasattr(jumanji_env, "_viewer"):
        viewer_class = type(jumanji_env._viewer)
        # Get viewer constructor parameters from current viewer
        if hasattr(jumanji_env._viewer, "_name"):
            name = jumanji_env._viewer._name
        else:
            name = env_id.split("-")[0]

        # Create new viewer with rgb_array mode
        try:
            # Try common viewer constructor signatures
            if hasattr(jumanji_env._viewer, "_board_size"):
                # Game2048, SlidingPuzzle style
                jumanji_env._viewer = viewer_class(
                    name=name,
                    board_size=jumanji_env._viewer._board_size,
                    render_mode=render_mode,
                )
            elif hasattr(jumanji_env._viewer, "_num_rows"):
                # Minesweeper style
                jumanji_env._viewer = viewer_class(
                    name=name,
                    num_rows=jumanji_env._viewer._num_rows,
                    num_cols=jumanji_env._viewer._num_cols,
                    render_mode=render_mode,
                )
            elif hasattr(jumanji_env._viewer, "_cube_size"):
                # RubiksCube style
                jumanji_env._viewer = viewer_class(
                    name=name,
                    cube_size=jumanji_env._viewer._cube_size,
                    render_mode=render_mode,
                )
            else:
                # Generic - try just name and render_mode
                jumanji_env._viewer = viewer_class(
                    name=name,
                    render_mode=render_mode,
                )
            LOGGER.debug(f"Created {viewer_class.__name__} with render_mode={render_mode}")
        except Exception as e:
            LOGGER.warning(f"Could not create viewer with rgb_array mode: {e}")

    return jumanji_env


def make_jumanji_gym_env(
    env_id: str,
    seed: int = 0,
    render_mode: Optional[str] = "rgb_array",
    flatten_obs: bool = False,
    backend: Optional[str] = None,
) -> gym.Env:
    """Factory function to create Gymnasium-wrapped Jumanji environment.

    This function uses Jumanji's built-in JumanjiToGymWrapper to bridge
    the JAX-based stateless environment to Gymnasium's stateful API.

    Args:
        env_id: Jumanji environment ID (e.g., "Game2048-v1")
        seed: Random seed for JAX PRNG
        render_mode: Render mode for viewer - "rgb_array" for GUI embedding,
            "human" for popup window. Defaults to "rgb_array".
        flatten_obs: If True, wrap with FlattenObservationWrapper
        backend: JAX backend ("cpu", "gpu", "tpu") or None for auto

    Returns:
        Gymnasium-compatible environment

    Raises:
        ImportError: If JAX or Jumanji not installed
        ValueError: If env_id not in supported environments

    Example:
        # Basic usage (returns RGB arrays for GUI)
        env = make_jumanji_gym_env("Game2048-v1", seed=42)

        # With flattened observations (for CleanRL, etc.)
        env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)

        # Force CPU backend
        env = make_jumanji_gym_env("RubiksCube-v0", backend="cpu")
    """
    if not _ensure_jumanji_imported():
        raise ImportError(
            "JAX and Jumanji are required for Jumanji environments. "
            "Install with: pip install jax jaxlib jumanji"
        )

    if env_id not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            f"env_id must be one of {sorted(SUPPORTED_ENVIRONMENTS)}, "
            f"got '{env_id}'"
        )

    # Create environment with proper viewer render mode
    jumanji_env = _create_jumanji_env_with_viewer(
        env_id, render_mode=render_mode or "rgb_array"
    )

    # Wrap with Jumanji's built-in Gymnasium wrapper
    # This handles: state management, PRNG keys, JIT compilation, pytree conversion
    env = _JumanjiToGymWrapper(jumanji_env, seed=seed, backend=backend)

    # Optionally flatten observations
    if flatten_obs:
        env = FlattenObservationWrapper(env)

    # Store render_mode for compatibility
    env.render_mode = render_mode

    return env


def register_jumanji_envs() -> None:
    """Register Jumanji environments with Gymnasium.

    This function registers all supported Jumanji environments with the
    Gymnasium registry under the "jumanji/" namespace.

    After calling this, environments can be created with:
        gymnasium.make("jumanji/Game2048-v1")

    Note:
        Registration happens lazily - the actual environment creation
        (and JAX import) only occurs when gymnasium.make() is called.
    """
    for env_id in SUPPORTED_ENVIRONMENTS:
        gym_id = f"jumanji/{env_id}"

        # Check if already registered
        try:
            gym.spec(gym_id)
            continue  # Already registered
        except gym.error.NameNotFound:
            pass

        # Register environment with factory function
        gym.register(
            id=gym_id,
            entry_point="jumanji_worker.gymnasium_adapter:make_jumanji_gym_env",
            kwargs={"env_id": env_id},
        )

    LOGGER.debug(f"Registered {len(SUPPORTED_ENVIRONMENTS)} Jumanji environments with Gymnasium")


# Backward compatibility: expose wrapper class name (delegates to Jumanji's implementation)
def JumanjiGymnasiumEnv(
    env_id: str,
    seed: int = 0,
    render_mode: Optional[str] = None,
    flatten_obs: bool = False,
    backend: Optional[str] = None,
) -> gym.Env:
    """Create a Gymnasium-wrapped Jumanji environment.

    This is an alias for make_jumanji_gym_env() for backward compatibility.

    Deprecated:
        Use make_jumanji_gym_env() instead.
    """
    LOGGER.warning(
        "JumanjiGymnasiumEnv is deprecated. Use make_jumanji_gym_env() instead."
    )
    return make_jumanji_gym_env(
        env_id=env_id,
        seed=seed,
        render_mode=render_mode,
        flatten_obs=flatten_obs,
        backend=backend,
    )


# Auto-register on import for convenience
try:
    register_jumanji_envs()
except Exception as e:
    LOGGER.warning(f"Failed to auto-register Jumanji environments: {e}")


__all__ = [
    "make_jumanji_gym_env",
    "register_jumanji_envs",
    "FlattenObservationWrapper",
    # Backward compatibility
    "JumanjiGymnasiumEnv",
]
