"""Tests for Jumanji Gymnasium adapter.

This test suite verifies the Gymnasium bridge that allows Jumanji environments
to be used by CleanRL, Ray, and XuanCe workers through the standard Gymnasium API.

The adapter uses Jumanji's built-in JumanjiToGymWrapper rather than reinventing
the wheel, following the pattern established by Shimmy and SuperSuit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jumanji_worker.config import LOGIC_ENVIRONMENTS


# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    import jumanji
    from jumanji.wrappers import JumanjiToGymWrapper
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX/Jumanji not installed")
class TestGymnasiumAdapterIntegration:
    """Integration tests for Gymnasium adapter (requires JAX)."""

    def test_jax_available(self):
        """Verify JAX is properly installed."""
        assert HAS_JAX
        assert jax is not None
        assert jnp is not None
        assert jumanji is not None

    def test_uses_jumanji_builtin_wrapper(self):
        """Verify we use Jumanji's built-in JumanjiToGymWrapper."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        env = make_jumanji_gym_env("Game2048-v1", seed=42)

        # The base environment should be JumanjiToGymWrapper
        assert isinstance(env, JumanjiToGymWrapper)
        env.close()

    def test_make_jumanji_gym_env_import(self):
        """make_jumanji_gym_env factory should be importable."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        assert callable(make_jumanji_gym_env)

    def test_register_jumanji_envs_import(self):
        """register_jumanji_envs should be importable."""
        from jumanji_worker.gymnasium_adapter import register_jumanji_envs
        assert callable(register_jumanji_envs)

    def test_flatten_observation_wrapper_import(self):
        """FlattenObservationWrapper should be importable."""
        from jumanji_worker.gymnasium_adapter import FlattenObservationWrapper
        assert FlattenObservationWrapper is not None

    def test_create_game2048_env(self):
        """Should create Game2048 environment."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        env = make_jumanji_gym_env("Game2048-v1", seed=42)
        assert env is not None

        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

        env.close()

    def test_env_has_action_space(self):
        """Environment should have action_space attribute."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import gymnasium as gym

        env = make_jumanji_gym_env("Game2048-v1", seed=42)
        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, gym.spaces.Space)
        env.close()

    def test_env_has_observation_space(self):
        """Environment should have observation_space attribute."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import gymnasium as gym

        env = make_jumanji_gym_env("Game2048-v1", seed=42)
        assert hasattr(env, "observation_space")
        assert isinstance(env.observation_space, gym.spaces.Space)
        env.close()

    def test_step_returns_five_values(self):
        """Step should return (obs, reward, terminated, truncated, info)."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        env = make_jumanji_gym_env("Game2048-v1", seed=42)
        env.reset()

        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_reset_with_seed(self):
        """Reset should accept seed parameter."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        env = make_jumanji_gym_env("Game2048-v1", seed=42)

        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        # Same seed should give same observation
        import numpy as np
        if isinstance(obs1, dict):
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
        else:
            np.testing.assert_array_equal(obs1, obs2)

        env.close()

    def test_flatten_obs_option(self):
        """flatten_obs=True should return flat numpy array."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import numpy as np

        env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)
        obs, _ = env.reset()

        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1  # Flattened to 1D

        env.close()

    def test_flatten_obs_uses_wrapper(self):
        """flatten_obs=True should use FlattenObservationWrapper."""
        from jumanji_worker.gymnasium_adapter import (
            make_jumanji_gym_env,
            FlattenObservationWrapper,
        )

        env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)

        # Should be wrapped with FlattenObservationWrapper
        assert isinstance(env, FlattenObservationWrapper)
        env.close()

    def test_all_logic_environments_work(self):
        """All logic environments should be creatable."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        for env_id in LOGIC_ENVIRONMENTS:
            try:
                env = make_jumanji_gym_env(env_id, seed=42)
                obs, info = env.reset()
                assert obs is not None
                env.close()
            except Exception as e:
                pytest.fail(f"Failed to create {env_id}: {e}")

    def test_gymnasium_registration(self):
        """Environments should be registered with Gymnasium."""
        from jumanji_worker.gymnasium_adapter import register_jumanji_envs
        import gymnasium as gym

        register_jumanji_envs()

        # Check if at least one environment is registered
        for env_id in LOGIC_ENVIRONMENTS:
            gym_id = f"jumanji/{env_id}"
            try:
                spec = gym.spec(gym_id)
                assert spec is not None
                break
            except gym.error.NameNotFound:
                continue
        else:
            pytest.fail("No Jumanji environments registered with Gymnasium")

    def test_gymnasium_make(self):
        """gymnasium.make() should work with registered environments."""
        from jumanji_worker.gymnasium_adapter import register_jumanji_envs
        import gymnasium as gym

        register_jumanji_envs()

        env = gym.make("jumanji/Game2048-v1")
        assert env is not None

        obs, info = env.reset()
        assert obs is not None

        env.close()


class TestGymnasiumAdapterNoJax:
    """Tests that don't require JAX (module structure tests)."""

    def test_config_imports(self):
        """Config module should import without JAX."""
        from jumanji_worker.config import (
            JumanjiWorkerConfig,
            load_worker_config,
            LOGIC_ENVIRONMENTS,
        )
        assert JumanjiWorkerConfig is not None
        assert callable(load_worker_config)
        assert len(LOGIC_ENVIRONMENTS) > 0

    def test_logic_environments_defined(self):
        """LOGIC_ENVIRONMENTS should contain expected environments."""
        expected = {
            "Game2048-v1",
            "Minesweeper-v0",
            "RubiksCube-v0",
            "Sudoku-v0",
        }
        assert expected.issubset(LOGIC_ENVIRONMENTS)

    def test_adapter_handles_missing_jax(self):
        """Adapter should handle missing JAX gracefully."""
        # Import should not fail even without JAX
        try:
            from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
            # If JAX is missing, creating an env should raise ImportError
            if not HAS_JAX:
                with pytest.raises(ImportError, match="JAX"):
                    make_jumanji_gym_env("Game2048-v1")
        except ImportError:
            # Expected if JAX is not installed
            pass

    def test_invalid_env_id_raises(self):
        """Invalid env_id should raise ValueError."""
        if not HAS_JAX:
            pytest.skip("JAX not installed")

        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env

        with pytest.raises(ValueError, match="env_id must be one of"):
            make_jumanji_gym_env("InvalidEnv-v999")

    def test_backward_compat_alias_exists(self):
        """JumanjiGymnasiumEnv should exist for backward compatibility."""
        from jumanji_worker.gymnasium_adapter import JumanjiGymnasiumEnv
        assert callable(JumanjiGymnasiumEnv)


@pytest.mark.skipif(not HAS_JAX, reason="JAX/Jumanji not installed")
class TestCrossWorkerCompatibility:
    """Tests verifying compatibility with other MOSAIC workers."""

    def test_cleanrl_compatible_env(self):
        """Environment should work with CleanRL's expected interface."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import numpy as np

        env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)

        # CleanRL expects:
        # 1. reset() returns (obs, info) tuple
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

        # 2. step() returns 5 values
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # 3. action_space.sample() works
        action = env.action_space.sample()
        assert action is not None

        # 4. observation_space exists
        assert env.observation_space is not None

        env.close()

    def test_ray_compatible_env(self):
        """Environment should work with Ray RLlib's expected interface."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import gymnasium as gym

        env = make_jumanji_gym_env("Game2048-v1", seed=42)

        # Ray RLlib expects standard Gymnasium interface
        assert isinstance(env, gym.Env)
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "close")

        env.close()

    def test_xuance_compatible_env(self):
        """Environment should work with XuanCe's expected interface."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import gymnasium as gym

        env = make_jumanji_gym_env("Game2048-v1", seed=42)

        # XuanCe expects standard Gymnasium interface
        assert isinstance(env, gym.Env)

        # XuanCe may use unwrapped
        assert hasattr(env, "unwrapped")

        # Should support render
        assert hasattr(env, "render")

        env.close()


@pytest.mark.skipif(not HAS_JAX, reason="JAX/Jumanji not installed")
class TestFlattenObservationWrapper:
    """Tests for the FlattenObservationWrapper."""

    def test_flattens_dict_observation(self):
        """Should flatten Dict observations to 1D array."""
        from jumanji_worker.gymnasium_adapter import FlattenObservationWrapper, make_jumanji_gym_env
        import numpy as np

        # Create env without flattening
        base_env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=False)
        obs_before, _ = base_env.reset()

        # Wrap with flattener
        flat_env = FlattenObservationWrapper(base_env)
        obs_after, _ = flat_env.reset()

        # Output should be 1D numpy array
        assert isinstance(obs_after, np.ndarray)
        assert len(obs_after.shape) == 1
        assert obs_after.dtype == np.float32

        flat_env.close()

    def test_observation_space_is_box(self):
        """Flattened observation space should be Box."""
        from jumanji_worker.gymnasium_adapter import FlattenObservationWrapper, make_jumanji_gym_env
        import gymnasium as gym

        base_env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=False)
        flat_env = FlattenObservationWrapper(base_env)

        assert isinstance(flat_env.observation_space, gym.spaces.Box)
        assert len(flat_env.observation_space.shape) == 1

        flat_env.close()

    def test_step_also_flattens(self):
        """Step should also return flattened observations."""
        from jumanji_worker.gymnasium_adapter import make_jumanji_gym_env
        import numpy as np

        env = make_jumanji_gym_env("Game2048-v1", seed=42, flatten_obs=True)
        env.reset()

        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert len(obs.shape) == 1

        env.close()
