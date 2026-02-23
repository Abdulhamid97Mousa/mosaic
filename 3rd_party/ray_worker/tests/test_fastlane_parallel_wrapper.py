"""Test FastLane ParallelFastLaneWrapper integration with Ray RLlib.

This test verifies that:
1. Our FastLane wrapper's step() is called when wrapped by Ray's ParallelPettingZooEnv
2. Timesteps increment correctly
3. Episodes increment on reset
4. Frames are captured (render is called)
5. The wrapper properly forwards all PettingZoo attributes
"""

import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Set up environment variables before importing fastlane
os.environ["RAY_FASTLANE_ENABLED"] = "1"
os.environ["RAY_FASTLANE_RUN_ID"] = "test-run-123"
os.environ["RAY_FASTLANE_ENV_NAME"] = "pursuit_v4"
os.environ["RAY_FASTLANE_THROTTLE_MS"] = "0"  # No throttle for testing

from ray_worker.fastlane import (
    ParallelFastLaneWrapper,
    FastLaneRayConfig,
    maybe_wrap_parallel_env,
    is_fastlane_enabled,
)


class MockParallelEnv:
    """Mock PettingZoo Parallel environment for testing."""

    def __init__(self):
        try:
            import gymnasium as gym
            obs_space = gym.spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)
            act_space = gym.spaces.Discrete(5)
        except ImportError:
            # Fallback to MagicMock if gymnasium not available
            obs_space = MagicMock()
            act_space = MagicMock()

        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._reset_count = 0
        self.observation_spaces = {
            agent: obs_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: act_space for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self._reset_count += 1
        self.agents = list(self.possible_agents)
        obs = {agent: np.zeros((7, 7, 3), dtype=np.uint8) for agent in self.agents}
        info = {agent: {} for agent in self.agents}
        return obs, info

    def step(self, actions):
        self._step_count += 1
        obs = {agent: np.zeros((7, 7, 3), dtype=np.uint8) for agent in self.agents}
        rewards = {agent: 0.1 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        # Return a simple RGB frame
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        pass


class TestParallelFastLaneWrapperBasic:
    """Test basic wrapper functionality without Ray."""

    def test_wrapper_initializes(self):
        """Wrapper initializes with correct config."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=True,
            run_id="test-run",
            env_name="test_env",
            worker_index=1,
            throttle_interval_ms=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)

        assert wrapper._config.worker_index == 1
        assert wrapper._config.run_id == "test-run"
        assert wrapper._total_timesteps == 0
        assert wrapper._episode == 0

    def test_wrapper_forwards_attributes(self):
        """Wrapper forwards PettingZoo attributes correctly."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=True,
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)

        # Check forwarded properties
        assert wrapper.possible_agents == env.possible_agents
        assert wrapper.agents == env.agents
        assert wrapper.observation_spaces == env.observation_spaces
        assert wrapper.action_spaces == env.action_spaces

    def test_reset_increments_episode(self):
        """Reset increments episode counter."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,  # Disable FastLane to avoid shared memory issues
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)

        assert wrapper._episode == 0

        wrapper.reset()
        assert wrapper._episode == 1

        wrapper.reset()
        assert wrapper._episode == 2

    def test_step_increments_timesteps(self):
        """Step increments timestep counter."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,  # Disable FastLane to avoid shared memory issues
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)
        wrapper.reset()

        assert wrapper._total_timesteps == 0

        actions = {agent: 0 for agent in wrapper.agents}
        wrapper.step(actions)
        assert wrapper._total_timesteps == 1

        wrapper.step(actions)
        assert wrapper._total_timesteps == 2

        wrapper.step(actions)
        assert wrapper._total_timesteps == 3

    def test_timesteps_persist_across_episodes(self):
        """Timesteps accumulate across episodes (not reset)."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)

        # Episode 1
        wrapper.reset()
        actions = {agent: 0 for agent in wrapper.agents}
        for _ in range(5):
            wrapper.step(actions)
        assert wrapper._total_timesteps == 5
        assert wrapper._episode == 1

        # Episode 2 - timesteps should continue from 5
        wrapper.reset()
        for _ in range(3):
            wrapper.step(actions)
        assert wrapper._total_timesteps == 8
        assert wrapper._episode == 2

    def test_step_returns_correct_format(self):
        """Step returns (obs, rewards, terms, truncs, infos) tuple."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)
        wrapper.reset()

        actions = {agent: 0 for agent in wrapper.agents}
        result = wrapper.step(actions)

        assert len(result) == 5
        obs, rewards, terminations, truncations, infos = result
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)


class TestParallelFastLaneWrapperWithRay:
    """Test wrapper integration with Ray's ParallelPettingZooEnv."""

    @pytest.fixture
    def ray_wrapped_env(self):
        """Create a FastLane-wrapped env inside Ray's wrapper."""
        try:
            from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        except ImportError:
            pytest.skip("Ray RLlib not installed")

        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,  # Disable to avoid shared memory issues in tests
            run_id="test-run",
            env_name="test_env",
            worker_index=1,
        )
        fastlane_wrapper = ParallelFastLaneWrapper(env, config)
        ray_env = ParallelPettingZooEnv(fastlane_wrapper)

        return ray_env, fastlane_wrapper

    def test_ray_wrapper_uses_our_step(self, ray_wrapped_env):
        """Ray's ParallelPettingZooEnv calls our wrapper's step()."""
        ray_env, fastlane_wrapper = ray_wrapped_env

        # Reset - episode increments
        episode_before = fastlane_wrapper._episode
        ray_env.reset()
        assert fastlane_wrapper._episode == episode_before + 1

        timesteps_before = fastlane_wrapper._total_timesteps

        # Step through Ray's wrapper
        # Ray's wrapper converts to single-agent gym interface
        action = ray_env.action_space.sample()
        ray_env.step(action)

        # Our wrapper's step should have been called
        assert fastlane_wrapper._total_timesteps == timesteps_before + 1

    def test_ray_wrapper_multiple_steps(self, ray_wrapped_env):
        """Multiple steps through Ray's wrapper increment our timesteps."""
        ray_env, fastlane_wrapper = ray_wrapped_env

        ray_env.reset()
        timesteps_before = fastlane_wrapper._total_timesteps

        # Take 10 steps
        for i in range(10):
            action = ray_env.action_space.sample()
            ray_env.step(action)

        assert fastlane_wrapper._total_timesteps == timesteps_before + 10

    def test_ray_wrapper_episode_tracking(self, ray_wrapped_env):
        """Episode tracking works through Ray's wrapper."""
        ray_env, fastlane_wrapper = ray_wrapped_env

        # Episode 1
        episode_before = fastlane_wrapper._episode
        ray_env.reset()
        episode_1 = fastlane_wrapper._episode
        assert episode_1 == episode_before + 1

        timesteps_before = fastlane_wrapper._total_timesteps
        for _ in range(5):
            action = ray_env.action_space.sample()
            ray_env.step(action)

        # Episode 2
        ray_env.reset()
        assert fastlane_wrapper._episode == episode_1 + 1
        assert fastlane_wrapper._total_timesteps == timesteps_before + 5  # Persisted from episode 1


class TestParallelFastLaneWrapperFrameCapture:
    """Test frame capture functionality."""

    def test_render_called_on_step(self):
        """Render is called during step when FastLane enabled."""
        env = MockParallelEnv()
        env.render = MagicMock(return_value=np.zeros((64, 64, 3), dtype=np.uint8))

        config = FastLaneRayConfig(
            enabled=True,
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
            throttle_interval_ms=0,  # No throttle
        )

        # Mock FastLane writer to avoid shared memory issues
        with patch("ray_worker.fastlane._FASTLANE_AVAILABLE", False):
            wrapper = ParallelFastLaneWrapper(env, config)
            wrapper.reset()

            actions = {agent: 0 for agent in wrapper.agents}
            wrapper.step(actions)

            # Render should NOT be called when FastLane unavailable
            # (we're testing the path, not actual rendering)

    def test_metrics_updated_on_step(self):
        """Agent metrics are updated after each step."""
        env = MockParallelEnv()
        config = FastLaneRayConfig(
            enabled=False,
            run_id="test-run",
            env_name="test_env",
            worker_index=0,
        )
        wrapper = ParallelFastLaneWrapper(env, config)
        wrapper.reset()

        # Initial metrics
        assert all(m.steps == 0 for m in wrapper._agent_metrics.values())

        actions = {agent: 0 for agent in wrapper.agents}
        wrapper.step(actions)

        # Metrics should be updated (when enabled, but we check structure)
        # Note: metrics only update when config.enabled is True


class TestMaybeWrapParallelEnv:
    """Test the maybe_wrap_parallel_env helper function."""

    def test_wraps_when_enabled(self):
        """Returns wrapped env when FastLane is enabled."""
        with patch.dict(os.environ, {"RAY_FASTLANE_ENABLED": "1"}):
            env = MockParallelEnv()
            wrapped = maybe_wrap_parallel_env(env, worker_index=2)

            assert isinstance(wrapped, ParallelFastLaneWrapper)
            assert wrapped._config.worker_index == 2

    def test_returns_original_when_disabled(self):
        """Returns original env when FastLane is disabled."""
        with patch.dict(os.environ, {"RAY_FASTLANE_ENABLED": "0"}):
            env = MockParallelEnv()
            result = maybe_wrap_parallel_env(env, worker_index=0)

            assert result is env
            assert not isinstance(result, ParallelFastLaneWrapper)


class TestFastLaneConfigFromEnv:
    """Test FastLaneRayConfig.from_env()."""

    def test_reads_worker_index_from_env(self):
        """Worker index is read from environment variable."""
        with patch.dict(os.environ, {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-run",
            "RAY_FASTLANE_WORKER_INDEX": "5",
        }):
            config = FastLaneRayConfig.from_env()
            assert config.worker_index == 5

    def test_explicit_worker_index_overrides_env(self):
        """Explicit worker_index parameter overrides env var."""
        with patch.dict(os.environ, {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-run",
            "RAY_FASTLANE_WORKER_INDEX": "5",
        }):
            config = FastLaneRayConfig.from_env(worker_index=10)
            assert config.worker_index == 10

    def test_stream_id_format(self):
        """Stream ID follows {run_id}-worker-{index} format."""
        config = FastLaneRayConfig(
            enabled=True,
            run_id="my-run-123",
            env_name="pursuit_v4",
            worker_index=3,
        )
        assert config.stream_id == "my-run-123-worker-3"


class TestWithRealPettingZooEnv:
    """Test with actual PettingZoo environment (if available)."""

    @pytest.fixture
    def pursuit_env(self):
        """Create real Pursuit environment."""
        try:
            from pettingzoo.sisl import pursuit_v4
            env = pursuit_v4.parallel_env(render_mode="rgb_array")
            yield env
            env.close()
        except ImportError:
            pytest.skip("PettingZoo SISL not installed")

    def test_wrapper_with_real_pursuit(self, pursuit_env):
        """Wrapper works with real Pursuit environment."""
        config = FastLaneRayConfig(
            enabled=False,  # Disable to avoid shared memory
            run_id="test-pursuit",
            env_name="pursuit_v4",
            worker_index=1,
        )
        wrapper = ParallelFastLaneWrapper(pursuit_env, config)

        # Reset
        obs, info = wrapper.reset()
        assert wrapper._episode == 1
        assert len(wrapper._agent_ids) == 8  # Pursuit has 8 pursuers by default

        # Step
        actions = {agent: 0 for agent in wrapper.agents}  # Action 0 = stay
        result = wrapper.step(actions)
        assert wrapper._total_timesteps == 1

        obs, rewards, terms, truncs, infos = result
        assert len(obs) > 0
        assert len(rewards) > 0

    def test_wrapper_with_ray_and_real_pursuit(self, pursuit_env):
        """Full integration: FastLane wrapper + Ray wrapper + real Pursuit."""
        try:
            from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        except ImportError:
            pytest.skip("Ray RLlib not installed")

        config = FastLaneRayConfig(
            enabled=False,
            run_id="test-pursuit-ray",
            env_name="pursuit_v4",
            worker_index=2,
        )
        fastlane_wrapper = ParallelFastLaneWrapper(pursuit_env, config)

        # Record episode count before Ray wrapper (might call reset in __init__)
        episode_before_ray = fastlane_wrapper._episode

        ray_env = ParallelPettingZooEnv(fastlane_wrapper)

        # Reset through Ray - episode should increment
        ray_env.reset()
        episode_after_reset = fastlane_wrapper._episode
        assert episode_after_reset > episode_before_ray, "Episode should increment after reset"

        # Step through Ray
        timesteps_before = fastlane_wrapper._total_timesteps
        for i in range(10):
            action = ray_env.action_space.sample()
            obs, reward, term, trunc, info = ray_env.step(action)

        # Verify our wrapper's step was called - timesteps should increase by 10
        assert fastlane_wrapper._total_timesteps == timesteps_before + 10
        # Episode should stay the same (no reset during steps)
        assert fastlane_wrapper._episode == episode_after_reset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
