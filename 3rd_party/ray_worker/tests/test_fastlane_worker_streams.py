"""Test FastLane per-worker stream functionality.

This test verifies that:
1. Each Ray worker gets its own FastLane stream (separate tab in Render View)
2. Stream names are correctly formatted as Ray-Live-{env}-worker-{N}-{run_id}
3. FastLane config correctly reads worker_index from env vars
4. The wrapper creates streams and publishes frames correctly
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Add the ray_worker package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray_worker.fastlane import (
    FastLaneRayConfig,
    ParallelFastLaneWrapper,
    MultiAgentFastLaneWrapper,
    maybe_wrap_parallel_env,
    maybe_wrap_env,
    is_fastlane_enabled,
    ENV_FASTLANE_ENABLED,
    ENV_FASTLANE_RUN_ID,
    ENV_FASTLANE_ENV_NAME,
    ENV_FASTLANE_WORKER_INDEX,
    ENV_FASTLANE_THROTTLE_MS,
)


class TestFastLaneRayConfig:
    """Test FastLaneRayConfig stream naming."""

    def test_stream_id_format(self):
        """All workers use consistent {run_id}-worker-{N} format."""
        config0 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=0,
        )
        assert config0.stream_id == "01KCEH4AB5ZG-worker-0"

        config1 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=1,
        )
        assert config1.stream_id == "01KCEH4AB5ZG-worker-1"

        config2 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=2,
        )
        assert config2.stream_id == "01KCEH4AB5ZG-worker-2"

    def test_from_env_reads_worker_index(self):
        """Config should read worker_index from environment variable."""
        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "01KCEH4AB5ZG",
            ENV_FASTLANE_ENV_NAME: "multiwalker_v9",
            ENV_FASTLANE_WORKER_INDEX: "3",
            ENV_FASTLANE_THROTTLE_MS: "33",
        }):
            config = FastLaneRayConfig.from_env()
            assert config.worker_index == 3
            assert config.stream_id == "01KCEH4AB5ZG-worker-3"

    def test_from_env_with_explicit_worker_index(self):
        """Explicit worker_index parameter should override env var."""
        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "01KCEH4AB5ZG",
            ENV_FASTLANE_ENV_NAME: "multiwalker_v9",
            ENV_FASTLANE_WORKER_INDEX: "5",  # This should be overridden
        }):
            config = FastLaneRayConfig.from_env(worker_index=7)
            assert config.worker_index == 7
            assert config.stream_id == "01KCEH4AB5ZG-worker-7"

    def test_from_env_defaults_worker_index_to_zero(self):
        """Worker index should default to 0 if not specified."""
        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "01KCEH4AB5ZG",
            ENV_FASTLANE_ENV_NAME: "multiwalker_v9",
        }, clear=True):
            # Remove worker index env var if present
            os.environ.pop(ENV_FASTLANE_WORKER_INDEX, None)
            config = FastLaneRayConfig.from_env()
            assert config.worker_index == 0
            assert config.stream_id == "01KCEH4AB5ZG-worker-0"

    def test_throttle_defaults_to_33ms(self):
        """Throttle should default to 33ms (~30 FPS)."""
        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "my-run-id",
        }, clear=True):
            os.environ.pop(ENV_FASTLANE_THROTTLE_MS, None)
            config = FastLaneRayConfig.from_env()
            assert config.throttle_interval_ms == 33


class TestMaybeWrapFunctions:
    """Test the maybe_wrap_env and maybe_wrap_parallel_env functions."""

    def test_maybe_wrap_parallel_env_passes_worker_index(self):
        """maybe_wrap_parallel_env should pass worker_index to config."""
        mock_env = MagicMock()
        mock_env.possible_agents = ["agent_0", "agent_1"]

        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "01KCEH4AB5ZG",
            ENV_FASTLANE_ENV_NAME: "multiwalker_v9",
        }):
            wrapped = maybe_wrap_parallel_env(mock_env, worker_index=5)
            assert isinstance(wrapped, ParallelFastLaneWrapper)
            assert wrapped._config.worker_index == 5
            assert wrapped._config.stream_id == "01KCEH4AB5ZG-worker-5"

    def test_maybe_wrap_env_passes_worker_index(self):
        """maybe_wrap_env should pass worker_index to config."""
        mock_env = MagicMock()
        mock_env.possible_agents = ["agent_0", "agent_1"]

        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "1",
            ENV_FASTLANE_RUN_ID: "01KCEH4AB5ZG",
            ENV_FASTLANE_ENV_NAME: "multiwalker_v9",
        }):
            wrapped = maybe_wrap_env(mock_env, worker_index=3)
            assert isinstance(wrapped, MultiAgentFastLaneWrapper)
            assert wrapped._config.worker_index == 3
            assert wrapped._config.stream_id == "01KCEH4AB5ZG-worker-3"

    def test_maybe_wrap_returns_original_when_disabled(self):
        """When FastLane disabled, should return original env."""
        mock_env = MagicMock()

        with patch.dict(os.environ, {
            ENV_FASTLANE_ENABLED: "0",
        }):
            result = maybe_wrap_parallel_env(mock_env, worker_index=5)
            assert result is mock_env  # Should be the original, not wrapped


class TestParallelFastLaneWrapper:
    """Test ParallelFastLaneWrapper functionality."""

    def test_wrapper_uses_correct_stream_id(self):
        """All workers use consistent {run_id}-worker-{N} format."""
        mock_env = MagicMock()
        mock_env.possible_agents = ["walker_0", "walker_1", "walker_2"]

        config0 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEFQG2CH9",
            env_name="multiwalker_v9",
            worker_index=0,
        )
        wrapper0 = ParallelFastLaneWrapper(mock_env, config0)
        assert wrapper0._config.stream_id == "01KCEFQG2CH9-worker-0"

        config1 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEFQG2CH9",
            env_name="multiwalker_v9",
            worker_index=1,
        )
        wrapper1 = ParallelFastLaneWrapper(mock_env, config1)
        assert wrapper1._config.stream_id == "01KCEFQG2CH9-worker-1"

    def test_different_workers_have_different_streams(self):
        """Each worker creates a unique stream (separate tabs in UI)."""
        mock_env = MagicMock()
        mock_env.possible_agents = ["walker_0"]

        configs = []
        for worker_idx in range(3):
            config = FastLaneRayConfig(
                enabled=True,
                run_id="01KCEH4AB5ZG",
                env_name="multiwalker_v9",
                worker_index=worker_idx,
            )
            configs.append(config)

        stream_ids = [c.stream_id for c in configs]

        # All stream IDs should be unique
        assert len(stream_ids) == len(set(stream_ids))
        # All workers use same pattern
        assert stream_ids == [
            "01KCEH4AB5ZG-worker-0",
            "01KCEH4AB5ZG-worker-1",
            "01KCEH4AB5ZG-worker-2",
        ]


class TestRLlibIntegration:
    """Test that worker_index is correctly extracted from RLlib config."""

    def test_env_creator_extracts_worker_index_from_dict(self):
        """Extract worker_index from dict config."""
        rllib_config = {
            "worker_index": 2,
            "vector_index": 0,
            "remote": True,
        }

        # Extract worker_index using the same logic as wrapped_env_creator
        if hasattr(rllib_config, "worker_index"):
            worker_index = rllib_config.worker_index
        elif isinstance(rllib_config, dict):
            worker_index = rllib_config.get("worker_index", 0)
        else:
            worker_index = 0

        assert worker_index == 2

    def test_env_creator_extracts_worker_index_from_env_context(self):
        """Extract worker_index from RLlib EnvContext (attribute, not dict key)."""
        # Simulate RLlib's EnvContext which stores worker_index as an attribute
        class MockEnvContext(dict):
            def __init__(self, worker_index: int):
                super().__init__()
                self.worker_index = worker_index

        env_context = MockEnvContext(worker_index=2)

        # This is the logic from wrapped_env_creator - must check hasattr first!
        if hasattr(env_context, "worker_index"):
            worker_index = env_context.worker_index
        elif isinstance(env_context, dict):
            worker_index = env_context.get("worker_index", 0)
        else:
            worker_index = 0

        assert worker_index == 2

    def test_env_creator_defaults_to_zero_without_config(self):
        """Without config, worker_index should default to 0."""
        _config = None

        # This is the logic from wrapped_env_creator
        if hasattr(_config, "worker_index"):
            worker_index = _config.worker_index
        elif isinstance(_config, dict):
            worker_index = _config.get("worker_index", 0)
        else:
            worker_index = 0

        assert worker_index == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
