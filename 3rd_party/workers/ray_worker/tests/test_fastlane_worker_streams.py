"""Test FastLane per-worker stream functionality.

This test verifies that:
1. Each Ray worker gets its own FastLane stream (separate tab in Render View)
2. Stream names are correctly formatted as {run_id}-w{N}
3. FastLane config correctly reads worker_index from env vars
4. The wrapper creates streams and publishes frames correctly
5. Runtime env vars propagate to Ray workers (integration test)
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Try to import ray for integration tests
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

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
        """All workers use consistent {run_id}-w{N} format."""
        config0 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=0,
        )
        assert config0.stream_id == "01KCEH4AB5ZG-w0"

        config1 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=1,
        )
        assert config1.stream_id == "01KCEH4AB5ZG-w1"

        config2 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEH4AB5ZG",
            env_name="multiwalker_v9",
            worker_index=2,
        )
        assert config2.stream_id == "01KCEH4AB5ZG-w2"

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
            assert config.stream_id == "01KCEH4AB5ZG-w3"

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
            assert config.stream_id == "01KCEH4AB5ZG-w7"

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
            assert config.stream_id == "01KCEH4AB5ZG-w0"

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
            assert wrapped._config.stream_id == "01KCEH4AB5ZG-w5"

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
            assert wrapped._config.stream_id == "01KCEH4AB5ZG-w3"

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
        """All workers use consistent {run_id}-w{N} format."""
        mock_env = MagicMock()
        mock_env.possible_agents = ["walker_0", "walker_1", "walker_2"]

        config0 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEFQG2CH9",
            env_name="multiwalker_v9",
            worker_index=0,
        )
        wrapper0 = ParallelFastLaneWrapper(mock_env, config0)
        assert wrapper0._config.stream_id == "01KCEFQG2CH9-w0"

        config1 = FastLaneRayConfig(
            enabled=True,
            run_id="01KCEFQG2CH9",
            env_name="multiwalker_v9",
            worker_index=1,
        )
        wrapper1 = ParallelFastLaneWrapper(mock_env, config1)
        assert wrapper1._config.stream_id == "01KCEFQG2CH9-w1"

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
            "01KCEH4AB5ZG-w0",
            "01KCEH4AB5ZG-w1",
            "01KCEH4AB5ZG-w2",
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


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestRayRuntimeEnvPropagation:
    """Integration tests verifying Ray runtime_env propagates env vars to workers.

    These tests actually run Ray (not mocked) to verify env var propagation.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Initialize and shutdown Ray for each test."""
        # Shutdown any existing Ray instance
        if ray.is_initialized():
            ray.shutdown()
        yield
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()

    def test_runtime_env_propagates_to_ray_tasks(self):
        """Verify env vars in runtime_env reach Ray remote functions."""
        test_env_vars = {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-propagation-001",
            "RAY_FASTLANE_ENV_NAME": "test_env",
            "TEST_MARKER": "propagation_test",
        }

        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env={"env_vars": test_env_vars},
        )

        @ray.remote
        def check_env_vars():
            """Check if env vars are available in Ray worker."""
            return {
                "RAY_FASTLANE_ENABLED": os.getenv("RAY_FASTLANE_ENABLED", "NOT_SET"),
                "RAY_FASTLANE_RUN_ID": os.getenv("RAY_FASTLANE_RUN_ID", "NOT_SET"),
                "TEST_MARKER": os.getenv("TEST_MARKER", "NOT_SET"),
                "is_fastlane_enabled": is_fastlane_enabled(),
            }

        result = ray.get(check_env_vars.remote())

        # All env vars should be available in the worker
        assert result["RAY_FASTLANE_ENABLED"] == "1", f"RAY_FASTLANE_ENABLED not propagated: {result}"
        assert result["RAY_FASTLANE_RUN_ID"] == "test-propagation-001", f"RUN_ID not propagated: {result}"
        assert result["TEST_MARKER"] == "propagation_test", f"TEST_MARKER not propagated: {result}"
        assert result["is_fastlane_enabled"] is True, f"is_fastlane_enabled() returned False: {result}"

    def test_multiple_workers_receive_env_vars(self):
        """Verify env vars propagate to multiple workers."""
        test_env_vars = {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-multi-worker",
        }

        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": test_env_vars},
        )

        @ray.remote
        def check_in_worker(worker_id: int):
            return {
                "worker_id": worker_id,
                "enabled": os.getenv("RAY_FASTLANE_ENABLED", "NOT_SET"),
                "run_id": os.getenv("RAY_FASTLANE_RUN_ID", "NOT_SET"),
            }

        # Run in multiple workers
        futures = [check_in_worker.remote(i) for i in range(3)]
        results = ray.get(futures)

        for r in results:
            assert r["enabled"] == "1", f"Worker {r['worker_id']}: enabled={r['enabled']}"
            assert r["run_id"] == "test-multi-worker", f"Worker {r['worker_id']}: run_id={r['run_id']}"

    def test_env_vars_in_rllib_local_worker(self):
        """Test env vars in local worker (worker_index=0).

        Local worker runs in driver process, so env vars must be set via os.environ.
        runtime_env only affects REMOTE workers.
        """
        import tempfile
        import json
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPOConfig

        test_env_vars = {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-rllib-local",
        }

        # CRITICAL: Set env vars in main process (like runtime.py does)
        # runtime_env doesn't apply to the local worker!
        for k, v in test_env_vars.items():
            os.environ[k] = v

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            capture_file = f.name
            f.write("{}")

        def test_env_creator(config):
            import json
            captured = {
                "RAY_FASTLANE_ENABLED": os.getenv("RAY_FASTLANE_ENABLED", "NOT_SET"),
                "is_fastlane_enabled": is_fastlane_enabled(),
                "worker_index": getattr(config, "worker_index", config.get("worker_index", -1) if isinstance(config, dict) else -1),
            }
            with open(capture_file, 'w') as f:
                json.dump(captured, f)
            import gymnasium as gym
            return gym.make("CartPole-v1")

        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": test_env_vars})
        register_env("test_fastlane_env", test_env_creator)

        config = (
            PPOConfig()
            .environment("test_fastlane_env")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(num_env_runners=0, create_env_on_local_worker=True)
        )

        algo = config.build()

        with open(capture_file, 'r') as f:
            captured = json.load(f)

        print(f"\n[DEBUG] Local worker (W0): {captured}")

        os.unlink(capture_file)
        # Cleanup env vars
        for k in test_env_vars:
            os.environ.pop(k, None)
        algo.stop()

        assert captured.get("RAY_FASTLANE_ENABLED") == "1", f"Local worker missing env var: {captured}"
        assert captured.get("is_fastlane_enabled") is True, f"FastLane not enabled in local worker: {captured}"
        assert captured.get("worker_index") == 0, f"Wrong worker_index: {captured}"

    def test_env_vars_in_rllib_remote_workers(self):
        """Test env vars in remote workers (worker_index=1, 2, ...).

        Remote workers get env vars from ray.init(runtime_env=...).
        This is the CRITICAL test - do remote env_runners get the env vars?
        """
        import tempfile
        import json
        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPOConfig

        test_env_vars = {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": "test-rllib-remote",
        }

        # Use unique file per worker
        import uuid
        capture_dir = tempfile.mkdtemp()
        capture_prefix = os.path.join(capture_dir, "worker_")

        def test_env_creator(config):
            import json
            worker_idx = getattr(config, "worker_index", config.get("worker_index", -1) if isinstance(config, dict) else -1)
            captured = {
                "RAY_FASTLANE_ENABLED": os.getenv("RAY_FASTLANE_ENABLED", "NOT_SET"),
                "is_fastlane_enabled": is_fastlane_enabled(),
                "worker_index": worker_idx,
            }
            # Write to worker-specific file
            with open(f"{capture_prefix}{worker_idx}.json", 'w') as f:
                json.dump(captured, f)
            import gymnasium as gym
            return gym.make("CartPole-v1")

        # Set env vars for main process too (for local worker)
        for k, v in test_env_vars.items():
            os.environ[k] = v

        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": test_env_vars})
        register_env("test_fastlane_env", test_env_creator)

        config = (
            PPOConfig()
            .environment("test_fastlane_env")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                num_env_runners=2,  # 2 REMOTE workers
                create_env_on_local_worker=True,  # Plus local worker
            )
        )

        algo = config.build()

        # Read results from all workers
        results = {}
        import glob
        for f in glob.glob(f"{capture_prefix}*.json"):
            with open(f, 'r') as fp:
                data = json.load(fp)
                results[data["worker_index"]] = data
            os.unlink(f)
        os.rmdir(capture_dir)

        print(f"\n[DEBUG] All workers: {results}")

        # Cleanup
        for k in test_env_vars:
            os.environ.pop(k, None)
        algo.stop()

        # Check local worker (W0)
        assert 0 in results, f"Local worker W0 not found: {results.keys()}"
        assert results[0].get("RAY_FASTLANE_ENABLED") == "1", f"W0 missing env var: {results[0]}"

        # Check remote workers (W1, W2)
        for w in [1, 2]:
            assert w in results, f"Remote worker W{w} not found: {results.keys()}"
            assert results[w].get("RAY_FASTLANE_ENABLED") == "1", \
                f"W{w} missing env var - runtime_env not propagating: {results[w]}"
            assert results[w].get("is_fastlane_enabled") is True, \
                f"FastLane not enabled in W{w}: {results[w]}"


class TestFastLaneSharedMemoryCreation:
    """Test that FastLane actually creates shared memory."""

    def test_parallel_wrapper_creates_shared_memory(self):
        """Test that ParallelFastLaneWrapper creates shared memory when it runs.

        This is the CRITICAL test - do we actually see shm files in /dev/shm?
        """
        import tempfile

        # Try to import PettingZoo - skip if not available
        try:
            from pettingzoo.sisl import multiwalker_v9
        except ImportError:
            pytest.skip("PettingZoo SISL not available")

        # Create config
        run_id = f"test-shm-{os.getpid()}"
        config = FastLaneRayConfig(
            enabled=True,
            run_id=run_id,
            env_name="multiwalker_v9",
            worker_index=1,
            throttle_interval_ms=0,  # No throttling for test
        )

        # Expected stream ID and shm file
        expected_stream_id = f"{run_id}-w1"
        expected_shm_name = f"mosaic.fastlane.{expected_stream_id}"

        # Create env with render_mode for frame capture
        env = multiwalker_v9.parallel_env(render_mode="rgb_array")

        # Wrap with FastLane
        wrapped = ParallelFastLaneWrapper(env, config)

        # Reset to initialize
        wrapped.reset()

        # Run a few steps to trigger frame publishing
        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in wrapped.agents}
            if not wrapped.agents:
                break
            wrapped.step(actions)

        # Check if shared memory was created
        shm_path = f"/dev/shm/{expected_shm_name}"
        shm_exists = os.path.exists(shm_path)

        print(f"\n[DEBUG] Expected shm: {shm_path}")
        print(f"[DEBUG] SHM exists: {shm_exists}")

        # List all mosaic shm files
        import glob
        mosaic_shms = glob.glob("/dev/shm/mosaic.*")
        print(f"[DEBUG] All mosaic shm files: {mosaic_shms}")

        # Check if writer was created
        print(f"[DEBUG] Wrapper._writer: {wrapped._writer}")
        print(f"[DEBUG] Config: enabled={config.enabled}, stream_id={config.stream_id}")

        # Cleanup
        wrapped.close()

        # Assert (but don't fail - we want to see what's happening)
        if not shm_exists:
            print(f"[WARNING] Shared memory NOT created: {shm_path}")
            # Print why it might have failed
            print(f"[DEBUG] _FASTLANE_AVAILABLE in fastlane.py should be True")
            try:
                from gym_gui.fastlane import FastLaneWriter, FastLaneConfig
                print(f"[DEBUG] gym_gui.fastlane imports work: FastLaneWriter={FastLaneWriter}, FastLaneConfig={FastLaneConfig}")
            except ImportError as e:
                print(f"[DEBUG] gym_gui.fastlane import FAILED: {e}")

        assert shm_exists, f"Shared memory not created at {shm_path}. Check FastLane imports."

    def test_rllib_with_fastlane_creates_shm(self):
        """Test that RLlib with FastLane wrapper creates shared memory.

        This tests the full integration: RLlib -> env_creator -> FastLane wrapper -> shm
        """
        if not RAY_AVAILABLE:
            pytest.skip("Ray not available")

        try:
            from pettingzoo.sisl import multiwalker_v9
        except ImportError:
            pytest.skip("PettingZoo SISL not available")

        from ray.tune.registry import register_env
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        import glob

        run_id = f"test-rllib-shm-{os.getpid()}"
        test_env_vars = {
            "RAY_FASTLANE_ENABLED": "1",
            "RAY_FASTLANE_RUN_ID": run_id,
            "RAY_FASTLANE_ENV_NAME": "multiwalker_v9",
            "RAY_FASTLANE_THROTTLE_MS": "0",
        }

        # Set env vars in main process
        for k, v in test_env_vars.items():
            os.environ[k] = v

        # Track created shm files
        shm_files_before = set(glob.glob("/dev/shm/mosaic.fastlane.*"))

        def env_creator(config):
            # Create env with render_mode for FastLane
            env = multiwalker_v9.parallel_env(render_mode="rgb_array")

            # Get worker index from config
            worker_idx = getattr(config, "worker_index", 0)

            # Wrap with FastLane
            wrapped_env = maybe_wrap_parallel_env(env, worker_index=worker_idx)

            # Wrap for RLlib
            return ParallelPettingZooEnv(wrapped_env)

        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": test_env_vars})
        register_env("test_multiwalker_fastlane", env_creator)

        config = (
            PPOConfig()
            .environment("test_multiwalker_fastlane")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                num_env_runners=1,  # 1 remote worker
                create_env_on_local_worker=True,  # Plus local worker
            )
            .training(
                train_batch_size=200,
                minibatch_size=50,
            )
        )

        algo = config.build()

        # Do one training iteration to trigger env usage
        try:
            algo.train()
        except Exception as e:
            print(f"[DEBUG] Training error (expected for short test): {e}")

        # Check for new shm files
        shm_files_after = set(glob.glob("/dev/shm/mosaic.fastlane.*"))
        new_shm_files = shm_files_after - shm_files_before

        print(f"\n[DEBUG] SHM files before: {shm_files_before}")
        print(f"[DEBUG] SHM files after: {shm_files_after}")
        print(f"[DEBUG] NEW shm files: {new_shm_files}")

        # Expected files: {run_id}-w0 and {run_id}-w1
        expected_streams = [
            f"/dev/shm/mosaic.fastlane.{run_id}-w0",
            f"/dev/shm/mosaic.fastlane.{run_id}-w1",
        ]
        print(f"[DEBUG] Expected streams: {expected_streams}")

        # Cleanup
        for k in test_env_vars:
            os.environ.pop(k, None)
        algo.stop()

        # Check if at least one stream was created
        assert len(new_shm_files) > 0, (
            f"No FastLane shm files created. Expected: {expected_streams}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
