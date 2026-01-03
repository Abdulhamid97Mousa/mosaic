"""Test that env_creator properly wraps environments with FastLane."""
import os
import pytest


def test_env_creator_wraps_parallel_env_with_fastlane():
    """Test that env_creator wraps Parallel envs with FastLane when enabled."""
    # Set FastLane env vars
    os.environ["RAY_FASTLANE_ENABLED"] = "1"
    os.environ["RAY_FASTLANE_RUN_ID"] = "01TESTENVWRAP123456789"
    os.environ["RAY_FASTLANE_ENV_NAME"] = "sisl/multiwalker_v9"
    os.environ["RAY_FASTLANE_THROTTLE_MS"] = "0"

    from ray_worker.runtime import RayWorkerRuntime
    from ray_worker.config import RayWorkerConfig, EnvironmentConfig, TrainingConfig, PettingZooAPIType
    from ray_worker.fastlane import ParallelFastLaneWrapper
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # Create config for Parallel multiwalker
    config = RayWorkerConfig(
        run_id="01TESTENVWRAP123456789",
        environment=EnvironmentConfig(
            family="sisl",
            env_id="multiwalker_v9",
            api_type=PettingZooAPIType.PARALLEL,
            env_kwargs={"render_mode": "rgb_array"},
        ),
        training=TrainingConfig(
            algorithm="PPO",
            total_timesteps=100,
            algo_params={},
        ),
        policy_configuration="parameter_sharing",
        fastlane_enabled=True,
        fastlane_throttle_ms=0,
    )

    # Create runtime
    runtime = RayWorkerRuntime(config)

    # Get env_creator
    env_creator = runtime._create_env_factory()

    # Create env with worker_index=1 (simulating remote worker)
    mock_config = type('Config', (), {'worker_index': 1})()
    env = env_creator(mock_config)

    # Verify wrapping chain
    print(f"\nEnvironment type: {type(env).__name__}")
    print(f"Is ParallelPettingZooEnv? {isinstance(env, ParallelPettingZooEnv)}")

    # Check if inner env is FastLane-wrapped
    if isinstance(env, ParallelPettingZooEnv):
        inner_env = env.par_env if hasattr(env, 'par_env') else env.env
        print(f"Inner env type: {type(inner_env).__name__}")
        print(f"Is ParallelFastLaneWrapper? {isinstance(inner_env, ParallelFastLaneWrapper)}")

        assert isinstance(inner_env, ParallelFastLaneWrapper), \
            f"Inner env should be ParallelFastLaneWrapper, got {type(inner_env).__name__}"

        # Check stream_id
        assert inner_env._config.stream_id == "01TESTENVWRAP123456789-worker-1", \
            f"Stream ID should be '01TESTENVWRAP123456789-worker-1', got '{inner_env._config.stream_id}'"

        print(f"✓ Stream ID: {inner_env._config.stream_id}")
        print(f"✓ Worker index: {inner_env._config.worker_index}")

    # Cleanup
    env.close()

    print("\n✅ Test passed! env_creator correctly wraps Parallel envs with FastLane")


if __name__ == "__main__":
    test_env_creator_wraps_parallel_env_with_fastlane()
