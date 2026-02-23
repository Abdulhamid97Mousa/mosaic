"""Test TensorBoard integration with actual training.

Uses PPO for faster testing (no replay buffer overhead).
"""

import os
import glob
import tempfile
import pytest

os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['RAY_memory_monitor_refresh_ms'] = '0'


class TestTensorBoardIntegration:
    """Test TensorBoard file creation during training."""

    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory for test outputs."""
        return tempfile.mkdtemp(prefix='tb_int_test_')

    def test_tensorboard_files_created_ppo(self, test_dir):
        """Test that TensorBoard files are created during PPO training."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig(
            run_id='ppo-tb-test',
            environment=EnvironmentConfig(family='classic', env_id='tictactoe_v3'),  # Small env
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(
                algorithm='PPO',
                total_timesteps=500,  # Very short
                algo_params={'train_batch_size': 256},
            ),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            output_dir=test_dir,
            tensorboard=True,
            wandb=False,
            fastlane_enabled=False,
        )

        runtime = RayWorkerRuntime(config)
        result = runtime.run()

        # Verify training completed
        assert result is not None

        # Check TensorBoard files in tensorboard subdirectory
        tb_dir = config.tensorboard_log_dir
        tb_files = glob.glob(f'{tb_dir}/events.out.tfevents.*')

        print(f"\nTensorBoard directory: {tb_dir}")
        print(f"Files found: {tb_files}")

        assert len(tb_files) >= 1, f"Expected TensorBoard files in {tb_dir}"

        # Check file has actual data (more than just header)
        max_size = max(os.path.getsize(f) for f in tb_files)
        print(f"Largest file size: {max_size} bytes")
        assert max_size > 100, f"TensorBoard files too small ({max_size} bytes)"

    def test_ray_writes_to_run_dir(self, test_dir):
        """Test that Ray's logs go to run_dir, not ~/ray_results."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig(
            run_id='ray-log-test',
            environment=EnvironmentConfig(family='classic', env_id='tictactoe_v3'),
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(
                algorithm='PPO',
                total_timesteps=500,
                algo_params={'train_batch_size': 256},
            ),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            output_dir=test_dir,
            tensorboard=True,
            wandb=False,
            fastlane_enabled=False,
        )

        runtime = RayWorkerRuntime(config)
        runtime.run()

        # Check that Ray's event files are in run_dir (not ~/ray_results)
        run_dir = config.run_dir
        all_event_files = glob.glob(f'{run_dir}/**/events.out.tfevents.*', recursive=True)

        print(f"\nRun directory: {run_dir}")
        print(f"All event files: {all_event_files}")

        assert len(all_event_files) >= 1, f"No event files in {run_dir}"

        # Verify no files in ~/ray_results for this run
        home_ray_files = glob.glob(f'{os.path.expanduser("~")}/ray_results/**/*{config.run_id}*', recursive=True)
        assert len(home_ray_files) == 0, f"Found files in ~/ray_results: {home_ray_files}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
