"""Test TensorBoard path configuration.

These tests verify the TensorBoard log directory respects output_dir.
"""

import os
import tempfile
import pytest

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'


class TestTensorBoardPath:
    """Test TensorBoard path configuration."""

    def test_tensorboard_path_respects_output_dir(self):
        """TensorBoard log dir should be inside run_dir when output_dir is set."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )

        test_dir = tempfile.mkdtemp(prefix='tb_test_')

        config = RayWorkerConfig(
            run_id='test-run',
            environment=EnvironmentConfig(family='classic', env_id='chess_v6'),
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(algorithm='DQN', total_timesteps=100),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            output_dir=test_dir,
            tensorboard=True,
        )

        # TensorBoard should be inside run_dir
        assert config.tensorboard_log_dir is not None
        assert str(config.tensorboard_log_dir).startswith(str(config.run_dir))
        assert config.tensorboard_log_dir == config.run_dir / "tensorboard"

    def test_tensorboard_path_default(self):
        """TensorBoard log dir should be inside run_dir by default."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )

        config = RayWorkerConfig(
            run_id='test-default',
            environment=EnvironmentConfig(family='classic', env_id='chess_v6'),
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(algorithm='DQN', total_timesteps=100),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            tensorboard=True,
        )

        # Should still be inside run_dir
        assert config.tensorboard_log_dir is not None
        assert config.tensorboard_log_dir == config.run_dir / "tensorboard"

    def test_tensorboard_disabled(self):
        """TensorBoard log dir should be None when disabled."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )

        config = RayWorkerConfig(
            run_id='test-disabled',
            environment=EnvironmentConfig(family='classic', env_id='chess_v6'),
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(algorithm='DQN', total_timesteps=100),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            tensorboard=False,
        )

        assert config.tensorboard_log_dir is None

    def test_custom_tensorboard_dir(self):
        """Custom tensorboard_dir should override default."""
        from ray_worker.config import (
            RayWorkerConfig, TrainingConfig, EnvironmentConfig,
            PolicyConfiguration, ResourceConfig
        )

        custom_tb_dir = tempfile.mkdtemp(prefix='custom_tb_')

        config = RayWorkerConfig(
            run_id='test-custom',
            environment=EnvironmentConfig(family='classic', env_id='chess_v6'),
            policy_configuration=PolicyConfiguration.SELF_PLAY,
            training=TrainingConfig(algorithm='DQN', total_timesteps=100),
            resources=ResourceConfig(num_workers=0, num_gpus=0),
            tensorboard=True,
            tensorboard_dir=custom_tb_dir,
        )

        from pathlib import Path
        assert config.tensorboard_log_dir == Path(custom_tb_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
