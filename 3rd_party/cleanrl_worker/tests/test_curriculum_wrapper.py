"""
Tests for curriculum learning wrappers using Syllabus-RL.

These tests verify:
1. Curriculum environment creation with make_curriculum_env
2. BabyAITaskWrapper can switch between environments
3. Syllabus SequentialCurriculum is properly configured
4. Environment transitions occur based on curriculum schedule
"""

from __future__ import annotations

import pytest
import gymnasium as gym
import numpy as np


class TestMakeCurriculumEnv:
    """Tests for make_curriculum_env function."""

    def test_creates_vectorized_environment(self):
        """make_curriculum_env should return a gym.vector.VectorEnv."""
        from cleanrl_worker.wrappers.curriculum import make_curriculum_env

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        envs = make_curriculum_env(schedule, num_envs=2)

        assert isinstance(envs, gym.vector.VectorEnv)
        assert envs.num_envs == 2
        envs.close()

    def test_creates_correct_number_of_envs(self):
        """make_curriculum_env should create the specified number of environments."""
        from cleanrl_worker.wrappers.curriculum import make_curriculum_env

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        for num_envs in [1, 2, 4]:
            envs = make_curriculum_env(schedule, num_envs=num_envs)
            assert envs.num_envs == num_envs
            envs.close()

    def test_environments_can_reset(self):
        """Created environments should be able to reset."""
        from cleanrl_worker.wrappers.curriculum import make_curriculum_env

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        envs = make_curriculum_env(schedule, num_envs=2)
        obs, info = envs.reset()

        assert obs is not None
        assert obs.shape[0] == 2  # num_envs
        envs.close()

    def test_environments_can_step(self):
        """Created environments should be able to take steps."""
        from cleanrl_worker.wrappers.curriculum import make_curriculum_env

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        envs = make_curriculum_env(schedule, num_envs=2)
        envs.reset()

        # Take a step with random actions
        actions = np.array([envs.single_action_space.sample() for _ in range(2)])
        obs, rewards, terminated, truncated, info = envs.step(actions)

        assert obs is not None
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        envs.close()


class TestBabyAITaskWrapper:
    """Tests for BabyAITaskWrapper environment switching."""

    @staticmethod
    def _make_minigrid_env(env_id: str) -> gym.Env:
        """Create a MiniGrid env with proper observation wrappers."""
        from minigrid.wrappers import ImgObsWrapper
        env = gym.make(env_id)
        env = ImgObsWrapper(env)  # Convert Dict obs to image
        env = gym.wrappers.FlattenObservation(env)
        return env

    def test_initializes_with_env_ids(self):
        """BabyAITaskWrapper should initialize with list of environment IDs."""
        from cleanrl_worker.wrappers.curriculum import BabyAITaskWrapper

        env_ids = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0"]
        base_env = self._make_minigrid_env(env_ids[0])

        wrapper = BabyAITaskWrapper(base_env, env_ids, apply_wrappers=False)

        assert wrapper.env_ids == env_ids
        assert wrapper.task == 0
        wrapper.close()

    def test_current_env_id_returns_correct_id(self):
        """current_env_id property should return the active environment ID."""
        from cleanrl_worker.wrappers.curriculum import BabyAITaskWrapper

        env_ids = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0"]
        base_env = self._make_minigrid_env(env_ids[0])

        wrapper = BabyAITaskWrapper(base_env, env_ids, apply_wrappers=False)

        assert wrapper.current_env_id == env_ids[0]
        wrapper.close()

    def test_change_task_switches_environment(self):
        """change_task should switch to a different environment."""
        from cleanrl_worker.wrappers.curriculum import BabyAITaskWrapper

        env_ids = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0"]
        base_env = self._make_minigrid_env(env_ids[0])

        # Use apply_wrappers=True so switching works
        wrapper = BabyAITaskWrapper(base_env, env_ids, apply_wrappers=True)

        # Change to second environment (pass env_id string, as Syllabus does)
        wrapper.change_task(env_ids[1])

        assert wrapper.task == env_ids[1]
        assert wrapper.current_env_id == env_ids[1]
        wrapper.close()

    def test_reset_with_new_task_switches_environment(self):
        """reset(new_task=X) should switch to environment X."""
        from cleanrl_worker.wrappers.curriculum import BabyAITaskWrapper

        env_ids = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0"]
        base_env = self._make_minigrid_env(env_ids[0])

        # Use apply_wrappers=True so switching works
        wrapper = BabyAITaskWrapper(base_env, env_ids, apply_wrappers=True)

        # Reset with new task (pass env_id string, as Syllabus does)
        obs, info = wrapper.reset(new_task=env_ids[1])

        assert wrapper.task == env_ids[1]
        assert obs is not None
        wrapper.close()


class TestMakeBabyAICurriculum:
    """Tests for make_babyai_curriculum function."""

    def test_creates_sequential_curriculum(self):
        """make_babyai_curriculum should return a SequentialCurriculum."""
        from cleanrl_worker.wrappers.curriculum import make_babyai_curriculum
        from syllabus.curricula import SequentialCurriculum

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        curriculum = make_babyai_curriculum(schedule)

        assert isinstance(curriculum, SequentialCurriculum)

    def test_curriculum_has_correct_number_of_stages(self):
        """Curriculum should have one stage per environment in schedule."""
        from cleanrl_worker.wrappers.curriculum import make_babyai_curriculum

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0", "steps": 2000},
            {"env_id": "MiniGrid-Empty-8x8-v0"},
        ]

        curriculum = make_babyai_curriculum(schedule)

        # SequentialCurriculum should have 3 stages
        assert len(curriculum.curriculum_list) == 3

    def test_curriculum_samples_initial_task(self):
        """Curriculum should sample task 0 initially."""
        from cleanrl_worker.wrappers.curriculum import make_babyai_curriculum

        schedule = [
            {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 1000},
            {"env_id": "MiniGrid-Empty-6x6-v0"},
        ]

        curriculum = make_babyai_curriculum(schedule)
        tasks = curriculum.sample(k=1)

        # First sample should be task 0
        assert tasks[0] == 0


class TestCurriculumConfigDetection:
    """Tests for detecting curriculum config in CleanRLWorkerConfig."""

    def test_detects_curriculum_schedule_in_config(self):
        """is_curriculum_config should return True when curriculum_schedule present."""
        from cleanrl_worker.curriculum_training import is_curriculum_config
        from cleanrl_worker.config import CleanRLWorkerConfig

        config = CleanRLWorkerConfig(
            run_id="test",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={
                "curriculum_schedule": [
                    {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 5000},
                    {"env_id": "MiniGrid-Empty-6x6-v0"},
                ]
            },
        )

        assert is_curriculum_config(config)

    def test_returns_false_without_curriculum_schedule(self):
        """is_curriculum_config should return False when no curriculum_schedule."""
        from cleanrl_worker.curriculum_training import is_curriculum_config
        from cleanrl_worker.config import CleanRLWorkerConfig

        config = CleanRLWorkerConfig(
            run_id="test",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={},
        )

        assert not is_curriculum_config(config)

    def test_returns_false_with_empty_curriculum_schedule(self):
        """is_curriculum_config should return False when curriculum_schedule is empty."""
        from cleanrl_worker.curriculum_training import is_curriculum_config
        from cleanrl_worker.config import CleanRLWorkerConfig

        config = CleanRLWorkerConfig(
            run_id="test",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={"curriculum_schedule": []},
        )

        assert not is_curriculum_config(config)


class TestCurriculumTrainingConfig:
    """Tests for CurriculumTrainingConfig creation."""

    def test_creates_config_from_worker_config(self):
        """CurriculumTrainingConfig.from_worker_config should create valid config."""
        from cleanrl_worker.curriculum_training import CurriculumTrainingConfig
        from cleanrl_worker.config import CleanRLWorkerConfig

        worker_config = CleanRLWorkerConfig(
            run_id="test-run",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            seed=42,
            extras={
                "curriculum_schedule": [
                    {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 5000},
                    {"env_id": "MiniGrid-Empty-6x6-v0"},
                ],
                "num_envs": 2,
            },
        )

        config = CurriculumTrainingConfig.from_worker_config(worker_config)

        assert config.run_id == "test-run"
        assert config.total_timesteps == 10000
        assert config.seed == 42
        assert config.num_envs == 2
        assert len(config.curriculum_schedule) == 2

    def test_raises_error_without_curriculum_schedule(self):
        """from_worker_config should raise ValueError without curriculum_schedule."""
        from cleanrl_worker.curriculum_training import CurriculumTrainingConfig
        from cleanrl_worker.config import CleanRLWorkerConfig

        worker_config = CleanRLWorkerConfig(
            run_id="test-run",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={},
        )

        with pytest.raises(ValueError, match="curriculum_schedule is required"):
            CurriculumTrainingConfig.from_worker_config(worker_config)

    def test_fastlane_config_defaults(self):
        """CurriculumTrainingConfig should have FastLane enabled by default."""
        from cleanrl_worker.curriculum_training import CurriculumTrainingConfig
        from cleanrl_worker.config import CleanRLWorkerConfig

        worker_config = CleanRLWorkerConfig(
            run_id="test-run",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={
                "curriculum_schedule": [
                    {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 5000},
                    {"env_id": "MiniGrid-Empty-6x6-v0"},
                ],
            },
        )

        config = CurriculumTrainingConfig.from_worker_config(worker_config)

        # FastLane should be enabled by default for curriculum training
        assert config.fastlane_only is True
        assert config.fastlane_slot == 0
        assert config.fastlane_video_mode == "grid"
        assert config.fastlane_grid_limit is None  # Defaults to num_envs at runtime

    def test_fastlane_config_from_extras(self):
        """CurriculumTrainingConfig should respect FastLane settings from extras."""
        from cleanrl_worker.curriculum_training import CurriculumTrainingConfig
        from cleanrl_worker.config import CleanRLWorkerConfig

        worker_config = CleanRLWorkerConfig(
            run_id="test-run",
            algo="ppo",
            env_id="MiniGrid-Empty-5x5-v0",
            total_timesteps=10000,
            extras={
                "curriculum_schedule": [
                    {"env_id": "MiniGrid-Empty-5x5-v0", "steps": 5000},
                    {"env_id": "MiniGrid-Empty-6x6-v0"},
                ],
                "fastlane_only": False,
                "fastlane_video_mode": "single",
                "fastlane_slot": 2,
                "fastlane_grid_limit": 8,
            },
        )

        config = CurriculumTrainingConfig.from_worker_config(worker_config)

        assert config.fastlane_only is False
        assert config.fastlane_video_mode == "single"
        assert config.fastlane_slot == 2
        assert config.fastlane_grid_limit == 8


class TestPresetCurricula:
    """Tests for preset curriculum schedules."""

    def test_babyai_goto_curriculum_has_four_stages(self):
        """BABYAI_GOTO_CURRICULUM should have 4 stages."""
        from cleanrl_worker.wrappers.curriculum import BABYAI_GOTO_CURRICULUM

        assert len(BABYAI_GOTO_CURRICULUM) == 4
        assert all("env_id" in stage for stage in BABYAI_GOTO_CURRICULUM)

    def test_babyai_doorkey_curriculum_has_four_stages(self):
        """BABYAI_DOORKEY_CURRICULUM should have 4 stages."""
        from cleanrl_worker.wrappers.curriculum import BABYAI_DOORKEY_CURRICULUM

        assert len(BABYAI_DOORKEY_CURRICULUM) == 4
        assert all("env_id" in stage for stage in BABYAI_DOORKEY_CURRICULUM)

    def test_preset_curricula_are_valid_schedules(self):
        """Preset curricula should be valid for make_curriculum_env."""
        from cleanrl_worker.wrappers.curriculum import (
            make_babyai_curriculum,
            BABYAI_GOTO_CURRICULUM,
            BABYAI_DOORKEY_CURRICULUM,
        )

        # Should not raise
        curriculum1 = make_babyai_curriculum(BABYAI_GOTO_CURRICULUM)
        curriculum2 = make_babyai_curriculum(BABYAI_DOORKEY_CURRICULUM)

        assert curriculum1 is not None
        assert curriculum2 is not None


class TestFastLaneEnvSetup:
    """Tests for FastLane environment variable setup."""

    def test_setup_fastlane_env_sets_variables(self):
        """_setup_fastlane_env should set the correct environment variables."""
        import os
        from cleanrl_worker.curriculum_training import (
            CurriculumTrainingConfig,
            _setup_fastlane_env,
        )

        config = CurriculumTrainingConfig(
            run_id="test-fastlane-run",
            curriculum_schedule=[{"env_id": "MiniGrid-Empty-5x5-v0"}],
            total_timesteps=1000,
            num_envs=4,
            fastlane_only=True,
            fastlane_slot=0,
            fastlane_video_mode="grid",
            fastlane_grid_limit=4,
        )

        # Clear any existing values
        for var in ["GYM_GUI_FASTLANE_ONLY", "GYM_GUI_FASTLANE_SLOT",
                    "GYM_GUI_FASTLANE_VIDEO_MODE", "GYM_GUI_FASTLANE_GRID_LIMIT",
                    "CLEANRL_NUM_ENVS", "CLEANRL_RUN_ID"]:
            os.environ.pop(var, None)

        _setup_fastlane_env(config)

        assert os.environ["GYM_GUI_FASTLANE_ONLY"] == "1"
        assert os.environ["GYM_GUI_FASTLANE_SLOT"] == "0"
        assert os.environ["GYM_GUI_FASTLANE_VIDEO_MODE"] == "grid"
        assert os.environ["GYM_GUI_FASTLANE_GRID_LIMIT"] == "4"
        assert os.environ["CLEANRL_NUM_ENVS"] == "4"
        assert os.environ["CLEANRL_RUN_ID"] == "test-fastlane-run"

    def test_setup_fastlane_env_respects_disabled(self):
        """_setup_fastlane_env should set FASTLANE_ONLY=0 when disabled."""
        import os
        from cleanrl_worker.curriculum_training import (
            CurriculumTrainingConfig,
            _setup_fastlane_env,
        )

        config = CurriculumTrainingConfig(
            run_id="test-disabled",
            curriculum_schedule=[{"env_id": "MiniGrid-Empty-5x5-v0"}],
            total_timesteps=1000,
            num_envs=2,
            fastlane_only=False,  # Disabled
        )

        _setup_fastlane_env(config)

        assert os.environ["GYM_GUI_FASTLANE_ONLY"] == "0"

    def test_setup_fastlane_env_defaults_grid_limit_to_num_envs(self):
        """_setup_fastlane_env should use num_envs as grid_limit when None."""
        import os
        from cleanrl_worker.curriculum_training import (
            CurriculumTrainingConfig,
            _setup_fastlane_env,
        )

        config = CurriculumTrainingConfig(
            run_id="test-grid-default",
            curriculum_schedule=[{"env_id": "MiniGrid-Empty-5x5-v0"}],
            total_timesteps=1000,
            num_envs=8,
            fastlane_grid_limit=None,  # Should default to num_envs
        )

        _setup_fastlane_env(config)

        assert os.environ["GYM_GUI_FASTLANE_GRID_LIMIT"] == "8"
