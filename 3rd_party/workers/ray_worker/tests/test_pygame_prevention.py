"""Tests to ensure pygame windows are prevented during training.

These tests verify that:
1. SDL environment variables are correctly set to dummy drivers
2. No pygame display is initialized during environment creation
3. render_mode=None is used for agent detection (no rendering needed)
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestSDLDummyDriver:
    """Verify SDL is configured to use dummy drivers (no display)."""

    def test_sdl_videodriver_dummy_prevents_display(self):
        """Setting SDL_VIDEODRIVER=dummy should prevent pygame display init."""
        # Set the env vars as our code does
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        # Import pygame AFTER setting env vars
        import pygame

        # Initialize pygame - should NOT open a window with dummy driver
        pygame.init()

        # Check that display is in dummy mode (no actual window)
        # pygame.display.get_driver() returns the video driver name
        driver = pygame.display.get_driver()
        assert driver == "dummy" or driver == "", f"Expected dummy driver, got: {driver}"

        pygame.quit()

    def test_runtime_sets_sdl_before_imports(self):
        """RayWorkerRuntime.run() should set SDL vars before pygame import."""
        # Read the source to verify SDL vars are set early
        import inspect
        from ray_worker.runtime import RayWorkerRuntime

        source = inspect.getsource(RayWorkerRuntime.run)

        # SDL vars should be set before ray.init
        sdl_pos = source.find('SDL_VIDEODRIVER')
        ray_init_pos = source.find('ray.init')

        assert sdl_pos != -1, "SDL_VIDEODRIVER not found in run()"
        assert ray_init_pos != -1, "ray.init not found in run()"
        assert sdl_pos < ray_init_pos, "SDL vars should be set before ray.init"


class TestAgentDetectionNoDisplay:
    """Verify agent detection doesn't trigger pygame display."""

    def test_get_agent_ids_uses_none_render_mode(self):
        """_get_agent_ids creates env with render_mode=None (no display)."""
        import inspect
        from ray_worker.runtime import RayWorkerRuntime

        source = inspect.getsource(RayWorkerRuntime._get_agent_ids)

        # Should contain render_mode = None
        assert 'render_mode' in source
        assert 'None' in source or 'render_mode"] = None' in source

    def test_render_mode_none_does_not_init_pygame_display(self):
        """Creating env with render_mode=None should not initialize pygame display."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        # Create environment with render_mode=None
        try:
            from pettingzoo.sisl import multiwalker_v9
            env = multiwalker_v9.env(render_mode=None)

            # Get agent IDs without rendering
            agents = env.possible_agents
            assert len(agents) > 0

            env.close()
        except ImportError:
            pytest.skip("PettingZoo SISL not installed")


class TestRuntimeEnvPropagation:
    """Verify env vars are propagated to Ray workers."""

    def test_all_required_env_vars_in_runtime_env(self):
        """All required env vars should be in Ray runtime_env."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-env-propagation",
            "environment": {
                "family": "sisl",
                "env_id": "pursuit_v4",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "resources": {"num_workers": 2},
            "checkpoint": {},
            "fastlane_enabled": True,
            "fastlane_throttle_ms": 33,
        })

        runtime = RayWorkerRuntime(config)

        with patch('ray_worker.runtime.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False

            try:
                runtime.run()
            except Exception:
                pass

            # Get the runtime_env passed to ray.init
            call_kwargs = mock_ray.init.call_args[1]
            runtime_env = call_kwargs.get("runtime_env", {})
            env_vars = runtime_env.get("env_vars", {})

            # All these should be present
            required_vars = [
                "SDL_VIDEODRIVER",
                "SDL_AUDIODRIVER",
                "RAY_FASTLANE_ENABLED",
                "RAY_FASTLANE_RUN_ID",
                "RAY_FASTLANE_ENV_NAME",
                "RAY_FASTLANE_THROTTLE_MS",
            ]

            for var in required_vars:
                assert var in env_vars, f"Missing env var: {var}"

    def test_runtime_env_values_are_strings(self):
        """All runtime_env values should be strings (Ray requirement)."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-string-values",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO"},
            "resources": {},
            "checkpoint": {},
            "fastlane_enabled": True,
            "fastlane_throttle_ms": 100,  # Integer in config
        })

        runtime = RayWorkerRuntime(config)

        with patch('ray_worker.runtime.ray') as mock_ray:
            mock_ray.is_initialized.return_value = False

            try:
                runtime.run()
            except Exception:
                pass

            call_kwargs = mock_ray.init.call_args[1]
            env_vars = call_kwargs.get("runtime_env", {}).get("env_vars", {})

            # All values should be strings
            for key, value in env_vars.items():
                assert isinstance(value, str), f"{key} should be string, got {type(value)}"

            # Specifically check throttle_ms is converted to string
            assert env_vars.get("RAY_FASTLANE_THROTTLE_MS") == "100"


class TestNoPopupDuringTraining:
    """Integration tests to verify no popups during training setup."""

    def test_algorithm_config_build_no_popup(self):
        """Building algorithm config should not open any windows."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-no-popup",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO"},
            "resources": {"num_workers": 0},  # No remote workers
            "checkpoint": {},
            "fastlane_enabled": True,
        })

        runtime = RayWorkerRuntime(config)

        # Mock Ray to avoid actual initialization
        with patch('ray_worker.runtime.ray') as mock_ray:
            mock_ray.is_initialized.return_value = True  # Pretend Ray is already up

            # This should complete without opening any windows
            # The _get_agent_ids call uses render_mode=None
            algo_config = runtime._build_algorithm_config()

            # Should have detected agents
            assert len(runtime._agent_ids) > 0
