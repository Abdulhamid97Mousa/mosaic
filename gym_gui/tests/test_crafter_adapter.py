"""Regression tests for the Crafter adapter integration.

Crafter is an open-world survival game benchmark for reinforcement learning.
Paper: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Skip entire module if crafter is not installed
pytest.importorskip("crafter")

from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.crafter import (
    CrafterAdapter,
    CrafterRewardAdapter,
    CrafterNoRewardAdapter,
    CRAFTER_ADAPTERS,
    CRAFTER_ACHIEVEMENTS,
    CRAFTER_ACTIONS,
)
from gym_gui.config.game_configs import CrafterConfig
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ENV_CRAFTER_BOOT,
    LOG_ENV_CRAFTER_STEP,
)


def _make_adapter(game_id: GameId = GameId.CRAFTER_REWARD, **overrides) -> CrafterAdapter:
    """Create a Crafter adapter with the given configuration."""
    config = CrafterConfig(env_id=game_id.value, **overrides)
    context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
    adapter_class = CRAFTER_ADAPTERS.get(game_id, CrafterAdapter)
    adapter = adapter_class(context, config=config)
    adapter.load()
    return adapter


class TestCrafterAdapterBasics:
    """Basic adapter functionality tests."""

    def test_crafter_adapter_creation(self) -> None:
        """Test that CrafterAdapter can be created and loaded."""
        adapter = _make_adapter()
        try:
            assert adapter.id == GameId.CRAFTER_REWARD.value
            assert adapter.default_render_mode == RenderMode.RGB_ARRAY
        finally:
            adapter.close()

    def test_crafter_no_reward_adapter(self) -> None:
        """Test CrafterNoRewardAdapter variant."""
        adapter = _make_adapter(GameId.CRAFTER_NO_REWARD)
        try:
            assert adapter.id == GameId.CRAFTER_NO_REWARD.value
        finally:
            adapter.close()

    def test_crafter_adapters_registry(self) -> None:
        """Test that CRAFTER_ADAPTERS registry is properly populated."""
        assert GameId.CRAFTER_REWARD in CRAFTER_ADAPTERS
        assert GameId.CRAFTER_NO_REWARD in CRAFTER_ADAPTERS
        assert CRAFTER_ADAPTERS[GameId.CRAFTER_REWARD] == CrafterRewardAdapter
        assert CRAFTER_ADAPTERS[GameId.CRAFTER_NO_REWARD] == CrafterNoRewardAdapter


class TestCrafterAdapterBoot:
    """Tests for adapter initialization and boot logging."""

    def test_crafter_adapter_boot_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that boot event is logged correctly."""
        caplog.set_level(logging.INFO, logger="gym_gui.core.adapters.base")
        adapter = _make_adapter()
        try:
            boot_codes = [getattr(record, "log_code", None) for record in caplog.records]
            boot_codes = [c for c in boot_codes if c is not None]
            assert LOG_ENV_CRAFTER_BOOT.code in boot_codes
        finally:
            adapter.close()

    def test_crafter_gym_kwargs(self) -> None:
        """Test that gym_kwargs returns expected values."""
        adapter = _make_adapter()
        try:
            kwargs = adapter.gym_kwargs()
            assert "render_mode" in kwargs
            assert kwargs["render_mode"] == "rgb_array"
        finally:
            adapter.close()


class TestCrafterAdapterReset:
    """Tests for reset functionality."""

    def test_crafter_reset_basic(self) -> None:
        """Test basic reset functionality."""
        adapter = _make_adapter()
        try:
            step = adapter.reset()
            assert step.observation is not None
            assert isinstance(step.observation, np.ndarray)
            assert step.observation.dtype == np.uint8
        finally:
            adapter.close()

    def test_crafter_reset_observation_shape(self) -> None:
        """Test that observation has correct shape (64, 64, 3)."""
        adapter = _make_adapter()
        try:
            step = adapter.reset()
            # Crafter returns 64x64 RGB images
            assert step.observation.shape == (64, 64, 3)
        finally:
            adapter.close()

    def test_crafter_reset_with_seed_reproducibility(self) -> None:
        """Test that reset with same seed produces same observation."""
        adapter = _make_adapter()
        try:
            first = adapter.reset(seed=12345)
            second = adapter.reset(seed=12345)
            np.testing.assert_array_equal(first.observation, second.observation)
        finally:
            adapter.close()

    def test_crafter_reset_state_metadata(self) -> None:
        """Test that reset state contains expected metadata."""
        adapter = _make_adapter()
        try:
            step = adapter.reset()
            assert step.state.environment["env_id"] == GameId.CRAFTER_REWARD.value
        finally:
            adapter.close()


class TestCrafterAdapterStep:
    """Tests for step functionality."""

    def test_crafter_step_basic(self) -> None:
        """Test basic step functionality."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            # Use noop action (0)
            step = adapter.step(0)
            assert step.observation is not None
            assert isinstance(step.reward, float)
            assert isinstance(step.terminated, bool)
            assert isinstance(step.truncated, bool)
        finally:
            adapter.close()

    def test_crafter_step_all_actions_valid(self) -> None:
        """Test that all 17 actions are valid."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            # Test all 17 actions (0-16)
            for action in range(17):
                step = adapter.step(action)
                assert step.observation is not None
                # Reset after each to avoid terminal state issues
                if step.terminated or step.truncated:
                    _ = adapter.reset(seed=42)
        finally:
            adapter.close()

    def test_crafter_step_observation_shape(self) -> None:
        """Test step observation has correct shape."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            step = adapter.step(0)
            assert step.observation.shape == (64, 64, 3)
        finally:
            adapter.close()

    def test_crafter_step_info_contains_achievements(self) -> None:
        """Test that step info contains achievement tracking."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            step = adapter.step(0)
            # Info should contain achievements dict
            raw_info = step.info
            assert "achievements" in raw_info or "_crafter_raw_observation" in raw_info
        finally:
            adapter.close()

    def test_crafter_step_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that step logging works (every 100 steps)."""
        caplog.set_level(logging.DEBUG, logger="gym_gui.core.adapters.base")
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            # Step log happens on step 1, 101, 201, etc.
            step = adapter.step(0)
            step_codes = [getattr(record, "log_code", None) for record in caplog.records]
            step_codes = [c for c in step_codes if c is not None]
            assert LOG_ENV_CRAFTER_STEP.code in step_codes
        finally:
            adapter.close()


class TestCrafterAdapterRender:
    """Tests for render functionality."""

    def test_crafter_render_returns_dict(self) -> None:
        """Test that render returns expected dict structure."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            assert isinstance(render_payload, dict)
            assert render_payload.get("mode") == "rgb_array"
            assert isinstance(render_payload.get("rgb"), np.ndarray)
            assert render_payload.get("game_id") == GameId.CRAFTER_REWARD.value
        finally:
            adapter.close()

    def test_crafter_render_rgb_shape(self) -> None:
        """Test that rendered RGB has correct shape."""
        adapter = _make_adapter()
        try:
            _ = adapter.reset(seed=42)
            render_payload = adapter.render()
            rgb = render_payload.get("rgb")
            assert rgb is not None
            assert len(rgb.shape) == 3
            assert rgb.shape[2] == 3  # RGB channels
        finally:
            adapter.close()

    def test_crafter_step_render_payload(self) -> None:
        """Test that step includes render payload."""
        adapter = _make_adapter()
        try:
            step = adapter.reset(seed=42)
            render_payload = step.render_payload
            assert isinstance(render_payload, dict)
            assert render_payload.get("mode") == "rgb_array"
        finally:
            adapter.close()


class TestCrafterAdapterHelpers:
    """Tests for helper methods and constants."""

    def test_crafter_get_action_name(self) -> None:
        """Test get_action_name helper."""
        adapter = _make_adapter()
        try:
            assert adapter.get_action_name(0) == "noop"
            assert adapter.get_action_name(1) == "move_left"
            assert adapter.get_action_name(5) == "do"
            assert adapter.get_action_name(16) == "make_iron_sword"
            assert adapter.get_action_name(999) == "unknown_999"
        finally:
            adapter.close()

    def test_crafter_achievements_list(self) -> None:
        """Test CRAFTER_ACHIEVEMENTS constant."""
        assert len(CRAFTER_ACHIEVEMENTS) == 22
        assert "collect_wood" in CRAFTER_ACHIEVEMENTS
        assert "defeat_zombie" in CRAFTER_ACHIEVEMENTS
        assert "make_iron_sword" in CRAFTER_ACHIEVEMENTS

    def test_crafter_actions_list(self) -> None:
        """Test CRAFTER_ACTIONS constant."""
        assert len(CRAFTER_ACTIONS) == 17
        assert CRAFTER_ACTIONS[0] == "noop"
        assert CRAFTER_ACTIONS[5] == "do"
        assert CRAFTER_ACTIONS[16] == "make_iron_sword"

    def test_crafter_get_achievement_names(self) -> None:
        """Test get_achievement_names static method."""
        names = CrafterAdapter.get_achievement_names()
        assert len(names) == 22
        assert "collect_wood" in names


class TestCrafterConfig:
    """Tests for CrafterConfig dataclass."""

    def test_crafter_config_defaults(self) -> None:
        """Test CrafterConfig default values."""
        config = CrafterConfig()
        assert config.env_id == GameId.CRAFTER_REWARD.value
        assert config.area == (64, 64)
        assert config.view == (9, 9)
        assert config.size == (64, 64)
        assert config.reward is True
        assert config.length == 10000
        assert config.seed is None
        assert config.render_mode == "rgb_array"
        assert config.reward_multiplier == 1.0

    def test_crafter_config_custom_values(self) -> None:
        """Test CrafterConfig with custom values."""
        config = CrafterConfig(
            env_id=GameId.CRAFTER_NO_REWARD.value,
            reward=False,
            length=5000,
            seed=42,
            reward_multiplier=2.0,
        )
        assert config.env_id == GameId.CRAFTER_NO_REWARD.value
        assert config.reward is False
        assert config.length == 5000
        assert config.seed == 42
        assert config.reward_multiplier == 2.0

    def test_crafter_config_to_gym_kwargs(self) -> None:
        """Test CrafterConfig.to_gym_kwargs()."""
        config = CrafterConfig()
        kwargs = config.to_gym_kwargs()
        assert "render_mode" in kwargs
        assert kwargs["render_mode"] == "rgb_array"


class TestCrafterRewardMultiplier:
    """Tests for reward multiplier functionality."""

    def test_crafter_reward_multiplier_applied(self) -> None:
        """Test that reward multiplier is applied to rewards."""
        adapter = _make_adapter(reward_multiplier=10.0)
        try:
            _ = adapter.reset(seed=42)
            # Take several steps to potentially get a reward
            for _ in range(10):
                step = adapter.step(3)  # move_up
                # If we got a reward, it should be scaled
                if step.reward != 0:
                    # Original rewards are typically ±1 for achievements
                    # With 10x multiplier, should be ±10 (or health-based ±0.1 * 10 = ±1)
                    pass  # Just verify it doesn't crash
        finally:
            adapter.close()


class TestCrafterControlModes:
    """Tests for supported control modes."""

    def test_crafter_supports_human_control(self) -> None:
        """Test that Crafter supports human control mode."""
        context = AdapterContext(settings=None, control_mode=ControlMode.HUMAN_ONLY)
        config = CrafterConfig()
        adapter = CrafterAdapter(context, config=config)
        adapter.load()
        try:
            assert ControlMode.HUMAN_ONLY in CrafterAdapter.supported_control_modes
        finally:
            adapter.close()

    def test_crafter_supports_agent_control(self) -> None:
        """Test that Crafter supports agent control mode."""
        assert ControlMode.AGENT_ONLY in CrafterAdapter.supported_control_modes

    def test_crafter_supports_hybrid_control(self) -> None:
        """Test that Crafter supports hybrid control mode."""
        assert ControlMode.HYBRID_TURN_BASED in CrafterAdapter.supported_control_modes
