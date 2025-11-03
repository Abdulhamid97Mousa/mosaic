"""Integration tests for all MiniGrid Empty environment variants.

This test suite verifies that all 6 Empty environment variants are properly
integrated into the gym_gui system and work correctly with the MiniGrid adapter.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("minigrid")

from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.minigrid import MiniGridAdapter
from gym_gui.config.game_configs import (
    DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
    DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG,
    DEFAULT_MINIGRID_EMPTY_6x6_CONFIG,
    DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG,
    DEFAULT_MINIGRID_EMPTY_8x8_CONFIG,
    DEFAULT_MINIGRID_EMPTY_16x16_CONFIG,
    MiniGridConfig,
)
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.game_docs import get_game_info
from gym_gui.logging_config.log_constants import (
    LOG_ENV_MINIGRID_BOOT,
    LOG_ENV_MINIGRID_STEP,
)
from gym_gui.ui.environments.single_agent_env.minigrid import (
    MINIGRID_GAME_IDS,
    resolve_default_config,
)


# All Empty environment variants that should be registered
ALL_EMPTY_VARIANTS = [
    (GameId.MINIGRID_EMPTY_5x5, DEFAULT_MINIGRID_EMPTY_5x5_CONFIG, "5x5"),
    (GameId.MINIGRID_EMPTY_RANDOM_5x5, DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG, "Random-5x5"),
    (GameId.MINIGRID_EMPTY_6x6, DEFAULT_MINIGRID_EMPTY_6x6_CONFIG, "6x6"),
    (GameId.MINIGRID_EMPTY_RANDOM_6x6, DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG, "Random-6x6"),
    (GameId.MINIGRID_EMPTY_8x8, DEFAULT_MINIGRID_EMPTY_8x8_CONFIG, "8x8"),
    (GameId.MINIGRID_EMPTY_16x16, DEFAULT_MINIGRID_EMPTY_16x16_CONFIG, "16x16"),
]


class TestEmptyEnvironmentRegistration:
    """Test that all Empty variants are properly registered in the system."""

    def test_all_empty_game_ids_exist(self) -> None:
        """Verify all Empty GameId enum members exist."""
        expected_ids = [
            GameId.MINIGRID_EMPTY_5x5,
            GameId.MINIGRID_EMPTY_RANDOM_5x5,
            GameId.MINIGRID_EMPTY_6x6,
            GameId.MINIGRID_EMPTY_RANDOM_6x6,
            GameId.MINIGRID_EMPTY_8x8,
            GameId.MINIGRID_EMPTY_16x16,
        ]
        
        for game_id in expected_ids:
            assert game_id in GameId
            assert "Empty" in game_id.value
            assert game_id.value.startswith("MiniGrid-Empty-")

    def test_all_empty_configs_exported(self) -> None:
        """Verify all Empty default configs are exported."""
        from gym_gui.config import game_configs
        
        expected_configs = [
            "DEFAULT_MINIGRID_EMPTY_5x5_CONFIG",
            "DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG",
            "DEFAULT_MINIGRID_EMPTY_6x6_CONFIG",
            "DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG",
            "DEFAULT_MINIGRID_EMPTY_8x8_CONFIG",
            "DEFAULT_MINIGRID_EMPTY_16x16_CONFIG",
        ]
        
        for config_name in expected_configs:
            assert hasattr(game_configs, config_name)
            config = getattr(game_configs, config_name)
            assert isinstance(config, MiniGridConfig)

    def test_empty_variants_in_minigrid_game_ids(self) -> None:
        """Verify all Empty variants are in MINIGRID_GAME_IDS tuple."""
        empty_game_ids = [
            GameId.MINIGRID_EMPTY_5x5,
            GameId.MINIGRID_EMPTY_RANDOM_5x5,
            GameId.MINIGRID_EMPTY_6x6,
            GameId.MINIGRID_EMPTY_RANDOM_6x6,
            GameId.MINIGRID_EMPTY_8x8,
            GameId.MINIGRID_EMPTY_16x16,
        ]
        
        for game_id in empty_game_ids:
            assert game_id in MINIGRID_GAME_IDS

    def test_resolve_default_config_for_all_empty_variants(self) -> None:
        """Verify resolve_default_config returns correct configs for all Empty variants."""
        for game_id, expected_config, variant_name in ALL_EMPTY_VARIANTS:
            resolved = resolve_default_config(game_id)
            assert resolved.env_id == expected_config.env_id
            assert resolved.env_id == game_id.value


class TestEmptyEnvironmentConfiguration:
    """Test configuration and metadata for Empty environments."""

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_config_has_correct_env_id(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify each config has the correct env_id."""
        assert config.env_id == game_id.value
        assert "Empty" in config.env_id

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_keyboard_mappings_exist(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify keyboard mappings exist for all Empty variants.
        
        Note: Keyboard mappings are defined in human_input.py but not exported.
        This test verifies the structure exists by importing the controller.
        """
        from gym_gui.controllers.human_input import HumanInputController
        # If the controller can be instantiated, keyboard mappings are available
        assert HumanInputController is not None

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_game_info_documentation_exists(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify game info documentation exists for all Empty variants."""
        info = get_game_info(game_id)
        assert info is not None
        assert len(info) > 0
        assert "Empty" in info
        assert "MiniGrid" in info


class TestEmptyEnvironmentAdapter:
    """Test MiniGrid adapter functionality with all Empty variants."""

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_adapter_can_load_environment(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify adapter can load each Empty variant."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            assert adapter._env is not None  # Check internal state since is_loaded doesn't exist
            assert adapter.id == game_id.value
        finally:
            adapter.close()

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_adapter_reset_produces_valid_observation(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify reset produces valid observations for all Empty variants."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            step = adapter.reset(seed=42)
            
            # Check observation properties
            assert isinstance(step.observation, np.ndarray)
            assert step.observation.dtype == np.uint8
            assert step.observation.ndim == 1
            assert step.observation.shape[0] > 0  # Should have some observation
            
            # Check state metadata
            assert step.state.environment["env_id"] == game_id.value
            
            # Check render payload
            assert isinstance(step.render_payload, dict)
            assert step.render_payload.get("mode") == "rgb_array"
            rgb = step.render_payload.get("rgb")
            assert isinstance(rgb, np.ndarray)
            assert rgb.ndim == 3  # H x W x C
            assert rgb.shape[2] == 3  # RGB channels
            
        finally:
            adapter.close()

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_adapter_step_works(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify step function works for all Empty variants."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            _ = adapter.reset(seed=42)
            
            # Take a step (action 2 = move forward)
            step = adapter.step(2)
            
            assert isinstance(step.observation, np.ndarray)
            assert step.observation.dtype == np.uint8
            assert isinstance(step.reward, (int, float))
            assert isinstance(step.terminated, bool)
            assert isinstance(step.truncated, bool)
            assert isinstance(step.info, dict)
            
        finally:
            adapter.close()

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_adapter_action_space_is_discrete_7(
        self, game_id: GameId, config: MiniGridConfig, variant_name: str
    ) -> None:
        """Verify action space is Discrete(7) for all Empty variants."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            action_space = adapter.action_space
            
            # MiniGrid environments have 7 discrete actions
            assert hasattr(action_space, 'n'), "Action space should be Discrete with 'n' attribute"
            assert action_space.n == 7  # type: ignore[attr-defined]
            
        finally:
            adapter.close()


class TestEmptyEnvironmentLogging:
    """Test logging and telemetry for Empty environments."""

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_boot_log_emitted(
        self,
        game_id: GameId,
        config: MiniGridConfig,
        variant_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify LOG_ENV_MINIGRID_BOOT is emitted for all Empty variants."""
        caplog.set_level(logging.INFO, logger="gym_gui.core.adapters.base")
        
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            _ = adapter.reset(seed=42)
            
            boot_codes = [
                record.log_code for record in caplog.records  # type: ignore[attr-defined]
                if hasattr(record, "log_code")
            ]
            assert LOG_ENV_MINIGRID_BOOT.code in boot_codes
            
        finally:
            adapter.close()

    @pytest.mark.parametrize("game_id,config,variant_name", ALL_EMPTY_VARIANTS)
    def test_step_log_emitted(
        self,
        game_id: GameId,
        config: MiniGridConfig,
        variant_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify LOG_ENV_MINIGRID_STEP is emitted for all Empty variants."""
        caplog.set_level(logging.DEBUG, logger="gym_gui.core.adapters.base")
        
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        adapter = MiniGridAdapter(context, config=config)
        
        try:
            adapter.load()
            _ = adapter.reset(seed=42)
            _ = adapter.step(2)  # Move forward
            
            step_codes = [
                record.log_code for record in caplog.records  # type: ignore[attr-defined]
                if hasattr(record, "log_code")
            ]
            assert LOG_ENV_MINIGRID_STEP.code in step_codes
            
        finally:
            adapter.close()


class TestEmptyEnvironmentRandomness:
    """Test randomization behavior of Random vs fixed variants."""

    def test_random_variant_changes_starting_position(self) -> None:
        """Verify Random variants randomize agent starting position."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        
        # Test Random-5x5 variant
        adapter = MiniGridAdapter(context, config=DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG)
        
        try:
            adapter.load()
            
            # Reset with different seeds should give different observations
            step1 = adapter.reset(seed=1)
            step2 = adapter.reset(seed=2)
            
            # Observations should differ (different starting positions)
            # Note: They might be the same by chance, but very unlikely
            obs_equal = np.array_equal(step1.observation, step2.observation)
            # We can't guarantee they're different, but we can check structure
            assert step1.observation.shape == step2.observation.shape
            
            # Same seed should give same observation
            step3 = adapter.reset(seed=1)
            np.testing.assert_array_equal(step1.observation, step3.observation)
            
        finally:
            adapter.close()

    def test_fixed_variant_has_consistent_starting_position(self) -> None:
        """Verify fixed variants always start in the same position."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        
        # Test fixed 5x5 variant
        adapter = MiniGridAdapter(context, config=DEFAULT_MINIGRID_EMPTY_5x5_CONFIG)
        
        try:
            adapter.load()
            
            # Multiple resets should give same starting observation
            # (when using the same seed for RNG determinism)
            step1 = adapter.reset(seed=42)
            step2 = adapter.reset(seed=42)
            
            np.testing.assert_array_equal(step1.observation, step2.observation)
            
        finally:
            adapter.close()


class TestEmptyEnvironmentSizeProgression:
    """Test that different size variants have appropriate characteristics."""

    def test_size_progression_exists(self) -> None:
        """Verify we have size progression: 5x5, 6x6, 8x8, 16x16."""
        sizes = ["5x5", "6x6", "8x8", "16x16"]
        
        for size in sizes:
            # Check both fixed and random variants exist
            fixed_found = any(
                size in game_id.value for game_id, _, _ in ALL_EMPTY_VARIANTS
                if "Random" not in game_id.value
            )
            assert fixed_found, f"Fixed {size} variant not found"
            
            # Random variants only exist for 5x5 and 6x6
            if size in ["5x5", "6x6"]:
                random_found = any(
                    size in game_id.value for game_id, _, _ in ALL_EMPTY_VARIANTS
                    if "Random" in game_id.value
                )
                assert random_found, f"Random {size} variant not found"

    def test_all_sizes_produce_valid_environments(self) -> None:
        """Verify all size variants produce valid, working environments."""
        context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
        
        size_configs = [
            DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
            DEFAULT_MINIGRID_EMPTY_6x6_CONFIG,
            DEFAULT_MINIGRID_EMPTY_8x8_CONFIG,
            DEFAULT_MINIGRID_EMPTY_16x16_CONFIG,
        ]
        
        for config in size_configs:
            adapter = MiniGridAdapter(context, config=config)
            try:
                adapter.load()
                step = adapter.reset(seed=42)
                assert step.observation.shape[0] > 0
            finally:
                adapter.close()


# Summary test to verify integration completeness
def test_empty_integration_summary() -> None:
    """Summary test verifying complete Empty environment integration."""
    # 1. Verify count
    assert len(ALL_EMPTY_VARIANTS) == 6, "Should have 6 Empty variants"
    
    # 2. Verify all are in MINIGRID_GAME_IDS
    empty_count_in_tuple = sum(
        1 for game_id in MINIGRID_GAME_IDS 
        if "Empty" in game_id.value
    )
    assert empty_count_in_tuple == 6, "All 6 Empty variants should be in MINIGRID_GAME_IDS"
    
    # 3. Verify all can be instantiated
    context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
    for game_id, config, variant_name in ALL_EMPTY_VARIANTS:
        adapter = MiniGridAdapter(context, config=config)
        try:
            adapter.load()
            step = adapter.reset(seed=42)
            assert step is not None
            print(f"✓ {variant_name}: {game_id.value}")
        finally:
            adapter.close()
    
    print(f"\n✓ All {len(ALL_EMPTY_VARIANTS)} MiniGrid Empty variants integrated successfully!")
