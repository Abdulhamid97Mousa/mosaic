"""End-to-end integration tests for Blackjack-v1 game.

Tests the complete Blackjack integration including:
1. Adapter creation, loading, and basic operations
2. RGB rendering with pygame
3. Render payload structure and serialization
4. Telemetry storage and retrieval
5. Human input keyboard mappings
6. Game info documentation
"""

from __future__ import annotations

import pytest
import numpy as np
from typing import TYPE_CHECKING

from gym_gui.core.enums import GameId, RenderMode, ControlMode
from gym_gui.core.adapters.toy_text import BlackjackAdapter
from gym_gui.config.game_configs import BlackjackConfig, DEFAULT_BLACKJACK_CONFIG
from gym_gui.core.factories.adapters import create_adapter
from gym_gui.game_docs import get_game_info
from gym_gui.utils import json_serialization
from gym_gui.controllers.human_input import _TOY_TEXT_MAPPINGS

if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterStep


class TestBlackjackAdapterBasics:
    """Test basic Blackjack adapter functionality."""

    def test_adapter_can_be_created_directly(self):
        """Test: Blackjack adapter can be created directly."""
        config = DEFAULT_BLACKJACK_CONFIG
        adapter = BlackjackAdapter(game_config=config)
        assert adapter is not None
        assert adapter.id == GameId.BLACKJACK.value

    def test_adapter_can_be_created_via_factory(self):
        """Test: Blackjack adapter can be created via factory."""
        config = DEFAULT_BLACKJACK_CONFIG
        adapter = create_adapter(GameId.BLACKJACK, game_config=config)
        assert adapter is not None
        assert adapter.id == GameId.BLACKJACK.value

    def test_adapter_loads_successfully(self):
        """Blackjack adapter loads environment without errors."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        assert adapter._env is not None

    def test_adapter_resets_successfully(self):
        """Blackjack adapter resets and returns valid observation."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        result = adapter.reset(seed=42)
        
        assert result is not None
        assert result.observation is not None
        # Observation is a tuple: (player_sum, dealer_card, usable_ace)
        assert isinstance(result.observation, tuple)
        assert len(result.observation) == 3
        assert isinstance(result.observation[0], int)  # player_sum
        assert isinstance(result.observation[1], int)  # dealer_card
        assert isinstance(result.observation[2], (bool, int))  # usable_ace

    def test_adapter_step_stick_action(self):
        """Blackjack adapter processes stick action (0) correctly."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        # Action 0 = Stick (stop taking cards)
        result = adapter.step(0)
        assert result is not None
        assert isinstance(result.reward, (int, float))
        # Episode should terminate after stick
        assert result.terminated or result.truncated

    def test_adapter_step_hit_action(self):
        """Blackjack adapter processes hit action (1) correctly."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        # Action 1 = Hit (take another card)
        result = adapter.step(1)
        assert result is not None
        assert isinstance(result.reward, (int, float))
        # Episode may or may not terminate (depends on whether we bust)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)


class TestBlackjackConfiguration:
    """Test Blackjack configuration options."""

    def test_default_config(self):
        """Default Blackjack config has expected values."""
        config = DEFAULT_BLACKJACK_CONFIG
        assert config.natural is False
        assert config.sab is False

    def test_custom_config_natural_true(self):
        """Blackjack config with natural=True."""
        config = BlackjackConfig(natural=True, sab=False)
        adapter = BlackjackAdapter(game_config=config)
        adapter.load()
        result = adapter.reset(seed=42)
        assert result is not None

    def test_custom_config_sab_true(self):
        """Blackjack config with sab=True (Sutton & Barto rules)."""
        config = BlackjackConfig(natural=False, sab=True)
        adapter = BlackjackAdapter(game_config=config)
        adapter.load()
        result = adapter.reset(seed=42)
        assert result is not None

    def test_config_to_gym_kwargs(self):
        """Config.to_gym_kwargs() returns correct dictionary."""
        config = BlackjackConfig(natural=True, sab=False)
        kwargs = config.to_gym_kwargs()
        assert kwargs == {"natural": True, "sab": False}


class TestBlackjackRendering:
    """Test Blackjack RGB rendering with pygame."""

    def test_render_mode_is_rgb_array(self):
        """Blackjack uses rgb_array render mode (not ansi)."""
        adapter = create_adapter(GameId.BLACKJACK)
        assert adapter._gym_render_mode == "rgb_array"  # type: ignore[attr-defined]

    def test_render_returns_valid_payload(self):
        """Render returns a properly structured payload."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert isinstance(payload, dict)
        assert "mode" in payload
        assert payload["mode"] == RenderMode.RGB_ARRAY.value
        assert "rgb" in payload
        assert "game_id" in payload
        assert payload["game_id"] == GameId.BLACKJACK.value

    def test_render_rgb_array_format(self):
        """RGB array has correct shape and dtype."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        rgb = payload["rgb"]
        
        assert isinstance(rgb, np.ndarray)
        assert rgb.ndim == 3
        assert rgb.shape == (500, 600, 3)  # Height, Width, Channels
        assert rgb.dtype == np.uint8

    def test_render_includes_game_state(self):
        """Render payload includes game state metadata."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert "player_sum" in payload
        assert "dealer_card" in payload
        assert "usable_ace" in payload
        assert "terminated" in payload
        assert "truncated" in payload
        
        # Validate types
        assert isinstance(payload["player_sum"], int)
        assert isinstance(payload["dealer_card"], int)
        assert isinstance(payload["usable_ace"], bool)
        assert isinstance(payload["terminated"], bool)
        assert isinstance(payload["truncated"], bool)

    def test_render_payload_key_not_frame(self):
        """Render uses 'rgb' key (not 'frame') for RgbRendererStrategy compatibility."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        assert "rgb" in payload
        assert "frame" not in payload  # Should NOT use 'frame'


class TestBlackjackPayloadSerialization:
    """Test that Blackjack render payloads can be serialized for telemetry."""

    def test_payload_serialization_with_numpy_array(self):
        """Render payload with numpy RGB array can be serialized."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        
        # Test serialization
        serialized = json_serialization.dumps(payload)
        assert serialized is not None
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_payload_deserialization_roundtrip(self):
        """Serialized payload can be deserialized back to original structure."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        original_payload = adapter.render()
        original_rgb = original_payload["rgb"]
        
        # Serialize and deserialize
        serialized = json_serialization.dumps(original_payload)
        deserialized = json_serialization.loads(serialized)
        
        # Verify structure
        assert isinstance(deserialized, dict)
        assert "rgb" in deserialized
        assert "mode" in deserialized
        assert deserialized["mode"] == RenderMode.RGB_ARRAY.value
        
        # Verify RGB array
        recovered_rgb = deserialized["rgb"]
        assert isinstance(recovered_rgb, np.ndarray)
        assert recovered_rgb.shape == original_rgb.shape
        assert recovered_rgb.dtype == original_rgb.dtype
        assert np.array_equal(recovered_rgb, original_rgb)

    def test_payload_metadata_preserved_after_serialization(self):
        """Game state metadata is preserved through serialization."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        original = adapter.render()
        serialized = json_serialization.dumps(original)
        recovered = json_serialization.loads(serialized)
        
        # Check metadata preservation
        assert recovered["player_sum"] == original["player_sum"]
        assert recovered["dealer_card"] == original["dealer_card"]
        assert recovered["usable_ace"] == original["usable_ace"]
        assert recovered["game_id"] == original["game_id"]


class TestBlackjackHumanInput:
    """Test Blackjack human input keyboard mappings."""

    def test_blackjack_has_keyboard_mappings(self):
        """Blackjack is registered in TOY_TEXT_MAPPINGS."""
        assert GameId.BLACKJACK in _TOY_TEXT_MAPPINGS

    def test_blackjack_has_two_actions(self):
        """Blackjack has exactly 2 action mappings (Stick, Hit)."""
        mappings = _TOY_TEXT_MAPPINGS[GameId.BLACKJACK]
        assert len(mappings) == 2

    def test_blackjack_stick_action_mapping(self):
        """Action 0 (Stick) is mapped correctly."""
        mappings = _TOY_TEXT_MAPPINGS[GameId.BLACKJACK]
        stick_mapping = next(m for m in mappings if m.action == 0)
        assert stick_mapping is not None
        # Should have key sequences (e.g., Key_1, Key_Q)
        assert len(stick_mapping.key_sequences) >= 1

    def test_blackjack_hit_action_mapping(self):
        """Action 1 (Hit) is mapped correctly."""
        mappings = _TOY_TEXT_MAPPINGS[GameId.BLACKJACK]
        hit_mapping = next(m for m in mappings if m.action == 1)
        assert hit_mapping is not None
        # Should have key sequences (e.g., Key_2, Key_E)
        assert len(hit_mapping.key_sequences) >= 1

    def test_keyboard_mappings_no_spacebar_conflict(self):
        """Keyboard mappings avoid Spacebar to prevent UI button conflicts."""
        mappings = _TOY_TEXT_MAPPINGS[GameId.BLACKJACK]
        for mapping in mappings:
            for sequence in mapping.key_sequences:
                key_string = sequence.toString()
                # Spacebar should NOT be used
                assert "Space" not in key_string


class TestBlackjackGameInfo:
    """Test Blackjack game info documentation."""

    def test_game_info_exists(self):
        """Blackjack game info is registered."""
        info = get_game_info(GameId.BLACKJACK)
        assert info is not None
        assert len(info) > 0

    def test_game_info_contains_description(self):
        """Game info contains game description."""
        info = get_game_info(GameId.BLACKJACK)
        assert "Blackjack" in info
        assert "dealer" in info.lower()
        assert "21" in info

    def test_game_info_contains_keyboard_controls(self):
        """Game info documents keyboard controls."""
        info = get_game_info(GameId.BLACKJACK)
        assert "Stick" in info or "stick" in info
        assert "Hit" in info or "hit" in info

    def test_game_info_contains_config_params(self):
        """Game info documents configuration parameters."""
        info = get_game_info(GameId.BLACKJACK)
        assert "natural" in info.lower()
        assert "sab" in info.lower()

    def test_game_info_contains_rewards(self):
        """Game info documents reward structure."""
        info = get_game_info(GameId.BLACKJACK)
        assert "+1" in info or "Win" in info
        assert "-1" in info or "Lose" in info


class TestBlackjackEndToEndScenario:
    """End-to-end scenario tests simulating actual gameplay."""

    def test_complete_episode_stick_immediately(self):
        """Play a complete episode by sticking immediately."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        
        # Reset
        reset_result = adapter.reset(seed=42)
        assert not reset_result.terminated
        
        # Render initial state
        initial_render = adapter.render()
        assert initial_render["rgb"] is not None
        
        # Action 0 = Stick
        step_result = adapter.step(0)
        
        # Episode should terminate
        assert step_result.terminated
        # Reward should be +1, -1, or 0
        assert step_result.reward in [-1, 0, 1, 1.5]
        
        # Render final state
        final_render = adapter.render()
        assert final_render["terminated"] is True

    def test_complete_episode_with_multiple_hits(self):
        """Play an episode with multiple hit actions."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=123)
        
        # Hit multiple times until episode ends
        max_steps = 10
        step_count = 0
        terminated = False
        
        while not terminated and step_count < max_steps:
            result = adapter.step(1)  # Action 1 = Hit
            step_count += 1
            terminated = result.terminated or result.truncated
            
            # Verify we can render at each step
            payload = adapter.render()
            assert payload["rgb"] is not None
            assert "player_sum" in payload
        
        # Episode should have ended (either bust or reached goal)
        assert terminated

    def test_payload_serialization_during_gameplay(self):
        """Render payloads can be serialized at every step of gameplay."""
        adapter = create_adapter(GameId.BLACKJACK)
        adapter.load()
        adapter.reset(seed=42)
        
        # Collect payloads from a few steps
        payloads = []
        payloads.append(adapter.render())
        
        # Take 2 hit actions
        adapter.step(1)
        payloads.append(adapter.render())
        
        adapter.step(1)
        payloads.append(adapter.render())
        
        # Verify all payloads can be serialized
        for i, payload in enumerate(payloads):
            serialized = json_serialization.dumps(payload)
            assert serialized is not None, f"Failed to serialize payload {i}"
            
            # Verify deserialization
            recovered = json_serialization.loads(serialized)
            assert recovered["mode"] == RenderMode.RGB_ARRAY.value
            assert isinstance(recovered["rgb"], np.ndarray)


class TestBlackjackEnumIntegration:
    """Test Blackjack integration with core enums."""

    def test_blackjack_in_game_id_enum(self):
        """BLACKJACK is defined in GameId enum."""
        assert hasattr(GameId, "BLACKJACK")
        assert GameId.BLACKJACK.value == "Blackjack-v1"

    def test_blackjack_default_render_mode(self):
        """Blackjack default render mode is RGB_ARRAY."""
        from gym_gui.core.enums import DEFAULT_RENDER_MODES
        assert GameId.BLACKJACK in DEFAULT_RENDER_MODES
        assert DEFAULT_RENDER_MODES[GameId.BLACKJACK] == RenderMode.RGB_ARRAY

    def test_blackjack_environment_family(self):
        """Blackjack belongs to TOY_TEXT family."""
        from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily
        assert GameId.BLACKJACK in ENVIRONMENT_FAMILY_BY_GAME
        assert ENVIRONMENT_FAMILY_BY_GAME[GameId.BLACKJACK] == EnvironmentFamily.TOY_TEXT

    def test_blackjack_control_modes(self):
        """Blackjack supports expected control modes."""
        from gym_gui.core.enums import DEFAULT_CONTROL_MODES
        assert GameId.BLACKJACK in DEFAULT_CONTROL_MODES
        modes = DEFAULT_CONTROL_MODES[GameId.BLACKJACK]
        assert ControlMode.HUMAN_ONLY in modes
        assert ControlMode.AGENT_ONLY in modes
        assert ControlMode.HYBRID_TURN_BASED in modes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
