"""Tests for simultaneous key input and key combination resolvers.

This module tests the state-based key tracking system that enables simultaneous
key combinations for real-time games like Procgen, ALE, and ViZDoom.

The key tracking system solves the problem of Qt's QShortcut mechanism only
detecting individual key presses, not multiple keys held simultaneously.

The input mode is now user-configurable via the Game Configuration panel,
allowing users to choose between:
- State-Based (Real-time): For simultaneous key combinations
- Shortcut-Based (Immediate): For single-key actions
"""

from __future__ import annotations

import pytest
from typing import Set
from unittest.mock import MagicMock, patch

# Import key resolvers and constants
from gym_gui.controllers.human_input import (
    ProcgenKeyCombinationResolver,
    AleKeyCombinationResolver,
    Box2DKeyCombinationResolver,
    ViZDoomKeyCombinationResolver,
    MeltingPotKeyCombinationResolver,
    get_key_combination_resolver,
    KeyCombinationResolver,
    _KEYS_UP,
    _KEYS_DOWN,
    _KEYS_LEFT,
    _KEYS_RIGHT,
    _KEY_SPACE,
    _KEY_UP,
    _KEY_DOWN,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_W,
    _KEY_A,
    _KEY_S,
    _KEY_D,
    _KEY_Q,
    _KEY_E,
    _KEY_G,
    _KEY_Z,
    _KEY_C,
    _KEY_X,
    _KEY_1,
    _KEY_2,
    _KEY_3,
)
from gym_gui.core.enums import EnvironmentFamily, GameId, InputMode, INPUT_MODE_INFO


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def procgen_resolver() -> ProcgenKeyCombinationResolver:
    """Create a Procgen key combination resolver."""
    return ProcgenKeyCombinationResolver()


@pytest.fixture
def ale_resolver() -> AleKeyCombinationResolver:
    """Create an ALE key combination resolver."""
    return AleKeyCombinationResolver()


@pytest.fixture
def box2d_resolver() -> Box2DKeyCombinationResolver:
    """Create a Box2D key combination resolver."""
    return Box2DKeyCombinationResolver()


# =============================================================================
# Procgen Resolver Tests
# =============================================================================

class TestProcgenKeyCombinationResolver:
    """Test key combination resolution for Procgen environments.

    Procgen action space (15 actions):
    0: down_left, 1: left, 2: up_left, 3: down, 4: noop, 5: up,
    6: down_right, 7: right, 8: up_right,
    9-14: action buttons
    """

    # ----- Diagonal Movement Tests -----

    def test_up_right_arrow_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Up+Right arrow keys produce up_right action (8)."""
        pressed = {_KEY_UP, _KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) == 8

    def test_up_left_arrow_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Up+Left arrow keys produce up_left action (2)."""
        pressed = {_KEY_UP, _KEY_LEFT}
        assert procgen_resolver.resolve(pressed) == 2

    def test_down_right_arrow_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Down+Right arrow keys produce down_right action (6)."""
        pressed = {_KEY_DOWN, _KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) == 6

    def test_down_left_arrow_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Down+Left arrow keys produce down_left action (0)."""
        pressed = {_KEY_DOWN, _KEY_LEFT}
        assert procgen_resolver.resolve(pressed) == 0

    def test_up_right_wasd_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test W+D keys produce up_right action (8)."""
        pressed = {_KEY_W, _KEY_D}
        assert procgen_resolver.resolve(pressed) == 8

    def test_up_left_wasd_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test W+A keys produce up_left action (2)."""
        pressed = {_KEY_W, _KEY_A}
        assert procgen_resolver.resolve(pressed) == 2

    def test_mixed_wasd_and_arrows(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test mixing WASD and arrow keys still works."""
        # W (up) + Right arrow
        pressed = {_KEY_W, _KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) == 8  # up_right

    # ----- Cardinal Direction Tests -----

    def test_up_only(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Up arrow produces up action (5)."""
        pressed = {_KEY_UP}
        assert procgen_resolver.resolve(pressed) == 5

    def test_down_only(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Down arrow produces down action (3)."""
        pressed = {_KEY_DOWN}
        assert procgen_resolver.resolve(pressed) == 3

    def test_left_only(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Left arrow produces left action (1)."""
        pressed = {_KEY_LEFT}
        assert procgen_resolver.resolve(pressed) == 1

    def test_right_only(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Right arrow produces right action (7)."""
        pressed = {_KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) == 7

    # ----- Direction Cancellation Tests -----

    def test_up_down_cancel(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Up+Down cancel each other out."""
        pressed = {_KEY_UP, _KEY_DOWN}
        # Should return None (no valid action) or fire button if pressed
        assert procgen_resolver.resolve(pressed) is None

    def test_left_right_cancel(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Left+Right cancel each other out."""
        pressed = {_KEY_LEFT, _KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) is None

    def test_all_directions_cancel(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test all four directions cancel out."""
        pressed = {_KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT}
        assert procgen_resolver.resolve(pressed) is None

    # ----- Action Button Tests -----

    def test_space_fires(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test Space bar produces fire action (9)."""
        pressed = {_KEY_SPACE}
        assert procgen_resolver.resolve(pressed) == 9

    # ----- Empty Set Test -----

    def test_no_keys_pressed(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test empty key set returns None."""
        pressed: Set[int] = set()
        assert procgen_resolver.resolve(pressed) is None


# =============================================================================
# ALE Resolver Tests
# =============================================================================

class TestAleKeyCombinationResolver:
    """Test key combination resolution for ALE/Atari environments.

    Standard ALE action space (18 actions):
    0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN,
    6: UPRIGHT, 7: UPLEFT, 8: DOWNRIGHT, 9: DOWNLEFT,
    10: UPFIRE, 11: RIGHTFIRE, 12: LEFTFIRE, 13: DOWNFIRE,
    14: UPRIGHTFIRE, 15: UPLEFTFIRE, 16: DOWNRIGHTFIRE, 17: DOWNLEFTFIRE
    """

    # ----- Diagonal Movement Without Fire -----

    def test_up_right(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up+Right produces UPRIGHT action (6)."""
        pressed = {_KEY_UP, _KEY_RIGHT}
        assert ale_resolver.resolve(pressed) == 6

    def test_up_left(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up+Left produces UPLEFT action (7)."""
        pressed = {_KEY_UP, _KEY_LEFT}
        assert ale_resolver.resolve(pressed) == 7

    def test_down_right(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down+Right produces DOWNRIGHT action (8)."""
        pressed = {_KEY_DOWN, _KEY_RIGHT}
        assert ale_resolver.resolve(pressed) == 8

    def test_down_left(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down+Left produces DOWNLEFT action (9)."""
        pressed = {_KEY_DOWN, _KEY_LEFT}
        assert ale_resolver.resolve(pressed) == 9

    # ----- Diagonal Movement With Fire -----

    def test_up_right_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up+Right+Space produces UPRIGHTFIRE action (14)."""
        pressed = {_KEY_UP, _KEY_RIGHT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 14

    def test_up_left_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up+Left+Space produces UPLEFTFIRE action (15)."""
        pressed = {_KEY_UP, _KEY_LEFT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 15

    def test_down_right_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down+Right+Space produces DOWNRIGHTFIRE action (16)."""
        pressed = {_KEY_DOWN, _KEY_RIGHT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 16

    def test_down_left_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down+Left+Space produces DOWNLEFTFIRE action (17)."""
        pressed = {_KEY_DOWN, _KEY_LEFT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 17

    # ----- Cardinal Direction With Fire -----

    def test_up_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up+Space produces UPFIRE action (10)."""
        pressed = {_KEY_UP, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 10

    def test_right_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Right+Space produces RIGHTFIRE action (11)."""
        pressed = {_KEY_RIGHT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 11

    def test_left_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Left+Space produces LEFTFIRE action (12)."""
        pressed = {_KEY_LEFT, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 12

    def test_down_fire(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down+Space produces DOWNFIRE action (13)."""
        pressed = {_KEY_DOWN, _KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 13

    # ----- Fire Only -----

    def test_fire_only(self, ale_resolver: AleKeyCombinationResolver):
        """Test Space alone produces FIRE action (1)."""
        pressed = {_KEY_SPACE}
        assert ale_resolver.resolve(pressed) == 1

    # ----- Cardinal Directions -----

    def test_up(self, ale_resolver: AleKeyCombinationResolver):
        """Test Up produces UP action (2)."""
        pressed = {_KEY_UP}
        assert ale_resolver.resolve(pressed) == 2

    def test_right(self, ale_resolver: AleKeyCombinationResolver):
        """Test Right produces RIGHT action (3)."""
        pressed = {_KEY_RIGHT}
        assert ale_resolver.resolve(pressed) == 3

    def test_left(self, ale_resolver: AleKeyCombinationResolver):
        """Test Left produces LEFT action (4)."""
        pressed = {_KEY_LEFT}
        assert ale_resolver.resolve(pressed) == 4

    def test_down(self, ale_resolver: AleKeyCombinationResolver):
        """Test Down produces DOWN action (5)."""
        pressed = {_KEY_DOWN}
        assert ale_resolver.resolve(pressed) == 5

    # ----- No Keys -----

    def test_no_keys(self, ale_resolver: AleKeyCombinationResolver):
        """Test empty key set returns None (NOOP)."""
        pressed: Set[int] = set()
        assert ale_resolver.resolve(pressed) is None


# =============================================================================
# Box2D Resolver Tests
# =============================================================================

class TestBox2DKeyCombinationResolver:
    """Test key combination resolution for Box2D environments (LunarLander)."""

    def test_up_fires_main_engine(self, box2d_resolver: Box2DKeyCombinationResolver):
        """Test Up produces main engine action (2)."""
        pressed = {_KEY_UP}
        assert box2d_resolver.resolve(pressed) == 2

    def test_left_fires_left_engine(self, box2d_resolver: Box2DKeyCombinationResolver):
        """Test Left produces left engine action (1)."""
        pressed = {_KEY_LEFT}
        assert box2d_resolver.resolve(pressed) == 1

    def test_right_fires_right_engine(self, box2d_resolver: Box2DKeyCombinationResolver):
        """Test Right produces right engine action (3)."""
        pressed = {_KEY_RIGHT}
        assert box2d_resolver.resolve(pressed) == 3

    def test_main_engine_priority(self, box2d_resolver: Box2DKeyCombinationResolver):
        """Test main engine (Up) has priority over side engines."""
        # Up + Left should fire main engine, not left
        pressed = {_KEY_UP, _KEY_LEFT}
        assert box2d_resolver.resolve(pressed) == 2  # Main engine

    def test_no_keys_idle(self, box2d_resolver: Box2DKeyCombinationResolver):
        """Test no keys produces None (idle)."""
        pressed: Set[int] = set()
        assert box2d_resolver.resolve(pressed) is None


# =============================================================================
# Resolver Factory Tests
# =============================================================================

class TestGetKeyCombinationResolver:
    """Test the resolver factory function."""

    def test_procgen_game_gets_procgen_resolver(self):
        """Test Procgen games get ProcgenKeyCombinationResolver."""
        resolver = get_key_combination_resolver(GameId.PROCGEN_COINRUN)
        assert isinstance(resolver, ProcgenKeyCombinationResolver)

    def test_ale_game_gets_ale_resolver(self):
        """Test ALE games get AleKeyCombinationResolver."""
        resolver = get_key_combination_resolver(GameId.ALE_SPACE_INVADERS_V5)
        assert isinstance(resolver, AleKeyCombinationResolver)

    def test_lunarlander_gets_box2d_resolver(self):
        """Test LunarLander gets Box2DKeyCombinationResolver."""
        resolver = get_key_combination_resolver(GameId.LUNAR_LANDER)
        assert isinstance(resolver, Box2DKeyCombinationResolver)

    def test_unsupported_game_returns_none(self):
        """Test unsupported games return None."""
        # FrozenLake is a turn-based game, should not have a resolver
        resolver = get_key_combination_resolver(GameId.FROZEN_LAKE)
        assert resolver is None


# =============================================================================
# Input Mode User Configuration Tests
# =============================================================================

class TestInputModeConfiguration:
    """Test that input mode is user-configurable for all games."""

    def test_input_mode_enum_values(self):
        """Test InputMode enum has correct values."""
        assert InputMode.STATE_BASED.value == "state_based"
        assert InputMode.SHORTCUT_BASED.value == "shortcut_based"

    def test_input_mode_info_complete(self):
        """Test INPUT_MODE_INFO has entries for all modes."""
        for mode in InputMode:
            assert mode in INPUT_MODE_INFO
            label, description = INPUT_MODE_INFO[mode]
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(description, str) and len(description) > 0

    def test_procgen_has_resolver(self):
        """Test Procgen games have a key combination resolver."""
        resolver = get_key_combination_resolver(GameId.PROCGEN_COINRUN)
        assert isinstance(resolver, ProcgenKeyCombinationResolver)

    def test_ale_has_resolver(self):
        """Test ALE games have a key combination resolver."""
        resolver = get_key_combination_resolver(GameId.ALE_SPACE_INVADERS_V5)
        assert isinstance(resolver, AleKeyCombinationResolver)

    def test_box2d_has_resolver(self):
        """Test Box2D games have a key combination resolver."""
        resolver = get_key_combination_resolver(GameId.LUNAR_LANDER)
        assert isinstance(resolver, Box2DKeyCombinationResolver)

    def test_toy_text_no_resolver(self):
        """Test ToyText games don't have a key combination resolver."""
        resolver = get_key_combination_resolver(GameId.FROZEN_LAKE)
        assert resolver is None

    def test_minigrid_has_resolver(self):
        """Test MiniGrid games now have a key combination resolver for multi-keyboard support."""
        from gym_gui.controllers.human_input import MiniGridKeyCombinationResolver
        resolver = get_key_combination_resolver(GameId.MINIGRID_EMPTY_RANDOM_5x5)
        assert isinstance(resolver, MiniGridKeyCombinationResolver)




# =============================================================================
# Key Set Helper Tests
# =============================================================================

class TestKeySetHelpers:
    """Test the key set constants."""

    def test_keys_up_contains_arrow_and_w(self):
        """Test _KEYS_UP contains both Up arrow and W."""
        assert _KEY_UP in _KEYS_UP
        assert _KEY_W in _KEYS_UP
        assert len(_KEYS_UP) == 2

    def test_keys_down_contains_arrow_and_s(self):
        """Test _KEYS_DOWN contains both Down arrow and S."""
        assert _KEY_DOWN in _KEYS_DOWN
        assert _KEY_S in _KEYS_DOWN
        assert len(_KEYS_DOWN) == 2

    def test_keys_left_contains_arrow_and_a(self):
        """Test _KEYS_LEFT contains both Left arrow and A."""
        assert _KEY_LEFT in _KEYS_LEFT
        assert _KEY_A in _KEYS_LEFT
        assert len(_KEYS_LEFT) == 2

    def test_keys_right_contains_arrow_and_d(self):
        """Test _KEYS_RIGHT contains both Right arrow and D."""
        assert _KEY_RIGHT in _KEYS_RIGHT
        assert _KEY_D in _KEYS_RIGHT
        assert len(_KEYS_RIGHT) == 2


# =============================================================================
# Integration Tests (with mocked Qt)
# =============================================================================

class TestHumanInputControllerIntegration:
    """Integration tests for HumanInputController state tracking."""

    def test_configure_with_state_based_override(self):
        """Test configure respects input_mode override."""
        # This test requires mocking Qt components
        # The actual integration is tested via manual testing
        pass  # Placeholder for manual test reference

    def test_configure_with_shortcut_based_override(self):
        """Test configure respects shortcut_based input_mode override."""
        pass  # Placeholder for manual test reference


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_resolver_handles_unknown_keys(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test resolver ignores unknown key codes."""
        pressed = {999999}  # Unknown key code
        assert procgen_resolver.resolve(pressed) is None

    def test_resolver_handles_mixed_known_unknown(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test resolver works with mix of known and unknown keys."""
        pressed = {_KEY_UP, 999999}
        assert procgen_resolver.resolve(pressed) == 5  # Up action

    def test_diagonal_priority_over_cardinal(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test that diagonal combinations are detected before cardinal directions."""
        # If up+right are pressed, should return diagonal, not just up or right
        pressed = {_KEY_UP, _KEY_RIGHT}
        result = procgen_resolver.resolve(pressed)
        assert result == 8  # up_right, not 5 (up) or 7 (right)

    def test_wasd_equivalence(self, procgen_resolver: ProcgenKeyCombinationResolver):
        """Test WASD keys produce same results as arrow keys."""
        # W+D should equal Up+Right
        wasd_result = procgen_resolver.resolve({_KEY_W, _KEY_D})
        arrow_result = procgen_resolver.resolve({_KEY_UP, _KEY_RIGHT})
        assert wasd_result == arrow_result == 8


# =============================================================================
# MeltingPot Resolver Tests
# =============================================================================

@pytest.fixture
def meltingpot_resolver_8() -> MeltingPotKeyCombinationResolver:
    """Create a MeltingPot resolver with 8 actions (basic substrates)."""
    return MeltingPotKeyCombinationResolver(num_actions=8)


@pytest.fixture
def meltingpot_resolver_11() -> MeltingPotKeyCombinationResolver:
    """Create a MeltingPot resolver with 11 actions (e.g., clean_up)."""
    return MeltingPotKeyCombinationResolver(num_actions=11)


class TestMeltingPotKeyCombinationResolver:
    """Test key combination resolution for MeltingPot multi-agent environments.

    MeltingPot action space (varies by substrate):
    0: NOOP - Do nothing
    1: FORWARD - Move forward
    2: BACKWARD - Move backward
    3: LEFT - Strafe left
    4: RIGHT - Strafe right
    5: TURN_LEFT - Turn left
    6: TURN_RIGHT - Turn right
    7: INTERACT - Interact/use
    8: FIRE_1 - Secondary action (e.g., fireZap) - only in 9+ action substrates
    9: FIRE_2 - Tertiary action (e.g., fireClean) - only in 10+ action substrates
    10: FIRE_3 - Quaternary action - only in 11+ action substrates
    """

    # ----- Movement Tests -----

    def test_forward_w_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test W key produces FORWARD action (1)."""
        pressed = {_KEY_W}
        assert meltingpot_resolver_8.resolve(pressed) == 1

    def test_forward_up_arrow(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test Up arrow produces FORWARD action (1)."""
        pressed = {_KEY_UP}
        assert meltingpot_resolver_8.resolve(pressed) == 1

    def test_backward_s_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test S key produces BACKWARD action (2)."""
        pressed = {_KEY_S}
        assert meltingpot_resolver_8.resolve(pressed) == 2

    def test_backward_down_arrow(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test Down arrow produces BACKWARD action (2)."""
        pressed = {_KEY_DOWN}
        assert meltingpot_resolver_8.resolve(pressed) == 2

    def test_strafe_left_a_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test A key produces LEFT strafe action (3)."""
        pressed = {_KEY_A}
        assert meltingpot_resolver_8.resolve(pressed) == 3

    def test_strafe_right_d_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test D key produces RIGHT strafe action (4)."""
        pressed = {_KEY_D}
        assert meltingpot_resolver_8.resolve(pressed) == 4

    # ----- Turning Tests -----

    def test_turn_left_q_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test Q key produces TURN_LEFT action (5)."""
        pressed = {_KEY_Q}
        assert meltingpot_resolver_8.resolve(pressed) == 5

    def test_turn_right_e_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test E key produces TURN_RIGHT action (6)."""
        pressed = {_KEY_E}
        assert meltingpot_resolver_8.resolve(pressed) == 6

    # ----- Interact Tests -----

    def test_interact_space_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test Space key produces INTERACT action (7)."""
        pressed = {_KEY_SPACE}
        assert meltingpot_resolver_8.resolve(pressed) == 7

    def test_interact_g_key(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test G key produces INTERACT action (7)."""
        pressed = {_KEY_G}
        assert meltingpot_resolver_8.resolve(pressed) == 7

    # ----- Extended Action Tests (11-action substrates) -----

    def test_fire1_z_key_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test Z key produces FIRE_1 action (8) in 11-action substrates."""
        pressed = {_KEY_Z}
        assert meltingpot_resolver_11.resolve(pressed) == 8

    def test_fire1_key1_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test 1 key produces FIRE_1 action (8) in 11-action substrates."""
        pressed = {_KEY_1}
        assert meltingpot_resolver_11.resolve(pressed) == 8

    def test_fire2_c_key_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test C key produces FIRE_2 action (9) in 11-action substrates."""
        pressed = {_KEY_C}
        assert meltingpot_resolver_11.resolve(pressed) == 9

    def test_fire2_key2_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test 2 key produces FIRE_2 action (9) in 11-action substrates."""
        pressed = {_KEY_2}
        assert meltingpot_resolver_11.resolve(pressed) == 9

    def test_fire3_x_key_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test X key produces FIRE_3 action (10) in 11-action substrates."""
        pressed = {_KEY_X}
        assert meltingpot_resolver_11.resolve(pressed) == 10

    def test_fire3_key3_11_actions(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test 3 key produces FIRE_3 action (10) in 11-action substrates."""
        pressed = {_KEY_3}
        assert meltingpot_resolver_11.resolve(pressed) == 10

    # ----- Action Space Boundary Tests -----

    def test_fire1_ignored_in_8_action_substrate(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test Z key is ignored in 8-action substrates (no FIRE_1 available)."""
        pressed = {_KEY_Z}
        # Should return None since FIRE_1 (action 8) doesn't exist in 8-action space
        assert meltingpot_resolver_8.resolve(pressed) is None

    def test_fire2_ignored_in_8_action_substrate(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test C key is ignored in 8-action substrates (no FIRE_2 available)."""
        pressed = {_KEY_C}
        assert meltingpot_resolver_8.resolve(pressed) is None

    def test_fire3_ignored_in_8_action_substrate(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test X key is ignored in 8-action substrates (no FIRE_3 available)."""
        pressed = {_KEY_X}
        assert meltingpot_resolver_8.resolve(pressed) is None

    # ----- Priority Tests -----

    def test_fire_priority_over_movement(self, meltingpot_resolver_11: MeltingPotKeyCombinationResolver):
        """Test fire actions have priority over movement in 11-action substrates."""
        # Z + W pressed: Z (FIRE_1) should take priority
        pressed = {_KEY_Z, _KEY_W}
        assert meltingpot_resolver_11.resolve(pressed) == 8  # FIRE_1

    def test_interact_priority_over_movement(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test interact has priority over movement."""
        # Space + W pressed: Space (INTERACT) should take priority
        pressed = {_KEY_SPACE, _KEY_W}
        assert meltingpot_resolver_8.resolve(pressed) == 7  # INTERACT

    def test_turn_priority_over_movement(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test turning has priority over movement."""
        # Q + W pressed: Q (TURN_LEFT) should take priority
        pressed = {_KEY_Q, _KEY_W}
        assert meltingpot_resolver_8.resolve(pressed) == 5  # TURN_LEFT

    # ----- No Keys Test -----

    def test_no_keys_returns_none(self, meltingpot_resolver_8: MeltingPotKeyCombinationResolver):
        """Test empty key set returns None (caller should use NOOP action 0)."""
        pressed: Set[int] = set()
        assert meltingpot_resolver_8.resolve(pressed) is None


class TestMeltingPotResolverFactory:
    """Test that MeltingPot games get correct resolvers via factory."""

    def test_meltingpot_game_gets_resolver(self):
        """Test MeltingPot games get MeltingPotKeyCombinationResolver."""
        resolver = get_key_combination_resolver(GameId.MELTINGPOT_CLEAN_UP)
        assert isinstance(resolver, MeltingPotKeyCombinationResolver)

    def test_meltingpot_resolver_with_action_space(self):
        """Test MeltingPot resolver respects action space size."""
        from gymnasium import spaces

        # Simulate an 11-action space (like clean_up)
        action_space = spaces.Discrete(11)
        resolver = get_key_combination_resolver(
            GameId.MELTINGPOT_CLEAN_UP,
            action_space=action_space,
        )

        assert isinstance(resolver, MeltingPotKeyCombinationResolver)
        # Should allow FIRE_1 (action 8)
        assert resolver.resolve({_KEY_Z}) == 8

    def test_meltingpot_resolver_with_8_action_space(self):
        """Test MeltingPot resolver with 8-action space ignores extended actions."""
        from gymnasium import spaces

        action_space = spaces.Discrete(8)
        resolver = get_key_combination_resolver(
            GameId.MELTINGPOT_COINS,  # Basic substrate
            action_space=action_space,
        )

        assert isinstance(resolver, MeltingPotKeyCombinationResolver)
        # Should NOT allow FIRE_1 (action 8) since action space is only 8
        assert resolver.resolve({_KEY_Z}) is None
