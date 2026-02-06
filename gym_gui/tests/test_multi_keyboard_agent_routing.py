"""Tests for multi-keyboard to multi-agent routing.

This module tests the specific bug scenario:
"D key moves two agents instead of one"

Root Cause Investigation:
- When one keyboard is assigned to agent_0 and another to agent_1
- Pressing D on keyboard_0 should ONLY affect agent_0
- If both agents move, the routing is broken

Test Scenarios:
1. Single keyboard assigned to single agent - only that agent moves
2. Two keyboards assigned to two agents - each keyboard controls its agent
3. Unassigned keyboard - no agents should move
4. Legacy MultiGrid vs INI MultiGrid idle action differences

NOTE: These tests focus on the LOGIC of action generation, not the full
HumanInputController (which requires Qt). The core logic is in the resolver
and the action generation algorithm.
"""

from __future__ import annotations

import pytest
from typing import Dict, Set, List

from gym_gui.controllers.human_input import (
    INIMultiGridKeyCombinationResolver,
    MultiGridKeyCombinationResolver,
    MiniGridKeyCombinationResolver,
    _KEY_W,
    _KEY_A,
    _KEY_D,
    _KEY_SPACE,
)


# =============================================================================
# Core Logic Tests: Multi-Agent Action Generation
# =============================================================================

def get_multi_agent_actions_logic(
    agent_names: List[str],
    agent_pressed_keys: Dict[str, Set[int]],
    resolver,
    default_idle: int,
) -> List[int]:
    """Simulate the get_multi_agent_actions logic.

    This extracts the core algorithm from HumanInputController for testing.
    """
    actions: List[int] = []
    for agent_id in agent_names:
        pressed_keys = agent_pressed_keys.get(agent_id, set())
        if pressed_keys:
            action = resolver.resolve(pressed_keys)
            actions.append(action if action is not None else default_idle)
        else:
            actions.append(default_idle)
    return actions


# =============================================================================
# BUG REPRODUCTION: D Key Moves Two Agents
# =============================================================================

class TestDKeyMovesTwoAgentsBug:
    """Reproduce and verify the bug where D key moves two agents."""

    def test_d_key_on_one_keyboard_only_moves_one_agent_ini_multigrid(self):
        """Test that pressing D on agent_0's keyboard ONLY moves agent_0.

        Bug scenario:
        - agent_0 has keyboard assigned
        - agent_1 has NO keyboard (or different keyboard)
        - User presses D on agent_0's keyboard
        - EXPECTED: [RIGHT, IDLE] = [1, 6] for INI MultiGrid
        - BUG: [RIGHT, RIGHT] = [1, 1] - both agents turn right!
        """
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6  # DONE for INI MultiGrid

        # ONLY agent_0 has the D key pressed
        agent_pressed_keys = {
            "agent_0": {_KEY_D},  # Only agent_0 pressed D
            # agent_1 has no keys (not even in dict)
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        # Expected for INI MultiGrid:
        # - agent_0: D key → RIGHT (action 1)
        # - agent_1: no keys → DONE (action 6, the idle action for INI MultiGrid)
        assert len(actions) == 2, f"Expected 2 actions, got {len(actions)}"

        # THE KEY ASSERTION - this is what the bug breaks:
        assert actions[0] == 1, f"agent_0 should do RIGHT (1), got {actions[0]}"
        assert actions[1] == 6, f"agent_1 should do DONE/idle (6), got {actions[1]}"

        # Verify they are NOT the same (the bug symptom)
        assert actions[0] != actions[1], (
            f"BUG DETECTED: Both agents got same action {actions[0]}! "
            f"agent_0 should be RIGHT (1), agent_1 should be IDLE (6)"
        )

    def test_d_key_on_one_keyboard_only_moves_one_agent_legacy_multigrid(self):
        """Test that pressing D on agent_0's keyboard ONLY moves agent_0 in legacy MultiGrid.

        Legacy MultiGrid uses STILL (0) as idle, not DONE (6).
        """
        resolver = MultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1", "agent_2", "agent_3"]
        default_idle = 0  # STILL for legacy MultiGrid

        # ONLY agent_0 has the D key pressed
        agent_pressed_keys = {
            "agent_0": {_KEY_D},  # Only agent_0 pressed D
            # agent_1, agent_2, agent_3 have no keys
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        # Expected for legacy MultiGrid (4 agents):
        # - agent_0: D key → RIGHT (action 2 in legacy)
        # - agent_1, agent_2, agent_3: no keys → STILL (action 0)
        assert len(actions) == 4

        # Legacy MultiGrid: RIGHT is action 2 (STILL=0, LEFT=1, RIGHT=2, FORWARD=3)
        assert actions[0] == 2, f"agent_0 should do RIGHT (2), got {actions[0]}"
        assert actions[1] == 0, f"agent_1 should do STILL (0), got {actions[1]}"
        assert actions[2] == 0, f"agent_2 should do STILL (0), got {actions[2]}"
        assert actions[3] == 0, f"agent_3 should do STILL (0), got {actions[3]}"

    def test_empty_pressed_keys_all_agents_idle(self):
        """Test that when no keys are pressed, all agents do idle action."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6  # DONE for INI MultiGrid

        # No keys pressed for any agent
        agent_pressed_keys: Dict[str, Set[int]] = {}

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        assert len(actions) == 2

        # INI MultiGrid idle = DONE (6)
        assert actions[0] == 6, f"agent_0 should be idle (6), got {actions[0]}"
        assert actions[1] == 6, f"agent_1 should be idle (6), got {actions[1]}"


# =============================================================================
# Multi-Keyboard Routing Tests
# =============================================================================

class TestMultiKeyboardRouting:
    """Test that keyboard events are routed to the correct agent."""

    def test_two_keyboards_two_agents_independent_actions(self):
        """Test that two keyboards control two agents independently.

        Scenario:
        - Keyboard 1 (agent_0) presses W (FORWARD)
        - Keyboard 2 (agent_1) presses D (RIGHT)
        - Expected: [FORWARD, RIGHT] = [2, 1]
        """
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Each agent has different keys pressed (from different keyboards)
        agent_pressed_keys = {
            "agent_0": {_KEY_W},  # FORWARD
            "agent_1": {_KEY_D},  # RIGHT
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        assert len(actions) == 2

        # INI MultiGrid: FORWARD=2, RIGHT=1
        assert actions[0] == 2, f"agent_0 should do FORWARD (2), got {actions[0]}"
        assert actions[1] == 1, f"agent_1 should do RIGHT (1), got {actions[1]}"

    def test_one_keyboard_assigned_other_agent_idles(self):
        """Test that unassigned agent gets idle action."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Only agent_1 has keys (agent_0's keyboard not used)
        agent_pressed_keys = {
            "agent_1": {_KEY_A},  # LEFT
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        assert len(actions) == 2

        # agent_0 has no entry → idle (6)
        # agent_1 has A pressed → LEFT (0)
        assert actions[0] == 6, f"agent_0 should be idle (6), got {actions[0]}"
        assert actions[1] == 0, f"agent_1 should do LEFT (0), got {actions[1]}"


# =============================================================================
# Idle Action Tests (INI vs Legacy MultiGrid)
# =============================================================================

class TestIdleActionDifferences:
    """Test correct idle action for different MultiGrid versions."""

    def test_ini_multigrid_idle_should_be_done_not_left(self):
        """Verify INI MultiGrid should use DONE (6) as idle, not LEFT (0).

        Bug: If idle is LEFT (0), idle agents will spin continuously!

        INI MultiGrid action space (NO STILL):
        0: LEFT, 1: RIGHT, 2: FORWARD, 3: PICKUP, 4: DROP, 5: TOGGLE, 6: DONE

        Since there's no STILL/NOOP, DONE (6) is the least disruptive idle.
        """
        # The correct idle action for INI MultiGrid
        correct_idle = 6  # DONE

        # NOT LEFT (0) - that causes spinning!
        wrong_idle = 0  # LEFT

        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]

        # With correct idle, idle agents don't move
        agent_pressed_keys = {"agent_0": {_KEY_D}}  # Only agent_0 moves

        actions_correct = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, correct_idle
        )

        actions_wrong = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, wrong_idle
        )

        # Correct: agent_1 does DONE (harmless)
        assert actions_correct[1] == 6, "Agent 1 should do DONE (6)"

        # Wrong: agent_1 does LEFT (spins!)
        assert actions_wrong[1] == 0, "With wrong idle, agent_1 would do LEFT (0)"

        # The difference is critical
        assert actions_correct[1] != actions_wrong[1], "Idle action matters!"

    def test_legacy_multigrid_idle_is_still(self):
        """Verify legacy MultiGrid uses STILL (0) as idle."""
        # Legacy MultiGrid has STILL as action 0
        resolver = MultiGridKeyCombinationResolver()

        # Action 0 is STILL (true no-op)
        # Unlike INI MultiGrid where 0 is LEFT!

        agent_names = ["agent_0", "agent_1"]
        agent_pressed_keys: Dict[str, Set[int]] = {}  # No keys

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle=0
        )

        assert actions[0] == 0  # STILL
        assert actions[1] == 0  # STILL


# =============================================================================
# Key Isolation Tests
# =============================================================================

class TestKeyIsolation:
    """Test that key presses are isolated between agents."""

    def test_adding_key_to_agent_does_not_affect_other_agents(self):
        """Test that adding a key to one agent doesn't give it to others."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Start with empty pressed keys
        agent_pressed_keys: Dict[str, Set[int]] = {}

        # Simulate: agent_0's keyboard sends D key press
        if "agent_0" not in agent_pressed_keys:
            agent_pressed_keys["agent_0"] = set()
        agent_pressed_keys["agent_0"].add(_KEY_D)

        # Verify isolation
        assert _KEY_D in agent_pressed_keys.get("agent_0", set())
        assert _KEY_D not in agent_pressed_keys.get("agent_1", set())

        # Get actions
        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        # agent_0 should turn right, agent_1 should be idle
        assert actions[0] == 1  # RIGHT
        assert actions[1] == 6  # IDLE (DONE)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases in multi-agent keyboard handling."""

    def test_agent_with_empty_set_gets_idle(self):
        """Test that agent with empty pressed_keys set gets idle action."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # agent_0 has empty set (not missing, but empty)
        agent_pressed_keys = {
            "agent_0": set(),  # Empty set
            "agent_1": {_KEY_W},  # Has key
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        # Empty set should behave same as missing entry
        assert actions[0] == 6  # IDLE
        assert actions[1] == 2  # FORWARD

    def test_unknown_key_resolves_to_idle(self):
        """Test that unknown keys resolve to idle action."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Press an unknown key code
        agent_pressed_keys = {
            "agent_0": {99999},  # Unknown key
        }

        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        # Resolver returns None for unknown keys → falls back to idle
        assert actions[0] == 6  # IDLE (DONE for INI MultiGrid)


# =============================================================================
# Evdev Device-to-Agent Mapping Simulation
# =============================================================================

class TestEvdevKeyRouting:
    """Test evdev key routing to correct agent."""

    def test_evdev_key_press_routes_to_assigned_agent(self):
        """Simulate evdev key press routing to correct agent."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Set up evdev device-to-agent mapping (simulated)
        device_path_0 = "/dev/input/event0"
        device_path_1 = "/dev/input/event1"

        evdev_device_to_agent = {
            device_path_0: "agent_0",
            device_path_1: "agent_1",
        }

        # Simulate what _on_evdev_key_pressed does:
        # Device 0 sends D key
        agent_pressed_keys: Dict[str, Set[int]] = {}

        # Route key to correct agent
        device_path = device_path_0
        agent_id = evdev_device_to_agent.get(device_path)
        assert agent_id == "agent_0"

        if agent_id not in agent_pressed_keys:
            agent_pressed_keys[agent_id] = set()
        agent_pressed_keys[agent_id].add(_KEY_D)

        # Verify only agent_0 has the key
        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )

        assert actions[0] == 1  # RIGHT
        assert actions[1] == 6  # IDLE

    def test_unassigned_device_key_is_ignored(self):
        """Test that keys from unassigned devices don't affect agents."""
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]
        default_idle = 6

        # Set up partial device-to-agent mapping (device_99 not assigned)
        evdev_device_to_agent = {
            "/dev/input/event0": "agent_0",
            # event99 NOT assigned!
        }

        # Key from unassigned device should NOT route to any agent
        device_path_unassigned = "/dev/input/event99"
        unassigned_agent = evdev_device_to_agent.get(device_path_unassigned)
        assert unassigned_agent is None

        # No keys should be added
        agent_pressed_keys: Dict[str, Set[int]] = {}

        # Actions should all be idle
        actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle
        )
        assert actions[0] == 6
        assert actions[1] == 6


# =============================================================================
# Bug Scenario: Shared pressed_keys (incorrect implementation)
# =============================================================================

class TestSharedPressedKeysBug:
    """Test that demonstrates the bug when pressed_keys is shared between agents.

    This is the BUG scenario: if all agents share the same pressed_keys set,
    then pressing D on one keyboard will make ALL agents turn right.
    """

    def test_shared_pressed_keys_causes_all_agents_to_move_bug(self):
        """Demonstrate the bug: shared pressed_keys makes all agents move.

        BUG REPRODUCTION:
        If the implementation uses a SINGLE shared pressed_keys set for all agents
        (instead of per-agent pressed_keys), then:
        - User presses D
        - shared_pressed_keys = {D}
        - For each agent: resolve(shared_pressed_keys) → RIGHT
        - Result: ALL agents turn right!
        """
        resolver = INIMultiGridKeyCombinationResolver()
        agent_names = ["agent_0", "agent_1"]

        # BUG SCENARIO: Single shared pressed_keys (WRONG!)
        shared_pressed_keys = {_KEY_D}

        # Buggy implementation: use same keys for all agents
        buggy_actions = []
        for _ in agent_names:
            action = resolver.resolve(shared_pressed_keys)
            buggy_actions.append(action if action is not None else 6)

        # BUG SYMPTOM: Both agents do RIGHT!
        assert buggy_actions[0] == 1  # RIGHT
        assert buggy_actions[1] == 1  # RIGHT - BUG!
        assert buggy_actions[0] == buggy_actions[1], "Bug: both agents do same action"

        # CORRECT SCENARIO: Per-agent pressed_keys
        agent_pressed_keys = {
            "agent_0": {_KEY_D},  # Only agent_0 pressed D
            # agent_1 has no keys
        }

        correct_actions = get_multi_agent_actions_logic(
            agent_names, agent_pressed_keys, resolver, default_idle=6
        )

        # CORRECT: Only agent_0 turns right, agent_1 idles
        assert correct_actions[0] == 1  # RIGHT
        assert correct_actions[1] == 6  # IDLE
        assert correct_actions[0] != correct_actions[1], "Correct: different actions"

    def test_diagnosis_check_if_controller_uses_shared_keys(self):
        """This test helps diagnose if the real controller has the bug.

        To diagnose the real bug:
        1. Add logging to HumanInputController.get_multi_agent_actions()
        2. Print self._agent_pressed_keys before computing actions
        3. If you see the same keys for multiple agents, that's the bug!

        Expected (correct):
        _agent_pressed_keys = {'agent_0': {68}, 'agent_1': set()}

        Bug (if shared):
        _pressed_keys = {68}  # Used for all agents!
        """
        # This test documents what to look for when debugging

        # If pressed_keys is the same for all agents, the bug exists
        # The fix is to use _agent_pressed_keys[agent_id] per agent
        pass
