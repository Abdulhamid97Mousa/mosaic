"""Tests for baseline operators.

Tests verify:
- Dynamic action space configuration
- Action selection behavior
- Trajectory tracking
- Seed reproducibility
"""

import pytest
from gymnasium import spaces
from operators_worker.operators import (
    RandomOperator,
    NoopOperator,
    CyclingOperator,
    create_baseline_operator,
)


class TestRandomOperator:
    """Tests for RandomOperator."""

    def test_action_space_required(self):
        """Test that select_action raises without action_space."""
        operator = RandomOperator()

        with pytest.raises(RuntimeError, match="action_space not configured"):
            operator.select_action(None)

    def test_set_action_space_validates(self):
        """Test that set_action_space validates input."""
        operator = RandomOperator()

        with pytest.raises(TypeError):
            operator.set_action_space("not a space")

    def test_discrete_action_space(self):
        """Test random action selection with Discrete space."""
        operator = RandomOperator()
        action_space = spaces.Discrete(7)
        operator.set_action_space(action_space)
        operator.reset(seed=42)

        # Sample 100 actions
        actions = [operator.select_action(None) for _ in range(100)]

        # All should be valid
        assert all(0 <= a < 7 for a in actions)

        # Should have variety (not all same action)
        assert len(set(actions)) > 1

    def test_seed_reproducibility(self):
        """Test that seed produces reproducible sequences."""
        operator1 = RandomOperator()
        operator1.set_action_space(spaces.Discrete(7))
        operator1.reset(seed=42)

        operator2 = RandomOperator()
        operator2.set_action_space(spaces.Discrete(7))
        operator2.reset(seed=42)

        # Same seed should produce same sequence
        actions1 = [operator1.select_action(None) for _ in range(20)]
        actions2 = [operator2.select_action(None) for _ in range(20)]

        assert actions1 == actions2

    def test_trajectory_tracking(self):
        """Test trajectory tracking via on_step_result."""
        operator = RandomOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        # Simulate episode
        operator.on_step_result(None, 0.0, False, False, {})
        operator.on_step_result(None, 0.5, False, False, {})
        operator.on_step_result(None, 1.0, True, False, {})

        trajectory = operator.get_trajectory()
        assert len(trajectory) == 3
        assert trajectory[0]["step"] == 0
        assert trajectory[1]["reward"] == 0.5
        assert trajectory[2]["terminated"] is True

    def test_episode_return(self):
        """Test episode return calculation."""
        operator = RandomOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        operator.on_step_result(None, 1.0, False, False, {})
        operator.on_step_result(None, 2.0, False, False, {})
        operator.on_step_result(None, 3.0, True, False, {})

        assert operator.get_episode_return() == 6.0

    def test_reset_clears_trajectory(self):
        """Test that reset clears previous trajectory."""
        operator = RandomOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        operator.on_step_result(None, 1.0, True, False, {})
        assert len(operator.get_trajectory()) == 1

        # Reset should clear
        operator.reset()
        assert len(operator.get_trajectory()) == 0


class TestNoopOperator:
    """Tests for NoopOperator."""

    def test_always_returns_same_action(self):
        """Test that no-op always returns configured action."""
        operator = NoopOperator(action_index=3)
        operator.set_action_space(spaces.Discrete(7))

        # Should always return 3
        for _ in range(20):
            assert operator.select_action(None) == 3

    def test_default_action_zero(self):
        """Test that default action is 0."""
        operator = NoopOperator()
        operator.set_action_space(spaces.Discrete(7))

        assert operator.select_action(None) == 0

    def test_validates_action_index(self):
        """Test that invalid action_index is caught."""
        operator = NoopOperator(action_index=10)

        with pytest.raises(ValueError, match="action_index 10 invalid"):
            operator.set_action_space(spaces.Discrete(7))

    def test_trajectory_tracking(self):
        """Test trajectory tracking works."""
        operator = NoopOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        operator.on_step_result(None, 0.5, False, False, {})
        operator.on_step_result(None, 1.0, True, False, {})

        trajectory = operator.get_trajectory()
        assert len(trajectory) == 2
        assert trajectory[1]["reward"] == 1.0


class TestCyclingOperator:
    """Tests for CyclingOperator."""

    def test_cycles_through_actions(self):
        """Test that operator cycles through action space."""
        operator = CyclingOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        # Should cycle: 0, 1, 2, 3, 0, 1, 2, 3, ...
        actions = [operator.select_action(None) for _ in range(12)]
        expected = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]

        assert actions == expected

    def test_requires_discrete_space(self):
        """Test that only Discrete spaces are supported."""
        operator = CyclingOperator()

        with pytest.raises(ValueError, match="only supports Discrete"):
            operator.set_action_space(spaces.Box(0, 1, shape=(2,)))

    def test_reset_restarts_cycle(self):
        """Test that reset restarts the action cycle."""
        operator = CyclingOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        # Advance cycle
        for _ in range(5):
            operator.select_action(None)

        # Reset should restart from 0
        operator.reset()
        assert operator.select_action(None) == 0
        assert operator.select_action(None) == 1

    def test_trajectory_tracking(self):
        """Test trajectory tracking works."""
        operator = CyclingOperator()
        operator.set_action_space(spaces.Discrete(4))
        operator.reset()

        operator.on_step_result(None, 1.0, False, False, {})
        operator.on_step_result(None, 2.0, False, False, {})

        trajectory = operator.get_trajectory()
        assert len(trajectory) == 2
        assert operator.get_episode_return() == 3.0


class TestFactoryFunction:
    """Tests for create_baseline_operator factory."""

    def test_creates_random(self):
        """Test factory creates RandomOperator."""
        operator = create_baseline_operator("random")
        assert isinstance(operator, RandomOperator)

    def test_creates_noop(self):
        """Test factory creates NoopOperator."""
        operator = create_baseline_operator("noop")
        assert isinstance(operator, NoopOperator)

    def test_creates_cycling(self):
        """Test factory creates CyclingOperator."""
        operator = create_baseline_operator("cycling")
        assert isinstance(operator, CyclingOperator)

    def test_invalid_behavior_raises(self):
        """Test that invalid behavior raises ValueError."""
        with pytest.raises(ValueError, match="Unknown baseline behavior"):
            create_baseline_operator("invalid")

    def test_sets_custom_id(self):
        """Test that custom operator_id is set."""
        operator = create_baseline_operator("random", operator_id="custom_001")
        assert operator.id == "custom_001"

    def test_sets_custom_name(self):
        """Test that custom operator_name is set."""
        operator = create_baseline_operator(
            "random",
            operator_name="My Custom Operator"
        )
        assert operator.name == "My Custom Operator"

    def test_noop_action_index(self):
        """Test that NoopOperator action_index can be customized."""
        operator = create_baseline_operator("noop", action_index=5)
        operator.set_action_space(spaces.Discrete(7))
        assert operator.select_action(None) == 5


class TestIntegration:
    """Integration tests with multiple operators."""

    def test_multiple_operators_independent(self):
        """Test that multiple operators can run independently."""
        op1 = RandomOperator()
        op1.set_action_space(spaces.Discrete(4))
        op1.reset(seed=42)

        op2 = NoopOperator()
        op2.set_action_space(spaces.Discrete(4))
        op2.reset()

        op3 = CyclingOperator()
        op3.set_action_space(spaces.Discrete(4))
        op3.reset()

        # Each should behave independently
        a1 = op1.select_action(None)  # Random
        a2 = op2.select_action(None)  # 0
        a3 = op3.select_action(None)  # 0 (first in cycle)

        assert 0 <= a1 < 4
        assert a2 == 0
        assert a3 == 0

        # Next actions
        a2_next = op2.select_action(None)  # Still 0
        a3_next = op3.select_action(None)  # 1 (next in cycle)

        assert a2_next == 0
        assert a3_next == 1

    def test_operators_with_different_action_spaces(self):
        """Test that operators adapt to different action spaces."""
        # Small action space
        op1 = RandomOperator()
        op1.set_action_space(spaces.Discrete(3))
        op1.reset(seed=42)
        actions_small = [op1.select_action(None) for _ in range(10)]
        assert all(0 <= a < 3 for a in actions_small)

        # Large action space
        op2 = RandomOperator()
        op2.set_action_space(spaces.Discrete(20))
        op2.reset(seed=42)
        actions_large = [op2.select_action(None) for _ in range(10)]
        assert all(0 <= a < 20 for a in actions_large)
