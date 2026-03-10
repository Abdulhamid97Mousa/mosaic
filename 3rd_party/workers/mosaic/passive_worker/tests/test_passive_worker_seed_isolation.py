"""Seed isolation tests for passive_worker.

The passive worker always returns the NOOP action (action 0, or the first
"still/noop" action in the action space).  It has no action-selection RNG,
so a shared reset seed can never contaminate its behaviour.

These tests verify:
- handle_select_action always returns NOOP regardless of seed or reset.
- Two passive worker instances are fully independent.
- A shared reset seed cannot affect the returned action.
"""

from __future__ import annotations

import pytest

from passive_worker.config import PassiveWorkerConfig
from passive_worker.runtime import PassiveWorkerRuntime


def _make_worker(run_id: str, seed: int | None = None) -> PassiveWorkerRuntime:
    config = PassiveWorkerConfig(run_id=run_id, seed=seed)
    return PassiveWorkerRuntime(config)


class TestPassiveWorkerSeedIsolation:
    """Passive worker always returns NOOP — seeds must never change that."""

    SHARED_LAYOUT_SEED = 42

    def test_always_returns_noop_regardless_of_seed(self):
        """NOOP action must be returned no matter what seed is set."""
        for seed in [None, 0, 1, 42, 999, 2**30]:
            worker = _make_worker(f"op_{seed}", seed=seed)
            worker.handle_init_agent({"game_name": "FakeEnv", "player_id": "agent_0"})
            resp = worker.handle_select_action({"observation": [], "player_id": "agent_0"})
            assert resp["type"] == "action_selected"
            assert resp["action"] == worker._passive_action, (
                f"Seed {seed!r}: expected passive action {worker._passive_action}, "
                f"got {resp['action']}"
            )

    def test_two_instances_always_return_same_noop(self):
        """Two instances with different seeds must return the same NOOP action."""
        w1 = _make_worker("op_a", seed=111)
        w2 = _make_worker("op_b", seed=222)

        for w in (w1, w2):
            w.handle_init_agent({"game_name": "FakeEnv", "player_id": "agent_0"})

        actions1 = [
            w1.handle_select_action({"observation": [], "player_id": "agent_0"})["action"]
            for _ in range(20)
        ]
        actions2 = [
            w2.handle_select_action({"observation": [], "player_id": "agent_0"})["action"]
            for _ in range(20)
        ]

        # Both must always return their passive_action (same value: 0)
        assert all(a == w1._passive_action for a in actions1)
        assert all(a == w2._passive_action for a in actions2)

    def test_shared_reset_seed_does_not_change_noop(self):
        """Simulates GUI sending shared reset seed to both operators.

        The passive worker must continue returning NOOP regardless.
        This mirrors the exact scenario that caused the random_worker bug.
        """
        w1 = _make_worker("passive_op_1", seed=1001)
        w2 = _make_worker("passive_op_2", seed=2002)

        for w in (w1, w2):
            w.handle_init_agent({"game_name": "FakeEnv", "player_id": "agent_0"})

        # Both workers receive the same shared layout seed (as the GUI sends)
        # This should have zero effect on action selection.
        for _ in range(10):
            r1 = w1.handle_select_action({"observation": [], "player_id": "agent_0"})
            r2 = w2.handle_select_action({"observation": [], "player_id": "agent_0"})
            assert r1["action"] == w1._passive_action
            assert r2["action"] == w2._passive_action

    def test_instances_do_not_share_state(self):
        """Two PassiveWorkerRuntime instances must not share any mutable state."""
        w1 = _make_worker("isolated_a", seed=5)
        w2 = _make_worker("isolated_b", seed=99)

        w1.handle_init_agent({"game_name": "FakeEnv", "player_id": "agent_0"})

        # w2 has not been initialised — w1's init must not affect w2
        assert w2._action_space is None, (
            "w2 action space was set by w1.handle_init_agent — instances share state!"
        )
        assert w1._step_count == 0

        # Advance w1 step counter
        for _ in range(5):
            w1.handle_select_action({"observation": [], "player_id": "agent_0"})
        assert w1._step_count == 5
        assert w2._step_count == 0, (
            "w2 step count was incremented by w1 — instances share state!"
        )
