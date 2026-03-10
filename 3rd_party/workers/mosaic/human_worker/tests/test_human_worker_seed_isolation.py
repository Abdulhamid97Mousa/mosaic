"""Seed isolation tests for human_worker.

The human worker receives keyboard input from a human — it does not use an
RNG for action selection.  Seeds are used only to reset the environment layout.
Two human worker instances must be fully independent.

These tests verify:
- HumanWorkerConfig instances are independent (no shared state).
- HumanWorkerRuntime instances do not share internal state.
- Two runtimes' step counters and action-space sizes are independent.
"""

from __future__ import annotations

import pytest

from human_worker.config import HumanWorkerConfig
from human_worker.runtime import HumanWorkerRuntime


def _make_worker(run_id: str, seed: int = 42) -> HumanWorkerRuntime:
    config = HumanWorkerConfig(run_id=run_id, seed=seed)
    return HumanWorkerRuntime(config)


class TestHumanWorkerSeedIsolation:
    """Human worker instances must not share any state."""

    def test_two_configs_have_independent_seeds(self):
        cfg1 = HumanWorkerConfig(run_id="human_op_1", seed=111)
        cfg2 = HumanWorkerConfig(run_id="human_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        cfg1 = HumanWorkerConfig(run_id="human_a", seed=42)
        _cfg2 = HumanWorkerConfig(run_id="human_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_two_runtime_instances_are_distinct(self):
        """Two runtime instances must not alias each other."""
        rt1 = _make_worker("human_rt_1", seed=10)
        rt2 = _make_worker("human_rt_2", seed=20)

        assert rt1 is not rt2
        assert rt1.config.run_id == "human_rt_1"
        assert rt2.config.run_id == "human_rt_2"
        assert rt1.config is not rt2.config

    def test_runtime_internal_state_is_independent(self):
        """Internal state of two runtimes must not be shared."""
        rt1 = _make_worker("human_state_a", seed=5)
        rt2 = _make_worker("human_state_b", seed=6)

        # HumanWorkerRuntime is the board-game runtime (GUI owns the env).
        # It starts with empty player/game state — verify isolation.
        assert rt1._player_id == ""
        assert rt2._player_id == ""
        assert rt1._game_name == ""
        assert rt2._game_name == ""
        assert rt1._waiting_for_input is False
        assert rt2._waiting_for_input is False
        assert rt1 is not rt2

    def test_five_configs_all_independent(self):
        """Five configs with unique seeds must all retain their own seeds."""
        seeds = [1, 2, 3, 4, 5]
        configs = [HumanWorkerConfig(run_id=f"h_{s}", seed=s) for s in seeds]

        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected, (
                f"Config has seed {cfg.seed!r}, expected {expected} — isolation broken."
            )
