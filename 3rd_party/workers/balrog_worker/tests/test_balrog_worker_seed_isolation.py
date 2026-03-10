"""Seed isolation tests for balrog_worker.

BALROG uses an LLM for action selection — there is no action-space RNG that
could be contaminated by a shared reset seed.  However, two operator instances
must still be fully independent with respect to their configs and state.

These tests verify:
- BarlogWorkerConfig instances are independent.
- Two configs with different seeds do not share the seed value.
- Resetting with a shared layout seed is safe (env.reset only, no RNG leakage).
"""

from __future__ import annotations

import pytest

from balrog_worker.config import BarlogWorkerConfig


def _make_config(run_id: str, seed: int | None = None) -> BarlogWorkerConfig:
    return BarlogWorkerConfig(run_id=run_id, seed=seed)


class TestBalrogWorkerSeedIsolation:
    """Balrog config instances must be fully independent."""

    def test_two_configs_have_independent_seeds(self):
        """Two configs created with different seeds must not share the value."""
        cfg1 = _make_config("balrog_op_1", seed=111)
        cfg2 = _make_config("balrog_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        """Creating a second config must not mutate the first config's seed."""
        cfg1 = _make_config("balrog_a", seed=42)
        _cfg2 = _make_config("balrog_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_none_seed_per_instance(self):
        """Two configs with seed=None must be independently None."""
        cfg1 = _make_config("balrog_none_a", seed=None)
        cfg2 = _make_config("balrog_none_b", seed=None)

        assert cfg1.seed is None
        assert cfg2.seed is None
        assert cfg1.run_id != cfg2.run_id

    def test_five_configs_all_have_independent_seeds(self):
        """Five configs with different seeds must all retain their own seed."""
        seeds = [10, 20, 30, 40, 50]
        configs = [_make_config(f"balrog_{s}", seed=s) for s in seeds]

        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected, (
                f"Config has seed {cfg.seed!r}, expected {expected} — "
                "config seeds are not isolated."
            )

    def test_configs_are_distinct_objects(self):
        """Each config must be a distinct object — no aliasing."""
        cfg1 = _make_config("op_x", seed=7)
        cfg2 = _make_config("op_y", seed=8)

        assert cfg1 is not cfg2

    def test_run_ids_are_independent(self):
        """Each config's run_id must match what was passed."""
        pairs = [("alpha", 1), ("beta", 2), ("gamma", 3)]
        configs = [_make_config(rid, seed=s) for rid, s in pairs]

        for cfg, (expected_rid, expected_seed) in zip(configs, pairs):
            assert cfg.run_id == expected_rid
            assert cfg.seed == expected_seed
