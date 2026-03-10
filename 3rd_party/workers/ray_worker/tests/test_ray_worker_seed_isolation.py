"""Seed isolation tests for ray_worker.

Ray worker is an RL training worker.  There is no interactive action-selection
RNG, so the shared-reset-seed contamination bug does not apply.  However, two
training run configs must be completely independent.

These tests verify:
- RayWorkerConfig instances have independent seeds (no shared state).
- Seeds stored in one config never bleed into another.
- Multiple configs with unique seeds all retain their own values.
"""

from __future__ import annotations

import pytest

from ray_worker.config import RayWorkerConfig, EnvironmentConfig


def _make_env_config() -> EnvironmentConfig:
    return EnvironmentConfig(family="pettingzoo", env_id="chess_v6")


def _make_config(run_id: str, seed: int | None = None) -> RayWorkerConfig:
    return RayWorkerConfig(run_id=run_id, environment=_make_env_config(), seed=seed)


class TestRayWorkerSeedIsolation:
    """RayWorkerConfig instances must be fully independent."""

    def test_two_configs_have_independent_seeds(self):
        cfg1 = _make_config("ray_op_1", seed=111)
        cfg2 = _make_config("ray_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        cfg1 = _make_config("ray_a", seed=42)
        _cfg2 = _make_config("ray_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_none_seed_per_instance(self):
        cfg1 = _make_config("ray_none_a", seed=None)
        cfg2 = _make_config("ray_none_b", seed=None)

        assert cfg1.seed is None
        assert cfg2.seed is None
        assert cfg1.run_id != cfg2.run_id

    def test_five_configs_all_retain_their_seeds(self):
        seeds = [10, 20, 30, 40, 50]
        configs = [_make_config(f"ray_{s}", seed=s) for s in seeds]

        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected, (
                f"Config has seed {cfg.seed!r}, expected {expected} — isolation broken."
            )

    def test_configs_are_distinct_objects(self):
        cfg1 = _make_config("ray_x", seed=7)
        cfg2 = _make_config("ray_y", seed=8)
        assert cfg1 is not cfg2

    def test_run_ids_are_independent(self):
        pairs = [("raya", 1), ("rayb", 2), ("rayc", 3)]
        configs = [_make_config(rid, seed=s) for rid, s in pairs]

        for cfg, (expected_rid, expected_seed) in zip(configs, pairs):
            assert cfg.run_id == expected_rid
            assert cfg.seed == expected_seed
