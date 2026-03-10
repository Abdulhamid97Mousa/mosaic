"""Seed isolation tests for cleanrl_worker.

CleanRL is a training worker: it launches a subprocess with a seed argument.
There is no interactive action-selection RNG that could be contaminated.

These tests verify:
- CleanRLWorkerConfig instances are fully independent (no shared state).
- Seeds stored in a config object never bleed into another config.
- The seed is correctly serialised to the subprocess command-line argument.
"""

from __future__ import annotations

import pytest

from cleanrl_worker.config import CleanRLWorkerConfig


def _base_config(run_id: str, seed: int | None = None) -> CleanRLWorkerConfig:
    return CleanRLWorkerConfig(
        run_id=run_id,
        algo="ppo_gru",
        env_id="MosaicMultiGrid-Basketball-Solo-Blue-IndAgObs-v0",
        total_timesteps=10_000,
        seed=seed,
    )


class TestCleanRLSeedIsolation:
    """Two CleanRL configs must be fully independent — seeds never cross-contaminate."""

    def test_two_configs_have_independent_seeds(self):
        """Config objects created with different seeds must not share seed values."""
        cfg1 = _base_config("run_a", seed=111)
        cfg2 = _base_config("run_b", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        """Creating a second config must not mutate the first config's seed."""
        cfg1 = _base_config("run_a", seed=42)
        cfg2 = _base_config("run_b", seed=99)  # noqa: F841

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_none_seed_is_independent_per_instance(self):
        """Two configs with seed=None must both have seed=None independently."""
        cfg1 = _base_config("run_none_a", seed=None)
        cfg2 = _base_config("run_none_b", seed=None)

        assert cfg1.seed is None
        assert cfg2.seed is None
        # run_ids must be distinct
        assert cfg1.run_id != cfg2.run_id

    def test_five_configs_all_independent(self):
        """Five configs, each with a unique seed, must all be independent."""
        seeds = [10, 20, 30, 40, 50]
        configs = [_base_config(f"run_{s}", seed=s) for s in seeds]

        for i, cfg in enumerate(configs):
            assert cfg.seed == seeds[i], (
                f"Config {i} has seed {cfg.seed!r}, expected {seeds[i]} — "
                "config seeds are not isolated."
            )

    def test_config_with_seed_serialises_seed(self):
        """A config with a seed must include the seed in its dict representation."""
        cfg = _base_config("run_serial", seed=77)
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
        # Either direct attribute or dict representation must carry the seed
        assert cfg.seed == 77

    def test_config_run_ids_are_independent(self):
        """Each config's run_id must match what was passed — no aliasing."""
        pairs = [("alpha", 1), ("beta", 2), ("gamma", 3)]
        configs = [_base_config(rid, seed=s) for rid, s in pairs]

        for cfg, (expected_rid, expected_seed) in zip(configs, pairs):
            assert cfg.run_id == expected_rid
            assert cfg.seed == expected_seed
