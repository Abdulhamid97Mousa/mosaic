"""Seed isolation tests for llm_worker.

LLM worker uses a language model for action selection — there is no
action-space RNG that could be contaminated by a shared reset seed.
Two operator instances must still be fully independent.

These tests verify:
- LLMWorkerConfig instances are independent (no shared state).
- Two configs with different seeds do not share seed values.
- Runtime instances derived from different configs are distinct.
"""

from __future__ import annotations

import pytest

from llm_worker.config import LLMWorkerConfig
from llm_worker.runtime import InteractiveLLMRuntime


def _make_config(run_id: str, seed: int | None = None) -> LLMWorkerConfig:
    return LLMWorkerConfig(run_id=run_id, seed=seed)


class TestLLMWorkerSeedIsolation:
    """LLMWorkerConfig instances must be fully independent."""

    def test_two_configs_have_independent_seeds(self):
        cfg1 = _make_config("llm_op_1", seed=111)
        cfg2 = _make_config("llm_op_2", seed=222)

        assert cfg1.seed == 111
        assert cfg2.seed == 222
        assert cfg1.seed != cfg2.seed

    def test_config_seed_unchanged_after_second_config_created(self):
        cfg1 = _make_config("llm_a", seed=42)
        _cfg2 = _make_config("llm_b", seed=99)

        assert cfg1.seed == 42, (
            "cfg1.seed was mutated after cfg2 was created — configs share state!"
        )

    def test_none_seed_per_instance(self):
        cfg1 = _make_config("llm_none_a", seed=None)
        cfg2 = _make_config("llm_none_b", seed=None)

        assert cfg1.seed is None
        assert cfg2.seed is None
        assert cfg1.run_id != cfg2.run_id

    def test_five_configs_all_retain_their_seeds(self):
        seeds = [10, 20, 30, 40, 50]
        configs = [_make_config(f"llm_{s}", seed=s) for s in seeds]

        for cfg, expected in zip(configs, seeds):
            assert cfg.seed == expected, (
                f"Config has seed {cfg.seed!r}, expected {expected} — isolation broken."
            )

    def test_configs_are_distinct_objects(self):
        cfg1 = _make_config("llm_x", seed=7)
        cfg2 = _make_config("llm_y", seed=8)
        assert cfg1 is not cfg2

    def test_run_ids_are_independent(self):
        pairs = [("llm_alpha", 1), ("llm_beta", 2), ("llm_gamma", 3)]
        configs = [_make_config(rid, seed=s) for rid, s in pairs]

        for cfg, (expected_rid, expected_seed) in zip(configs, pairs):
            assert cfg.run_id == expected_rid
            assert cfg.seed == expected_seed

    def test_two_interactive_runtimes_are_distinct_objects(self):
        """Two InteractiveRuntime instances must not alias each other."""
        cfg1 = _make_config("llm_rt_1", seed=10)
        cfg2 = _make_config("llm_rt_2", seed=20)

        rt1 = InteractiveLLMRuntime(cfg1)
        rt2 = InteractiveLLMRuntime(cfg2)

        assert rt1 is not rt2
        assert rt1.config.run_id == "llm_rt_1"
        assert rt2.config.run_id == "llm_rt_2"
        assert rt1.config is not rt2.config
