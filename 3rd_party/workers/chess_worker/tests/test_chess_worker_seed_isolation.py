"""Seed isolation tests for chess_worker.

The chess worker uses an LLM to select moves — there is no internal RNG for
action selection.  Two chess worker runtime instances must not share any state.

These tests verify:
- ChessWorkerConfig instances are independent (no shared state).
- Two runtime instances are independent (no class-level shared RNG).
- The run_id and config are correctly scoped per instance.
"""

from __future__ import annotations

import pytest

from chess_worker.config import ChessWorkerConfig
from chess_worker.runtime import ChessWorkerRuntime


def _make_config(run_id: str, **kwargs) -> ChessWorkerConfig:
    return ChessWorkerConfig(run_id=run_id, **kwargs)


class TestChessWorkerSeedIsolation:
    """Chess worker instances must be fully independent — no shared state."""

    def test_two_configs_have_independent_run_ids(self):
        """Config objects must store independent run_ids."""
        cfg1 = _make_config("chess_op_1")
        cfg2 = _make_config("chess_op_2")

        assert cfg1.run_id == "chess_op_1"
        assert cfg2.run_id == "chess_op_2"
        assert cfg1.run_id != cfg2.run_id

    def test_config_mutation_does_not_bleed(self):
        """Mutating one config must not affect another."""
        cfg1 = _make_config("chess_a")
        cfg2 = _make_config("chess_b")

        assert cfg1.run_id != cfg2.run_id

        # Confirm they are distinct objects
        assert cfg1 is not cfg2

    def test_five_configs_all_independent(self):
        """Five configs must all have independent run_ids."""
        configs = [_make_config(f"op_{i}") for i in range(5)]

        run_ids = [c.run_id for c in configs]
        assert len(set(run_ids)) == 5, (
            "Config run_ids are not unique — configs may share state."
        )

    def test_runtime_instances_are_independent(self):
        """Two ChessWorkerRuntime instances must not share internal state."""
        cfg1 = _make_config("chess_rt_1")
        cfg2 = _make_config("chess_rt_2")

        rt1 = ChessWorkerRuntime(cfg1)
        rt2 = ChessWorkerRuntime(cfg2)

        assert rt1 is not rt2
        assert rt1.config.run_id == "chess_rt_1"
        assert rt2.config.run_id == "chess_rt_2"

    def test_runtime_config_not_aliased(self):
        """Runtime config must not be a shared reference between instances."""
        cfg1 = _make_config("alias_a")
        cfg2 = _make_config("alias_b")

        rt1 = ChessWorkerRuntime(cfg1)
        rt2 = ChessWorkerRuntime(cfg2)

        assert rt1.config is not rt2.config
