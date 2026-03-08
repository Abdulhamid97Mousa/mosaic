"""Integration-oriented tests to ensure CleanRL helper features behave as expected."""

from __future__ import annotations

from gym_gui.Algo_docs.cleanrl_worker import get_algo_doc
from gym_gui.validations.validation_cleanrl_worker_form import run_cleanrl_dry_run
from .test_cleanrl_dry_run import _minimal_config


def test_algorithm_doc_lookup():
    assert "PPO" in get_algo_doc("ppo")
    assert "DQN" in get_algo_doc("dqn")
    # Unknown algorithms fall back to the generic help block
    assert "CleanRL" in get_algo_doc("unknown_algo")


def test_dry_run_payload_includes_artifacts():
    config = _minimal_config()
    assert "artifacts" in config
    success, output = run_cleanrl_dry_run(config)
    assert success, f"Dry-run failed: {output}"

