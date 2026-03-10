"""Seed isolation tests for mujoco_mpc_worker.

MuJoCoMPCConfig is a connection config for an external MJPC server.
It does not carry a training seed — isolation tests verify that two
MuJoCoMPCConfig instances never share mutable state.
"""

from __future__ import annotations

import pytest

from mujoco_mpc_worker.config import MuJoCoMPCConfig, MuJoCoMPCPlannerType, MuJoCoMPCTaskId


def _make_config(**kwargs) -> MuJoCoMPCConfig:
    return MuJoCoMPCConfig(**kwargs)


class TestMuJoCoMPCConfigIsolation:
    """MuJoCoMPCConfig instances must be fully independent (no shared mutable state)."""

    def test_two_configs_are_distinct_objects(self):
        cfg1 = _make_config(port=8000)
        cfg2 = _make_config(port=8001)

        assert cfg1 is not cfg2
        assert cfg1.port == 8000
        assert cfg2.port == 8001

    def test_task_id_is_independent(self):
        cfg1 = _make_config(task_id="Cartpole")
        cfg2 = _make_config(task_id="Humanoid")

        assert cfg1.task_id == "Cartpole"
        assert cfg2.task_id == "Humanoid"
        assert cfg1.task_id != cfg2.task_id

    def test_planner_type_is_independent(self):
        cfg1 = _make_config(planner_type=MuJoCoMPCPlannerType.PREDICTIVE_SAMPLING)
        cfg2 = _make_config(planner_type=MuJoCoMPCPlannerType.GRADIENT_DESCENT)

        assert cfg1.planner_type != cfg2.planner_type

    def test_mutating_cost_weights_does_not_affect_another(self):
        cfg1 = _make_config()
        cfg2 = _make_config()

        cfg1.cost_weights["velocity"] = 0.5

        assert "velocity" not in cfg2.cost_weights, (
            "Mutating cfg1.cost_weights leaked into cfg2 — configs share state!"
        )

    def test_mutating_task_parameters_does_not_affect_another(self):
        cfg1 = _make_config()
        cfg2 = _make_config()

        cfg1.task_parameters["mass"] = 1.2

        assert "mass" not in cfg2.task_parameters, (
            "Mutating cfg1.task_parameters leaked into cfg2 — configs share state!"
        )

    def test_five_configs_all_independent(self):
        configs = [_make_config(task_id=f"Task{i}", port=9000 + i) for i in range(5)]

        for i, cfg in enumerate(configs):
            assert cfg.task_id == f"Task{i}"
            assert cfg.port == 9000 + i
