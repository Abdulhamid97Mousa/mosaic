"""Regression test: curriculum environment swap must update agents.envs.

XuanCe's OnPolicyMARLAgents stores environments in ``self.envs``.
The on-policy training loop (on_policy_marl.py) reads ``self.envs``
exclusively (buf_obs, step(), buf_state, etc.).  A previous bug set
``runner.agents.train_envs`` (which doesn't exist on the agent) instead
of ``runner.agents.envs``, causing the agent to keep training on the
closed Phase-1 environment and crash.

This test inspects the swap code in ``run_multi_agent_curriculum_training``
to verify it targets the correct attribute.
"""

import types
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_runner(initial_env_id: str = "collect_1vs1"):
    """Build a mock RunnerMARL with the attributes the swap code touches."""
    runner = MagicMock()
    runner.n_envs = 4

    # runner.config must be deepcopy-able (SimpleNamespace works)
    runner.config = types.SimpleNamespace(
        env_name="multigrid",
        env_id=initial_env_id,
        env_seed=0,
        parallels=4,
        vectorize="DummyVecMultiAgentEnv",
        render_mode=None,
    )

    # The agent exposes envs (used by XuanCe training loop)
    agents = MagicMock()
    agents.current_step = 0
    agents.envs = MagicMock(name="old_collect_envs")
    runner.agents = agents

    # runner.envs is the vectorized env holder on the runner side
    runner.envs = MagicMock(name="old_runner_envs")

    return runner


class TestEnvironmentSwapTargetsEnvs:
    """Verify that environment swap sets agents.envs."""

    def test_swap_code_sets_agents_envs(self):
        """The swap block must assign new_envs to runner.agents.envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "runner.agents.envs = new_envs" in source, (
            "Environment swap must set runner.agents.envs.  XuanCe's "
            "on_policy_marl.train() reads from self.envs exclusively."
        )

    def test_swap_code_does_not_set_train_envs(self):
        """The swap block must NOT set runner.agents.train_envs (wrong attribute)."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "runner.agents.train_envs = new_envs" not in source, (
            "Environment swap must NOT set runner.agents.train_envs -- "
            "that attribute does not exist in XuanCe's MARLAgents.  "
            "Use runner.agents.envs instead."
        )

    def test_assertion_guard_present(self):
        """The swap code must contain a runtime assertion on agents.envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "assert runner.agents.envs is new_envs" in source, (
            "Environment swap must include a runtime assertion that "
            "verifies agents.envs was correctly set."
        )
