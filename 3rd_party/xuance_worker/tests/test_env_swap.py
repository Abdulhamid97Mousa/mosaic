"""Regression test: curriculum environment swap must update agent.train_envs.

XuanCe's MARLAgents stores training environments in ``self.train_envs``.
The on-policy training loop (MAPPO_Agents.train()) reads ``self.train_envs``
exclusively (buf_obs, step(), buf_state, etc.).  A previous bug set
``runner.agents`` (which doesn't exist — the attribute is ``runner.agent``)
causing an AttributeError at runtime.

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

    # The agent exposes train_envs (used by XuanCe training loop)
    agent = MagicMock()
    agent.current_step = 0
    agent.train_envs = MagicMock(name="old_collect_envs")
    runner.agent = agent

    # runner.envs is the vectorized env holder on the runner side
    runner.envs = MagicMock(name="old_runner_envs")

    return runner


class TestEnvironmentSwapTargetsEnvs:
    """Verify that environment swap sets agent.train_envs."""

    def test_swap_code_sets_agent_train_envs(self):
        """The swap block must assign new_envs to runner.agent.train_envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "runner.agent.train_envs = new_envs" in source, (
            "Environment swap must set runner.agent.train_envs.  XuanCe's "
            "MAPPO_Agents.train() reads from self.train_envs exclusively."
        )

    def test_swap_code_does_not_use_agents_plural(self):
        """The swap block must NOT use runner.agents (wrong attribute name)."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "runner.agents." not in source, (
            "Environment swap must NOT use runner.agents (plural) -- "
            "RunnerMARL uses self.agent (singular)."
        )

    def test_assertion_guard_present(self):
        """The swap code must contain a runtime assertion on agent.train_envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "assert runner.agent.train_envs is new_envs" in source, (
            "Environment swap must include a runtime assertion that "
            "verifies agent.train_envs was correctly set."
        )
