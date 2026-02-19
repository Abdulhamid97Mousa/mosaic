"""Regression test: curriculum environment swap must update agent.train_envs.

XuanCe's MARLAgents stores environments in ``self.train_envs`` (not
``self.envs``).  The on-policy training loop reads ``self.train_envs``
exclusively.  A previous bug set ``runner.agent.envs`` (which doesn't exist
on the agent) instead of ``runner.agent.train_envs``, causing the agent to
silently keep training on the Phase-1 environment for the entire run.

This test mocks the runner/agent structure and verifies that the swap code
in ``run_multi_agent_curriculum_training`` targets the correct attribute.
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


class TestEnvironmentSwapTargetsTrainEnvs:
    """Verify that environment swap sets agent.train_envs, not agent.envs."""

    def test_swap_code_sets_train_envs(self):
        """The swap block must assign new_envs to runner.agent.train_envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        # Read the source and verify train_envs is the target
        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "runner.agent.train_envs = new_envs" in source, (
            "Environment swap must set runner.agent.train_envs (not "
            "runner.agent.envs).  XuanCe's training loop reads from "
            "train_envs exclusively."
        )

    def test_swap_code_does_not_set_agent_dot_envs(self):
        """The swap block must NOT set runner.agent.envs (wrong attribute)."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)

        # runner.envs = new_envs is OK (that's the runner-level attribute)
        # runner.agent.envs = new_envs is WRONG
        assert "runner.agent.envs = new_envs" not in source, (
            "Environment swap must NOT set runner.agent.envs -- that "
            "attribute is not used by XuanCe's training loop.  Use "
            "runner.agent.train_envs instead."
        )

    def test_assertion_guard_present(self):
        """The swap code must contain a runtime assertion on train_envs."""
        import xuance_worker.multi_agent_curriculum_training as mod

        import inspect
        source = inspect.getsource(mod.run_multi_agent_curriculum_training)
        assert "assert runner.agent.train_envs is new_envs" in source, (
            "Environment swap must include a runtime assertion that "
            "verifies train_envs was correctly set."
        )
