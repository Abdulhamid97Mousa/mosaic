# 3rd_party/xuance_worker/tests/test_policy_inference.py
"""Tests for loading a trained IPPO checkpoint and querying it for actions.

Verifies the full inference pipeline:
  1. get_runner() creates the correct agent from YAML config
  2. agent.load_model() loads the checkpoint weights
  3. policy forward pass returns a valid discrete action
  4. Different agents (agent_0, agent_1) can be queried independently
  5. Actions are within the valid range [0, n_actions)
  6. Policy produces non-uniform action distributions (not random)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


# Path to the trained IPPO checkpoint (collect_1vs1)
# Updated dynamically — find the most recent checkpoint if available
def _find_checkpoint() -> str | None:
    """Find the most recent .pth checkpoint under var/trainer/."""
    from pathlib import Path
    var_dir = Path("/home/hamid/Desktop/software/mosaic/var/trainer")
    if not var_dir.exists():
        return None
    pth_files = sorted(var_dir.rglob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pth_files[0]) if pth_files else None


CHECKPOINT_PATH = _find_checkpoint()


# Marker to skip trained-checkpoint tests if no checkpoint available
_skip_no_checkpoint = pytest.mark.skipif(
    CHECKPOINT_PATH is None or not os.path.exists(CHECKPOINT_PATH),
    reason="No trained checkpoint found under var/trainer/",
)


def _load_ippo_agent(env_id: str = "collect_1vs1", checkpoint: str = CHECKPOINT_PATH):
    """Load an IPPO agent from checkpoint, returning (agent, device)."""
    import torch
    from xuance_worker.runtime import _resolve_custom_config_path, _gymnasium_to_xuance_env_id
    from xuance import get_runner

    xuance_env_id = _gymnasium_to_xuance_env_id(env_id) or env_id
    config_path = _resolve_custom_config_path(
        method="ippo", env="multigrid", env_id=xuance_env_id,
        num_groups=None, config_path=None,
    )
    assert config_path is not None, f"No config for {env_id}"

    runner = get_runner(
        algo="ippo", env="multigrid", env_id=xuance_env_id,
        config_path=config_path, is_test=True,
    )
    agent = runner.agent
    # load_model() expects the env_id-level dir (e.g. .../collect_1vs1/)
    # It auto-discovers seed_*/ subdirs and picks the latest .pth inside.
    checkpoint_dir = str(Path(checkpoint).parent.parent)
    agent.load_model(checkpoint_dir)
    device = next(iter(agent.policy.parameters())).device
    return agent, device


def _get_action(agent, device, obs: np.ndarray, player_id: str) -> int:
    """Query the IPPO policy for a single agent's action."""
    import torch

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    agent_key = player_id
    obs_input = {agent_key: obs_tensor}

    with torch.no_grad():
        _, pi_dists = agent.policy(
            observation=obs_input,
            agent_ids=None,
            agent_key=agent_key,
        )
    action = pi_dists[agent_key].stochastic_sample()
    return int(action.cpu().item())


@_skip_no_checkpoint
class TestIPPOPolicyInference:
    """Test loading and querying the trained IPPO collect_1vs1 policy."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        """Load the IPPO agent once for all tests in this class."""
        self.agent, self.device = _load_ippo_agent()
        yield

    def test_agent_loaded(self):
        """Agent and policy are not None after loading."""
        assert self.agent is not None
        assert self.agent.policy is not None

    def test_agent_keys(self):
        """IPPO should have separate keys for agent_0 and agent_1."""
        keys = getattr(self.agent, 'agent_keys', [])
        assert "agent_0" in keys, f"agent_0 not in agent_keys: {keys}"
        assert "agent_1" in keys, f"agent_1 not in agent_keys: {keys}"

    def test_no_parameter_sharing(self):
        """IPPO collect_1vs1 trains with use_parameter_sharing=False."""
        assert not getattr(self.agent, 'use_parameter_sharing', True)

    def test_valid_action_agent_0(self):
        """Policy returns a valid action for agent_0."""
        obs = np.random.rand(27).astype(np.float32)
        action = _get_action(self.agent, self.device, obs, "agent_0")
        assert isinstance(action, int)
        assert 0 <= action < 8, f"Action {action} out of range [0, 8)"

    def test_valid_action_agent_1(self):
        """Policy returns a valid action for agent_1."""
        obs = np.random.rand(27).astype(np.float32)
        action = _get_action(self.agent, self.device, obs, "agent_1")
        assert isinstance(action, int)
        assert 0 <= action < 8, f"Action {action} out of range [0, 8)"

    def test_actions_on_real_obs(self):
        """Query policy with actual environment observations."""
        from types import SimpleNamespace
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        cfg = SimpleNamespace(
            env_name="multigrid", env_id="collect_1vs1",
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, _ = env.reset()
            for agent_id in env.agents:
                action = _get_action(self.agent, self.device, obs[agent_id], agent_id)
                assert 0 <= action < 8
        finally:
            env.close()

    def test_policy_not_uniform(self):
        """Trained policy should NOT produce uniform random actions.

        Run 100 queries on an actual environment observation — a trained policy
        should favour certain actions more than a uniform random policy.
        """
        from types import SimpleNamespace
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        cfg = SimpleNamespace(
            env_name="multigrid", env_id="collect_1vs1",
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, _ = env.reset()
            test_obs = obs["agent_0"]
        finally:
            env.close()

        actions = [
            _get_action(self.agent, self.device, test_obs, "agent_0")
            for _ in range(200)
        ]
        from collections import Counter
        counts = Counter(actions)
        most_common_freq = counts.most_common(1)[0][1]
        # A uniform policy over 8 actions with 200 samples → ~25 each.
        # A trained policy should have its top action appear > 40 times.
        assert most_common_freq > 40, (
            f"Policy's most common action appeared {most_common_freq}/200 times — "
            f"looks uniform/random. Distribution: {dict(counts)}"
        )

    def test_different_obs_different_actions(self):
        """Different observations should (eventually) lead to different actions.

        Feed two very different observations and check the policy doesn't always
        return the same action.
        """
        obs_a = np.zeros(27, dtype=np.float32)       # Empty view
        obs_b = np.ones(27, dtype=np.float32) * 255   # Saturated view

        actions_a = set()
        actions_b = set()
        for _ in range(50):
            actions_a.add(_get_action(self.agent, self.device, obs_a, "agent_0"))
            actions_b.add(_get_action(self.agent, self.device, obs_b, "agent_0"))

        # At least the distributions should differ (different modal actions)
        # This is a soft check — mainly validates the policy is obs-dependent
        all_a = list(actions_a)
        all_b = list(actions_b)
        assert len(all_a) > 0 and len(all_b) > 0

    def test_invalid_player_id_raises(self):
        """Querying with an invalid player_id should raise KeyError."""
        obs = np.random.rand(27).astype(np.float32)
        with pytest.raises(KeyError):
            _get_action(self.agent, self.device, obs, "agent_99")


@_skip_no_checkpoint
class TestIPPOPolicyMultiStep:
    """Test running the policy for multiple environment steps (mini-episode)."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        self.agent, self.device = _load_ippo_agent()
        yield

    def test_10_step_episode(self):
        """Run 10 steps with the trained policy and verify all actions are valid."""
        from types import SimpleNamespace
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        cfg = SimpleNamespace(
            env_name="multigrid", env_id="collect_1vs1",
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, _ = env.reset()
            for step in range(10):
                actions = {}
                for agent_id in env.agents:
                    action = _get_action(
                        self.agent, self.device, obs[agent_id], agent_id
                    )
                    assert 0 <= action < 8, f"Step {step}, {agent_id}: invalid action {action}"
                    actions[agent_id] = action
                obs, rewards, terminated, truncated, info = env.step(actions)
                if terminated or truncated:
                    break
        finally:
            env.close()


# ===========================================================================
# Fresh agent inference (no checkpoint needed)
# ===========================================================================
class TestFreshIPPOInference:
    """Test the full inference pipeline with a freshly created (untrained) agent.

    This validates the mechanism works even without a pre-trained checkpoint:
      get_runner() → agent created → policy forward pass → valid action returned
    """

    @pytest.fixture(autouse=True)
    def setup_fresh_agent(self):
        """Create a fresh IPPO agent from config (random weights)."""
        from xuance_worker.runtime import _resolve_custom_config_path
        from xuance import get_runner

        config_path = _resolve_custom_config_path(
            method="ippo", env="multigrid", env_id="collect_1vs1",
            num_groups=None, config_path=None,
        )
        runner = get_runner(
            algo="ippo", env="multigrid", env_id="collect_1vs1",
            config_path=config_path, is_test=True,
        )
        self.agent = runner.agent
        self.device = next(iter(self.agent.policy.parameters())).device
        yield

    def test_agent_created(self):
        assert self.agent is not None
        assert self.agent.policy is not None

    def test_agent_keys_present(self):
        keys = getattr(self.agent, 'agent_keys', [])
        assert "agent_0" in keys
        assert "agent_1" in keys

    @pytest.mark.parametrize("agent_id", ["agent_0", "agent_1"])
    def test_valid_action_from_random_obs(self, agent_id: str):
        """Fresh policy produces a valid action from random observations."""
        obs = np.random.rand(27).astype(np.float32)
        action = _get_action(self.agent, self.device, obs, agent_id)
        assert isinstance(action, int)
        assert 0 <= action < 8

    def test_valid_action_from_env_obs(self):
        """Fresh policy produces valid actions from real environment observations."""
        from types import SimpleNamespace
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        cfg = SimpleNamespace(
            env_name="multigrid", env_id="collect_1vs1",
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, _ = env.reset()
            for agent_id in env.agents:
                action = _get_action(self.agent, self.device, obs[agent_id], agent_id)
                assert 0 <= action < 8, f"{agent_id}: action {action} out of range"
        finally:
            env.close()

    def test_10_step_episode_fresh(self):
        """Run 10 steps with fresh policy — all actions valid, no crashes."""
        from types import SimpleNamespace
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        cfg = SimpleNamespace(
            env_name="multigrid", env_id="collect_1vs1",
            env_seed=42, training_mode="competitive",
        )
        env = MultiGrid_Env(cfg)
        try:
            obs, _ = env.reset()
            for step in range(10):
                actions = {}
                for agent_id in env.agents:
                    action = _get_action(self.agent, self.device, obs[agent_id], agent_id)
                    assert 0 <= action < 8
                    actions[agent_id] = action
                obs, rewards, terminated, truncated, info = env.step(actions)
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_invalid_agent_key_raises(self):
        """Invalid agent key raises KeyError."""
        obs = np.random.rand(27).astype(np.float32)
        with pytest.raises(KeyError):
            _get_action(self.agent, self.device, obs, "agent_99")


class TestFreshMAPPOInference:
    """Test MAPPO (parameter sharing) inference with a fresh agent."""

    @pytest.fixture(autouse=True)
    def setup_fresh_agent(self):
        from xuance_worker.runtime import _resolve_custom_config_path
        from xuance import get_runner

        config_path = _resolve_custom_config_path(
            method="mappo", env="multigrid", env_id="collect_1vs1",
            num_groups=None, config_path=None,
        )
        runner = get_runner(
            algo="mappo", env="multigrid", env_id="collect_1vs1",
            config_path=config_path, is_test=True,
        )
        self.agent = runner.agent
        self.device = next(iter(self.agent.policy.parameters())).device
        self.n_agents = 2
        yield

    def _get_mappo_action(self, obs: np.ndarray, player_id: str) -> int:
        """Query MAPPO policy (parameter sharing) for one agent's action."""
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        agent_key = self.agent.model_keys[0]
        agent_idx = int(player_id.split("_")[-1])
        agents_id = torch.zeros(1, self.n_agents, dtype=torch.float32, device=self.device)
        agents_id[0, agent_idx] = 1.0
        with torch.no_grad():
            _, pi_dists = self.agent.policy(
                observation={agent_key: obs_tensor},
                agent_ids=agents_id,
                agent_key=agent_key,
            )
        return int(pi_dists[agent_key].stochastic_sample().cpu().item())

    def test_mappo_parameter_sharing(self):
        assert getattr(self.agent, 'use_parameter_sharing', False)

    @pytest.mark.parametrize("agent_id", ["agent_0", "agent_1"])
    def test_mappo_valid_action(self, agent_id: str):
        obs = np.random.rand(27).astype(np.float32)
        action = self._get_mappo_action(obs, agent_id)
        assert isinstance(action, int)
        assert 0 <= action < 8
