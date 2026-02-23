"""Test XuanCe MAPPO policy inference from trained checkpoint.

Verifies the trained MAPPO checkpoint can be loaded and queried for actions
using the Collect 1vs1 IndAgObs environment (the Phase 1 curriculum env).

Checkpoint: var/trainer/custom_scripts/01KHFGBCEXSVRTB3WA3QFWRJ91/checkpoints/torch/collect_1vs1/seed_1_2026_0215_095902/final_train_model.pth
Architecture: MAPPO with parameter sharing, Basic_MLP representation, Categorical actor
  - Representation: Linear(27,64) -> ReLU -> LayerNorm(64) -> Linear(64,64) -> ReLU -> LayerNorm(64)
  - Actor: Linear(66,64) -> ReLU -> LayerNorm(64) -> Linear(64,64) -> ReLU -> LayerNorm(64) -> Linear(64,7)
  - The extra 2 dims in actor input = agent one-hot (parameter sharing with 2 agents)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "var/trainer/custom_scripts/01KHFGBCEXSVRTB3WA3QFWRJ91"
    / "checkpoints/torch/collect_1vs1/seed_1_2026_0215_095902"
    / "final_train_model.pth"
)
ENV_ID = "MosaicMultiGrid-Collect-1vs1-IndAgObs-v0"

# Architecture constants (from collect_1vs1.yaml)
OBS_DIM = 27       # IndAgObs (3,3,3) flattened
N_ACTIONS = 7      # 7 discrete actions
HIDDEN = 64        # representation_hidden_size and actor_hidden_size
N_AGENTS = 2       # 1v1 → 2 agents with parameter sharing


# ---------------------------------------------------------------------------
# Model reconstruction (mirrors XuanCe's Basic_MLP + Categorical_MAAC_Policy)
# ---------------------------------------------------------------------------

class BasicMLP(nn.Module):
    """Matches XuanCe Basic_MLP representation with LayerNorm."""

    def __init__(self, input_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),   # model.0
            nn.ReLU(),                      # model.1
            nn.LayerNorm(hidden),           # model.2
            nn.Linear(hidden, hidden),      # model.3
            nn.ReLU(),                      # model.4
            nn.LayerNorm(hidden),           # model.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CategoricalActor(nn.Module):
    """Matches XuanCe Categorical actor head (takes repr output + agent id)."""

    def __init__(self, input_dim: int, n_actions: int, hidden: int = HIDDEN):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),   # model.0
            nn.ReLU(),                      # model.1
            nn.LayerNorm(hidden),           # model.2
            nn.Linear(hidden, hidden),      # model.3
            nn.ReLU(),                      # model.4
            nn.LayerNorm(hidden),           # model.5
            nn.Linear(hidden, n_actions),   # model.6
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MAPPOPolicy(nn.Module):
    """Minimal reconstruction of XuanCe Categorical_MAAC_Policy for inference.

    Parameter-sharing MAPPO: one set of weights for agent_0, applied to all
    agents by concatenating a one-hot agent ID to the representation output.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        n_agents: int = N_AGENTS,
        hidden: int = HIDDEN,
    ):
        super().__init__()
        self.n_agents = n_agents
        # XuanCe stores per-agent modules in dicts keyed by agent name
        self.actor_representation = nn.ModuleDict({
            "agent_0": BasicMLP(obs_dim, hidden),
        })
        self.actor = nn.ModuleDict({
            "agent_0": CategoricalActor(hidden + n_agents, n_actions, hidden),
        })

    def select_action(
        self, obs: np.ndarray, agent_index: int = 0
    ) -> int:
        """Run actor forward pass and return greedy action."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
            repr_out = self.actor_representation["agent_0"](obs_t)  # (1, hidden)
            # Concatenate one-hot agent ID
            agent_id = torch.zeros(1, self.n_agents)
            agent_id[0, agent_index] = 1.0
            actor_input = torch.cat([repr_out, agent_id], dim=-1)  # (1, hidden+n_agents)
            logits = self.actor["agent_0"](actor_input)  # (1, n_actions)
            return int(logits.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint():
    """Load checkpoint state_dict, skip if file missing."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return torch.load(str(CHECKPOINT_PATH), map_location="cpu")


@pytest.fixture(scope="module")
def policy(checkpoint):
    """Build MAPPOPolicy and load weights from checkpoint."""
    model = MAPPOPolicy()
    # Filter to only actor keys (skip critic)
    actor_keys = {
        k: v for k, v in checkpoint.items()
        if k.startswith("actor_representation.") or
        (k.startswith("actor.") and not k.startswith("actor_representation"))
    }
    model.load_state_dict(actor_keys, strict=True)
    model.eval()
    return model


@pytest.fixture(scope="module")
def env():
    """Create the Collect 1vs1 IndAgObs environment (unwrapped).

    The raw env uses integer-keyed dicts: ``{0: {image, direction, mission}, 1: ...}``.
    XuanCe flattens ``image`` to (27,) and renames keys to ``agent_0``, ``agent_1``.
    """
    import gymnasium as gym

    # Register mosaic_multigrid environments
    import mosaic_multigrid.envs  # noqa: F401

    e = gym.make(ENV_ID, render_mode=None)
    yield e
    e.close()


def _extract_obs(raw_obs: dict) -> dict[str, np.ndarray]:
    """Flatten raw multigrid observations like XuanCe wrapper does.

    ``raw_obs`` is ``{0: {image: (3,3,3), ...}, 1: ...}`` →
    returns ``{agent_0: (27,), agent_1: (27,)}``.
    """
    return {
        f"agent_{i}": raw_obs[i]["image"].flatten().astype(np.float32)
        for i in sorted(raw_obs.keys())
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    """Verify the checkpoint file structure and loadability."""

    def test_checkpoint_is_ordered_dict(self, checkpoint):
        assert isinstance(checkpoint, dict)

    def test_checkpoint_has_actor_keys(self, checkpoint):
        actor_keys = [k for k in checkpoint if k.startswith("actor.")]
        assert len(actor_keys) > 0, "No actor keys in checkpoint"

    def test_checkpoint_has_representation_keys(self, checkpoint):
        repr_keys = [k for k in checkpoint if k.startswith("actor_representation.")]
        assert len(repr_keys) > 0, "No representation keys in checkpoint"

    def test_actor_output_shape_matches_n_actions(self, checkpoint):
        output_weight = checkpoint["actor.agent_0.model.6.weight"]
        assert output_weight.shape[0] == N_ACTIONS

    def test_representation_input_matches_obs_dim(self, checkpoint):
        input_weight = checkpoint["actor_representation.agent_0.model.0.weight"]
        assert input_weight.shape[1] == OBS_DIM


class TestPolicyInference:
    """Verify the loaded policy produces valid actions."""

    def test_policy_loads_weights(self, policy):
        """Policy loads without errors."""
        assert policy is not None

    def test_select_action_returns_int(self, policy):
        """select_action returns a Python int."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = policy.select_action(obs, agent_index=0)
        assert isinstance(action, int)

    def test_action_in_valid_range(self, policy):
        """Action is within [0, N_ACTIONS)."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = policy.select_action(obs, agent_index=0)
        assert 0 <= action < N_ACTIONS

    def test_both_agents_produce_actions(self, policy):
        """Both agent_0 and agent_1 produce valid actions (parameter sharing)."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for idx in range(N_AGENTS):
            action = policy.select_action(obs, agent_index=idx)
            assert 0 <= action < N_ACTIONS, f"Agent {idx} gave invalid action {action}"

    def test_deterministic_with_same_input(self, policy):
        """Same observation should produce same greedy action."""
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        a1 = policy.select_action(obs, agent_index=0)
        a2 = policy.select_action(obs, agent_index=0)
        assert a1 == a2

    def test_different_observations_can_differ(self, policy):
        """Different observations should (eventually) produce different actions."""
        actions = set()
        for _ in range(100):
            obs = np.random.randn(OBS_DIM).astype(np.float32) * 5.0
            actions.add(policy.select_action(obs, agent_index=0))
        assert len(actions) > 1, "Policy always returns the same action (possibly collapsed)"


class TestPolicyWithEnvironment:
    """End-to-end: load env, reset, get obs, query policy."""

    def test_env_obs_shape_matches_policy(self, env):
        """Environment observation shape matches what the policy expects."""
        raw_obs, _ = env.reset(seed=42)
        obs = _extract_obs(raw_obs)
        assert obs["agent_0"].shape == (OBS_DIM,), (
            f"Expected ({OBS_DIM},), got {obs['agent_0'].shape}"
        )

    def test_policy_on_real_observation(self, policy, env):
        """Policy produces valid action from real environment observation."""
        raw_obs, _ = env.reset(seed=42)
        obs = _extract_obs(raw_obs)
        for agent_id, agent_obs in obs.items():
            idx = int(agent_id.split("_")[-1])
            action = policy.select_action(agent_obs, agent_index=idx)
            assert 0 <= action < N_ACTIONS, (
                f"{agent_id} action={action} not in [0, {N_ACTIONS})"
            )

    def test_full_episode_step(self, policy, env):
        """Run one full step: reset → infer actions → step."""
        raw_obs, _ = env.reset(seed=42)
        obs = _extract_obs(raw_obs)
        actions = {}
        for agent_id, agent_obs in obs.items():
            idx = int(agent_id.split("_")[-1])
            actions[idx] = policy.select_action(agent_obs, agent_index=idx)

        raw_obs2, rewards, terminated, truncated, infos = env.step(actions)
        obs2 = _extract_obs(raw_obs2)
        for agent_id, agent_obs in obs2.items():
            assert agent_obs.shape == (OBS_DIM,)

    def test_multi_step_rollout(self, policy, env):
        """Run 10 steps to verify the policy doesn't crash mid-episode."""
        raw_obs, _ = env.reset(seed=42)
        for step in range(10):
            obs = _extract_obs(raw_obs)
            actions = {}
            for agent_id, agent_obs in obs.items():
                idx = int(agent_id.split("_")[-1])
                actions[idx] = policy.select_action(agent_obs, agent_index=idx)
            raw_obs, rewards, terminated, truncated, infos = env.step(actions)
            # Check for natural termination
            if all(terminated.get(i, False) for i in range(N_AGENTS)):
                raw_obs, _ = env.reset(seed=42 + step)
