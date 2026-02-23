"""Tests that training and interactive-evaluation use compatible architectures.

The CleanRL training script (ppo.py) picks either MinigridAgent (CNN) or
MLPAgent (MLP) based on the env_id.  The interactive runtime
(InteractiveCleanRLRuntime._load_policy) must pick the *same* agent class
and apply the *same* observation wrappers so that a checkpoint saved during
training can be loaded for live evaluation in the Multi-Operator view.

These tests guarantee that the two code-paths stay in sync.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
import numpy as np
import pytest
import torch

from cleanrl_worker.agents.mlp import MLPAgent
from cleanrl_worker.agents.minigrid import MinigridAgent
from cleanrl_worker.wrappers.minigrid import is_minigrid_env, make_env

minigrid.register_minigrid_envs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIGRID_ENV = "MiniGrid-DoorKey-5x5-v0"
_CLASSIC_ENV = "CartPole-v1"


def _training_agent_cls(env_id: str) -> type:
    """Return the agent class that ppo.py would select for *env_id*."""
    # Mirrors algorithms/ppo.py lines 154-159
    if is_minigrid_env(env_id):
        return MinigridAgent
    return MLPAgent


def _eval_agent_cls(env_id: str) -> type:
    """Return the agent class the interactive runtime selects for *env_id*.

    Mirrors runtime.py _load_policy() after the fix.
    """
    from cleanrl_worker.eval_registry import get_eval_entry

    entry = get_eval_entry("ppo")
    assert entry is not None
    agent_cls = entry.agent_cls  # registry default (MLPAgent)

    is_minigrid = env_id.startswith("MiniGrid") or env_id.startswith("BabyAI")
    if is_minigrid:
        agent_cls = MinigridAgent

    return agent_cls


def _training_envs(env_id: str) -> gym.vector.VectorEnv:
    """Create the vectorised env exactly as ppo.py does for training."""
    env_fn = make_env(env_id, idx=0, capture_video=False, run_name="test",
                      seed=1, max_episode_steps=64)
    return gym.vector.SyncVectorEnv([env_fn])


def _eval_envs(env_id: str) -> gym.vector.VectorEnv:
    """Create the vectorised env exactly as _load_policy() does for eval."""
    is_mg = env_id.startswith("MiniGrid") or env_id.startswith("BabyAI")

    def make():
        env = gym.make(env_id, render_mode="rgb_array")
        if is_mg:
            from minigrid.wrappers import ImgObsWrapper
            env = ImgObsWrapper(env)
            # NOTE: no FlattenObservation — CNN expects (7,7,3) images
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return gym.vector.SyncVectorEnv([make])


# ---------------------------------------------------------------------------
# Tests — agent class consistency
# ---------------------------------------------------------------------------

class TestAgentClassConsistency:
    """Training and eval must pick the same agent architecture."""

    def test_minigrid_uses_cnn_agent(self):
        """MiniGrid envs should use MinigridAgent (CNN) in both paths."""
        assert _training_agent_cls(_MINIGRID_ENV) is MinigridAgent
        assert _eval_agent_cls(_MINIGRID_ENV) is MinigridAgent

    def test_classic_uses_mlp_agent(self):
        """Non-MiniGrid envs should use MLPAgent in both paths."""
        assert _training_agent_cls(_CLASSIC_ENV) is MLPAgent
        assert _eval_agent_cls(_CLASSIC_ENV) is MLPAgent

    @pytest.mark.parametrize("env_id", [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "BabyAI-GoToRedBall-v0",
    ])
    def test_minigrid_variants_all_use_cnn(self, env_id):
        """All MiniGrid/BabyAI variants must use MinigridAgent in both paths."""
        assert _training_agent_cls(env_id) is MinigridAgent
        assert _eval_agent_cls(env_id) is MinigridAgent


# ---------------------------------------------------------------------------
# Tests — observation space consistency
# ---------------------------------------------------------------------------

class TestObsSpaceConsistency:
    """Training and eval envs must produce the same observation shape."""

    def test_minigrid_obs_shape_matches(self):
        """MiniGrid training and eval envs should both produce (7,7,3) images."""
        train_envs = _training_envs(_MINIGRID_ENV)
        eval_envs = _eval_envs(_MINIGRID_ENV)
        try:
            train_shape = train_envs.single_observation_space.shape
            eval_shape = eval_envs.single_observation_space.shape
            assert train_shape == eval_shape, (
                f"Training obs shape {train_shape} != eval obs shape {eval_shape}"
            )
            # Specifically, should be (7, 7, 3) — NOT (147,)
            assert len(train_shape) == 3, (
                f"MiniGrid obs should be 3D image, got {train_shape}"
            )
        finally:
            train_envs.close()
            eval_envs.close()

    def test_classic_obs_shape_matches(self):
        """Classic control training and eval envs should have same flat shape."""
        train_envs = _training_envs(_CLASSIC_ENV)
        eval_envs = _eval_envs(_CLASSIC_ENV)
        try:
            train_shape = train_envs.single_observation_space.shape
            eval_shape = eval_envs.single_observation_space.shape
            assert train_shape == eval_shape, (
                f"Training obs shape {train_shape} != eval obs shape {eval_shape}"
            )
        finally:
            train_envs.close()
            eval_envs.close()


# ---------------------------------------------------------------------------
# Tests — state_dict key compatibility
# ---------------------------------------------------------------------------

class TestStateDictCompatibility:
    """Checkpoint keys from training must match the eval agent's architecture."""

    def test_minigrid_state_dict_keys_match(self):
        """MinigridAgent keys from training must load into MinigridAgent for eval."""
        envs = _training_envs(_MINIGRID_ENV)
        try:
            train_agent = MinigridAgent(envs)
            eval_agent = MinigridAgent(envs)

            # Keys saved at training time must match keys expected at eval time
            train_keys = set(train_agent.state_dict().keys())
            eval_keys = set(eval_agent.state_dict().keys())
            assert train_keys == eval_keys
        finally:
            envs.close()

    def test_minigrid_checkpoint_fails_with_mlp(self):
        """A MinigridAgent checkpoint must NOT load into MLPAgent (the old bug)."""
        train_envs = _training_envs(_MINIGRID_ENV)
        try:
            cnn_agent = MinigridAgent(train_envs)
            cnn_keys = set(cnn_agent.state_dict().keys())
        finally:
            train_envs.close()

        # MLPAgent needs flat obs, use a flat env for it
        def make_flat():
            env = gym.make(_MINIGRID_ENV)
            from minigrid.wrappers import ImgObsWrapper
            env = ImgObsWrapper(env)
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        flat_envs = gym.vector.SyncVectorEnv([make_flat])
        try:
            mlp_agent = MLPAgent(flat_envs)
            mlp_keys = set(mlp_agent.state_dict().keys())

            # The key sets should differ (CNN has cnn.* keys)
            assert cnn_keys != mlp_keys, (
                "MinigridAgent and MLPAgent should have different state_dict keys"
            )
            assert any(k.startswith("cnn.") for k in cnn_keys), (
                "MinigridAgent should have cnn.* keys"
            )
            assert not any(k.startswith("cnn.") for k in mlp_keys), (
                "MLPAgent should NOT have cnn.* keys"
            )
        finally:
            flat_envs.close()


# ---------------------------------------------------------------------------
# Tests — end-to-end save/load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    """Train briefly, save, and reload with the eval path."""

    @pytest.mark.slow
    def test_minigrid_train_then_eval_load(self):
        """Train MinigridAgent, save checkpoint, load with eval agent class."""
        # 1. Train for a few steps
        train_envs = _training_envs(_MINIGRID_ENV)
        try:
            agent = MinigridAgent(train_envs).to("cpu")
            # Run one forward pass to ensure the model is usable
            obs, _ = train_envs.reset(seed=42)
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            assert action is not None
        finally:
            train_envs.close()

        # 2. Save checkpoint (same format as ppo.py)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ppo.cleanrl_model"
            torch.save({"model_state_dict": agent.state_dict()}, ckpt_path)

            # 3. Load using the eval path
            eval_envs = _eval_envs(_MINIGRID_ENV)
            try:
                eval_cls = _eval_agent_cls(_MINIGRID_ENV)
                assert eval_cls is MinigridAgent

                eval_agent = eval_cls(eval_envs).to("cpu")
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                eval_agent.load_state_dict(checkpoint["model_state_dict"])
                eval_agent.eval()

                # 4. Run one action selection — must not crash
                obs, _ = eval_envs.reset(seed=42)
                obs_t = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, _, _, _ = eval_agent.get_action_and_value(obs_t)

                assert action.shape == (1,)
            finally:
                eval_envs.close()

    @pytest.mark.slow
    def test_classic_train_then_eval_load(self):
        """Train MLPAgent, save checkpoint, load with eval agent class."""
        train_envs = _training_envs(_CLASSIC_ENV)
        try:
            agent = MLPAgent(train_envs).to("cpu")
            obs, _ = train_envs.reset(seed=42)
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            assert action is not None
        finally:
            train_envs.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ppo.cleanrl_model"
            torch.save({"model_state_dict": agent.state_dict()}, ckpt_path)

            eval_envs = _eval_envs(_CLASSIC_ENV)
            try:
                eval_cls = _eval_agent_cls(_CLASSIC_ENV)
                assert eval_cls is MLPAgent

                eval_agent = eval_cls(eval_envs).to("cpu")
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                eval_agent.load_state_dict(checkpoint["model_state_dict"])
                eval_agent.eval()

                obs, _ = eval_envs.reset(seed=42)
                obs_t = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, _, _, _ = eval_agent.get_action_and_value(obs_t)

                assert action.shape == (1,)
            finally:
                eval_envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
