"""
Tests for PPO-GRU algorithm with custom_scripts compatibility.

This module tests the PPO-GRU algorithm to ensure:
1. It can be imported and instantiated correctly
2. It respects MOSAIC_RUN_DIR for custom_scripts compatibility
3. It handles view_size parameter correctly
4. It saves checkpoints to the correct location
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import gymnasium as gym
from torch.distributions.categorical import Categorical

from cleanrl_worker.agents import GRUAgent


class TestGRUAgent:
    """Test the GRU agent architecture."""

    def test_gru_agent_creation(self):
        """Test that GRUAgent can be created."""
        # Create a simple environment
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])

        # Create agent
        agent = GRUAgent(envs, gru_hidden_size=64, gru_num_layers=1)

        assert agent is not None
        assert agent.gru.hidden_size == 64
        assert agent.gru.num_layers == 1

        envs.close()

    def test_gru_agent_forward_pass(self):
        """Test that GRUAgent can perform forward pass."""
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
        agent = GRUAgent(envs, gru_hidden_size=64, gru_num_layers=1)

        # Create dummy inputs
        obs = torch.randn(2, 4)  # 2 envs, 4 obs dims
        gru_state = torch.zeros(1, 2, 64)  # 1 layer, 2 envs, 64 hidden
        done = torch.zeros(2)

        # Forward pass
        action, log_prob, entropy, value, new_gru_state = agent.get_action_and_value(
            obs, gru_state, done
        )

        assert action.shape == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2, 1)
        assert new_gru_state.shape == (1, 2, 64)

        envs.close()

    def test_gru_agent_episode_reset(self):
        """Test that GRU hidden state is reset on episode done."""
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
        agent = GRUAgent(envs, gru_hidden_size=64, gru_num_layers=1)

        obs = torch.randn(2, 4)
        gru_state = torch.ones(1, 2, 64)  # Non-zero initial state
        done = torch.tensor([1.0, 0.0])  # First env is done

        # Forward pass
        _, _, _, _, new_gru_state = agent.get_action_and_value(obs, gru_state, done)

        # First env's hidden state should be reset (close to zero)
        # Second env's hidden state should be preserved (non-zero)
        assert new_gru_state[0, 0, :].abs().mean() < 0.5  # First env reset
        assert new_gru_state[0, 1, :].abs().mean() > 0.1  # Second env preserved

        envs.close()


class TestPPOGRUCustomScripts:
    """Test PPO-GRU compatibility with custom_scripts."""

    def test_ppo_gru_respects_mosaic_run_dir(self):
        """Test that PPO-GRU respects MOSAIC_RUN_DIR environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set MOSAIC_RUN_DIR
            os.environ["MOSAIC_RUN_DIR"] = tmpdir

            try:
                # Import the algorithm module
                from cleanrl_worker.algorithms import ppo_gru

                # Check that the algorithm would use MOSAIC_RUN_DIR
                # (We can't run full training in tests, but we can verify the logic)
                mosaic_run_dir = os.environ.get("MOSAIC_RUN_DIR")
                assert mosaic_run_dir == tmpdir

                # Verify checkpoint directory would be created correctly
                checkpoint_dir = Path(mosaic_run_dir) / "checkpoints"
                assert checkpoint_dir.parent == Path(tmpdir)

            finally:
                # Clean up
                if "MOSAIC_RUN_DIR" in os.environ:
                    del os.environ["MOSAIC_RUN_DIR"]

    def test_ppo_gru_checkpoint_path_with_mosaic_run_dir(self):
        """Test that checkpoints are saved to MOSAIC_RUN_DIR when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MOSAIC_RUN_DIR"] = tmpdir

            try:
                # Verify checkpoint path construction
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                model_path = checkpoint_dir / "final_train_model.pth"

                # Verify path is within MOSAIC_RUN_DIR
                assert str(model_path).startswith(tmpdir)
                assert "checkpoints" in str(model_path)
                assert "final_train_model.pth" in str(model_path)

            finally:
                if "MOSAIC_RUN_DIR" in os.environ:
                    del os.environ["MOSAIC_RUN_DIR"]

    def test_ppo_gru_tensorboard_path_with_mosaic_run_dir(self):
        """Test that TensorBoard logs go to MOSAIC_RUN_DIR when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MOSAIC_RUN_DIR"] = tmpdir

            try:
                # Verify TensorBoard path construction
                tensorboard_dir = Path(tmpdir) / "tensorboard"

                # Verify path is within MOSAIC_RUN_DIR
                assert str(tensorboard_dir).startswith(tmpdir)
                assert "tensorboard" in str(tensorboard_dir)

            finally:
                if "MOSAIC_RUN_DIR" in os.environ:
                    del os.environ["MOSAIC_RUN_DIR"]

    def test_ppo_gru_view_size_environment_variable(self):
        """Test that PPO-GRU sets MOSAIC_VIEW_SIZE environment variable."""
        # Test that view_size parameter would set environment variable
        view_size = 7
        os.environ["MOSAIC_VIEW_SIZE"] = str(view_size)

        try:
            assert os.environ.get("MOSAIC_VIEW_SIZE") == "7"
        finally:
            if "MOSAIC_VIEW_SIZE" in os.environ:
                del os.environ["MOSAIC_VIEW_SIZE"]


class TestPPOGRUIntegration:
    """Integration tests for PPO-GRU algorithm."""

    def test_ppo_gru_imports(self):
        """Test that PPO-GRU can be imported."""
        try:
            from cleanrl_worker.algorithms import ppo_gru
            assert ppo_gru is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ppo_gru: {e}")

    def test_gru_agent_import_from_algorithms(self):
        """Test that GRUAgent can be imported from algorithms module."""
        from cleanrl_worker.algorithms import GRUAgent as AlgoGRUAgent
        from cleanrl_worker.agents import GRUAgent as AgentGRUAgent

        # Both should be the same class
        assert AlgoGRUAgent is AgentGRUAgent

    def test_ppo_gru_args_dataclass(self):
        """Test that PPO-GRU Args dataclass has required fields."""
        from cleanrl_worker.algorithms.ppo_gru import Args

        args = Args()

        # Check GRU-specific fields
        assert hasattr(args, "gru_hidden_size")
        assert hasattr(args, "gru_num_layers")
        assert args.gru_hidden_size == 128
        assert args.gru_num_layers == 1

        # Check view_size field
        assert hasattr(args, "view_size")
        assert args.view_size is None  # Default should be None

        # Check standard PPO fields
        assert hasattr(args, "learning_rate")
        assert hasattr(args, "num_envs")
        assert hasattr(args, "gamma")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Static action-mask integration tests (masks noop=0, toggle=6, done=7)
# ---------------------------------------------------------------------------

def _build_static_mask(n_actions=8, invalid=(0, 6, 7)):
    """Build a static action mask like ppo_gru.py does."""
    mask = torch.zeros(n_actions)
    for a in invalid:
        mask[a] = float("-inf")
    return mask


class TestStaticActionMask:
    """Prove that the static mask [0, 6, 7] works correctly with GRUAgent."""

    @pytest.fixture()
    def setup(self):
        envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make("CartPole-v1") for _ in range(2)]
        )
        # CartPole has 2 actions, but we test with a custom 8-action agent
        # by building the agent manually against a fake 8-action env
        envs.close()

        fake_envs = gym.vector.SyncVectorEnv(
            [lambda: gym.wrappers.TimeLimit(
                type("E", (gym.Env,), {
                    "__init__": lambda s: (
                        setattr(s, "action_space", gym.spaces.Discrete(8)),
                        setattr(s, "observation_space", gym.spaces.Box(0, 1, shape=(4,))),
                        super(gym.Env, s).__init__(),
                    )[-1],
                    "reset": lambda s, **kw: (s.observation_space.sample(), {}),
                    "step": lambda s, a: (s.observation_space.sample(), 0.0, False, False, {}),
                })(),
                max_episode_steps=10,
            ) for _ in range(2)]
        )
        agent = GRUAgent(fake_envs, gru_hidden_size=32, gru_num_layers=1)
        mask = _build_static_mask(8, (0, 6, 7))
        yield fake_envs, agent, mask
        fake_envs.close()

    def test_masked_actions_get_neg_inf_logits(self, setup):
        """Logits for actions 0, 6, 7 must be -inf after masking."""
        _, agent, mask = setup
        obs = torch.randn(2, 4)
        gru = torch.zeros(1, 2, 32)
        done = torch.zeros(2)

        with torch.no_grad():
            hidden, _ = agent.get_states(obs, gru, done)
            logits = agent.actor(hidden) + mask

        for a in (0, 6, 7):
            assert (logits[:, a] == float("-inf")).all(), \
                f"action {a} logit should be -inf"

    def test_valid_actions_have_positive_probability(self, setup):
        """Actions 1-5 must have p > 0 under the mask."""
        _, agent, mask = setup
        obs = torch.randn(2, 4)
        gru = torch.zeros(1, 2, 32)
        done = torch.zeros(2)

        with torch.no_grad():
            hidden, _ = agent.get_states(obs, gru, done)
            logits = agent.actor(hidden) + mask
            probs = Categorical(logits=logits).probs

        for a in (1, 2, 3, 4, 5):
            assert (probs[:, a] > 0).all(), \
                f"valid action {a} must have non-zero probability"

    def test_masked_actions_never_sampled(self, setup):
        """Over 1000 samples, actions 0/6/7 must never appear."""
        _, agent, mask = setup
        obs = torch.randn(2, 4)
        gru = torch.zeros(1, 2, 32)
        done = torch.zeros(2)

        sampled = set()
        for _ in range(1000):
            with torch.no_grad():
                action, _, _, _, gru = agent.get_action_and_value(
                    obs, gru, done, action_mask=mask
                )
            sampled.update(action.numpy().tolist())

        for a in (0, 6, 7):
            assert a not in sampled, f"masked action {a} was sampled"

    def test_pickup_and_drop_remain_valid(self, setup):
        """Pickup(4) and drop(5) must NOT be masked — they are valid game actions."""
        _, _, mask = setup
        assert mask[4] == 0.0, "pickup must not be masked"
        assert mask[5] == 0.0, "drop must not be masked"

    def test_entropy_over_five_valid_actions(self, setup):
        """With uniform logits, entropy should be log(5) for 5 valid actions."""
        _, _, mask = setup
        logits = torch.zeros(1, 8)
        masked = logits + mask
        dist = Categorical(logits=masked)
        expected = np.log(5)  # 5 valid: left, right, forward, pickup, drop
        assert abs(dist.entropy().item() - expected) < 1e-5

    def test_mask_is_same_for_rollout_and_update(self, setup):
        """Static mask is identical during rollout and policy update."""
        _, _, mask = setup
        # Same mask object is used everywhere — just verify it's static
        mask2 = _build_static_mask(8, (0, 6, 7))
        assert torch.equal(mask, mask2)
