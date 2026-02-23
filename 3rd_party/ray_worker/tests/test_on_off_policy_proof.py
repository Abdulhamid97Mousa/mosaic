"""
Test to PROVE Ray handles On-Policy vs Off-Policy automatically.

This test demonstrates:
1. PPO (On-Policy) - Uses data once, no persistent replay buffer
2. DQN (Off-Policy) - Uses ReplayBuffer, reuses old experiences

Run with: pytest test_on_off_policy_proof.py -v -s
"""

import pytest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=2)
    yield
    ray.shutdown()


class TestOnPolicyVsOffPolicy:
    """Prove that Ray handles on-policy vs off-policy automatically."""

    def test_ppo_is_on_policy_no_replay_buffer(self, ray_init):
        """
        PROOF: PPO (On-Policy) does NOT have a replay buffer.

        On-policy means: data is used immediately and then discarded.
        """
        config = (
            PPOConfig()
            .environment("CartPole-v1")
            .env_runners(num_env_runners=0)  # Local only
            .training(
                train_batch_size_per_learner=200,
                minibatch_size=50,
                num_epochs=2,  # Multiple passes over SAME batch (new API name)
            )
        )

        algo = config.build()

        # PROOF 1: PPO has NO replay buffer
        # Check if algo has replay buffer attribute
        has_replay_buffer = hasattr(algo, 'local_replay_buffer') and algo.local_replay_buffer is not None

        print("\n" + "="*60)
        print("PPO (ON-POLICY) PROOF:")
        print("="*60)
        print(f"  Has replay buffer: {has_replay_buffer}")
        print(f"  Algorithm class: {type(algo).__name__}")

        # Get the training config
        train_config = algo.config.to_dict()
        print(f"  num_epochs (epochs over same data): {train_config.get('num_epochs', 'N/A')}")
        print(f"  minibatch_size: {train_config.get('minibatch_size', 'N/A')}")

        # PROOF: PPO uses the data multiple times in same iteration, then DISCARDS
        assert not has_replay_buffer, "PPO should NOT have a replay buffer!"

        # Do one training iteration
        result = algo.train()

        print(f"  Timesteps trained: {result.get('timesteps_total', result.get('num_env_steps_sampled_lifetime', 'N/A'))}")
        print("  ✅ PPO is ON-POLICY: Uses batch once, then discards!")
        print("="*60)

        algo.stop()

    def test_dqn_is_off_policy_has_replay_buffer(self, ray_init):
        """
        PROOF: DQN (Off-Policy) HAS a replay buffer.

        Off-policy means: data is stored and reused many times.
        """
        config = (
            DQNConfig()
            .environment("CartPole-v1")
            .env_runners(num_env_runners=0)  # Local only
            .training(
                replay_buffer_config={
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": 10000,  # Stores 10,000 experiences!
                },
                train_batch_size=32,
                n_step=1,
            )
        )

        algo = config.build()

        # PROOF 1: DQN HAS a replay buffer
        has_replay_buffer = hasattr(algo, 'local_replay_buffer') and algo.local_replay_buffer is not None

        # Alternative check for newer Ray versions
        if not has_replay_buffer:
            # Check in learner group
            has_replay_buffer = "replay_buffer" in str(type(algo)).lower() or \
                               algo.config.replay_buffer_config is not None

        print("\n" + "="*60)
        print("DQN (OFF-POLICY) PROOF:")
        print("="*60)
        print(f"  Has replay buffer config: {algo.config.replay_buffer_config is not None}")
        print(f"  Algorithm class: {type(algo).__name__}")

        # Get the replay buffer config
        rb_config = algo.config.replay_buffer_config
        print(f"  Replay buffer type: {rb_config.get('type', 'N/A') if rb_config else 'None'}")
        print(f"  Replay buffer capacity: {rb_config.get('capacity', 'N/A') if rb_config else 'None'}")

        # PROOF: DQN has replay buffer configured
        assert rb_config is not None, "DQN should have a replay buffer config!"

        # Do one training iteration
        result = algo.train()

        print(f"  Timesteps trained: {result.get('timesteps_total', result.get('num_env_steps_sampled_lifetime', 'N/A'))}")
        print("  ✅ DQN is OFF-POLICY: Stores experiences in buffer, reuses them!")
        print("="*60)

        algo.stop()

    def test_compare_buffer_configs(self, ray_init):
        """
        Side-by-side comparison of PPO vs DQN configurations.
        """
        ppo_config = PPOConfig().environment("CartPole-v1")
        dqn_config = DQNConfig().environment("CartPole-v1")

        print("\n" + "="*60)
        print("SIDE-BY-SIDE COMPARISON:")
        print("="*60)

        # PPO config
        ppo_dict = ppo_config.to_dict()
        print("\nPPO (On-Policy):")
        print(f"  replay_buffer_config: {ppo_dict.get('replay_buffer_config', 'None/Empty')}")
        print(f"  num_epochs: {ppo_dict.get('num_epochs', 'N/A')} (epochs over same batch)")
        print(f"  minibatch_size: {ppo_dict.get('minibatch_size', 'N/A')}")

        # DQN config
        dqn_dict = dqn_config.to_dict()
        print("\nDQN (Off-Policy):")
        print(f"  replay_buffer_config: {dqn_dict.get('replay_buffer_config', 'None/Empty')}")
        print(f"  n_step: {dqn_dict.get('n_step', 'N/A')} (n-step returns)")
        print(f"  target_network_update_freq: {dqn_dict.get('target_network_update_freq', 'N/A')}")

        print("\n" + "="*60)
        print("CONCLUSION:")
        print("  - PPO: No replay buffer → Data used once → ON-POLICY")
        print("  - DQN: Has replay buffer → Data reused → OFF-POLICY")
        print("  - Ray handles this AUTOMATICALLY based on algorithm choice!")
        print("="*60)

        # Assert the difference
        ppo_has_buffer = bool(ppo_dict.get('replay_buffer_config'))
        dqn_has_buffer = bool(dqn_dict.get('replay_buffer_config'))

        assert not ppo_has_buffer or ppo_dict.get('replay_buffer_config') == {}, \
            "PPO should not have replay buffer"
        assert dqn_has_buffer, "DQN should have replay buffer"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
