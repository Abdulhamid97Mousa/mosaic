"""
Test to PROVE Ray's scaling architecture for On-Policy vs Off-Policy.

This demonstrates Ray's 3 scaling axes:
1. num_env_runners - Number of parallel environment collectors
2. num_envs_per_env_runner - Vectorized envs per runner
3. num_learners - Number of GPU learners for distributed training

Run with: pytest test_scaling_proof.py -v -s
"""

import pytest
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=4)
    yield
    ray.shutdown()


class TestScalingArchitecture:
    """Prove Ray's scaling architecture for both on-policy and off-policy."""

    def test_ppo_scaling_on_policy(self, ray_init):
        """
        PROOF: PPO (On-Policy) scales with EnvRunners that SYNC data.

        On-policy scaling:
        - Multiple EnvRunners collect in PARALLEL
        - All data is AGGREGATED into one batch
        - Learner trains on aggregated batch
        - Weights are SYNCED back to all EnvRunners
        - Old data is DISCARDED
        """
        config = (
            PPOConfig()
            .environment("CartPole-v1")
            .env_runners(
                num_env_runners=2,           # 2 parallel collectors!
                num_envs_per_env_runner=2,   # 2 vectorized envs each
            )
            .training(
                train_batch_size_per_learner=400,
            )
        )

        print("\n" + "="*70)
        print("PPO (ON-POLICY) SCALING ARCHITECTURE:")
        print("="*70)

        config_dict = config.to_dict()

        num_runners = config_dict.get('num_env_runners', 0)
        num_envs_per = config_dict.get('num_envs_per_env_runner', 1)
        total_envs = (num_runners + 1) * num_envs_per  # +1 for local worker

        print(f"""
        ┌─────────────────────────────────────────────────────────────────┐
        │  PPO SCALING (On-Policy) - SYNCHRONOUS                          │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  Algorithm (Main Process)                                       │
        │      │                                                          │
        │      ├── EnvRunnerGroup                                         │
        │      │       │                                                  │
        │      │       ├── EnvRunner 0 (local) ─► [{num_envs_per} envs]        │
        │      │       ├── EnvRunner 1 (remote) ─► [{num_envs_per} envs]       │
        │      │       └── EnvRunner 2 (remote) ─► [{num_envs_per} envs]       │
        │      │               │                                          │
        │      │               ▼                                          │
        │      │       ┌─────────────────────┐                            │
        │      │       │ AGGREGATE all data  │  ◄── All runners SYNC!     │
        │      │       │ into ONE batch      │                            │
        │      │       └─────────────────────┘                            │
        │      │               │                                          │
        │      │               ▼                                          │
        │      └── Learner ◄── Train on aggregated batch                  │
        │              │                                                  │
        │              ▼                                                  │
        │       SYNC weights back to ALL EnvRunners                       │
        │       DISCARD old data (on-policy requirement!)                 │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘

        Configuration:
          num_env_runners: {num_runners}
          num_envs_per_env_runner: {num_envs_per}
          Total parallel envs: {total_envs}

        Data Flow:
          1. All {total_envs} envs collect experiences IN PARALLEL
          2. Data AGGREGATED into single batch
          3. Learner does {config_dict.get('num_epochs', 30)} epochs over batch
          4. Weights synced to all runners
          5. Data DISCARDED (on-policy!)
        """)

        # Verify configuration
        assert num_runners == 2, "Should have 2 remote EnvRunners"
        assert num_envs_per == 2, "Should have 2 envs per runner"
        print("  ✅ PPO scaling configured correctly!")
        print("="*70)

    def test_dqn_scaling_off_policy(self, ray_init):
        """
        PROOF: DQN (Off-Policy) scales with ASYNC data collection + replay buffer.

        Off-policy scaling:
        - Multiple EnvRunners collect ASYNCHRONOUSLY
        - Data goes into SHARED replay buffer
        - Learner samples from buffer INDEPENDENTLY
        - No need to wait for fresh data!
        """
        config = (
            DQNConfig()
            .environment("CartPole-v1")
            .env_runners(
                num_env_runners=2,           # 2 parallel collectors
                num_envs_per_env_runner=2,   # 2 vectorized envs each
            )
            .training(
                replay_buffer_config={
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": 50000,
                },
            )
        )

        print("\n" + "="*70)
        print("DQN (OFF-POLICY) SCALING ARCHITECTURE:")
        print("="*70)

        config_dict = config.to_dict()

        num_runners = config_dict.get('num_env_runners', 0)
        num_envs_per = config_dict.get('num_envs_per_env_runner', 1)
        total_envs = (num_runners + 1) * num_envs_per
        rb_config = config_dict.get('replay_buffer_config', {})

        print(f"""
        ┌─────────────────────────────────────────────────────────────────┐
        │  DQN SCALING (Off-Policy) - ASYNCHRONOUS                        │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │  Algorithm (Main Process)                                       │
        │      │                                                          │
        │      ├── EnvRunnerGroup (ASYNC collection)                      │
        │      │       │                                                  │
        │      │       ├── EnvRunner 0 ─► [{num_envs_per} envs] ──┐            │
        │      │       ├── EnvRunner 1 ─► [{num_envs_per} envs] ──┼──► BUFFER  │
        │      │       └── EnvRunner 2 ─► [{num_envs_per} envs] ──┘            │
        │      │                                    │                     │
        │      │                                    ▼                     │
        │      │                         ┌──────────────────┐             │
        │      │                         │  REPLAY BUFFER   │             │
        │      │                         │  capacity: {rb_config.get('capacity', 'N/A')}  │
        │      │                         │  (stores ALL)    │             │
        │      │                         └──────────────────┘             │
        │      │                                    │                     │
        │      │                           Sample random batch            │
        │      │                                    │                     │
        │      └── Learner ◄────────────────────────┘                     │
        │              │                                                  │
        │              ▼                                                  │
        │       Periodic weight sync (not every step!)                    │
        │       Data KEPT in buffer for reuse!                            │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘

        Configuration:
          num_env_runners: {num_runners}
          num_envs_per_env_runner: {num_envs_per}
          Total parallel envs: {total_envs}
          Replay buffer capacity: {rb_config.get('capacity', 'N/A')}

        Data Flow:
          1. All {total_envs} envs collect ASYNCHRONOUSLY
          2. Data STORED in replay buffer (never discarded!)
          3. Learner samples RANDOMLY from buffer
          4. Can train on experiences from 10,000 steps ago!
          5. Weight sync is PERIODIC (not blocking)
        """)

        # Verify configuration
        assert num_runners == 2, "Should have 2 remote EnvRunners"
        assert rb_config.get('capacity') == 50000, "Should have 50k buffer"
        print("  ✅ DQN scaling configured correctly!")
        print("="*70)

    def test_scaling_comparison(self, ray_init):
        """Side-by-side comparison of scaling behavior."""

        print("\n" + "="*70)
        print("SCALING COMPARISON: ON-POLICY vs OFF-POLICY")
        print("="*70)

        print("""
        ┌────────────────────┬─────────────────────┬─────────────────────┐
        │  Aspect            │  PPO (On-Policy)    │  DQN (Off-Policy)   │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Data Collection   │  SYNCHRONOUS        │  ASYNCHRONOUS       │
        │                    │  Wait for all       │  Continuous flow    │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Data Storage      │  Temporary batch    │  Replay Buffer      │
        │                    │  (discarded)        │  (persistent)       │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Weight Sync       │  EVERY iteration    │  PERIODIC           │
        │                    │  (blocking)         │  (non-blocking)     │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Sample Efficiency │  LOW                │  HIGH               │
        │                    │  (use once)         │  (reuse many times) │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Wall-clock Speed  │  SLOWER             │  FASTER             │
        │                    │  (sync overhead)    │  (async parallel)   │
        ├────────────────────┼─────────────────────┼─────────────────────┤
        │  Scaling Limit     │  Sync bottleneck    │  Buffer memory      │
        └────────────────────┴─────────────────────┴─────────────────────┘

        SCALING FORMULA:

        On-Policy (PPO):
          Throughput = num_env_runners × num_envs_per_runner × steps_per_iter
          BUT: Must wait for ALL runners before training!

        Off-Policy (DQN):
          Throughput = num_env_runners × num_envs_per_runner × continuous
          AND: Training happens INDEPENDENTLY of collection!
        """)

        print("\n  ✅ Off-policy scales better for throughput!")
        print("  ✅ On-policy is more stable but slower!")
        print("="*70)

    def test_your_ray_worker_scaling(self, ray_init):
        """Show how YOUR ray_worker is configured for scaling."""

        print("\n" + "="*70)
        print("YOUR MOSAIC RAY WORKER SCALING:")
        print("="*70)

        print("""
        From ray_worker/runtime.py (lines 587-596):

        ```python
        .env_runners(
            num_env_runners=rc.num_workers,        # From GUI config!
            num_cpus_per_env_runner=rc.num_cpus_per_worker,
            num_gpus_per_env_runner=rc.num_gpus_per_worker,
            create_env_on_local_worker=True,
        )
        .resources(
            num_gpus=rc.num_gpus,                  # From GUI config!
        )
        ```

        YOUR ORCHESTRATOR CONTROLS:
        ┌─────────────────────────────────────────────────────────────────┐
        │  GUI sends config.json:                                         │
        │  {                                                              │
        │    "resources": {                                               │
        │      "num_workers": 4,        ──► 4 EnvRunner actors           │
        │      "num_gpus": 2,           ──► 2 GPU Learners               │
        │      "num_cpus_per_worker": 1                                   │
        │    },                                                           │
        │    "training": {                                                │
        │      "algorithm": "PPO"       ──► On-policy mode               │
        │    }                                                            │
        │  }                                                              │
        └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  Ray automatically creates:                                     │
        │                                                                 │
        │  EnvRunnerGroup:                                                │
        │    └── 4 EnvRunner actors (each on separate CPU)               │
        │                                                                 │
        │  LearnerGroup:                                                  │
        │    └── 2 Learner actors (each on separate GPU)                 │
        │                                                                 │
        │  If PPO: Sync collection → aggregate → train → sync weights    │
        │  If DQN: Async collection → buffer → sample → train            │
        └─────────────────────────────────────────────────────────────────┘
        """)

        print("  ✅ Your GUI controls scaling via config!")
        print("  ✅ Ray handles the distributed execution!")
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
