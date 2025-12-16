"""XuanCe worker integration for MOSAIC BDI-RL framework.

This module provides the integration layer between MOSAIC's GUI and
XuanCe's comprehensive deep reinforcement learning library.

Key Features:
- Single-agent RL: DQN, PPO, SAC, TD3, DDPG, A2C, and many more
- Multi-agent RL (MARL): MAPPO, MADDPG, QMIX, VDN, COMA, and more
- Support for PyTorch, TensorFlow, and MindSpore backends
- Integration with MOSAIC's PolicyMappingService

Supported Algorithms:
    Single-Agent: DQN, DDQN, DuelingDQN, NoisyDQN, C51, QRDQN, PG, A2C, PPO,
                  PPG, DDPG, TD3, SAC, DRQN, DreamerV2, DreamerV3
    Multi-Agent:  IQL, VDN, QMIX, WQMIX, QTRAN, DCG, MAPPO, MADDPG, MATD3,
                  MASAC, IPPO, ISAC, IAC, COMA, MeanField

Example:
    from xuance_worker import XuanCeWorker

    worker = XuanCeWorker(config={
        "algorithm": "MAPPO",
        "env_id": "simple_spread_v3",
        "backend": "torch",
    })
    worker.train(total_timesteps=100000)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
