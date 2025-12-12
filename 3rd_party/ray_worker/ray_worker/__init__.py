"""Ray/RLlib worker integration for MOSAIC BDI-RL framework.

This module provides the integration layer between MOSAIC's GUI and
Ray/RLlib's distributed reinforcement learning capabilities.

Key Features:
- Multi-agent training with RLlib algorithms (PPO, DQN, A2C, IMPALA, APPO)
- Distributed training across multiple workers
- Support for both discrete and continuous action spaces
- Integration with MOSAIC's PolicyMappingService

Example:
    from ray_worker import RLlibWorker

    worker = RLlibWorker(config={
        "algorithm": "PPO",
        "num_workers": 4,
        "framework": "torch",
    })
    worker.train(env_id="CartPole-v1", total_timesteps=100000)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
