"""Standalone actor extractors for deploying XuanCe-trained MARL policies.

Option B deployment: extract just the actor network from a trained checkpoint
and run single-agent inference without XuanCe's multi-agent batching.

Supports:
    - MAPPOActor:    MLP-based MAPPO/IPPO policies
    - MAPPOGRUActor: GRU-based MAPPO/IPPO policies (maintains hidden state)
"""

from xuance_worker.algorithms.mappo_actor import MAPPOActor
from xuance_worker.algorithms.mappo_gru_actor import MAPPOGRUActor

__all__ = ["MAPPOActor", "MAPPOGRUActor"]
