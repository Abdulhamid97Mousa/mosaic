"""Operators Worker - Baseline operators for credit assignment experiments.

This worker provides simple baseline operators (Random, No-op, Cycling) for ablation
studies and credit assignment research in hybrid teams.

Multi-Operator Support:
    Multiple instances can run simultaneously, each as a separate subprocess with
    independent telemetry logging to var/operators/telemetry/.

Usage:
    python -m operators_worker.cli --run-id operator_0_abc123 --behavior random --task BabyAI-GoToRedBall-v0

Operators:
    - RandomOperator: Selects uniformly random actions
    - NoopOperator: Always returns action 0 (no-operation)
    - CyclingOperator: Cycles through actions sequentially (0, 1, 2, ..., n-1, repeat)
"""

__version__ = "0.1.0"

from operators_worker.config import OperatorsWorkerConfig
from operators_worker.operators import RandomOperator, NoopOperator, CyclingOperator
from operators_worker.runtime import OperatorsWorkerRuntime

__all__ = [
    "OperatorsWorkerConfig",
    "RandomOperator",
    "NoopOperator",
    "CyclingOperator",
    "OperatorsWorkerRuntime",
]
