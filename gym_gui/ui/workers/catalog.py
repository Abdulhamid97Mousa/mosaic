"""Worker catalog for training and analytics integrations.

This module centralises metadata about available worker integrations so the UI can
present consistent choices. Each worker definition describes how the control panel
should expose forms, whether evaluation (policy load) is supported, and which
capabilities (telemetry vs. analytics) the worker provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class WorkerDefinition:
    """Describe a worker integration exposed through the GUI."""

    worker_id: str
    display_name: str
    description: str
    supports_training: bool = True
    supports_policy_load: bool = False
    requires_live_telemetry: bool = True
    provides_fast_analytics: bool = False

    def capabilities(self) -> Tuple[str, ...]:
        """Return a human-readable tuple of capability labels."""
        labels: list[str] = []
        if self.requires_live_telemetry:
            labels.append("Live telemetry")
        if self.provides_fast_analytics:
            labels.append("Analytics artifacts")
        if self.supports_training:
            labels.append("Training")
        if self.supports_policy_load:
            labels.append("Policy evaluation")
        return tuple(labels)


def get_worker_catalog() -> Tuple[WorkerDefinition, ...]:
    """Return the catalog of worker integrations recognised by the UI.

    Note: SPADE-BDI and CleanRL workers have been removed from Multi-Agent Mode.
    Only Ray RLlib and PettingZoo workers are available for multi-agent training.
    """
    return (
        WorkerDefinition(
            worker_id="ray_worker",
            display_name="Ray RLlib Worker",
            description=(
                "Distributed multi-agent reinforcement learning using Ray RLlib. "
                "Supports multiple training paradigms: Parameter Sharing, Independent Learning, "
                "Self-Play, and Shared Value Function (CTDE). "
                "Works with PettingZoo environments (SISL, Classic, Butterfly, MPE). "
                "FastLane integration for live training visualization with per-worker grid."
            ),
            supports_training=True,
            supports_policy_load=True,
            requires_live_telemetry=False,
            provides_fast_analytics=True,
        ),
        WorkerDefinition(
            worker_id="pettingzoo_worker",
            display_name="PettingZoo Worker",
            description=(
                "Multi-agent reinforcement learning using PettingZoo environments. "
                "Supports both AEC (turn-based) and Parallel (simultaneous) APIs. "
                "Includes classic board games (Chess, Go), cooperative environments (MPE, SISL), "
                "and competitive scenarios. Human control available for turn-based games."
            ),
            supports_training=True,
            supports_policy_load=True,
            requires_live_telemetry=True,
            provides_fast_analytics=True,
        ),
    )


__all__ = ["WorkerDefinition", "get_worker_catalog"]
