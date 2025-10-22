"""Core runtime components for the refactored SPADE-BDI + RL stack."""

from typing import TYPE_CHECKING

from .config import PolicyStrategy, RunConfig
from .runtime import HeadlessTrainer
from .bdi_trainer import BDITrainer
from .worker_telemetry import TelemetryEmitter

if TYPE_CHECKING:
    from .agent import (
        DEFAULT_JID,
        DEFAULT_PASSWORD,
        AgentHandle,
        create_agent,
        create_and_start_agent,
        docker_compose_path,
        resolve_asl,
    )

# Lazy imports to avoid pulling in SPADE/BDI dependencies for pure RL workers
__all__ = [
    "AgentHandle",
    "create_agent",
    "create_and_start_agent",
    "docker_compose_path",
    "resolve_asl",
    "DEFAULT_JID",
    "DEFAULT_PASSWORD",
    "RunConfig",
    "PolicyStrategy",
    "HeadlessTrainer",
    "BDITrainer",
    "TelemetryEmitter",
]


def __getattr__(name: str):
    """Lazy-load BDI agent components only when needed."""
    if name in (
        "AgentHandle",
        "create_agent",
        "create_and_start_agent",
        "docker_compose_path",
        "resolve_asl",
        "DEFAULT_JID",
        "DEFAULT_PASSWORD",
    ):
        from .agent import (
            DEFAULT_JID,
            DEFAULT_PASSWORD,
            AgentHandle,
            create_agent,
            create_and_start_agent,
            docker_compose_path,
            resolve_asl,
        )

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
