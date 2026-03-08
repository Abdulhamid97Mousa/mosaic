"""Configuration dataclasses for MuJoCo MPC Worker.

This module defines configuration structures for the MuJoCo MPC integration,
including planner types, task identifiers, and runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MuJoCoMPCPlannerType(str, Enum):
    """MuJoCo MPC planner algorithms.

    These correspond to the planners available in MJPC:
    - iLQG: Iterative Linear Quadratic Gaussian (derivative-based)
    - Gradient Descent: Simple gradient-based optimization
    - Predictive Sampling: Derivative-free sampling-based planner
    - Cross Entropy: Cross-entropy method for optimization
    """
    ILQG = "ilqg"
    GRADIENT_DESCENT = "gradient_descent"
    PREDICTIVE_SAMPLING = "predictive_sampling"
    CROSS_ENTROPY = "cross_entropy"


class MuJoCoMPCTaskId(str, Enum):
    """MJPC built-in task identifiers.

    These are the standard tasks that come with MuJoCo MPC.
    Task names must match exactly what MJPC expects.
    """
    CARTPOLE = "Cartpole"
    PARTICLE = "Particle"
    SWIMMER = "Swimmer"
    QUADRUPED = "Quadruped"
    HUMANOID_TRACK = "Humanoid Track"
    HUMANOID_STAND = "Humanoid Stand"
    WALKER = "Walker"
    HAND = "Hand"
    PANDA = "Panda"
    SHADOW_HAND = "Shadow Hand"

    @classmethod
    def from_string(cls, value: str) -> "MuJoCoMPCTaskId":
        """Get task ID from string, case-insensitive."""
        value_lower = value.lower().replace(" ", "_").replace("-", "_")
        for task in cls:
            if task.value.lower().replace(" ", "_") == value_lower:
                return task
        raise ValueError(f"Unknown task: {value}")


@dataclass
class MuJoCoMPCConfig:
    """Configuration for MuJoCo MPC Worker.

    Attributes:
        task_id: The MJPC task identifier (e.g., "Cartpole", "Humanoid Track")
        planner_type: The planner algorithm to use
        port: gRPC port for MJPC server (None = auto-assign)
        real_time_speed: Ratio of simulation speed to wall clock (0.0 to 1.0)
        server_binary_path: Path to agent_server binary (None = auto-detect)
        model_xml_path: Optional custom MuJoCo model XML path
    """
    task_id: str = "Cartpole"
    planner_type: MuJoCoMPCPlannerType = MuJoCoMPCPlannerType.PREDICTIVE_SAMPLING
    port: Optional[int] = None
    real_time_speed: float = 1.0
    server_binary_path: Optional[str] = None
    model_xml_path: Optional[str] = None

    # Cost weights (task-specific, applied after init)
    cost_weights: dict[str, float] = field(default_factory=dict)

    # Task parameters (task-specific)
    task_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.real_time_speed < 0.0 or self.real_time_speed > 1.0:
            raise ValueError(
                f"real_time_speed must be between 0.0 and 1.0, got {self.real_time_speed}"
            )
        if self.port is not None and (self.port < 0 or self.port > 65535):
            raise ValueError(
                f"port must be between 0 and 65535, got {self.port}"
            )


@dataclass
class MuJoCoMPCState:
    """Runtime state of a MuJoCo MPC session.

    Attributes:
        is_running: Whether the MJPC agent is currently running
        current_task: The currently loaded task ID
        current_planner: The currently active planner
        server_port: The gRPC port the server is listening on
        total_cost: Current total cost from the planner
        step_count: Number of simulation steps executed
    """
    is_running: bool = False
    current_task: Optional[str] = None
    current_planner: Optional[MuJoCoMPCPlannerType] = None
    server_port: Optional[int] = None
    total_cost: float = 0.0
    step_count: int = 0
