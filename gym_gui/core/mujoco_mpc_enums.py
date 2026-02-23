"""Enumerations specific to MuJoCo MPC integration.

This module defines enums for MuJoCo MPC (MJPC) that are SEPARATE from the
RL-focused enums in enums.py. MuJoCo MPC is a real-time predictive controller,
not an RL training system, so it has its own domain model.

Do NOT mix these with ControlMode or other RL-focused enums.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """Minimal stand-in for :class:`enum.StrEnum` (Python 3.11+)."""

    def __new__(cls, value: str) -> "StrEnum":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj


class MuJoCoMPCPlannerType(StrEnum):
    """MuJoCo MPC planner algorithms.

    These correspond to the planners available in MJPC:
    - iLQG: Iterative Linear Quadratic Gaussian (derivative-based)
    - Gradient Descent: Simple gradient-based optimization
    - Predictive Sampling: Derivative-free sampling-based planner (recommended)
    - Cross Entropy: Cross-entropy method for optimization
    """
    ILQG = "ilqg"
    GRADIENT_DESCENT = "gradient_descent"
    PREDICTIVE_SAMPLING = "predictive_sampling"
    CROSS_ENTROPY = "cross_entropy"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            "ilqg": "iLQG",
            "gradient_descent": "Gradient Descent",
            "predictive_sampling": "Predictive Sampling",
            "cross_entropy": "Cross Entropy",
        }
        return names.get(self.value, self.value)


class MuJoCoMPCTaskId(StrEnum):
    """MJPC built-in task identifiers.

    These are the standard tasks that come with MuJoCo MPC.
    Task names must match exactly what MJPC expects (case-sensitive with spaces).
    """
    # Classic control
    CARTPOLE = "Cartpole"
    PARTICLE = "Particle"

    # Locomotion
    SWIMMER = "Swimmer"
    WALKER = "Walker"
    QUADRUPED = "Quadruped"

    # Humanoid tasks
    HUMANOID_TRACK = "Humanoid Track"
    HUMANOID_STAND = "Humanoid Stand"

    # Manipulation
    HAND = "Hand"
    PANDA = "Panda"
    SHADOW_HAND = "Shadow Hand"

    # Bimanual
    BIMANUAL = "Bimanual"

    @classmethod
    def from_string(cls, value: str) -> "MuJoCoMPCTaskId":
        """Get task ID from string, case-insensitive.

        Args:
            value: Task name string (e.g., "cartpole", "Humanoid Track")

        Returns:
            The matching MuJoCoMPCTaskId enum value

        Raises:
            ValueError: If no matching task is found
        """
        # First try exact match
        for task in cls:
            if task.value == value:
                return task

        # Then try case-insensitive match
        value_normalized = value.lower().replace(" ", "_").replace("-", "_")
        for task in cls:
            task_normalized = task.value.lower().replace(" ", "_")
            if task_normalized == value_normalized:
                return task

        raise ValueError(f"Unknown MuJoCo MPC task: {value}")

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return self.value


class MuJoCoMPCSessionState(StrEnum):
    """State of a MuJoCo MPC session."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


# Mapping of tasks to their category for UI grouping
MUJOCO_MPC_TASK_CATEGORIES: dict[MuJoCoMPCTaskId, str] = {
    MuJoCoMPCTaskId.CARTPOLE: "Classic Control",
    MuJoCoMPCTaskId.PARTICLE: "Classic Control",
    MuJoCoMPCTaskId.SWIMMER: "Locomotion",
    MuJoCoMPCTaskId.WALKER: "Locomotion",
    MuJoCoMPCTaskId.QUADRUPED: "Locomotion",
    MuJoCoMPCTaskId.HUMANOID_TRACK: "Humanoid",
    MuJoCoMPCTaskId.HUMANOID_STAND: "Humanoid",
    MuJoCoMPCTaskId.HAND: "Manipulation",
    MuJoCoMPCTaskId.PANDA: "Manipulation",
    MuJoCoMPCTaskId.SHADOW_HAND: "Manipulation",
    MuJoCoMPCTaskId.BIMANUAL: "Manipulation",
}


__all__ = [
    "MuJoCoMPCPlannerType",
    "MuJoCoMPCTaskId",
    "MuJoCoMPCSessionState",
    "MUJOCO_MPC_TASK_CATEGORIES",
]
