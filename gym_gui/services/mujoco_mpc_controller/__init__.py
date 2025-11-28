"""MuJoCo MPC Controller Service.

This service manages MuJoCo MPC sessions for real-time predictive control
visualization within the GUI_BDI_RL application.

Note: This is SEPARATE from the trainer service which handles RL training.
MuJoCo MPC is a real-time controller, not a learning system.
"""

from gym_gui.services.mujoco_mpc_controller.service import (
    MuJoCoMPCControllerService,
)

__all__ = [
    "MuJoCoMPCControllerService",
]
