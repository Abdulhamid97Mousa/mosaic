"""MuJoCo MPC Worker - Integration wrapper for MOSAIC BDI-RL framework.

This module provides a wrapper around the vendored mujoco_mpc library,
enabling MuJoCo MPC visualization and control within the GUI_BDI_RL application.

The vendored mujoco_mpc code lives in ../mujoco_mpc/ and should NOT be modified.
All customizations and integrations belong in this mujoco_mpc_worker package.
"""

__version__ = "0.1.0"

from mujoco_mpc_worker.config import MuJoCoMPCConfig
from mujoco_mpc_worker.launcher import MJPCLauncher, MJPCProcess, get_launcher

__all__ = [
    "__version__",
    "MuJoCoMPCConfig",
    "MJPCLauncher",
    "MJPCProcess",
    "get_launcher",
]
