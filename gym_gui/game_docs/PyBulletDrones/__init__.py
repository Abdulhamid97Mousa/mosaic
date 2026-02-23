"""PyBullet Drones game documentation module.

gym-pybullet-drones is a PyBullet-based Gymnasium environment for single and
multi-agent reinforcement learning of quadcopter control. It provides realistic
physics simulation including aerodynamic effects (drag, ground effect, downwash).

Environments:
    - hover-aviary-v0: Single-agent hover task
    - multihover-aviary-v0: Multi-agent hover task
    - ctrl-aviary-v0: Low-level RPM control
    - velocity-aviary-v0: High-level velocity control

Reference:
    Panerati, J., et al. (2021). Learning to Fly - a Gym Environment with PyBullet
    Physics for Reinforcement Learning of Multi-agent Quadcopter Control.
    https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations

from .HoverAviary import (
    HOVER_AVIARY_HTML,
    get_hover_aviary_html,
)
from .MultiHoverAviary import (
    MULTIHOVER_AVIARY_HTML,
    get_multihover_aviary_html,
)

__all__ = [
    "HOVER_AVIARY_HTML",
    "MULTIHOVER_AVIARY_HTML",
    "get_hover_aviary_html",
    "get_multihover_aviary_html",
]
