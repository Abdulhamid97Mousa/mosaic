"""Overcooked-AI game documentation module.

Overcooked-AI is a benchmark environment for fully cooperative human-AI task
performance, based on the popular video game Overcooked. Developed by UC Berkeley
CHAI (Center for Human-Compatible AI) for studying human-AI coordination and
zero-shot cooperation.

Research Paper:
    Carroll, M., Shah, R., Ho, M. K., Griffiths, T. L., Seshia, S. A., Abbeel, P.,
    & Dragan, A. (2019). "On the Utility of Learning about Humans for Human-AI
    Coordination". NeurIPS 2019.
    https://arxiv.org/abs/1910.05789

Repository:
    https://github.com/HumanCompatibleAI/overcooked_ai
"""
from __future__ import annotations

from .CrampedRoom import OVERCOOKED_CRAMPED_ROOM_HTML, get_cramped_room_html
from .AsymmetricAdvantages import (
    OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML,
    get_asymmetric_advantages_html,
)
from .CoordinationRing import (
    OVERCOOKED_COORDINATION_RING_HTML,
    get_coordination_ring_html,
)
from .ForcedCoordination import (
    OVERCOOKED_FORCED_COORDINATION_HTML,
    get_forced_coordination_html,
)
from .CounterCircuit import OVERCOOKED_COUNTER_CIRCUIT_HTML, get_counter_circuit_html

__all__ = [
    "OVERCOOKED_CRAMPED_ROOM_HTML",
    "get_cramped_room_html",
    "OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML",
    "get_asymmetric_advantages_html",
    "OVERCOOKED_COORDINATION_RING_HTML",
    "get_coordination_ring_html",
    "OVERCOOKED_FORCED_COORDINATION_HTML",
    "get_forced_coordination_html",
    "OVERCOOKED_COUNTER_CIRCUIT_HTML",
    "get_counter_circuit_html",
]
