"""Algorithm registry for XuanCe worker.

This module provides mappings between:
- Deep learning backends (torch, tensorflow, mindspore)
- Algorithm paradigms (single-agent, multi-agent)
- Available algorithms

Used by the UI to dynamically filter algorithm options based on
user selections.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Mapping


class Backend(str, Enum):
    """Supported deep learning backends."""

    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    MINDSPORE = "mindspore"


class Paradigm(str, Enum):
    """Algorithm paradigms."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


@dataclass(frozen=True)
class AlgorithmInfo:
    """Metadata for an algorithm."""

    key: str  # Registry key (e.g., "PPO_Clip")
    display_name: str  # Human-readable name
    category: str  # Category (e.g., "Policy Gradient", "Value-based")
    paradigm: Paradigm
    description: str = ""


# ============================================================================
# Single-Agent Algorithms
# ============================================================================

# Core single-agent algorithms (available in ALL backends)
_CORE_SINGLE_AGENT: tuple[AlgorithmInfo, ...] = (
    # Policy Gradient
    AlgorithmInfo("PG", "PG", "Policy Gradient", Paradigm.SINGLE_AGENT, "REINFORCE policy gradient"),
    AlgorithmInfo("A2C", "A2C", "Actor-Critic", Paradigm.SINGLE_AGENT, "Advantage Actor-Critic"),
    AlgorithmInfo("PPO_Clip", "PPO (Clip)", "Policy Optimization", Paradigm.SINGLE_AGENT, "Proximal Policy Optimization with clipping"),
    AlgorithmInfo("PPO_KL", "PPO (KL)", "Policy Optimization", Paradigm.SINGLE_AGENT, "PPO with KL divergence penalty"),
    AlgorithmInfo("PPG", "PPG", "Policy Optimization", Paradigm.SINGLE_AGENT, "Phasic Policy Gradient"),
    # Continuous Control
    AlgorithmInfo("DDPG", "DDPG", "Continuous Control", Paradigm.SINGLE_AGENT, "Deep Deterministic Policy Gradient"),
    AlgorithmInfo("TD3", "TD3", "Continuous Control", Paradigm.SINGLE_AGENT, "Twin Delayed DDPG"),
    AlgorithmInfo("SAC", "SAC", "Continuous Control", Paradigm.SINGLE_AGENT, "Soft Actor-Critic"),
    # Parameterized Action
    AlgorithmInfo("PDQN", "P-DQN", "Parameterized Action", Paradigm.SINGLE_AGENT, "Parameterized DQN"),
    AlgorithmInfo("MPDQN", "MP-DQN", "Parameterized Action", Paradigm.SINGLE_AGENT, "Multi-pass P-DQN"),
    AlgorithmInfo("SPDQN", "SP-DQN", "Parameterized Action", Paradigm.SINGLE_AGENT, "Split P-DQN"),
    # Value-based (Q-learning)
    AlgorithmInfo("DQN", "DQN", "Value-based", Paradigm.SINGLE_AGENT, "Deep Q-Network"),
    AlgorithmInfo("DDQN", "Double DQN", "Value-based", Paradigm.SINGLE_AGENT, "Double DQN"),
    AlgorithmInfo("Duel_DQN", "Dueling DQN", "Value-based", Paradigm.SINGLE_AGENT, "Dueling network architecture"),
    AlgorithmInfo("NoisyDQN", "Noisy DQN", "Value-based", Paradigm.SINGLE_AGENT, "DQN with noisy networks"),
    AlgorithmInfo("PerDQN", "PER DQN", "Value-based", Paradigm.SINGLE_AGENT, "Prioritized Experience Replay DQN"),
    AlgorithmInfo("C51DQN", "C51", "Distributional", Paradigm.SINGLE_AGENT, "Categorical DQN (51 atoms)"),
    AlgorithmInfo("QRDQN", "QR-DQN", "Distributional", Paradigm.SINGLE_AGENT, "Quantile Regression DQN"),
    AlgorithmInfo("DRQN", "DRQN", "Recurrent", Paradigm.SINGLE_AGENT, "Deep Recurrent Q-Network"),
)

# PyTorch-only single-agent algorithms
_TORCH_ONLY_SINGLE_AGENT: tuple[AlgorithmInfo, ...] = (
    AlgorithmInfo("NPG", "NPG", "Policy Gradient", Paradigm.SINGLE_AGENT, "Natural Policy Gradient"),
    # Model-based RL
    AlgorithmInfo("DreamerV2", "DreamerV2", "Model-based", Paradigm.SINGLE_AGENT, "World model with actor-critic"),
    AlgorithmInfo("DreamerV3", "DreamerV3", "Model-based", Paradigm.SINGLE_AGENT, "Improved DreamerV2"),
    # Contrastive RL
    AlgorithmInfo("CURL", "CURL", "Contrastive", Paradigm.SINGLE_AGENT, "Contrastive Unsupervised RL"),
    AlgorithmInfo("SPR", "SPR", "Contrastive", Paradigm.SINGLE_AGENT, "Self-Predictive Representations"),
    AlgorithmInfo("DrQ", "DrQ", "Contrastive", Paradigm.SINGLE_AGENT, "Data-regularized Q"),
    # Offline RL
    AlgorithmInfo("TD3BC", "TD3+BC", "Offline", Paradigm.SINGLE_AGENT, "TD3 with Behavior Cloning"),
)

# ============================================================================
# Multi-Agent Algorithms
# ============================================================================

# Core multi-agent algorithms (available in ALL backends)
_CORE_MULTI_AGENT: tuple[AlgorithmInfo, ...] = (
    # Independent Learning
    AlgorithmInfo("IQL", "IQL", "Independent", Paradigm.MULTI_AGENT, "Independent Q-Learning"),
    AlgorithmInfo("IAC", "IAC", "Independent", Paradigm.MULTI_AGENT, "Independent Actor-Critic"),
    AlgorithmInfo("IPPO", "IPPO", "Independent", Paradigm.MULTI_AGENT, "Independent PPO"),
    AlgorithmInfo("IDDPG", "IDDPG", "Independent", Paradigm.MULTI_AGENT, "Independent DDPG"),
    AlgorithmInfo("ISAC", "ISAC", "Independent", Paradigm.MULTI_AGENT, "Independent SAC"),
    # Value Decomposition
    AlgorithmInfo("VDN", "VDN", "Value Decomposition", Paradigm.MULTI_AGENT, "Value Decomposition Networks"),
    AlgorithmInfo("QMIX", "QMIX", "Value Decomposition", Paradigm.MULTI_AGENT, "Q-value mixing network"),
    AlgorithmInfo("CWQMIX", "CW-QMIX", "Value Decomposition", Paradigm.MULTI_AGENT, "Centrally-Weighted QMIX"),
    AlgorithmInfo("OWQMIX", "OW-QMIX", "Value Decomposition", Paradigm.MULTI_AGENT, "Optimistically-Weighted QMIX"),
    AlgorithmInfo("QTRAN_base", "QTRAN", "Value Decomposition", Paradigm.MULTI_AGENT, "Q-Transformation"),
    AlgorithmInfo("QTRAN_alt", "QTRAN (alt)", "Value Decomposition", Paradigm.MULTI_AGENT, "QTRAN alternative"),
    AlgorithmInfo("DCG", "DCG", "Value Decomposition", Paradigm.MULTI_AGENT, "Deep Coordination Graphs"),
    AlgorithmInfo("DCG_S", "DCG-S", "Value Decomposition", Paradigm.MULTI_AGENT, "DCG Sparse"),
    # Centralized Training
    AlgorithmInfo("VDAC", "VDAC", "Centralized", Paradigm.MULTI_AGENT, "Value-Decomposition Actor-Critic"),
    AlgorithmInfo("COMA", "COMA", "Centralized", Paradigm.MULTI_AGENT, "Counterfactual Multi-Agent"),
    AlgorithmInfo("MADDPG", "MADDPG", "Centralized", Paradigm.MULTI_AGENT, "Multi-Agent DDPG"),
    AlgorithmInfo("MAPPO", "MAPPO", "Centralized", Paradigm.MULTI_AGENT, "Multi-Agent PPO"),
    AlgorithmInfo("MASAC", "MASAC", "Centralized", Paradigm.MULTI_AGENT, "Multi-Agent SAC"),
    AlgorithmInfo("MATD3", "MATD3", "Centralized", Paradigm.MULTI_AGENT, "Multi-Agent TD3"),
    # Mean Field
    AlgorithmInfo("MFQ", "MFQ", "Mean Field", Paradigm.MULTI_AGENT, "Mean Field Q-learning"),
    AlgorithmInfo("MFAC", "MFAC", "Mean Field", Paradigm.MULTI_AGENT, "Mean Field Actor-Critic"),
)

# PyTorch-only multi-agent algorithms
_TORCH_ONLY_MULTI_AGENT: tuple[AlgorithmInfo, ...] = (
    # Communication
    AlgorithmInfo("CommNet", "CommNet", "Communication", Paradigm.MULTI_AGENT, "Communication Neural Net"),
    AlgorithmInfo("IC3Net", "IC3Net", "Communication", Paradigm.MULTI_AGENT, "Individualized Controlled Continuous Communication"),
    AlgorithmInfo("TarMAC", "TarMAC", "Communication", Paradigm.MULTI_AGENT, "Targeted Multi-Agent Communication"),
)


# ============================================================================
# Registry Construction
# ============================================================================

def _build_algorithm_set(
    include_core_sa: bool = True,
    include_core_ma: bool = True,
    include_torch_only_sa: bool = False,
    include_torch_only_ma: bool = False,
) -> FrozenSet[str]:
    """Build a set of algorithm keys based on inclusion flags."""
    algorithms: set[str] = set()

    if include_core_sa:
        algorithms.update(a.key for a in _CORE_SINGLE_AGENT)
    if include_core_ma:
        algorithms.update(a.key for a in _CORE_MULTI_AGENT)
    if include_torch_only_sa:
        algorithms.update(a.key for a in _TORCH_ONLY_SINGLE_AGENT)
    if include_torch_only_ma:
        algorithms.update(a.key for a in _TORCH_ONLY_MULTI_AGENT)

    return frozenset(algorithms)


# Backend → Available algorithms
BACKEND_ALGORITHMS: Mapping[Backend, FrozenSet[str]] = {
    Backend.TORCH: _build_algorithm_set(
        include_core_sa=True,
        include_core_ma=True,
        include_torch_only_sa=True,
        include_torch_only_ma=True,
    ),
    Backend.TENSORFLOW: _build_algorithm_set(
        include_core_sa=True,
        include_core_ma=True,
        include_torch_only_sa=False,
        include_torch_only_ma=False,
    ),
    Backend.MINDSPORE: _build_algorithm_set(
        include_core_sa=True,
        include_core_ma=True,
        include_torch_only_sa=False,
        include_torch_only_ma=False,
    ),
}


# All algorithm info by key
_ALL_ALGORITHMS: tuple[AlgorithmInfo, ...] = (
    *_CORE_SINGLE_AGENT,
    *_TORCH_ONLY_SINGLE_AGENT,
    *_CORE_MULTI_AGENT,
    *_TORCH_ONLY_MULTI_AGENT,
)

ALGORITHM_INFO: Mapping[str, AlgorithmInfo] = {a.key: a for a in _ALL_ALGORITHMS}


# ============================================================================
# Public API
# ============================================================================

def get_algorithms_for_backend(backend: str | Backend) -> FrozenSet[str]:
    """Get available algorithm keys for a backend.

    Args:
        backend: Backend name ("torch", "tensorflow", "mindspore") or Backend enum.

    Returns:
        Frozenset of algorithm keys available for this backend.
    """
    if isinstance(backend, str):
        backend = Backend(backend)
    return BACKEND_ALGORITHMS[backend]


def get_algorithms_for_paradigm(paradigm: str | Paradigm) -> FrozenSet[str]:
    """Get algorithm keys for a paradigm (single-agent or multi-agent).

    Args:
        paradigm: Paradigm name or Paradigm enum.

    Returns:
        Frozenset of algorithm keys for this paradigm.
    """
    if isinstance(paradigm, str):
        paradigm = Paradigm(paradigm)

    return frozenset(
        a.key for a in _ALL_ALGORITHMS if a.paradigm == paradigm
    )


def get_algorithms(
    backend: str | Backend,
    paradigm: str | Paradigm,
) -> FrozenSet[str]:
    """Get algorithms filtered by both backend and paradigm.

    Args:
        backend: Backend name or Backend enum.
        paradigm: Paradigm name or Paradigm enum.

    Returns:
        Frozenset of algorithm keys matching both filters.
    """
    backend_algos = get_algorithms_for_backend(backend)
    paradigm_algos = get_algorithms_for_paradigm(paradigm)
    return backend_algos & paradigm_algos


def get_algorithm_info(key: str) -> AlgorithmInfo | None:
    """Get algorithm metadata by key.

    Args:
        key: Algorithm registry key (e.g., "PPO_Clip").

    Returns:
        AlgorithmInfo if found, None otherwise.
    """
    return ALGORITHM_INFO.get(key)


def get_algorithm_choices(
    backend: str | Backend,
    paradigm: str | Paradigm,
) -> list[tuple[str, str]]:
    """Get algorithm choices as (key, display_name) tuples for UI dropdowns.

    Args:
        backend: Backend name or Backend enum.
        paradigm: Paradigm name or Paradigm enum.

    Returns:
        List of (key, display_name) tuples sorted by category then name.
    """
    algo_keys = get_algorithms(backend, paradigm)
    choices = []

    for key in algo_keys:
        info = ALGORITHM_INFO.get(key)
        if info:
            choices.append((key, info.display_name))

    # Sort by display name
    return sorted(choices, key=lambda x: x[1])


def get_algorithms_by_category(
    backend: str | Backend,
    paradigm: str | Paradigm,
) -> dict[str, list[AlgorithmInfo]]:
    """Get algorithms grouped by category.

    Args:
        backend: Backend name or Backend enum.
        paradigm: Paradigm name or Paradigm enum.

    Returns:
        Dictionary mapping category names to lists of AlgorithmInfo.
    """
    algo_keys = get_algorithms(backend, paradigm)
    categories: dict[str, list[AlgorithmInfo]] = {}

    for key in algo_keys:
        info = ALGORITHM_INFO.get(key)
        if info:
            if info.category not in categories:
                categories[info.category] = []
            categories[info.category].append(info)

    # Sort algorithms within each category
    for cat in categories:
        categories[cat].sort(key=lambda a: a.display_name)

    return categories


def is_algorithm_available(algorithm: str, backend: str | Backend) -> bool:
    """Check if an algorithm is available for a backend.

    Args:
        algorithm: Algorithm key.
        backend: Backend name or Backend enum.

    Returns:
        True if the algorithm is available for this backend.
    """
    return algorithm in get_algorithms_for_backend(backend)


# ============================================================================
# Summary Statistics
# ============================================================================

def get_backend_summary() -> dict[str, dict[str, int]]:
    """Get algorithm counts per backend and paradigm.

    Returns:
        Dictionary with backend names mapping to paradigm counts.
    """
    summary = {}
    for backend in Backend:
        summary[backend.value] = {
            "single_agent": len(get_algorithms(backend, Paradigm.SINGLE_AGENT)),
            "multi_agent": len(get_algorithms(backend, Paradigm.MULTI_AGENT)),
            "total": len(BACKEND_ALGORITHMS[backend]),
        }
    return summary


__all__ = [
    "Backend",
    "Paradigm",
    "AlgorithmInfo",
    "BACKEND_ALGORITHMS",
    "ALGORITHM_INFO",
    "get_algorithms_for_backend",
    "get_algorithms_for_paradigm",
    "get_algorithms",
    "get_algorithm_info",
    "get_algorithm_choices",
    "get_algorithms_by_category",
    "is_algorithm_available",
    "get_backend_summary",
]
