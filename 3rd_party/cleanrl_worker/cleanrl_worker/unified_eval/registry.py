"""Registry mapping algorithms to their ActionSelector adapters."""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import ActionSelector
from .adapters import (
    PPOSelector,
    DQNSelector,
    C51Selector,
    DDPGSelector,
    TD3Selector,
    SACSelector,
)


# Algorithm to Adapter class mapping
ADAPTER_REGISTRY: Dict[str, Type[ActionSelector]] = {
    # PPO Family - all use PPOSelector
    "ppo": PPOSelector,
    "ppo_continuous_action": PPOSelector,
    "ppo_atari": PPOSelector,
    "ppo_atari_lstm": PPOSelector,  # May need specialized LSTM selector
    "ppo_atari_envpool": PPOSelector,
    "ppo_atari_multigpu": PPOSelector,
    "ppo_procgen": PPOSelector,
    "ppo_pettingzoo_ma_atari": PPOSelector,
    "ppo_rnd_envpool": PPOSelector,
    "ppo_continuous_action_isaacgym": PPOSelector,
    "ppo_trxl": PPOSelector,
    "ppg_procgen": PPOSelector,
    "rpo_continuous_action": PPOSelector,

    # DQN Family - all use DQNSelector
    "dqn": DQNSelector,
    "dqn_atari": DQNSelector,
    "pqn": DQNSelector,
    "pqn_atari_envpool": DQNSelector,
    "pqn_atari_envpool_lstm": DQNSelector,
    "qdagger_dqn_atari_impalacnn": DQNSelector,
    "rainbow_atari": DQNSelector,

    # C51 Family - distributional RL
    "c51": C51Selector,
    "c51_atari": C51Selector,

    # DDPG
    "ddpg_continuous_action": DDPGSelector,

    # TD3
    "td3_continuous_action": TD3Selector,

    # SAC
    "sac_continuous_action": SACSelector,
    "sac_atari": SACSelector,
}


def get_adapter(algo: str) -> Optional[ActionSelector]:
    """Get an ActionSelector instance for the given algorithm.

    Args:
        algo: Algorithm name (e.g., "ppo", "dqn", "ddpg")

    Returns:
        ActionSelector instance or None if algorithm not registered
    """
    adapter_cls = ADAPTER_REGISTRY.get(algo)
    if adapter_cls is None:
        return None
    return adapter_cls()


def get_adapter_class(algo: str) -> Optional[Type[ActionSelector]]:
    """Get the ActionSelector class for the given algorithm.

    Args:
        algo: Algorithm name

    Returns:
        ActionSelector class or None if algorithm not registered
    """
    return ADAPTER_REGISTRY.get(algo)


def register_adapter(algo: str, adapter_cls: Type[ActionSelector]) -> None:
    """Register a custom adapter for an algorithm.

    Args:
        algo: Algorithm name
        adapter_cls: ActionSelector class to use
    """
    ADAPTER_REGISTRY[algo] = adapter_cls


def list_supported_algorithms() -> list[str]:
    """Return list of all supported algorithm names."""
    return sorted(ADAPTER_REGISTRY.keys())
