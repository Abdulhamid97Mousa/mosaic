"""Adapter factory used by the headless worker."""

from __future__ import annotations

from typing import Any, Dict, Type, Union

from .frozenlake import FrozenLakeAdapter, FrozenLakeV2Adapter
from .cliffwalking import CliffWalkingAdapter
from .taxi import TaxiAdapter

# Type alias for all adapter types
AdapterType = Union[FrozenLakeAdapter, FrozenLakeV2Adapter, CliffWalkingAdapter, TaxiAdapter]

_ADAPTERS: Dict[str, Type[AdapterType]] = {
    # FrozenLake-v1 (original)
    "FrozenLake-v1": FrozenLakeAdapter,
    "frozenlake": FrozenLakeAdapter,
    
    # FrozenLake-v2 (configurable)
    "FrozenLake-v2": FrozenLakeV2Adapter,
    "FrozenLake-v2-headless": FrozenLakeV2Adapter,
    "frozenlake-v2": FrozenLakeV2Adapter,
    
    # CliffWalking-v0
    "CliffWalking-v0": CliffWalkingAdapter,
    "CliffWalking-v1": CliffWalkingAdapter,  # Alias
    "cliffwalking": CliffWalkingAdapter,
    
    # Taxi-v3
    "Taxi-v3": TaxiAdapter,
    "taxi": TaxiAdapter,
}


def create_adapter(env_id: str, **kwargs: Any) -> AdapterType:
    """Create an environment adapter based on environment ID.
    
    Args:
        env_id: Environment identifier (e.g., 'FrozenLake-v2', 'CliffWalking-v0', 'Taxi-v3')
        **kwargs: Additional keyword arguments passed to the adapter constructor
        
    Returns:
        Configured adapter instance
        
    Raises:
        ValueError: If env_id is not supported
    """
    try:
        adapter_cls = _ADAPTERS[env_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported env_id '{env_id}'. Available adapters: {sorted(_ADAPTERS.keys())}"
        ) from exc
    return adapter_cls(**kwargs)  # type: ignore[return-value]


__all__ = [
    "create_adapter",
    "FrozenLakeAdapter",
    "FrozenLakeV2Adapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "AdapterType",
]
