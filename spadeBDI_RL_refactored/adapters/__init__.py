"""Adapter factory used by the headless worker."""

from __future__ import annotations

from typing import Any, Dict, Type

from .frozenlake import FrozenLakeAdapter

_ADAPTERS: Dict[str, Type[FrozenLakeAdapter]] = {
    "FrozenLake-v2": FrozenLakeAdapter,
    "FrozenLake-v2-headless": FrozenLakeAdapter,
    "frozenlake": FrozenLakeAdapter,
}


def create_adapter(env_id: str, **kwargs: Any) -> FrozenLakeAdapter:
    try:
        adapter_cls = _ADAPTERS[env_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported env_id '{env_id}'. Available adapters: {sorted(_ADAPTERS.keys())}"
        ) from exc
    return adapter_cls(**kwargs)


__all__ = ["create_adapter", "FrozenLakeAdapter"]
