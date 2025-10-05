from __future__ import annotations

"""In-memory cache abstractions used across the GUI."""

from functools import lru_cache
from typing import Callable, TypeVar

T = TypeVar("T")


def memoize(maxsize: int = 128) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator wrapper around :func:`functools.lru_cache`."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        return lru_cache(maxsize=maxsize)(func)

    return decorator


__all__ = ["memoize"]
