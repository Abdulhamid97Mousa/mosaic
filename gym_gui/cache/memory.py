from __future__ import annotations

"""In-memory cache abstractions used across the GUI."""

from functools import lru_cache, _lru_cache_wrapper
from typing import Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def memoize(
    maxsize: int = 128,
) -> Callable[[Callable[P, T]], _lru_cache_wrapper[T]]:
    """Decorator wrapper around :func:`functools.lru_cache`.

    Returns:
        A decorator that wraps functions with lru_cache, preserving
        cache_clear(), cache_info(), and other lru_cache methods.
    """

    def decorator(func: Callable[P, T]) -> _lru_cache_wrapper[T]:
        return lru_cache(maxsize=maxsize)(func)

    return decorator


__all__ = ["memoize"]
