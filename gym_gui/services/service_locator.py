from __future__ import annotations

"""Simple service locator used to wire core subsystems together."""

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Optional, Type, TypeVar

ServiceT = TypeVar("ServiceT")


@dataclass(slots=True)
class ServiceRegistration:
    """Record describing a registered service instance."""

    instance: Any
    eager: bool = True


class ServiceLocator:
    """In-process registry for shared services.

    This provides a lightweight alternative to a full dependency injection
    framework while keeping the wiring explicit. Services can be registered at
    application startup and looked up on demand by controllers, presenters, or
    CLI entrypoints.
    """

    def __init__(self) -> None:
        self._services: Dict[Type[Any] | str, ServiceRegistration] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, key: Type[ServiceT] | str, instance: ServiceT) -> None:
        """Register ``instance`` under ``key``.

        Args:
            key: Either the service class or an arbitrary string identifier.
            instance: Concrete service implementation.
        """

        with self._lock:
            self._services[key] = ServiceRegistration(instance=instance)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def resolve(self, key: Type[ServiceT] | str) -> Optional[ServiceT]:
        """Return the service previously registered for ``key``.

        Returns ``None`` if the key has not been registered."""

        with self._lock:
            registration = self._services.get(key)
        if registration is None:
            return None
        return registration.instance  # type: ignore[return-value]

    def require(self, key: Type[ServiceT] | str) -> ServiceT:
        """Resolve a service and raise ``KeyError`` if it is missing."""

        instance = self.resolve(key)
        if instance is None:
            raise KeyError(f"Service '{key}' has not been registered")
        return instance

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove all registered services (primarily for tests)."""

        with self._lock:
            self._services.clear()


_GLOBAL_LOCATOR: ServiceLocator | None = None


def get_service_locator() -> ServiceLocator:
    """Return the process-wide service locator instance."""

    global _GLOBAL_LOCATOR
    if _GLOBAL_LOCATOR is None:
        _GLOBAL_LOCATOR = ServiceLocator()
    return _GLOBAL_LOCATOR


__all__ = ["ServiceLocator", "get_service_locator"]
