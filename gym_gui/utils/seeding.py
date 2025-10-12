"""Helpers for consistent session-wide random seeding."""


from collections.abc import Callable
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gym_gui.utils import json_serialization

try:  # pragma: no cover - optional Qt dependency during testing
    from qtpy import QtCore
except Exception:  # noqa: BLE001 - fallback when Qt bindings unavailable
    QtCore = None  # type: ignore[assignment]


logger = logging.getLogger("gym_gui.utils.seeding")


def _seed_qt_random(seed: int) -> None:
    del seed
    if QtCore is None:  # Qt bindings not available (e.g., during headless tests)
        return
    # Qt aborts if the global random generator is reseeded after initialization.
    # Instead of forcing determinism here (which would require invasive hooks),
    # we simply log the requested seed so other subsystems remain deterministic.
    logger.debug("Skipping Qt random reseed; relying on Python/NumPy determinism")


def _capture_qt_state() -> dict[str, Any] | None:
    if QtCore is None:
        return None
    try:
        generator = QtCore.QRandomGenerator.global_()  # type: ignore[attr-defined]
        state: dict[str, Any] = {"type": generator.__class__.__name__}
        if hasattr(generator, "state"):
            qt_state = generator.state()  # type: ignore[attr-defined]
            state["state"] = int(qt_state)
        return state
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Failed to capture Qt random generator state")
        return None


def _serialize_python_state(state: Any) -> dict[str, Any]:
    version, internal_state, gauss = state
    return {
        "version": int(version),
        "state": list(internal_state),
        "gauss": gauss,
    }


def _serialize_numpy_state(state: Any) -> dict[str, Any]:
    bit_generator, keys, pos, has_gauss, cached_gauss = state
    return {
        "bit_generator": bit_generator,
        "keys": np.asarray(keys).tolist(),
        "pos": int(pos),
        "has_gauss": bool(has_gauss),
        "cached_gaussian": float(cached_gauss),
    }


def _encode_json(obj: Any) -> str:
    return json_serialization.dumps(obj).decode("utf-8")


def _digest(obj: Any) -> str:
    return hashlib.sha1(json_serialization.dumps(obj)).hexdigest()


Callback = Callable[[int], None]


@dataclass(slots=True)
class SessionSeedManager:
    """Orchestrates deterministic seeding across libraries and consumers."""

    _last_applied_seed: int | None = None
    _callbacks: dict[str, Callback] = field(default_factory=dict)

    def register_consumer(self, name: str, callback: Callback) -> None:
        """Register a callable invoked whenever a seed is applied."""

        self._callbacks[name] = callback

    def apply(self, seed: int) -> None:
        """Seed Python, NumPy, Qt, and registered consumers with ``seed``."""

        random.seed(seed)
        np.random.seed(seed)
        _seed_qt_random(seed)
        for name, callback in self._callbacks.items():
            try:
                callback(seed)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Seed callback '%s' failed", name)
        self._last_applied_seed = seed

    def capture_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of RNG state across subsystems."""

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        serialized_python = _serialize_python_state(python_state)
        serialized_numpy = _serialize_numpy_state(numpy_state)
        snapshot: dict[str, Any] = {
            "seed": self._last_applied_seed,
            "python_random": serialized_python,
            "numpy_random": serialized_numpy,
            "python_random_json": _encode_json(serialized_python),
            "numpy_random_json": _encode_json(serialized_numpy),
            "python_random_digest": _digest(serialized_python),
            "numpy_random_digest": _digest(serialized_numpy),
        }
        qt_state = _capture_qt_state()
        if qt_state is not None:
            snapshot["qt_random"] = qt_state
        return snapshot

    @property
    def last_seed(self) -> int | None:
        return self._last_applied_seed


__all__ = ["SessionSeedManager"]
