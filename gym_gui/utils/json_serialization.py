from __future__ import annotations

"""Safe JSON-based serialization helpers for telemetry payloads."""

from collections.abc import Mapping, Sequence, Set
import base64
import json
import math
from datetime import datetime
from typing import Any

import numpy as np
import logging
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED


class SerializationError(RuntimeError):
    """Raised when an object cannot be safely serialized or deserialized."""


class _JsonSerializer(LogConstantMixin):
    """Internal serializer with structured logging via LogConstantMixin."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._log_once_flags = {"numpy_scalar": False}

    def _to_json_compatible(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                # Represent non-finite floats using a structured marker
                return {"__type__": "float", "value": repr(obj)}
            return obj
        # Normalize NumPy scalar types to built-in Python primitives
        # This covers np.integer, np.floating, np.bool_, and other np.generic subclasses.
        # Using .item() preserves NaN/Inf behaviour which is handled by the float branch above on recursion.
        try:  # fast path without importing numpy types explicitly
            import numpy as _np  # local import to avoid global hard dependency during type checks
            if isinstance(obj, _np.generic):  # type: ignore[attr-defined]
                return self._to_json_compatible(obj.item())
        except Exception as exc:  # pragma: no cover - defensive: numpy unavailable or other edge
            if not self._log_once_flags["numpy_scalar"]:
                self.log_constant(
                    LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED,
                    extra={"obj_type": getattr(type(obj), "__name__", str(type(obj)))},
                    exc_info=exc,
                )
                self._log_once_flags["numpy_scalar"] = True
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}
        if isinstance(obj, bytes):
            return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}
        if isinstance(obj, np.ndarray):
            array = np.asarray(obj)
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            encoded = base64.b64encode(array.tobytes()).decode("ascii")
            return {
                "__type__": "ndarray",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "encoding": "base64",
                "data": encoded,
            }
        if isinstance(obj, Mapping):
            return {str(key): self._to_json_compatible(value) for key, value in obj.items()}
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "data": [self._to_json_compatible(item) for item in obj]}
        if isinstance(obj, Set):
            return {"__type__": "set", "data": [self._to_json_compatible(item) for item in obj]}
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [self._to_json_compatible(item) for item in obj]

        raise TypeError(f"Type {type(obj)!r} is not supported for safe serialization")

    def _from_json_compatible(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, str)):
            return obj
        if isinstance(obj, float):
            return obj
        if isinstance(obj, list):
            return [self._from_json_compatible(item) for item in obj]
        if isinstance(obj, Mapping):
            marker = obj.get("__type__")
            if marker == "datetime":
                return datetime.fromisoformat(obj["value"])
            if marker == "bytes":
                return base64.b64decode(obj["data"])
            if marker == "ndarray":
                dtype = np.dtype(obj["dtype"])
                shape = tuple(obj["shape"])
                encoding = obj.get("encoding")
                if encoding == "base64":
                    raw = base64.b64decode(obj["data"])
                    array = np.frombuffer(raw, dtype=dtype)
                    return array.reshape(shape)
                data = obj.get("data")
                array = np.array(data, dtype=dtype)
                return array.reshape(shape)
            if marker == "tuple":
                return tuple(self._from_json_compatible(item) for item in obj["data"])
            if marker == "set":
                return {self._from_json_compatible(item) for item in obj["data"]}
            if marker == "float":
                value = obj.get("value", "nan")
                return float(value)
            return {key: self._from_json_compatible(value) for key, value in obj.items()}

        raise SerializationError(f"Unsupported JSON structure encountered: {obj!r}")


def dumps(value: Any) -> bytes:
    """Serialize a Python object to UTF-8 encoded JSON bytes.

    Only a restricted set of types is supported to reduce the attack surface:
    primitives, mappings with string keys, lists/tuples, sets, bytes, datetime,
    and numpy arrays. Objects outside this set will raise ``SerializationError``.
    """

    s = _JsonSerializer()
    try:
        prepared = s._to_json_compatible(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise SerializationError(f"Unsupported value for serialization: {value!r}") from exc
    try:
        return json.dumps(prepared, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise SerializationError("Failed to encode payload as JSON") from exc


def loads(payload: bytes | bytearray | str | memoryview) -> Any:
    """Deserialize JSON payload produced by :func:`dumps`."""

    text: str
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - legacy data
            raise SerializationError("Payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, str):  # pragma: no cover - defensive
        raise SerializationError("JSON payload must be text or bytes")
    text = payload

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SerializationError("Invalid JSON payload") from exc
    s = _JsonSerializer()
    return s._from_json_compatible(decoded)



__all__ = ["dumps", "loads", "SerializationError"]
