from __future__ import annotations

"""Utilities for describing Gymnasium spaces in a JSON-friendly format."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class SpaceDescriptor:
    """Serializable descriptor for a Gymnasium space."""

    type: str
    shape: Sequence[int] | None = None
    dtype: str | None = None
    bounds: Mapping[str, Any] | None = None
    values: Mapping[str, Any] | None = None
    children: Mapping[str, "SpaceDescriptor"] | Sequence["SpaceDescriptor"] | None = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""

        payload: Dict[str, Any] = {"type": self.type}
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.bounds is not None:
            payload["bounds"] = _json_safe(self.bounds)
        if self.values is not None:
            payload["values"] = _json_safe(self.values)
        if self.children is not None:
            if isinstance(self.children, Mapping):
                payload["children"] = {
                    key: child.as_dict() for key, child in self.children.items()
                }
            else:
                payload["children"] = [child.as_dict() for child in self.children]
        if self.metadata is not None:
            payload["metadata"] = _json_safe(self.metadata)
        return payload


def describe_space(space: gym.Space[Any]) -> Dict[str, Any]:
    """Describe a Gymnasium space in a structured way suitable for telemetry."""

    descriptor = _build_descriptor(space)
    return descriptor.as_dict()


def describe_spaces(
    observation_space: gym.Space[Any] | None,
    action_space: gym.Space[Any] | None,
) -> Dict[str, Any]:
    """Describe observation and action spaces for an environment."""

    payload: Dict[str, Any] = {}
    if observation_space is not None:
        payload["observation"] = describe_space(observation_space)
    if action_space is not None:
        payload["action"] = describe_space(action_space)
    return payload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_descriptor(space: gym.Space[Any]) -> SpaceDescriptor:
    if isinstance(space, gym.spaces.Box):
        return SpaceDescriptor(
            type="Box",
            shape=space.shape,
            dtype=str(space.dtype),
            bounds={
                "low": _array_like(space.low),
                "high": _array_like(space.high),
                "bounded_below": bool(np.all(np.isfinite(space.low))),
                "bounded_above": bool(np.all(np.isfinite(space.high))),
            },
        )
    if isinstance(space, gym.spaces.Discrete):
        return SpaceDescriptor(
            type="Discrete",
            values={"n": int(space.n)},
        )
    if isinstance(space, gym.spaces.MultiBinary):
        size = getattr(space, "shape", None) or getattr(space, "n", None)
        return SpaceDescriptor(
            type="MultiBinary",
            shape=_ensure_shape(size),
        )
    if isinstance(space, gym.spaces.MultiDiscrete):
        return SpaceDescriptor(
            type="MultiDiscrete",
            values={"nvec": _array_like(space.nvec)},
        )
    if isinstance(space, gym.spaces.Tuple):
        return SpaceDescriptor(
            type="Tuple",
            children=[_build_descriptor(child) for child in space.spaces],
        )
    if isinstance(space, gym.spaces.Dict):
        return SpaceDescriptor(
            type="Dict",
            children={key: _build_descriptor(child) for key, child in space.spaces.items()},
        )
    # Sequence space is optional in some Gymnasium versions
    sequence_space = getattr(gym.spaces, "Sequence", None)
    if sequence_space and isinstance(space, sequence_space):  # type: ignore[arg-type]
        metadata: Dict[str, Any] = {}
        maxlen = getattr(space, "maxlen", None)
        if maxlen is not None:
            metadata["max_length"] = int(maxlen)
        sequence_child = getattr(space, "space", None)
        children = [_build_descriptor(sequence_child)] if sequence_child else []
        return SpaceDescriptor(
            type="Sequence",
            children=children or None,
            metadata=metadata or None,
        )
    text_space = getattr(gym.spaces, "Text", None)
    if text_space and isinstance(space, text_space):  # type: ignore[arg-type]
        metadata = {
            "max_length": int(getattr(space, "max_length", 0)),
            "min_length": int(getattr(space, "min_length", 0)),
        }
        charset = getattr(space, "charset", None)
        if charset is not None:
            metadata["charset"] = list(charset)
        return SpaceDescriptor(
            type="Text",
            metadata=metadata,
        )
    graph_space = getattr(gym.spaces, "Graph", None)
    if graph_space and isinstance(space, graph_space):  # type: ignore[arg-type]
        node_space = getattr(space, "node_space", None)
        edge_space = getattr(space, "edge_space", None)
        metadata = {
            "num_nodes": getattr(space, "num_nodes", None),
            "num_edges": getattr(space, "num_edges", None),
            "node_space": _build_descriptor(node_space).as_dict() if node_space else None,
            "edge_space": _build_descriptor(edge_space).as_dict() if edge_space else None,
        }
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        return SpaceDescriptor(
            type="Graph",
            metadata=clean_metadata or None,
        )

    # Fallback descriptor for custom spaces
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    metadata = {}
    if hasattr(space, "n"):
        metadata["n"] = int(getattr(space, "n"))
    if hasattr(space, "nvec"):
        metadata["nvec"] = _array_like(getattr(space, "nvec"))

    return SpaceDescriptor(
        type=space.__class__.__name__,
        shape=tuple(int(dim) for dim in shape) if shape is not None else None,
        dtype=str(dtype) if dtype is not None else None,
        metadata=metadata or None,
    )


def _array_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and callable(value.tolist):  # type: ignore[attr-defined]
        try:
            return value.tolist()
        except TypeError:
            pass
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        return [_array_like(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _ensure_shape(value: Any) -> Sequence[int] | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(v) for v in value]
    if hasattr(value, "shape"):
        return list(getattr(value, "shape"))
    return None


__all__ = ["describe_space", "describe_spaces", "SpaceDescriptor"]
