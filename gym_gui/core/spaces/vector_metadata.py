from __future__ import annotations

"""Helpers for extracting vector-environment metadata for telemetry."""

from collections.abc import Mapping
from typing import Any

try:  # pragma: no cover - optional dependency guard
    from gymnasium.vector.vector_env import AutoresetMode, VectorEnv
except ImportError:  # pragma: no cover - legacy gymnasium
    AutoresetMode = None  # type: ignore[assignment]
    VectorEnv = None  # type: ignore[assignment]

from gym_gui.constants.constants_vector import (
    DEFAULT_AUTORESET_MODE,
    RESET_MASK_KEY,
    VECTOR_ENV_BATCH_SIZE_KEY,
    VECTOR_ENV_INDEX_KEY,
    VECTOR_SEED_KEY,
)
from gym_gui.core.spaces.serializer import describe_space


def describe_vector_environment(env: Any) -> dict[str, Any] | None:
    """Return static vector-environment metadata if the env supports it."""

    base_env = _unwrap(env)
    if VectorEnv is None or not isinstance(base_env, VectorEnv):
        return None

    metadata: dict[str, Any] = {"vectorized": True}

    num_envs = getattr(base_env, "num_envs", None)
    if num_envs is not None:
        metadata[VECTOR_ENV_BATCH_SIZE_KEY] = int(num_envs)

    vector_metadata = getattr(base_env, "metadata", {}) or {}
    autoreset_mode = DEFAULT_AUTORESET_MODE
    autoreset = vector_metadata.get("autoreset_mode")
    if AutoresetMode is not None and isinstance(autoreset, AutoresetMode):
        autoreset_mode = autoreset.value
    elif isinstance(autoreset, str):
        autoreset_mode = autoreset
    if autoreset_mode:
        metadata["autoreset_mode"] = autoreset_mode

    render_modes = vector_metadata.get("render_modes")
    if render_modes:
        metadata["render_modes"] = list(render_modes)

    fps = vector_metadata.get("render_fps")
    if fps is not None:
        try:
            metadata["render_fps"] = float(fps)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass

    # Capture both batched and single-space descriptors for downstream tooling.
    if hasattr(base_env, "observation_space"):
        metadata["observation_space"] = describe_space(base_env.observation_space)
    if hasattr(base_env, "single_observation_space"):
        metadata["single_observation_space"] = describe_space(base_env.single_observation_space)
    if hasattr(base_env, "action_space"):
        metadata["action_space"] = describe_space(base_env.action_space)
    if hasattr(base_env, "single_action_space"):
        metadata["single_action_space"] = describe_space(base_env.single_action_space)

    return metadata


def extract_vector_step_details(info: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract per-step vector metadata (indices, masks, seeds) from info payloads."""

    if not info:
        return None

    candidates: list[Mapping[str, Any]] = [info]
    nested = info.get("vector") if isinstance(info, Mapping) else None
    if isinstance(nested, Mapping):
        candidates.append(nested)

    payload: dict[str, Any] = {}
    for candidate in candidates:
        if VECTOR_ENV_INDEX_KEY in candidate and VECTOR_ENV_INDEX_KEY not in payload:
            payload[VECTOR_ENV_INDEX_KEY] = candidate[VECTOR_ENV_INDEX_KEY]
        if RESET_MASK_KEY in candidate and RESET_MASK_KEY not in payload:
            payload[RESET_MASK_KEY] = candidate[RESET_MASK_KEY]
        if VECTOR_SEED_KEY in candidate and VECTOR_SEED_KEY not in payload:
            payload[VECTOR_SEED_KEY] = candidate[VECTOR_SEED_KEY]
        autoreset = candidate.get("autoreset_mode")
        if autoreset and "autoreset_mode" not in payload:
            payload["autoreset_mode"] = autoreset

    return payload or None


def _unwrap(env: Any) -> Any:
    """Return the deepest underlying environment, following .unwrapped when available."""

    current = env
    visited: set[int] = set()
    while hasattr(current, "unwrapped"):
        next_env = getattr(current, "unwrapped")
        if next_env is current or id(next_env) in visited:
            break
        if next_env is None:
            break
        visited.add(id(current))
        current = next_env
    return current


__all__ = [
    "describe_vector_environment",
    "extract_vector_step_details",
]
