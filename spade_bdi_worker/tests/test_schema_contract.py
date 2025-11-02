from __future__ import annotations

"""Tests for the telemetry schema scaffolding and vector constants."""

from typing import Any, Dict

import pytest

from gym_gui.constants import (
    DEFAULT_AUTORESET_MODE,
    RESET_MASK_KEY,
    SUPPORTED_AUTORESET_MODES,
    VECTOR_ENV_BATCH_SIZE_KEY,
    VECTOR_ENV_INDEX_KEY,
    VECTOR_SEED_KEY,
)
from gym_gui.core.schema import schema_registry
from gym_gui.validations.validations_telemetry import ValidationService


@pytest.fixture(scope="module")
def default_schema() -> Any:
    schema = schema_registry.get("default")
    assert schema is not None, "Default telemetry schema must be registered"
    return schema


def test_default_schema_includes_required_fields(default_schema: Any) -> None:
    required = set(default_schema.required_fields)
    assert {"observation", "time_step", "space_signature"}.issubset(required)

    json_schema: Dict[str, Any] = default_schema.as_json_schema()
    properties = json_schema.get("properties", {})
    for field in ("observation", "time_step", "space_signature"):
        assert field in properties, f"Schema properties missing {field}"


def test_default_schema_vector_fragment(default_schema: Any) -> None:
    vector_fragment = default_schema.vector_metadata
    assert vector_fragment is not None, "Vector metadata schema must be defined"
    assert "vectorized" in vector_fragment.required_keys

    json_fragment = vector_fragment.as_json_schema()
    props = json_fragment.get("properties", {})
    for key in (
        VECTOR_ENV_BATCH_SIZE_KEY,
        VECTOR_ENV_INDEX_KEY,
        RESET_MASK_KEY,
        VECTOR_SEED_KEY,
    ):
        assert key in props, f"Vector metadata schema missing property: {key}"


def test_schema_registry_alias_points_to_default(default_schema: Any) -> None:
    assert schema_registry.get("gymnasium") is default_schema
    assert schema_registry.get("classic-control") is default_schema


def test_validation_service_exports_schema(default_schema: Any) -> None:
    validator = ValidationService(strict_mode=True)
    exported = validator.get_step_schema("default")
    assert exported is not None
    assert exported.get("$id") == default_schema.schema_id


def test_vector_autoreset_defaults_are_consistent() -> None:
    assert DEFAULT_AUTORESET_MODE in SUPPORTED_AUTORESET_MODES
    assert sorted(SUPPORTED_AUTORESET_MODES) == sorted(set(SUPPORTED_AUTORESET_MODES))


def test_minigrid_schema_requires_render_payload(default_schema: Any) -> None:
    minigrid = schema_registry.get("minigrid")
    assert minigrid is not None
    assert "render_payload" in minigrid.required_fields
    assert minigrid.vector_metadata is default_schema.vector_metadata


def test_atari_schema_requires_frame_ref(default_schema: Any) -> None:
    atari = schema_registry.get("atari")
    assert atari is not None
    assert "frame_ref" in atari.required_fields
    assert "render_payload" in atari.required_fields
    # Ensure alias lookup works for ALE naming.
    assert schema_registry.get("ale") is atari
