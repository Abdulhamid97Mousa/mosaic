"""Telemetry payload schema definitions.

This module provides lightweight dataclass wrappers that describe the shape of
`StepRecord` telemetry payloads in a JSON-schema compatible form.  The schema
objects are intentionally declarative so that both the GUI and remote trainers
can serialise or validate payloads without depending on concrete Python
classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from gym_gui.constants.constants_telemetry import (
    RENDER_PAYLOAD_GRAPH,
    RENDER_PAYLOAD_GRID,
    RENDER_PAYLOAD_RGB,
    TELEMETRY_KEY_AUTORESET_MODE,
    TELEMETRY_KEY_SPACE_SIGNATURE,
    TELEMETRY_KEY_TIME_STEP,
    TELEMETRY_KEY_VECTOR_METADATA,
)
from gym_gui.constants.constants_vector import (
    RESET_MASK_KEY,
    VECTOR_ENV_BATCH_SIZE_KEY,
    VECTOR_ENV_INDEX_KEY,
    VECTOR_SEED_KEY,
)

# ---------------------------------------------------------------------------
# Schema primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderPayloadSchema:
    """Describe a render payload variant shipped alongside telemetry."""

    variant: str
    description: str
    required_fields: Sequence[str] = field(default_factory=tuple)
    optional_fields: Sequence[str] = field(default_factory=tuple)

    def as_json_schema(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        for field_name in self.required_fields:
            properties[field_name] = {"type": ["number", "integer", "string", "array", "object", "boolean"]}
        for field_name in self.optional_fields:
            if field_name not in properties:
                properties[field_name] = {"type": ["number", "integer", "string", "array", "object", "boolean"]}
        return {
            "$id": f"render_payload/{self.variant}",
            "type": "object",
            "description": self.description,
            "properties": properties,
            "required": list(self.required_fields),
            "additionalProperties": True,
        }


@dataclass(frozen=True)
class VectorMetadataSchema:
    """Schema fragment describing per-step vector metadata."""

    required_keys: Sequence[str] = field(default_factory=tuple)
    optional_keys: Sequence[str] = field(default_factory=tuple)

    def as_json_schema(self) -> Dict[str, Any]:
        properties: MutableMapping[str, Any] = {
            "vectorized": {"type": "boolean"},
            VECTOR_ENV_BATCH_SIZE_KEY: {"type": "integer", "minimum": 1},
            TELEMETRY_KEY_AUTORESET_MODE: {"type": "string"},
            "render_modes": {"type": "array", "items": {"type": "string"}},
            "render_fps": {"type": ["number", "integer"]},
            "observation_space": {"type": "object"},
            "single_observation_space": {"type": "object"},
            "action_space": {"type": "object"},
            "single_action_space": {"type": "object"},
            VECTOR_ENV_INDEX_KEY: {"type": ["integer", "array"]},
            RESET_MASK_KEY: {"type": ["array", "boolean"]},
            VECTOR_SEED_KEY: {"type": ["array", "integer", "number"]},
        }

        required = set(self.required_keys)
        optional = set(self.optional_keys)
        # Ensure any explicitly declared optional keys exist in properties with a generic shape.
        for key in optional:
            properties.setdefault(key, {"type": ["number", "integer", "string", "array", "object", "boolean"]})
        for key in required:
            properties.setdefault(key, {"type": ["number", "integer", "string", "array", "object", "boolean"]})

        return {
            "$id": "vector_metadata/base",
            "type": "object",
            "properties": dict(properties),
            "required": sorted(required),
            "additionalProperties": True,
        }


@dataclass(frozen=True)
class BaseStepSchema:
    """Top-level schema describing telemetry step payloads."""

    schema_id: str
    version: int
    required_fields: Sequence[str]
    optional_fields: Sequence[str] = field(default_factory=tuple)
    vector_metadata: VectorMetadataSchema | None = None
    render_payloads: Sequence[RenderPayloadSchema] = field(default_factory=tuple)

    def as_json_schema(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {
            "episode_id": {"type": "string"},
            "step_index": {"type": "integer", "minimum": 0},
            "reward": {"type": "number"},
            "terminated": {"type": "boolean"},
            "truncated": {"type": "boolean"},
            "info": {"type": "object"},
        }

        for field_name in self.required_fields:
            properties.setdefault(field_name, {"type": ["string", "number", "integer", "boolean", "object"]})

        for field_name in self.optional_fields:
            properties.setdefault(field_name, {"type": ["string", "number", "integer", "boolean", "object"]})

        if self.vector_metadata is not None:
            properties[TELEMETRY_KEY_VECTOR_METADATA] = self.vector_metadata.as_json_schema()
        if TELEMETRY_KEY_SPACE_SIGNATURE not in properties:
            properties[TELEMETRY_KEY_SPACE_SIGNATURE] = {"type": "object"}
        if TELEMETRY_KEY_TIME_STEP not in properties:
            properties[TELEMETRY_KEY_TIME_STEP] = {"type": ["integer", "null"]}

        definitions: Dict[str, Any] = {}
        if self.render_payloads:
            render_refs = []
            for idx, payload in enumerate(self.render_payloads):
                schema = payload.as_json_schema()
                definition_key = f"render_payload_{idx}"
                definitions[definition_key] = schema
                render_refs.append({"$ref": f"#/$defs/{definition_key}"})
            properties["render_payload"] = {
                "oneOf": render_refs,
                "description": "Render payload variants emitted for UI consumption.",
            }

        return {
            "$id": self.schema_id,
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
            "required": sorted({"episode_id", "step_index", "reward", "terminated", "truncated", *self.required_fields}),
            "additionalProperties": True,
            "$defs": definitions or None,
            "metadata": {
                "version": self.version,
            },
        }


def build_default_step_schema() -> BaseStepSchema:
    """Return the default schema shared across most Gymnasium-derived environments."""

    vector_schema = VectorMetadataSchema(
        required_keys=("vectorized",),
        optional_keys=(
            VECTOR_ENV_BATCH_SIZE_KEY,
            TELEMETRY_KEY_AUTORESET_MODE,
            VECTOR_ENV_INDEX_KEY,
            RESET_MASK_KEY,
            VECTOR_SEED_KEY,
            "render_modes",
            "render_fps",
            "observation_space",
            "single_observation_space",
            "action_space",
            "single_action_space",
        ),
    )

    render_payloads = (
        RenderPayloadSchema(
            variant=RENDER_PAYLOAD_GRID,
            description="Grid-based board representations (FrozenLake, CliffWalking, MiniGrid).",
            required_fields=("tiles", "width", "height"),
            optional_fields=("agent_position", "goal_positions", "hazards"),
        ),
        RenderPayloadSchema(
            variant=RENDER_PAYLOAD_RGB,
            description="RGB frames from Box2D or classic control environments.",
            required_fields=("pixels",),
            optional_fields=("width", "height", "mode"),
        ),
        RenderPayloadSchema(
            variant=RENDER_PAYLOAD_GRAPH,
            description="Graph observations (nodes/edges) typically used by NetHack or custom agents.",
            required_fields=("nodes", "edges"),
            optional_fields=("node_features", "edge_features"),
        ),
    )

    return BaseStepSchema(
        schema_id="telemetry.step.default",
        version=1,
        required_fields=(
            "observation",
            TELEMETRY_KEY_TIME_STEP,
            TELEMETRY_KEY_SPACE_SIGNATURE,
        ),
        optional_fields=(
            TELEMETRY_KEY_VECTOR_METADATA,
            "agent_id",
            "render_hint",
            "frame_ref",
            "payload_version",
            "run_id",
            "worker_id",
        ),
        vector_metadata=vector_schema,
        render_payloads=render_payloads,
    )


def clone_step_schema(base: BaseStepSchema, **updates: Any) -> BaseStepSchema:
    """Return a copy of ``base`` with the provided dataclass field overrides."""

    return replace(base, **updates)
