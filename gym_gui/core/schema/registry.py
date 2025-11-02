"""Registry for telemetry schema variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

from .step_payload import BaseStepSchema, build_default_step_schema, clone_step_schema


@dataclass
class SchemaEntry:
    """Internal container storing schema definitions and metadata."""

    schema: BaseStepSchema
    aliases: tuple[str, ...]


class TelemetrySchemaRegistry:
    """In-memory registry mapping environment families to telemetry schemas."""

    def __init__(self) -> None:
        self._schemas: MutableMapping[str, SchemaEntry] = {}

    def register(self, key: str, schema: BaseStepSchema, *, aliases: Iterable[str] | None = None) -> None:
        canonical_key = key.lower()
        alias_tuple = tuple(alias.lower() for alias in aliases or ())
        self._schemas[canonical_key] = SchemaEntry(schema=schema, aliases=alias_tuple)
        for alias in alias_tuple:
            self._schemas.setdefault(alias, SchemaEntry(schema=schema, aliases=alias_tuple))

    def get(self, key: str | None) -> Optional[BaseStepSchema]:
        if key is None:
            return None
        entry = self._schemas.get(key.lower())
        return entry.schema if entry else None

    def available(self) -> Mapping[str, BaseStepSchema]:
        return {key: entry.schema for key, entry in self._schemas.items()}


schema_registry = TelemetrySchemaRegistry()

_default_schema = build_default_step_schema()
schema_registry.register("default", _default_schema, aliases=("gymnasium", "classic-control"))


def _require_fields(schema: BaseStepSchema, extra: Sequence[str]) -> Sequence[str]:
    return tuple(sorted(set(schema.required_fields).union(extra)))


_minigrid_schema = clone_step_schema(
    _default_schema,
    schema_id="telemetry.step.minigrid",
    required_fields=_require_fields(_default_schema, ("render_payload",)),
)

schema_registry.register(
    "minigrid",
    _minigrid_schema,
    aliases=("gym-minigrid", "gymnasium-minigrid", "minigrid-env"),
)


_atari_schema = clone_step_schema(
    _default_schema,
    schema_id="telemetry.step.atari",
    required_fields=_require_fields(_default_schema, ("render_payload", "frame_ref")),
)

schema_registry.register(
    "atari",
    _atari_schema,
    aliases=("ale", "gym-atari", "gymnasium-atari"),
)


def resolve_schema_for_game(game_id: str | None) -> BaseStepSchema | None:
    """Return the best matching schema for a Gymnasium environment identifier."""

    if not game_id:
        return schema_registry.get("default")

    lowered = game_id.lower()
    candidates = [game_id, lowered]
    if "minigrid" in lowered:
        candidates.append("minigrid")
    if lowered.startswith("ale/") or "noframeskip" in lowered or "atari" in lowered:
        candidates.append("atari")
    candidates.append("default")

    for key in candidates:
        schema = schema_registry.get(key)
        if schema is not None:
            return schema
    return None
