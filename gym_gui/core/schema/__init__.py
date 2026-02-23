"""Schema definitions describing telemetry payload contracts.

The schema layer complements the runtime validation helpers in
``gym_gui.validations`` by providing reusable JSON-schema snippets that can be
shared between workers, trainers, and the GUI.  Individual environment
families register their schema variants with :class:`TelemetrySchemaRegistry` so
that downstream services can select the correct contract at runtime.
"""

from __future__ import annotations

from .registry import TelemetrySchemaRegistry, resolve_schema_for_game, schema_registry
from .step_payload import (
    BaseStepSchema,
    RenderPayloadSchema,
    VectorMetadataSchema,
    build_default_step_schema,
    clone_step_schema,
)

__all__ = [
    "BaseStepSchema",
    "RenderPayloadSchema",
    "TelemetrySchemaRegistry",
    "VectorMetadataSchema",
    "build_default_step_schema",
    "clone_step_schema",
    "resolve_schema_for_game",
    "schema_registry",
]
