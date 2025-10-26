from __future__ import annotations

"""Configuration helpers for trainer runs and schema validation."""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from jsonschema import Draft202012Validator, ValidationError

from gym_gui.utils import json_serialization
from . import constants as trainer_constants


SCHEMA_DEFAULTS = trainer_constants.TRAINER_DEFAULTS.schema


_TRAIN_RUN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TrainRunConfig",
    "type": "object",
    "required": [
        "run_name",
        "entry_point",
        "arguments",
        "environment",
        "resources",
        "artifacts",
    ],
    "properties": {
        "run_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": SCHEMA_DEFAULTS.run_name_max_length,
        },
        "entry_point": {"type": "string", "minLength": 1},
        "arguments": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "environment": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "default": {},
        },
        "resources": {
            "type": "object",
            "required": ["cpus", "memory_mb", "gpus"],
            "properties": {
                "cpus": {
                    "type": "integer",
                    "minimum": SCHEMA_DEFAULTS.resources_min_cpus,
                },
                "memory_mb": {
                    "type": "integer",
                    "minimum": SCHEMA_DEFAULTS.resources_min_memory_mb,
                },
                "gpus": {
                    "type": "object",
                    "required": ["requested"],
                    "properties": {
                        "requested": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": SCHEMA_DEFAULTS.resources_max_requested_gpus,
                        },
                        "mandatory": {
                            "type": "boolean",
                            "default": SCHEMA_DEFAULTS.resources_default_gpu_mandatory,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "artifacts": {
            "type": "object",
            "properties": {
                "output_prefix": {"type": "string", "minLength": 1},
                "persist_logs": {
                    "type": "boolean",
                    "default": SCHEMA_DEFAULTS.artifacts_default_persist_logs,
                },
                "keep_checkpoints": {
                    "type": "boolean",
                    "default": SCHEMA_DEFAULTS.artifacts_default_keep_checkpoints,
                },
            },
            "additionalProperties": False,
            "default": {},
        },
        "metadata": {
            "type": "object",
            "default": {},
        },
        "schedule": {
            "type": "object",
            "properties": {
                "max_duration_seconds": {
                    "type": "integer",
                    "minimum": SCHEMA_DEFAULTS.schedule_min_duration_s,
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": SCHEMA_DEFAULTS.schedule_min_steps,
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

_VALIDATOR = Draft202012Validator(_TRAIN_RUN_SCHEMA)


@dataclass(slots=True, frozen=True)
class TrainerRunMetadata:
    """Runtime identifiers derived from validated configuration."""

    run_id: str
    digest: str
    submitted_at: datetime


@dataclass(slots=True, frozen=True)
class TrainRunConfig:
    """Normalized configuration for a trainer run."""

    payload: Mapping[str, Any]
    metadata: TrainerRunMetadata

    def to_json(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"), sort_keys=True)


def _canonicalize_config(data: Mapping[str, Any]) -> MutableMapping[str, Any]:
    resources_input = data.get("resources", {})
    gpus_input = resources_input.get("gpus", {})
    cpus = int(
        resources_input.get("cpus", SCHEMA_DEFAULTS.resources_min_cpus)
    )
    cpus = max(SCHEMA_DEFAULTS.resources_min_cpus, cpus)
    memory_mb = int(
        resources_input.get("memory_mb", SCHEMA_DEFAULTS.resources_min_memory_mb)
    )
    memory_mb = max(SCHEMA_DEFAULTS.resources_min_memory_mb, memory_mb)
    gpus_requested = int(gpus_input.get("requested", 0))
    gpus_requested = max(0, min(SCHEMA_DEFAULTS.resources_max_requested_gpus, gpus_requested))
    gpus_mandatory = bool(
        gpus_input.get("mandatory", SCHEMA_DEFAULTS.resources_default_gpu_mandatory)
    )

    canonical: MutableMapping[str, Any] = {
        "run_name": data.get("run_name"),
        "entry_point": data.get("entry_point"),
        "arguments": list(data.get("arguments", [])),
        "environment": dict(data.get("environment", {})),
        "resources": {
            "cpus": cpus,
            "memory_mb": memory_mb,
            "gpus": {
                "requested": gpus_requested,
                "mandatory": gpus_mandatory,
            },
        },
        "artifacts": {
            "output_prefix": data.get("artifacts", {}).get("output_prefix"),
            "persist_logs": bool(
                data.get("artifacts", {}).get(
                    "persist_logs", SCHEMA_DEFAULTS.artifacts_default_persist_logs
                )
            ),
            "keep_checkpoints": bool(
                data.get("artifacts", {}).get(
                    "keep_checkpoints", SCHEMA_DEFAULTS.artifacts_default_keep_checkpoints
                )
            ),
        },
        "metadata": dict(data.get("metadata", {})),
    }
    if "schedule" in data:
        schedule = data["schedule"]
        max_duration = schedule.get("max_duration_seconds")
        if max_duration is not None:
            max_duration = max(
                SCHEMA_DEFAULTS.schedule_min_duration_s, int(max_duration)
            )
        max_steps = schedule.get("max_steps")
        if max_steps is not None:
            max_steps = max(SCHEMA_DEFAULTS.schedule_min_steps, int(max_steps))
        canonical["schedule"] = {
            key: value
            for key, value in (
                ("max_duration_seconds", max_duration),
                ("max_steps", max_steps),
            )
            if value is not None
        }
    return canonical


def _stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json_serialization.dumps(payload)
    return hashlib.sha256(encoded).hexdigest()


def validate_train_run_config(raw: Mapping[str, Any]) -> TrainRunConfig:
    """Validate a run config against the schema and return a canonical form."""

    errors = sorted(_VALIDATOR.iter_errors(raw), key=lambda err: err.path)
    if errors:
        messages = "; ".join(f"{'/'.join(str(p) for p in error.path)}: {error.message}" for error in errors)
        raise ValidationError(f"Invalid TrainRunConfig: {messages}")

    canonical = _canonicalize_config(raw)
    submitted = datetime.utcnow().replace(tzinfo=None)
    digest = _stable_digest(canonical)
    run_id_seed = f"{canonical['run_name']}::{submitted.isoformat()}::{digest}".encode(
        "utf-8"
    )
    run_id = hashlib.sha1(run_id_seed, usedforsecurity=False).hexdigest()

    # CRITICAL FIX: Update worker config with the correct hash-based run_id
    # The UI builds config with run_name, but we need to use the hash-based run_id
    # so that telemetry from the worker matches the database run_id
    if "metadata" in canonical and "worker" in canonical["metadata"]:
        if "config" in canonical["metadata"]["worker"]:
            canonical["metadata"]["worker"]["config"]["run_id"] = run_id

    metadata = TrainerRunMetadata(run_id=run_id, digest=digest, submitted_at=submitted)
    return TrainRunConfig(payload=canonical, metadata=metadata)


def load_train_run_config(path: Path) -> TrainRunConfig:
    """Load and validate a run config from JSON file."""

    data = json.loads(path.read_text("utf-8"))
    return validate_train_run_config(data)


__all__ = [
    "TrainRunConfig",
    "TrainerRunMetadata",
    "validate_train_run_config",
    "load_train_run_config",
]
