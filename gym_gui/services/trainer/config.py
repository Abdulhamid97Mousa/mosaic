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
        "run_name": {"type": "string", "minLength": 1, "maxLength": 120},
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
                "cpus": {"type": "integer", "minimum": 1},
                "memory_mb": {"type": "integer", "minimum": 256},
                "gpus": {
                    "type": "object",
                    "required": ["requested"],
                    "properties": {
                        "requested": {"type": "integer", "minimum": 0, "maximum": 8},
                        "mandatory": {"type": "boolean", "default": False},
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
                "persist_logs": {"type": "boolean", "default": True},
                "keep_checkpoints": {"type": "boolean", "default": True},
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
                "max_duration_seconds": {"type": "integer", "minimum": 1},
                "max_steps": {"type": "integer", "minimum": 1},
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
    canonical: MutableMapping[str, Any] = {
        "run_name": data.get("run_name"),
        "entry_point": data.get("entry_point"),
        "arguments": list(data.get("arguments", [])),
        "environment": dict(data.get("environment", {})),
        "resources": {
            "cpus": int(data["resources"]["cpus"]),
            "memory_mb": int(data["resources"]["memory_mb"]),
            "gpus": {
                "requested": int(data["resources"]["gpus"]["requested"]),
                "mandatory": bool(data["resources"]["gpus"].get("mandatory", False)),
            },
        },
        "artifacts": {
            "output_prefix": data.get("artifacts", {}).get("output_prefix"),
            "persist_logs": bool(data.get("artifacts", {}).get("persist_logs", True)),
            "keep_checkpoints": bool(
                data.get("artifacts", {}).get("keep_checkpoints", True)
            ),
        },
        "metadata": dict(data.get("metadata", {})),
    }
    if "schedule" in data:
        schedule = data["schedule"]
        canonical["schedule"] = {
            key: schedule[key]
            for key in ("max_duration_seconds", "max_steps")
            if key in schedule
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
