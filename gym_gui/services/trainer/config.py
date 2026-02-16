from __future__ import annotations

"""Configuration helpers for trainer runs and schema validation."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from jsonschema import Draft202012Validator, ValidationError
from ulid import ULID

from gym_gui.constants import TRAINER_DEFAULTS
from gym_gui.config.paths import VAR_TENSORBOARD_DIR, VAR_WANDB_DIR
from gym_gui.utils import json_serialization


SCHEMA_DEFAULTS = TRAINER_DEFAULTS.schema


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

    metadata_payload = canonical.get("metadata") if isinstance(canonical.get("metadata"), MutableMapping) else None
    worker_meta = None
    worker_config = None
    existing_run_id: str | None = None
    if isinstance(metadata_payload, MutableMapping):
        worker_meta = metadata_payload.get("worker") if isinstance(metadata_payload.get("worker"), MutableMapping) else None
        if isinstance(worker_meta, MutableMapping):
            worker_config = worker_meta.get("config") if isinstance(worker_meta.get("config"), MutableMapping) else None
            if isinstance(worker_config, MutableMapping):
                candidate_run_id = worker_config.get("run_id")
                if isinstance(candidate_run_id, str):
                    try:
                        ULID.from_str(candidate_run_id)
                        existing_run_id = candidate_run_id
                    except ValueError:
                        existing_run_id = None

    # Generate sortable run_id using ULID unless a valid ULID was already supplied.
    run_id = existing_run_id or str(ULID())

    # CRITICAL: Update worker config with the correct ULID-based run_id
    # The UI builds config with run_name, but we need to use the ULID-based run_id
    # so that telemetry from the worker matches the database run_id
    if isinstance(worker_config, MutableMapping):
        worker_config["run_id"] = run_id
        worker_id_from_config = worker_config.get("worker_id")
        if worker_id_from_config and isinstance(worker_meta, MutableMapping) and not worker_meta.get("worker_id"):
            worker_meta["worker_id"] = worker_id_from_config

    # Custom script detection: the form sets "script" (no "module") in worker
    # metadata.  Custom script forms are the single source of truth for ALL
    # paths (tensorboard, wandb, MOSAIC env vars).  We must NOT rewrite them
    # here -- doing so would redirect artifacts to var/trainer/runs/ instead
    # of var/trainer/custom_scripts/ where scripts actually write output.
    is_custom_script = (
        isinstance(worker_meta, MutableMapping)
        and worker_meta.get("script")
        and not worker_meta.get("module")
    )

    if not is_custom_script:
        # --- Standard training: compute and inject artifact paths ---
        worker_extras = {}
        if isinstance(worker_config, Mapping):
            worker_extras = worker_config.get("extras", {}) if isinstance(worker_config.get("extras"), Mapping) else {}
        tensorboard_dirname = worker_extras.get("tensorboard_dir") if isinstance(worker_extras, Mapping) else None
        if not isinstance(tensorboard_dirname, str) or not tensorboard_dirname.strip():
            tensorboard_dirname = "tensorboard"
        tensorboard_relative = tensorboard_dirname
        tensorboard_absolute = (VAR_TENSORBOARD_DIR / run_id / tensorboard_dirname).resolve()

        # WANDB manifest file path (worker will write run_path here after wandb.init())
        wandb_manifest_file = (VAR_WANDB_DIR / run_id / "wandb.json").resolve()
        wandb_relative = f"var/trainer/runs/{run_id}/wandb.json"

        if isinstance(metadata_payload, MutableMapping):
            artifacts_meta = metadata_payload.get("artifacts")
            if not isinstance(artifacts_meta, MutableMapping):
                artifacts_meta = {}
            tensorboard_meta = artifacts_meta.get("tensorboard") if isinstance(artifacts_meta, MutableMapping) else None
            if not isinstance(tensorboard_meta, MutableMapping):
                tensorboard_meta = {}
                artifacts_meta["tensorboard"] = tensorboard_meta
            tensorboard_meta["relative_path"] = tensorboard_relative
            tensorboard_meta.setdefault("enabled", True)
            tensorboard_meta["log_dir"] = str(tensorboard_absolute)

            wandb_meta = artifacts_meta.get("wandb")
            if not isinstance(wandb_meta, MutableMapping):
                wandb_meta = {}
                artifacts_meta["wandb"] = wandb_meta

            worker_extra = worker_config.get("extras", {}) if isinstance(worker_config, Mapping) else {}
            track_wandb = bool(worker_extra.get("track_wandb", False))
            wandb_meta.setdefault("enabled", track_wandb)

            wandb_meta["manifest_file"] = str(wandb_manifest_file)
            wandb_meta["relative_path"] = wandb_relative
            wandb_meta.setdefault("run_path", "")

            metadata_payload["artifacts"] = artifacts_meta

    metadata = TrainerRunMetadata(run_id=run_id, digest=digest, submitted_at=submitted)

    environment = canonical.get("environment")
    if isinstance(environment, MutableMapping) and not is_custom_script:
        # --- Standard training: rewrite env vars with canonical ULID ---
        if "CLEANRL_RUN_ID" in environment:
            environment["CLEANRL_RUN_ID"] = run_id

        if "XUANCE_RUN_ID" in environment:
            environment["XUANCE_RUN_ID"] = run_id
        if "XUANCE_TENSORBOARD_DIR" in environment:
            environment["XUANCE_TENSORBOARD_DIR"] = f"var/trainer/runs/{run_id}/tensorboard"

        if "MOSAIC_RUN_ID" in environment:
            environment["MOSAIC_RUN_ID"] = run_id
        run_dir = VAR_TENSORBOARD_DIR / run_id
        if "MOSAIC_RUN_DIR" in environment:
            environment["MOSAIC_RUN_DIR"] = str(run_dir)
        if "MOSAIC_CHECKPOINT_DIR" in environment:
            environment["MOSAIC_CHECKPOINT_DIR"] = str(run_dir / "checkpoints")
        if "MOSAIC_CONFIG_FILE" in environment:
            old_config = Path(environment["MOSAIC_CONFIG_FILE"])
            new_config_dir = run_dir / "config"
            new_config_dir.mkdir(parents=True, exist_ok=True)
            new_config_path = new_config_dir / old_config.name
            if old_config.exists() and not new_config_path.exists():
                import shutil
                shutil.move(str(old_config), str(new_config_path))
            environment["MOSAIC_CONFIG_FILE"] = str(new_config_path)
        if "MOSAIC_CUSTOM_SCRIPTS_DIR" in environment:
            environment["MOSAIC_CUSTOM_SCRIPTS_DIR"] = str(run_dir / "config")

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
