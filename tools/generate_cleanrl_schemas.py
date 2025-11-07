#!/usr/bin/env python3
"""Generate CleanRL algorithm parameter schemas for the GUI."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args, get_origin, Union

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "3rd_party" / "cleanrl_worker"
CLEANRL_PACKAGE_DIR = VENDOR_ROOT / "cleanrl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "metadata" / "cleanrl"
SUPPORTED_SIMPLE_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}
RUNTIME_ONLY_FIELDS = {"batch_size", "minibatch_size", "num_iterations"}


def _load_cleanrl_version() -> str:
    pyproject = VENDOR_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return data.get("project", {}).get("version", "vendored")


def _normalize_type(annotation: Any) -> tuple[str | None, bool]:
    origin = get_origin(annotation)
    if origin is None:
        return SUPPORTED_SIMPLE_TYPES.get(annotation), False

    args = get_args(annotation)
    if origin is type(None):
        return None, True
    if origin in (list, tuple, dict, set):
        return None, False
    if origin is None:
        return SUPPORTED_SIMPLE_TYPES.get(annotation), False

    # Optional[T] / Union[T, None]
    if origin is Union and args:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            simple, _ = _normalize_type(non_none[0])
            return simple, True
        return None, True

    return SUPPORTED_SIMPLE_TYPES.get(origin, None), False


def _serialize_field(field: dataclasses.Field[Any]) -> dict[str, Any] | None:
    type_name_optional = _normalize_type(field.type)
    type_name, optional = type_name_optional
    if not type_name:
        return None

    default: Any = None
    if field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
        try:
            default = field.default_factory()  # type: ignore[misc]
        except Exception:
            default = None

    return {
        "name": field.name,
        "type": type_name,
        "optional": optional,
        "default": default,
        "help": field.metadata.get("help", "") if field.metadata else "",
        "runtime_only": field.name in RUNTIME_ONLY_FIELDS,
    }


def generate_schemas(output_dir: Path) -> Path:
    version = _load_cleanrl_version()
    schemas: dict[str, Any] = {
        "cleanrl_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithms": {},
    }

    sys.path.insert(0, str(VENDOR_ROOT))

    for module_path in sorted(CLEANRL_PACKAGE_DIR.glob("*.py")):
        if module_path.stem.startswith("__"):
            continue
        module_name = f"cleanrl.{module_path.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import guard
            print(f"[WARN] Failed to import {module_name}: {exc}")
            continue

        args_cls = getattr(module, "Args", None)
        if args_cls is None or not dataclasses.is_dataclass(args_cls):
            continue

        field_specs = []
        for field in dataclasses.fields(args_cls):
            spec = _serialize_field(field)
            if spec is None:
                continue
            field_specs.append(spec)

        schemas["algorithms"][module_path.stem] = {
            "module": module_name,
            "fields": field_specs,
        }

    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    output_file = version_dir / "schemas.json"
    output_file.write_text(json.dumps(schemas, indent=2, sort_keys=True))
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated metadata (default: metadata/cleanrl)",
    )
    args = parser.parse_args()
    output = generate_schemas(args.output_dir)
    print(f"Wrote CleanRL schemas to {output}")


if __name__ == "__main__":
    main()
