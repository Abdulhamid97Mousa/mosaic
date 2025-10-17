"""Headless worker entrypoint producing JSONL telemetry for trainer orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import create_adapter
from .core import RunConfig, TelemetryEmitter
from .core.runtime import HeadlessTrainer


def _read_config_from_path(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config(args: argparse.Namespace) -> RunConfig:
    if args.config_path:
        config_payload = _read_config_from_path(Path(args.config_path).expanduser().resolve())
    else:
        try:
            config_payload = json.load(sys.stdin)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Failed to parse JSON config from stdin: {exc}") from exc
    return RunConfig.from_dict(config_payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to a JSON config file. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without running episodes.",
    )
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Stream telemetry via gRPC to trainer daemon (overrides config).",
    )
    parser.add_argument(
        "--grpc-target",
        default="127.0.0.1:50055",
        help="Daemon gRPC address (default: 127.0.0.1:50055).",
    )
    parsed = parser.parse_args(argv)

    run_config = _load_config(parsed)
    
    # Override config with CLI flags
    if parsed.grpc:
        run_config.extra["use_grpc_telemetry"] = True
        run_config.extra["grpc_target"] = parsed.grpc_target
    
    emitter = TelemetryEmitter()

    adapter_kwargs = run_config.extra.get("adapter_kwargs", {})
    adapter = create_adapter(run_config.env_id, **adapter_kwargs)

    if parsed.dry_run:
        emitter.run_started(run_config.run_id, _dry_run_payload(run_config))
        emitter.run_completed(run_config.run_id, status="skipped", reason="dry-run")
        return 0

    trainer = HeadlessTrainer(adapter, run_config, emitter)

    try:
        return trainer.run()
    except KeyboardInterrupt:
        emitter.run_completed(run_config.run_id, status="cancelled")
        return 2


def _dry_run_payload(config: RunConfig) -> Dict[str, Any]:
    return {
        "env_id": config.env_id,
        "policy_strategy": config.policy_strategy.value,
        "policy_path": str(config.ensure_policy_path()),
        "agent_id": config.agent_id,
        "headless": config.headless,
    }


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    sys.exit(main())
