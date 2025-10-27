"""Headless worker entrypoint producing JSONL telemetry for trainer orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import create_adapter
from .core import RunConfig, TelemetryEmitter
from .core.runtime import HeadlessTrainer
from .core.bdi_trainer import BDITrainer


def _read_config_from_path(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_config(args: argparse.Namespace) -> RunConfig:
    # Priority: CLI arg > environment variable > stdin
    config_path = args.config_path

    # If no CLI arg, check environment variable
    if not config_path:
        config_path = os.environ.get("TRAINER_WORKER_CONFIG_PATH")

    if config_path:
        config_payload = _read_config_from_path(Path(config_path).expanduser().resolve())
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
    parser.add_argument(
        "--bdi",
        action="store_true",
        help="Use BDI-RL mode (requires SPADE-BDI and ejabberd).",
    )
    parser.add_argument(
        "--bdi-jid",
        default="agent@localhost",
        help="XMPP JID for BDI agent (default: agent@localhost).",
    )
    parser.add_argument(
        "--bdi-password",
        default="secret",
        help="XMPP password for BDI agent (default: secret).",
    )
    parser.add_argument(
        "--asl-file",
        default=None,
        help="Path to custom AgentSpeak (.asl) file for BDI agent.",
    )
    parsed = parser.parse_args(argv)

    run_config = _load_config(parsed)

    # Override config with CLI flags
    if parsed.grpc:
        run_config.extra["use_grpc_telemetry"] = True
        run_config.extra["grpc_target"] = parsed.grpc_target

    emitter = TelemetryEmitter()

    # Extract game_config from run_config.extra (set by GUI's Agent Train Form)
    # and pass it to the adapter factory
    game_config = run_config.extra.pop("game_config", None)
    adapter_kwargs = run_config.extra.get("adapter_kwargs", {})
    
    # Include game_config in kwargs if provided
    if game_config is not None:
        adapter_kwargs["game_config"] = game_config
    
    adapter = create_adapter(run_config.game_id, **adapter_kwargs)
    
    # GUI adapters require explicit load() call to initialize the environment
    adapter.load()

    if parsed.dry_run:
        emitter.run_started(run_config.run_id, _dry_run_payload(run_config))
        emitter.run_completed(run_config.run_id, status="skipped", reason="dry-run")
        return 0

    # Choose trainer based on BDI flag
    if parsed.bdi:
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid=parsed.bdi_jid,
            password=parsed.bdi_password,
            asl_file=parsed.asl_file,
        )
    else:
        trainer = HeadlessTrainer(adapter, run_config, emitter)

    try:
        return trainer.run()
    except KeyboardInterrupt:
        emitter.run_completed(run_config.run_id, status="cancelled")
        return 2


def _dry_run_payload(config: RunConfig) -> Dict[str, Any]:
    return {
        "game_id": config.game_id,
        "policy_strategy": config.policy_strategy.value,
        "policy_path": str(config.ensure_policy_path()),
        "agent_id": config.agent_id,
        "headless": config.headless,
    }


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    sys.exit(main())
