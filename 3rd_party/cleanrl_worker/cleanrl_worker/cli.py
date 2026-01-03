"""Command-line entry point for the CleanRL worker."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from gym_gui.config.paths import VAR_TRAINER_DIR
from .config import WorkerConfig, load_worker_config
from .runtime import CleanRLWorkerRuntime


def _bootstrap_vendor_packages() -> None:
    """Ensure cleanrl and cleanrl_utils are importable.

    With proper package installation (pip install -e), these are already
    available as top-level packages. This bootstrap is a no-op but kept
    for compatibility.
    """
    # cleanrl and cleanrl_utils are now proper top-level packages
    # installed via pyproject.toml, no aliasing needed
    pass


def _parse_cli(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to trainer-issued worker config JSON.")
    parser.add_argument("--algo", help="Override algorithm identifier.")
    parser.add_argument("--env-id", help="Override environment id.")
    parser.add_argument("--total-timesteps", type=int, help="Override total timesteps.")
    parser.add_argument("--seed", type=int, help="Override algorithm seed.")
    parser.add_argument("--worker-id", help="Override worker id.")
    parser.add_argument("--extras", help="JSON blob merged into config.extras overrides.")
    parser.add_argument("--grpc", action="store_true", help="Enable gRPC telemetry handshake.")
    parser.add_argument("--grpc-target", default="127.0.0.1:50055", help="gRPC target (host:port).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve module and exit without executing.")
    parser.add_argument("--emit-summary", action="store_true", help="Print resolved summary JSON to stdout.")
    return parser.parse_args(argv)


def _parse_extras_overrides(extras_json: Optional[str]) -> Dict[str, Any]:
    if not extras_json:
        return {}
    try:
        parsed = json.loads(extras_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse --extras JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SystemExit("--extras must be a JSON object")
    return parsed


def main(argv: Optional[list[str]] = None) -> int:
    _bootstrap_vendor_packages()
    args = _parse_cli(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Config path does not exist: {config_path}")

    config = load_worker_config(config_path)
    overrides = config.with_overrides(
        algo=args.algo,
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        worker_id=args.worker_id,
        extras=_parse_extras_overrides(args.extras),
    )

    runtime = CleanRLWorkerRuntime(
        overrides,
        use_grpc=args.grpc,
        grpc_target=args.grpc_target,
        dry_run=args.dry_run,
    )

    if runtime.dry_run:
        summary = runtime.run()
        if args.emit_summary:
            print(json.dumps(summary["config"], indent=2, sort_keys=True), file=sys.stdout)
        return 0

    try:
        runtime.run()
        return 0
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    except Exception:  # noqa: BLE001
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
