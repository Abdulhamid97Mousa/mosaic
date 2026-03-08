"""Command-line entry point for the CleanRL worker.

Supports two modes:
1. Training mode: Requires --config file with full training parameters
2. Interactive mode: For GUI step-by-step policy evaluation (--interactive)

Example usage:
    # Training mode (requires config file)
    cleanrl-worker --config /path/to/config.json

    # Interactive mode (for GUI control)
    cleanrl-worker --interactive --run-id test123 --algo ppo \\
        --env-id MiniGrid-Empty-8x8-v0 --policy-path /path/to/model.cleanrl_model
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import WorkerConfig, CleanRLWorkerConfig, load_worker_config


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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for GUI step-by-step policy evaluation. "
             "Reads commands from stdin, emits telemetry to stdout.",
    )

    # Config file (for training mode)
    parser.add_argument(
        "--config",
        help="Path to trainer-issued worker config JSON (required for training mode).",
    )

    # Direct arguments (for interactive mode)
    interactive_group = parser.add_argument_group("Interactive Mode Arguments")
    interactive_group.add_argument("--run-id", help="Unique run identifier.")
    interactive_group.add_argument("--algo", default="ppo", help="Algorithm (default: ppo).")
    interactive_group.add_argument("--env-id", help="Environment ID (e.g., MiniGrid-Empty-8x8-v0).")
    interactive_group.add_argument("--policy-path", help="Path to trained policy checkpoint.")
    interactive_group.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    # Overrides (work with both modes)
    override_group = parser.add_argument_group("Overrides")
    override_group.add_argument("--total-timesteps", type=int, help="Override total timesteps.")
    override_group.add_argument("--worker-id", help="Override worker id.")
    override_group.add_argument("--extras", help="JSON blob merged into config.extras overrides.")

    # gRPC settings
    grpc_group = parser.add_argument_group("gRPC")
    grpc_group.add_argument("--grpc", action="store_true", help="Enable gRPC telemetry handshake.")
    grpc_group.add_argument("--grpc-target", default="127.0.0.1:50055", help="gRPC target (host:port).")

    # Other options
    parser.add_argument("--dry-run", action="store_true", help="Resolve module and exit without executing.")
    parser.add_argument("--emit-summary", action="store_true", help="Print resolved summary JSON to stdout.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

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


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the worker."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_interactive_config(args: argparse.Namespace) -> CleanRLWorkerConfig:
    """Build config from CLI arguments for interactive mode."""
    if not args.run_id:
        raise ValueError("--run-id is required for interactive mode")
    if not args.env_id:
        raise ValueError("--env-id is required for interactive mode")
    if not args.policy_path:
        raise ValueError("--policy-path is required for interactive mode")

    extras = _parse_extras_overrides(args.extras) if args.extras else {}
    extras["mode"] = "interactive"
    extras["policy_path"] = args.policy_path

    return CleanRLWorkerConfig(
        run_id=args.run_id,
        algo=args.algo or "ppo",
        env_id=args.env_id,
        total_timesteps=1,  # Not used in interactive mode
        seed=args.seed,
        worker_id=args.worker_id,
        extras=extras,
    )


def main(argv: Optional[list[str]] = None) -> int:
    _bootstrap_vendor_packages()
    args = _parse_cli(argv)

    _setup_logging(verbose=getattr(args, 'verbose', False))
    logger = logging.getLogger("cleanrl_worker")

    # Interactive mode - step-by-step policy evaluation
    if args.interactive:
        try:
            config = _build_interactive_config(args)
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return 1

        logger.info(f"CleanRL Worker - Interactive Mode")
        logger.info(f"Run ID: {config.run_id}")
        logger.info(f"Environment: {config.env_id}")
        logger.info(f"Algorithm: {config.algo}")
        logger.info(f"Policy: {config.extras.get('policy_path')}")

        try:
            from .runtime import InteractiveRuntime
            runtime = InteractiveRuntime(config)
            runtime.run()
            logger.info("Interactive mode completed")
            return 0
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            return 130
        except Exception as e:
            logger.exception(f"Interactive mode failed: {e}")
            return 1

    # Training mode - requires config file
    if not args.config:
        logger.error("--config is required for training mode (or use --interactive)")
        return 1

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Config path does not exist: {config_path}")

    config = load_worker_config(config_path)
    overrides = config.with_overrides(
        algo=args.algo if args.algo != "ppo" else None,  # Don't override default
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        worker_id=args.worker_id,
        extras=_parse_extras_overrides(args.extras),
    )

    # Check for curriculum training mode
    from .curriculum_training import is_curriculum_config
    if is_curriculum_config(overrides):
        logger.info("Detected curriculum_schedule in config - using curriculum training mode")
        from .curriculum_training import CurriculumTrainingConfig, run_curriculum_training

        try:
            curriculum_config = CurriculumTrainingConfig.from_worker_config(overrides)
            result = run_curriculum_training(curriculum_config)
            logger.info(f"Curriculum training complete: {result}")
            return 0
        except Exception as e:
            logger.exception(f"Curriculum training failed: {e}")
            return 1

    # Standard training mode
    from .runtime import CleanRLWorkerRuntime
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
