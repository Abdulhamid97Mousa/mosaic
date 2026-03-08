"""CLI entry point for the MARLlib worker.

Supports two modes:

1. **Config file** (from MOSAIC GUI/trainer)::

       marllib-worker --config /path/to/config.json

2. **Direct arguments** (interactive / scripting)::

       marllib-worker --run-id run1 --algo mappo \\
           --environment-name mpe --map-name simple_spread
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MARLlibWorkerConfig, load_worker_config

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------


def _parse_cli(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="marllib-worker",
        description="MARLlib Multi-Agent RL worker for MOSAIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Config file ---
    p.add_argument("--config", help="Path to trainer-issued worker config JSON.")

    # --- Direct configuration ---
    g = p.add_argument_group("Direct configuration")
    g.add_argument("--run-id", help="Unique run identifier.")
    g.add_argument("--algo", help="Algorithm (e.g. mappo, qmix, ippo).")
    g.add_argument(
        "--environment-name",
        "--env",
        dest="environment_name",
        help="Environment (e.g. mpe, smac).",
    )
    g.add_argument("--map-name", help="Map/scenario (e.g. simple_spread, 3m).")
    g.add_argument("--force-coop", action="store_true", help="Force global reward.")
    g.add_argument(
        "--hyperparam-source",
        default="common",
        help="Hyperparameter source: 'common', 'test', or env name.",
    )
    g.add_argument(
        "--share-policy",
        default="all",
        choices=["all", "group", "individual"],
    )
    g.add_argument(
        "--core-arch", default="mlp", choices=["mlp", "gru", "lstm"]
    )
    g.add_argument("--encode-layer", default="128-256")

    # --- Training ---
    t = p.add_argument_group("Training")
    t.add_argument("--num-gpus", type=int, default=1)
    t.add_argument("--num-workers", type=int, default=2)
    t.add_argument("--local-mode", action="store_true", help="Ray local/debug mode.")
    t.add_argument("--framework", default="torch", choices=["torch", "tf"])
    t.add_argument("--stop-timesteps", type=int, default=1_000_000)
    t.add_argument("--stop-reward", type=float, default=999_999.0)
    t.add_argument("--stop-iters", type=int, default=9_999_999)
    t.add_argument("--checkpoint-freq", type=int, default=100)
    t.add_argument(
        "--no-checkpoint-end",
        action="store_true",
        help="Do not save checkpoint at training end.",
    )
    t.add_argument("--seed", type=int, default=None)

    # --- Restore ---
    r = p.add_argument_group("Checkpoint restore")
    r.add_argument("--restore-model-path", default="")
    r.add_argument("--restore-params-path", default="")

    # --- JSON overrides ---
    j = p.add_argument_group("JSON overrides")
    j.add_argument("--algo-params", help="JSON dict of algo hyperparameter overrides.")
    j.add_argument("--env-params", help="JSON dict of environment parameter overrides.")
    j.add_argument("--extras", help="JSON dict merged into config.extras.")

    # --- Flags ---
    p.add_argument("--dry-run", action="store_true", help="Validate without training.")
    p.add_argument(
        "--emit-summary", action="store_true", help="Print JSON summary to stdout."
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    return p.parse_args(argv)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_json_arg(value: Optional[str], label: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse {label} JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{label} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _build_config_from_args(args: argparse.Namespace) -> MARLlibWorkerConfig:
    if not args.run_id:
        raise SystemExit("--run-id is required when not using --config")
    if not args.algo:
        raise SystemExit("--algo is required when not using --config")
    if not args.environment_name:
        raise SystemExit("--environment-name is required when not using --config")
    if not args.map_name:
        raise SystemExit("--map-name is required when not using --config")

    return MARLlibWorkerConfig(
        run_id=args.run_id,
        algo=args.algo,
        environment_name=args.environment_name,
        map_name=args.map_name,
        force_coop=args.force_coop,
        hyperparam_source=args.hyperparam_source,
        share_policy=args.share_policy,
        core_arch=args.core_arch,
        encode_layer=args.encode_layer,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        local_mode=args.local_mode,
        framework=args.framework,
        checkpoint_end=not args.no_checkpoint_end,
        stop_timesteps=args.stop_timesteps,
        stop_reward=args.stop_reward,
        stop_iters=args.stop_iters,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        restore_model_path=args.restore_model_path or "",
        restore_params_path=args.restore_params_path or "",
        algo_params=_parse_json_arg(args.algo_params, "--algo-params"),
        env_params=_parse_json_arg(args.env_params, "--env-params"),
        extras=_parse_json_arg(args.extras, "--extras"),
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    args = _parse_cli(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Build config ---
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.exists():
            LOGGER.error("Config file not found: %s", config_path)
            return 1
        try:
            config = load_worker_config(config_path)
        except Exception:
            LOGGER.exception("Failed to load config from %s", config_path)
            return 1
    else:
        try:
            config = _build_config_from_args(args)
        except (SystemExit, ValueError) as exc:
            LOGGER.error("%s", exc)
            return 1

    # --- Execute ---
    from .runtime import MARLlibWorkerRuntime

    runtime = MARLlibWorkerRuntime(config, dry_run=args.dry_run)

    try:
        summary = runtime.run()
        if args.emit_summary or args.dry_run:
            print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
        return 130
    except Exception:
        LOGGER.exception("MARLlib worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
