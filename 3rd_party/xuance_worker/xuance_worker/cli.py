"""CLI entry point for XuanCe worker.

This module provides the command-line interface for running XuanCe
training jobs, either directly or as a subprocess launched by MOSAIC.

Usage:
    # Direct parameter mode
    xuance-worker --method ppo --env classic_control --env-id CartPole-v1

    # Config file mode
    xuance-worker --config /path/to/config.json

    # Dry-run mode (validate without executing)
    xuance-worker --method dqn --env atari --env-id Pong-v5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from .config import XuanCeWorkerConfig
from .runtime import XuanCeWorkerRuntime


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (None for sys.argv).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="XuanCe Worker - MOSAIC Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file mode
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file",
    )

    # Direct parameter mode
    parser.add_argument(
        "--method",
        type=str,
        default="ppo",
        help="Algorithm name (dqn, ppo, sac, ddpg, td3, mappo, qmix, maddpg, etc.)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="classic_control",
        help="Environment family (classic_control, atari, mujoco, box2d, mpe, smac, football)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Specific environment ID (CartPole-v1, Pong-v5, simple_spread_v3, etc.)",
    )
    parser.add_argument(
        "--running-steps",
        type=int,
        default=1_000_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Computing device",
    )
    parser.add_argument(
        "--parallels",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--dl-toolbox",
        type=str,
        default="torch",
        choices=["torch", "tensorflow", "mindspore"],
        help="Deep learning backend (torch, tensorflow, mindspore)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (load model and evaluate)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode (training with periodic evaluation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without executing",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Custom YAML config path for XuanCe (overrides defaults)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for xuance-worker CLI.

    Args:
        args: Command-line arguments (None for sys.argv).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parsed = parse_args(args)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("xuance_worker")

    # Build configuration
    if parsed.config is not None:
        # Load from JSON file
        if not parsed.config.exists():
            logger.error("Config file not found: %s", parsed.config)
            return 1
        try:
            config = XuanCeWorkerConfig.from_json_file(parsed.config)
            logger.info("Loaded config from: %s", parsed.config)
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return 1
    else:
        # Build from CLI arguments
        config = XuanCeWorkerConfig(
            run_id=parsed.run_id or str(uuid.uuid4())[:8],
            method=parsed.method,
            env=parsed.env,
            env_id=parsed.env_id,
            dl_toolbox=parsed.dl_toolbox,
            running_steps=parsed.running_steps,
            seed=parsed.seed,
            device=parsed.device,
            parallels=parsed.parallels,
            test_mode=parsed.test,
            config_path=parsed.config_path,
        )

    logger.info(
        "XuanCe Worker Config: method=%s env=%s env_id=%s backend=%s steps=%d device=%s",
        config.method,
        config.env,
        config.env_id,
        config.dl_toolbox,
        config.running_steps,
        config.device,
    )

    # Dry-run mode: print config and exit
    if parsed.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Create runtime and execute
    runtime = XuanCeWorkerRuntime(config, dry_run=False)

    try:
        if parsed.benchmark:
            logger.info("Running in benchmark mode")
            summary = runtime.benchmark()
        else:
            summary = runtime.run()

        logger.info(
            "Run completed: status=%s runner=%s",
            summary.status,
            summary.runner_type,
        )
        return 0

    except RuntimeError as e:
        # XuanCe not installed or configuration error
        logger.error("Runtime error: %s", e)
        return 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130  # Standard Unix SIGINT exit code

    except Exception as e:
        logger.exception("XuanCe worker failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
