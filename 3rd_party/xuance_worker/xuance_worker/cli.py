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

# CRITICAL: Set MPI environment variables BEFORE any imports that might load mpi4py.
# mpi4py initializes MPI on import, which can hang indefinitely in single-process mode
# or when spawned as a subprocess. Setting MPI4PY_RC_INITIALIZE=false prevents this.
import os
os.environ.setdefault('MPI4PY_RC_INITIALIZE', 'false')
os.environ.setdefault('OMPI_MCA_btl', '^openib')
os.environ.setdefault('OMPI_MCA_btl_base_warn_component_unused', '0')

# NOTE: sitecustomize import is DEFERRED until after arg parsing
# to avoid loading heavy dependencies (torch, gymnasium, mpi4py) in dry-run mode.
# This prevents MPI from hanging the process on exit during validation.
# See: _import_sitecustomize() function below.

import argparse
import json
import logging
import sys
import uuid
from dataclasses import asdict
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
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing training",
    )
    parser.add_argument(
        "--emit-summary",
        action="store_true",
        help="Emit resolved configuration summary as JSON (use with --dry-run)",
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

    # Interactive mode for Multi-Operator view (step-by-step policy evaluation)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for GUI step-by-step policy evaluation",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Path to trained policy checkpoint (required for interactive mode)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling in interactive mode (default: deterministic/argmax)",
    )

    # MOSAIC integration arguments (passed by trainer dispatcher)
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Enable gRPC telemetry (passed by MOSAIC dispatcher)",
    )
    parser.add_argument(
        "--grpc-target",
        type=str,
        default="127.0.0.1:50055",
        help="gRPC telemetry server address",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker identifier (for multi-worker training)",
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
            # Override worker_id from CLI if provided
            if parsed.worker_id:
                config = XuanCeWorkerConfig(
                    run_id=config.run_id,
                    method=config.method,
                    env=config.env,
                    env_id=config.env_id,
                    dl_toolbox=config.dl_toolbox,
                    running_steps=config.running_steps,
                    seed=config.seed,
                    device=config.device,
                    parallels=config.parallels,
                    test_mode=config.test_mode,
                    config_path=config.config_path,
                    worker_id=parsed.worker_id,
                    extras=config.extras,
                )
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
            worker_id=parsed.worker_id,
        )

    # Log gRPC settings if enabled
    if parsed.grpc:
        logger.info("gRPC telemetry enabled, target: %s", parsed.grpc_target)

    logger.info(
        "XuanCe Worker Config: method=%s env=%s env_id=%s backend=%s steps=%d device=%s",
        config.method,
        config.env,
        config.env_id,
        config.dl_toolbox,
        config.running_steps,
        config.device,
    )

    # Dry-run mode: validate configuration without executing training
    if parsed.dry_run:
        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        try:
            summary = runtime.run()
            if parsed.emit_summary:
                # Emit resolved configuration as JSON for GUI validation
                # Convert dataclass to dict for JSON serialization
                print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            else:
                # Legacy behavior: just print config
                print(json.dumps(config.to_dict(), indent=2))
            return 0
        except Exception as e:
            logger.error("Dry-run validation failed: %s", e)
            if parsed.emit_summary:
                error_summary = {
                    "status": "error",
                    "error": str(e),
                    "config": config.to_dict(),
                }
                print(json.dumps(error_summary, indent=2, sort_keys=True))
            return 1

    # CRITICAL: Import sitecustomize NOW to patch gym.make() with FastLane wrapper
    # This is deferred from module-level to avoid loading heavy dependencies
    # (torch, gymnasium, mpi4py) in dry-run mode, which can hang on MPI cleanup.
    from . import sitecustomize  # noqa: F401 - patches gym.make for FastLane

    # Interactive mode: run step-by-step policy evaluation
    if parsed.interactive:
        from .runtime import InteractiveRuntime, InteractiveConfig

        if not parsed.policy_path:
            logger.error("--policy-path is required for interactive mode")
            return 1

        interactive_config = InteractiveConfig(
            run_id=config.run_id,
            env_id=config.env_id,
            method=config.method,
            policy_path=parsed.policy_path,
            device=config.device,
            deterministic=not getattr(parsed, 'stochastic', False),
        )
        runtime = InteractiveRuntime(interactive_config)
        runtime.run()
        return 0

    # Check for multi-agent curriculum first (more specific: env=multigrid + schedule)
    from .multi_agent_curriculum_training import is_multi_agent_curriculum_config
    if is_multi_agent_curriculum_config(config):
        logger.info("Detected multi-agent curriculum_schedule - using MARL curriculum mode")
        from .multi_agent_curriculum_training import (
            MultiAgentCurriculumConfig, run_multi_agent_curriculum_training,
        )

        try:
            ma_config = MultiAgentCurriculumConfig.from_worker_config(config)
            result = run_multi_agent_curriculum_training(ma_config)
            logger.info(f"Multi-agent curriculum training complete: {result}")
            return 0
        except Exception as e:
            logger.exception(f"Multi-agent curriculum training failed: {e}")
            return 1

    # Check for single-agent curriculum (BabyAI/MiniGrid with Syllabus-RL)
    from .single_agent_curriculum_training import is_curriculum_config
    if is_curriculum_config(config):
        logger.info("Detected curriculum_schedule in config - using single-agent curriculum mode")
        from .single_agent_curriculum_training import CurriculumTrainingConfig, run_curriculum_training

        try:
            curriculum_config = CurriculumTrainingConfig.from_worker_config(config)
            result = run_curriculum_training(curriculum_config)
            logger.info(f"Curriculum training complete: {result}")
            return 0
        except Exception as e:
            logger.exception(f"Curriculum training failed: {e}")
            return 1

    # Create runtime and execute
    runtime = XuanCeWorkerRuntime(config, dry_run=False)

    try:
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
