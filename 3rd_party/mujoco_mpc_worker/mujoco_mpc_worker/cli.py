"""CLI entry point for MuJoCo MPC Worker.

Usage:
    mujoco-mpc-worker --task Cartpole
    mujoco-mpc-worker --task "Humanoid Track" --planner predictive_sampling
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from mujoco_mpc_worker.config import MuJoCoMPCConfig, MuJoCoMPCPlannerType

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MuJoCo MPC Worker for MOSAIC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="Cartpole",
        help="MJPC task identifier (e.g., 'Cartpole', 'Humanoid Track')",
    )

    parser.add_argument(
        "--planner",
        type=str,
        choices=[p.value for p in MuJoCoMPCPlannerType],
        default=MuJoCoMPCPlannerType.PREDICTIVE_SAMPLING.value,
        help="Planner algorithm to use",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="gRPC port (0 = auto-assign)",
    )

    parser.add_argument(
        "--real-time-speed",
        type=float,
        default=1.0,
        help="Ratio of simulation speed to wall clock (0.0 to 1.0)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for MuJoCo MPC Worker CLI."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("MuJoCo MPC Worker starting...")
    _LOGGER.info(f"Task: {args.task}")
    _LOGGER.info(f"Planner: {args.planner}")

    # Create configuration
    config = MuJoCoMPCConfig(
        task_id=args.task,
        planner_type=MuJoCoMPCPlannerType(args.planner),
        port=args.port if args.port > 0 else None,
        real_time_speed=args.real_time_speed,
    )

    _LOGGER.info(f"Configuration: {config}")

    # TODO: Launch MJPC agent with configuration
    # This will be implemented when we add the agent_wrapper
    _LOGGER.warning("Agent launch not yet implemented - CLI scaffold only")

    return 0


if __name__ == "__main__":
    sys.exit(main())
