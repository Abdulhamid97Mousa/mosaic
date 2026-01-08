"""Command-line interface for MCTX GPU-accelerated worker.

This module provides the CLI entry point for launching MCTS-based training
with mctx and Pgx. It's invoked by the MOSAIC TrainerDispatcher.

Usage:
    python -m mctx_worker.cli --config /path/to/config.json

    # Or using the installed entry point:
    mctx-worker --config /path/to/config.json

The config file should contain the training configuration in JSON format.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import MCTXWorkerConfig, load_worker_config
from .runtime import MCTXWorkerRuntime

_LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="mctx-worker",
        description="MCTX GPU-accelerated MCTS training worker for MOSAIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train AlphaZero on chess
  python -m mctx_worker.cli --config var/trainer/runs/chess_alphazero/config.json

  # Train with verbose output
  python -m mctx_worker.cli --config config.json --verbose

  # Dry run (validate config only)
  python -m mctx_worker.cli --config config.json --dry-run

Supported Games (via Pgx):
  - chess, go_9x9, go_19x19, shogi
  - connect_four, tic_tac_toe, othello, hex
  - backgammon, kuhn_poker, leduc_holdem
  - 2048, minatar-* (single-player)
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configuration without running training",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "gpu", "tpu"],
        help="Override device from config (cpu, gpu, tpu)",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path from config",
    )

    # gRPC arguments (for compatibility with dispatcher)
    parser.add_argument(
        "--grpc",
        action="store_true",
        default=False,
        help="Enable gRPC telemetry handshake (reserved for future use)",
    )

    parser.add_argument(
        "--grpc-target",
        type=str,
        default="127.0.0.1:50055",
        help="gRPC target host:port (reserved for future use)",
    )

    parser.add_argument(
        "--worker-id",
        type=str,
        default="mctx_worker",
        help="Worker identifier for distributed runs",
    )

    return parser


def validate_config(config: MCTXWorkerConfig) -> bool:
    """Validate the configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    errors = config.validate()

    if errors:
        for error in errors:
            _LOGGER.error(f"Configuration error: {error}")
        return False

    return True


def print_config_summary(config: MCTXWorkerConfig) -> None:
    """Print a summary of the configuration.

    Args:
        config: Configuration to summarize
    """
    print("=" * 60)
    print("MCTX GPU-Accelerated Training Configuration")
    print("=" * 60)
    print(f"Run ID:         {config.run_id}")
    print(f"Environment:    {config.env_id}")
    print(f"Algorithm:      {config.algorithm.value}")
    print(f"Mode:           {config.mode}")
    print(f"Device:         {config.device}")
    print(f"Seed:           {config.seed}")
    print("-" * 60)
    print("Network:")
    print(f"  Res Blocks:   {config.network.num_res_blocks}")
    print(f"  Channels:     {config.network.channels}")
    print(f"  Hidden Dims:  {config.network.hidden_dims}")
    print("-" * 60)
    print("MCTS:")
    print(f"  Simulations:  {config.mcts.num_simulations}")
    print(f"  Dirichlet α:  {config.mcts.dirichlet_alpha}")
    print(f"  Temperature:  {config.mcts.temperature}")
    print("-" * 60)
    print("Training:")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Batch Size:    {config.training.batch_size}")
    print(f"  Buffer Size:   {config.training.replay_buffer_size}")
    print(f"  Max Steps:     {config.max_steps or 'unlimited'}")
    print("=" * 60)
    sys.stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        _LOGGER.error(f"Configuration file not found: {config_path}")
        print(f"[ERROR] config_file_not_found path={config_path}")
        return 1

    try:
        config = load_worker_config(str(config_path))
    except Exception as e:
        _LOGGER.error(f"Failed to load configuration: {e}")
        print(f"[ERROR] config_load_failed error={e}")
        return 1

    # Override device if specified
    if args.device:
        config.device = args.device

    # Override checkpoint path if specified
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path

    # Validate configuration
    if not validate_config(config):
        print("[ERROR] config_validation_failed")
        return 1

    # Print configuration summary
    print_config_summary(config)

    # Dry run mode
    if args.dry_run:
        _LOGGER.info("Dry run mode - configuration is valid")
        print("[DRY_RUN] config_valid=true")
        return 0

    # Check JAX/GPU availability
    try:
        import jax
        devices = jax.devices()
        _LOGGER.info(f"JAX devices available: {devices}")
    except ImportError:
        _LOGGER.error("JAX not installed. Install with: pip install jax jaxlib")
        print("[ERROR] jax_not_installed")
        return 1

    # Run training
    try:
        print(f"[START] run_id={config.run_id}")
        sys.stdout.flush()

        runtime = MCTXWorkerRuntime(config)
        result = runtime.run()

        _LOGGER.info("Training completed successfully")
        return 0

    except KeyboardInterrupt:
        _LOGGER.warning("Training interrupted by user")
        print(f"[INTERRUPTED] run_id={config.run_id}")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        _LOGGER.error(f"Training failed: {e}", exc_info=True)
        print(f"[FAILED] run_id={config.run_id} error={e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
