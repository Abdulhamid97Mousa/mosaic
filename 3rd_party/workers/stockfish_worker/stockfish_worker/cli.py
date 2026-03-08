"""Command-line interface for Stockfish Worker.

Usage:
    stockfish-worker --config config.json
    stockfish-worker --difficulty medium --run-id test_run
    stockfish-worker --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import StockfishWorkerConfig, load_worker_config, DIFFICULTY_PRESETS
from .runtime import StockfishWorkerRuntime


def setup_logging(debug: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        stream=sys.stderr,  # Log to stderr, keep stdout for JSONL output
    )


def main() -> int:
    """Main entry point for stockfish-worker CLI.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        prog="stockfish-worker",
        description="Stockfish chess engine worker for MOSAIC operators",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Unique run identifier",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=list(DIFFICULTY_PRESETS.keys()),
        default="medium",
        help="Difficulty preset (default: medium)",
    )
    parser.add_argument(
        "--skill-level",
        type=int,
        help="Stockfish skill level (0-20, overrides difficulty)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Search depth (1-30, overrides difficulty)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1000,
        help="Time limit per move in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        help="Path to Stockfish binary (auto-detected if not provided)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (stdin/stdout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()
    setup_logging(debug=args.debug)
    logger = logging.getLogger("stockfish_worker.cli")

    try:
        # Load or build config
        if args.config:
            logger.info(f"Loading config from: {args.config}")
            config = load_worker_config(args.config)
            # Override with CLI args if provided
            if args.run_id:
                config.run_id = args.run_id
        else:
            # Build config from CLI args
            config = StockfishWorkerConfig(
                run_id=args.run_id,
                difficulty=args.difficulty,
                skill_level=args.skill_level,
                depth=args.depth,
                time_limit_ms=args.time_limit,
                stockfish_path=args.stockfish_path,
            )

        logger.info(f"Config: {config.to_dict()}")

        if args.dry_run:
            logger.info("Dry run - config validated successfully")
            print(json.dumps({"status": "dry-run", "config": config.to_dict()}))
            return 0

        # Create and run runtime
        runtime = StockfishWorkerRuntime(config)

        if args.interactive:
            logger.info("Starting interactive mode...")
            summary = runtime.run_interactive()
        else:
            # For non-interactive mode, just validate and exit
            logger.info("Non-interactive mode - validating setup...")
            with runtime:
                # Engine started successfully
                logger.info("Stockfish engine ready")
                print(json.dumps({
                    "status": "ready",
                    "config": config.to_dict(),
                    "stockfish_path": runtime._stockfish_path,
                }))
            summary = {"status": "completed"}

        logger.info(f"Completed: {summary.get('status', 'unknown')}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 2
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return 3
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
