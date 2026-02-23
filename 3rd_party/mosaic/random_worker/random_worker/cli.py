"""CLI entry point for Random Worker.

Usage (autonomous — env-owning, for script experiments):
    random-worker --run-id test123 --task MiniGrid-Empty-8x8-v0

Usage (interactive — action-selector, for manual multi-agent mode):
    random-worker --run-id test123 --interactive

    python -m random_worker --run-id test123 --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys

from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Random action-selector worker for MOSAIC multi-agent environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--env-name", default="", help="Environment family")
    parser.add_argument("--task", default="", help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--behavior", default="random",
                        choices=("random", "noop", "cycling"),
                        help="Action selection strategy (default: random)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode (action-selector protocol)")
    parser.add_argument("--telemetry-dir", default=None, help="Telemetry output directory (unused)")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Configure logging to stderr (stdout is reserved for JSON protocol)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    config = RandomWorkerConfig(
        run_id=args.run_id,
        env_name=args.env_name,
        task=args.task,
        seed=args.seed,
        behavior=args.behavior,
    )

    try:
        runtime = RandomWorkerRuntime(config)

        if args.interactive:
            runtime.run()
        else:
            runtime.run_autonomous()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
