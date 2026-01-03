"""Human Worker CLI - Entry point for the human worker."""

import argparse
import sys

from .config import HumanWorkerConfig
from .runtime import HumanWorkerRuntime


def main() -> int:
    """Main entry point for human_worker CLI."""
    parser = argparse.ArgumentParser(
        description="Human Worker - Human-in-the-loop action selection via GUI"
    )

    # Run identification
    parser.add_argument("--run-id", type=str, default="", help="Unique run identifier")

    # Player settings
    parser.add_argument(
        "--player-name",
        type=str,
        default="Human",
        help="Human player's display name",
    )

    # Timeout settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Timeout for human input in seconds (0 = no timeout)",
    )

    # UI hints
    parser.add_argument(
        "--show-legal-moves",
        action="store_true",
        default=True,
        help="Highlight legal moves in UI",
    )
    parser.add_argument(
        "--no-show-legal-moves",
        action="store_false",
        dest="show_legal_moves",
        help="Don't highlight legal moves in UI",
    )
    parser.add_argument(
        "--confirm-moves",
        action="store_true",
        default=False,
        help="Require move confirmation before submitting",
    )

    # Telemetry
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="var/telemetry",
        help="Directory for telemetry output",
    )

    args = parser.parse_args()

    # Create config from args
    config = HumanWorkerConfig(
        run_id=args.run_id,
        player_name=args.player_name,
        timeout_seconds=args.timeout,
        show_legal_moves=args.show_legal_moves,
        confirm_moves=args.confirm_moves,
        telemetry_dir=args.telemetry_dir,
    )

    # Create and run the worker
    runtime = HumanWorkerRuntime(config)
    runtime.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())
