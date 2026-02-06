"""Command-line interface for operators worker.

This is the entry point for launching operator worker subprocesses.

Usage:
    python -m operators_worker.cli \
        --run-id operator_0_abc123 \
        --behavior random \
        --env-name babyai \
        --task BabyAI-GoToRedBall-v0 \
        --interactive

Examples:
    # Random baseline operator
    python -m operators_worker.cli --run-id op_rand_001 --behavior random --task BabyAI-GoToRedBall-v0

    # No-op baseline operator
    python -m operators_worker.cli --run-id op_noop_001 --behavior noop --task MultiGrid-Empty-8x8-v0

    # Cycling baseline operator
    python -m operators_worker.cli --run-id op_cycle_001 --behavior cycling --task MiniGrid-Empty-8x8-v0
"""

import argparse
import sys
from operators_worker.config import OperatorsWorkerConfig
from operators_worker.runtime import OperatorsWorkerRuntime


def parse_args(argv=None):
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Operators Worker - Baseline operators for credit assignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random baseline operator
  %(prog)s --run-id op_rand_001 --behavior random --task BabyAI-GoToRedBall-v0

  # No-op baseline operator
  %(prog)s --run-id op_noop_001 --behavior noop --task MultiGrid-Empty-8x8-v0

  # Cycling baseline operator with seed
  %(prog)s --run-id op_cycle_001 --behavior cycling --task MiniGrid-Empty-8x8-v0 --seed 42

Interactive Mode:
  When --interactive is set, the worker reads JSON commands from stdin:
    {"cmd": "reset", "seed": 42, "env_name": "babyai", "task": "BabyAI-GoToRedBall-v0"}
    {"cmd": "step"}
    {"cmd": "stop"}

  Responses are emitted to stdout as JSON:
    {"type": "init"}
    {"type": "ready", "render_payload": {...}}
    {"type": "step", "reward": 0.0, "terminated": false, "render_payload": {...}}
        """
    )

    # Required arguments
    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique identifier for this worker run (e.g., operator_0_abc123)",
    )

    # Operator configuration
    parser.add_argument(
        "--behavior",
        default="random",
        choices=["random", "noop", "cycling"],
        help="Operator behavior (default: random)",
    )

    # Environment configuration
    parser.add_argument(
        "--env-name",
        default="babyai",
        help="Environment family (default: babyai)",
    )

    parser.add_argument(
        "--task",
        default="BabyAI-GoToRedBall-v0",
        help="Specific environment task (default: BabyAI-GoToRedBall-v0)",
    )

    # Telemetry configuration
    parser.add_argument(
        "--telemetry-dir",
        default=None,
        help="Directory to write JSONL telemetry files (default: auto-resolved)",
    )

    parser.add_argument(
        "--no-emit-jsonl",
        action="store_true",
        help="Disable JSONL telemetry emission",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (read commands from stdin)",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for operators worker CLI.

    Args:
        argv: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args(argv)

    # Create configuration
    config = OperatorsWorkerConfig(
        run_id=args.run_id,
        behavior=args.behavior,
        env_name=args.env_name,
        task=args.task,
        telemetry_dir=args.telemetry_dir,
        emit_jsonl=not args.no_emit_jsonl,
        seed=args.seed,
        interactive=args.interactive,
    )

    try:
        # Create runtime
        runtime = OperatorsWorkerRuntime(config)

        if config.interactive:
            # Run interactive mode (stdin/stdout protocol)
            runtime.run_interactive()
        else:
            # Non-interactive mode: just run a single episode
            # (Useful for testing)
            print(f"Non-interactive mode not fully implemented yet.", file=sys.stderr)
            print(f"Use --interactive for subprocess mode.", file=sys.stderr)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
