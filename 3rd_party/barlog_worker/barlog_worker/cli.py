"""Command-line interface for BARLOG Worker.

Entry point: barlog-worker

Example usage:
    # Using OpenRouter (default, supports all major models)
    export OPENROUTER_API_KEY=sk-or-...
    barlog-worker --run-id test123 --env babyai --task BabyAI-GoToRedBall-v0

    # Using specific model via OpenRouter
    barlog-worker --run-id test123 --model anthropic/claude-3.5-sonnet

    # Using local vLLM
    barlog-worker --run-id test123 --client vllm --model meta-llama/Llama-3.1-8B-Instruct

    # Load from config file
    barlog-worker --config config.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from barlog_worker import __version__
from barlog_worker.config import (
    AGENT_TYPES,
    CLIENT_NAMES,
    ENV_NAMES,
    BarlogWorkerConfig,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the worker."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for barlog-worker CLI."""
    parser = argparse.ArgumentParser(
        prog="barlog-worker",
        description="BARLOG Worker - Run LLM agents on BALROG benchmark environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI on BabyAI
  barlog-worker --run-id test123 --env babyai --task BabyAI-GoToRedBall-v0

  # Run with Claude on MiniHack
  barlog-worker --run-id test456 --env minihack --task MiniHack-Room-5x5-v0 \\
      --client anthropic --model claude-3-5-sonnet-20241022

  # Load config from file
  barlog-worker --config config.json

  # Use chain-of-thought reasoning
  barlog-worker --run-id test789 --env crafter --agent-type cot
        """,
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"barlog-worker {__version__}",
    )

    # Config file (mutually exclusive with individual args)
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to JSON config file (overrides other arguments)",
    )

    # Required arguments
    parser.add_argument(
        "--run-id",
        type=str,
        required=False,  # Required unless --config is provided
        help="Unique run identifier (from GUI)",
    )

    # Environment settings
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--env",
        type=str,
        choices=ENV_NAMES,
        default="babyai",
        dest="env_name",
        help="Environment to use (default: babyai)",
    )
    env_group.add_argument(
        "--task",
        type=str,
        default="BabyAI-GoToRedBall-v0",
        help="Task/level within the environment (default: BabyAI-GoToRedBall-v0)",
    )
    env_group.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)",
    )
    env_group.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    env_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # LLM client settings
    llm_group = parser.add_argument_group("LLM Client")
    llm_group.add_argument(
        "--client",
        type=str,
        choices=CLIENT_NAMES,
        default="openrouter",
        dest="client_name",
        help="LLM client to use (default: openrouter). OpenRouter provides unified access to all major models.",
    )
    llm_group.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        dest="model_id",
        help="Model identifier (default: openai/gpt-4o-mini). For OpenRouter use format: provider/model",
    )
    llm_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to OPENROUTER_API_KEY, OPENAI_API_KEY, etc.)",
    )
    llm_group.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for API (OpenRouter: https://openrouter.ai/api/v1)",
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    llm_group.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )

    # Agent settings
    agent_group = parser.add_argument_group("Agent")
    agent_group.add_argument(
        "--agent-type",
        type=str,
        choices=AGENT_TYPES,
        default="naive",
        help="Agent reasoning strategy (default: naive)",
    )
    agent_group.add_argument(
        "--max-image-history",
        type=int,
        default=0,
        help="Max images in history. 0=text-only (default), >=1=VLM mode. "
             "Use 0 for text-only models like Qwen2.5. Use >=1 for multimodal models like GPT-4V.",
    )

    # Output settings
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--telemetry-dir",
        type=str,
        default="./telemetry",
        help="Directory for telemetry output (default: ./telemetry)",
    )
    output_group.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable JSONL telemetry output",
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    # Interactive mode for GUI step-by-step control
    output_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode - read commands from stdin, emit telemetry to stdout. "
             "Enables step-by-step control from GUI for scientific comparison.",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> BarlogWorkerConfig:
    """Build BarlogWorkerConfig from parsed arguments."""
    # If config file provided, load it
    if args.config:
        config = BarlogWorkerConfig.from_json_file(args.config)
        # Override run_id if provided on command line
        if args.run_id:
            config = BarlogWorkerConfig(
                run_id=args.run_id,
                **{k: v for k, v in config.to_dict().items() if k != "run_id" and k != "api_key"}
            )
        return config

    # Otherwise build from CLI args
    if not args.run_id:
        raise ValueError("--run-id is required when not using --config")

    return BarlogWorkerConfig(
        run_id=args.run_id,
        env_name=args.env_name,
        task=args.task,
        client_name=args.client_name,
        model_id=args.model_id,
        agent_type=args.agent_type,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        telemetry_dir=args.telemetry_dir,
        emit_jsonl=not args.no_jsonl,
        seed=args.seed,
        timeout=args.timeout,
        max_image_history=args.max_image_history,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for barlog-worker CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("barlog_worker")

    try:
        config = build_config_from_args(args)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return 1

    logger.info(f"BARLOG Worker v{__version__}")
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Environment: {config.env_name} / {config.task}")
    logger.info(f"LLM: {config.client_name} / {config.model_id}")
    logger.info(f"Agent: {config.agent_type}")
    logger.info(f"Episodes: {config.num_episodes}, Max Steps: {config.max_steps}")

    try:
        # Lazy import to avoid dependency issues when just parsing args
        if args.interactive:
            # Interactive mode for GUI step-by-step control
            from barlog_worker.runtime import InteractiveRuntime

            runtime = InteractiveRuntime(config)
            runtime.run()
            logger.info("Interactive mode completed")
        else:
            # Autonomous mode - run all episodes
            from barlog_worker.runtime import BarlogWorkerRuntime

            runtime = BarlogWorkerRuntime(config)
            runtime.run()
            logger.info("BARLOG Worker completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"BARLOG Worker failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
