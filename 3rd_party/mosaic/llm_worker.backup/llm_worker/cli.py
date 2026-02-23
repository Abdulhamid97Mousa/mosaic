"""MOSAIC LLM Worker CLI - Command Line Interface.

Entry point for the mosaic-llm-worker command.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from .config import LLMWorkerConfig, load_worker_config
from .runtime import LLMWorkerRuntime, InteractiveLLMRuntime


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr, telemetry to stdout
    )


def main() -> int:
    """Main entry point for mosaic-llm-worker CLI."""
    parser = argparse.ArgumentParser(
        description="MOSAIC LLM Worker - Multi-agent LLM coordination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run autonomous mode with config file
  mosaic-llm-worker --config config.json

  # Run with CLI arguments
  mosaic-llm-worker --task MultiGrid-Soccer-v0 --model anthropic/claude-3.5-sonnet

  # Interactive mode for GUI
  mosaic-llm-worker --interactive --task MultiGrid-Soccer-v0
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )

    # Mode selection
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (JSON stdin/stdout for GUI)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing",
    )

    # Run identification
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Unique run identifier (auto-generated if not provided)",
    )

    # Environment settings
    parser.add_argument(
        "--env-name",
        type=str,
        default="multigrid",
        help="Environment family (multigrid, pettingzoo, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="MultiGrid-Soccer-v0",
        help="Specific environment task",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Number of agents (default: 4 for Soccer)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment",
    )

    # LLM settings
    parser.add_argument(
        "--client",
        type=str,
        default="openrouter",
        choices=["openrouter", "openai", "anthropic", "vllm"],
        help="LLM client to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-sonnet",
        help="Model identifier (e.g., anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (0.0-2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Custom API base URL (for vLLM, etc.)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider (or use env var)",
    )

    # Multi-agent coordination
    parser.add_argument(
        "--coordination-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Coordination strategy: 1=emergent, 2=hints, 3=role-based",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="egocentric",
        choices=["egocentric", "visible_teammates"],
        help="Observation text mode",
    )
    parser.add_argument(
        "--agent-roles",
        type=str,
        default=None,
        help="Comma-separated agent roles (e.g., 'forward,defender,forward,defender')",
    )

    # Execution settings
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode",
    )

    # Rendering settings
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["human", "rgb_array"],
        help="Rendering mode for environment visualization",
    )

    # Telemetry settings
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="var/telemetry",
        help="Directory for telemetry output",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable JSONL file output",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Disable stdout telemetry output",
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config from file or build from args
    if args.config:
        try:
            config = load_worker_config(args.config)
            logger.info(f"Loaded config from: {args.config}")
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
    else:
        # Build config from CLI args
        run_id = args.run_id or f"llm-{uuid.uuid4().hex[:8]}"

        # Parse agent roles if provided
        agent_roles = None
        if args.agent_roles:
            agent_roles = [r.strip() for r in args.agent_roles.split(",")]

        config = LLMWorkerConfig(
            run_id=run_id,
            seed=args.seed,
            env_name=args.env_name,
            task=args.task,
            num_agents=args.num_agents,
            client_name=args.client,
            model_id=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            coordination_level=args.coordination_level,
            observation_mode=args.observation_mode,
            agent_roles=agent_roles,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            render_mode=args.render_mode,
            telemetry_dir=args.telemetry_dir,
            emit_jsonl=not args.no_jsonl,
            emit_stdout=not args.no_stdout,
        )

    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Coordination Level: {config.coordination_level}")

    # Run in appropriate mode
    try:
        if args.interactive:
            logger.info("Starting interactive mode...")
            runtime = InteractiveLLMRuntime(config)
            runtime.run()
            return 0
        else:
            logger.info("Starting autonomous mode...")
            runtime = LLMWorkerRuntime(config, dry_run=args.dry_run)
            result = runtime.run()

            if result["status"] == "completed":
                logger.info(
                    f"Completed {config.num_episodes} episodes | "
                    f"Total reward: {result.get('total_reward', 0):.2f}"
                )
                return 0
            elif result["status"] == "dry-run":
                logger.info("Dry-run completed successfully")
                return 0
            else:
                logger.error(f"Run failed: {result.get('error', 'Unknown error')}")
                return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
