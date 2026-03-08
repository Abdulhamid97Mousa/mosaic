"""Chess Worker CLI - Entry point for the chess worker."""

import argparse
import sys

from .config import ChessWorkerConfig
from .runtime import ChessWorkerRuntime


def main() -> int:
    """Main entry point for chess_worker CLI."""
    parser = argparse.ArgumentParser(
        description="Chess Worker - LLM-based chess player using llm_chess prompting style"
    )

    # Run identification
    parser.add_argument("--run-id", type=str, default="", help="Unique run identifier")

    # LLM settings
    parser.add_argument(
        "--client-name",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "anthropic"],
        help="LLM client to use",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base URL for vLLM or compatible API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional for local vLLM)",
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens in response",
    )

    # Chess-specific settings
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max invalid move attempts before fallback",
    )
    parser.add_argument(
        "--max-dialog-turns",
        type=int,
        default=10,
        help="Max conversation turns per move",
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
    config = ChessWorkerConfig(
        run_id=args.run_id,
        client_name=args.client_name,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_dialog_turns=args.max_dialog_turns,
        telemetry_dir=args.telemetry_dir,
    )

    # Create and run the worker
    runtime = ChessWorkerRuntime(config)
    runtime.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())
