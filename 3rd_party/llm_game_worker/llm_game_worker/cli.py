"""Command-line interface for LLM Game Worker."""

import argparse
import logging
import sys
from typing import Optional

from .config import LLMGameWorkerConfig, SupportedGame, PlayMode
from .runtime import LLMGameWorkerRuntime


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for LLM Game Worker CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="LLM Game Worker - LLM-based player for PettingZoo classic games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported games:
  tictactoe_v3    Tic-Tac-Toe (3x3 grid, positions 0-8)
  connect_four_v3 Connect Four (7 columns, drop to columns 0-6)
  go_v5           Go (19x19 board, coordinates or 'pass')

Play modes:
  self_play     LLM vs LLM (both players controlled by LLM)
  human_vs_ai   Human plays first (player_1), AI plays second (player_2)
  ai_vs_human   AI plays first (player_1), Human plays second (player_2)
  ai_only       AI controls only the specified player (use --play-as)

Examples:
  %(prog)s --task tictactoe_v3
  %(prog)s --task tictactoe_v3 --play-mode human_vs_ai
  %(prog)s --task connect_four_v3 --play-as player_1 --model-id gpt-4
  %(prog)s --task go_v5 --board-size 9 --play-mode self_play
        """,
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Unique run identifier",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="tictactoe_v3",
        choices=[g.value for g in SupportedGame],
        help="Game to play (default: tictactoe_v3)",
    )

    parser.add_argument(
        "--play-mode",
        type=str,
        default="self_play",
        choices=[m.value for m in PlayMode],
        help="Play mode: self_play, human_vs_ai, ai_vs_human, ai_only (default: self_play)",
    )

    parser.add_argument(
        "--play-as",
        type=str,
        default="both",
        help="Which player LLM controls: player_1, player_2, or both (default: both)",
    )

    parser.add_argument(
        "--client-name",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "anthropic"],
        help="LLM client type (default: vllm)",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model identifier (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base URL (default: http://127.0.0.1:8000/v1)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional for local vLLM)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens in response (default: 256)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max invalid move attempts (default: 3)",
    )

    parser.add_argument(
        "--max-dialog-turns",
        type=int,
        default=10,
        help="Max conversation turns per move (default: 10)",
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=19,
        help="Board size for Go (default: 19)",
    )

    parser.add_argument(
        "--komi",
        type=float,
        default=7.5,
        help="Komi value for Go (default: 7.5)",
    )

    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="var/telemetry",
        help="Telemetry output directory",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parsed = parser.parse_args(args)

    # Setup logging
    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    # Create config
    config = LLMGameWorkerConfig(
        run_id=parsed.run_id,
        task=parsed.task,
        play_mode=parsed.play_mode,
        play_as=parsed.play_as,
        client_name=parsed.client_name,
        model_id=parsed.model_id,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        temperature=parsed.temperature,
        max_tokens=parsed.max_tokens,
        max_retries=parsed.max_retries,
        max_dialog_turns=parsed.max_dialog_turns,
        board_size=parsed.board_size,
        komi=parsed.komi,
        telemetry_dir=parsed.telemetry_dir,
    )

    # Create and run worker
    runtime = LLMGameWorkerRuntime(config)
    runtime.run_interactive()

    return 0


if __name__ == "__main__":
    sys.exit(main())
