"""Presenter for Chess LLM Worker.

This presenter handles:
1. Building training configurations for chess LLM agent
2. Creating worker-specific UI tabs for chess game visualization
3. Configuring LLM settings (vLLM, OpenAI, Anthropic)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class ChessWorkerPresenter:
    """Presenter for Chess LLM Worker.

    Handles configuration building for LLM-based chess players using
    the llm_chess prompting style with multi-turn conversation.

    Features:
    - Multi-turn conversation: LLM can request board state, legal moves
    - Regex validation with retry on invalid moves
    - Supports OpenAI, Anthropic, and local vLLM backends
    """

    @property
    def id(self) -> str:
        """Return unique identifier for this presenter."""
        return "chess_worker"

    def build_train_request(
        self,
        policy_path: Any,
        current_game: Optional[Any],
    ) -> dict:
        """Build a training request for chess LLM evaluation.

        Note: Chess worker does not support training (LLM inference only).
        This builds an evaluation/play request.

        Args:
            policy_path: Path to the LLM configuration file
            current_game: Currently selected game (should be chess)

        Returns:
            dict: Configuration dictionary for TrainerClient submission
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"chess-llm-{timestamp}"

        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "chess_worker.cli"],
            "environment": {
                "PETTINGZOO_ENV_ID": "chess_v6",
                "PETTINGZOO_FAMILY": "classic",
                "PETTINGZOO_API_TYPE": "aec",
            },
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
            "metadata": {
                "ui": {
                    "worker_id": self.id,
                    "env_id": "chess_v6",
                    "family": "classic",
                    "is_parallel": False,
                    "mode": "evaluation",
                },
                "worker": {
                    "module": "chess_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "policy_path": str(policy_path) if policy_path else None,
                        "render": True,
                    },
                },
            },
        }

        _LOGGER.info(
            "Built Chess LLM request: run=%s",
            run_name,
        )

        return config

    def build_llm_config(
        self,
        client_name: str = "vllm",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
        max_retries: int = 3,
        max_dialog_turns: int = 10,
        **kwargs: Any,
    ) -> dict:
        """Build LLM configuration for the chess worker.

        Args:
            client_name: LLM client (vllm, openai, anthropic)
            model_id: Model identifier
            base_url: API base URL for vLLM or compatible API
            api_key: API key (optional for local vLLM)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Max invalid move attempts before fallback
            max_dialog_turns: Max conversation turns per move
            **kwargs: Additional configuration options

        Returns:
            dict: LLM configuration dictionary
        """
        return {
            "client_name": client_name,
            "model_id": model_id,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "max_dialog_turns": max_dialog_turns,
            **kwargs,
        }

    def create_tabs(
        self,
        run_id: str,
        agent_id: str,
        first_payload: dict,
        parent: Any,
    ) -> List[Any]:
        """Create worker-specific UI tabs for a running chess session.

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier (player_0 or player_1)
            first_payload: First telemetry payload containing metadata
            parent: Parent Qt widget

        Returns:
            list: List of QWidget tab instances (empty for now)
        """
        # Future tabs could include:
        # - Move history with notation
        # - LLM reasoning/conversation log
        # - Position evaluation
        # - Time per move statistics

        _LOGGER.debug(
            "Creating Chess worker tabs for run=%s, agent=%s",
            run_id,
            agent_id,
        )

        return []

    def extract_game_metadata(self, fen: str) -> Dict[str, Any]:
        """Extract game metadata from FEN string.

        Args:
            fen: FEN notation of the current position

        Returns:
            dict: Game metadata including position info
        """
        parts = fen.split()
        if len(parts) >= 6:
            return {
                "fen": fen,
                "active_color": "white" if parts[1] == "w" else "black",
                "castling": parts[2],
                "en_passant": parts[3],
                "halfmove_clock": int(parts[4]) if parts[4].isdigit() else 0,
                "fullmove_number": int(parts[5]) if parts[5].isdigit() else 1,
            }
        return {"fen": fen}


__all__ = ["ChessWorkerPresenter"]
