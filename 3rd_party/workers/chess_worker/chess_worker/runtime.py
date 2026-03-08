"""Chess Worker Runtime - LLM-based chess player using llm_chess prompting style.

This worker implements the prompting strategy from llm_chess:
1. LLM can request: get_current_board, get_legal_moves, make_move <uci>
2. Multi-turn conversation with retry on invalid moves
3. Regex parsing of make_move pattern
"""

import json
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import ChessWorkerConfig

logger = logging.getLogger("chess_worker")


class ChessWorkerRuntime:
    """Chess Worker Runtime using llm_chess prompting style."""

    # Action patterns (from llm_chess)
    GET_BOARD_ACTION = "get_current_board"
    GET_LEGAL_MOVES_ACTION = "get_legal_moves"
    MAKE_MOVE_ACTION = "make_move"

    def __init__(self, config: ChessWorkerConfig):
        """Initialize the chess worker.

        Args:
            config: Worker configuration.
        """
        self.config = config
        self._client: Optional[OpenAI] = None
        self._conversation: List[Dict[str, str]] = []
        self._player_id: str = ""
        self._game_name: str = "chess_v6"

        # Stats
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._llm_calls = 0
        self._failed_attempts = 0

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _init_client(self) -> None:
        """Initialize the OpenAI-compatible client."""
        if self._client is not None:
            return

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "not-needed",
        )
        logger.info(f"Chess Worker v{self.config.run_id}")
        logger.info(f"LLM: {self.config.client_name} / {self.config.model_id}")
        logger.info(f"Base URL: {self.config.base_url}")

    def _get_system_prompt(self, player_color: str) -> str:
        """Get the system prompt for the chess player.

        Args:
            player_color: "white" or "black"

        Returns:
            System prompt string.
        """
        return (
            f"You are a professional chess player and you play as {player_color}. "
            "Now is your turn to make a move. Before making a move you can pick one of the following actions:\n"
            f"- '{self.GET_BOARD_ACTION}' to get the schema and current status of the board\n"
            f"- '{self.GET_LEGAL_MOVES_ACTION}' to get a UCI formatted list of available moves\n"
            f"- '{self.MAKE_MOVE_ACTION} <UCI formatted move>' when you are ready to complete your turn "
            f"(e.g., '{self.MAKE_MOVE_ACTION} e2e4')\n\n"
            "Always respond with ONLY one of the above actions. No explanations."
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM and get response.

        Args:
            messages: Conversation messages.

        Returns:
            LLM response text.
        """
        if self._client is None:
            self._init_client()

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            self._llm_calls += 1

            # Track tokens
            if hasattr(response, "usage") and response.usage:
                self._total_input_tokens += response.usage.prompt_tokens or 0
                self._total_output_tokens += response.usage.completion_tokens or 0

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_action(self, response: str) -> tuple[str, Optional[str]]:
        """Parse the LLM response to extract action.

        Args:
            response: LLM response text.

        Returns:
            Tuple of (action_type, move_uci or None)
        """
        response_lower = response.lower().strip()

        # Check for make_move action with UCI move
        # Pattern: make_move followed by 4-5 character UCI move
        match = re.search(
            rf"{self.MAKE_MOVE_ACTION}\s+([a-zA-Z0-9]{{4,5}})",
            response_lower,
        )
        if match:
            return self.MAKE_MOVE_ACTION, match.group(1)

        # Check for get_current_board
        if self.GET_BOARD_ACTION in response_lower:
            return self.GET_BOARD_ACTION, None

        # Check for get_legal_moves
        if self.GET_LEGAL_MOVES_ACTION in response_lower:
            return self.GET_LEGAL_MOVES_ACTION, None

        # Try to find a bare UCI move (4-5 chars like e2e4, g1f3)
        bare_match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", response_lower)
        if bare_match:
            return self.MAKE_MOVE_ACTION, bare_match.group(1)

        return "invalid", None

    def init_agent(self, game_name: str, player_id: str) -> None:
        """Initialize the agent for a game.

        Args:
            game_name: Name of the game (e.g., "chess_v6").
            player_id: Player identifier (e.g., "player_0").
        """
        self._init_client()
        self._game_name = game_name
        self._player_id = player_id
        self._conversation = []
        self._failed_attempts = 0

        # Determine color from player_id
        player_color = "white" if player_id == "player_0" else "black"

        # Initialize conversation with system prompt
        self._conversation = [
            {"role": "system", "content": self._get_system_prompt(player_color)}
        ]

        logger.info(f"Agent initialized for {game_name} as {player_id} ({player_color})")

    def select_action(
        self,
        observation: str,
        legal_moves: List[str],
        board_str: str,
    ) -> Dict[str, Any]:
        """Select an action using multi-turn LLM conversation.

        This implements the llm_chess prompting style:
        1. Show the board and legal moves
        2. LLM can request more info or make a move
        3. Validate move and retry if invalid

        Args:
            observation: Current game observation string.
            legal_moves: List of legal UCI moves.
            board_str: String representation of the board.

        Returns:
            Dict with action_str, action_index, reasoning, tokens, etc.
        """
        if not self._conversation:
            self.init_agent(self._game_name, self._player_id)

        # Build initial observation message
        obs_msg = (
            f"Current position:\n{board_str}\n\n"
            f"Legal moves: {', '.join(legal_moves)}\n\n"
            "What is your action?"
        )

        self._conversation.append({"role": "user", "content": obs_msg})

        # Multi-turn conversation loop
        for turn in range(self.config.max_dialog_turns):
            # Get LLM response
            try:
                response = self._call_llm(self._conversation)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                # Return random move on error
                import random
                fallback = random.choice(legal_moves) if legal_moves else "e2e4"
                return {
                    "action_str": fallback,
                    "action_index": None,
                    "reasoning": f"LLM error: {e}",
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                    "success": False,
                }

            logger.debug(f"LLM response (turn {turn + 1}): {response}")

            # Add response to conversation
            self._conversation.append({"role": "assistant", "content": response})

            # Parse the action
            action_type, move_uci = self._parse_action(response)

            if action_type == self.GET_BOARD_ACTION:
                # Provide board state
                reply = f"Current board:\n{board_str}"
                self._conversation.append({"role": "user", "content": reply})
                continue

            elif action_type == self.GET_LEGAL_MOVES_ACTION:
                # Provide legal moves
                reply = f"Legal moves: {', '.join(legal_moves)}"
                self._conversation.append({"role": "user", "content": reply})
                continue

            elif action_type == self.MAKE_MOVE_ACTION and move_uci:
                # Validate move
                if move_uci in legal_moves:
                    logger.info(f"Valid move selected: {move_uci}")
                    return {
                        "action_str": move_uci,
                        "action_index": None,
                        "reasoning": response,
                        "input_tokens": self._total_input_tokens,
                        "output_tokens": self._total_output_tokens,
                        "success": True,
                    }
                else:
                    # Invalid move - provide feedback and retry
                    self._failed_attempts += 1
                    if self._failed_attempts >= self.config.max_retries:
                        logger.warning(f"Max retries reached, using random move")
                        import random
                        fallback = random.choice(legal_moves) if legal_moves else "e2e4"
                        return {
                            "action_str": fallback,
                            "action_index": None,
                            "reasoning": f"Max retries, last attempt: {response}",
                            "input_tokens": self._total_input_tokens,
                            "output_tokens": self._total_output_tokens,
                            "success": False,
                        }

                    reply = (
                        f"Invalid move '{move_uci}'. That move is not legal.\n"
                        f"Legal moves are: {', '.join(legal_moves)}\n"
                        "Please choose a valid move."
                    )
                    self._conversation.append({"role": "user", "content": reply})
                    continue

            else:
                # Invalid action format
                self._failed_attempts += 1
                if self._failed_attempts >= self.config.max_retries:
                    logger.warning(f"Max retries reached, using random move")
                    import random
                    fallback = random.choice(legal_moves) if legal_moves else "e2e4"
                    return {
                        "action_str": fallback,
                        "action_index": None,
                        "reasoning": f"Invalid format, last: {response}",
                        "input_tokens": self._total_input_tokens,
                        "output_tokens": self._total_output_tokens,
                        "success": False,
                    }

                reply = (
                    f"Invalid action. Please respond with one of:\n"
                    f"- '{self.GET_BOARD_ACTION}' to see the board\n"
                    f"- '{self.GET_LEGAL_MOVES_ACTION}' to see legal moves\n"
                    f"- '{self.MAKE_MOVE_ACTION} <move>' to make a move (e.g., 'make_move e2e4')"
                )
                self._conversation.append({"role": "user", "content": reply})
                continue

        # Max turns reached
        logger.warning("Max dialog turns reached, using random move")
        import random
        fallback = random.choice(legal_moves) if legal_moves else "e2e4"
        return {
            "action_str": fallback,
            "action_index": None,
            "reasoning": "Max turns reached",
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "success": False,
        }

    def run_interactive(self) -> None:
        """Run in interactive mode, reading commands from stdin."""
        logger.info("Interactive mode started. Waiting for commands on stdin...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
                cmd_type = cmd.get("type", "")

                if cmd_type == "init_agent":
                    game_name = cmd.get("game_name", "chess_v6")
                    player_id = cmd.get("player_id", "player_0")
                    self.init_agent(game_name, player_id)
                    self._emit({
                        "type": "agent_initialized",
                        "run_id": self.config.run_id,
                        "game_name": game_name,
                        "player_id": player_id,
                    })

                elif cmd_type == "select_action":
                    observation = cmd.get("observation", "")
                    info = cmd.get("info", {})
                    legal_moves = info.get("legal_moves", [])
                    player_id = cmd.get("player_id", self._player_id)

                    # Extract board from observation
                    board_str = observation

                    result = self.select_action(observation, legal_moves, board_str)

                    self._emit({
                        "type": "action_selected",
                        "run_id": self.config.run_id,
                        "player_id": player_id,
                        "action": result["action_str"],
                        "action_str": result["action_str"],
                        "reasoning": result.get("reasoning", ""),
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "success": result.get("success", True),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif cmd_type == "stop":
                    logger.info("Stop command received, exiting")
                    self._emit({"type": "stopped", "run_id": self.config.run_id})
                    break

                else:
                    logger.warning(f"Unknown command type: {cmd_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.exception(f"Command processing error: {e}")
                self._emit({"type": "error", "message": str(e)})

    def _emit(self, data: Dict[str, Any]) -> None:
        """Emit a response to stdout."""
        print(json.dumps(data), flush=True)
