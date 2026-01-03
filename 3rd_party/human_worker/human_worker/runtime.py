"""Human Worker Runtime - Wait for human input via GUI.

This worker implements human-in-the-loop action selection:
1. GUI sends select_action when it's the human's turn
2. Worker emits 'waiting_for_human' signal
3. GUI enables click-on-board and waits for human click
4. GUI sends 'human_input' with the selected move
5. Worker validates and returns the action

Protocol:
    GUI -> Worker: {"cmd": "select_action", "info": {"legal_moves": [...]}}
    Worker -> GUI: {"type": "waiting_for_human", "legal_moves": [...]}
    (Human clicks on board)
    GUI -> Worker: {"cmd": "human_input", "move": "e7e5"}
    Worker -> GUI: {"type": "action_selected", "action_str": "e7e5", "success": true}
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import HumanWorkerConfig

logger = logging.getLogger("human_worker")


class HumanWorkerRuntime:
    """Human Worker Runtime - waits for human input via GUI."""

    def __init__(self, config: HumanWorkerConfig):
        """Initialize the human worker.

        Args:
            config: Worker configuration.
        """
        self.config = config
        self._player_id: str = ""
        self._game_name: str = ""
        self._waiting_for_input: bool = False
        self._current_legal_moves: List[str] = []

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def init_agent(self, game_name: str, player_id: str) -> None:
        """Initialize the agent for a game.

        Args:
            game_name: Name of the game (e.g., "chess_v6").
            player_id: Player identifier (e.g., "player_0").
        """
        self._game_name = game_name
        self._player_id = player_id
        self._waiting_for_input = False
        self._current_legal_moves = []

        logger.info(
            f"Human agent initialized for {game_name} as {player_id} "
            f"(player: {self.config.player_name})"
        )

    def request_human_input(
        self,
        observation: str,
        legal_moves: List[str],
        board_str: str,
    ) -> None:
        """Signal that we're waiting for human input.

        This emits a 'waiting_for_human' message to the GUI, which should:
        1. Enable click-on-board for move selection
        2. Optionally highlight legal moves
        3. Wait for human to click and send 'human_input' command

        Args:
            observation: Current game observation string.
            legal_moves: List of legal moves (UCI for chess).
            board_str: String representation of the board.
        """
        self._waiting_for_input = True
        self._current_legal_moves = legal_moves

        # Emit signal to GUI
        self._emit({
            "type": "waiting_for_human",
            "run_id": self.config.run_id,
            "player_id": self._player_id,
            "player_name": self.config.player_name,
            "legal_moves": legal_moves,
            "show_legal_moves": self.config.show_legal_moves,
            "confirm_moves": self.config.confirm_moves,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(f"Waiting for human input ({len(legal_moves)} legal moves)")

    def process_human_input(self, move: str) -> Dict[str, Any]:
        """Process human input and validate the move.

        Args:
            move: The move selected by the human (e.g., "e7e5").

        Returns:
            Dict with action_str, success, and optional error message.
        """
        if not self._waiting_for_input:
            logger.warning("Received human input but not waiting for input")
            return {
                "action_str": move,
                "success": False,
                "error": "Not waiting for input",
            }

        # Validate move against legal moves
        if move not in self._current_legal_moves:
            logger.warning(f"Invalid move '{move}' - not in legal moves")
            return {
                "action_str": move,
                "success": False,
                "error": f"Invalid move '{move}'. Legal moves: {', '.join(self._current_legal_moves[:10])}...",
            }

        self._waiting_for_input = False
        logger.info(f"Human selected move: {move}")

        return {
            "action_str": move,
            "success": True,
        }

    def run_interactive(self) -> None:
        """Run in interactive mode, reading commands from stdin.

        Protocol:
            - init_agent: Initialize for a game/player
            - select_action: Start waiting for human input
            - human_input: Receive the human's move selection
            - stop: Terminate gracefully
        """
        logger.info(
            f"Human Worker started for {self.config.player_name}. "
            "Waiting for commands on stdin..."
        )

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
                cmd_type = cmd.get("cmd") or cmd.get("type", "")

                if cmd_type == "init_agent":
                    game_name = cmd.get("game_name", "chess_v6")
                    player_id = cmd.get("player_id", "player_0")
                    self.init_agent(game_name, player_id)
                    self._emit({
                        "type": "agent_initialized",
                        "run_id": self.config.run_id,
                        "game_name": game_name,
                        "player_id": player_id,
                        "player_name": self.config.player_name,
                    })

                elif cmd_type == "select_action":
                    # GUI is asking us to select an action
                    # We emit 'waiting_for_human' and wait for 'human_input'
                    observation = cmd.get("observation", "")
                    info = cmd.get("info", {})
                    legal_moves = info.get("legal_moves", [])

                    self.request_human_input(observation, legal_moves, observation)
                    # Don't emit action_selected yet - wait for human_input

                elif cmd_type == "human_input":
                    # Human has made their selection via GUI
                    move = cmd.get("move", "")
                    player_id = cmd.get("player_id", self._player_id)

                    result = self.process_human_input(move)

                    self._emit({
                        "type": "action_selected",
                        "run_id": self.config.run_id,
                        "player_id": player_id,
                        "action": result["action_str"],
                        "action_str": result["action_str"],
                        "success": result["success"],
                        "error": result.get("error", ""),
                        "source": "human",
                        "player_name": self.config.player_name,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif cmd_type == "cancel_input":
                    # GUI cancelled the human input request
                    self._waiting_for_input = False
                    logger.info("Human input cancelled")
                    self._emit({
                        "type": "input_cancelled",
                        "run_id": self.config.run_id,
                        "player_id": self._player_id,
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
