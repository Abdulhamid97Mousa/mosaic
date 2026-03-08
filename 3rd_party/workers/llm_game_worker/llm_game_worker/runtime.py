"""LLM Game Worker Runtime - LLM-based player for PettingZoo classic games.

This worker implements multi-turn conversation prompting for various board games:
1. LLM can request: get_board, get_legal_moves, make_move <action>
2. Multi-turn conversation with retry on invalid moves
3. Game-specific board representations and action formats

Supported games:
- Tic-Tac-Toe: 3x3 grid, actions are positions 0-8
- Connect Four: 7 columns, actions are column numbers 0-6
- Go: 19x19 board (configurable), actions are (row, col) coordinates or "pass"
"""

import json
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import LLMGameWorkerConfig, SupportedGame, AgentType

logger = logging.getLogger("llm_game_worker")


class LLMGameWorkerRuntime:
    """LLM Game Worker Runtime for PettingZoo classic games."""

    # Action patterns
    GET_BOARD_ACTION = "get_board"
    GET_LEGAL_MOVES_ACTION = "get_legal_moves"
    MAKE_MOVE_ACTION = "make_move"

    def __init__(self, config: LLMGameWorkerConfig):
        """Initialize the game worker.

        Args:
            config: Worker configuration.
        """
        self.config = config
        self._client: Optional[OpenAI] = None
        self._conversation: List[Dict[str, str]] = []
        self._player_id: str = ""
        self._game_name: str = config.task

        # Stats
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._llm_calls = 0
        self._failed_attempts = 0

        # RL Policy cache (loaded on demand)
        self._rl_policies: Dict[str, Any] = {}

        # Pending human action (for async human input)
        self._pending_human_action: Optional[int] = None

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        if not logger.handlers:
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
        logger.info(f"LLM Game Worker v{self.config.run_id}")
        logger.info(f"Game: {self.config.task}")
        logger.info(f"LLM: {self.config.client_name} / {self.config.model_id}")
        logger.info(f"Base URL: {self.config.base_url}")

    def _get_game_description(self) -> str:
        """Get game-specific description for the system prompt."""
        game_type = self.config.game_type

        if game_type == SupportedGame.TIC_TAC_TOE:
            return (
                "Tic-Tac-Toe is a two-player game on a 3x3 grid. "
                "Players take turns placing their mark (X or O) in empty cells. "
                "The goal is to get three marks in a row (horizontally, vertically, or diagonally). "
                "Positions are numbered 0-8:\n"
                " 0 | 1 | 2\n"
                "-----------\n"
                " 3 | 4 | 5\n"
                "-----------\n"
                " 6 | 7 | 8"
            )

        elif game_type == SupportedGame.CONNECT_FOUR:
            return (
                "Connect Four is a two-player game on a 6x7 vertical grid. "
                "Players take turns dropping discs into columns. "
                "Discs fall to the lowest available row in the chosen column. "
                "The goal is to get four discs in a row (horizontally, vertically, or diagonally). "
                "Columns are numbered 0-6 from left to right."
            )

        elif game_type == SupportedGame.GO:
            return (
                f"Go is a two-player strategy game on a {self.config.board_size}x{self.config.board_size} board. "
                "Players take turns placing stones (Black or White) on intersections. "
                "The goal is to control more territory than your opponent. "
                "Stones are captured when surrounded. "
                f"Actions are given as coordinates (row, column) from 0-{self.config.board_size - 1}, "
                "or 'pass' to pass your turn."
            )

        return "Unknown game."

    def _get_action_format(self) -> str:
        """Get game-specific action format description."""
        game_type = self.config.game_type

        if game_type == SupportedGame.TIC_TAC_TOE:
            return f"'{self.MAKE_MOVE_ACTION} <position>' where position is 0-8 (e.g., '{self.MAKE_MOVE_ACTION} 4' for center)"

        elif game_type == SupportedGame.CONNECT_FOUR:
            return f"'{self.MAKE_MOVE_ACTION} <column>' where column is 0-6 (e.g., '{self.MAKE_MOVE_ACTION} 3' for center column)"

        elif game_type == SupportedGame.GO:
            return f"'{self.MAKE_MOVE_ACTION} <row> <col>' or '{self.MAKE_MOVE_ACTION} pass' (e.g., '{self.MAKE_MOVE_ACTION} 3 3' or '{self.MAKE_MOVE_ACTION} pass')"

        return f"'{self.MAKE_MOVE_ACTION} <action>'"

    def _get_system_prompt(self, player_symbol: str) -> str:
        """Get the system prompt for the game player.

        Args:
            player_symbol: Player symbol/color (e.g., "X", "Black")

        Returns:
            System prompt string.
        """
        game_desc = self._get_game_description()
        action_format = self._get_action_format()

        return (
            f"You are playing {self.config.task}. {game_desc}\n\n"
            f"You are playing as {player_symbol}. "
            "Now is your turn to make a move. Before making a move you can pick one of the following actions:\n"
            f"- '{self.GET_BOARD_ACTION}' to see the current board state\n"
            f"- '{self.GET_LEGAL_MOVES_ACTION}' to get a list of available moves\n"
            f"- {action_format}\n\n"
            "Always respond with ONLY one of the above actions. No explanations."
        )

    def _get_player_symbol(self, player_id: str) -> str:
        """Get the player symbol based on game and player ID."""
        game_type = self.config.game_type

        if game_type == SupportedGame.TIC_TAC_TOE:
            return "X" if player_id == "player_1" else "O"

        elif game_type == SupportedGame.CONNECT_FOUR:
            return "Red" if player_id == "player_0" else "Yellow"

        elif game_type == SupportedGame.GO:
            return "Black" if player_id == "black_0" else "White"

        return player_id

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
            Tuple of (action_type, action_value or None)
        """
        response_lower = response.lower().strip()
        game_type = self.config.game_type

        # Check for make_move action
        if self.MAKE_MOVE_ACTION in response_lower:
            if game_type == SupportedGame.TIC_TAC_TOE:
                # Pattern: make_move followed by 0-8
                match = re.search(rf"{self.MAKE_MOVE_ACTION}\s+(\d)", response_lower)
                if match:
                    return self.MAKE_MOVE_ACTION, match.group(1)

            elif game_type == SupportedGame.CONNECT_FOUR:
                # Pattern: make_move followed by 0-6
                match = re.search(rf"{self.MAKE_MOVE_ACTION}\s+(\d)", response_lower)
                if match:
                    return self.MAKE_MOVE_ACTION, match.group(1)

            elif game_type == SupportedGame.GO:
                # Pattern: make_move pass or make_move row col
                if "pass" in response_lower:
                    return self.MAKE_MOVE_ACTION, "pass"
                match = re.search(rf"{self.MAKE_MOVE_ACTION}\s+(\d+)\s+(\d+)", response_lower)
                if match:
                    return self.MAKE_MOVE_ACTION, f"{match.group(1)},{match.group(2)}"

        # Check for get_board
        if self.GET_BOARD_ACTION in response_lower:
            return self.GET_BOARD_ACTION, None

        # Check for get_legal_moves
        if self.GET_LEGAL_MOVES_ACTION in response_lower:
            return self.GET_LEGAL_MOVES_ACTION, None

        # Try to find a bare number (for TTT and Connect Four)
        if game_type in (SupportedGame.TIC_TAC_TOE, SupportedGame.CONNECT_FOUR):
            bare_match = re.search(r"\b(\d)\b", response_lower)
            if bare_match:
                return self.MAKE_MOVE_ACTION, bare_match.group(1)

        return "invalid", None

    def _convert_action_to_index(self, action_value: str, legal_moves: List[int]) -> Optional[int]:
        """Convert action value string to action index.

        Args:
            action_value: Action value from LLM (e.g., "4", "3,3", "pass")
            legal_moves: List of legal action indices

        Returns:
            Action index if valid, None otherwise
        """
        game_type = self.config.game_type

        if game_type in (SupportedGame.TIC_TAC_TOE, SupportedGame.CONNECT_FOUR):
            try:
                action_idx = int(action_value)
                return action_idx if action_idx in legal_moves else None
            except ValueError:
                return None

        elif game_type == SupportedGame.GO:
            if action_value == "pass":
                # Pass is typically the last action (board_size * board_size)
                pass_action = self.config.board_size * self.config.board_size
                return pass_action if pass_action in legal_moves else None

            try:
                row, col = action_value.split(",")
                action_idx = int(row) * self.config.board_size + int(col)
                return action_idx if action_idx in legal_moves else None
            except (ValueError, AttributeError):
                return None

        return None

    def init_agent(self, game_name: str, player_id: str) -> None:
        """Initialize the agent for a game.

        Args:
            game_name: Name of the game (e.g., "tictactoe_v3").
            player_id: Player identifier (e.g., "player_0").
        """
        self._init_client()
        self._game_name = game_name
        self._player_id = player_id
        self._conversation = []
        self._failed_attempts = 0

        # Get player symbol for prompting
        player_symbol = self._get_player_symbol(player_id)

        # Initialize conversation with system prompt
        self._conversation = [
            {"role": "system", "content": self._get_system_prompt(player_symbol)}
        ]

        logger.info(f"Agent initialized for {game_name} as {player_id} ({player_symbol})")

    def select_action_for_player(
        self,
        player_id: str,
        observation: str,
        legal_moves: List[int],
        board_str: str,
        raw_observation: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Select an action based on the player's agent type.

        Routes to the appropriate method based on agent_type:
        - llm: Multi-turn LLM conversation
        - human: Request human input (returns pending action request)
        - random: Random legal move
        - rl_policy: Trained RL policy inference

        Args:
            player_id: Player identifier.
            observation: Current game observation string.
            legal_moves: List of legal action indices.
            board_str: String representation of the board.
            raw_observation: Raw numpy observation (for RL policies).

        Returns:
            Dict with action, action_index, agent_type, reasoning, etc.
        """
        player_config = self.config.get_player_config(player_id)
        agent_type = player_config.agent_type

        logger.debug(f"Selecting action for {player_id} (agent_type={agent_type})")

        if agent_type == AgentType.HUMAN.value:
            return self._select_action_human(player_id, legal_moves, board_str)

        elif agent_type == AgentType.RL_POLICY.value:
            return self._select_action_rl_policy(
                player_id, legal_moves, raw_observation, player_config
            )

        else:  # Default to LLM
            return self.select_action(observation, legal_moves, board_str)

    def _select_action_human(
        self,
        player_id: str,
        legal_moves: List[int],
        board_str: str,
    ) -> Dict[str, Any]:
        """Request action from human player.

        This emits a request and expects the action to be provided
        via the interactive protocol (human_action command).

        Args:
            player_id: Player identifier.
            legal_moves: List of legal action indices.
            board_str: String representation of the board.

        Returns:
            Dict with pending status or action if already provided.
        """
        # Check if we have a pending human action
        if self._pending_human_action is not None:
            action = self._pending_human_action
            self._pending_human_action = None

            if action in legal_moves:
                logger.info(f"Human ({player_id}) action received: {action}")
                return {
                    "action": action,
                    "action_index": action,
                    "agent_type": "human",
                    "reasoning": "Human player input",
                    "success": True,
                }
            else:
                logger.warning(f"Human action {action} not in legal moves")
                return {
                    "action": None,
                    "action_index": None,
                    "agent_type": "human",
                    "reasoning": f"Invalid action {action}, not in legal moves",
                    "success": False,
                    "awaiting_input": True,
                    "legal_moves": legal_moves,
                }

        # No pending action - request human input
        legal_moves_str = self._format_legal_moves(legal_moves)
        logger.info(f"Awaiting human input for {player_id}")

        return {
            "action": None,
            "action_index": None,
            "agent_type": "human",
            "reasoning": "Awaiting human input",
            "success": False,
            "awaiting_input": True,
            "player_id": player_id,
            "board": board_str,
            "legal_moves": legal_moves,
            "legal_moves_str": legal_moves_str,
        }

    def _select_action_rl_policy(
        self,
        player_id: str,
        legal_moves: List[int],
        raw_observation: Optional[Any],
        player_config: Any,
    ) -> Dict[str, Any]:
        """Select action using a trained RL policy.

        Args:
            player_id: Player identifier.
            legal_moves: List of legal action indices.
            raw_observation: Raw numpy observation for the policy.
            player_config: PlayerConfig with policy_path and policy_type.

        Returns:
            Dict with action and metadata.
        """
        policy_path = player_config.policy_path
        policy_type = player_config.policy_type or "sb3"

        if policy_path is None:
            logger.error(f"No policy_path configured for RL agent {player_id}")
            return {
                "action": None,
                "action_index": None,
                "agent_type": "rl_policy",
                "reasoning": "Error: policy_path not configured for RL agent",
                "success": False,
            }

        try:
            # Load policy if not cached
            policy = self._load_rl_policy(policy_path, policy_type)

            if policy is None:
                logger.error(f"Failed to load policy for {player_id} from {policy_path}")
                return {
                    "action": None,
                    "action_index": None,
                    "agent_type": "rl_policy",
                    "policy_path": policy_path,
                    "reasoning": f"Error: Failed to load policy from {policy_path}",
                    "success": False,
                }

            # Get action from policy
            if raw_observation is None:
                logger.error(f"No raw_observation provided for RL policy {player_id}")
                return {
                    "action": None,
                    "action_index": None,
                    "agent_type": "rl_policy",
                    "policy_path": policy_path,
                    "reasoning": "Error: raw_observation required for RL policy",
                    "success": False,
                }

            # Predict action based on policy type
            if policy_type == "sb3":
                action, _states = policy.predict(raw_observation, deterministic=True)
                action = int(action)
            elif policy_type == "cleanrl":
                # CleanRL policies typically use .get_action() or direct forward
                import torch
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(raw_observation).unsqueeze(0)
                    if hasattr(policy, "get_action"):
                        action = policy.get_action(obs_tensor)
                    else:
                        logits = policy(obs_tensor)
                        action = logits.argmax(dim=-1).item()
                action = int(action)
            else:
                logger.error(f"Unknown policy_type {policy_type} for {player_id}")
                return {
                    "action": None,
                    "action_index": None,
                    "agent_type": "rl_policy",
                    "policy_path": policy_path,
                    "reasoning": f"Error: Unknown policy_type '{policy_type}'",
                    "success": False,
                }

            # Validate action is legal
            if action not in legal_moves:
                logger.error(f"RL policy action {action} not in legal moves {legal_moves}")
                return {
                    "action": None,
                    "action_index": None,
                    "agent_type": "rl_policy",
                    "policy_path": policy_path,
                    "reasoning": f"Error: RL policy returned illegal action {action}",
                    "success": False,
                }

            logger.info(f"RL policy ({player_id}) selected action: {action}")

            return {
                "action": action,
                "action_index": action,
                "agent_type": "rl_policy",
                "policy_path": policy_path,
                "policy_type": policy_type,
                "reasoning": f"RL policy prediction from {policy_path}",
                "success": True,
            }

        except Exception as e:
            logger.error(f"RL policy error for {player_id}: {e}")
            return {
                "action": None,
                "action_index": None,
                "agent_type": "rl_policy",
                "reasoning": f"RL policy error: {e}",
                "success": False,
            }

    def _load_rl_policy(self, policy_path: str, policy_type: str) -> Optional[Any]:
        """Load an RL policy from checkpoint.

        Args:
            policy_path: Path to the policy checkpoint.
            policy_type: Type of policy ("sb3", "cleanrl", "xuance").

        Returns:
            Loaded policy model or None if loading fails.
        """
        cache_key = f"{policy_type}:{policy_path}"

        if cache_key in self._rl_policies:
            return self._rl_policies[cache_key]

        try:
            if policy_type == "sb3":
                # Stable-Baselines3
                from stable_baselines3 import PPO, A2C, DQN
                # Try common algorithms
                for algo_cls in [PPO, A2C, DQN]:
                    try:
                        policy = algo_cls.load(policy_path)
                        self._rl_policies[cache_key] = policy
                        logger.info(f"Loaded SB3 policy from {policy_path}")
                        return policy
                    except Exception:
                        continue
                logger.error(f"Could not load SB3 policy from {policy_path}")
                return None

            elif policy_type == "cleanrl":
                # CleanRL - load PyTorch model
                import torch
                policy = torch.load(policy_path)
                policy.eval()
                self._rl_policies[cache_key] = policy
                logger.info(f"Loaded CleanRL policy from {policy_path}")
                return policy

            elif policy_type == "xuance":
                # XuanCe - similar to SB3
                logger.warning("XuanCe policy loading not yet implemented")
                return None

            else:
                logger.error(f"Unknown policy_type: {policy_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to load policy from {policy_path}: {e}")
            return None

    def set_human_action(self, action: int) -> None:
        """Set the pending human action.

        Called when human input is received via interactive protocol.

        Args:
            action: The action index chosen by the human.
        """
        self._pending_human_action = action
        logger.debug(f"Human action set: {action}")

    def select_action(
        self,
        observation: str,
        legal_moves: List[int],
        board_str: str,
    ) -> Dict[str, Any]:
        """Select an action using multi-turn LLM conversation.

        This is the LLM-specific implementation. For routing based on
        agent type, use select_action_for_player() instead.

        Args:
            observation: Current game observation string.
            legal_moves: List of legal action indices.
            board_str: String representation of the board.

        Returns:
            Dict with action, action_index, reasoning, tokens, etc.
        """
        if not self._conversation:
            self.init_agent(self._game_name, self._player_id)

        # Format legal moves for display
        legal_moves_str = self._format_legal_moves(legal_moves)

        # Build initial observation message
        obs_msg = (
            f"Current board:\n{board_str}\n\n"
            f"Legal moves: {legal_moves_str}\n\n"
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
                fallback = random.choice(legal_moves) if legal_moves else 0
                return {
                    "action": fallback,
                    "action_index": fallback,
                    "reasoning": f"LLM error: {e}",
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                    "success": False,
                }

            logger.debug(f"LLM response (turn {turn + 1}): {response}")

            # Add response to conversation
            self._conversation.append({"role": "assistant", "content": response})

            # Parse the action
            action_type, action_value = self._parse_action(response)

            if action_type == self.GET_BOARD_ACTION:
                # Provide board state
                reply = f"Current board:\n{board_str}"
                self._conversation.append({"role": "user", "content": reply})
                continue

            elif action_type == self.GET_LEGAL_MOVES_ACTION:
                # Provide legal moves
                reply = f"Legal moves: {legal_moves_str}"
                self._conversation.append({"role": "user", "content": reply})
                continue

            elif action_type == self.MAKE_MOVE_ACTION and action_value:
                # Convert and validate move
                action_idx = self._convert_action_to_index(action_value, legal_moves)

                if action_idx is not None:
                    logger.info(f"Valid move selected: {action_value} -> index {action_idx}")
                    return {
                        "action": action_idx,
                        "action_index": action_idx,
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
                        fallback = random.choice(legal_moves) if legal_moves else 0
                        return {
                            "action": fallback,
                            "action_index": fallback,
                            "reasoning": f"Max retries, last attempt: {response}",
                            "input_tokens": self._total_input_tokens,
                            "output_tokens": self._total_output_tokens,
                            "success": False,
                        }

                    reply = (
                        f"Invalid move '{action_value}'. That move is not legal.\n"
                        f"Legal moves are: {legal_moves_str}\n"
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
                    fallback = random.choice(legal_moves) if legal_moves else 0
                    return {
                        "action": fallback,
                        "action_index": fallback,
                        "reasoning": f"Invalid format, last: {response}",
                        "input_tokens": self._total_input_tokens,
                        "output_tokens": self._total_output_tokens,
                        "success": False,
                    }

                action_format = self._get_action_format()
                reply = (
                    f"Invalid action. Please respond with one of:\n"
                    f"- '{self.GET_BOARD_ACTION}' to see the board\n"
                    f"- '{self.GET_LEGAL_MOVES_ACTION}' to see legal moves\n"
                    f"- {action_format}"
                )
                self._conversation.append({"role": "user", "content": reply})
                continue

        # Max turns reached
        logger.warning("Max dialog turns reached, using random move")
        import random
        fallback = random.choice(legal_moves) if legal_moves else 0
        return {
            "action": fallback,
            "action_index": fallback,
            "reasoning": "Max turns reached",
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "success": False,
        }

    def _format_legal_moves(self, legal_moves: List[int]) -> str:
        """Format legal moves for display based on game type."""
        game_type = self.config.game_type

        if game_type in (SupportedGame.TIC_TAC_TOE, SupportedGame.CONNECT_FOUR):
            return ", ".join(str(m) for m in legal_moves)

        elif game_type == SupportedGame.GO:
            formatted = []
            board_size = self.config.board_size
            pass_action = board_size * board_size

            for move in legal_moves:
                if move == pass_action:
                    formatted.append("pass")
                else:
                    row = move // board_size
                    col = move % board_size
                    formatted.append(f"({row},{col})")

            return ", ".join(formatted)

        return ", ".join(str(m) for m in legal_moves)

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
                    game_name = cmd.get("game_name", self.config.task)
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
                    raw_observation = cmd.get("raw_observation")  # For RL policies

                    # Use observation as board string
                    board_str = observation

                    # Route to appropriate agent type handler
                    result = self.select_action_for_player(
                        player_id, observation, legal_moves, board_str, raw_observation
                    )

                    response = {
                        "type": "action_selected",
                        "run_id": self.config.run_id,
                        "player_id": player_id,
                        "action": result.get("action"),
                        "action_index": result.get("action_index"),
                        "agent_type": result.get("agent_type", "llm"),
                        "reasoning": result.get("reasoning", ""),
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                        "success": result.get("success", True),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Include human input request info if awaiting
                    if result.get("awaiting_input"):
                        response["type"] = "awaiting_human_input"
                        response["legal_moves"] = result.get("legal_moves", [])
                        response["legal_moves_str"] = result.get("legal_moves_str", "")
                        response["board"] = result.get("board", "")

                    self._emit(response)

                elif cmd_type == "human_action":
                    # Receive human player's action
                    action = cmd.get("action")
                    player_id = cmd.get("player_id", self._player_id)

                    if action is not None:
                        self.set_human_action(int(action))
                        self._emit({
                            "type": "human_action_received",
                            "run_id": self.config.run_id,
                            "player_id": player_id,
                            "action": action,
                        })
                    else:
                        self._emit({
                            "type": "error",
                            "message": "human_action command missing 'action' field",
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
