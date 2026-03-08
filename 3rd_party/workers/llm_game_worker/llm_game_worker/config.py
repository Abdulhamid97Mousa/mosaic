"""Configuration for LLM Game Worker."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class SupportedGame(str, Enum):
    """Supported PettingZoo classic games."""
    TIC_TAC_TOE = "tictactoe_v3"
    CONNECT_FOUR = "connect_four_v3"
    GO = "go_v5"


class PlayMode(str, Enum):
    """Play mode for the worker."""
    SELF_PLAY = "self_play"          # LLM vs LLM (same or different models)
    HUMAN_VS_AI = "human_vs_ai"      # Human plays first, AI plays second
    AI_VS_HUMAN = "ai_vs_human"      # AI plays first, Human plays second
    AI_ONLY = "ai_only"              # AI controls specified player(s)


class AgentType(str, Enum):
    """Types of agents that can control a player."""
    LLM = "llm"              # LLM-based agent (OpenAI, Anthropic, vLLM)
    HUMAN = "human"          # Human player (waits for input)
    RL_POLICY = "rl_policy"  # Trained RL policy (SB3, CleanRL, etc.)


@dataclass
class PlayerConfig:
    """Configuration for a single player/agent.

    Attributes:
        player_id: Player identifier (e.g., "player_1", "black_0")
        agent_type: Type of agent ("llm", "human", "rl_policy")
        model_id: Model identifier (for LLM agents)
        temperature: Sampling temperature (for LLM agents)
        policy_path: Path to trained RL policy checkpoint (for rl_policy agents)
        policy_type: Type of RL policy ("sb3", "cleanrl", "xuance")
    """
    player_id: str = "player_1"
    agent_type: str = "llm"  # "llm", "human", "rl_policy"
    model_id: Optional[str] = None  # Override main model_id for this player
    temperature: Optional[float] = None  # Override main temperature
    policy_path: Optional[str] = None  # Path to RL policy checkpoint
    policy_type: Optional[str] = None  # "sb3", "cleanrl", "xuance"


@dataclass
class LLMGameWorkerConfig:
    """Configuration for LLM Game Worker.

    Supports multiple PettingZoo classic games with a unified LLM interface.

    Attributes:
        run_id: Unique run identifier.
        env_name: Environment family (always "pettingzoo" for these games).
        task: Specific game task (e.g., "tictactoe_v3", "connect_four_v3").
        play_mode: Play mode (self_play, human_vs_ai, ai_vs_human, ai_only).
        play_as: Which player(s) the LLM controls ("player_1", "player_2", "both").
        players: List of player configurations for fine-grained control.
        client_name: LLM client ("vllm", "openai", "anthropic").
        model_id: Model identifier (default for all LLM players).
        base_url: API base URL for vLLM or compatible API.
        api_key: API key (optional for local vLLM).
        temperature: Sampling temperature (default for all LLM players).
        max_tokens: Maximum tokens in response.
        max_retries: Max retries for invalid moves.
        max_dialog_turns: Max conversation turns per move.
        board_size: Board size (for Go, default 19).
        komi: Komi value for Go (default 7.5).
    """
    run_id: str = ""
    env_name: str = "pettingzoo"
    task: str = "tictactoe_v3"

    # Player assignment
    play_mode: str = "self_play"  # self_play, human_vs_ai, ai_vs_human, ai_only
    play_as: str = "both"  # "player_1", "player_2", "both" (for self_play)
    players: List[PlayerConfig] = field(default_factory=list)

    # LLM settings (defaults for all LLM players)
    client_name: str = "vllm"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None

    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 256

    # Game-specific settings
    max_retries: int = 3  # Max invalid move attempts before giving up
    max_dialog_turns: int = 10  # Max conversation turns per move

    # Go-specific settings
    board_size: int = 19
    komi: float = 7.5

    # Telemetry
    telemetry_dir: str = "var/telemetry"

    def get_player_config(self, player_id: str) -> PlayerConfig:
        """Get configuration for a specific player.

        Args:
            player_id: Player identifier (e.g., "player_1")

        Returns:
            PlayerConfig for the player, or default LLM config if not specified.
        """
        # First check for explicit player configuration
        for player in self.players:
            if player.player_id == player_id:
                return player

        # Handle play modes with specific player assignments
        if self.play_mode == PlayMode.HUMAN_VS_AI.value:
            # Human plays first (player_1/player_0), AI plays second
            if player_id in ("player_1", "player_0", "black_0"):
                return PlayerConfig(player_id=player_id, agent_type="human")
            return PlayerConfig(player_id=player_id, agent_type="llm")

        if self.play_mode == PlayMode.AI_VS_HUMAN.value:
            # AI plays first (player_1/player_0), Human plays second
            if player_id in ("player_1", "player_0", "black_0"):
                return PlayerConfig(player_id=player_id, agent_type="llm")
            return PlayerConfig(player_id=player_id, agent_type="human")

        if self.play_mode == PlayMode.SELF_PLAY.value:
            # Both players are LLM-controlled
            return PlayerConfig(player_id=player_id, agent_type="llm")

        # AI_ONLY mode - check play_as
        if self.play_as == player_id or self.play_as == "both":
            return PlayerConfig(player_id=player_id, agent_type="llm")
        return PlayerConfig(player_id=player_id, agent_type="human")

    def is_llm_controlled(self, player_id: str) -> bool:
        """Check if a player is controlled by the LLM.

        Args:
            player_id: Player identifier

        Returns:
            True if the player is LLM-controlled.
        """
        config = self.get_player_config(player_id)
        return config.agent_type == "llm"

    def get_model_for_player(self, player_id: str) -> str:
        """Get the model ID for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            Model ID to use for this player.
        """
        config = self.get_player_config(player_id)
        return config.model_id or self.model_id

    def get_temperature_for_player(self, player_id: str) -> float:
        """Get the temperature for a specific player.

        Args:
            player_id: Player identifier

        Returns:
            Temperature to use for this player.
        """
        config = self.get_player_config(player_id)
        return config.temperature if config.temperature is not None else self.temperature

    @property
    def game_type(self) -> SupportedGame:
        """Get the game type from task string."""
        for game in SupportedGame:
            if game.value == self.task:
                return game
        return SupportedGame.TIC_TAC_TOE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "task": self.task,
            "play_mode": self.play_mode,
            "play_as": self.play_as,
            "players": [
                {
                    "player_id": p.player_id,
                    "agent_type": p.agent_type,
                    "model_id": p.model_id,
                    "temperature": p.temperature,
                }
                for p in self.players
            ],
            "client_name": self.client_name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "max_dialog_turns": self.max_dialog_turns,
            "board_size": self.board_size,
            "komi": self.komi,
            "telemetry_dir": self.telemetry_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMGameWorkerConfig":
        """Create from dictionary."""
        # Parse player configs
        players_data = data.get("players", [])
        players = [
            PlayerConfig(
                player_id=p.get("player_id", "player_1"),
                agent_type=p.get("agent_type", "llm"),
                model_id=p.get("model_id"),
                temperature=p.get("temperature"),
            )
            for p in players_data
        ]

        return cls(
            run_id=data.get("run_id", ""),
            env_name=data.get("env_name", "pettingzoo"),
            task=data.get("task", "tictactoe_v3"),
            play_mode=data.get("play_mode", "self_play"),
            play_as=data.get("play_as", "both"),
            players=players,
            client_name=data.get("client_name", "vllm"),
            model_id=data.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct"),
            base_url=data.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature", 0.3),
            max_tokens=data.get("max_tokens", 256),
            max_retries=data.get("max_retries", 3),
            max_dialog_turns=data.get("max_dialog_turns", 10),
            board_size=data.get("board_size", 19),
            komi=data.get("komi", 7.5),
            telemetry_dir=data.get("telemetry_dir", "var/telemetry"),
        )
