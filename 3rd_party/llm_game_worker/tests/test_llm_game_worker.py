"""Tests for LLM Game Worker."""

import pytest

from llm_game_worker import __version__, get_worker_metadata
from llm_game_worker.config import LLMGameWorkerConfig, SupportedGame, PlayMode, PlayerConfig, AgentType
from llm_game_worker.runtime import LLMGameWorkerRuntime


class TestVersion:
    """Test version info."""

    def test_version_is_string(self) -> None:
        """Test that version is a string."""
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"


class TestWorkerMetadata:
    """Test worker metadata."""

    def test_get_worker_metadata_returns_tuple(self) -> None:
        """Test that get_worker_metadata returns a tuple."""
        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_worker_metadata_fields(self) -> None:
        """Test metadata fields."""
        metadata, capabilities = get_worker_metadata()

        assert metadata.name == "LLM Game Worker"
        assert metadata.version == __version__
        assert "PettingZoo" in metadata.description

    def test_worker_capabilities_fields(self) -> None:
        """Test capabilities fields."""
        metadata, capabilities = get_worker_metadata()

        assert capabilities.worker_type == "llm_game"
        assert "pettingzoo" in capabilities.env_families
        assert capabilities.max_agents == 2
        assert capabilities.supports_self_play is True


class TestConfig:
    """Test LLMGameWorkerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LLMGameWorkerConfig()

        assert config.task == "tictactoe_v3"
        assert config.env_name == "pettingzoo"
        assert config.client_name == "vllm"
        assert config.temperature == 0.3
        assert config.max_retries == 3

    def test_config_to_dict(self) -> None:
        """Test config serialization."""
        config = LLMGameWorkerConfig(task="connect_four_v3")
        data = config.to_dict()

        assert data["task"] == "connect_four_v3"
        assert "model_id" in data
        assert "base_url" in data

    def test_config_from_dict(self) -> None:
        """Test config deserialization."""
        data = {
            "task": "go_v5",
            "temperature": 0.5,
            "board_size": 9,
        }
        config = LLMGameWorkerConfig.from_dict(data)

        assert config.task == "go_v5"
        assert config.temperature == 0.5
        assert config.board_size == 9

    def test_game_type_property(self) -> None:
        """Test game_type property."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        assert config.game_type == SupportedGame.TIC_TAC_TOE

        config = LLMGameWorkerConfig(task="connect_four_v3")
        assert config.game_type == SupportedGame.CONNECT_FOUR

        config = LLMGameWorkerConfig(task="go_v5")
        assert config.game_type == SupportedGame.GO


class TestSupportedGame:
    """Test SupportedGame enum."""

    def test_supported_games(self) -> None:
        """Test all supported games are defined."""
        assert SupportedGame.TIC_TAC_TOE.value == "tictactoe_v3"
        assert SupportedGame.CONNECT_FOUR.value == "connect_four_v3"
        assert SupportedGame.GO.value == "go_v5"


class TestRuntime:
    """Test LLMGameWorkerRuntime."""

    def test_runtime_creation(self) -> None:
        """Test runtime can be created."""
        config = LLMGameWorkerConfig()
        runtime = LLMGameWorkerRuntime(config)

        assert runtime.config == config
        assert runtime._player_id == ""

    def test_runtime_init_agent(self) -> None:
        """Test agent initialization."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        # Note: This doesn't call the LLM, just sets up state
        runtime._game_name = "tictactoe_v3"
        runtime._player_id = "player_1"
        runtime._conversation = []

        assert runtime._player_id == "player_1"

    def test_format_legal_moves_tictactoe(self) -> None:
        """Test legal moves formatting for Tic-Tac-Toe."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        moves = [0, 1, 4, 8]
        formatted = runtime._format_legal_moves(moves)
        assert formatted == "0, 1, 4, 8"

    def test_format_legal_moves_connect_four(self) -> None:
        """Test legal moves formatting for Connect Four."""
        config = LLMGameWorkerConfig(task="connect_four_v3")
        runtime = LLMGameWorkerRuntime(config)

        moves = [0, 2, 3, 6]
        formatted = runtime._format_legal_moves(moves)
        assert formatted == "0, 2, 3, 6"

    def test_format_legal_moves_go(self) -> None:
        """Test legal moves formatting for Go."""
        config = LLMGameWorkerConfig(task="go_v5", board_size=9)
        runtime = LLMGameWorkerRuntime(config)

        # Move 0 = (0,0), Move 10 = (1,1), Move 81 = pass
        moves = [0, 10, 81]
        formatted = runtime._format_legal_moves(moves)
        assert "(0,0)" in formatted
        assert "(1,1)" in formatted
        assert "pass" in formatted

    def test_get_player_symbol_tictactoe(self) -> None:
        """Test player symbol for Tic-Tac-Toe."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        assert runtime._get_player_symbol("player_1") == "X"
        assert runtime._get_player_symbol("player_2") == "O"

    def test_get_player_symbol_connect_four(self) -> None:
        """Test player symbol for Connect Four."""
        config = LLMGameWorkerConfig(task="connect_four_v3")
        runtime = LLMGameWorkerRuntime(config)

        assert runtime._get_player_symbol("player_0") == "Red"
        assert runtime._get_player_symbol("player_1") == "Yellow"

    def test_get_player_symbol_go(self) -> None:
        """Test player symbol for Go."""
        config = LLMGameWorkerConfig(task="go_v5")
        runtime = LLMGameWorkerRuntime(config)

        assert runtime._get_player_symbol("black_0") == "Black"
        assert runtime._get_player_symbol("white_0") == "White"

    def test_convert_action_tictactoe(self) -> None:
        """Test action conversion for Tic-Tac-Toe."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 1, 4, 8]

        assert runtime._convert_action_to_index("4", legal_moves) == 4
        assert runtime._convert_action_to_index("0", legal_moves) == 0
        assert runtime._convert_action_to_index("5", legal_moves) is None  # Not in legal
        assert runtime._convert_action_to_index("invalid", legal_moves) is None

    def test_convert_action_connect_four(self) -> None:
        """Test action conversion for Connect Four."""
        config = LLMGameWorkerConfig(task="connect_four_v3")
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 2, 3, 6]

        assert runtime._convert_action_to_index("3", legal_moves) == 3
        assert runtime._convert_action_to_index("1", legal_moves) is None

    def test_convert_action_go(self) -> None:
        """Test action conversion for Go."""
        config = LLMGameWorkerConfig(task="go_v5", board_size=9)
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 10, 81]  # (0,0), (1,1), pass

        assert runtime._convert_action_to_index("0,0", legal_moves) == 0
        assert runtime._convert_action_to_index("1,1", legal_moves) == 10
        assert runtime._convert_action_to_index("pass", legal_moves) == 81
        assert runtime._convert_action_to_index("5,5", legal_moves) is None

    def test_parse_action_tictactoe(self) -> None:
        """Test action parsing for Tic-Tac-Toe."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        action_type, value = runtime._parse_action("make_move 4")
        assert action_type == "make_move"
        assert value == "4"

        action_type, value = runtime._parse_action("get_board")
        assert action_type == "get_board"

        action_type, value = runtime._parse_action("get_legal_moves")
        assert action_type == "get_legal_moves"

    def test_parse_action_go(self) -> None:
        """Test action parsing for Go."""
        config = LLMGameWorkerConfig(task="go_v5")
        runtime = LLMGameWorkerRuntime(config)

        action_type, value = runtime._parse_action("make_move 3 3")
        assert action_type == "make_move"
        assert value == "3,3"

        action_type, value = runtime._parse_action("make_move pass")
        assert action_type == "make_move"
        assert value == "pass"

    def test_game_description(self) -> None:
        """Test game description generation."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)
        desc = runtime._get_game_description()
        assert "3x3" in desc
        assert "Tic-Tac-Toe" in desc

        config = LLMGameWorkerConfig(task="connect_four_v3")
        runtime = LLMGameWorkerRuntime(config)
        desc = runtime._get_game_description()
        assert "Connect Four" in desc
        assert "6x7" in desc

        config = LLMGameWorkerConfig(task="go_v5", board_size=9)
        runtime = LLMGameWorkerRuntime(config)
        desc = runtime._get_game_description()
        assert "Go" in desc
        assert "9x9" in desc


class TestPlayMode:
    """Test PlayMode enum."""

    def test_play_modes_defined(self) -> None:
        """Test all play modes are defined."""
        assert PlayMode.SELF_PLAY.value == "self_play"
        assert PlayMode.HUMAN_VS_AI.value == "human_vs_ai"
        assert PlayMode.AI_VS_HUMAN.value == "ai_vs_human"
        assert PlayMode.AI_ONLY.value == "ai_only"


class TestPlayerConfig:
    """Test PlayerConfig dataclass."""

    def test_default_player_config(self) -> None:
        """Test default PlayerConfig values."""
        config = PlayerConfig()
        assert config.player_id == "player_1"
        assert config.agent_type == "llm"
        assert config.model_id is None
        assert config.temperature is None

    def test_custom_player_config(self) -> None:
        """Test custom PlayerConfig values."""
        config = PlayerConfig(
            player_id="player_2",
            agent_type="human",
            model_id="gpt-4",
            temperature=0.7,
        )
        assert config.player_id == "player_2"
        assert config.agent_type == "human"
        assert config.model_id == "gpt-4"
        assert config.temperature == 0.7


class TestPlayerAssignment:
    """Test player assignment configuration."""

    def test_default_play_mode(self) -> None:
        """Test default play mode is self_play."""
        config = LLMGameWorkerConfig()
        assert config.play_mode == "self_play"
        assert config.play_as == "both"

    def test_is_llm_controlled_self_play(self) -> None:
        """Test LLM controls both players in self_play mode."""
        config = LLMGameWorkerConfig(play_mode="self_play")
        assert config.is_llm_controlled("player_1") is True
        assert config.is_llm_controlled("player_2") is True

    def test_is_llm_controlled_human_vs_ai(self) -> None:
        """Test player assignment in human_vs_ai mode."""
        config = LLMGameWorkerConfig(play_mode="human_vs_ai")
        # Player 1 is human, Player 2 is AI
        assert config.is_llm_controlled("player_1") is False
        assert config.is_llm_controlled("player_2") is True

    def test_is_llm_controlled_ai_vs_human(self) -> None:
        """Test player assignment in ai_vs_human mode."""
        config = LLMGameWorkerConfig(play_mode="ai_vs_human")
        # Player 1 is AI, Player 2 is human
        assert config.is_llm_controlled("player_1") is True
        assert config.is_llm_controlled("player_2") is False

    def test_is_llm_controlled_ai_only_player1(self) -> None:
        """Test player assignment in ai_only mode with player_1."""
        config = LLMGameWorkerConfig(play_mode="ai_only", play_as="player_1")
        assert config.is_llm_controlled("player_1") is True
        assert config.is_llm_controlled("player_2") is False

    def test_is_llm_controlled_ai_only_player2(self) -> None:
        """Test player assignment in ai_only mode with player_2."""
        config = LLMGameWorkerConfig(play_mode="ai_only", play_as="player_2")
        assert config.is_llm_controlled("player_1") is False
        assert config.is_llm_controlled("player_2") is True

    def test_custom_player_configs(self) -> None:
        """Test custom PlayerConfig list."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="llm", model_id="gpt-4"),
            PlayerConfig(player_id="player_2", agent_type="human"),
        ]
        config = LLMGameWorkerConfig(players=players)

        assert config.is_llm_controlled("player_1") is True
        assert config.is_llm_controlled("player_2") is False
        assert config.get_model_for_player("player_1") == "gpt-4"

    def test_get_model_for_player_default(self) -> None:
        """Test default model is used when not specified."""
        config = LLMGameWorkerConfig(model_id="default-model")
        assert config.get_model_for_player("player_1") == "default-model"

    def test_get_model_for_player_override(self) -> None:
        """Test player-specific model overrides default."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="llm", model_id="player1-model"),
        ]
        config = LLMGameWorkerConfig(model_id="default-model", players=players)
        assert config.get_model_for_player("player_1") == "player1-model"
        assert config.get_model_for_player("player_2") == "default-model"

    def test_get_temperature_for_player(self) -> None:
        """Test temperature per player."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="llm", temperature=0.9),
        ]
        config = LLMGameWorkerConfig(temperature=0.3, players=players)
        assert config.get_temperature_for_player("player_1") == 0.9
        assert config.get_temperature_for_player("player_2") == 0.3

    def test_config_serialization_with_players(self) -> None:
        """Test config to_dict/from_dict with players."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="llm", model_id="gpt-4"),
            PlayerConfig(player_id="player_2", agent_type="human"),
        ]
        config = LLMGameWorkerConfig(
            play_mode="ai_only",
            play_as="player_1",
            players=players,
        )

        data = config.to_dict()
        assert data["play_mode"] == "ai_only"
        assert data["play_as"] == "player_1"
        assert len(data["players"]) == 2

        restored = LLMGameWorkerConfig.from_dict(data)
        assert restored.play_mode == "ai_only"
        assert restored.play_as == "player_1"
        assert len(restored.players) == 2
        assert restored.players[0].model_id == "gpt-4"


class TestAgentType:
    """Test AgentType enum."""

    def test_agent_types_defined(self) -> None:
        """Test all agent types are defined."""
        assert AgentType.LLM.value == "llm"
        assert AgentType.HUMAN.value == "human"
        assert AgentType.RL_POLICY.value == "rl_policy"


class TestAgentTypeSelection:
    """Test agent type routing in runtime."""

    def test_select_action_human_awaiting(self) -> None:
        """Test human agent awaiting input."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 1, 4, 8]
        result = runtime._select_action_human("player_1", legal_moves, "board")

        assert result["action"] is None
        assert result["agent_type"] == "human"
        assert result["awaiting_input"] is True
        assert result["legal_moves"] == legal_moves

    def test_select_action_human_with_pending(self) -> None:
        """Test human agent with pending action."""
        config = LLMGameWorkerConfig(task="tictactoe_v3")
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 1, 4, 8]
        runtime.set_human_action(4)

        result = runtime._select_action_human("player_1", legal_moves, "board")

        assert result["action"] == 4
        assert result["agent_type"] == "human"
        assert result["success"] is True

    def test_select_action_for_player_routes_correctly(self) -> None:
        """Test that select_action_for_player routes to correct handler."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="human"),
            PlayerConfig(player_id="player_2", agent_type="human"),
        ]
        config = LLMGameWorkerConfig(task="tictactoe_v3", players=players)
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 1, 4, 8]

        # Human player 1 (should be awaiting)
        result1 = runtime.select_action_for_player(
            "player_1", "obs", legal_moves, "board"
        )
        assert result1["agent_type"] == "human"
        assert result1["awaiting_input"] is True

        # Human player 2 (should be awaiting)
        result2 = runtime.select_action_for_player(
            "player_2", "obs", legal_moves, "board"
        )
        assert result2["agent_type"] == "human"
        assert result2["awaiting_input"] is True

    def test_rl_policy_config(self) -> None:
        """Test PlayerConfig with RL policy settings."""
        player = PlayerConfig(
            player_id="player_1",
            agent_type="rl_policy",
            policy_path="/path/to/model.zip",
            policy_type="sb3",
        )

        assert player.agent_type == "rl_policy"
        assert player.policy_path == "/path/to/model.zip"
        assert player.policy_type == "sb3"

    def test_rl_policy_error_when_no_policy_path(self) -> None:
        """Test RL policy returns error when no policy_path configured."""
        players = [
            PlayerConfig(player_id="player_1", agent_type="rl_policy"),
        ]
        config = LLMGameWorkerConfig(task="tictactoe_v3", players=players)
        runtime = LLMGameWorkerRuntime(config)

        legal_moves = [0, 1, 4, 8]
        result = runtime.select_action_for_player(
            "player_1", "obs", legal_moves, "board"
        )

        # Should return error since no policy_path configured
        assert result["action"] is None
        assert result["agent_type"] == "rl_policy"
        assert result["success"] is False
        assert "Error" in result["reasoning"]
