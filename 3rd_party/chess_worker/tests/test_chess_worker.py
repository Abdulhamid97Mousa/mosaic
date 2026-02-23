"""Tests for Chess Worker."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from chess_worker import ChessWorkerConfig, ChessWorkerRuntime


class TestChessWorkerConfig:
    """Tests for ChessWorkerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChessWorkerConfig()
        assert config.env_name == "pettingzoo"
        assert config.task == "chess_v6"
        assert config.client_name == "vllm"
        assert config.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.base_url == "http://127.0.0.1:8000/v1"
        assert config.temperature == 0.3
        assert config.max_tokens == 256
        assert config.max_retries == 3
        assert config.max_dialog_turns == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ChessWorkerConfig(run_id="test-123")
        d = config.to_dict()
        assert d["run_id"] == "test-123"
        assert d["env_name"] == "pettingzoo"
        assert d["task"] == "chess_v6"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "run_id": "test-456",
            "client_name": "openai",
            "model_id": "gpt-4",
            "temperature": 0.7,
        }
        config = ChessWorkerConfig.from_dict(data)
        assert config.run_id == "test-456"
        assert config.client_name == "openai"
        assert config.model_id == "gpt-4"
        assert config.temperature == 0.7


class TestChessWorkerRuntime:
    """Tests for ChessWorkerRuntime."""

    def test_init(self):
        """Test runtime initialization."""
        config = ChessWorkerConfig(run_id="test-init")
        runtime = ChessWorkerRuntime(config)
        assert runtime.config == config
        assert runtime._client is None
        assert runtime._conversation == []
        assert runtime._llm_calls == 0

    def test_get_system_prompt_white(self):
        """Test system prompt for white player."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)
        prompt = runtime._get_system_prompt("white")
        assert "white" in prompt
        assert "get_current_board" in prompt
        assert "get_legal_moves" in prompt
        assert "make_move" in prompt

    def test_get_system_prompt_black(self):
        """Test system prompt for black player."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)
        prompt = runtime._get_system_prompt("black")
        assert "black" in prompt

    def test_parse_action_make_move(self):
        """Test parsing make_move action."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        # Standard format
        action, move = runtime._parse_action("make_move e2e4")
        assert action == "make_move"
        assert move == "e2e4"

        # With promotion
        action, move = runtime._parse_action("make_move a7a8q")
        assert action == "make_move"
        assert move == "a7a8q"

    def test_parse_action_get_board(self):
        """Test parsing get_current_board action."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        action, move = runtime._parse_action("get_current_board")
        assert action == "get_current_board"
        assert move is None

    def test_parse_action_get_legal_moves(self):
        """Test parsing get_legal_moves action."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        action, move = runtime._parse_action("get_legal_moves")
        assert action == "get_legal_moves"
        assert move is None

    def test_parse_action_bare_uci_move(self):
        """Test parsing bare UCI move without make_move prefix."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        # Bare UCI move should be detected
        action, move = runtime._parse_action("e2e4")
        assert action == "make_move"
        assert move == "e2e4"

        # Knight move
        action, move = runtime._parse_action("g1f3")
        assert action == "make_move"
        assert move == "g1f3"

    def test_parse_action_invalid(self):
        """Test parsing invalid action."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        action, move = runtime._parse_action("I want to think about this")
        assert action == "invalid"
        assert move is None

    def test_init_agent(self):
        """Test agent initialization."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        with patch.object(runtime, "_init_client"):
            runtime.init_agent("chess_v6", "player_0")

        assert runtime._game_name == "chess_v6"
        assert runtime._player_id == "player_0"
        assert len(runtime._conversation) == 1
        assert runtime._conversation[0]["role"] == "system"
        assert "white" in runtime._conversation[0]["content"]

    def test_init_agent_black(self):
        """Test agent initialization for black player."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)

        with patch.object(runtime, "_init_client"):
            runtime.init_agent("chess_v6", "player_1")

        assert runtime._player_id == "player_1"
        assert "black" in runtime._conversation[0]["content"]

    @patch.object(ChessWorkerRuntime, "_call_llm")
    @patch.object(ChessWorkerRuntime, "_init_client")
    def test_select_action_valid_move(self, mock_init, mock_llm):
        """Test selecting a valid move."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)
        runtime.init_agent("chess_v6", "player_0")

        # LLM returns a valid move
        mock_llm.return_value = "make_move e2e4"

        result = runtime.select_action(
            observation="Starting position",
            legal_moves=["e2e4", "d2d4", "g1f3"],
            board_str="[board representation]",
        )

        assert result["action_str"] == "e2e4"
        assert result["success"] is True

    @patch.object(ChessWorkerRuntime, "_call_llm")
    @patch.object(ChessWorkerRuntime, "_init_client")
    def test_select_action_multi_turn(self, mock_init, mock_llm):
        """Test multi-turn conversation (LLM asks for info first)."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)
        runtime.init_agent("chess_v6", "player_0")

        # First LLM asks for legal moves, then makes a move
        mock_llm.side_effect = ["get_legal_moves", "make_move e2e4"]

        result = runtime.select_action(
            observation="Starting position",
            legal_moves=["e2e4", "d2d4", "g1f3"],
            board_str="[board representation]",
        )

        assert result["action_str"] == "e2e4"
        assert result["success"] is True
        assert mock_llm.call_count == 2

    @patch.object(ChessWorkerRuntime, "_call_llm")
    @patch.object(ChessWorkerRuntime, "_init_client")
    def test_select_action_invalid_then_valid(self, mock_init, mock_llm):
        """Test retry on invalid move."""
        config = ChessWorkerConfig()
        runtime = ChessWorkerRuntime(config)
        runtime.init_agent("chess_v6", "player_0")

        # First move is invalid, second is valid
        mock_llm.side_effect = ["make_move a1a2", "make_move e2e4"]

        result = runtime.select_action(
            observation="Starting position",
            legal_moves=["e2e4", "d2d4", "g1f3"],
            board_str="[board representation]",
        )

        assert result["action_str"] == "e2e4"
        assert result["success"] is True
        assert mock_llm.call_count == 2


class TestGetWorkerMetadata:
    """Tests for get_worker_metadata function."""

    def test_metadata_structure(self):
        """Test metadata has required fields."""
        from chess_worker import get_worker_metadata

        metadata = get_worker_metadata()
        assert "name" in metadata
        assert "version" in metadata
        assert "supported_envs" in metadata
        assert "supported_tasks" in metadata
        assert metadata["name"] == "chess_worker"
        assert "pettingzoo" in metadata["supported_envs"]
        assert "chess_v6" in metadata["supported_tasks"]
