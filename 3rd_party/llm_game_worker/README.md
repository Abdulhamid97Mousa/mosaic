# LLM Game Worker

LLM-based player for PettingZoo classic board games. Uses multi-turn conversation prompting similar to llm_chess.

## Supported Games

| Game | Environment ID | Description |
|------|----------------|-------------|
| Tic-Tac-Toe | `tictactoe_v3` | 3x3 grid, positions 0-8 |
| Connect Four | `connect_four_v3` | 7 columns, drop to 0-6 |
| Go | `go_v5` | 19x19 board (configurable), coordinates or 'pass' |

## Installation

```bash
# From MOSAIC root directory
pip install -e 3rd_party/llm_game_worker

# Or with development dependencies
pip install -e "3rd_party/llm_game_worker[dev]"
```

## Usage

### Command Line

```bash
# Play Tic-Tac-Toe with local vLLM
llm-game-worker --task tictactoe_v3

# Play Connect Four with OpenAI
llm-game-worker --task connect_four_v3 --client-name openai --model-id gpt-4

# Play Go (9x9 board) with Anthropic
llm-game-worker --task go_v5 --board-size 9 --client-name anthropic --model-id claude-3-sonnet
```

### Python API

```python
from llm_game_worker import LLMGameWorkerRuntime, LLMGameWorkerConfig

# Create config
config = LLMGameWorkerConfig(
    task="tictactoe_v3",
    client_name="openai",
    model_id="gpt-4",
    api_key="your-api-key",
)

# Create runtime
runtime = LLMGameWorkerRuntime(config)

# Initialize for a game
runtime.init_agent("tictactoe_v3", "player_1")

# Select an action
result = runtime.select_action(
    observation="board state",
    legal_moves=[0, 1, 4, 8],
    board_str=" | | \n-+-+-\n |X| \n-+-+-\n | | ",
)

print(f"Selected action: {result['action']}")
```

## Interactive Protocol

The worker reads JSON commands from stdin and writes responses to stdout:

### Commands

```json
// Initialize agent
{"type": "init_agent", "game_name": "tictactoe_v3", "player_id": "player_1"}

// Select action
{"type": "select_action", "observation": "...", "info": {"legal_moves": [0, 1, 4]}}

// Stop worker
{"type": "stop"}
```

### Responses

```json
// Agent initialized
{"type": "agent_initialized", "run_id": "...", "game_name": "...", "player_id": "..."}

// Action selected
{"type": "action_selected", "action": 4, "reasoning": "...", "success": true}

// Stopped
{"type": "stopped", "run_id": "..."}
```

## MOSAIC Integration

The worker registers with MOSAIC via entry points:

```python
from llm_game_worker import get_worker_metadata

metadata, capabilities = get_worker_metadata()
# metadata.name == "LLM Game Worker"
# capabilities.worker_type == "llm_game"
# capabilities.env_families == ("pettingzoo",)
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `task` | `tictactoe_v3` | Game to play |
| `client_name` | `vllm` | LLM client (vllm, openai, anthropic) |
| `model_id` | `Qwen/Qwen2.5-1.5B-Instruct` | Model identifier |
| `base_url` | `http://127.0.0.1:8000/v1` | API endpoint |
| `temperature` | `0.3` | Sampling temperature |
| `max_retries` | `3` | Invalid move retry limit |
| `board_size` | `19` | Go board size |
