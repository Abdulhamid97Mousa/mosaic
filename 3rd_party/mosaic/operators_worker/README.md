# Operators Worker - Baseline Operators for Credit Assignment

Simple baseline operators for ablation studies and credit assignment research in hybrid agent teams.

## Overview

The `operators_worker` provides three baseline operators:

| Operator | Behavior | Use Case |
|----------|----------|----------|
| **RandomOperator** | Uniformly random actions | Measure trained agent contribution |
| **NoopOperator** | Always action 0 (stay/no-op) | Extreme ablation - can trained agent compensate? |
| **CyclingOperator** | Cycles through actions (0, 1, 2, ..., n-1, repeat) | Systematic exploration baseline |

## Key Features

- **Dynamic Action Spaces**: Operators configure themselves from environment at runtime (no hardcoded action counts)
- **Multi-Operator Support**: Run multiple operators simultaneously as separate subprocesses
- **Telemetry Logging**: Automatic JSONL logging to `var/operators/telemetry/` for post-hoc analysis
- **Visual Feedback**: Integrates with MOSAIC UI for real-time visualization
- **Reproducibility**: Seed control for deterministic experiments

## Installation

```bash
# From the GUI_BDI_RL root directory
cd 3rd_party/mosaic/operators_worker
pip install -e .

# Install with environment dependencies
pip install -e ".[babyai]"     # For BabyAI environments
pip install -e ".[minigrid]"   # For MiniGrid environments
pip install -e ".[multigrid]"  # For MultiGrid environments
```

## Usage

### As a Subprocess (GUI Integration)

The primary use case is as a subprocess controlled by the MOSAIC GUI:

```bash
python -m operators_worker \
    --run-id operator_0_abc123 \
    --behavior random \
    --task BabyAI-GoToRedBall-v0 \
    --interactive
```

#### Interactive Protocol

The worker reads JSON commands from stdin and emits JSON responses to stdout:

**Commands (stdin)**:
```json
{"cmd": "reset", "seed": 42, "env_name": "babyai", "task": "BabyAI-GoToRedBall-v0"}
{"cmd": "step"}
{"cmd": "stop"}
```

**Responses (stdout)**:
```json
{"type": "init", "run_id": "operator_0_abc123"}
{"type": "ready", "render_payload": {"rgb_array": [...], "shape": [64, 64, 3]}}
{"type": "step", "reward": 0.0, "terminated": false, "render_payload": {...}}
{"type": "episode_end", "episode_return": 10.0, "episode_steps": 42}
```

### As a Python Library

You can also use the operators directly in Python code:

```python
import gymnasium as gym
from operators_worker.operators import RandomOperator

# Create environment
env = gym.make("BabyAI-GoToRedBall-v0")

# Create operator
operator = RandomOperator()
operator.set_action_space(env.action_space)

# Run episode
obs, info = env.reset(seed=42)
operator.reset(seed=42)

done = False
while not done:
    action = operator.select_action(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    operator.on_step_result(obs, reward, terminated, truncated, info)
    done = terminated or truncated

# Analyze trajectory
trajectory = operator.get_trajectory()
episode_return = operator.get_episode_return()
print(f"Episode return: {episode_return}")
```

### Factory Function

Use the factory for convenient operator creation:

```python
from operators_worker.operators import create_baseline_operator

# Create random operator
operator = create_baseline_operator("random", operator_id="op_001")
operator.set_action_space(env.action_space)

# Create no-op operator
operator = create_baseline_operator("noop", operator_id="op_002")
operator.set_action_space(env.action_space)

# Create cycling operator
operator = create_baseline_operator("cycling", operator_id="op_003")
operator.set_action_space(env.action_space)
```

## Configuration

The `OperatorsWorkerConfig` dataclass controls worker behavior:

```python
from operators_worker.config import OperatorsWorkerConfig

config = OperatorsWorkerConfig(
    run_id="operator_0_abc123",          # Unique identifier
    behavior="random",                    # "random", "noop", or "cycling"
    env_name="babyai",                    # Environment family
    task="BabyAI-GoToRedBall-v0",        # Specific task
    telemetry_dir=None,                   # Auto-resolved to var/operators/telemetry
    emit_jsonl=True,                      # Enable telemetry logging
    seed=42,                              # Random seed (optional)
    interactive=True,                     # Interactive mode (stdin/stdout)
)
```

## Telemetry Output

When `emit_jsonl=True`, the worker writes two JSONL files to `var/operators/telemetry/`:

### Steps File: `{run_id}_steps.jsonl`

Per-step data for detailed analysis:

```json
{"run_id": "operator_0_abc123", "episode": 0, "step": 0, "action": 2, "reward": 0.0, "terminated": false, "truncated": false}
{"run_id": "operator_0_abc123", "episode": 0, "step": 1, "action": 5, "reward": 0.0, "terminated": false, "truncated": false}
{"run_id": "operator_0_abc123", "episode": 0, "step": 2, "action": 1, "reward": 1.0, "terminated": true, "truncated": false}
```

### Episodes File: `{run_id}_episodes.jsonl`

Per-episode summaries:

```json
{"run_id": "operator_0_abc123", "episode": 0, "seed": 42, "return": 1.0, "steps": 3, "success": true}
{"run_id": "operator_0_abc123", "episode": 1, "seed": 43, "return": 0.0, "steps": 100, "success": false}
```

## Multi-Operator Experiments

Run multiple operators simultaneously for comparison:

```bash
# Terminal 1: Random baseline
python -m operators_worker \
    --run-id op_rand_001 \
    --behavior random \
    --task MultiGrid-RedBlueDoors-6x6-v0 \
    --interactive

# Terminal 2: No-op baseline
python -m operators_worker \
    --run-id op_noop_001 \
    --behavior noop \
    --task MultiGrid-RedBlueDoors-6x6-v0 \
    --interactive

# Terminal 3: Cycling baseline
python -m operators_worker \
    --run-id op_cycle_001 \
    --behavior cycling \
    --task MultiGrid-RedBlueDoors-6x6-v0 \
    --interactive
```

Each operator gets:
- Independent subprocess
- Own telemetry files in `var/operators/telemetry/`
- Own render view in GUI

## Architecture

### Subprocess Communication

```
GUI (MainWindow)
  |
  +-- launch_operator("operator_0_abc123") ----+
  |                                             |
  +-- stdin: {"cmd": "reset", "seed": 42} -----+--> Worker
  |                                             |    Subprocess
  +-- stdin: {"cmd": "step"} ------------------+--> (Python)
  |                                             |
  +-- stdout: {"type": "step", ...} <----------+
  |    |
  |    +-> OperatorRenderContainer
  |        +-> Display RGB array
  |
  +-- QTimer.singleShot(50ms) --> next step (paced)
```

### Automatic Execution Flow (Script Mode)

The GUI uses `OperatorScriptExecutionManager` to drive automatic multi-episode execution:

```
ScriptExperimentWidget (UI: Browse Script, Run/Stop, Step Delay SpinBox)
  |
  +-> OperatorScriptExecutionManager (state machine)
  |     |
  |     +-> launch_operator signal ---> MainWindow launches subprocess
  |     +-> step_operator signal -----> MainWindow sends step via stdin
  |     +-> reset_operator signal ----> MainWindow sends reset via stdin
  |     +-> stop_operator signal -----> MainWindow terminates subprocess
  |     |
  |     +-- on_step_received() -------> QTimer.singleShot(step_delay_ms)
  |     |                                  +-> step_operator.emit() (paced)
  |     |
  |     +-- on_episode_ended() -------> advance seed, emit reset_operator
  |     +-- experiment_completed -----> all episodes done
  |
  +-> MainWindow._handle_operator_response() routes stdout back to manager
```

**Step pacing**: A configurable delay (default 50ms) between steps prevents the
step loop from starving Qt paint events. Without pacing, steps fire faster than
the UI can repaint, causing visual jitter (e.g. step 26 jumps to 37).

The delay is adjustable in real-time via the "Step delay (ms)" SpinBox:
- 0 ms = fastest (may skip frames visually)
- 50 ms = smooth frame-by-frame rendering (default)
- 200 ms = slow-motion for analysis

## Research Use Cases

### Credit Assignment in Hybrid Teams

Compare performance of:
1. **RL + LLM**: Hybrid team
2. **RL + Random**: Ablation (measures RL contribution)
3. **RL + No-op**: Extreme ablation (can RL compensate?)
4. **Random + Random**: Double baseline (environment difficulty)

```python
# Analysis script
import json

def analyze_episode(steps_file):
    """Analyze who contributed to success."""
    with open(steps_file) as f:
        steps = [json.loads(line) for line in f]

    # Who got positive rewards?
    agent_0_rewards = sum(s["reward"] for s in steps if s["agent"] == "agent_0")
    agent_1_rewards = sum(s["reward"] for s in steps if s["agent"] == "agent_1")

    return agent_0_rewards, agent_1_rewards

# Compare hybrid vs ablation
rl_llm = analyze_episode("var/operators/telemetry/hybrid_rl_llm_steps.jsonl")
rl_rand = analyze_episode("var/operators/telemetry/ablation_rl_random_steps.jsonl")

print(f"RL+LLM return: {sum(rl_llm)}")
print(f"RL+Random return: {sum(rl_rand)}")
print(f"LLM contribution: {sum(rl_llm) - sum(rl_rand)}")
```

## Operator Protocol

All operators implement the `Operator` protocol from `gym_gui/services/operator.py`:

```python
class Operator(Protocol):
    """Universal agent interface."""

    def select_action(self, observation, info=None) -> Any:
        """Select action based on observation."""
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset operator state for new episode."""
        ...

    def on_step_result(
        self, observation, reward, terminated, truncated, info
    ) -> None:
        """Process step result (for trajectory tracking)."""
        ...
```

This protocol is framework-agnostic and can be implemented by:
- Baseline operators (this package)
- RL policies (CleanRL, XuanCe)
- LLM agents (OpenAI, Anthropic)
- BDI reasoning systems
- Human keyboard input

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run operator worker tests
pytest tests/

# Run Script Mode execution manager tests (uses pytest-qt)
pytest gym_gui/tests/test_script_execution_manager.py -v

# With coverage
pytest --cov=operators_worker --cov-report=html tests/
```

### Testing Qt Signals with pytest-qt

The execution manager tests use **pytest-qt** (`qtbot.waitSignal`) to properly test
deferred signals from `QTimer.singleShot` (step pacing). This avoids hardcoded sleeps
and manual `processEvents()` calls:

```python
def test_on_step_triggers_next_step(execution_manager, qtbot):
    # ... setup ...
    with qtbot.waitSignal(execution_manager.step_operator, timeout=1000) as blocker:
        execution_manager.on_step_received("op1")
    assert blocker.signal_triggered
```

### Project Structure

```
operators_worker/
├── operators_worker/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # Python -m entry point
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration dataclass
│   ├── operators.py         # Baseline operator implementations
│   └── runtime.py           # Subprocess runtime + telemetry
├── tests/
│   ├── test_operators.py    # Operator unit tests
│   ├── test_runtime.py      # Runtime integration tests
│   └── test_cli.py          # CLI tests
├── pyproject.toml           # Package metadata
└── README.md                # This file
```

## License

MIT License - Part of the MOSAIC project

## Citation

If you use this in your research, please cite:

```bibtex
@software{operators_worker,
  title = {Operators Worker: Baseline Operators for Credit Assignment},
  author = {MOSAIC Project},
  year = {2026},
  url = {https://github.com/your-org/GUI_BDI_RL/tree/main/3rd_party/mosaic/operators_worker}
}
```
