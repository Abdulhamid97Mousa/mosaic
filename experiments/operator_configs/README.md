# Operator Configuration Scripts

This directory contains Python scripts that define operator configurations for MOSAIC experiments.

## What Are These Scripts?

Instead of manually clicking through the UI to configure operators, you write a Python script that defines:
- **Operators**: What agents to run (RL, LLM, baseline)
- **Environments**: Which gridworld tasks to test
- **Execution**: How many episodes, seeds, auto-run settings

## Supported Environments

**✅ Valid Environment Types:**
- **MiniGrid** (`env_name: "minigrid"`) - Single-agent gridworld navigation
- **MultiGrid** (`env_name: "multigrid"`) - Multi-agent cooperative gridworld
- **BabyAI** (`env_name: "babyai"`) - Instruction-following in gridworlds
- **MeltingPot** (`env_name: "meltingpot"`) - Multi-agent social dilemmas

**❌ Not Supported:**
- CartPole, MountainCar, or other classic control environments
- This is a **gridworld multi-agent coordination research platform**

## Script Format

```python
# Define operators (agents to run)
operators = [
    {
        "id": "unique_operator_id",           # Unique identifier
        "name": "Display Name",               # Shown in UI
        "env_name": "minigrid",               # Environment family
        "task": "MiniGrid-Empty-8x8-v0",      # Specific task
        "workers": {
            "agent": {                         # Single-agent
                "type": "baseline",            # Worker type
                "behavior": "random",          # Baseline behavior
                "seed": 42,                    # Random seed
            },
        }
    },
]

# Optional: Auto-execution settings
execution = {
    "auto_run": True,                         # Run automatically
    "num_episodes": 100,                      # Episodes to run
    "seeds": range(1000, 1100),               # Seed range
}
```

## Worker Types

| Type | Description | Example Settings |
|------|-------------|------------------|
| `baseline` | Simple baseline (random, noop, cycling) | `{"behavior": "random", "seed": 42}` |
| `rl` | Trained RL policy | `{"policy_path": "/path/model.pt", "algorithm": "ppo"}` |
| `llm` | Language model agent | `{"model_id": "gpt-4o-mini", "client_name": "openai"}` |
| `human` | Human keyboard input | `{"player_name": "Human"}` |

## Example Scripts

### 1. Simple Single-Agent Test
**File:** `simple_random_baseline.py`
**Purpose:** Test baseline operator with MiniGrid
**Environment:** MiniGrid-Empty-8x8-v0 (single-agent)

### 2. Credit Assignment Experiment
**File:** `credit_assignment_experiment.py`
**Purpose:** Compare hybrid teams vs ablation baselines
**Environment:** MultiGrid-RedBlueDoors-6x6-v0 (multi-agent)

## Usage in MOSAIC

1. **Open MOSAIC GUI**
2. **Go to Operators Tab**
3. **Select "Script Mode"** (instead of Manual Mode)
4. **Click "Load Script"** → Select a script from this directory
5. **Click "Validate"** → Check syntax and structure
6. **Click "Apply & Configure"** → Operators launch with visual renders!

## Multi-Agent Configuration

For multi-agent environments (MultiGrid, MeltingPot):

```python
operators = [
    {
        "id": "hybrid_team",
        "name": "RL + LLM Hybrid Team",
        "env_name": "multigrid",
        "task": "MultiGrid-RedBlueDoors-6x6-v0",
        "workers": {
            "agent_0": {
                "type": "rl",
                "policy_path": "/path/to/model.pt",
            },
            "agent_1": {
                "type": "llm",
                "model_id": "gpt-4o-mini",
            },
        }
    },
]
```

## Telemetry Output

When operators run, telemetry is saved to:
- **Steps**: `var/operators/telemetry/{run_id}_steps.jsonl`
- **Episodes**: `var/operators/telemetry/{run_id}_episodes.jsonl`

## Future: Shell Script Export

Currently using Python scripts for GUI integration. Future versions will support exporting to shell scripts (`.sh`) for headless batch execution.

---

**Remember:** This is a **gridworld research platform**. Use MiniGrid, MultiGrid, BabyAI, or MeltingPot only!
