# Advanced Tab UI Reference

## Overview

The **Advanced Tab** is the unified configuration interface for MOSAIC's multi-framework RL system. It provides a 4-step flow for configuring any RL scenario, from single-agent Gymnasium environments to complex multi-agent PettingZoo setups with mixed human/AI agents.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ADVANCED TAB                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Environment Selection                                        │    │
│  │ ┌──────────────┐ ┌────────────────────────┐ ┌────────┐ ┌──────────┐ │    │
│  │ │ Family    ▼  │ │ Environment         ▼  │ │ Seed   │ │  Load    │ │    │
│  │ └──────────────┘ └────────────────────────┘ └────────┘ └──────────┘ │    │
│  │                                                                      │    │
│  │ ┌──────────────────────────────────────────────────────────────────┐│    │
│  │ │ Environment: simple_spread_v3                                    ││    │
│  │ │ Type: Multi-Agent (3 agents)                                     ││    │
│  │ │ Paradigm: Sequential                                             ││    │
│  │ │ Agents: agent_0, agent_1, agent_2                                ││    │
│  │ └──────────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Agent Configuration                                          │    │
│  │ ┌────────────────────────────────────────────────────────────────┐  │    │
│  │ │ ☐ Apply to all: [Actor ▼] [Worker ▼] [Mode ▼] [Apply]          │  │    │
│  │ └────────────────────────────────────────────────────────────────┘  │    │
│  │ ┌──────────┬───────────────┬─────────────┬──────────┐               │    │
│  │ │ Agent    │ Actor/Policy  │ Worker      │ Mode     │               │    │
│  │ ├──────────┼───────────────┼─────────────┼──────────┤               │    │
│  │ │ agent_0  │ [Human     ▼] │ [Local   ▼] │ [Play ▼] │               │    │
│  │ │ agent_1  │ [XuanCe   ▼]  │ [XuanCe ▼]  │ [Train▼] │               │    │
│  │ │ agent_2  │ [XuanCe   ▼]  │ [XuanCe ▼]  │ [Train▼] │               │    │
│  │ └──────────┴───────────────┴─────────────┴──────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: Worker Configuration                                         │    │
│  │ ┌──────────────────────────────────────────────────────────────────┐│    │
│  │ │ Local Execution (used by: agent_0)                               ││    │
│  │ │   Render Mode: [human ▼]    Record Video: [ ]                    ││    │
│  │ └──────────────────────────────────────────────────────────────────┘│    │
│  │ ┌──────────────────────────────────────────────────────────────────┐│    │
│  │ │ XuanCe Worker (used by: agent_1, agent_2)                        ││    │
│  │ │   Algorithm: [MAPPO ▼]      Learning Rate: [0.0005]              ││    │
│  │ │   Batch Size: [256]         Backend: [torch ▼]                   ││    │
│  │ └──────────────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: Run Mode                                                     │    │
│  │   ◉ Interactive (with rendering)                                     │    │
│  │   ○ Headless Training (no rendering)                                 │    │
│  │   ○ Evaluation (load trained policy)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│                              [ Launch Session ]                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Environment Selection

### Purpose
Select the environment and see auto-detected information about agents and stepping paradigm.

### UI Elements

| Element | Type | Description |
|---------|------|-------------|
| Family | Dropdown | Environment family (Gymnasium, PettingZoo MPE, etc.) |
| Environment | Dropdown | Specific environment within family |
| Seed | SpinBox | Random seed (1 - 10,000,000) |
| Load | Button | Load environment and detect agents |
| Info Panel | Label | Shows environment details |

### Info Panel Display

The info panel shows:
- **Environment**: The environment ID (e.g., `simple_spread_v3`)
- **Type**: `Single-Agent` or `Multi-Agent (N agents)` (highlighted in blue)
- **Paradigm**: `Single Agent`, `Sequential`, or `Simultaneous`
- **Agents**: List of agent IDs (truncated if > 5 agents)
- **Description**: Brief description of the environment

### Supported Environment Families

| Family | Example Environments | Agent Count |
|--------|---------------------|-------------|
| **Gymnasium Classic** | CartPole-v1, MountainCar-v0, Acrobot-v1 | 1 |
| **Gymnasium Box2D** | LunarLander-v3, BipedalWalker-v3 | 1 |
| **Gymnasium MuJoCo** | HalfCheetah-v5, Ant-v5, Humanoid-v5 | 1 |
| **PettingZoo Classic** | chess_v6, go_v5, connect_four_v3 | 2 |
| **PettingZoo MPE** | simple_spread_v3, simple_tag_v3 | 2-6 |
| **PettingZoo SISL** | waterworld_v4, pursuit_v4, multiwalker_v9 | 3-8 |
| **PettingZoo Butterfly** | pistonball_v6, knights_archers_zombies_v10 | 2-20 |
| **PettingZoo Atari** | pong_v3, space_invaders_v2 | 2 |
| **ViZDoom** | ViZDoom-Basic-v0, ViZDoom-DefendTheCenter-v0 | 1 |
| **MiniGrid** | MiniGrid-Empty-5x5-v0, MiniGrid-DoorKey-5x5-v0 | 1 |

---

## Step 2: Agent Configuration

### Purpose
Configure which policy and worker controls each agent in the environment.

### UI Elements

| Element | Type | Description |
|---------|------|-------------|
| Bulk Apply Checkbox | CheckBox | Enable bulk configuration |
| Bulk Actor | Dropdown | Actor to apply to all agents |
| Bulk Worker | Dropdown | Worker to apply to all agents |
| Bulk Mode | Dropdown | Mode to apply to all agents |
| Apply Button | Button | Apply bulk settings |
| Agent Table | Table | Per-agent configuration rows |

### Available Actors/Policies

| ID | Display Name | Description | Best For |
|----|--------------|-------------|----------|
| `human_keyboard` | Human (Keyboard) | Human player using keyboard | Interactive play |
| `random` | Random | Uniform random actions | Baseline testing |
| `cleanrl_ppo` | CleanRL PPO | PPO from CleanRL | Single-agent continuous |
| `cleanrl_dqn` | CleanRL DQN | DQN from CleanRL | Single-agent discrete |
| `cleanrl_sac` | CleanRL SAC | SAC from CleanRL | Single-agent continuous |
| `rllib_ppo` | RLlib PPO | PPO from Ray RLlib | Distributed training |
| `rllib_dqn` | RLlib DQN | DQN from Ray RLlib | Distributed training |
| `xuance_mappo` | XuanCe MAPPO | Multi-Agent PPO | Cooperative MARL |
| `xuance_maddpg` | XuanCe MADDPG | Multi-Agent DDPG | Competitive MARL |
| `xuance_qmix` | XuanCe QMIX | QMIX value decomposition | Cooperative MARL |
| `stockfish` | Stockfish | Chess engine | Chess only |
| `llm` | LLM Agent | Language model | Experimental |
| `bdi` | BDI Agent | Belief-Desire-Intention | Goal-driven agents |

### Available Workers

| ID | Display Name | Description | Paradigms |
|----|--------------|-------------|-----------|
| `local` | Local | Run in main process | All |
| `cleanrl` | CleanRL | CleanRL training | Single-Agent |
| `rllib` | Ray RLlib | Distributed RL | All |
| `xuance` | XuanCe | Multi-agent RL | Multi-Agent |
| `llm` | LLM | Language model | All |
| `jason` | Jason BDI | AgentSpeak BDI | Sequential |
| `spade_bdi` | SPADE BDI | Python BDI | Sequential |

### Available Modes

| ID | Display Name | Description |
|----|--------------|-------------|
| `play` | Play | Interactive play, no training |
| `train` | Train | Training mode, policy updates |
| `eval` | Evaluate | Evaluation, frozen policy |
| `frozen` | Frozen | Frozen snapshot for self-play |

---

## Step 3: Worker Configuration

### Purpose
Configure worker-specific parameters. Shows only workers that are actually selected in Step 2.

### Worker Configuration Schemas

#### Local Execution
*For running without an external worker*

| Parameter | Type | Default | Options/Range |
|-----------|------|---------|---------------|
| Render Mode | Choice | `human` | human, rgb_array, ansi, none |
| Record Video | Bool | `false` | - |

#### CleanRL Worker
*Single-agent RL training with CleanRL algorithms*

| Parameter | Type | Default | Options/Range |
|-----------|------|---------|---------------|
| Algorithm | Choice | `PPO` | PPO, DQN, A2C, SAC, TD3, DDPG |
| Learning Rate | Float | `0.0003` | 0.000001 - 1.0 |
| Total Timesteps | Int | `100000` | 1,000 - 10,000,000 |
| Parallel Envs | Int | `4` | 1 - 64 |
| Capture Video | Bool | `false` | - |

#### Ray RLlib Worker
*Distributed RL training with Ray RLlib*

| Parameter | Type | Default | Options/Range |
|-----------|------|---------|---------------|
| Algorithm | Choice | `PPO` | PPO, DQN, A2C, IMPALA, APPO, SAC |
| Num Workers | Int | `2` | 0 - 64 |
| Envs per Worker | Int | `1` | 1 - 16 |
| Framework | Choice | `torch` | torch, tf2 |

#### XuanCe Worker
*Multi-agent RL training with XuanCe MARL algorithms*

| Parameter | Type | Default | Options/Range |
|-----------|------|---------|---------------|
| Algorithm | Choice | `MAPPO` | MAPPO, MADDPG, QMIX, VDN, COMA, IPPO, IQL |
| Learning Rate | Float | `0.0005` | 0.000001 - 1.0 |
| Batch Size | Int | `256` | 32 - 4096 |
| Backend | Choice | `torch` | torch, tensorflow, mindspore |

#### LLM Worker
*Language model-based decision making*

| Parameter | Type | Default | Options/Range |
|-----------|------|---------|---------------|
| Model | Choice | `gpt-4` | gpt-4, gpt-3.5-turbo, claude-3, llama-3, ollama-local |
| Temperature | Float | `0.7` | 0.0 - 2.0 |
| Max Tokens | Int | `256` | 16 - 4096 |
| System Prompt | Text | "You are an RL agent..." | - |

#### Jason BDI Worker
*Belief-Desire-Intention agents with AgentSpeak*

| Parameter | Type | Default |
|-----------|------|---------|
| Agent File (.asl) | Text | `agent.asl` |
| MAS File (.mas2j) | Text | `project.mas2j` |
| Debug Mode | Bool | `false` |

#### SPADE BDI Worker
*Python-based BDI agents with SPADE framework*

| Parameter | Type | Default |
|-----------|------|---------|
| XMPP Server | Text | `localhost` |
| Agent JID | Text | `agent@localhost` |
| Debug Mode | Bool | `false` |

---

## Step 4: Run Mode Selection

### Purpose
Select how to execute the configured session.

### Available Modes

| Mode | Icon | Description | Use Case |
|------|------|-------------|----------|
| **Interactive** | ▶️ | Full visualization, real-time rendering | Human play, debugging, demonstrations |
| **Headless Training** | 🚀 | No rendering, maximum speed | Long training runs |
| **Evaluation** | 📊 | Load trained policy, with rendering | Testing trained models |

---

## Launch Configuration

When the user clicks **Launch Session**, the Advanced Tab emits a `LaunchConfig` with all settings:

```python
@dataclass
class LaunchConfig:
    env_id: str                              # e.g., "simple_spread_v3"
    seed: int                                # e.g., 42
    paradigm: SteppingParadigm               # SEQUENTIAL, SIMULTANEOUS, etc.
    agent_bindings: Dict[str, AgentRowConfig]  # Per-agent config
    worker_configs: Dict[str, Dict[str, Any]]  # Per-worker config
    run_mode: RunMode                        # INTERACTIVE, HEADLESS, EVALUATION
```

---

## Example Configurations

### Example 1: Human vs AI in Chess

```
Environment: chess_v6 (PettingZoo Classic)
Type: Multi-Agent (2 agents)

Agent Configuration:
  player_0: Human (Keyboard) | Local    | Play
  player_1: Stockfish        | Local    | Play

Worker Config:
  Local: render_mode=human, record_video=false

Run Mode: Interactive
```

### Example 2: MARL Training on Simple Spread

```
Environment: simple_spread_v3 (PettingZoo MPE)
Type: Multi-Agent (3 agents)

Agent Configuration:
  agent_0: XuanCe MAPPO | XuanCe | Train
  agent_1: XuanCe MAPPO | XuanCe | Train
  agent_2: XuanCe MAPPO | XuanCe | Train

Worker Config:
  XuanCe: algorithm=MAPPO, learning_rate=0.0005, batch_size=256, backend=torch

Run Mode: Headless Training
```

### Example 3: Human + LLM Cooperation

```
Environment: cooperative_pong_v5 (PettingZoo Butterfly)
Type: Multi-Agent (2 agents)

Agent Configuration:
  paddle_0: Human (Keyboard) | Local | Play
  paddle_1: LLM Agent        | LLM   | Play

Worker Config:
  Local: render_mode=human
  LLM: model=gpt-4, temperature=0.7, max_tokens=256

Run Mode: Interactive
```

### Example 4: Self-Play Training

```
Environment: connect_four_v3 (PettingZoo Classic)
Type: Multi-Agent (2 agents)

Agent Configuration:
  player_0: CleanRL DQN | CleanRL | Train
  player_1: CleanRL DQN | CleanRL | Frozen

Worker Config:
  CleanRL: algorithm=DQN, learning_rate=0.0001, total_timesteps=500000

Run Mode: Headless Training
```

---

## File Locations

| Component | File Path |
|-----------|-----------|
| AdvancedConfigTab | `gym_gui/ui/widgets/advanced_config/advanced_config_tab.py` |
| EnvironmentSelector | `gym_gui/ui/widgets/advanced_config/environment_selector.py` |
| AgentConfigTable | `gym_gui/ui/widgets/advanced_config/agent_config_table.py` |
| WorkerConfigPanel | `gym_gui/ui/widgets/advanced_config/worker_config_panel.py` |
| RunModeSelector | `gym_gui/ui/widgets/advanced_config/run_mode_selector.py` |
| Module __init__ | `gym_gui/ui/widgets/advanced_config/__init__.py` |

---

## Related Documentation

- [TASK_3 README](./README.md) - UI Architecture overview
- [UI Migration Plan](./01_ui_migration_plan.md) - Migration strategy
- [Advanced Tab Components](./02_advanced_tab_components.md) - Technical component details
- [TASK_1: PolicyMappingService](../TASK_1/03_policy_mapping_service_plan.md) - Backend service
- [Worker Requirements](../TASK_1/04_worker_requirements.md) - Worker installation guide
