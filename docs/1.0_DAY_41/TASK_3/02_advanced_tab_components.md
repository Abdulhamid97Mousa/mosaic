# Advanced Tab UI Components

## Overview

The Advanced Tab provides the **Unified Flow** for configuring any RL scenario. It consists of 4 steps that guide users through environment selection, agent configuration, worker setup, and run mode selection.

## Component Architecture

```
AdvancedConfigTab (QWidget)
├── EnvironmentSelector (Step 1)
│   ├── Family dropdown
│   ├── Environment dropdown
│   ├── Seed spinner
│   ├── Load button
│   └── Info panel (shows paradigm, agent count, description)
│
├── AgentConfigTable (Step 2)
│   ├── Bulk apply section
│   ├── Table header (Agent, Actor/Policy, Worker, Mode)
│   └── Per-agent rows (dynamically generated)
│
├── WorkerConfigPanel (Step 3)
│   └── Dynamic config forms per worker type
│
├── RunModeSelector (Step 4)
│   ├── Interactive (with rendering)
│   ├── Headless Training (no rendering)
│   └── Evaluation (load trained policy)
│
└── Launch button
```

## Step 1: Environment Selector

**File:** `gym_gui/ui/widgets/advanced_config/environment_selector.py`

### Features

- **Family Selection**: Groups environments by source (Gymnasium, PettingZoo MPE, PettingZoo SISL, etc.)
- **Auto-detection**: Automatically detects stepping paradigm and agent list
- **Info Panel**: Shows environment details including:
  - Environment ID
  - Type: "Single-Agent" or "Multi-Agent (N agents)"
  - Paradigm: SINGLE_AGENT, SEQUENTIAL, SIMULTANEOUS
  - Agent list (truncated for environments with many agents)
  - Description

### Signals

```python
environment_changed = pyqtSignal(str)      # env_id
paradigm_detected = pyqtSignal(object)     # SteppingParadigm
agents_detected = pyqtSignal(list)         # List[str] agent IDs
```

### Supported Environment Families

| Family | Environments | Paradigm |
|--------|-------------|----------|
| Gymnasium Classic | CartPole, MountainCar, Acrobot | SINGLE_AGENT |
| Gymnasium Box2D | LunarLander, BipedalWalker | SINGLE_AGENT |
| Gymnasium MuJoCo | HalfCheetah, Ant, Humanoid | SINGLE_AGENT |
| PettingZoo Classic | Chess, Go, Connect Four, TicTacToe | SEQUENTIAL |
| PettingZoo MPE | Simple Spread, Simple Tag, Simple Adversary, etc. | SEQUENTIAL |
| PettingZoo SISL | Multiwalker, Pursuit, Waterworld | SEQUENTIAL |
| PettingZoo Butterfly | Pistonball, Knights Archers Zombies, Cooperative Pong | SEQUENTIAL |
| PettingZoo Atari | Pong, Space Invaders, Tennis | SEQUENTIAL |
| ViZDoom | Basic, Defend The Center, Deadly Corridor | SINGLE_AGENT |
| MiniGrid | Empty, DoorKey, LavaGap | SINGLE_AGENT |

## Step 2: Agent Configuration Table

**File:** `gym_gui/ui/widgets/advanced_config/agent_config_table.py`

### Features

- **Per-agent Configuration**: Each agent gets its own row with:
  - Actor/Policy selection
  - Worker selection
  - Mode selection (play/train/eval/frozen)
- **Bulk Apply**: Apply same config to all agents at once
- **Dynamic Rows**: Rows generated based on environment's agent list

### Available Actors/Policies

| ID | Label | Description |
|----|-------|-------------|
| human_keyboard | Human (Keyboard) | Human player using keyboard input |
| random | Random | Uniform random action selection |
| cleanrl_ppo | CleanRL PPO | PPO policy from CleanRL |
| cleanrl_dqn | CleanRL DQN | DQN policy from CleanRL |
| cleanrl_sac | CleanRL SAC | SAC policy from CleanRL |
| rllib_ppo | RLlib PPO | PPO policy from Ray RLlib |
| rllib_dqn | RLlib DQN | DQN policy from Ray RLlib |
| xuance_mappo | XuanCe MAPPO | Multi-Agent PPO from XuanCe |
| xuance_maddpg | XuanCe MADDPG | Multi-Agent DDPG from XuanCe |
| xuance_qmix | XuanCe QMIX | QMIX from XuanCe |
| stockfish | Stockfish | Stockfish chess engine (Chess only) |
| llm | LLM Agent | Language model decision-maker |
| bdi | BDI Agent | Belief-Desire-Intention agent |

### Available Workers

| ID | Label | Description |
|----|-------|-------------|
| local | Local | Run in main process (no worker) |
| cleanrl | CleanRL | CleanRL single-agent RL training |
| rllib | Ray RLlib | Distributed RL with Ray RLlib |
| xuance | XuanCe | Multi-agent RL with XuanCe MARL |
| llm | LLM | Language model decision-making |
| jason | Jason BDI | AgentSpeak BDI agents |
| spade_bdi | SPADE BDI | Python BDI with SPADE |

### Available Modes

| ID | Label | Description |
|----|-------|-------------|
| play | Play | Interactive play (no training) |
| train | Train | Training mode |
| eval | Evaluate | Evaluation mode (frozen policy) |
| frozen | Frozen | Frozen snapshot (for self-play) |

## Step 3: Worker Configuration Panel

**File:** `gym_gui/ui/widgets/advanced_config/worker_config_panel.py`

### Features

- **Dynamic Forms**: Configuration forms generated from schemas
- **Per-worker Sections**: Each unique worker type gets its own section
- **Real-time Updates**: Updates when agent bindings change

### Worker Configuration Schemas

#### Local Execution
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| render_mode | choice | human | Rendering mode |
| record_video | bool | false | Record video |

#### CleanRL Worker
| Field | Type | Default | Range |
|-------|------|---------|-------|
| algorithm | choice | PPO | PPO, DQN, A2C, SAC, TD3, DDPG |
| learning_rate | float | 0.0003 | 0.000001 - 1.0 |
| total_timesteps | int | 100000 | 1000 - 10000000 |
| num_envs | int | 4 | 1 - 64 |
| capture_video | bool | false | - |

#### Ray RLlib Worker
| Field | Type | Default | Range |
|-------|------|---------|-------|
| algorithm | choice | PPO | PPO, DQN, A2C, IMPALA, APPO, SAC |
| num_workers | int | 2 | 0 - 64 |
| num_envs_per_worker | int | 1 | 1 - 16 |
| framework | choice | torch | torch, tf2 |

#### XuanCe Worker
| Field | Type | Default | Range |
|-------|------|---------|-------|
| algorithm | choice | MAPPO | MAPPO, MADDPG, QMIX, VDN, COMA, IPPO, IQL |
| learning_rate | float | 0.0005 | 0.000001 - 1.0 |
| batch_size | int | 256 | 32 - 4096 |
| backend | choice | torch | torch, tensorflow, mindspore |

#### LLM Worker
| Field | Type | Default |
|-------|------|---------|
| model | choice | gpt-4 |
| temperature | float | 0.7 |
| max_tokens | int | 256 |
| system_prompt | text | "You are an RL agent..." |

#### Jason BDI Worker
| Field | Type | Default |
|-------|------|---------|
| agent_file | text | agent.asl |
| mas_file | text | project.mas2j |
| debug_mode | bool | false |

#### SPADE BDI Worker
| Field | Type | Default |
|-------|------|---------|
| xmpp_server | text | localhost |
| agent_jid | text | agent@localhost |
| debug_mode | bool | false |

## Step 4: Run Mode Selector

**File:** `gym_gui/ui/widgets/advanced_config/run_mode_selector.py`

### Available Modes

| Mode | Description |
|------|-------------|
| **Interactive** | Run with full visualization. Use for human play, demonstrations, or debugging trained agents. |
| **Headless Training** | Maximum training speed without visualization. Telemetry and metrics still collected. |
| **Evaluation** | Load a trained policy and evaluate with rendering. No training updates applied. |

## Signal Flow

```
EnvironmentSelector
    │
    │ environment_changed(env_id)
    │ agents_detected([agent_ids])
    ▼
AgentConfigTable.set_agents([agent_ids])
    │
    │ bindings_changed({agent_id: AgentRowConfig})
    ▼
WorkerConfigPanel.update_from_bindings(bindings)
    │
    │ config_changed({worker_id: config})
    ▼
AdvancedConfigTab._update_launch_button_state()
    │
    │ launch_requested(LaunchConfig)
    ▼
MainWindow._on_advanced_launch(config)
```

## LaunchConfig Dataclass

```python
@dataclass
class LaunchConfig:
    """Complete configuration for launching a session from Advanced tab."""
    env_id: str
    seed: int
    paradigm: SteppingParadigm
    agent_bindings: Dict[str, AgentRowConfig]
    worker_configs: Dict[str, Dict[str, Any]]
    run_mode: RunMode
```

## Usage Example

1. **Select Environment**: Choose "PettingZoo MPE" → "simple_spread_v3"
   - Info panel shows: "Multi-Agent (3 agents): agent_0, agent_1, agent_2"
   - Paradigm: SEQUENTIAL

2. **Configure Agents**:
   - agent_0: Human (Keyboard), Local, Play
   - agent_1: XuanCe MAPPO, XuanCe, Train
   - agent_2: XuanCe MAPPO, XuanCe, Train

3. **Configure Worker**:
   - XuanCe Worker appears with fields:
     - Algorithm: MAPPO
     - Learning Rate: 0.0005
     - Batch Size: 256
     - Backend: torch

4. **Select Run Mode**: Interactive (with rendering)

5. **Launch**: Click "Launch Session" button

## Related Documents

- [TASK_3 README](./README.md) - UI Architecture overview
- [UI Migration Plan](./01_ui_migration_plan.md) - Migration strategy
- [TASK_1 README](../TASK_1/README.md) - Backend PolicyMappingService
