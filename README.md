# MOSAIC

**Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue)](https://www.gymlibrary.dev/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![License](https://img.shields.io/github/license/Abdulhamid97Mousa/MOSAIC)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue)](https://mosaic-multi-agent-orchestration-system.readthedocs.io/)

A unified platform that orchestrates diverse agents, paradigms, and workers to create cohesive intelligent systems — like tiles in a mosaic forming a complete picture.

**[Read the Documentation](https://mosaic-multi-agent-orchestration-system.readthedocs.io/)**

![MOSAIC Platform](docs/source/images/Platform_Main_View.png)

## Why MOSAIC?

Today's AI landscape offers powerful but **fragmented** tools: RL frameworks (CleanRL, RLlib, XuanCe), symbolic AI (Jason BDI, SPADE), language models (GPT, Claude), robotics simulators (MuJoCo), and 3D game engines (Godot). Each excels in isolation, but **no platform bridges them together** under a unified, visual-first interface.

### The Problem

| Domain | Framework | What It Offers | What It Lacks |
|--------|-----------|----------------|---------------|
| **RL** | CleanRL, RLlib, XuanCe | Neural policy training | BDI reasoning, LLM planning |
| **Symbolic AI** | Jason, SPADE | Goal-driven BDI agents | Neural learning, 3D environments |
| **LLM** | GPT, Claude | Natural language reasoning | RL training loops, real-time control |
| **Robotics** | MuJoCo MPC | Physics simulation, MPC | Multi-agent coordination |
| **3D Simulation** | Godot, AirSim | Rich game/drone environments | RL integration, agent management |

### The MOSAIC Solution

MOSAIC is a **visual-first orchestration platform** that bridges these domains:

- **Unified Framework Bridge**: Connect RL, LLM, BDI, Robotics, and 3D Simulation in a single platform
- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface — no code required for setup
- **Heterogeneous Agent Mixing**: Run Human + RL + BDI + LLM agents in the same environment
- **Resource Management & Quotas**: GPU allocation, queue limits, credit-based backpressure, and health monitoring
- **Per-Agent Policy Binding**: Route each agent to different workers (CleanRL, Jason, LLM) via `PolicyMappingService`
- **Worker Lifecycle Orchestration**: Subprocess management with heartbeat monitoring and graceful termination

## Supported Paradigms

| Paradigm | Description | Example Environments |
|----------|-------------|---------------------|
| **Gymnasium** | Standard single-agent RL API | CartPole, MountainCar, Atari, MuJoCo |
| **PettingZoo AEC** | Turn-based multi-agent (Alternating Environment Cycle) | Chess, Go, Tic-Tac-Toe, Connect Four |
| **PettingZoo Parallel** | Simultaneous multi-agent environments | MPE, SISL, Butterfly |
| **MiniGrid** | Procedural grid-world environments | Empty, DoorKey, MultiRoom, RedBlueDoors |
| **ViZDoom** | Doom-based visual RL platform | Basic, Deadly Corridor, Defend the Center |
| **MuJoCo MPC** | Model Predictive Control for robotics | Humanoid, Quadruped, Manipulation tasks |
| **Godot UE** | Custom 3D game environments | Custom game AI training scenarios |
| **ALE Atari** | Arcade Learning Environment | Breakout, Pong, Space Invaders |

## Supported Workers

MOSAIC integrates multiple training backends through a worker architecture:

| Worker | Type | Capabilities |
|--------|------|--------------|
| **CleanRL** | Neural RL | Single-file implementations (PPO, DQN, SAC, TD3) |
| **XuanCe** | Multi-agent RL | MAPPO, QMIX, MADDPG algorithms |
| **RLlib** | Distributed RL | Scalable training with Ray |
| **Jason BDI** | Symbolic AI | AgentSpeak agents via Java/gRPC bridge |
| **MuJoCo MPC** | Control | Model Predictive Control worker |
| **Godot** | 3D Engine | Game AI training integration |
| **LLM** | Language Models | GPT/Claude agents (planned) |

## Architecture

MOSAIC follows a layered architecture:

**Visual Layer** → **Service Layer** → **Adapter Layer** ↔ **Workers**

| Layer | Components | Purpose |
|-------|------------|---------|
| Visual | MainWindow, ControlPanel, RenderTabs | PyQt6 interface |
| Service | PolicyMappingService, ActorService, TelemetryService | Business logic |
| Adapter | ParadigmAdapter, PettingZooAdapter | Environment normalization |
| Workers | CleanRL, XuanCe, RLlib, Jason BDI, LLM | Training backends (gRPC/IPC) |

## Installation

### Prerequisites

- Python 3.10+
- PyQt6
- CUDA-capable GPU (optional, for neural training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
cd MOSAIC

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python -m gym_gui
```

### Optional Dependencies

```bash
# For ViZDoom environments
pip install vizdoom

# For MuJoCo environments
pip install mujoco

# For PettingZoo games
pip install pettingzoo[classic]

# For CleanRL training
pip install cleanrl
```

## Quick Start

### Launch the GUI

```bash
python -m gym_gui
```

### Programmatic Usage

```python
from gym_gui.services import PolicyMappingService
from gym_gui.core.enums import SteppingParadigm

# Configure heterogeneous agents for a Chess game
policy_service = PolicyMappingService()
policy_service.set_paradigm(SteppingParadigm.SEQUENTIAL)

# Player 0: Human control
policy_service.bind_agent_policy("player_0", "human_keyboard")

# Player 1: Trained RL policy
policy_service.bind_agent_policy("player_1", "cleanrl_ppo")
```

### Multi-Paradigm Example

```python
from gym_gui.core.enums import SteppingParadigm
from gym_gui.core.adapters import PettingZooAdapter

# AEC (turn-based) for Chess
adapter = PettingZooAdapter("chess_v6", paradigm=SteppingParadigm.SEQUENTIAL)
adapter.reset()

for agent in adapter.agent_iter():
    obs, reward, terminated, truncated, info = adapter.last()
    if terminated or truncated:
        action = None
    else:
        action = policy_service.get_action(agent, obs)
    adapter.step(action)

# Parallel (simultaneous) for MPE
adapter = PettingZooAdapter("simple_spread_v3", paradigm=SteppingParadigm.SIMULTANEOUS)
obs, infos = adapter.reset()

while not all(adapter.terminations.values()):
    actions = {agent: policy_service.get_action(agent, obs[agent])
               for agent in adapter.agents}
    obs, rewards, terminations, truncations, infos = adapter.step(actions)
```

## Core Features

### Visual-First Configuration
Configure complex multi-agent experiments through an intuitive PyQt6 interface. No code required for environment setup, agent configuration, or training launch.

### Framework Bridge (RL + LLM + BDI + Robotics + 3D)
Unified platform connecting:
- **RL**: CleanRL, RLlib, XuanCe for neural policy training
- **Symbolic AI**: Jason BDI (via gRPC), SPADE for goal-driven agents
- **LLM**: GPT/Claude agents for natural language reasoning (planned)
- **Robotics**: MuJoCo MPC for model predictive control
- **3D Simulation**: Godot for game environments, AirSim for drones (planned)

### Resource Management & Quotas
- **GPU Allocation**: `GPUAllocator` coordinates slot reservations across runs
- **Queue Limits**: Configurable telemetry buffers prevent runaway event rates
- **Credit-Based Backpressure**: Automatic flow control for telemetry streams
- **Health Monitoring**: Heartbeat-based worker lifecycle management

### Per-Agent Policy Binding
Route each agent to different workers via `PolicyMappingService`:
```python
mapping.bind_agent_policy("player_0", "human_keyboard")
mapping.bind_agent_policy("player_1", "cleanrl_ppo", worker_id="cleanrl")
mapping.bind_agent_policy("coach", "llm_planner", worker_id="llm")
```

### Worker Lifecycle Orchestration
`TrainerDispatcher` manages subprocess lifecycle:
- Async worker spawning and termination
- Heartbeat monitoring with configurable intervals
- Graceful shutdown with reason tracking
- GPU quota enforcement per run

### Telemetry & Logging
- TensorBoard integration
- Weights & Biases support
- HDF5 replay storage
- Real-time metrics dashboard

## Documentation

Full documentation is available at the [MOSAIC Documentation](docs/source/index.rst).

### Key Documentation

- [Installation Guide](docs/source/documents/tutorials/installation.rst)
- [Quick Start](docs/source/documents/tutorials/quickstart.rst)
- [Architecture Overview](docs/source/documents/architecture/overview.rst)
- [Paradigm Guide](docs/source/documents/architecture/paradigms.rst)
- [Policy Mapping](docs/source/documents/architecture/policy_mapping.rst)
- [API Reference](docs/source/documents/api/core.rst)

## Who Is MOSAIC For?

- **Researchers** exploring multi-agent RL with heterogeneous agents
- **Developers** building RL applications with visual configuration
- **Students** learning about different RL paradigms and agent architectures
- **AI practitioners** interested in combining symbolic AI (BDI) with neural methods (RL)
- **Game developers** training AI agents in custom 3D environments

## Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/source/documents/contributing.rst) for details.

## Citation

If you use MOSAIC in your research, please cite:

```bibtex
@software{mosaic2025,
  title = {MOSAIC: Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads},
  author = {Mousa, Abdulhamid},
  year = {2025},
  url = {https://github.com/Abdulhamid97Mousa/MOSAIC}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) - Standard RL API
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent environments
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean RL implementations
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [ViZDoom](https://vizdoom.cs.put.edu.pl/) - Visual RL platform
