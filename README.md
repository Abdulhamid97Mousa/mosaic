<div align="center">
    <a href="https://mosaic-multi-agent-orchestration-system.readthedocs.io/"><img width="1000px" height="auto" src="docs/source/images/Platform_Main_View.png"></a>
</div>

---

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D1.1.0-blue)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![Read the Docs](https://img.shields.io/badge/docs-ReadTheDocs-blue)](https://mosaic-multi-agent-orchestration-system.readthedocs.io/)

[![GitHub stars](https://img.shields.io/github/stars/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/network)
[![GitHub issues](https://img.shields.io/github/issues/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/pulls)
[![Contributors](https://img.shields.io/github/contributors/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/blob/main/LICENSE)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Abdulhamid97Mousa/MOSAIC)

## Introduction to MOSAIC

[Documentation](https://mosaic-multi-agent-orchestration-system.readthedocs.io/) | [Tutorials](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/tutorials/quickstart.html) | [Installation](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/tutorials/installation.html) | [Architecture](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/architecture/overview.html) | [Roadmap](docs/source/documents/roadmap.rst)

**MOSAIC** (**M**ulti-Agent **O**rchestration **S**ystem with **A**daptive **I**ntelligent **C**ontrol) is an open-source platform for **cross-paradigm comparison** of RL, LLM, VLM, and Human agents.

Gymnasium standardized the **environment** interface (`reset`/`step`), enabling interoperability across simulators. However, no equivalent standardization exists for the **agent** interface. MOSAIC addresses this gap through three contributions:

1. **Operator Abstraction** — a minimal interface unifying heterogeneous decision-makers (RL policies, LLM/VLM agents, human operators) under a common `select_action(obs)` protocol
2. **Process-Isolated Worker Protocol** — a versioned IPC protocol (JSONL over stdout + gRPC) that wraps diverse RL libraries and LLM benchmarks as isolated sub-processes with unified telemetry
3. **Deterministic Cross-Paradigm Evaluation** — shared seed schedules enabling multiple operators to execute on identical environment instances for reproducible head-to-head comparison

<details open>
<summary><b>Supported Agent Paradigms</b> (Click to Collapse)</summary>

| Paradigm | Description | Example |
|----------|-------------|---------|
| **RL Operator** | Trained neural policy $\pi_\theta(o)$ | CleanRL DQN, XuanCe MAPPO, RLlib PPO |
| **LLM Operator** | Zero-shot prompting $g(f(o))$ | GPT-4 via BALROG, Claude, LLM Chess |
| **VLM Operator** | Vision-language model | GPT-4V, Claude Vision |
| **Human Operator** | Keyboard/mouse input $h(o)$ | Interactive GUI control |


</details>

## Outline

- [Introduction to MOSAIC](#introduction-to-mosaic)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Workers](#supported-workers)
- [Environment Versatility](#environment-versatility)
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Documentation](#documentation)
- [Who Is MOSAIC For?](#who-is-mosaic-for)
- [Feedback and Contribution](#feedback-and-contribution)
- [Supporters](#supporters)
  - [&#8627; Stargazers](#-stargazers)
  - [&#8627; Forkers](#-forkers)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

```bash
# Clone the repository
git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
cd MOSAIC

# Create virtual environment (Python 3.10-3.12)
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install core GUI
pip install -e .
```

Install only what you need. **Workers** (training backends) and **environment families** are independent extras:

```bash
# Worker (CleanRL) + environment family (MiniGrid)
pip install -e ".[cleanrl,minigrid]"

# Multi-agent worker (XuanCe) + competitive environments
pip install -e ".[xuance,mosaic_multigrid]"

# Everything
pip install -e ".[full]"
```

<details>
<summary><b>All Available Extras</b> (Click to Expand)</summary>

| Extra | Type | Key Dependencies |
|-------|------|------------------|
| `cleanrl` | Worker | `torch`, `stable-baselines3`, `tensorboard` |
| `xuance` | Worker | `torch`, `mpi4py`, `xuance` |
| `ray-rllib` | Worker | `ray[rllib]`, `torch` |
| `balrog` | Worker | `omegaconf`, `openai`, `anthropic` |
| `chat` | Worker | `vllm`, `huggingface_hub`, `requests` |
| `mctx` | Worker | `jax`, `pgx`, `mctx` |
| `minigrid` | Environment | `minigrid` |
| `pettingzoo` | Environment | `pettingzoo`, `supersuit` |
| `atari` | Environment | `gymnasium[atari]`, `autorom` |
| `box2d` | Environment | `gymnasium[box2d]` |
| `mujoco` | Environment | `gymnasium[mujoco]` |
| `vizdoom` | Environment | `vizdoom` |
| `nethack` | Environment | `nle`, `minihack` |
| `crafter` | Environment | `crafter` |
| `smac` | Environment | `smac`, `pygame` |
| `rware` | Environment | `gymnasium`, `pyglet` |
| `mosaic_multigrid` | Environment | `mosaic-multigrid` |
| `overcooked` | Environment | `dill`, `gymnasium` |

</details>

For detailed installation instructions, see the [Installation Guide](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/tutorials/installation.html).

## Quick Start

```bash
# Launch with trainer daemon (recommended)
./run.sh

# Or launch GUI only
python -m gym_gui
```

<details open>
<summary><b>Programmatic Usage</b> (Click to Collapse)</summary>

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

```python
# Cross-paradigm comparison with shared seeds
mapping.bind_agent_policy("player_0", "human_keyboard")
mapping.bind_agent_policy("player_1", "cleanrl_ppo", worker_id="cleanrl")
mapping.bind_agent_policy("coach", "llm_planner", worker_id="llm")
```

</details>

## Supported Workers

MOSAIC integrates multiple training backends through a process-isolated worker architecture:

| Worker | Type | Capabilities |
|--------|------|--------------|
| **[CleanRL](https://github.com/vwxyzjn/cleanrl)** | Neural RL | Single-file implementations: PPO, DQN, SAC, TD3 |
| **[XuanCe](https://github.com/agi-brain/xuance)** | Multi-agent RL | MAPPO, QMIX, MADDPG algorithms |
| **[Ray RLlib](https://docs.ray.io/en/latest/rllib/)** | Distributed RL | Scalable training with Ray |
| **[BALROG](https://github.com/balrog-ai/BALROG)** | LLM/VLM Eval | LLM agents on BabyAI, MiniHack, Crafter |
| **[DI-engine](https://github.com/opendilab/DI-engine)** | Decision AI | DQN, Rainbow, PPO, SAC, R2D2, IMPALA, and 50+ algorithms |

| **MuJoCo MPC** | Control | Model Predictive Control worker |
| **Godot** | 3D Engine | Game AI training integration |
| **LLM Chat** | Language Models | GPT/Claude agents via OpenRouter or local vLLM |

## Environment Versatility

<details open>
<summary>(Click to Collapse)</summary>

| Family | Install | Environments |
|--------|---------|-------------|
| **Gymnasium Core** | `pip install -e ".[gymnasium]"` | FrozenLake, Taxi, CartPole, Pendulum |
| **Box2D** | `pip install -e ".[box2d]"` | LunarLander, BipedalWalker, CarRacing |
| **MuJoCo** | `pip install -e ".[mujoco]"` | Ant, HalfCheetah, Humanoid, Walker2d, Hopper |
| **Atari / ALE** | `pip install -e ".[atari]"` | Breakout, Pong, SpaceInvaders (128 games) |
| **MiniGrid** | `pip install -e ".[minigrid]"` | Empty, DoorKey, MultiRoom, RedBlueDoors |
| **BabyAI** | `pip install -e ".[minigrid]"` | GoTo, Open, Pickup, BossLevel (language-grounded) |
| **ViZDoom** | `pip install -e ".[vizdoom]"` | Basic, DeadlyCorridor, DefendTheCenter |
| **NetHack** | `pip install -e ".[nethack]"` | Room, MazeWalk, NetHackChallenge |
| **Crafter** | `pip install -e ".[crafter]"` | CrafterReward, CrafterNoReward (open-world survival) |
| **Procgen** | `pip install -e ".[procgen]"` | CoinRun, StarPilot, Maze, Heist (16 envs) |
| **PettingZoo** | `pip install -e ".[pettingzoo]"` | Chess, Go, Connect Four, TicTacToe, MPE |
| **SMAC** | `pip install -e ".[smac]"` | 3m, 8m, 2s3z, MMM2 (StarCraft cooperative) |
| **MOSAIC MultiGrid** | `pip install -e ".[mosaic_multigrid]"` | Soccer 2v2, Basketball 3v3 (competitive) |
| **Overcooked** | `pip install -e ".[overcooked]"` | CrampedRoom, CoordinationRing (cooperative cooking) |
| **RWARE** | `pip install -e ".[rware]"` | Warehouse delivery (cooperative logistics) |

</details>

## Architecture

MOSAIC follows a three-tier architecture separating **orchestration**, **communication**, and **execution**:

```
┌─────────────────────────────────────────────────────────┐
│  Qt6 Main Process (GUI)                                 │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Control  │ │ Render   │ │ Telemetry │ │ TensorBoard│  │
│  │ Panel    │ │ Tabs     │ │ Dashboard │ │ Viewer    │  │
│  └─────────┘ └──────────┘ └───────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────┤
│  Daemon Process (gRPC Server)                           │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │ Run      │ │ Telemetry│ │ GPU       │ │ Operator  │  │
│  │ Registry │ │ Proxy    │ │ Allocator │ │ Service   │  │
│  └──────────┘ └──────────┘ └───────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────┤
│  Worker Sub-Processes (isolated)                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌───────┐  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ ┌────────┐  │
│  │CleanRL │ │ XuanCe │ │ BALROG │ │ Human │ │DI-eng. │  │
│  │(PPO)   │ │(MAPPO) │ │(GPT-4) │ │(GUI)  │ │(50+)   │  │
│  └────────┘ └────────┘ └────────┘ └───────┘ └────────┘  │
└─────────────────────────────────────────────────────────┘
```

| Layer | Components | Purpose |
|-------|------------|---------|
| **Visual** | MainWindow, ControlPanel, RenderTabs | PyQt6 interface and real-time visualization |
| **Service** | PolicyMappingService, OperatorService, TelemetryService | Business logic and cross-paradigm orchestration |
| **Adapter** | ParadigmAdapter, PettingZooAdapter | Environment normalization (Gymnasium, AEC, Parallel) |
| **Workers** | CleanRL, XuanCe, RLlib, BALROG, DI-engine, LLM | Training backends communicating via gRPC/JSONL |

## Core Features

<details open>
<summary><b>Key Capabilities</b> (Click to Collapse)</summary>

### Visual-First Configuration
Configure complex multi-agent experiments through an intuitive PyQt6 interface. No code required for environment setup, agent configuration, or training launch.

### Cross-Paradigm Comparison (RL vs LLM vs Human)
The Operator Abstraction enables direct comparison: given identical initial conditions and random seeds, how does a zero-shot GPT-4 compare to a trained DQN policy or a human demonstrator?

### Framework Bridge (RL + LLM + Robotics + 3D)
- **RL**: CleanRL, RLlib, XuanCe, DI-engine for neural policy training
- **LLM/VLM**: GPT, Claude, local models via vLLM for language-based reasoning
- **Robotics**: MuJoCo MPC for model predictive control
- **3D Simulation**: Godot for game environments

### Process-Isolated Workers
Each agent runs as an isolated subprocess. Workers emit JSONL telemetry to stdout; a telemetry proxy sidecar translates events to gRPC Protocol Buffers. No source code modifications to algorithms required.

### Deterministic Evaluation
Shared seed schedules ensure all operators face identical scenarios. Trajectory logging enables step-by-step comparison: where does GPT-4 diverge from a trained DQN?

### Resource Management
- **GPU Allocation**: `GPUAllocator` coordinates slot reservations across runs
- **Queue Limits**: Configurable telemetry buffers prevent runaway event rates
- **Health Monitoring**: Heartbeat-based worker lifecycle management

### Telemetry & Logging
- TensorBoard integration (embedded viewer in GUI)
- Weights & Biases support
- SQLite-backed telemetry persistence
- Real-time metrics dashboard

</details>

## Documentation

Full documentation: **[mosaic-multi-agent-orchestration-system.readthedocs.io](https://mosaic-multi-agent-orchestration-system.readthedocs.io/)**

| Topic | Link |
|-------|------|
| Installation Guide | [tutorials/installation](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/tutorials/installation.html) |
| Quick Start | [tutorials/quickstart](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/tutorials/quickstart.html) |
| Architecture Overview | [architecture/overview](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/architecture/overview.html) |
| Paradigm Guide | [architecture/paradigms](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/architecture/paradigms.html) |
| Worker Development | [workers/guide](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/workers/guide.html) |
| API Reference | [api/core](https://mosaic-multi-agent-orchestration-system.readthedocs.io/en/latest/documents/api/core.html) |

## Who Is MOSAIC For?

- **Researchers** answering "given this task, which paradigm should I deploy?" across RL, LLM, and human agents
- **Developers** building RL applications with visual configuration and live telemetry
- **Students** learning about different RL paradigms, agent architectures, and cross-paradigm evaluation
- **AI practitioners** combining neural methods (RL) with foundation models (LLM/VLM)
- **Game developers** training AI agents in custom 3D environments

## Feedback and Contribution

- [File an issue](https://github.com/Abdulhamid97Mousa/MOSAIC/issues) on GitHub
- [Open or participate in discussions](https://github.com/Abdulhamid97Mousa/MOSAIC/discussions)
- Contact us by email: [mousa.abdulhamid@bit.edu.cn](mailto:mousa.abdulhamid@bit.edu.cn)
- Read our [Contributing Guide](docs/source/documents/contributing.rst) for contribution guidelines
- Check our [Roadmap](docs/source/documents/roadmap.rst) and contribute to future plans

We appreciate all feedback and contributions to improve MOSAIC, both algorithms and system designs. [CONTRIBUTING.md](CONTRIBUTING.md) offers the necessary information for getting started.

## Supporters

### &#8627; Stargazers

[![Stargazers repo roster for @Abdulhamid97Mousa/MOSAIC](https://reporoster.com/stars/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/stargazers)

### &#8627; Forkers

[![Forkers repo roster for @Abdulhamid97Mousa/MOSAIC](https://reporoster.com/forks/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/network/members)

## Citation

If you use MOSAIC in your research, please cite our paper:

```bibtex
@misc{mousa2025mosaic,
    title={MOSAIC: A Unified Platform for Cross-Paradigm Comparison of RL, LLM, VLM, and Human Agents},
    author={Mousa, Abdulhamid M. and Daoui, Zahra and Liu, Ming},
    publisher={GitHub},
    howpublished={\url{https://github.com/Abdulhamid97Mousa/MOSAIC}},
    year={2025},
}
```

## License

MOSAIC is released under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) — Standard RL API
- [PettingZoo](https://pettingzoo.farama.org/) — Multi-agent environments
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — Clean RL implementations
- [XuanCe](https://github.com/agi-brain/xuance) — Multi-agent RL algorithms
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) — Distributed RL training
- [BALROG](https://github.com/balrog-ai/BALROG) — LLM/VLM agent benchmark
- [DI-engine](https://github.com/opendilab/DI-engine) — Decision intelligence engine
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — GUI framework
- [MuJoCo](https://mujoco.org/) — Physics simulation
- [ViZDoom](https://vizdoom.cs.put.edu.pl/) — Visual RL platform
