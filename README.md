<div align="center">
    <a href="https://github.com/Abdulhamid97Mousa/mosaic"><img width="1000px" height="auto" src="docs/source/_static/figures/A_Full_Architecture.png"></a>
</div>

---

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Abdulhamid97Mousa/mosaic/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://mosaic-platform.readthedocs.io/en/latest/index.html)

# MOSAIC

**A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers**

MOSAIC is a visual-first platform that enables researchers to configure, run, and compare experiments across RL, LLM, VLM, and human decision-makers in the same multi-agent environment. Different paradigms like tiles in a mosaic come together to form a complete picture of agent performance.

| **Documentation**: [mosaic-platform.readthedocs.io](https://mosaic-platform.readthedocs.io/en/latest/index.html) | **GitHub**: [github.com/Abdulhamid97Mousa/mosaic](https://github.com/Abdulhamid97Mousa/mosaic) |
|---|---|

## Two Evaluation Modes

MOSAIC provides two evaluation modes designed for reproducibility:

- **Manual Mode:** side-by-side comparison where multiple operators step through the same environment with shared seeds, letting researchers visually inspect decision-making differences between paradigms in real time.

- **Script Mode:** automated, long-running evaluation driven by Python scripts that define operator configurations, worker assignments, seed sequences, and episode counts. Scripts execute deterministically with no manual intervention, producing reproducible telemetry logs (JSONL) for every step and episode.

All evaluation runs share **identical conditions**: same environment seeds, same observations, and unified telemetry. Script Mode additionally supports **procedural seeds** (different seed per episode to test generalization) and **fixed seeds** (same seed every episode to isolate agent behaviour), with configurable step pacing for visual inspection or headless batch execution.

## Why MOSAIC?

Today's AI landscape offers powerful but **fragmented** tools: RL frameworks ([CleanRL](https://github.com/vwxyzjn/cleanrl), [RLlib](https://docs.ray.io/en/latest/rllib/index.html), [XuanCe](https://github.com/agi-brain/xuance)), language models (GPT, Claude), and robotics simulators (MuJoCo). Each excels in isolation, but **no platform bridges them together** under a unified, visual-first interface.

**MOSAIC provides:**

- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface, **almost no code required**.
- **Heterogeneous Agent Mixing**: Deploy Human, RL, and LLM agents in the same environment.
- **Resource Management & Quotas**: GPU allocation, queue limits, credit-based backpressure, health monitoring.
- **Per-Agent Policy Binding**: Route each agent to different workers via `PolicyMappingService`.
- **Worker Lifecycle Orchestration**: Subprocess management with heartbeat monitoring and graceful termination.

## Supported Environment Families (26)

MOSAIC supports **26 environment families** spanning single-agent, multi-agent, and cooperative/competitive paradigms. See the full [Environment Families](https://mosaic-platform.readthedocs.io/en/latest/documents/environments/index.html) reference for installation instructions, environment lists, and academic citations.

| Family | Description | Example |
|--------|-------------|---------|
| **Gymnasium** | Standard single-agent RL (Toy Text, Classic Control, Box2D, MuJoCo) | <img src="docs/source/images/envs/gymnasium/cartpole.gif" width="200"> |
| **Atari / ALE** | 128 classic Atari 2600 games | <img src="docs/source/images/envs/atari/atari.gif" width="200"> |
| **MiniGrid** | Procedural grid-world navigation | <img src="docs/source/images/envs/minigrid/minigrid.gif" width="200"> |
| **BabyAI** | Language-grounded instruction following | <img src="docs/source/images/envs/babyai/GoTo.gif" width="200"> |
| **ViZDoom** | Doom-based first-person visual RL | <img src="docs/source/images/envs/vizdoom/vizdoom.gif" width="200"> |
| **MiniHack / NetHack** | Roguelike dungeon crawling (NLE) | <img src="docs/source/images/envs/minihack/minihack.gif" width="200"> |
| **Crafter** | Open-world survival benchmark | <img src="docs/source/images/envs/crafter/crafter.gif" width="200"> |
| **Procgen** | 16 procedurally generated environments | <img src="docs/source/images/envs/procgen/coinrun.gif" width="200"> |
| **BabaIsAI** | Rule-manipulation puzzles | <img src="docs/source/images/envs/babaisai/babaisai.png" width="200"> |
| **Jumanji** | JAX-accelerated logic/routing/packing (25 envs) | <img src="docs/source/images/envs/jumanji/jumanji.gif" width="200"> |
| **PyBullet Drones** | Quadcopter physics simulation | <img src="docs/source/images/envs/pybullet_drones/pybullet_drones.gif" width="200"> |
| **PettingZoo Classic** | Turn-based board games (AEC) | <img src="docs/source/images/envs/pettingzoo/pettingzoo.gif" width="200"> |
| **MOSAIC MultiGrid** | Competitive team sports (view_size=3) | <img src="docs/source/images/envs/mosaic_multigrid/mosaic_multigrid.gif" width="200"> |
| **INI MultiGrid** | Cooperative exploration (view_size=7) | <img src="docs/source/images/envs/multigrid_ini/multigrid_ini.gif" width="200"> |
| **Melting Pot** | Social multi-agent scenarios (up to 16 agents) | <img src="docs/source/images/envs/meltingpot/meltingpot.gif" width="200"> |
| **Overcooked** | Cooperative cooking (2 agents) | <img src="docs/source/images/envs/overcooked/overcooked_layouts.gif" width="200"> |
| **SMAC** | StarCraft Multi-Agent Challenge (hand-designed maps) | <img src="docs/source/images/envs/smac/smac.gif" width="200"> |
| **SMACv2** | StarCraft Multi-Agent Challenge v2 (procedural units) | <img src="docs/source/images/envs/smacv2/smacv2.png" width="200"> |
| **RWARE** | Cooperative warehouse delivery | <img src="docs/source/images/envs/rware/rware.gif" width="200"> |
| **MuJoCo** | Continuous-control robotics tasks | <img src="docs/source/images/envs/mujoco/ant.gif" width="200"> |

## Supported Workers (8)

| Worker | Description |
|--------|-------------|
| **[CleanRL](https://github.com/vwxyzjn/cleanrl)** | Single-file RL implementations (PPO, DQN, SAC, TD3, DDPG, C51) |
| **[XuanCe](https://github.com/agi-brain/xuance)** | Modular RL framework with flexible algorithm composition. Multi-agent algorithms (MAPPO, QMIX, MADDPG, VDN, COMA) |
| **[Ray RLlib](https://docs.ray.io/en/latest/rllib/)** | RL with distributed training and large-batch optimization (PPO, IMPALA, APPO) |
| **[BALROG](https://github.com/balrog-ai/BALROG)** | LLM/VLM agentic evaluation (GPT-4o, Claude 3, Gemini; NetHack, BabyAI, Crafter) |
| **[MOSAIC LLM](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_LLM_Worker)** | Multi-agent LLM with coordination strategies and Theory of Mind (MultiGrid, BabyAI, MeltingPot, PettingZoo) |
| **[Chess LLM](docs/source/documents/architecture/workers/integrated_workers/Chess_LLM_Worker)** | LLM chess play with multi-turn dialog (PettingZoo Chess) |
| **[MOSAIC Human Worker](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_Human_Worker)** | Human-in-the-loop play via keyboard for any Gymnasium-compatible environment (MiniGrid, Crafter, Chess, NetHack) |
| **[MOSAIC Random Worker](docs/source/documents/architecture/workers/integrated_workers/MOSAIC_Random_Worker)** | Baseline agents with random, no-op, and cycling action behaviours across all 26 environment families |

## Installation

```bash
# Clone the repository
git clone https://github.com/Abdulhamid97Mousa/mosaic.git
cd mosaic

# Create virtual environment (Python 3.10-3.12)
python3.11 -m venv .venv
source .venv/bin/activate

# Install core GUI
pip install -e .
```

Install only what you need. **Workers** and **environment families** are independent extras:

```bash
# Worker (CleanRL) + environment family (MiniGrid)
pip install -e ".[cleanrl,minigrid]"

# Multi-agent worker (XuanCe) + competitive environments
pip install -e ".[xuance,mosaic_multigrid]"

# Everything
pip install -e ".[full]"
```

## Quick Start

```bash
# Launch with trainer daemon (recommended)
./run.sh

# Or launch GUI only
python -m gym_gui
```

## Citing MOSAIC

If you use MOSAIC in your research, please cite:

```bibtex
@article{mousa2026mosaic,
  title   = {{MOSAIC}: A Unified Platform for Cross-Paradigm Comparison
             and Evaluation of Homogeneous and Heterogeneous Multi-Agent
             {RL}, {LLM}, {VLM}, and Human Decision-Makers},
  author  = {Mousa, Abdulhamid M. and Daoui, Zahra and Khajiev, Rakhmonberdi
             and Azzabi, Jalaledin M. and Mousa, Abdulkarim M. and Liu, Ming},
  year    = {2026},
  url     = {https://github.com/Abdulhamid97Mousa/mosaic},
  note    = {Available at \url{https://github.com/Abdulhamid97Mousa/mosaic}}
}
```

## License

MOSAIC is released under the [MIT License](https://github.com/Abdulhamid97Mousa/mosaic/blob/main/LICENSE).

## Acknowledgments

**RL & MARL Frameworks**
- [Gymnasium](https://gymnasium.farama.org/): Standard RL API
- [PettingZoo](https://pettingzoo.farama.org/): Multi-agent environment API
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Single-file RL implementations
- [XuanCe](https://github.com/agi-brain/xuance): Multi-agent RL algorithms
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/): Distributed RL training
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): RL algorithms library (PPO, A2C, DQN)
- [MARLlib](https://github.com/Farama-Foundation/Marllib): Multi-agent RL library
- [SuperSuit](https://github.com/Farama-Foundation/SuperSuit): PettingZoo wrapper utilities
- [Shimmy](https://shimmy.farama.org/): Environment compatibility wrappers

**Environment Families**
- [Atari / ALE](https://github.com/mgbellemare/Arcade-Learning-Environment): Classic Atari 2600 games
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid): Procedural grid-world navigation
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom): Doom-based first-person visual RL
- [NLE](https://github.com/facebookresearch/nle) / [MiniHack](https://github.com/facebookresearch/minihack): Roguelike dungeon crawling
- [Crafter](https://github.com/danijar/crafter): Open-world survival benchmark
- [Procgen](https://github.com/openai/procgen): Procedurally generated environments
- [Baba Is AI](https://github.com/nacloos/baba-is-ai): Rule-manipulation puzzles
- [Jumanji](https://github.com/google-deepmind/jumanji): JAX-accelerated logic/routing/packing environments
- [PyBullet Drones](https://github.com/utiasDSL/gym-pybullet-drones): Quadcopter physics simulation
- [INI MultiGrid](https://github.com/ini/multigrid): Cooperative multi-agent exploration
- [SMAC](https://github.com/oxwhirl/smac) / [SMACv2](https://github.com/oxwhirl/smacv2): StarCraft Multi-Agent Challenge
- [RWARE](https://github.com/uoe-agents/robotic-warehouse): Cooperative warehouse delivery
- [Melting Pot](https://github.com/google-deepmind/meltingpot): Multi-agent social scenarios
- [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai): Cooperative cooking benchmark
- [OpenSpiel](https://github.com/google-deepmind/open_spiel): Board games collection
- [MuJoCo](https://mujoco.org/): Continuous-control physics engine

**LLM / VLM Agents**
- [BALROG](https://github.com/balrog-ai/BALROG): LLM/VLM agentic evaluation benchmark
- [llm_chess](https://github.com/maxim-saplin/llm_chess): LLM chess evaluation
- [python-chess](https://github.com/niklasf/python-chess): Chess move generation and validation
- [Stockfish](https://stockfishchess.org/): Chess engine for evaluation
- [vLLM](https://github.com/vllm-project/vllm): Local GPU LLM inference

**JAX Ecosystem**
- [JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs
- [Flax](https://github.com/google/flax): Neural network library for JAX
- [Optax](https://github.com/google-deepmind/optax): Gradient transformation library
- [MCTX](https://github.com/google-deepmind/mctx): GPU-accelerated Monte-Carlo Tree Search
- [Pgx](https://github.com/sotetsuk/pgx): JAX-based board game environments

**GUI, Visualization & Core**
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/): GUI framework
- [PyTorch](https://pytorch.org/): Deep learning framework
- [gRPC](https://grpc.io/) / [Protobuf](https://developers.google.com/protocol-buffers): Inter-process communication
- [TensorBoard](https://www.tensorflow.org/tensorboard) / [Weights & Biases](https://wandb.ai/): Experiment tracking
