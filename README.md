<div align="center">
    <a href="https://github.com/Abdulhamid97Mousa/MOSAIC"><img width="1000px" height="auto" src="docs/source/_static/figures/A_Full_Architecture.png"></a>
</div>

---

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![GitHub license](https://img.shields.io/github/license/Abdulhamid97Mousa/MOSAIC)](https://github.com/Abdulhamid97Mousa/MOSAIC/blob/main/LICENSE)

# MOSAIC

**A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers**

MOSAIC is a visual-first platform that enables researchers to configure, run, and compare experiments across RL, LLM, VLM, and human decision-makers in the same multi-agent environment. Different paradigms like tiles in a mosaic come together to form a complete picture of agent performance.

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

MOSAIC supports **26 environment families** spanning single-agent, multi-agent, and cooperative/competitive paradigms.

| Family | Description |
|--------|-------------|
| **Gymnasium** | Standard single-agent RL (Toy Text, Classic Control, Box2D, MuJoCo) |
| **Atari / ALE** | 128 classic Atari 2600 games |
| **MiniGrid** | Procedural grid-world navigation |
| **BabyAI** | Language-grounded instruction following |
| **ViZDoom** | Doom-based first-person visual RL |
| **MiniHack / NetHack** | Roguelike dungeon crawling (NLE) |
| **Crafter** | Open-world survival benchmark |
| **Procgen** | 16 procedurally generated environments |
| **BabaIsAI** | Rule-manipulation puzzles |
| **Jumanji** | JAX-accelerated logic/routing/packing (25 envs) |
| **PyBullet Drones** | Quadcopter physics simulation |
| **PettingZoo Classic** | Turn-based board games (AEC) |
| **MOSAIC MultiGrid** | Competitive team sports (view_size=3) |
| **INI MultiGrid** | Cooperative exploration (view_size=7) |
| **Melting Pot** | Social multi-agent scenarios (up to 16 agents) |
| **Overcooked** | Cooperative cooking (2 agents) |
| **SMAC** | StarCraft Multi-Agent Challenge (hand-designed maps) |
| **SMACv2** | StarCraft Multi-Agent Challenge v2 (procedural units) |
| **RWARE** | Cooperative warehouse delivery |
| **MuJoCo** | Continuous-control robotics tasks |

## Supported Workers (8)

| Worker | Description |
|--------|-------------|
| **[CleanRL](https://github.com/vwxyzjn/cleanrl)** | Single-file RL implementations (PPO, DQN, SAC, TD3, DDPG, C51) |
| **[XuanCe](https://github.com/agi-brain/xuance)** | Modular RL framework with flexible algorithm composition. Multi-agent algorithms (MAPPO, QMIX, MADDPG, VDN, COMA) |
| **[Ray RLlib](https://docs.ray.io/en/latest/rllib/)** | RL with distributed training and large-batch optimization (PPO, IMPALA, APPO) |
| **[BALROG](https://github.com/balrog-ai/BALROG)** | LLM/VLM agentic evaluation (GPT-4o, Claude 3, Gemini; NetHack, BabyAI, Crafter) |
| **MOSAIC LLM** | Multi-agent LLM with coordination strategies and Theory of Mind (MultiGrid, BabyAI, MeltingPot, PettingZoo) |
| **Chess LLM** | LLM chess play with multi-turn dialog (PettingZoo Chess) |
| **MOSAIC Human Worker** | Human-in-the-loop play via keyboard for any Gymnasium-compatible environment (MiniGrid, Crafter, Chess, NetHack) |
| **MOSAIC Random Worker** | Baseline agents with random, no-op, and cycling action behaviours across all 26 environment families |

## Installation

```bash
# Clone the repository
git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
cd MOSAIC

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
  url     = {https://github.com/Abdulhamid97Mousa/MOSAIC},
  note    = {Available at \url{https://github.com/Abdulhamid97Mousa/MOSAIC}}
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
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — GUI framework
