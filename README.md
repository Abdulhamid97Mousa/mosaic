<div align="center">
    <a href="https://github.com/Abdulhamid97Mousa/mosaic"><img width="1000px" height="auto" src="docs/source/_static/figures/A_Full_Architecture.png"></a>
</div>

---

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red)](https://pytorch.org/get-started/locally/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue)](https://pettingzoo.farama.org/)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Abdulhamid97Mousa/mosaic/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://mosaic-agent-level-interface.readthedocs.io/en/latest/)

# MOSAIC

**A Unified Platform for Cross-Paradigm Comparison and Evaluation of Homogeneous and Heterogeneous Multi-Agent RL, LLM, VLM, and Human Decision-Makers**

MOSAIC is a visual-first platform that enables researchers to configure, run, and compare experiments across RL, LLM, VLM, and human decision-makers in the same multi-agent environment. Different paradigms like tiles in a mosaic come together to form a complete picture of agent performance.

| **Documentation**: [mosaic-platform.readthedocs.io](https://mosaic-platform.readthedocs.io/en/latest/) | **GitHub**: [github.com/Abdulhamid97Mousa/mosaic](https://github.com/Abdulhamid97Mousa/mosaic) |
|---|---|

## Two Evaluation Modes

MOSAIC provides two evaluation modes designed for reproducibility:

<video src="https://private-user-images.githubusercontent.com/80536675/553915409-ea9ebc18-2216-4fb2-913c-5d354ebea56e.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU0MDktZWE5ZWJjMTgtMjIxNi00ZmIyLTkxM2MtNWQzNTRlYmVhNTZlLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyZDkzMWQwZjczOTI1NzFiNjk2MGU4N2I2ZTAxNDE0M2Y2YmQxNjM5ZTAxMTkxYzA5NGU4ZGE3YzZkZmJkZWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.9BimKYjxGrdN_g1eDbXC69weDC7-My85Gl-ou2wNzxQ" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Manual Mode:</b> Side-by-side lock-step evaluation with shared seeds.</p>

- **Manual Mode:** side-by-side comparison where multiple operators step through the same environment with shared seeds, letting researchers visually inspect decision-making differences between paradigms in real time.

<video src="https://private-user-images.githubusercontent.com/80536675/553915854-a9b3f6f4-661c-492f-b43f-34d7125a6d2e.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU4NTQtYTliM2Y2ZjQtNjYxYy00OTJmLWI0M2YtMzRkNzEyNWE2ZDJlLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNhMjM0ZjBjYjU1NWFlNmYxOGU2Yzc2N2U0ODE4OTYzZGVkYTc5YTIyMjM5YzRjODU0MTRhODFhOWI4ZDU3NmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.tk8Ezu0ivgFlp-xm6YEkIsWlbFPcpOgSq30Hq_JEJvs" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Script Mode:</b> Automated batch evaluation with deterministic seed sequences.</p>

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

<video src="https://private-user-images.githubusercontent.com/80536675/553915983-ded17cdc-f23c-404f-a9f6-074fbe74816c.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTU5ODMtZGVkMTdjZGMtZjIzYy00MDRmLWE5ZjYtMDc0ZmJlNzQ4MTZjLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWUwZmZiZTlmZjQ3OGQ4NTk1ZWQzODEzMGVlYTMyY2E0Y2E1ZjJjZTU0NTg0MDMxYmU0OThlMGVkNjc1ZGVjNDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.1DWDiXB20L5dr8Hx60M70zd0pL1JKkOO5roOjr2kkaQ" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Human vs Human:</b> Two human players competing via dedicated USB keyboards.</p>

<video src="https://private-user-images.githubusercontent.com/80536675/553916105-2625a8f8-476c-4171-86cc-a9970cbf1665.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTYxMDUtMjYyNWE4ZjgtNDc2Yy00MTcxLTg2Y2MtYTk5NzBjYmYxNjY1Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQwMzljOWNhYWQwNjZjNzRlZDU4ZmM3M2M0YWJlNjYwNWM0NzM2YjkxNmQ4YTQxYmEzYTNmNzJkYWQwZGI3MWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Lvwy-I3p1fNP-kmUhHtUXO7FtJ9Q_K_NCVuGbxZ6VbY" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Random Agents:</b> Baseline agents across 26 environment families.</p>

<video src="https://private-user-images.githubusercontent.com/80536675/553916227-f2d79901-a93d-465b-9058-1b9cdabf311a.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTYyMjctZjJkNzk5MDEtYTkzZC00NjViLTkwNTgtMWI5Y2RhYmYzMTFhLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTU2MzNiMDJjM2ZjOWQwNjA2Y2Q1NWI0MzljMzhmNTNlOTlmMzU4OWNjOGQ1Y2NmMDdlODVkZDBjZTRhOWI4ODgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.NQwmmI3Yyu_ePoPhLvU8ZAvp-Jn-q3q90j48n7-TEFk" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Heterogeneous Multi-Agent Ad-Hoc Teamwork in Adversarial Settings:</b> Different decision-making paradigms (RL, LLM, Random) competing head-to-head in the same multi-agent environment.</p>

<video src="https://private-user-images.githubusercontent.com/80536675/553916417-2ae1665b-3a57-44be-98a3-4e7223b37628.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzIxNjgzMzQsIm5iZiI6MTc3MjE2ODAzNCwicGF0aCI6Ii84MDUzNjY3NS81NTM5MTY0MTctMmFlMTY2NWItM2E1Ny00NGJlLTk4YTMtNGU3MjIzYjM3NjI4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjI3VDA0NTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyYzllNjc4OTYwNzg5MGFkZDA5N2Q4ZTZmMzk4Yzg0ZTYzOGNmYTgxNGZkYTU2OWYzMGZiNDk2NmJjY2FiYjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.FK9hfbmCA1GOl80CJkenINfr5iD5CckeipYVYfFy3_Y" controls autoplay muted loop style="width:100%; max-width:100%; height:auto; border-radius:8px;"></video>
<p align="center"><b>Homogeneous Teams: Random vs LLM:</b> Two homogeneous teams (all-Random vs all-LLM) competing in the same multi-agent environment.</p>

## Experimental Configurations

Heterogeneous decision-making enables a systematic ablation matrix for cross-paradigm research. The following configurations illustrate the design using 2v2 soccer in [MOSAIC MultiGrid](https://mosaic-agent-level-interface.readthedocs.io/en/latest/documents/environments/mosaic_multigrid/index.html). Notation follows the paper's appendix.
- $\pi^{RL}$ denotes a solo‑trained RL policy (frozen at evaluation).
- $\lambda^{LLM}$ denotes an LLM agent.
- $\rho$ denotes a uniform random policy.
- $\nu$ denotes a no‑op (null action) policy.

### Adversarial Cross‑Paradigm Matchups

Testing how paradigms perform against each other. Each configuration pits two homogeneous teams against one another.

| Configuration | Team A | Team B | Purpose |
|---------------|--------|--------|---------|
| **A1** (RL vs RL) | $\pi^{RL}_1 + \pi^{RL}_2$ | $\pi^{RL}_3 + \pi^{RL}_4$ | Homogeneous RL baseline (ceiling) |
| **A2** (LLM vs LLM) | $\lambda^{LLM}_1 + \lambda^{LLM}_2$ | $\lambda^{LLM}_3 + \lambda^{LLM}_4$ | Homogeneous LLM baseline |
| **A3** (RL vs LLM) | $\pi^{RL}_1 + \pi^{RL}_2$ | $\lambda^{LLM}_1 + \lambda^{LLM}_2$ | Central cross‑paradigm comparison |
| **A4** (RL vs Random) | $\pi^{RL}_1 + \pi^{RL}_2$ | $\rho_1 + \rho_2$ | Sanity check (trained vs random) |

### Cooperative Heterogeneous Teams

Testing how paradigms work together **within** a team. All RL policies are trained solo (1v1) and frozen before deployment; LLM agents are zero‑shot.

| Configuration | Green Team | Blue Team |
|---------------|------------|-----------|
| **C1** (Heterogeneous vs Crippled) | $\pi^{RL} + \lambda^{LLM}$ | $\pi^{RL} + \rho$ |
| **C2** (Heterogeneous vs Solo) | $\pi^{RL} + \lambda^{LLM}$ | $\pi^{RL} + \nu$ |
| **C3** (Solo‑pair vs Solo‑pair) | $\pi^{RL}_i + \pi^{RL}_j$ | $\pi^{RL}_k + \pi^{RL}_l$ |
| **C4** (Heterogeneous vs Co‑trained) | $\pi^{RL} + \lambda^{LLM}$ | $\pi^{RL}_{2v2} + \pi^{RL}_{2v2}$ |

> **1v1‑to‑2v2 Transfer Design – Why Solo Training?**  
>  
> RL agents are trained as **solo experts in 1v1** (single‑agent environment), then deployed as teammates in 2v2 **without any fine‑tuning**. This design eliminates the *co‑training confound*: if agents were trained together in 2v2 via MAPPO self‑play, their policies would encode implicit partner models calibrated against another MAPPO agent. Swapping one teammate with an LLM would then conflate two effects – the paradigm difference **and** the partner mismatch. With 1v1‑trained agents, the RL policy carries **zero partner expectations** because it never had a partner, cleanly isolating the paradigm variable.  
>  
> This is distinct from **zero‑shot coordination (ZSC)** in the ad‑hoc teamwork literature. ZSC studies RL agents cooperating with unknown *RL* partners, agents that share the same observation and action representations ($\mathcal{O} = \mathbb{R}^d$, $\mathcal{A}$ discrete). Here we study an LLM as an ad‑hoc partner for a frozen RL policy – the partner is not only unknown but operates through a fundamentally different paradigm (text‑based reasoning vs. learned tensor‑to‑action mapping). The fair comparison baseline also changes: in ZSC the reference is a co‑trained RL+RL team, while here the appropriate baseline is **C3**: two independently trained 1v1 solo experts paired in 2v2, since neither agent was trained with any partner.


## Supported Environment Families (26)

MOSAIC supports **26 environment families** spanning single-agent, multi-agent, and cooperative/competitive paradigms. See the full [Environment Families](https://mosaic-agent-level-interface.readthedocs.io/en/latest/documents/environments/index.html) reference for installation instructions, environment lists, and academic citations.

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

- [Gymnasium](https://gymnasium.farama.org/): Standard RL API
- [PettingZoo](https://pettingzoo.farama.org/): Multi-agent environments
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Clean RL implementations
- [XuanCe](https://github.com/agi-brain/xuance): Multi-agent RL algorithms
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/): Distributed RL training
- [BALROG](https://github.com/balrog-ai/BALROG): LLM/VLM agent benchmark
- [llm_chess](https://github.com/maxim-saplin/llm_chess): LLM chess evaluation
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/): GUI framework
