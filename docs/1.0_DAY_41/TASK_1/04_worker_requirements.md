# Worker Requirements Architecture

## Overview

MOSAIC supports multiple RL training backends (workers) with modular dependency management. Each worker has its own requirements file and package structure, allowing users to install only what they need.

## Directory Structure

```
GUI_BDI_RL/
├── requirements/
│   ├── base.txt                 # Core GUI + shared infrastructure
│   ├── cleanrl_worker.txt       # CleanRL dependencies
│   ├── ray_worker.txt           # Ray/RLlib dependencies
│   ├── xuance_worker.txt        # XuanCe MARL dependencies
│   ├── jason_worker.txt         # Jason BDI dependencies
│   ├── llm_worker.txt           # LLM/Ollama dependencies
│   ├── spade_bdi_worker.txt     # SPADE BDI dependencies
│   ├── pettingzoo.txt           # PettingZoo environments
│   └── vizdoom.txt              # ViZDoom environments
│
├── 3rd_party/
│   ├── cleanrl_worker/
│   │   ├── pyproject.toml       # mosaic-cleanrl package
│   │   ├── cleanrl_worker/      # Integration harness
│   │   └── cleanrl/             # Original CleanRL (git submodule)
│   │
│   ├── ray_worker/
│   │   ├── pyproject.toml       # mosaic-ray-worker package
│   │   ├── ray_worker/          # Integration harness
│   │   └── ray/                 # Original Ray (git submodule)
│   │
│   └── xuance_worker/
│       ├── pyproject.toml       # mosaic-xuance-worker package
│       ├── xuance_worker/       # Integration harness
│       └── xuance/              # Original XuanCe (git submodule)
│
└── pyproject.toml               # Main package with optional dependencies
```

## Installation Options

### Via requirements.txt (Traditional)

```bash
# GUI only (no workers)
pip install -r requirements/base.txt

# With CleanRL worker
pip install -r requirements/cleanrl_worker.txt

# With Ray/RLlib worker
pip install -r requirements/ray_worker.txt

# With XuanCe MARL worker
pip install -r requirements/xuance_worker.txt

# Full installation (all default workers)
pip install -r requirements.txt
```

### Via pyproject.toml (Modern)

```bash
# Minimal GUI only
pip install -e .

# With specific workers
pip install -e ".[cleanrl]"
pip install -e ".[ray-rllib]"
pip install -e ".[xuance]"

# With multiple workers
pip install -e ".[cleanrl,ray-rllib]"

# Full development setup
pip install -e ".[full]"
```

## Worker Comparison

| Worker | Paradigm | Use Case | Key Dependencies |
|--------|----------|----------|------------------|
| **CleanRL** | Single-Agent | Simple RL training | torch, tensorboard, wandb |
| **Ray RLlib** | Multi-Agent | Distributed training | ray[rllib], ray[tune] |
| **XuanCe** | MARL | Multi-agent algorithms | torch, mpi4py, pettingzoo |
| **LLM** | Any | Language model agents | ollama, langchain |
| **Jason BDI** | Sequential | BDI agents | Java runtime |
| **SPADE BDI** | Sequential | Python BDI agents | spade, spade-bdi |

## pyproject.toml Optional Dependencies

```toml
[project.optional-dependencies]
# Workers
cleanrl = ["torch>=2.0.0", "tensorboard>=2.11.0", "wandb>=0.22.3", ...]
ray-rllib = ["ray[rllib]>=2.9.0", "ray[tune]>=2.9.0", "torch>=2.0.0", ...]
xuance = ["torch>=2.0.0", "scipy>=1.5.0", "mpi4py>=3.1.0", ...]
llm = ["ollama>=0.3.0", "langchain>=0.3.0", ...]
spade-bdi = ["spade>=4.0.0", "spade-bdi>=0.3.0"]

# Environment families
pettingzoo = ["pettingzoo[classic,butterfly,mpe,sisl]>=1.24.0", ...]
vizdoom = ["vizdoom>=1.2.0,<2.0.0"]

# Convenience bundles
all-envs = ["gym-gui[box2d,mujoco,atari,minigrid,pettingzoo,vizdoom]"]
full = ["gym-gui[all-envs,cleanrl,dev]"]
```

## Worker Configuration in UI

The Advanced tab's **Worker Configuration Panel** shows different options based on the selected worker:

### Local (Default)
- Render Mode: human, rgb_array, ansi, none
- Record Video: bool

### CleanRL
- Algorithm: PPO, DQN, A2C, SAC, TD3, DDPG
- Learning Rate: float
- Total Timesteps: int
- Parallel Envs: int
- Capture Video: bool

### Ray RLlib
- Algorithm: PPO, DQN, A2C, IMPALA, APPO, SAC
- Num Workers: int
- Envs per Worker: int
- Framework: torch, tf2

### XuanCe
- Algorithm: MAPPO, MADDPG, QMIX, VDN, COMA, IPPO, IQL
- Learning Rate: float
- Batch Size: int
- Backend: torch, tensorflow, mindspore

### LLM
- Model: gpt-4, gpt-3.5-turbo, claude-3, llama-3, ollama-local
- Temperature: float
- Max Tokens: int
- System Prompt: text

### Jason BDI
- Agent File (.asl): text
- MAS File (.mas2j): text
- Debug Mode: bool

### SPADE BDI
- XMPP Server: text
- Agent JID: text
- Debug Mode: bool

## Adding a New Worker

1. Create directory structure:
   ```
   3rd_party/myworker/
   ├── pyproject.toml
   ├── myworker/
   │   └── __init__.py
   └── original_lib/  (optional git submodule)
   ```

2. Create requirements file:
   ```
   requirements/myworker.txt
   ```

3. Add to `pyproject.toml`:
   ```toml
   myworker = ["dep1>=1.0", "dep2>=2.0"]
   ```

4. Add worker config schema in `worker_config_panel.py`:
   ```python
   WORKER_CONFIG_SCHEMAS["myworker"] = {
       "display_name": "My Worker",
       "fields": [...]
   }
   ```

5. Add to `AVAILABLE_WORKERS` in `agent_config_table.py`:
   ```python
   ("myworker", "My Worker", "Description")
   ```

## Related Documents

- [TASK_1 README](./README.md) - Multi-Paradigm Orchestrator overview
- [PolicyMappingService Plan](./03_policy_mapping_service_plan.md) - Service architecture
- [TASK_3 README](../TASK_3/README.md) - UI Architecture
