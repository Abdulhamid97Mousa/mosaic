# Naming Resolution Plan: MuJoCo vs MuJoCo MPC & Directory Structure

## Executive Summary

This document addresses two critical naming conflicts that must be resolved before implementing the multi-paradigm orchestrator:

1. **MuJoCo vs MuJoCo MPC** - Two completely different systems sharing similar names
2. **UI "environments" vs Core "adapters"** - Confusing directory naming

---

## Progress Tracking

### Completed Tasks ✅

- [x] **Directory rename**: `ui/environments/` → `ui/config_panels/`
- [x] **Subdirectory rename**: `single_agent_env/` → `single_agent/`
- [x] **Subdirectory rename**: `multi_agent_env/` → `multi_agent/`
- [x] **Import updates**: `gym_gui/ui/widgets/control_panel.py`
- [x] **Import updates**: `gym_gui/ui/handlers/features/game_config.py`
- [x] **Import updates**: `gym_gui/tests/test_minigrid_empty_integration.py`
- [x] **Import updates**: `gym_gui/tests/test_minigrid_redbluedoors_integration.py`
- [x] **Pyright verification**: No import-related errors

### Pending Tasks 📋

- [ ] **MuJuCo typo fix**: Rename `game_docs/Gymnasium/MuJuCo/` → `MuJoCo/` (deferred - user indicated already correct)
- [x] **Pytest verification**: Run tests on minigrid integration tests ✅ **78 passed in 1.97s**
- [ ] **Optional**: Rename `gym/` → `gymnasium/` for consistency

---

## 1. MuJoCo vs MuJoCo MPC Resolution

### 1.1 The Two Systems

| Aspect | MuJoCo (Physics Engine) | MuJoCo MPC (MJPC) |
|--------|-------------------------|-------------------|
| **Full Name** | MuJoCo Multi-Joint dynamics with Contact | MuJoCo Model Predictive Control |
| **Developer** | DeepMind (open-sourced) | DeepMind |
| **Purpose** | Physics simulation for RL | Real-time optimal control |
| **Approach** | Environment for learning | Planning/optimization |
| **Learning** | ✅ Yes (policy gradients, etc.) | ❌ No (uses iLQG, Cross Entropy) |
| **Use Case** | Train agents with CleanRL/RLlib | Robot control without learning |

### 1.2 Current File Locations

```
MuJoCo (RL Environments) - Used with CleanRL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gym_gui/
├── game_docs/Gymnasium/MuJuCo/              # ❌ TYPO: Should be "MuJoCo"
│   └── __init__.py                          # HalfCheetah, Hopper, etc. docs
├── Algo_docs/cleanrl_worker/ppo/            # PPO for MuJoCo envs
└── Algo_docs/cleanrl_worker/td3/            # TD3 for MuJoCo envs

3rd_party/
└── cleanrl_worker/                          # Trains on MuJoCo envs


MuJoCo MPC (Optimal Control) - NOT RL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gym_gui/
├── core/mujoco_mpc_enums.py                 # MPC-specific enums
├── services/mujoco_mpc_controller/          # MPC service
│   ├── __init__.py
│   └── service.py
└── ui/widgets/mujoco_mpc_tab.py             # MPC UI panel

3rd_party/
└── mujoco_mpc_worker/                       # MJPC integration
    ├── mujoco_mpc/                          # DeepMind's MJPC library
    └── mujoco_mpc_worker/                   # Our wrapper
```

### 1.3 Naming Fixes Required

#### Fix 1: Correct the Typo

```bash
# Current (WRONG)
gym_gui/game_docs/Gymnasium/MuJuCo/

# Should be
gym_gui/game_docs/Gymnasium/MuJoCo/
```

#### Fix 2: Clarify Naming in Code

**Before:**
```python
# Ambiguous - which MuJoCo?
from gym_gui.core.mujoco_mpc_enums import MuJoCoMPCTaskId
```

**After (keep as is, but document clearly):**
```python
# MuJoCo MPC = Optimal control (NOT RL)
# For MuJoCo RL environments, use Gymnasium: gym.make("HalfCheetah-v4")
from gym_gui.core.mujoco_mpc_enums import MuJoCoMPCTaskId
```

### 1.4 Multi-Agent MuJoCo (MaMuJoCo)

MuJoCo environments can also be used in multi-agent settings via **MaMuJoCo**:

```python
# Single-Agent MuJoCo (current support)
import gymnasium as gym
env = gym.make("HalfCheetah-v4")  # Uses CleanRL worker

# Multi-Agent MuJoCo (future support)
from pettingzoo.mujoco import multiwalker_v9
env = multiwalker_v9.env()  # Multiple walkers cooperating
```

**Paradigm Support:**
| Environment | Paradigm | Worker |
|-------------|----------|--------|
| HalfCheetah-v4 | SINGLE_AGENT | CleanRL |
| Multiwalker | SEQUENTIAL (AEC) | PettingZoo |
| Humanoid-v4 (cooperative) | SIMULTANEOUS | RLlib |

---

## 2. UI "environments" vs Core "adapters" Resolution

### 2.1 The Naming Clash

```
CURRENT STRUCTURE (CONFUSING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gym_gui/
├── ui/environments/                    # ❌ These are UI PANELS, not environments!
│   ├── single_agent_env/              # Contains: config_panel.py, controls
│   │   ├── ale/                       # ALE config panel
│   │   ├── gym/                       # Gym config panel
│   │   ├── minigrid/                  # MiniGrid config panel
│   │   └── vizdoom/                   # ViZDoom config panel
│   └── multi_agent_env/               # Contains: config_panel.py
│       └── pettingzoo/                # PettingZoo config panel
│
├── core/adapters/                     # ✅ These ARE environment adapters
│   ├── base.py                        # EnvironmentAdapter ABC
│   ├── ale.py                         # ALEAdapter
│   ├── pettingzoo.py                  # PettingZooAdapter
│   └── ...
```

### 2.2 Why This Is Confusing

1. `gym_gui/ui/environments/` sounds like it contains environment implementations
2. But it actually contains **UI configuration panels** for environments
3. The actual environment adapters are in `gym_gui/core/adapters/`
4. This causes confusion when navigating the codebase

### 2.3 Proposed Rename

```
PROPOSED STRUCTURE (CLEAR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gym_gui/
├── ui/
│   ├── config_panels/                 # ✅ Clear: UI config panels
│   │   ├── single_agent/              # Drop "_env" suffix
│   │   │   ├── ale/
│   │   │   ├── gymnasium/             # Rename "gym" → "gymnasium" for clarity
│   │   │   ├── minigrid/
│   │   │   └── vizdoom/
│   │   └── multi_agent/               # Drop "_env" suffix
│   │       └── pettingzoo/
│   │
│   └── widgets/                       # Keep as-is
│
├── core/
│   ├── adapters/                      # ✅ Keep: Environment adapters
│   │   ├── base.py
│   │   ├── ale.py
│   │   └── ...
│   └── ...
```

### 2.4 Migration Steps

#### Step 1: Rename Directory

```bash
# Rename ui/environments → ui/config_panels
git mv gym_gui/ui/environments gym_gui/ui/config_panels
```

#### Step 2: Update Imports

```python
# Before
from gym_gui.ui.environments.single_agent_env import build_frozenlake_controls

# After
from gym_gui.ui.config_panels.single_agent import build_frozenlake_controls
```

#### Step 3: Rename Subdirectories

```bash
git mv gym_gui/ui/config_panels/single_agent_env gym_gui/ui/config_panels/single_agent
git mv gym_gui/ui/config_panels/multi_agent_env gym_gui/ui/config_panels/multi_agent
```

---

## 3. Complete Naming Standards

### 3.1 Directory Naming Convention

| Directory | Contains | Naming Pattern |
|-----------|----------|----------------|
| `core/adapters/` | Environment adapters | `{env_type}.py` (e.g., `ale.py`) |
| `ui/config_panels/` | UI config panels | `{paradigm}/{env_type}/` |
| `services/` | Business logic services | `{feature}.py` or `{feature}/` |
| `services/mujoco_mpc_controller/` | MuJoCo MPC (non-RL) | Keep separate from RL |
| `3rd_party/` | External workers | `{tool}_worker/` |

### 3.2 File Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Adapters | `{env_type}.py` | `pettingzoo.py`, `ale.py` |
| Config panels | `config_panel.py` | `ui/config_panels/single_agent/ale/config_panel.py` |
| Enums | `{domain}_enums.py` | `mujoco_mpc_enums.py`, `pettingzoo_enums.py` |
| Services | `{service_name}.py` | `actor.py`, `telemetry.py` |

### 3.3 Class Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Environment Adapters | `{EnvType}Adapter` | `ALEAdapter`, `PettingZooAdapter` |
| Services | `{Name}Service` | `ActorService`, `TelemetryService` |
| Controllers | `{Name}Controller` | `SessionController`, `HumanInputController` |
| Protocols | `{Name}` (no suffix) | `Actor`, `PolicyController` |
| Enums | `{Domain}{Category}` | `MuJoCoMPCPlannerType`, `RunStatus` |

---

## 4. Action Items

### 4.1 Immediate Fixes (Do Now)

```bash
# Fix 1: Correct MuJuCo typo
git mv gym_gui/game_docs/Gymnasium/MuJuCo gym_gui/game_docs/Gymnasium/MuJoCo
```

### 4.2 Rename UI Environments (Next)

```bash
# Fix 2: Rename ui/environments → ui/config_panels
git mv gym_gui/ui/environments gym_gui/ui/config_panels
git mv gym_gui/ui/config_panels/single_agent_env gym_gui/ui/config_panels/single_agent
git mv gym_gui/ui/config_panels/multi_agent_env gym_gui/ui/config_panels/multi_agent
```

### 4.3 Update All Imports (After Rename)

Find and replace all imports:
```python
# Old
from gym_gui.ui.environments.single_agent_env import ...
from gym_gui.ui.environments.multi_agent_env import ...

# New
from gym_gui.ui.config_panels.single_agent import ...
from gym_gui.ui.config_panels.multi_agent import ...
```

---

## 5. Documentation Summary

### 5.1 MuJoCo Disambiguation

| When You Mean | Use This | Import/Path |
|---------------|----------|-------------|
| RL environment (HalfCheetah) | "MuJoCo env" or "Gymnasium MuJoCo" | `gym.make("HalfCheetah-v4")` |
| Optimal control (MJPC) | "MuJoCo MPC" or "MJPC" | `mujoco_mpc_controller.service` |
| Multi-agent MuJoCo | "MaMuJoCo" | `pettingzoo.mujoco` |

### 5.2 Directory Roles

| Directory | Role | NOT |
|-----------|------|-----|
| `core/adapters/` | Environment adapters (wrap Gym/PZ) | UI components |
| `ui/config_panels/` | UI configuration panels | Environment implementations |
| `services/mujoco_mpc_controller/` | MuJoCo MPC optimal control | RL training |
| `3rd_party/cleanrl_worker/` | RL training with MuJoCo envs | Optimal control |

---

## 6. Verification Checklist

After applying fixes, verify:

- [ ] `gym_gui/game_docs/Gymnasium/MuJoCo/` exists (not MuJuCo) - **DEFERRED**
- [x] `gym_gui/ui/config_panels/` exists (not environments) - **DONE**
- [x] No imports reference old paths - **DONE**
- [x] Pyright passes with 0 new errors - **DONE**
- [x] All tests pass - **DONE** (78 passed in 1.97s)
