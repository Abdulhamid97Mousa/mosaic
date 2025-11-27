# MuJoCo MPC Integration Plan

## Scope

**Goal**: Display MuJoCo MPC visualization in the GUI's render area.

This is a visualization integration - we launch MJPC and display it in our render area. The MuJoCo MPC library handles all the control/planning internally.

## Key Architectural Challenge

### Current Architecture

The existing `gym_gui` is built around **RL training paradigms**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Current Tabs                              │
├─────────────────────────────────────────────────────────────────┤
│  Human Control Tab    │  Uses ControlMode enum                   │
│  Single-Agent Mode    │  CleanRL trainer via gRPC                │
│  Multi-Agent Mode     │  Jason BDI + SPADE workers               │
└─────────────────────────────────────────────────────────────────┘
```

**Core Assumption**: All modes interact with Gymnasium environments using:
- `ControlMode` enum (`HUMAN_ONLY`, `AGENT_ONLY`, `HYBRID_*`, `MULTI_AGENT_*`)
- The trainer service (`gym_gui/services/trainer/`) for RL training
- Episode/Step telemetry model (`EpisodeRollup`, `StepRecord`)

### MuJoCo MPC is Different

MuJoCo MPC is **NOT** an RL training system. It is a **real-time predictive controller**:

| Aspect | CleanRL (Current) | MuJoCo MPC |
|--------|-------------------|------------|
| **Purpose** | Train policies via RL | Execute optimal control in real-time |
| **Learning** | Learns from experience | No learning - solves optimization |
| **Planner** | Neural network policy | iLQG, Gradient Descent, Predictive Sampling |
| **Output** | Trained model (.pt) | Real-time control actions |
| **Interaction** | Gym env step loop | gRPC server with `Agent` class |
| **Model** | MuJoCo via Gymnasium | Native MuJoCo (mjModel, mjData) |
| **Tasks** | Generic Gym tasks | MJPC-specific tasks (Cartpole, Humanoid Track, etc.) |

## The Fundamental Question

**Should MuJoCo MPC fit into the existing `ControlMode` paradigm?**

### Option A: Extend ControlMode (Not Recommended)

Add new control modes like `MPC_CONTROL`:
```python
class ControlMode(StrEnum):
    # ... existing modes ...
    MPC_CONTROL = "mpc_control"  # New
```

**Problems**:
1. `ControlMode` is designed for RL agent control, not optimal control planners
2. MPC doesn't "learn" - it solves trajectory optimization each timestep
3. MPC uses its own task definitions, not Gymnasium envs
4. Would pollute the RL-focused enums with non-RL concepts

### Option B: Separate Domain (Recommended)

Create a **parallel domain** for robotics/MPC control:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GUI_BDI_RL Domains                          │
├─────────────────────────────────────────────────────────────────┤
│  RL Domain                   │  Robotics/MPC Domain             │
│  ─────────────────────────   │  ─────────────────────────       │
│  • Human Control Tab         │  • MuJoCo MPC Tab (NEW)          │
│  • Single-Agent Mode         │    - Task selection              │
│  • Multi-Agent Mode          │    - Planner configuration       │
│  • CleanRL trainer           │    - Real-time visualization     │
│  • ControlMode enum          │    - Cost/residual monitoring    │
│  • Gymnasium environments    │  • Native MuJoCo tasks           │
│  • Episode/Step telemetry    │  • Trajectory telemetry          │
└─────────────────────────────────────────────────────────────────┘
```

## Naming Convention

**Explicit naming throughout** - always use `mujoco_mpc` prefix:

| Component | Path |
|-----------|------|
| Enums | `gym_gui/core/mujoco_mpc_enums.py` |
| Service | `gym_gui/services/mujoco_mpc_controller/` |
| UI Tab | `gym_gui/ui/widgets/mujoco_mpc_tab.py` |
| Worker | `3rd_party/mujoco_mpc_worker/` |

## Proposed Architecture

### 1. New Enums (Don't Extend Existing)

Create `gym_gui/core/mujoco_mpc_enums.py`:

```python
class MuJoCoMPCPlannerType(StrEnum):
    """MuJoCo MPC planner algorithms."""
    ILQG = "ilqg"
    GRADIENT_DESCENT = "gradient_descent"
    PREDICTIVE_SAMPLING = "predictive_sampling"
    CROSS_ENTROPY = "cross_entropy"

class MuJoCoMPCTaskId(StrEnum):
    """MJPC built-in task identifiers."""
    CARTPOLE = "Cartpole"
    PARTICLE = "Particle"
    SWIMMER = "Swimmer"
    QUADRUPED = "Quadruped"
    HUMANOID_TRACK = "Humanoid Track"
    HUMANOID_STAND = "Humanoid Stand"
    WALKER = "Walker"
    # ... etc
```

### 2. Directory Structure

Following the same pattern as `cleanrl_worker`:

```
3rd_party/
└── mujoco_mpc_worker/
    ├── mujoco_mpc/                  # Git submodule (VENDORED - DO NOT TOUCH)
    │   ├── mjpc/                    # C++ MJPC core
    │   ├── python/                  # Python bindings
    │   │   └── mujoco_mpc/
    │   │       ├── agent.py         # gRPC client for MJPC
    │   │       ├── direct.py        # Direct trajectory optimization
    │   │       └── filter.py        # State estimation
    │   └── README.md
    ├── mujoco_mpc_worker/           # OUR wrapper code (lives here)
    │   ├── __init__.py
    │   ├── cli.py                   # CLI entry point
    │   ├── config.py                # Configuration management
    │   ├── agent_wrapper.py         # Wrapper around vendored Agent class
    │   ├── task_registry.py         # MJPC task definitions
    │   └── telemetry.py             # Cost/trajectory telemetry adapter
    ├── pyproject.toml
    └── tests/
```

**Key Point**:
- `mujoco_mpc/` = vendored submodule (DO NOT TOUCH)
- `mujoco_mpc_worker/` = our wrapper code

This mirrors the `cleanrl_worker` pattern:
- `cleanrl_worker/cleanrl/` = vendored
- `cleanrl_worker/cleanrl_worker/` = our code

### 3. Service Architecture

**Key Insight**: MuJoCo MPC already uses gRPC internally!

```
┌─────────────────────────────────────────────────────────────────┐
│                     MuJoCo MPC Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  gym_gui (PyQt)                                                  │
│       │                                                          │
│       ▼                                                          │
│  mujoco_mpc_worker (our wrapper)                                 │
│       │           └── 3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/
│       ▼                                                          │
│  mujoco_mpc.Agent (Python gRPC client)                           │
│       │           └── 3rd_party/mujoco_mpc_worker/mujoco_mpc/python/
│       │                                                          │
│       │ gRPC (localhost:port)                                    │
│       ▼                                                          │
│  agent_server (C++ binary, launched by Agent)                    │
│       │                                                          │
│       ▼                                                          │
│  MJPC Planner (iLQG / Gradient / Sampling)                       │
│       │                                                          │
│       ▼                                                          │
│  MuJoCo Physics + Visualization                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Do NOT reuse `gym_gui/services/trainer/`** - it's designed for RL training runs, not real-time control.

Instead, create a new service: `gym_gui/services/mujoco_mpc_controller/`

### 4. UI Integration

Create a new tab that is **not** a `ControlMode` variant:

```
gym_gui/ui/widgets/
├── mujoco_mpc_tab.py                    # NEW: Main MPC control tab
├── mujoco_mpc_task_selector.py          # NEW: Task/model selection
├── mujoco_mpc_planner_config.py         # NEW: Planner parameter tuning
├── mujoco_mpc_cost_monitor.py           # NEW: Real-time cost visualization
└── mujoco_mpc_trajectory_view.py        # NEW: Trajectory preview
```

### 5. Requirements

Create `requirements/mujoco_mpc_worker.txt`:

```txt
# MuJoCo MPC Worker Dependencies
# This bundle contains dependencies required exclusively by the mujoco_mpc_worker.
# Install this when using MuJoCo MPC for robotics control visualization.

# Include base requirements
-r base.txt

# Core MuJoCo
mujoco>=3.0.0

# gRPC for MJPC communication
grpcio>=1.50.0
grpcio-tools>=1.50.0

# Protobuf
protobuf>=4.21.0

# NumPy for array operations
numpy>=1.24.0

# Note: The MJPC agent_server binary must be built separately
# See: 3rd_party/mujoco_mpc_worker/mujoco_mpc/README.md
```

### 6. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mosaic-mujoco-mpc"
version = "0.1.0"
description = "MuJoCo MPC integration for MOSAIC BDI-RL framework"
requires-python = ">=3.10"
license = {text = "Apache-2.0"}
# Dependencies are managed via the main project's requirements.txt
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[project.scripts]
mujoco-mpc-worker = "mujoco_mpc_worker.cli:main"

[tool.setuptools]
packages = [
    # Original MuJoCo MPC Python API (from mujoco_mpc/python/ subdirectory)
    "mujoco_mpc",
    # Our integration wrapper
    "mujoco_mpc_worker",
]

[tool.setuptools.package-dir]
# Map vendored mujoco_mpc Python API
"mujoco_mpc" = "mujoco_mpc/python/mujoco_mpc"
# Our wrapper is directly in mujoco_mpc_worker/
"mujoco_mpc_worker" = "mujoco_mpc_worker"
```

## Implementation Phases

### Phase 1: Worker Setup
1. Create `requirements/mujoco_mpc_worker.txt`
2. Create `3rd_party/mujoco_mpc_worker/pyproject.toml`
3. Create `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/` scaffold:
   - `__init__.py`
   - `cli.py`
   - `config.py`
4. Update `.gitmodules` with mujoco_mpc submodule entry
5. Test basic import: `from mujoco_mpc import Agent`

### Phase 2: Wrapper Layer
All code in `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/`:

1. `agent_wrapper.py` - wraps vendored `mujoco_mpc.Agent`
2. `task_registry.py` - enumerate MJPC tasks
3. `config.py` - configuration dataclasses
4. `telemetry.py` - adapt MJPC metrics to our format

### Phase 3: Service Layer
1. Create `gym_gui/services/mujoco_mpc_controller/` service
2. Implement lifecycle management (start/stop MJPC server)
3. Implement state synchronization
4. Implement telemetry streaming (costs, trajectories)

### Phase 4: UI Integration
1. Create `gym_gui/core/mujoco_mpc_enums.py` (separate from `enums.py`)
2. Create `gym_gui/ui/widgets/mujoco_mpc_tab.py`
3. Integrate into `main_window.py` as new top-level tab
4. Embed MJPC visualization in render area

### Phase 5: Build System
1. Document MJPC C++ build requirements (CMake, clang)
2. Create build script for `agent_server` binary
3. Add CI/CD configuration for MPC worker

## Open Questions

1. **Binary Distribution**: How to distribute pre-built `agent_server`?
   - Build from source (current approach in MJPC)
   - Ship pre-built binaries
   - Docker container

2. **Task Model Loading**: How to handle custom MuJoCo models?
   - Use MJPC's built-in tasks only
   - Allow custom XML loading
   - Both

3. **Visualization Embedding**: How to embed MJPC visualization?
   - Capture frames and display in PyQt
   - Launch MJPC GUI as separate window
   - Use MuJoCo's passive viewer via Python

4. **Telemetry Unification**: Should MPC telemetry use same SQLite store?
   - Pro: Unified analytics
   - Con: Schema mismatch (no episodes/steps)

## Summary

MuJoCo MPC should be integrated as a **separate domain** within GUI_BDI_RL, not as an extension of the existing RL control modes. This keeps the architecture clean and reflects the fundamental difference between:

- **RL Training**: Learning optimal behavior from experience
- **MPC Control**: Solving trajectory optimization in real-time

### Key Files to Create

| File | Purpose |
|------|---------|
| `requirements/mujoco_mpc_worker.txt` | Dependencies |
| `3rd_party/mujoco_mpc_worker/pyproject.toml` | Package definition |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/__init__.py` | Worker module |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/agent_wrapper.py` | Wrapper for vendored Agent |
| `gym_gui/core/mujoco_mpc_enums.py` | MPC-specific enums |
| `gym_gui/services/mujoco_mpc_controller/service.py` | Service layer |
| `gym_gui/ui/widgets/mujoco_mpc_tab.py` | UI tab |

### Pattern Reference

This follows the same pattern as `cleanrl_worker`:

```
cleanrl_worker/cleanrl/           →  mujoco_mpc_worker/mujoco_mpc/
cleanrl_worker/cleanrl_worker/    →  mujoco_mpc_worker/mujoco_mpc_worker/
```

Vendored code stays untouched. Our wrapper code lives in the `*_worker/` subdirectory.
