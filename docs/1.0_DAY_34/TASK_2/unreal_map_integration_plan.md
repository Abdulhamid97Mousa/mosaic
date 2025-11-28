# Unreal-MAP Integration Plan

**Date:** 2025-11-27
**Status:** Planning
**Goal:** Integrate Unreal-MAP as a worker for Multi-Agent Mode in the GUI sidebar

---

## 1. Overview

### What is Unreal-MAP?

Unreal-MAP (Multi-Agent Playground) is a multi-agent simulation platform based on Unreal Engine, designed for:
- **Multi-Agent Reinforcement Learning (MARL)** training
- **Large-scale, heterogeneous, multi-team** simulations
- **Adversarial training** between swarms and algorithms
- High-performance training (TPS up to 10k+, FPS up to 10M+)

### Key Architectural Difference from Single-Agent

| Aspect | Single-Agent (CleanRL) | Multi-Agent (Unreal-MAP) |
|--------|------------------------|--------------------------|
| **Agents** | One agent per environment | Multiple agents per team, multiple teams |
| **Environment** | Gymnasium-based | Unreal Engine-based (custom protocol) |
| **Training** | PPO, DQN, etc. for single policy | MARL algorithms (PPO-MA, etc.) |
| **Communication** | gRPC to trainer | TCP/UDP to Unreal Engine server |
| **Tasks** | Gym env IDs | UHMAP SubTasks (Adversial, Formation, etc.) |
| **Rendering** | Gym's rgb_array | Cross-platform (Linux train, Windows render) |

---

## 2. Integration Architecture

### 2.1 Following the Worker Pattern

Following the established pattern from `cleanrl_worker` and `mujoco_mpc_worker`:

```
3rd_party/unreal_map_worker/
├── unreal-map/                      # Git submodule (VENDORED - DO NOT MODIFY)
│   ├── PythonExample/
│   │   └── hmp_minimal_modules/     # Python training framework
│   │       ├── ALGORITHM/           # MARL algorithms
│   │       ├── MISSION/             # Task definitions
│   │       ├── UTIL/                # Utilities
│   │       ├── config.py            # GlobalConfig
│   │       ├── main.py              # Entry point
│   │       └── task_runner.py       # Training runner
│   ├── Source/                      # UE4 C++ source
│   └── README.md
│
└── unreal_map_worker/               # OUR wrapper code (lives here)
    ├── __init__.py                  # Package exports
    ├── cli.py                       # CLI entry point
    ├── config.py                    # Configuration dataclasses
    ├── enums.py                     # Task and algorithm enums
    ├── launcher.py                  # Process launcher for Unreal server
    ├── client.py                    # Client wrapper for GUI integration
    └── telemetry.py                 # Telemetry adapter for GUI
```

### 2.2 GUI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GUI_BDI_RL Domains                          │
├─────────────────────────────────────────────────────────────────┤
│  Single-Agent Domain             │  Multi-Agent Domain          │
│  ─────────────────────────       │  ─────────────────────────   │
│  • Human Control Tab             │  • Multi-Agent Tab (NEW)     │
│  • Single-Agent Mode             │    - Environment selection   │
│  • CleanRL trainer               │    - Task configuration      │
│  • Gymnasium environments        │    - Team configuration      │
│                                  │    - Algorithm per team      │
│  Robotics/MPC Domain             │    - Training controls       │
│  ─────────────────────────       │  • Unreal-MAP tasks          │
│  • MuJoCo MPC Tab                │  • MARL algorithms           │
│                                  │  • Multi-team telemetry      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 New Enums

Create `gym_gui/core/unreal_map_enums.py`:

```python
from enum import Enum
from gym_gui.core.enums import StrEnum


class UnrealMAPTaskId(StrEnum):
    """Unreal-MAP built-in task identifiers (SubTasks)."""
    UHMAP_ADVERSIAL = "uhmap_adversial"
    UHMAP_FORMATION = "uhmap_formation"
    UHMAP_CARRIER = "uhmap_carrier"
    UHMAP_ESCAPE = "uhmap_escape"
    UHMAP_BREAKING_BAD = "uhmap_breaking_bad"
    UHMAP_LARGE_SCALE = "uhmap_large_scale"
    UHMAP_ATTACK_POST = "uhmap_attack_post"
    UHMAP_INTERCEPT = "uhmap_intercept"
    UHMAP_PREY_PREDATOR = "uhmap_prey_predator"
    UHMAP_WATERDROP = "uhmap_waterdrop"
    UHMAP_JUST_AN_ISLAND = "uhmap_just_an_island"
    UHMAP_REPRODUCE = "uhmap_reproduce"


class UnrealMAPAlgorithm(StrEnum):
    """MARL algorithms available in Unreal-MAP."""
    PPO_MA = "ppo_ma"
    HETE_LEAGUE = "hete_league_onenet_fix"
    SCRIPT_AI = "script_ai"
    RANDOM = "random"
    MY_AI = "my_ai"


class UnrealMAPDrawMode(StrEnum):
    """Visualization modes for Unreal-MAP."""
    OFF = "OFF"
    IMG = "Img"
    THREEJS = "Threejs"
    WEB = "Web"
    NATIVE = "Native"
```

### 3.2 Worker Package Structure

**`3rd_party/unreal_map_worker/unreal_map_worker/__init__.py`**:

```python
"""Unreal-MAP worker integration for MOSAIC BDI-RL framework."""

from .config import UnrealMAPConfig, TeamConfig
from .enums import UnrealMAPTaskId, UnrealMAPAlgorithm, UnrealMAPDrawMode
from .launcher import UnrealMAPLauncher, get_launcher

__all__ = [
    "UnrealMAPConfig",
    "TeamConfig",
    "UnrealMAPTaskId",
    "UnrealMAPAlgorithm",
    "UnrealMAPDrawMode",
    "UnrealMAPLauncher",
    "get_launcher",
]
```

**`3rd_party/unreal_map_worker/unreal_map_worker/config.py`**:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from .enums import UnrealMAPTaskId, UnrealMAPAlgorithm, UnrealMAPDrawMode


@dataclass
class TeamConfig:
    """Configuration for a single team in multi-agent training."""
    team_id: int
    algorithm: UnrealMAPAlgorithm
    n_agents: int
    policy_path: Optional[str] = None  # For loading pre-trained policy


@dataclass
class UnrealMAPConfig:
    """Configuration for Unreal-MAP training session."""
    # Task configuration
    task_id: UnrealMAPTaskId = UnrealMAPTaskId.UHMAP_ADVERSIAL

    # Teams configuration (multi-team support)
    teams: List[TeamConfig] = field(default_factory=list)

    # Training parameters
    seed: int = 42
    num_threads: int = 64  # Parallel environments
    n_parallel_frame: int = 5_000_000
    max_n_episode: int = 200_000

    # Visualization
    draw_mode: UnrealMAPDrawMode = UnrealMAPDrawMode.OFF

    # Device
    device: str = "cuda"  # 'cpu', 'cuda', 'cuda:0'

    # Logging
    note: str = "mosaic_training"
    activate_logger: bool = True

    # Episode alignment
    align_episode: bool = True

    # Testing
    train_time_testing: bool = True
    test_interval: int = 2048
    test_epoch: int = 32
```

**`3rd_party/unreal_map_worker/unreal_map_worker/launcher.py`**:

```python
"""Launcher for Unreal-MAP training processes."""

import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class UnrealMAPProcess:
    """Represents a running Unreal-MAP training process."""
    instance_id: int
    process: subprocess.Popen
    task_id: str
    config_path: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    def terminate(self) -> None:
        if self.is_running:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class UnrealMAPLauncher:
    """Launcher for Unreal-MAP training instances."""

    def __init__(self):
        self._base_path = Path(__file__).parent.parent / "unreal-map"
        self._python_path = self._base_path / "PythonExample" / "hmp_minimal_modules"
        self._processes: dict[int, UnrealMAPProcess] = {}
        self._next_id = 1

    def is_available(self) -> bool:
        """Check if Unreal-MAP Python modules are available."""
        main_py = self._python_path / "main.py"
        return main_py.exists()

    def get_status(self) -> dict:
        """Get detailed status information."""
        return {
            "available": self.is_available(),
            "python_path": str(self._python_path),
            "running_instances": len(self.list_running()),
        }

    def launch(
        self,
        config_path: Optional[str] = None,
        task_id: str = "uhmap_adversial",
    ) -> Tuple[Optional[UnrealMAPProcess], str]:
        """Launch a new Unreal-MAP training instance."""
        if not self.is_available():
            return None, "Unreal-MAP Python modules not found"

        instance_id = self._next_id
        self._next_id += 1

        cmd = ["python", "main.py"]
        if config_path:
            cmd.extend(["--cfg", config_path])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._python_path)

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self._python_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc = UnrealMAPProcess(
                instance_id=instance_id,
                process=process,
                task_id=task_id,
                config_path=config_path,
            )
            self._processes[instance_id] = proc
            return proc, f"Launched Unreal-MAP instance {instance_id}"
        except Exception as e:
            return None, f"Failed to launch: {e}"

    def terminate(self, instance_id: int) -> bool:
        """Terminate a specific instance."""
        proc = self._processes.get(instance_id)
        if proc is None:
            return False
        proc.terminate()
        del self._processes[instance_id]
        return True

    def terminate_all(self) -> int:
        """Terminate all instances."""
        count = 0
        for proc in list(self._processes.values()):
            proc.terminate()
            count += 1
        self._processes.clear()
        return count

    def list_running(self) -> List[UnrealMAPProcess]:
        """List only running processes."""
        running = []
        to_remove = []
        for instance_id, proc in self._processes.items():
            if proc.is_running:
                running.append(proc)
            else:
                to_remove.append(instance_id)
        for instance_id in to_remove:
            del self._processes[instance_id]
        return running


_launcher: Optional[UnrealMAPLauncher] = None


def get_launcher() -> UnrealMAPLauncher:
    """Get the singleton launcher instance."""
    global _launcher
    if _launcher is None:
        _launcher = UnrealMAPLauncher()
    return _launcher
```

### 3.3 GUI Multi-Agent Tab

**`gym_gui/ui/widgets/unreal_map_tab.py`**:

```python
"""Sidebar tab for Unreal-MAP multi-agent configuration."""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.core.unreal_map_enums import (
    UnrealMAPTaskId,
    UnrealMAPAlgorithm,
    UnrealMAPDrawMode,
)


class UnrealMAPTab(QtWidgets.QWidget):
    """Tab widget for Unreal-MAP multi-agent configuration."""

    # Signals
    launch_training_requested = pyqtSignal(dict)  # config dict
    stop_training_requested = pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Task Selection Group
        task_group = QtWidgets.QGroupBox("Task Selection", self)
        task_layout = QtWidgets.QFormLayout(task_group)

        self._task_combo = QtWidgets.QComboBox(task_group)
        for task in UnrealMAPTaskId:
            display = task.value.replace("_", " ").title()
            self._task_combo.addItem(display, task)
        task_layout.addRow("Task:", self._task_combo)

        layout.addWidget(task_group)

        # Team Configuration Group
        team_group = QtWidgets.QGroupBox("Team Configuration", self)
        team_layout = QtWidgets.QVBoxLayout(team_group)

        # Number of teams
        teams_row = QtWidgets.QHBoxLayout()
        teams_row.addWidget(QtWidgets.QLabel("Number of Teams:"))
        self._num_teams_spin = QtWidgets.QSpinBox(team_group)
        self._num_teams_spin.setRange(1, 4)
        self._num_teams_spin.setValue(2)
        teams_row.addWidget(self._num_teams_spin)
        team_layout.addLayout(teams_row)

        # Team details (dynamic based on num_teams)
        self._team_widgets: list[dict] = []
        self._team_container = QtWidgets.QWidget(team_group)
        self._team_container_layout = QtWidgets.QVBoxLayout(self._team_container)
        team_layout.addWidget(self._team_container)

        layout.addWidget(team_group)

        # Training Parameters Group
        params_group = QtWidgets.QGroupBox("Training Parameters", self)
        params_layout = QtWidgets.QFormLayout(params_group)

        self._seed_spin = QtWidgets.QSpinBox(params_group)
        self._seed_spin.setRange(1, 1_000_000)
        self._seed_spin.setValue(42)
        params_layout.addRow("Seed:", self._seed_spin)

        self._threads_spin = QtWidgets.QSpinBox(params_group)
        self._threads_spin.setRange(1, 256)
        self._threads_spin.setValue(64)
        params_layout.addRow("Parallel Envs:", self._threads_spin)

        self._episodes_spin = QtWidgets.QSpinBox(params_group)
        self._episodes_spin.setRange(1000, 1_000_000)
        self._episodes_spin.setValue(200_000)
        self._episodes_spin.setSingleStep(10000)
        params_layout.addRow("Max Episodes:", self._episodes_spin)

        self._device_combo = QtWidgets.QComboBox(params_group)
        self._device_combo.addItems(["cuda", "cpu", "cuda:0", "cuda:1"])
        params_layout.addRow("Device:", self._device_combo)

        layout.addWidget(params_group)

        # Visualization Group
        viz_group = QtWidgets.QGroupBox("Visualization", self)
        viz_layout = QtWidgets.QFormLayout(viz_group)

        self._draw_mode_combo = QtWidgets.QComboBox(viz_group)
        for mode in UnrealMAPDrawMode:
            self._draw_mode_combo.addItem(mode.value, mode)
        viz_layout.addRow("Draw Mode:", self._draw_mode_combo)

        layout.addWidget(viz_group)

        # Control Buttons
        buttons_layout = QtWidgets.QHBoxLayout()

        self._launch_button = QtWidgets.QPushButton("Launch Training", self)
        self._launch_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; "
            "background-color: #1976d2; color: white; }"
            "QPushButton:hover { background-color: #1565c0; }"
        )
        buttons_layout.addWidget(self._launch_button)

        self._stop_button = QtWidgets.QPushButton("Stop All", self)
        self._stop_button.setEnabled(False)
        buttons_layout.addWidget(self._stop_button)

        layout.addLayout(buttons_layout)

        # Status
        self._status_label = QtWidgets.QLabel("No training running", self)
        self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_label)

        layout.addStretch(1)

        # Initialize team widgets
        self._update_team_widgets()

    def _connect_signals(self) -> None:
        self._num_teams_spin.valueChanged.connect(self._update_team_widgets)
        self._launch_button.clicked.connect(self._on_launch_clicked)
        self._stop_button.clicked.connect(self._on_stop_clicked)

    def _update_team_widgets(self) -> None:
        """Update team configuration widgets based on number of teams."""
        # Clear existing
        for widget_dict in self._team_widgets:
            for w in widget_dict.values():
                if isinstance(w, QtWidgets.QWidget):
                    w.deleteLater()
        self._team_widgets.clear()

        # Clear layout
        while self._team_container_layout.count():
            item = self._team_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new
        num_teams = self._num_teams_spin.value()
        for i in range(num_teams):
            team_frame = QtWidgets.QFrame(self._team_container)
            team_frame.setFrameStyle(QtWidgets.QFrame.Shape.StyledPanel)
            frame_layout = QtWidgets.QFormLayout(team_frame)

            label = QtWidgets.QLabel(f"<b>Team {i + 1}</b>")
            frame_layout.addRow(label)

            algo_combo = QtWidgets.QComboBox(team_frame)
            for algo in UnrealMAPAlgorithm:
                display = algo.value.replace("_", " ").title()
                algo_combo.addItem(display, algo)
            frame_layout.addRow("Algorithm:", algo_combo)

            agents_spin = QtWidgets.QSpinBox(team_frame)
            agents_spin.setRange(1, 100)
            agents_spin.setValue(10)
            frame_layout.addRow("Agents:", agents_spin)

            self._team_container_layout.addWidget(team_frame)
            self._team_widgets.append({
                "frame": team_frame,
                "algo": algo_combo,
                "agents": agents_spin,
            })

    def _on_launch_clicked(self) -> None:
        config = self._build_config()
        self.launch_training_requested.emit(config)
        self._stop_button.setEnabled(True)
        self._status_label.setText("Training started...")

    def _on_stop_clicked(self) -> None:
        self.stop_training_requested.emit()
        self._stop_button.setEnabled(False)
        self._status_label.setText("No training running")

    def _build_config(self) -> dict:
        """Build configuration dictionary from UI state."""
        teams = []
        for i, widget_dict in enumerate(self._team_widgets):
            teams.append({
                "team_id": i,
                "algorithm": widget_dict["algo"].currentData().value,
                "n_agents": widget_dict["agents"].value(),
            })

        return {
            "task_id": self._task_combo.currentData().value,
            "teams": teams,
            "seed": self._seed_spin.value(),
            "num_threads": self._threads_spin.value(),
            "max_n_episode": self._episodes_spin.value(),
            "device": self._device_combo.currentText(),
            "draw_mode": self._draw_mode_combo.currentData().value,
        }

    def update_status(self, running_count: int) -> None:
        """Update status display."""
        if running_count > 0:
            self._status_label.setText(f"{running_count} training instance(s) running")
            self._stop_button.setEnabled(True)
        else:
            self._status_label.setText("No training running")
            self._stop_button.setEnabled(False)
```

### 3.4 pyproject.toml

**`3rd_party/unreal_map_worker/pyproject.toml`**:

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mosaic-unreal-map"
version = "0.1.0"
description = "Unreal-MAP integration for MOSAIC BDI-RL framework"
requires-python = ">=3.10"
license = {text = "MIT"}
# Dependencies are managed via the main project's requirements.txt
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[project.scripts]
unreal-map-worker = "unreal_map_worker.cli:main"

[tool.setuptools]
packages = [
    # Our integration wrapper
    "unreal_map_worker",
]

[tool.setuptools.package-dir]
# Our wrapper is directly in unreal_map_worker/
"unreal_map_worker" = "unreal_map_worker"
```

### 3.5 Requirements File

**`requirements/unreal_map_worker.txt`**:

```txt
# Unreal-MAP Worker Dependencies
# Install this when using Unreal-MAP for multi-agent training.

# Include base requirements
-r base.txt

# Core dependencies for Unreal-MAP training
torch>=2.0.0
numpy>=1.24.0
cython>=3.0.0  # For shm_pool.pyx compilation
pyximport>=0.3.0

# For visualization (optional)
tensorboard>=2.11.0
matplotlib>=3.7.0

# Note: The Unreal Engine client binaries must be built separately
# See: 3rd_party/unreal_map_worker/unreal-map/README.md
```

---

## 4. Implementation Phases

### Phase 1: Worker Package Setup
1. Create `3rd_party/unreal_map_worker/unreal_map_worker/` directory structure
2. Create `__init__.py` with package exports
3. Create `enums.py` with task and algorithm enums
4. Create `config.py` with configuration dataclasses
5. Create `pyproject.toml`
6. Create `requirements/unreal_map_worker.txt`
7. Test basic import: `from unreal_map_worker import get_launcher`

### Phase 2: Launcher Implementation
1. Create `launcher.py` with `UnrealMAPLauncher` class
2. Implement process management (launch, terminate, list)
3. Handle configuration file generation for training
4. Test launching training from command line

### Phase 3: GUI Enums
1. Create `gym_gui/core/unreal_map_enums.py`
2. Keep separate from `enums.py` (different domain)

### Phase 4: GUI Tab Widget
1. Create `gym_gui/ui/widgets/unreal_map_tab.py`
2. Implement task selection
3. Implement team configuration (dynamic based on num_teams)
4. Implement training parameters
5. Add launch/stop buttons

### Phase 5: Control Panel Integration
1. Replace placeholder in `control_panel.py` Multi-Agent tab
2. Connect signals to main window handlers
3. Add `UnrealMAPTab` to sidebar

### Phase 6: Main Window Handlers
1. Add handlers for `launch_training_requested`
2. Add handlers for `stop_training_requested`
3. Implement Render View tab for training visualization

### Phase 7: Telemetry Integration
1. Create `telemetry.py` adapter
2. Stream training metrics (reward, win rate, etc.)
3. Display in GUI

---

## 5. Key Differences from MuJoCo MPC

| Aspect | MuJoCo MPC | Unreal-MAP |
|--------|-----------|------------|
| **Purpose** | Real-time optimal control visualization | Multi-agent RL training |
| **Process Type** | Single MJPC GUI or agent_server | Training process with parallel envs |
| **Display** | External window or embedded render | Training metrics + optional UE render |
| **Configuration** | Task + planner selection | Task + multi-team + algorithm per team |
| **Output** | Real-time control | Trained policies + telemetry |
| **Binary Dependency** | MJPC build required | UE4 client build (optional) |

---

## 6. Open Questions

1. **UE Client Binary**: How to handle the Unreal Engine client?
   - Option A: External window mode only (user runs UE separately)
   - Option B: Launch UE client as subprocess
   - Option C: Headless training only (no visualization)

2. **Cross-Platform Rendering**: Unreal-MAP supports training on Linux and rendering on Windows. Should we support this?

3. **Telemetry Format**: Should we unify with existing SQLite telemetry or use separate format?

4. **Policy Loading**: How to integrate trained policies back for evaluation?

---

## 7. Files to Create

| File | Purpose |
|------|---------|
| `3rd_party/unreal_map_worker/pyproject.toml` | Package definition |
| `3rd_party/unreal_map_worker/unreal_map_worker/__init__.py` | Package exports |
| `3rd_party/unreal_map_worker/unreal_map_worker/enums.py` | Task and algorithm enums |
| `3rd_party/unreal_map_worker/unreal_map_worker/config.py` | Configuration dataclasses |
| `3rd_party/unreal_map_worker/unreal_map_worker/launcher.py` | Process launcher |
| `3rd_party/unreal_map_worker/unreal_map_worker/cli.py` | CLI entry point |
| `requirements/unreal_map_worker.txt` | Dependencies |
| `gym_gui/core/unreal_map_enums.py` | GUI-side enums |
| `gym_gui/ui/widgets/unreal_map_tab.py` | Multi-Agent tab widget |

---

## 8. Summary

This integration follows the established worker pattern:
- **Vendored code** (`unreal-map/`) stays untouched
- **Our wrapper** (`unreal_map_worker/`) provides the integration layer
- **GUI tab** replaces the placeholder in Multi-Agent Mode
- **Separate enums** keep the multi-agent domain clean

The key insight is that Unreal-MAP is a **training framework** (like CleanRL), not a real-time controller (like MuJoCo MPC). The GUI will primarily:
1. Configure training parameters
2. Launch/stop training processes
3. Display training progress and metrics
4. Optionally integrate with UE visualization
