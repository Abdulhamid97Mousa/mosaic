# MuJoCo MPC Integration - Complete Documentation

**Date:** 2025-11-26
**Status:** Complete (External Mode) / Planned (Embedded Mode)
**Author:** Claude Code Assistant

---

## Overview

This document describes the integration of MuJoCo MPC (MJPC) into the Gym GUI application. MJPC is a real-time Model Predictive Control framework for MuJoCo physics simulation, developed by DeepMind.

The integration provides:
- A dedicated **MuJoCo MPC tab** in the Control Panel sidebar
- **Two display modes**: External Window and Embedded (coming soon)
- **Multi-instance support**: Launch multiple MJPC instances simultaneously
- **Process management**: Start/stop individual instances or all at once

---

## Architecture

### Directory Structure

```
3rd_party/mujoco_mpc_worker/
├── mujoco_mpc/                    # Vendored MuJoCo MPC repository (DO NOT MODIFY)
│   ├── build/                     # Build output directory
│   │   └── bin/
│   │       ├── mjpc              # Main MJPC GUI binary
│   │       ├── agent_server      # gRPC server for headless MPC
│   │       └── ...
│   ├── mjpc/
│   │   └── grpc/
│   │       └── agent.proto       # gRPC service definition
│   └── python/
│       └── mujoco_mpc/
│           └── agent.py          # Python gRPC client
│
└── mujoco_mpc_worker/             # Our integration wrapper
    ├── __init__.py               # Package exports
    ├── config.py                 # Configuration dataclasses
    ├── enums.py                  # Task and planner enums
    ├── launcher.py               # Process launcher for MJPC
    └── CMakeLists.txt            # Build configuration (references vendored source)
```

### Key Components

#### 1. Launcher (`launcher.py`)

The `MJPCLauncher` class manages MJPC process lifecycle:

```python
from mujoco_mpc_worker import get_launcher

launcher = get_launcher()

# Check if MJPC is built
if launcher.is_built():
    # Launch a new instance
    process, message = launcher.launch()
    print(f"Launched MJPC with PID: {process.process.pid}")

    # Later, terminate it
    launcher.terminate(process.instance_id)
```

**Key Methods:**
- `is_built()` - Check if MJPC binaries exist
- `get_build_status()` - Get detailed build information
- `launch(task_id=None)` - Launch new MJPC instance
- `terminate(instance_id)` - Stop specific instance
- `terminate_all()` - Stop all instances
- `list_running()` - Get list of running processes

#### 2. GUI Tab (`mujoco_mpc_tab.py`)

The sidebar tab provides:
- Display mode selector (External Window / Embedded)
- Launch MJPC button
- Stop All Instances button
- Running instance count display

```python
from gym_gui.ui.widgets.mujoco_mpc_tab import MuJoCoMPCTab, MJPCDisplayMode

# Signals emitted:
# - launch_mpc_requested(str)  # display_mode: "external" or "embedded"
# - stop_all_requested()
```

#### 3. Main Window Handlers (`main_window.py`)

The MainWindow connects sidebar signals to launcher actions:

```python
# Signal connections
self._control_panel.mpc_launch_requested.connect(self._on_mpc_launch_requested)
self._control_panel.mpc_stop_all_requested.connect(self._on_mpc_stop_all_requested)

# Handler creates appropriate tab based on display mode
def _on_mpc_launch_requested(self, display_mode: str):
    if display_mode == "embedded":
        self._create_embedded_mpc_tab(...)  # Coming soon
    else:
        self._create_external_mpc_tab(...)  # Working
```

---

## Display Modes

### External Window Mode (Implemented)

In this mode, MJPC launches as a completely separate window outside of Gym GUI.

**Behavior:**
1. User clicks "Launch MJPC" with "External Window" selected
2. MJPC GUI opens as a standalone window
3. A status tab is created in Render View showing PID and stop button
4. User interacts with MJPC in its native window
5. Closing the tab or clicking "Stop" terminates the MJPC process

**Advantages:**
- Full MJPC functionality available
- Native OpenGL rendering performance
- No embedding issues

**Limitations:**
- Window is separate from Gym GUI
- Cannot be docked/arranged within the main application

### Embedded Mode (Planned - Coming Soon)

In this mode, MJPC rendering will appear directly inside the Gym GUI Render View tab.

**Planned Implementation:**
1. Launch `agent_server` (headless gRPC service) instead of `mjpc` GUI
2. Use MuJoCo Python bindings for rendering in a Qt widget
3. Communicate with agent_server via gRPC for MPC computation
4. Display rendered frames in the Render View tab

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│                   Gym GUI                        │
│  ┌─────────────────────────────────────────────┐│
│  │            Render View Tab                   ││
│  │  ┌───────────────────────────────────────┐  ││
│  │  │     MuJoCo Python Renderer Widget     │  ││
│  │  │  (renders mujoco.MjData to QImage)    │  ││
│  │  └───────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
              │ gRPC (localhost:PORT)
              ▼
┌─────────────────────────────────────────────────┐
│              agent_server (C++)                  │
│  - Runs MPC planning loop                       │
│  - Provides actions via GetAction RPC           │
│  - Accepts state updates via SetState RPC       │
└─────────────────────────────────────────────────┘
```

**Required for Implementation:**
1. Install `mujoco_mpc` Python package with gRPC stubs
2. Create Qt widget for MuJoCo rendering
3. Implement simulation loop with agent_server communication

---

## Building MJPC

### Prerequisites

```bash
# Build tools
sudo apt install cmake ninja-build

# For gRPC support (optional but recommended)
# gRPC will be fetched automatically during build
```

### Build Steps

```bash
cd 3rd_party/mujoco_mpc_worker/mujoco_mpc

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja -DMJPC_BUILD_GRPC_SERVICE=ON

# Build (this takes 10-30 minutes due to gRPC)
ninja -j$(nproc)
```

### Verify Build

```bash
# Check binaries exist
ls -la build/bin/mjpc build/bin/agent_server

# Test MJPC launches
./build/bin/mjpc
```

---

## Usage Guide

### Launching MJPC from GUI

1. Start the application:
   ```bash
   bash run.sh
   ```

2. Navigate to **MuJoCo MPC** tab in the Control Panel sidebar

3. Select display mode:
   - **External Window** - Opens MJPC as separate window (recommended)
   - **Embedded in Gym GUI** - Coming soon

4. Click **Launch MJPC**

5. A new tab appears in Render View:
   - For External mode: Shows status and stop button
   - For Embedded mode: Shows "coming soon" message

6. Multiple instances can be launched (MuJoCo-MPC-1, MuJoCo-MPC-2, etc.)

7. Click **Stop All Instances** to terminate all MJPC processes

### MJPC GUI Features

Once launched, the MJPC GUI provides:

- **Task Selection**: Choose from various MPC tasks (Cartpole, Quadruped, Humanoid, etc.)
- **Planner Selection**: iLQG, Gradient Descent, Predictive Sampling, etc.
- **Real-time Visualization**: 3D rendering of simulation
- **Cost Visualization**: Graphs showing objective function components
- **Parameter Tuning**: Adjust weights, gains, and other parameters

---

## API Reference

### MJPCLauncher

```python
class MJPCLauncher:
    """Launcher for MuJoCo MPC GUI instances."""

    def is_built(self) -> bool:
        """Check if MJPC binaries are built."""

    def get_build_status(self) -> dict:
        """Get detailed build status information."""

    def build(self, num_jobs: int = None) -> tuple[bool, str]:
        """Build MJPC from source."""

    def launch(self, task_id: str = None) -> tuple[MJPCProcess | None, str]:
        """Launch a new MJPC GUI instance."""

    def terminate(self, instance_id: int) -> bool:
        """Terminate a specific MJPC instance."""

    def terminate_all(self) -> int:
        """Terminate all MJPC instances. Returns count terminated."""

    def list_running(self) -> list[MJPCProcess]:
        """List only running processes."""
```

### MJPCProcess

```python
@dataclass
class MJPCProcess:
    """Represents a running MJPC process."""

    instance_id: int
    process: subprocess.Popen
    task_id: Optional[str] = None
    port: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""

    def terminate(self) -> None:
        """Terminate the process."""
```

### MJPCDisplayMode

```python
class MJPCDisplayMode(Enum):
    """Display mode for MJPC GUI."""
    EXTERNAL = "external"  # Separate popup window
    EMBEDDED = "embedded"  # Embedded in Render View tab
```

---

## Files Modified/Created

### New Files

| File | Description |
|------|-------------|
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/__init__.py` | Package exports |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/config.py` | Configuration dataclasses |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/enums.py` | Task and planner enums |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/launcher.py` | Process launcher |
| `3rd_party/mujoco_mpc_worker/mujoco_mpc_worker/CMakeLists.txt` | Build configuration |
| `gym_gui/ui/widgets/mujoco_mpc_tab.py` | Sidebar tab widget |
| `gym_gui/core/mujoco_mpc_enums.py` | GUI-side enums |

### Modified Files

| File | Changes |
|------|---------|
| `gym_gui/ui/widgets/control_panel.py` | Added MuJoCo MPC tab and signals |
| `gym_gui/ui/main_window.py` | Added MPC launch/stop handlers |

---

## Future Enhancements

### Embedded Mode Implementation

1. **Install mujoco_mpc Python package**
   ```bash
   cd 3rd_party/mujoco_mpc_worker/mujoco_mpc/python
   pip install -e .
   ```

2. **Create MuJoCo Qt Renderer Widget**
   - Use `mujoco.Renderer` for offscreen rendering
   - Convert rendered frames to QImage
   - Display in QLabel or custom OpenGL widget

3. **Implement gRPC Client Integration**
   - Connect to agent_server
   - Send state updates
   - Receive actions
   - Run simulation loop

### Additional Features

- [ ] Task preset selector in sidebar
- [ ] Cost weight adjustment UI
- [ ] Trajectory visualization overlay
- [ ] Recording/playback functionality
- [ ] Integration with training pipelines

---

## Troubleshooting

### MJPC Not Built Error

**Symptom:** "MuJoCo MPC needs to be built first" dialog appears

**Solution:**
```bash
cd 3rd_party/mujoco_mpc_worker/mujoco_mpc/build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -j$(nproc)
```

### Build Fails with libstdc++ Error

**Symptom:** `cannot find -lstdc++: No such file or directory`

**Solution:**
```bash
sudo apt install libstdc++-12-dev
```

### MJPC Window Doesn't Appear

**Symptom:** Launch button clicked but no window appears

**Check:**
1. Verify build completed: `ls 3rd_party/mujoco_mpc_worker/mujoco_mpc/build/bin/mjpc`
2. Check process is running: `ps aux | grep mjpc`
3. Check for OpenGL errors in terminal output

### gRPC Build Takes Too Long

**Note:** gRPC is a large dependency and initial download/build can take 10-30 minutes. This is expected.

---

## References

- [MuJoCo MPC GitHub](https://github.com/google-deepmind/mujoco_mpc)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MJPC Paper](https://arxiv.org/abs/2212.00541)
- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
