# AirSim Worker Integration Plan (Multi-Agent Drones/Rovers)

**Date:** 2025-11-27
**Status:** Planning
**Goal:** Integrate Project AirSim as a multi-agent worker for drone/rover swarm simulation in Mosaic/gym_gui

---

## 1. Overview

### What is Project AirSim?

[Project AirSim](https://github.com/iamaisim/ProjectAirSim) is a simulation platform for drones, robots, and autonomous systems:

- **Successor to Microsoft AirSim** - Maintained by IAMAI Simulations (former Microsoft engineers)
- **Unreal Engine 5** - Photo-realistic visuals and physics
- **Multi-Robot Support** - Multiple drones/rovers in same simulation
- **Rich Sensor Suite** - Cameras (RGB, Depth, Segmentation), LiDAR, Radar, IMU, GPS, Barometer
- **Gymnasium Integration** - Built-in `gym.Env` support for RL training
- **Python API** - Async client for robot control and sensor data

### Multi-Agent Capabilities

Project AirSim natively supports multiple robots in the same simulation:

```json
"actors": [
  {
    "type": "robot",
    "name": "Drone1",
    "origin": { "xyz": "0.0 0.0 -10.0", "rpy-deg": "0 0 0" },
    "ref": "robot_quadrotor_fastphysics.jsonc"
  },
  {
    "type": "robot",
    "name": "Drone2",
    "origin": { "xyz": "0.0 5.0 -10.0", "rpy-deg": "0 0 0" },
    "ref": "robot_quadrotor_fastphysics.jsonc"
  }
]
```

### Use Cases in Mosaic/gym_gui

| Use Case | Description |
|----------|-------------|
| **Drone Swarm RL** | Train multi-drone coordination policies |
| **Search & Rescue** | Cooperative exploration and target detection |
| **Formation Flight** | Maintain formations while navigating |
| **Adversarial Drones** | Pursuit-evasion scenarios |
| **Rover Coordination** | Ground-based multi-robot tasks |
| **Heterogeneous Teams** | Mixed drone + rover agents |

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| OS | Ubuntu 22.04 / Windows 11 |
| GPU | NVIDIA with Vulkan support |
| RAM | 16 GB minimum (32 GB recommended) |
| Disk | ~30 GB for UE5 + environments |
| Python | 3.7-3.9 (64-bit) |

---

## 2. Repository Structure

### Current Layout

```
3rd_party/airsim_worker/
├── ProjectAirSim/                    # Vendored repo (cloned)
│   ├── client/
│   │   ├── python/
│   │   │   ├── projectairsim/        # Python client package
│   │   │   │   ├── src/projectairsim/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── client.py     # ProjectAirSimClient
│   │   │   │   │   ├── drone.py      # Drone class
│   │   │   │   │   ├── rover.py      # Rover class
│   │   │   │   │   ├── world.py      # World class
│   │   │   │   │   ├── gym_envs/     # Gymnasium environments
│   │   │   │   │   └── autonomy/     # Autonomy modules
│   │   │   │   ├── pyproject.toml
│   │   │   │   └── requirements.txt
│   │   │   └── example_user_scripts/ # Demo scripts
│   │   └── cpp/                      # C++ client (not used)
│   ├── core_sim/                     # Simulation core (C++)
│   ├── physics/                      # Physics engines
│   ├── rendering/                    # Unreal Plugin
│   ├── docs/                         # Documentation
│   ├── Linux_Unreal_Engine_5.5.0.zip # UE5 installation
│   ├── build.sh                      # Build script
│   └── README.md
│
└── airsim_worker/                    # Our wrapper code (TO CREATE)
    ├── __init__.py
    ├── client.py                     # AirSim client wrapper
    ├── config.py                     # Configuration dataclasses
    ├── enums.py                      # Robot types, sensors, etc.
    ├── multi_agent_env.py            # Multi-agent Gymnasium env
    ├── launcher.py                   # UE5 server launcher
    └── cli.py                        # CLI entry point
```

---

## 3. Architecture

### 3.1 Overall Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Mosaic GUI                              │
├─────────────────────────────────────────────────────────────────┤
│  Human Control │ Single-Agent │ Multi-Agent │ AirSim Tab (NEW) │
│       Tab      │     Tab      │     Tab     │                  │
└────────────────┴──────────────┴─────────────┴──────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AirSim Worker                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ AirSimClient    │  │ MultiAgentEnv   │  │ ServerLauncher │  │
│  │ (API wrapper)   │  │ (Gymnasium)     │  │ (UE5 process)  │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Project AirSim Server (UE5)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Drones: Drone1, Drone2, ...  │  Rovers: Rover1, ...    │   │
│  │  Sensors: Camera, LiDAR, IMU  │  Physics: FastPhysics   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  TCP/UDP: localhost:41451 (services), localhost:41452 (topics)  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Communication Protocol

Project AirSim uses **nanomsg-next-gen (nng)** for communication:

```
┌──────────────┐                    ┌──────────────┐
│   Python     │  Request/Response  │   UE5        │
│   Client     │ ◄────────────────► │   Server     │
│              │   Port 41451       │              │
│              │                    │              │
│              │  Pub/Sub (Topics)  │              │
│              │ ◄────────────────► │              │
│              │   Port 41452       │              │
└──────────────┘                    └──────────────┘
```

- **Services (41451)**: Request/response for commands (takeoff, move, etc.)
- **Topics (41452)**: Pub/sub for sensor data (camera images, IMU, etc.)

---

## 4. Component Design

### 4.1 Enums (`airsim_worker/enums.py`)

```python
"""Enums for AirSim worker."""

from enum import Enum, auto


class RobotType(Enum):
    """Supported robot types."""
    QUADROTOR = "quadrotor"
    ROVER = "rover"


class PhysicsType(Enum):
    """Physics engine types."""
    FAST_PHYSICS = "fastphysics"
    EXTERNAL = "external"


class CameraType(Enum):
    """Camera sensor types."""
    SCENE = "scene_camera"
    DEPTH = "depth_camera"
    SEGMENTATION = "segmentation_camera"
    INFRARED = "infrared_camera"


class SensorType(Enum):
    """Available sensor types."""
    CAMERA = auto()
    LIDAR = auto()
    RADAR = auto()
    IMU = auto()
    GPS = auto()
    BAROMETER = auto()
    MAGNETOMETER = auto()
    AIRSPEED = auto()
    BATTERY = auto()


class DroneCommand(Enum):
    """Drone control commands."""
    TAKEOFF = auto()
    LAND = auto()
    HOVER = auto()
    GO_HOME = auto()
    MOVE_BY_VELOCITY = auto()
    MOVE_TO_POSITION = auto()
    MOVE_ON_PATH = auto()
    ROTATE_TO_YAW = auto()


class SimClockType(Enum):
    """Simulation clock types."""
    REAL_TIME = "real-time"
    STEPPABLE = "steppable"
```

### 4.2 Configuration (`airsim_worker/config.py`)

```python
"""Configuration for AirSim worker."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .enums import RobotType, PhysicsType


@dataclass
class RobotConfig:
    """Configuration for a single robot."""
    name: str
    robot_type: RobotType = RobotType.QUADROTOR
    physics_type: PhysicsType = PhysicsType.FAST_PHYSICS
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, -10.0)
    origin_rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    config_ref: str = "robot_quadrotor_fastphysics.jsonc"


@dataclass
class ServerConfig:
    """Configuration for AirSim server connection."""
    host: str = "localhost"
    port_services: int = 41451
    port_topics: int = 41452
    timeout: float = 30.0


@dataclass
class SceneConfig:
    """Configuration for simulation scene."""
    scene_name: str = "scene_basic_drone.jsonc"
    delay_after_load_sec: float = 2.0
    clock_type: str = "steppable"  # or "real-time"
    time_step_ns: int = 10_000_000  # 10ms = 100Hz


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent simulation."""
    server: ServerConfig = field(default_factory=ServerConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    robots: List[RobotConfig] = field(default_factory=list)

    # Environment settings
    max_episode_steps: int = 1000
    render_mode: Optional[str] = "rgb_array"

    @classmethod
    def create_drone_swarm(
        cls,
        num_drones: int,
        spacing: float = 5.0,
        altitude: float = -10.0,
    ) -> "MultiAgentConfig":
        """Create config for a drone swarm."""
        robots = []
        for i in range(num_drones):
            robots.append(RobotConfig(
                name=f"Drone{i+1}",
                robot_type=RobotType.QUADROTOR,
                origin_xyz=(0.0, i * spacing, altitude),
            ))
        return cls(robots=robots)
```

### 4.3 Client Wrapper (`airsim_worker/client.py`)

```python
"""AirSim client wrapper for multi-agent control."""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np

# Import from vendored projectairsim
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ProjectAirSim" / "client" / "python" / "projectairsim" / "src"))

from projectairsim import ProjectAirSimClient, World, Drone
from projectairsim.rover import Rover

from .config import MultiAgentConfig, RobotConfig
from .enums import RobotType, CameraType


@dataclass
class RobotState:
    """State of a single robot."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [roll, pitch, yaw]
    velocity: np.ndarray  # [vx, vy, vz]
    is_armed: bool = False
    is_api_control: bool = False


class AirSimMultiAgentClient:
    """Client for controlling multiple robots in AirSim."""

    def __init__(self, config: MultiAgentConfig):
        self._config = config
        self._client: Optional[ProjectAirSimClient] = None
        self._world: Optional[World] = None
        self._robots: Dict[str, Drone | Rover] = {}
        self._robot_states: Dict[str, RobotState] = {}
        self._subscriptions: Dict[str, List] = {}

    def connect(self) -> bool:
        """Connect to AirSim server."""
        try:
            self._client = ProjectAirSimClient(
                address=self._config.server.host,
                port_services=self._config.server.port_services,
                port_topics=self._config.server.port_topics,
            )
            self._client.connect()
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from AirSim server."""
        if self._client:
            self._client.disconnect()
            self._client = None

    def load_scene(self) -> bool:
        """Load the simulation scene with configured robots."""
        if not self._client:
            return False

        try:
            # Create world with scene config
            self._world = World(
                self._client,
                self._config.scene.scene_name,
                delay_after_load_sec=self._config.scene.delay_after_load_sec,
            )

            # Create robot objects for each configured robot
            for robot_config in self._config.robots:
                if robot_config.robot_type == RobotType.QUADROTOR:
                    robot = Drone(self._client, self._world, robot_config.name)
                else:
                    robot = Rover(self._client, self._world, robot_config.name)

                self._robots[robot_config.name] = robot
                self._robot_states[robot_config.name] = RobotState(
                    position=np.array(robot_config.origin_xyz),
                    orientation=np.zeros(3),
                    velocity=np.zeros(3),
                )

            return True
        except Exception as e:
            print(f"Failed to load scene: {e}")
            return False

    def get_robot(self, name: str) -> Optional[Drone | Rover]:
        """Get robot by name."""
        return self._robots.get(name)

    def get_all_robots(self) -> Dict[str, Drone | Rover]:
        """Get all robots."""
        return self._robots

    def get_robot_names(self) -> List[str]:
        """Get list of robot names."""
        return list(self._robots.keys())

    # ═══════════════════════════════════════════════════════════════
    # Control API
    # ═══════════════════════════════════════════════════════════════

    def enable_api_control(self, robot_name: str) -> bool:
        """Enable API control for a robot."""
        robot = self._robots.get(robot_name)
        if robot:
            robot.enable_api_control()
            self._robot_states[robot_name].is_api_control = True
            return True
        return False

    def enable_all_api_control(self):
        """Enable API control for all robots."""
        for name in self._robots:
            self.enable_api_control(name)

    def arm(self, robot_name: str) -> bool:
        """Arm a robot."""
        robot = self._robots.get(robot_name)
        if robot and isinstance(robot, Drone):
            robot.arm()
            self._robot_states[robot_name].is_armed = True
            return True
        return False

    def arm_all(self):
        """Arm all drones."""
        for name, robot in self._robots.items():
            if isinstance(robot, Drone):
                self.arm(name)

    async def takeoff_async(self, robot_name: str, timeout_sec: float = 20.0):
        """Takeoff a drone asynchronously."""
        robot = self._robots.get(robot_name)
        if robot and isinstance(robot, Drone):
            task = await robot.takeoff_async(timeout_sec=timeout_sec)
            await task

    async def takeoff_all_async(self, timeout_sec: float = 20.0):
        """Takeoff all drones concurrently."""
        tasks = []
        for name, robot in self._robots.items():
            if isinstance(robot, Drone):
                tasks.append(self.takeoff_async(name, timeout_sec))
        await asyncio.gather(*tasks)

    async def move_by_velocity_async(
        self,
        robot_name: str,
        v_north: float,
        v_east: float,
        v_down: float,
        duration: float,
    ):
        """Move robot by velocity."""
        robot = self._robots.get(robot_name)
        if robot and isinstance(robot, Drone):
            task = await robot.move_by_velocity_async(
                v_north=v_north,
                v_east=v_east,
                v_down=v_down,
                duration=duration,
            )
            await task

    async def move_to_position_async(
        self,
        robot_name: str,
        north: float,
        east: float,
        down: float,
        velocity: float = 5.0,
        timeout_sec: float = 60.0,
    ):
        """Move robot to position."""
        robot = self._robots.get(robot_name)
        if robot and isinstance(robot, Drone):
            task = await robot.move_to_position_async(
                north=north,
                east=east,
                down=down,
                velocity=velocity,
                timeout_sec=timeout_sec,
            )
            await task

    # ═══════════════════════════════════════════════════════════════
    # Sensor API
    # ═══════════════════════════════════════════════════════════════

    def subscribe_camera(
        self,
        robot_name: str,
        camera_name: str,
        camera_type: CameraType,
        callback: Callable,
    ):
        """Subscribe to camera sensor."""
        robot = self._robots.get(robot_name)
        if robot:
            topic = robot.sensors[camera_name][camera_type.value]
            self._client.subscribe(topic, callback)

            if robot_name not in self._subscriptions:
                self._subscriptions[robot_name] = []
            self._subscriptions[robot_name].append(topic)

    def get_ground_truth_pose(self, robot_name: str) -> Optional[Dict]:
        """Get ground truth pose of robot."""
        robot = self._robots.get(robot_name)
        if robot:
            return robot.get_ground_truth_pose()
        return None

    def get_ground_truth_kinematics(self, robot_name: str) -> Optional[Dict]:
        """Get ground truth kinematics of robot."""
        robot = self._robots.get(robot_name)
        if robot:
            return robot.get_ground_truth_kinematics()
        return None

    def get_imu_data(self, robot_name: str, sensor_name: str = "imu") -> Optional[Dict]:
        """Get IMU sensor data."""
        robot = self._robots.get(robot_name)
        if robot:
            return robot.get_imu_data(sensor_name)
        return None

    def get_gps_data(self, robot_name: str, sensor_name: str = "gps") -> Optional[Dict]:
        """Get GPS sensor data."""
        robot = self._robots.get(robot_name)
        if robot:
            return robot.get_gps_data(sensor_name)
        return None

    # ═══════════════════════════════════════════════════════════════
    # Simulation Control
    # ═══════════════════════════════════════════════════════════════

    def pause(self):
        """Pause simulation."""
        if self._world:
            self._world.pause()

    def resume(self):
        """Resume simulation."""
        if self._world:
            self._world.resume()

    def step(self, time_step_ns: Optional[int] = None):
        """Step simulation by time delta."""
        if self._world:
            ns = time_step_ns or self._config.scene.time_step_ns
            self._world.continue_for_sim_time(ns, wait_until_complete=True)

    def reset(self):
        """Reset simulation by reloading scene."""
        self.load_scene()
```

### 4.4 Multi-Agent Gymnasium Environment (`airsim_worker/multi_agent_env.py`)

```python
"""Multi-agent Gymnasium environment for AirSim."""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .client import AirSimMultiAgentClient
from .config import MultiAgentConfig


class AirSimMultiAgentEnv(gym.Env):
    """
    Multi-agent Gymnasium environment for drone swarms.

    Follows PettingZoo-like API for multi-agent RL.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        config: MultiAgentConfig,
        render_mode: Optional[str] = "rgb_array",
    ):
        super().__init__()
        self._config = config
        self.render_mode = render_mode

        # Create client
        self._client = AirSimMultiAgentClient(config)

        # Agent info
        self.agents = [r.name for r in config.robots]
        self.num_agents = len(self.agents)

        # Observation space: [x, y, z, vx, vy, vz, roll, pitch, yaw] per agent
        obs_dim = 9
        self.observation_space = spaces.Dict({
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for agent in self.agents
        })

        # Action space: [v_north, v_east, v_down] per agent
        action_dim = 3
        self.action_space = spaces.Dict({
            agent: spaces.Box(
                low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32
            )
            for agent in self.agents
        })

        # State
        self._step_count = 0
        self._connected = False
        self._loop = asyncio.new_event_loop()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Connect if not connected
        if not self._connected:
            self._client.connect()
            self._connected = True

        # Load/reload scene
        self._client.load_scene()

        # Enable control and arm all drones
        self._client.enable_all_api_control()
        self._client.arm_all()

        # Takeoff all drones
        self._loop.run_until_complete(self._client.takeoff_all_async())

        self._step_count = 0

        # Get initial observations
        obs = self._get_observations()
        info = {agent: {} for agent in self.agents}

        return obs, info

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminations
        Dict[str, bool],        # truncations
        Dict[str, Any],         # infos
    ]:
        """Execute actions for all agents."""
        # Apply actions to each robot
        async def apply_actions():
            tasks = []
            for agent, action in actions.items():
                tasks.append(
                    self._client.move_by_velocity_async(
                        robot_name=agent,
                        v_north=float(action[0]),
                        v_east=float(action[1]),
                        v_down=float(action[2]),
                        duration=0.1,  # 100ms
                    )
                )
            await asyncio.gather(*tasks)

        self._loop.run_until_complete(apply_actions())

        # Step simulation
        self._client.step()
        self._step_count += 1

        # Get observations
        obs = self._get_observations()

        # Calculate rewards (example: negative distance from origin)
        rewards = self._compute_rewards(obs)

        # Check terminations (example: collision or out of bounds)
        terminations = {agent: False for agent in self.agents}

        # Check truncations (max steps reached)
        truncated = self._step_count >= self._config.max_episode_steps
        truncations = {agent: truncated for agent in self.agents}

        # Info
        infos = {agent: {"step": self._step_count} for agent in self.agents}

        return obs, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "rgb_array":
            # Get camera image from first drone
            # TODO: Implement camera rendering
            return None
        return None

    def close(self):
        """Clean up resources."""
        self._client.disconnect()
        self._loop.close()

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        obs = {}
        for agent in self.agents:
            kinematics = self._client.get_ground_truth_kinematics(agent)
            if kinematics:
                pose = kinematics.get("pose", {})
                twist = kinematics.get("twist", {})

                position = pose.get("position", {})
                orientation = pose.get("orientation", {})
                linear = twist.get("linear", {})

                obs[agent] = np.array([
                    position.get("x", 0.0),
                    position.get("y", 0.0),
                    position.get("z", 0.0),
                    linear.get("x", 0.0),
                    linear.get("y", 0.0),
                    linear.get("z", 0.0),
                    orientation.get("x", 0.0),  # Simplified - should convert quaternion
                    orientation.get("y", 0.0),
                    orientation.get("z", 0.0),
                ], dtype=np.float32)
            else:
                obs[agent] = np.zeros(9, dtype=np.float32)

        return obs

    def _compute_rewards(self, obs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute rewards for all agents."""
        rewards = {}
        for agent, o in obs.items():
            # Example: negative L2 distance from hover position
            position = o[:3]
            target = np.array([0.0, 0.0, -10.0])  # Hover at altitude
            distance = np.linalg.norm(position - target)
            rewards[agent] = -distance
        return rewards
```

### 4.5 Server Launcher (`airsim_worker/launcher.py`)

```python
"""Launcher for AirSim UE5 server."""

import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional
import os


class AirSimServerLauncher:
    """Manages the AirSim UE5 server process."""

    def __init__(
        self,
        ue5_binary_path: Optional[Path] = None,
        render_offscreen: bool = False,
        disable_rendering: bool = False,
    ):
        self._binary_path = ue5_binary_path
        self._render_offscreen = render_offscreen
        self._disable_rendering = disable_rendering
        self._process: Optional[subprocess.Popen] = None

    def find_binary(self) -> Optional[Path]:
        """Find AirSim binary in common locations."""
        # Check environment variable
        env_path = os.environ.get("AIRSIM_BINARY")
        if env_path and Path(env_path).exists():
            return Path(env_path)

        # Check common locations
        search_paths = [
            Path.home() / "AirSim" / "Blocks" / "Binaries" / "Linux" / "Blocks.sh",
            Path.home() / "ProjectAirSim" / "Blocks.sh",
            Path("/opt/airsim/Blocks.sh"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def start(self, timeout: float = 30.0) -> bool:
        """Start the AirSim server."""
        binary = self._binary_path or self.find_binary()
        if not binary or not binary.exists():
            print(f"AirSim binary not found: {binary}")
            return False

        # Build command
        cmd = [str(binary)]

        if self._render_offscreen:
            cmd.append("-RenderOffScreen")

        if self._disable_rendering:
            cmd.append("-nullrhi")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=binary.parent,
            )

            # Wait for server to be ready
            time.sleep(timeout)

            if self._process.poll() is not None:
                print("AirSim server failed to start")
                return False

            return True
        except Exception as e:
            print(f"Failed to start AirSim server: {e}")
            return False

    def stop(self):
        """Stop the AirSim server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._process is not None and self._process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
```

---

## 5. GUI Integration

### 5.1 AirSim Tab Widget

```python
"""AirSim tab for control panel."""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal


class AirSimTab(QtWidgets.QWidget):
    """Tab widget for AirSim multi-agent control."""

    # Signals
    connect_requested = pyqtSignal(str, int)  # host, port
    disconnect_requested = pyqtSignal()
    start_simulation_requested = pyqtSignal(dict)  # config
    stop_simulation_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # ═══════════════════════════════════════════════════════════════
        # Connection Group
        # ═══════════════════════════════════════════════════════════════
        conn_group = QtWidgets.QGroupBox("Server Connection", self)
        conn_layout = QtWidgets.QFormLayout(conn_group)

        self._host_edit = QtWidgets.QLineEdit("localhost")
        conn_layout.addRow("Host:", self._host_edit)

        self._port_spin = QtWidgets.QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(41451)
        conn_layout.addRow("Port:", self._port_spin)

        self._status_label = QtWidgets.QLabel("Disconnected")
        self._status_label.setStyleSheet("color: red;")
        conn_layout.addRow("Status:", self._status_label)

        conn_buttons = QtWidgets.QHBoxLayout()
        self._connect_btn = QtWidgets.QPushButton("Connect")
        self._disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self._disconnect_btn.setEnabled(False)
        conn_buttons.addWidget(self._connect_btn)
        conn_buttons.addWidget(self._disconnect_btn)
        conn_layout.addRow(conn_buttons)

        layout.addWidget(conn_group)

        # ═══════════════════════════════════════════════════════════════
        # Swarm Configuration Group
        # ═══════════════════════════════════════════════════════════════
        swarm_group = QtWidgets.QGroupBox("Swarm Configuration", self)
        swarm_layout = QtWidgets.QFormLayout(swarm_group)

        self._num_drones_spin = QtWidgets.QSpinBox()
        self._num_drones_spin.setRange(1, 20)
        self._num_drones_spin.setValue(4)
        swarm_layout.addRow("Number of Drones:", self._num_drones_spin)

        self._spacing_spin = QtWidgets.QDoubleSpinBox()
        self._spacing_spin.setRange(1.0, 50.0)
        self._spacing_spin.setValue(5.0)
        swarm_layout.addRow("Spacing (m):", self._spacing_spin)

        self._altitude_spin = QtWidgets.QDoubleSpinBox()
        self._altitude_spin.setRange(-100.0, 0.0)
        self._altitude_spin.setValue(-10.0)
        swarm_layout.addRow("Altitude (m):", self._altitude_spin)

        self._scene_combo = QtWidgets.QComboBox()
        self._scene_combo.addItems([
            "scene_basic_drone.jsonc",
            "scene_bonsai_drone_landing.jsonc",
            "scene_urban.jsonc",
        ])
        swarm_layout.addRow("Scene:", self._scene_combo)

        layout.addWidget(swarm_group)

        # ═══════════════════════════════════════════════════════════════
        # Control Group
        # ═══════════════════════════════════════════════════════════════
        control_group = QtWidgets.QGroupBox("Simulation Control", self)
        control_layout = QtWidgets.QVBoxLayout(control_group)

        control_buttons = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("Start Simulation")
        self._stop_btn = QtWidgets.QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        control_buttons.addWidget(self._start_btn)
        control_buttons.addWidget(self._stop_btn)
        control_layout.addLayout(control_buttons)

        # Swarm commands
        swarm_commands = QtWidgets.QHBoxLayout()
        self._takeoff_btn = QtWidgets.QPushButton("Takeoff All")
        self._land_btn = QtWidgets.QPushButton("Land All")
        self._hover_btn = QtWidgets.QPushButton("Hover All")
        swarm_commands.addWidget(self._takeoff_btn)
        swarm_commands.addWidget(self._land_btn)
        swarm_commands.addWidget(self._hover_btn)
        control_layout.addLayout(swarm_commands)

        layout.addWidget(control_group)

        # ═══════════════════════════════════════════════════════════════
        # Status Group
        # ═══════════════════════════════════════════════════════════════
        status_group = QtWidgets.QGroupBox("Swarm Status", self)
        status_layout = QtWidgets.QVBoxLayout(status_group)

        self._drone_table = QtWidgets.QTableWidget()
        self._drone_table.setColumnCount(5)
        self._drone_table.setHorizontalHeaderLabels([
            "Name", "X", "Y", "Z", "Status"
        ])
        self._drone_table.setMaximumHeight(200)
        status_layout.addWidget(self._drone_table)

        layout.addWidget(status_group)

        layout.addStretch()

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

    def _on_connect(self):
        self.connect_requested.emit(
            self._host_edit.text(),
            self._port_spin.value(),
        )

    def _on_disconnect(self):
        self.disconnect_requested.emit()

    def _on_start(self):
        config = {
            "num_drones": self._num_drones_spin.value(),
            "spacing": self._spacing_spin.value(),
            "altitude": self._altitude_spin.value(),
            "scene": self._scene_combo.currentText(),
        }
        self.start_simulation_requested.emit(config)

    def _on_stop(self):
        self.stop_simulation_requested.emit()

    def set_connected(self, connected: bool):
        """Update connection status."""
        if connected:
            self._status_label.setText("Connected")
            self._status_label.setStyleSheet("color: green;")
            self._connect_btn.setEnabled(False)
            self._disconnect_btn.setEnabled(True)
        else:
            self._status_label.setText("Disconnected")
            self._status_label.setStyleSheet("color: red;")
            self._connect_btn.setEnabled(True)
            self._disconnect_btn.setEnabled(False)

    def update_drone_status(self, drone_states: dict):
        """Update drone status table."""
        self._drone_table.setRowCount(len(drone_states))
        for row, (name, state) in enumerate(drone_states.items()):
            self._drone_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self._drone_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{state['x']:.2f}"))
            self._drone_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{state['y']:.2f}"))
            self._drone_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{state['z']:.2f}"))
            self._drone_table.setItem(row, 4, QtWidgets.QTableWidgetItem(state.get('status', 'Unknown')))
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure
1. Create `3rd_party/airsim_worker/airsim_worker/` directory structure
2. Implement enums and configuration dataclasses
3. Create client wrapper with basic connection handling
4. Create `requirements/airsim_worker.txt`
5. Test basic connection to running AirSim server

### Phase 2: Multi-Agent Control
1. Implement multi-robot creation from config
2. Add control API (takeoff, move, land) for all robots
3. Add sensor subscription handling
4. Implement simulation stepping (steppable clock)
5. Test multi-drone control scripts

### Phase 3: Gymnasium Environment
1. Create `AirSimMultiAgentEnv` class
2. Implement observation/action spaces
3. Add reward computation
4. Implement reset/step cycle
5. Test with random actions

### Phase 4: GUI Integration
1. Create `AirSimTab` widget
2. Add tab to control panel
3. Connect signals to handlers
4. Display drone status table
5. Add camera feed visualization

### Phase 5: Advanced Features
1. Support heterogeneous teams (drones + rovers)
2. Add formation control primitives
3. Implement collision detection
4. Add weather effects control
5. Integrate with RL training loop

---

## 7. Files to Create

### New Files

| File | Purpose |
|------|---------|
| `3rd_party/airsim_worker/airsim_worker/__init__.py` | Package exports |
| `3rd_party/airsim_worker/airsim_worker/enums.py` | Enum definitions |
| `3rd_party/airsim_worker/airsim_worker/config.py` | Configuration dataclasses |
| `3rd_party/airsim_worker/airsim_worker/client.py` | Multi-agent client wrapper |
| `3rd_party/airsim_worker/airsim_worker/multi_agent_env.py` | Gymnasium environment |
| `3rd_party/airsim_worker/airsim_worker/launcher.py` | UE5 server launcher |
| `3rd_party/airsim_worker/airsim_worker/cli.py` | CLI entry point |
| `3rd_party/airsim_worker/pyproject.toml` | Package definition |
| `requirements/airsim_worker.txt` | Dependencies |
| `gym_gui/ui/widgets/airsim_tab.py` | GUI tab widget |

### Modified Files

| File | Change |
|------|--------|
| `gym_gui/ui/widgets/control_panel.py` | Add AirSim tab |
| `gym_gui/core/enums.py` | Add AirSim environment type |

---

## 8. Dependencies

### Location: `requirements/airsim_worker.txt`

```txt
# AirSim Worker Dependencies
# Install with: pip install -r requirements/airsim_worker.txt
#
# This file contains dependencies for Project AirSim multi-agent
# drone/rover simulation.

# Include base requirements
-r base.txt

# ═══════════════════════════════════════════════════════════════
# Project AirSim Python Client Dependencies
# (from ProjectAirSim/client/python/projectairsim/pyproject.toml)
# ═══════════════════════════════════════════════════════════════

# NNG (nanomsg-next-gen) for communication
pynng>=0.5.0

# Message serialization
msgpack>=1.0.5

# Image processing
opencv-python>=4.2.0.32
numpy

# Visualization
matplotlib

# JSON configuration
commentjson>=0.9.0
jsonschema>=4.4.0

# Input handling
inputs

# Priority queue dictionary
pqdict

# Security
cryptography

# KML file support (flight paths)
pykml

# Testing output
junit-xml
jsonlines

# Geometry
Shapely

# Point cloud processing
open3d>=0.16.0

# ═══════════════════════════════════════════════════════════════
# Optional: Autonomy Dependencies
# (for RL training and perception)
# ═══════════════════════════════════════════════════════════════

# PyTorch (uncomment for RL training)
# torch>=1.8.0
# torchvision>=0.9.0

# Image processing for perception
# scikit-image>=0.18.1
# Pillow<=8.2.0

# Gymnasium for RL environments
gymnasium>=0.29.0

# ONNX runtime (for model inference)
# onnxruntime-gpu

# API server (for remote control)
# pydantic
# python-multipart
# fastapi
# uvicorn
```

---

## 9. UE5 Installation

### Extract Unreal Engine 5.5.0

The UE5 binary is already downloaded at:
```
/home/hamid/Desktop/Projects/GUI_BDI_RL/3rd_party/airsim_worker/ProjectAirSim/Linux_Unreal_Engine_5.5.0.zip
```

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL/3rd_party/airsim_worker/ProjectAirSim

# Extract UE5 (this will take a while, ~30GB extracted)
unzip Linux_Unreal_Engine_5.5.0.zip -d UnrealEngine

# Set environment variable
export UE_ROOT=/home/hamid/Desktop/Projects/GUI_BDI_RL/3rd_party/airsim_worker/ProjectAirSim/UnrealEngine
```

### Install Linux Prerequisites

```bash
# Vulkan support
sudo apt update
sudo apt install -y libvulkan1 vulkan-tools

# Verify Vulkan
vulkaninfo

# NVIDIA drivers (if using NVIDIA GPU)
nvidia-smi
```

### Build Project AirSim

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL/3rd_party/airsim_worker/ProjectAirSim

# Install dev tools
./setup_linux_dev_tools.sh

# Build simulation libraries
./build.sh simlibs_debug

# Generate project files
./blocks_genprojfiles_vscode.sh
```

### Install Python Client

```bash
cd /home/hamid/Desktop/Projects/GUI_BDI_RL/3rd_party/airsim_worker/ProjectAirSim/client/python/projectairsim

# Install in editable mode
pip install -e .
```

---

## 10. Usage Examples

### Basic Multi-Drone Control

```python
import asyncio
from airsim_worker import AirSimMultiAgentClient, MultiAgentConfig

async def main():
    # Create config for 4-drone swarm
    config = MultiAgentConfig.create_drone_swarm(
        num_drones=4,
        spacing=5.0,
        altitude=-10.0,
    )

    # Create client
    client = AirSimMultiAgentClient(config)

    # Connect and load scene
    client.connect()
    client.load_scene()

    # Enable control and arm
    client.enable_all_api_control()
    client.arm_all()

    # Takeoff all drones
    await client.takeoff_all_async()

    # Move all drones forward
    for name in client.get_robot_names():
        await client.move_by_velocity_async(
            robot_name=name,
            v_north=2.0,
            v_east=0.0,
            v_down=0.0,
            duration=5.0,
        )

    # Disconnect
    client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### Gymnasium Environment for RL

```python
from airsim_worker import AirSimMultiAgentEnv, MultiAgentConfig

# Create config
config = MultiAgentConfig.create_drone_swarm(num_drones=4)

# Create environment
env = AirSimMultiAgentEnv(config)

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(1000):
    # Random actions for all agents
    actions = {
        agent: env.action_space[agent].sample()
        for agent in env.agents
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    if any(terminations.values()) or any(truncations.values()):
        break

env.close()
```

---

## 11. Summary

The AirSim Worker provides:

1. **Multi-Agent Drone/Rover Control** - Control swarms via unified API
2. **Gymnasium Integration** - Standard RL environment interface
3. **Rich Sensor Suite** - Camera, LiDAR, IMU, GPS, etc.
4. **UE5 Rendering** - Photo-realistic visuals
5. **GUI Integration** - Tab in Mosaic control panel

This enables multi-agent RL research with realistic drone physics and visuals.

---

## 12. References

- [Project AirSim GitHub](https://github.com/iamaisim/ProjectAirSim)
- [Project AirSim API Docs](https://github.com/iamaisim/ProjectAirSim/blob/main/docs/api.md)
- [Multiple Robots Docs](https://github.com/iamaisim/ProjectAirSim/blob/main/docs/multiple_robots.md)
- [Original AirSim (Microsoft)](https://github.com/microsoft/AirSim)
- [Unreal Engine 5](https://www.unrealengine.com/)
- [PettingZoo Multi-Agent API](https://pettingzoo.farama.org/)
