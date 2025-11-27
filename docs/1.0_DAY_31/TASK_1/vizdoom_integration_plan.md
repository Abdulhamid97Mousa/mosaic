# ViZDoom Integration Plan for gym_gui Human Control Mode

## Status: COMPLETED

This document outlines the plan that was used for integrating ViZDoom into the gym_gui application, enabling Human Control Mode for Doom-based RL environments. The implementation is now complete.

## Overview

## Table of Contents

1. [Architecture Analysis](#1-architecture-analysis)
2. [Requirements Management](#2-requirements-management)
3. [Component Implementation Plan](#3-component-implementation-plan)
4. [File Structure](#4-file-structure)
5. [Implementation Phases](#5-implementation-phases)
6. [Human Control Mode Design](#6-human-control-mode-design)
7. [Risk Assessment](#7-risk-assessment)

---

## 1. Architecture Analysis

### 1.1 Current gym_gui Architecture

The gym_gui follows a well-defined pattern for environment integration:

```
┌─────────────────────────────────────────────────────────────────┐
│                         gym_gui                                  │
├─────────────────────────────────────────────────────────────────┤
│  core/                                                           │
│  ├── enums.py           → GameId, ControlMode, RenderMode        │
│  ├── adapters/          → Environment lifecycle managers         │
│  │   ├── base.py        → EnvironmentAdapter ABC                 │
│  │   ├── ale.py         → ALE/Atari adapter pattern              │
│  │   ├── minigrid.py    → MiniGrid adapter pattern               │
│  │   └── vizdoom.py     → [NEW] ViZDoom adapter                  │
│  └── factories/         → Adapter registry & creation            │
├─────────────────────────────────────────────────────────────────┤
│  game_docs/                                                      │
│  ├── ALE/               → Atari game documentation               │
│  ├── MiniGrid/          → MiniGrid game documentation            │
│  └── ViZDoom/           → [NEW] ViZDoom documentation            │
├─────────────────────────────────────────────────────────────────┤
│  rendering/                                                      │
│  ├── strategies/        → Render strategy implementations        │
│  │   ├── rgb.py         → RGB array renderer (reusable)          │
│  │   └── vizdoom.py     → [OPTIONAL] Custom ViZDoom renderer     │
│  └── registry.py        → Strategy registration                  │
├─────────────────────────────────────────────────────────────────┤
│  ui/environments/                                                │
│  └── single_agent_env/                                           │
│      └── vizdoom/       → [NEW] ViZDoom UI components            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 ViZDoom API Overview

ViZDoom provides two integration approaches:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Native API** | Direct `vizdoom.DoomGame` | Full control, SPECTATOR mode | More boilerplate |
| **Gymnasium Wrapper** | `vizdoom.gymnasium_wrapper` | Standard Gymnasium API | Less control over modes |

**Recommendation**: Use **Native API** for Human Control Mode because:
- SPECTATOR mode allows human keyboard input natively
- Better control over game variables (health, ammo, kills)
- Direct access to depth/labels/automap buffers

### 1.3 Control Mode Mapping

| gym_gui ControlMode | ViZDoom Mode | Description |
|--------------------|--------------|-------------|
| `HUMAN_ONLY` | `Mode.SPECTATOR` | Human plays via keyboard, agent observes |
| `AGENT_ONLY` | `Mode.PLAYER` | Agent controls, human observes |
| `HYBRID_TURN_BASED` | Custom logic | Alternating control |
| `HYBRID_HUMAN_AGENT` | Custom logic | Concurrent control |

---

## 2. Requirements Management

### 2.1 New Requirements File

**File**: `requirements/vizdoom.txt`

```txt
# ViZDoom environment dependencies
# Include this file for ViZDoom support: pip install -r requirements/vizdoom.txt

-r base.txt

# ViZDoom - Doom-based RL research platform
# Requires OpenAL on Linux: sudo apt install libopenal-dev
vizdoom>=1.2.0,<2.0.0
```

### 2.2 Root requirements.txt Update

Add to `/home/hamid/Desktop/Projects/GUI_BDI_RL/requirements.txt`:

```txt
# ViZDoom worker stack (Doom environments)
# -r requirements/vizdoom.txt  # Uncomment to enable ViZDoom
```

**Rationale**: Keep ViZDoom optional since:
- Large dependency (includes game engine)
- Requires OpenAL system library
- Not needed for basic gym_gui usage

### 2.3 pyproject.toml Update

Add optional dependency group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
vizdoom = [
    "vizdoom>=1.2.0,<2.0.0",
]
```

This enables: `pip install -e .[vizdoom]`

### 2.4 System Dependencies

Document in requirements.txt header:

```txt
# ViZDoom System Requirements:
# Linux:   sudo apt install libopenal-dev
# macOS:   brew install openal-soft
# Windows: OpenAL included with ViZDoom wheel
```

---

## 3. Component Implementation Plan

### 3.1 Enums Extension (`gym_gui/core/enums.py`)

Add new entries:

```python
class EnvironmentFamily(StrEnum):
    # ... existing ...
    VIZDOOM = "vizdoom"

class GameId(StrEnum):
    # ... existing ...
    # ViZDoom Scenarios
    VIZDOOM_BASIC = "ViZDoom-Basic-v0"
    VIZDOOM_DEADLY_CORRIDOR = "ViZDoom-DeadlyCorridor-v0"
    VIZDOOM_DEFEND_THE_CENTER = "ViZDoom-DefendTheCenter-v0"
    VIZDOOM_DEFEND_THE_LINE = "ViZDoom-DefendTheLine-v0"
    VIZDOOM_HEALTH_GATHERING = "ViZDoom-HealthGathering-v0"
    VIZDOOM_HEALTH_GATHERING_SUPREME = "ViZDoom-HealthGatheringSupreme-v0"
    VIZDOOM_MY_WAY_HOME = "ViZDoom-MyWayHome-v0"
    VIZDOOM_PREDICT_POSITION = "ViZDoom-PredictPosition-v0"
    VIZDOOM_TAKE_COVER = "ViZDoom-TakeCover-v0"
    VIZDOOM_DEATHMATCH = "ViZDoom-Deathmatch-v0"
```

Update mappings:
- `ENVIRONMENT_FAMILY_BY_GAME`
- `DEFAULT_RENDER_MODES` (all → `RenderMode.RGB_ARRAY`)
- `DEFAULT_CONTROL_MODES`

### 3.2 Adapter Implementation (`gym_gui/core/adapters/vizdoom.py`)

```python
"""ViZDoom environment adapters for Doom-based RL scenarios."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode

# Lazy import to handle optional dependency
vizdoom = None

def _ensure_vizdoom():
    global vizdoom
    if vizdoom is None:
        try:
            import vizdoom as vzd
            vizdoom = vzd
        except ImportError as e:
            raise ImportError(
                "ViZDoom not installed. Install with: pip install vizdoom"
            ) from e
    return vizdoom


@dataclass(slots=True)
class ViZDoomConfig:
    """Configuration for ViZDoom environments."""

    screen_resolution: str = "RES_640X480"
    screen_format: str = "RGB24"
    render_hud: bool = True
    render_weapon: bool = True
    render_crosshair: bool = False
    episode_timeout: int = 2100  # ~60 seconds at 35 fps
    living_reward: float = 0.0
    death_penalty: float = 100.0
    freelook: bool = True
    sound_enabled: bool = False
    depth_buffer: bool = False
    labels_buffer: bool = False
    automap_buffer: bool = False


class ViZDoomAdapter(EnvironmentAdapter[np.ndarray, list[int]]):
    """Base adapter for all ViZDoom scenarios."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    )

    # Subclasses override
    _scenario_file: str = "basic.cfg"
    _available_buttons: list[str] = []
    _available_game_variables: list[str] = []

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: ViZDoomConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or ViZDoomConfig()
        self._game: Any = None  # vizdoom.DoomGame
        self._step_counter = 0
        self._episode_return = 0.0

    def load(self) -> None:
        """Initialize the ViZDoom game instance."""
        vzd = _ensure_vizdoom()

        self._game = vzd.DoomGame()
        self._game.load_config(vzd.scenarios_path + "/" + self._scenario_file)

        # Apply configuration
        res = getattr(vzd.ScreenResolution, self._config.screen_resolution)
        fmt = getattr(vzd.ScreenFormat, self._config.screen_format)
        self._game.set_screen_resolution(res)
        self._game.set_screen_format(fmt)

        self._game.set_render_hud(self._config.render_hud)
        self._game.set_render_weapon(self._config.render_weapon)
        self._game.set_render_crosshair(self._config.render_crosshair)
        self._game.set_episode_timeout(self._config.episode_timeout)
        self._game.set_living_reward(self._config.living_reward)
        self._game.set_death_penalty(self._config.death_penalty)
        self._game.set_sound_enabled(self._config.sound_enabled)

        # Buffer configuration
        self._game.set_depth_buffer_enabled(self._config.depth_buffer)
        self._game.set_labels_buffer_enabled(self._config.labels_buffer)
        self._game.set_automap_buffer_enabled(self._config.automap_buffer)

        # Control mode determines ViZDoom mode
        if self._context and self._context.control_mode == ControlMode.HUMAN_ONLY:
            self._game.set_mode(vzd.Mode.SPECTATOR)
            self._game.set_window_visible(True)
            if self._config.freelook:
                self._game.add_game_args("+freelook 1")
        else:
            self._game.set_mode(vzd.Mode.PLAYER)
            self._game.set_window_visible(False)

        self._game.init()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        """Start a new episode."""
        if seed is not None:
            self._game.set_seed(seed)

        self._game.new_episode()
        self._step_counter = 0
        self._episode_return = 0.0

        state = self._game.get_state()
        obs = state.screen_buffer if state else np.zeros((480, 640, 3), dtype=np.uint8)
        info = self._build_info(state)

        return AdapterStep(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            render_payload=self.render(),
            state=self.build_step_state(obs, info),
        )

    def step(self, action: list[int]) -> AdapterStep[np.ndarray]:
        """Execute action and return result."""
        vzd = _ensure_vizdoom()

        if self._context and self._context.control_mode == ControlMode.HUMAN_ONLY:
            # Human control: just advance, don't set action
            self._game.advance_action()
            reward = self._game.get_last_reward()
        else:
            # Agent control
            reward = self._game.make_action(action)

        self._step_counter += 1
        self._episode_return += reward

        terminated = self._game.is_episode_finished()
        state = self._game.get_state()

        if state is not None:
            obs = state.screen_buffer
        else:
            obs = np.zeros((480, 640, 3), dtype=np.uint8)

        info = self._build_info(state)

        return AdapterStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=False,
            info=info,
            render_payload=self.render(),
            state=self.build_step_state(obs, info),
        )

    def render(self) -> dict[str, Any]:
        """Return RGB render payload."""
        state = self._game.get_state() if self._game else None
        if state is not None:
            frame = np.asarray(state.screen_buffer)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self.id,
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._game:
            self._game.close()
            self._game = None

    def _build_info(self, state: Any) -> dict[str, Any]:
        """Extract game state info."""
        if state is None:
            return {}

        vzd = _ensure_vizdoom()
        info = {
            "tic": state.tic,
            "frame_number": state.number,
        }

        # Extract game variables if available
        if state.game_variables is not None:
            for i, var in enumerate(self._available_game_variables):
                if i < len(state.game_variables):
                    info[var.lower()] = state.game_variables[i]

        # Human action (for SPECTATOR mode)
        if self._context and self._context.control_mode == ControlMode.HUMAN_ONLY:
            info["last_action"] = self._game.get_last_action()

        return info

    def build_step_state(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> StepState:
        """Build machine-readable state snapshot."""
        metrics = {
            "step": self._step_counter,
            "episode_return": float(self._episode_return),
            "tic": info.get("tic", 0),
        }

        environment = {
            "scenario": self._scenario_file,
            "health": info.get("health", 100),
            "ammo": info.get("ammo", 0),
            "kills": info.get("killcount", 0),
        }

        return StepState(
            metrics=metrics,
            environment=environment,
            raw=dict(info),
        )


# Concrete scenario adapters

class ViZDoomBasicAdapter(ViZDoomAdapter):
    """Basic shooting scenario - shoot the monster."""

    id = GameId.VIZDOOM_BASIC.value
    _scenario_file = "basic.cfg"
    _available_buttons = ["ATTACK", "MOVE_LEFT", "MOVE_RIGHT"]
    _available_game_variables = ["AMMO2"]


class ViZDoomDeadlyCorridorAdapter(ViZDoomAdapter):
    """Navigate corridor while avoiding enemies."""

    id = GameId.VIZDOOM_DEADLY_CORRIDOR.value
    _scenario_file = "deadly_corridor.cfg"
    _available_buttons = [
        "ATTACK", "MOVE_LEFT", "MOVE_RIGHT",
        "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"
    ]
    _available_game_variables = ["HEALTH", "AMMO2"]


class ViZDoomDefendTheCenterAdapter(ViZDoomAdapter):
    """Defend position from incoming enemies."""

    id = GameId.VIZDOOM_DEFEND_THE_CENTER.value
    _scenario_file = "defend_the_center.cfg"
    _available_buttons = ["ATTACK", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = ["HEALTH", "AMMO2"]


class ViZDoomDefendTheLineAdapter(ViZDoomAdapter):
    """Defend a line from enemies."""

    id = GameId.VIZDOOM_DEFEND_THE_LINE.value
    _scenario_file = "defend_the_line.cfg"
    _available_buttons = ["ATTACK", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = ["HEALTH", "AMMO2"]


class ViZDoomHealthGatheringAdapter(ViZDoomAdapter):
    """Collect health packs to survive."""

    id = GameId.VIZDOOM_HEALTH_GATHERING.value
    _scenario_file = "health_gathering.cfg"
    _available_buttons = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = ["HEALTH"]


class ViZDoomHealthGatheringSupremeAdapter(ViZDoomAdapter):
    """Complex health gathering with hazards."""

    id = GameId.VIZDOOM_HEALTH_GATHERING_SUPREME.value
    _scenario_file = "health_gathering_supreme.cfg"
    _available_buttons = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = ["HEALTH"]


class ViZDoomMyWayHomeAdapter(ViZDoomAdapter):
    """Navigate maze to find the goal."""

    id = GameId.VIZDOOM_MY_WAY_HOME.value
    _scenario_file = "my_way_home.cfg"
    _available_buttons = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = []


class ViZDoomPredictPositionAdapter(ViZDoomAdapter):
    """Predict enemy position and shoot."""

    id = GameId.VIZDOOM_PREDICT_POSITION.value
    _scenario_file = "predict_position.cfg"
    _available_buttons = ["ATTACK", "TURN_LEFT", "TURN_RIGHT"]
    _available_game_variables = ["AMMO2"]


class ViZDoomTakeCoverAdapter(ViZDoomAdapter):
    """Take cover from incoming projectiles."""

    id = GameId.VIZDOOM_TAKE_COVER.value
    _scenario_file = "take_cover.cfg"
    _available_buttons = ["MOVE_LEFT", "MOVE_RIGHT"]
    _available_game_variables = ["HEALTH"]


class ViZDoomDeathmatchAdapter(ViZDoomAdapter):
    """Full deathmatch scenario."""

    id = GameId.VIZDOOM_DEATHMATCH.value
    _scenario_file = "deathmatch.cfg"
    _available_buttons = [
        "ATTACK", "USE", "MOVE_FORWARD", "MOVE_BACKWARD",
        "MOVE_LEFT", "MOVE_RIGHT", "TURN_LEFT", "TURN_RIGHT"
    ]
    _available_game_variables = ["HEALTH", "AMMO2", "ARMOR", "KILLCOUNT"]


# Adapter registry
VIZDOOM_ADAPTERS: dict[GameId, type[ViZDoomAdapter]] = {
    GameId.VIZDOOM_BASIC: ViZDoomBasicAdapter,
    GameId.VIZDOOM_DEADLY_CORRIDOR: ViZDoomDeadlyCorridorAdapter,
    GameId.VIZDOOM_DEFEND_THE_CENTER: ViZDoomDefendTheCenterAdapter,
    GameId.VIZDOOM_DEFEND_THE_LINE: ViZDoomDefendTheLineAdapter,
    GameId.VIZDOOM_HEALTH_GATHERING: ViZDoomHealthGatheringAdapter,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: ViZDoomHealthGatheringSupremeAdapter,
    GameId.VIZDOOM_MY_WAY_HOME: ViZDoomMyWayHomeAdapter,
    GameId.VIZDOOM_PREDICT_POSITION: ViZDoomPredictPositionAdapter,
    GameId.VIZDOOM_TAKE_COVER: ViZDoomTakeCoverAdapter,
    GameId.VIZDOOM_DEATHMATCH: ViZDoomDeathmatchAdapter,
}
```

### 3.3 Game Documentation (`gym_gui/game_docs/ViZDoom/`)

Structure:
```
gym_gui/game_docs/ViZDoom/
├── __init__.py           # Aggregator exports
├── basic.py              # Basic scenario docs
├── combat.py             # Combat scenario docs (Defend*, Deadly*)
├── navigation.py         # Navigation scenario docs (MyWayHome, HealthGathering)
└── multiplayer.py        # Deathmatch docs
```

### 3.4 UI Configuration Panel (`gym_gui/ui/environments/single_agent_env/vizdoom/`)

Structure:
```
gym_gui/ui/environments/single_agent_env/vizdoom/
├── __init__.py           # Exports VIZDOOM_GAME_IDS, build_vizdoom_controls
└── config_panel.py       # Configuration UI builders
```

---

## 4. File Structure

### 4.1 New Files to Create

```
gym_gui/
├── core/
│   └── adapters/
│       └── vizdoom.py              # [NEW] ViZDoom adapter
├── game_docs/
│   └── ViZDoom/                    # [NEW] Directory
│       ├── __init__.py
│       ├── basic.py
│       ├── combat.py
│       ├── navigation.py
│       └── multiplayer.py
└── ui/
    └── environments/
        └── single_agent_env/
            └── vizdoom/            # [NEW] Directory
                ├── __init__.py
                └── config_panel.py

requirements/
└── vizdoom.txt                     # [NEW] ViZDoom dependencies
```

### 4.2 Files to Modify

| File | Changes |
|------|---------|
| `gym_gui/core/enums.py` | Add `VIZDOOM` family, `GameId` entries, mappings |
| `gym_gui/core/adapters/__init__.py` | Export ViZDoom adapters |
| `gym_gui/core/factories/adapters.py` | Register ViZDoom adapters |
| `gym_gui/game_docs/__init__.py` | Import and export ViZDoom docs |
| `gym_gui/ui/environments/single_agent_env/__init__.py` | Export ViZDoom UI |
| `requirements.txt` | Add commented ViZDoom include |
| `pyproject.toml` | Add vizdoom optional dependency |

---

## 5. Implementation Phases

### Phase 1: Foundation (Core Infrastructure)

**Tasks:**
1. Create `requirements/vizdoom.txt`
2. Update `pyproject.toml` with optional dependency
3. Add `EnvironmentFamily.VIZDOOM` to enums
4. Add initial `GameId` entries (Basic, HealthGathering, DefendTheCenter)
5. Update `ENVIRONMENT_FAMILY_BY_GAME`, `DEFAULT_RENDER_MODES`, `DEFAULT_CONTROL_MODES`

**Deliverable:** ViZDoom can be optionally installed

### Phase 2: Adapter Implementation

**Tasks:**
1. Create `gym_gui/core/adapters/vizdoom.py`
2. Implement `ViZDoomConfig` dataclass
3. Implement base `ViZDoomAdapter` class
4. Implement 3 initial scenario adapters:
   - `ViZDoomBasicAdapter`
   - `ViZDoomHealthGatheringAdapter`
   - `ViZDoomDefendTheCenterAdapter`
5. Create `VIZDOOM_ADAPTERS` registry
6. Register in `gym_gui/core/factories/adapters.py`

**Deliverable:** Can load and step through ViZDoom environments

### Phase 3: Human Control Mode

**Tasks:**
1. Implement SPECTATOR mode integration in adapter
2. Handle keyboard input passthrough
3. Add `get_last_action()` tracking for demonstration collection
4. Test human control with visible window

**Deliverable:** Human can play ViZDoom scenarios through gym_gui

### Phase 4: Game Documentation

**Tasks:**
1. Create `gym_gui/game_docs/ViZDoom/` directory
2. Implement HTML documentation for each scenario
3. Document action space, observation space, rewards
4. Register in main `game_docs/__init__.py`

**Deliverable:** Users can browse ViZDoom scenario documentation

### Phase 5: UI Configuration Panel

**Tasks:**
1. Create `gym_gui/ui/environments/single_agent_env/vizdoom/`
2. Implement `build_vizdoom_controls()` function
3. Add controls for:
   - Screen resolution
   - Render options (HUD, weapon, crosshair)
   - Episode timeout
   - Buffer options (depth, labels, automap)
4. Export `VIZDOOM_GAME_IDS`
5. Register in single_agent_env `__init__.py`

**Deliverable:** Full GUI configuration for ViZDoom

### Phase 6: Remaining Scenarios

**Tasks:**
1. Add remaining scenario adapters:
   - DeadlyCorridor, DefendTheLine
   - HealthGatheringSupreme, MyWayHome
   - PredictPosition, TakeCover
   - Deathmatch
2. Add corresponding documentation
3. Update UI to support all scenarios

**Deliverable:** Complete ViZDoom integration

---

## 6. Human Control Mode Design

### 6.1 Control Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Human Control Mode Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User selects ViZDoom scenario in gym_gui                    │
│  2. User selects "Human Only" control mode                      │
│  3. Adapter sets Mode.SPECTATOR + window_visible=True           │
│  4. ViZDoom window opens (captures keyboard)                    │
│  5. gym_gui displays RGB observation in its viewer              │
│                                                                  │
│  Loop:                                                           │
│    a. Human presses keys in ViZDoom window                      │
│    b. Adapter calls advance_action() (no agent input)           │
│    c. Adapter retrieves:                                         │
│       - screen_buffer → RGB observation                          │
│       - get_last_action() → human's action                      │
│       - get_last_reward() → reward signal                       │
│    d. gym_gui updates display + telemetry                       │
│                                                                  │
│  6. Episode ends (death/timeout)                                 │
│  7. User can start new episode                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Keyboard Mapping

ViZDoom natively handles keyboard in SPECTATOR mode:

| Key | Action | ViZDoom Button |
|-----|--------|----------------|
| W/Arrow Up | Move Forward | MOVE_FORWARD |
| S/Arrow Down | Move Backward | MOVE_BACKWARD |
| A | Strafe Left | MOVE_LEFT |
| D | Strafe Right | MOVE_RIGHT |
| Arrow Left | Turn Left | TURN_LEFT |
| Arrow Right | Turn Right | TURN_RIGHT |
| Ctrl/LMB | Attack | ATTACK |
| Space | Use/Open | USE |
| Shift | Run | SPEED |

### 6.3 Demonstration Collection

In HUMAN_ONLY mode, the adapter collects:
```python
{
    "observation": np.ndarray,  # RGB frame
    "action": list[int],        # Human's action (from get_last_action())
    "reward": float,            # Reward received
    "game_variables": dict,     # Health, ammo, kills, etc.
    "tic": int,                 # Game tick
}
```

This enables imitation learning pipelines.

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OpenAL not installed | Medium | High | Clear error message, install instructions |
| Window focus issues | Medium | Medium | Document focus requirements |
| Version compatibility | Low | Medium | Pin vizdoom version range |
| Performance with buffers | Low | Low | Make buffers optional in config |

### 7.2 Integration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Adapter pattern mismatch | Low | High | Follow ALE adapter pattern closely |
| Rendering performance | Low | Medium | Reuse existing RGB renderer |
| Qt/ViZDoom thread conflict | Medium | High | Test thoroughly, document limitations |

### 7.3 User Experience Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Confusing dual windows | Medium | Medium | Document in game docs |
| Keyboard not responding | Medium | High | Clear focus instructions |
| Large download size | Low | Low | Make dependency optional |

---

## Summary

This integration plan follows the established gym_gui patterns:
- Adapter pattern for environment lifecycle
- Registry pattern for discovery
- Strategy pattern for rendering
- Builder pattern for UI configuration

Key decisions:
1. **Native ViZDoom API** over Gymnasium wrapper for SPECTATOR mode
2. **Optional dependency** to avoid bloating base install
3. **Reuse RGB renderer** since ViZDoom outputs RGB arrays
4. **Phased implementation** starting with 3 core scenarios

The plan enables Human Control Mode through ViZDoom's SPECTATOR mode, which naturally handles keyboard input while allowing the adapter to observe human actions for demonstration collection.
