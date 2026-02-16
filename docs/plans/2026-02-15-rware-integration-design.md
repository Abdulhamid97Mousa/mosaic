# RWARE (Robotic Warehouse) Integration into MOSAIC

## Status: DESIGN APPROVED

## Overview

Integrate the Robotic Warehouse (RWARE) multi-agent cooperative environment into MOSAIC
as a new environment family. RWARE simulates a warehouse where robots pick up shelves,
deliver them to workstations, and return them to empty locations.

- **Source:** `3rd_party/robotic-warehouse/` (local clone)
- **Package:** `rware` v2.0.0
- **API:** Pure Gymnasium with tuple-based multi-agent spaces (NOT PettingZoo)
- **Rendering:** Pyglet-based 2D grid (supports `rgb_array` mode)
- **Actions:** 5 discrete (NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD)
- **Observations:** 4 types (Flattened, Dict, Image, Image+Dict)
- **Rewards:** 3 types (Global, Individual, Two-Stage)
- **Agents:** 1-19 configurable, cooperative
- **Human play:** Supported (arrow keys, Tab for agent switching, P for pickup)

## Integration Approach: Direct Adapter

Like SMAC, a direct adapter wrapping RWARE's `Warehouse` class. This gives full control
over multi-agent observation packaging, action spaces, and reward handling.

Rejected alternatives:
- PettingZoo wrapper: RWARE is not PettingZoo-native; forcing it adds unnecessary indirection
- Gymnasium passthrough: Loses multi-agent semantics entirely

---

## Environment Family and GameIds

**One family:** `EnvironmentFamily.RWARE = "rware"`

**12 GameId variants (full coverage):**

| GameId Enum | Env ID String | Size | Agents | Difficulty |
|-------------|---------------|------|--------|------------|
| RWARE_TINY_2AG | rware-tiny-2ag-v2 | tiny (1x3) | 2 | normal |
| RWARE_TINY_4AG | rware-tiny-4ag-v2 | tiny (1x3) | 4 | normal |
| RWARE_SMALL_2AG | rware-small-2ag-v2 | small (2x3) | 2 | normal |
| RWARE_SMALL_4AG | rware-small-4ag-v2 | small (2x3) | 4 | normal |
| RWARE_MEDIUM_2AG | rware-medium-2ag-v2 | medium (2x5) | 2 | normal |
| RWARE_MEDIUM_4AG | rware-medium-4ag-v2 | medium (2x5) | 4 | normal |
| RWARE_MEDIUM_4AG_EASY | rware-medium-4ag-easy-v2 | medium (2x5) | 4 | easy |
| RWARE_MEDIUM_4AG_HARD | rware-medium-4ag-hard-v2 | medium (2x5) | 4 | hard |
| RWARE_LARGE_4AG | rware-large-4ag-v2 | large (3x5) | 4 | normal |
| RWARE_LARGE_4AG_HARD | rware-large-4ag-hard-v2 | large (3x5) | 4 | hard |
| RWARE_LARGE_8AG | rware-large-8ag-v2 | large (3x5) | 8 | normal |
| RWARE_LARGE_8AG_HARD | rware-large-8ag-hard-v2 | large (3x5) | 8 | hard |

**Mapping dict values:**
- `ENVIRONMENT_FAMILY_BY_GAME`: All 12 -> `EnvironmentFamily.RWARE`
- `DEFAULT_RENDER_MODES`: All 12 -> `RenderMode.RGB_ARRAY`
- `DEFAULT_CONTROL_MODES`: All 12 -> `(ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP)`
- `DEFAULT_PARADIGM_BY_FAMILY`: `EnvironmentFamily.RWARE` -> `SteppingParadigm.SIMULTANEOUS`

---

## RWAREConfig Dataclass

```python
@dataclass
class RWAREConfig:
    """Configuration for Robotic Warehouse (RWARE) multi-agent environments."""

    # Map parameters (derived from GameId, but overridable)
    shelf_columns: int = 3
    column_height: int = 8
    shelf_rows: int = 1
    n_agents: int = 2

    # Observation
    observation_type: str = "flattened"  # "flattened", "dict", "image", "image_dict"
    sensor_range: int = 1               # 1-5 cells visible around agent
    normalised_coordinates: bool = False

    # Reward
    reward_type: str = "global"  # "global", "individual", "two_stage"

    # Communication
    msg_bits: int = 0  # 0 = silent, >0 = communication channels

    # Episode limits
    max_steps: int = 500
    max_inactivity_steps: int | None = None  # None = use env default
    request_queue_size: int | None = None    # None = auto from difficulty

    # Misc
    seed: int | None = None
    render_mode: str = "rgb_array"
```

**Config panel widgets:**
- Observation Type: QComboBox (Flattened / Dict / Image / Image+Dict)
- Sensor Range: QSpinBox (1-5, default 1)
- Reward Type: QComboBox (Global / Individual / Two-Stage)
- Communication Bits: QSpinBox (0-8, default 0)
- Max Steps: QSpinBox (100-10000, default 500)
- Seed: QSpinBox (-1 for random, 0-99999)

---

## Adapter Architecture

```python
class RWAREAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Base adapter for Robotic Warehouse environments."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    # Subclasses override these
    _default_n_agents: int = 2
    _default_shelf_columns: int = 3
    _default_column_height: int = 8
    _default_shelf_rows: int = 1
    _default_difficulty: str | None = None  # None, "easy", "hard"

    @property
    def stepping_paradigm(self) -> SteppingParadigm:
        return SteppingParadigm.SIMULTANEOUS

    def load(self) -> None:
        """Create the Warehouse env via gym.make() with config parameters."""
        ...

    def reset(self, *, seed=None, options=None) -> AdapterStep:
        """Reset and package per-agent observations."""
        ...

    def step(self, action: List[int]) -> AdapterStep:
        """Execute actions for all agents simultaneously."""
        ...

    def render(self) -> np.ndarray | None:
        """Return pyglet rgb_array frame."""
        ...

    def close(self) -> None:
        """Release pyglet resources."""
        ...
```

**12 concrete subclasses** (e.g.):
```python
class RWARETiny2AgAdapter(RWAREAdapter):
    _default_shelf_columns = 1
    _default_shelf_rows = 1
    _default_n_agents = 2

class RWARELarge8AgHardAdapter(RWAREAdapter):
    _default_shelf_columns = 3
    _default_shelf_rows = 3
    _default_n_agents = 8
    _default_difficulty = "hard"

RWARE_ADAPTERS: Dict[GameId, type[RWAREAdapter]] = {
    GameId.RWARE_TINY_2AG: RWARETiny2AgAdapter,
    # ... 11 more
}
```

---

## Human Play Support

**Keyboard mapping (human_input.py):**

| Key | Action | Index |
|-----|--------|-------|
| Up Arrow | FORWARD | 1 |
| Left Arrow | LEFT (rotate) | 2 |
| Right Arrow | RIGHT (rotate) | 3 |
| P / L | TOGGLE_LOAD (pick up / drop shelf) | 4 |
| Space | NOOP | 0 |

**Agent switching:** Tab key cycles the active agent. The interaction controller
tracks `_current_agent_idx` and applies the human action only to that agent while
others get NOOP.

**Control modes:**
- `HUMAN_ONLY`: Human controls one agent via keyboard, others get NOOP (Tab to switch)
- `AGENT_ONLY`: All agents controlled by RL policy
- `MULTI_AGENT_COOP`: All agents controlled by RL policies cooperatively

---

## Rendering

RWARE's pyglet renderer produces 2D grid visualizations:
- Shelves: Dark slate blue
- Requested shelves: Teal
- Agents: Dark orange (red when carrying)
- Goals: Dark gray
- Grid: Black lines on white background

The adapter creates the env with `render_mode="rgb_array"` and calls `env.render()`
to get numpy arrays. No custom renderer needed -- pyglet's rgb_array mode works
headlessly.

---

## Game Documentation

```
gym_gui/game_docs/RWARE/
    __init__.py               # Re-exports all HTML constants
    _shared.py                # Shared fragments: actions table, obs description, reward types
    RWARE_Tiny/__init__.py    # Docs for tiny variants (2ag, 4ag)
    RWARE_Small/__init__.py   # Docs for small variants (2ag, 4ag)
    RWARE_Medium/__init__.py  # Docs for medium variants (2ag, 4ag, easy, hard)
    RWARE_Large/__init__.py   # Docs for large variants (4ag, 4ag-hard, 8ag, 8ag-hard)
```

Each doc module exports HTML strings per GameId (e.g., `RWARE_TINY_2AG_HTML`).
Multiple GameIds per doc file (grouped by size) since they share most content
with only agent count and difficulty differing.

---

## Log Constants

Reserve LOG970-LOG979 for RWARE (following SMAC's LOG960-LOG969):

| Constant | ID | Level | Description |
|----------|----|-------|-------------|
| LOG_RWARE_ENV_CREATED | LOG970 | INFO | RWARE environment created |
| LOG_RWARE_ENV_RESET | LOG971 | INFO | RWARE environment reset |
| LOG_RWARE_STEP_SUMMARY | LOG972 | DEBUG | RWARE step summary |
| LOG_RWARE_ENV_CLOSED | LOG973 | INFO | RWARE environment closed |
| LOG_RWARE_RENDER_ERROR | LOG974 | WARNING | RWARE render error |
| LOG_RWARE_DELIVERY | LOG975 | INFO | Shelf delivery event |

---

## Dependencies

**requirements/rware.txt:**
```
# Robotic Warehouse (RWARE) multi-agent cooperative environment
# Install: pip install -e 3rd_party/robotic-warehouse/
# Source: https://github.com/uoe-agents/robotic-warehouse
numpy
gymnasium>=0.26.0
pyglet<2
networkx
```

**pyproject.toml:**
```toml
rware = [
    "gymnasium>=1.1.0",
    "pyglet<2.0.0",
    "networkx>=2.8.0",
]
```

**Installation:** `pip install -e 3rd_party/robotic-warehouse/`

---

## New Files (10)

| File | Purpose |
|------|---------|
| `gym_gui/core/adapters/rware.py` | Adapter base + 12 subclasses |
| `gym_gui/game_docs/RWARE/__init__.py` | Game doc re-exports |
| `gym_gui/game_docs/RWARE/_shared.py` | Shared HTML fragments |
| `gym_gui/game_docs/RWARE/RWARE_Tiny/__init__.py` | Tiny variant docs |
| `gym_gui/game_docs/RWARE/RWARE_Small/__init__.py` | Small variant docs |
| `gym_gui/game_docs/RWARE/RWARE_Medium/__init__.py` | Medium variant docs |
| `gym_gui/game_docs/RWARE/RWARE_Large/__init__.py` | Large variant docs |
| `gym_gui/ui/config_panels/multi_agent/rware/__init__.py` | Config panel package |
| `gym_gui/ui/config_panels/multi_agent/rware/config_panel.py` | UI controls |
| `requirements/rware.txt` | Dependencies |

## Modified Files (11)

| File | Changes |
|------|---------|
| `gym_gui/core/enums.py` | +1 family, +12 GameIds, +4 mapping dicts |
| `gym_gui/logging_config/log_constants.py` | +6 RWARE log constants (LOG970-975) |
| `gym_gui/config/game_configs.py` | +RWAREConfig dataclass, +GameConfig alias |
| `gym_gui/config/game_config_builder.py` | +RWARE branch in build_config() |
| `gym_gui/core/adapters/__init__.py` | +RWARE conditional import |
| `gym_gui/core/factories/adapters.py` | +RWARE try/except + registry |
| `gym_gui/game_docs/__init__.py` | +RWARE doc registration (12 maps) |
| `gym_gui/ui/widgets/control_panel.py` | +RWARE panel dispatch |
| `gym_gui/app.py` | +rware dependency detection |
| `pyproject.toml` | +rware optional dep group, update all-envs |
| `gym_gui/controllers/human_input.py` | +RWARE keyboard mapping |

---

## Sources

- RWARE repository: https://github.com/uoe-agents/robotic-warehouse
- Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep Reinforcement Learning
  Algorithms in Cooperative Tasks"
- Local source: 3rd_party/robotic-warehouse/
