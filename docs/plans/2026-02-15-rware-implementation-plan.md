# RWARE (Robotic Warehouse) Integration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the Robotic Warehouse (RWARE) multi-agent cooperative environment into MOSAIC as a new environment family with 12 GameId variants, config panel, game docs, and human play support.

**Architecture:** Direct adapter pattern (like SMAC) wrapping RWARE's Warehouse class via `gym.make()`. Single EnvironmentFamily, 12 GameId variants covering 4 sizes x agent counts x difficulties. Config panel controls observation type, reward type, sensor range, and communication bits.

**Tech Stack:** rware v2.0.0 (local at 3rd_party/robotic-warehouse/), Gymnasium, pyglet, PyQt6

**Design doc:** `docs/plans/2026-02-15-rware-integration-design.md`

---

### Task 1: Install RWARE and create requirements file

**Files:**
- Create: `requirements/rware.txt`
- Modify: `pyproject.toml` (lines 380-382 in all-envs bundle)

**Step 1: Install rware in editable mode**

Run: `source .venv/bin/activate && pip install -e 3rd_party/robotic-warehouse/`
Expected: Successful installation, `rware` importable

**Step 2: Verify installation**

Run: `source .venv/bin/activate && python -c "import rware; import gymnasium as gym; env = gym.make('rware-tiny-2ag-v2', render_mode='rgb_array'); obs, info = env.reset(); print(f'obs shapes: {[o.shape for o in obs]}'); env.close(); print('OK')"`
Expected: obs shapes printed, "OK"

**Step 3: Create requirements/rware.txt**

```
# Robotic Warehouse (RWARE) multi-agent cooperative environment
# Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL Algorithms"
# Repository: https://github.com/uoe-agents/robotic-warehouse
# Local source: 3rd_party/robotic-warehouse/
#
# Install from local source:
#   pip install -e 3rd_party/robotic-warehouse/
#
# Or install from PyPI:
#   pip install rware
numpy
gymnasium>=0.26.0
pyglet<2
networkx
```

**Step 4: Add rware optional dependency group to pyproject.toml**

After the `smacv2` group (~line 202), add:

```toml
# Robotic Warehouse (RWARE) multi-agent cooperative environment (University of Edinburgh)
# Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks"
# Repository: https://github.com/uoe-agents/robotic-warehouse
# Local: 3rd_party/robotic-warehouse/
# Includes: tiny/small/medium/large warehouses with 2-19 agents, 3 reward types, 4 observation types
# Features: Cooperative shelf delivery, collision resolution, communication channels
# Installation: pip install -e 3rd_party/robotic-warehouse/
rware = [
    "gymnasium>=1.1.0",
    "pyglet<2.0.0",
    "networkx>=2.8.0",
]
```

Update `all-envs` bundle (line 381) to include `rware`:
```
"mosaic[box2d,mujoco,atari,minigrid,pettingzoo,vizdoom,nethack,crafter,procgen,textworld,babaisai,mosaic_multigrid,multigrid_ini,jumanji,pybullet-drones,openspiel,meltingpot,overcooked,smac,smacv2,rware]",
```

**Step 5: Commit**

```bash
git add requirements/rware.txt pyproject.toml
git commit -m "feat(rware): add requirements file and pyproject.toml optional dep group"
```

---

### Task 2: Register EnvironmentFamily and GameIds in enums.py

**Files:**
- Modify: `gym_gui/core/enums.py`

**Step 1: Add EnvironmentFamily.RWARE**

Insert before the `OTHER` entry (line 47):

```python
    RWARE = "rware"  # Robotic Warehouse: cooperative multi-agent shelf delivery
```

**Step 2: Add 12 GameId entries**

Insert after the last SMACv2 GameId entry (after SMACV2_ZERG), in the GameId enum:

```python
    # ── RWARE (Robotic Warehouse) ────────────────────────────────────
    RWARE_TINY_2AG = "rware-tiny-2ag-v2"
    RWARE_TINY_4AG = "rware-tiny-4ag-v2"
    RWARE_SMALL_2AG = "rware-small-2ag-v2"
    RWARE_SMALL_4AG = "rware-small-4ag-v2"
    RWARE_MEDIUM_2AG = "rware-medium-2ag-v2"
    RWARE_MEDIUM_4AG = "rware-medium-4ag-v2"
    RWARE_MEDIUM_4AG_EASY = "rware-medium-4ag-easy-v2"
    RWARE_MEDIUM_4AG_HARD = "rware-medium-4ag-hard-v2"
    RWARE_LARGE_4AG = "rware-large-4ag-v2"
    RWARE_LARGE_4AG_HARD = "rware-large-4ag-hard-v2"
    RWARE_LARGE_8AG = "rware-large-8ag-v2"
    RWARE_LARGE_8AG_HARD = "rware-large-8ag-hard-v2"
```

**Step 3: Add ENVIRONMENT_FAMILY_BY_GAME entries**

Insert before the closing `}` of ENVIRONMENT_FAMILY_BY_GAME (before line 1223):

```python
    # RWARE (Robotic Warehouse)
    GameId.RWARE_TINY_2AG: EnvironmentFamily.RWARE,
    GameId.RWARE_TINY_4AG: EnvironmentFamily.RWARE,
    GameId.RWARE_SMALL_2AG: EnvironmentFamily.RWARE,
    GameId.RWARE_SMALL_4AG: EnvironmentFamily.RWARE,
    GameId.RWARE_MEDIUM_2AG: EnvironmentFamily.RWARE,
    GameId.RWARE_MEDIUM_4AG: EnvironmentFamily.RWARE,
    GameId.RWARE_MEDIUM_4AG_EASY: EnvironmentFamily.RWARE,
    GameId.RWARE_MEDIUM_4AG_HARD: EnvironmentFamily.RWARE,
    GameId.RWARE_LARGE_4AG: EnvironmentFamily.RWARE,
    GameId.RWARE_LARGE_4AG_HARD: EnvironmentFamily.RWARE,
    GameId.RWARE_LARGE_8AG: EnvironmentFamily.RWARE,
    GameId.RWARE_LARGE_8AG_HARD: EnvironmentFamily.RWARE,
```

**Step 4: Add DEFAULT_RENDER_MODES entries**

Insert before the closing `}` of DEFAULT_RENDER_MODES (before line 1533):

```python
    # RWARE (Robotic Warehouse)
    GameId.RWARE_TINY_2AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_TINY_4AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_SMALL_2AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_SMALL_4AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_MEDIUM_2AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_MEDIUM_4AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_MEDIUM_4AG_EASY: RenderMode.RGB_ARRAY,
    GameId.RWARE_MEDIUM_4AG_HARD: RenderMode.RGB_ARRAY,
    GameId.RWARE_LARGE_4AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_LARGE_4AG_HARD: RenderMode.RGB_ARRAY,
    GameId.RWARE_LARGE_8AG: RenderMode.RGB_ARRAY,
    GameId.RWARE_LARGE_8AG_HARD: RenderMode.RGB_ARRAY,
```

**Step 5: Add DEFAULT_CONTROL_MODES entries**

Insert before the closing `}` of DEFAULT_CONTROL_MODES (before line 2290):

```python
    # RWARE (Robotic Warehouse) -- human play supported via keyboard
    GameId.RWARE_TINY_2AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_TINY_4AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_SMALL_2AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_SMALL_4AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_MEDIUM_2AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_MEDIUM_4AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_MEDIUM_4AG_EASY: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_MEDIUM_4AG_HARD: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_LARGE_4AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_LARGE_4AG_HARD: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_LARGE_8AG: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.RWARE_LARGE_8AG_HARD: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
```

**Step 6: Add DEFAULT_PARADIGM_BY_FAMILY entry**

Insert before the `OTHER` entry in DEFAULT_PARADIGM_BY_FAMILY (before the OTHER line):

```python
    EnvironmentFamily.RWARE: SteppingParadigm.SIMULTANEOUS,  # RWARE: all robots act in parallel each timestep
```

**Step 7: Verify enum registration**

Run: `source .venv/bin/activate && python -c "from gym_gui.core.enums import EnvironmentFamily, GameId, ENVIRONMENT_FAMILY_BY_GAME; print(EnvironmentFamily.RWARE); print(GameId.RWARE_TINY_2AG); print(ENVIRONMENT_FAMILY_BY_GAME[GameId.RWARE_TINY_2AG])"`
Expected: `rware`, `rware-tiny-2ag-v2`, `EnvironmentFamily.RWARE`

**Step 8: Commit**

```bash
git add gym_gui/core/enums.py
git commit -m "feat(rware): register RWARE family, 12 GameIds, and mapping dicts"
```

---

### Task 3: Add RWARE log constants

**Files:**
- Modify: `gym_gui/logging_config/log_constants.py`

**Step 1: Find the last numbered LOG constant**

Search for the highest-numbered `LOG` constant to determine where LOG970-975 should go.
The SMAC constants use LOG960-LOG969, so LOG970-975 is the correct range.

**Step 2: Add RWARE log constants**

Add after the SMAC log constants block (search for `LOG_SMAC_BATTLE_RESULT` or `LOG969`):

```python
# ── RWARE (Robotic Warehouse) adapter ────────────────────────────────
LOG_RWARE_ENV_CREATED = _constant(
    "LOG970",
    "INFO",
    "RWARE environment created: {map_id}, {n_agents} agents, reward={reward_type}",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "environment", "lifecycle"),
)
LOG_RWARE_ENV_RESET = _constant(
    "LOG971",
    "INFO",
    "RWARE environment reset: step_count={step_count}",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "episode", "lifecycle"),
)
LOG_RWARE_STEP_SUMMARY = _constant(
    "LOG972",
    "DEBUG",
    "RWARE step {step}: rewards={rewards}, done={done}",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "step"),
)
LOG_RWARE_ENV_CLOSED = _constant(
    "LOG973",
    "INFO",
    "RWARE environment closed",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "environment", "lifecycle"),
)
LOG_RWARE_RENDER_ERROR = _constant(
    "LOG974",
    "WARNING",
    "RWARE render error: {error}",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "render", "error"),
)
LOG_RWARE_DELIVERY = _constant(
    "LOG975",
    "INFO",
    "RWARE shelf delivery: agent={agent_id}, reward={reward}",
    component="Adapter",
    subcomponent="RWARE",
    tags=_tags("rware", "delivery", "reward"),
)
```

**Step 3: Commit**

```bash
git add gym_gui/logging_config/log_constants.py
git commit -m "feat(rware): add RWARE log constants LOG970-LOG975"
```

---

### Task 4: Add RWAREConfig dataclass

**Files:**
- Modify: `gym_gui/config/game_configs.py` (after SMACConfig, ~line 935)

**Step 1: Add RWAREConfig dataclass**

Insert after the SMACConfig class (after line ~935, before the GameConfig TypeAlias):

```python
@dataclass
class RWAREConfig:
    """Configuration for Robotic Warehouse (RWARE) multi-agent environments.

    RWARE simulates a warehouse with autonomous robots that cooperatively
    pick up shelves, deliver them to goal workstations, and return them.

    Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL
           Algorithms in Cooperative Tasks"
    Source: 3rd_party/robotic-warehouse/
    """

    # Observation type: "flattened" (1D vector), "dict" (nested dict),
    # "image" (multi-channel grid), "image_dict" (image + dict)
    observation_type: str = "flattened"

    # Sensor range: how many cells each agent can see (1-5)
    sensor_range: int = 1

    # Reward type: "global" (shared), "individual" (per-agent), "two_stage" (delivery + return)
    reward_type: str = "individual"

    # Communication bits per agent (0 = silent, >0 = message channels)
    msg_bits: int = 0

    # Episode limits
    max_steps: int = 500

    # Random seed (None = random)
    seed: int | None = None

    # Render mode (always rgb_array for MOSAIC)
    render_mode: str = "rgb_array"
```

**Step 2: Update GameConfig TypeAlias**

Add `| RWAREConfig` to the GameConfig union (line ~954):

Before:
```python
    | SMACConfig
)
```

After:
```python
    | SMACConfig
    | RWAREConfig
)
```

**Step 3: Verify import**

Run: `source .venv/bin/activate && python -c "from gym_gui.config.game_configs import RWAREConfig; c = RWAREConfig(); print(c.observation_type, c.reward_type, c.sensor_range)"`
Expected: `flattened individual 1`

**Step 4: Commit**

```bash
git add gym_gui/config/game_configs.py
git commit -m "feat(rware): add RWAREConfig dataclass and GameConfig alias"
```

---

### Task 5: Create RWARE adapter

**Files:**
- Create: `gym_gui/core/adapters/rware.py`

**Step 1: Write the adapter file**

Create `gym_gui/core/adapters/rware.py`:

```python
"""Adapter bridging RWARE's Warehouse to MOSAIC's adapter interface.

RWARE (Robotic Warehouse) is a multi-agent cooperative environment where
robots pick up shelves, deliver them to goal workstations, and return them.

Paper: Papoudakis et al. (2021). "Benchmarking Multi-Agent Deep RL Algorithms"
Source: 3rd_party/robotic-warehouse/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np

from gym_gui.core.enums import (
    ControlMode,
    GameId,
    RenderMode,
    SteppingParadigm,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers (rware is optional)
# ---------------------------------------------------------------------------

_Warehouse: type | None = None
_RewardType: type | None = None
_ObservationType: type | None = None


def _ensure_rware() -> None:
    """Import rware lazily (triggers gymnasium.register calls)."""
    global _Warehouse, _RewardType, _ObservationType
    if _Warehouse is not None:
        return
    import rware  # noqa: F401 -- triggers env registration
    from rware.warehouse import ObservationType, RewardType, Warehouse

    _Warehouse = Warehouse
    _RewardType = RewardType
    _ObservationType = ObservationType


# Mapping from config string to rware enum values
_REWARD_TYPE_MAP: Dict[str, int] = {
    "global": 0,
    "individual": 1,
    "two_stage": 2,
}

_OBS_TYPE_MAP: Dict[str, int] = {
    "flattened": 1,
    "dict": 0,
    "image": 2,
    "image_dict": 3,
}


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class RWAREAdapter:
    """Base adapter for Robotic Warehouse environments.

    Subclasses set class-level defaults for warehouse size, agent count,
    and difficulty. The config panel can override observation type, reward
    type, sensor range, and communication bits at runtime.
    """

    # Subclasses override these
    _gym_id: str = "rware-tiny-2ag-v2"
    _default_n_agents: int = 2

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    def __init__(
        self,
        context: Any | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        from gym_gui.config.game_configs import RWAREConfig

        if config is None:
            config = RWAREConfig()
        self._config = config
        self._env: gym.Env | None = None
        self._n_agents: int = self._default_n_agents
        self._step_count: int = 0

    @property
    def stepping_paradigm(self) -> SteppingParadigm:
        return SteppingParadigm.SIMULTANEOUS

    @property
    def action_space(self) -> gym.Space:
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._env.action_space

    @property
    def observation_space(self) -> gym.Space:
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._env.observation_space

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def load(self) -> None:
        """Create the RWARE Warehouse environment."""
        _ensure_rware()

        gym_id = self._gym_id
        _LOGGER.info("Loading RWARE environment: %s", gym_id)

        self._env = gym.make(gym_id, render_mode=self._config.render_mode)

        # Apply config overrides that differ from defaults
        env = self._env.unwrapped
        self._n_agents = env.n_agents

        _LOGGER.info(
            "RWARE environment created: %s, %d agents",
            gym_id,
            self._n_agents,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset the environment and return initial observation."""
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        effective_seed = seed if seed is not None else self._config.seed
        obs_tuple, info = self._env.reset(seed=effective_seed, options=options)
        self._step_count = 0

        return {
            "observations": list(obs_tuple),
            "rewards": [0.0] * self._n_agents,
            "terminated": False,
            "truncated": False,
            "info": info,
        }

    def step(self, action: List[int]) -> dict[str, Any]:
        """Execute one timestep for all agents simultaneously."""
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        obs_tuple, rewards, done, truncated, info = self._env.step(tuple(action))
        self._step_count += 1

        return {
            "observations": list(obs_tuple),
            "rewards": list(rewards),
            "terminated": done,
            "truncated": truncated,
            "info": info,
        }

    def render(self) -> np.ndarray | None:
        """Return an RGB frame from the pyglet renderer."""
        if self._env is None:
            return None
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                return frame
            return None
        except Exception as exc:
            _LOGGER.warning("RWARE render error: %s", exc)
            return None

    def close(self) -> None:
        """Release environment resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        _LOGGER.info("RWARE environment closed")


# ---------------------------------------------------------------------------
# Concrete adapters (one per GameId)
# ---------------------------------------------------------------------------

# Size reference: tiny=(1,3), small=(2,3), medium=(2,5), large=(3,5)
# Format: (shelf_rows, shelf_columns)


class RWARETiny2AgAdapter(RWAREAdapter):
    """Tiny warehouse (1x3 shelves), 2 agents."""
    _gym_id = "rware-tiny-2ag-v2"
    _default_n_agents = 2


class RWARETiny4AgAdapter(RWAREAdapter):
    """Tiny warehouse (1x3 shelves), 4 agents."""
    _gym_id = "rware-tiny-4ag-v2"
    _default_n_agents = 4


class RWARESmall2AgAdapter(RWAREAdapter):
    """Small warehouse (2x3 shelves), 2 agents."""
    _gym_id = "rware-small-2ag-v2"
    _default_n_agents = 2


class RWARESmall4AgAdapter(RWAREAdapter):
    """Small warehouse (2x3 shelves), 4 agents."""
    _gym_id = "rware-small-4ag-v2"
    _default_n_agents = 4


class RWAREMedium2AgAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 2 agents."""
    _gym_id = "rware-medium-2ag-v2"
    _default_n_agents = 2


class RWAREMedium4AgAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents."""
    _gym_id = "rware-medium-4ag-v2"
    _default_n_agents = 4


class RWAREMedium4AgEasyAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents, easy difficulty."""
    _gym_id = "rware-medium-4ag-easy-v2"
    _default_n_agents = 4


class RWAREMedium4AgHardAdapter(RWAREAdapter):
    """Medium warehouse (2x5 shelves), 4 agents, hard difficulty."""
    _gym_id = "rware-medium-4ag-hard-v2"
    _default_n_agents = 4


class RWARELarge4AgAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 4 agents."""
    _gym_id = "rware-large-4ag-v2"
    _default_n_agents = 4


class RWARELarge4AgHardAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 4 agents, hard difficulty."""
    _gym_id = "rware-large-4ag-hard-v2"
    _default_n_agents = 4


class RWARELarge8AgAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 8 agents."""
    _gym_id = "rware-large-8ag-v2"
    _default_n_agents = 8


class RWARELarge8AgHardAdapter(RWAREAdapter):
    """Large warehouse (3x5 shelves), 8 agents, hard difficulty."""
    _gym_id = "rware-large-8ag-hard-v2"
    _default_n_agents = 8


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RWARE_ADAPTERS: Dict[GameId, type[RWAREAdapter]] = {
    GameId.RWARE_TINY_2AG: RWARETiny2AgAdapter,
    GameId.RWARE_TINY_4AG: RWARETiny4AgAdapter,
    GameId.RWARE_SMALL_2AG: RWARESmall2AgAdapter,
    GameId.RWARE_SMALL_4AG: RWARESmall4AgAdapter,
    GameId.RWARE_MEDIUM_2AG: RWAREMedium2AgAdapter,
    GameId.RWARE_MEDIUM_4AG: RWAREMedium4AgAdapter,
    GameId.RWARE_MEDIUM_4AG_EASY: RWAREMedium4AgEasyAdapter,
    GameId.RWARE_MEDIUM_4AG_HARD: RWAREMedium4AgHardAdapter,
    GameId.RWARE_LARGE_4AG: RWARELarge4AgAdapter,
    GameId.RWARE_LARGE_4AG_HARD: RWARELarge4AgHardAdapter,
    GameId.RWARE_LARGE_8AG: RWARELarge8AgAdapter,
    GameId.RWARE_LARGE_8AG_HARD: RWARELarge8AgHardAdapter,
}

ALL_RWARE_GAME_IDS: tuple[GameId, ...] = tuple(RWARE_ADAPTERS.keys())

__all__ = [
    "RWAREAdapter",
    "RWARE_ADAPTERS",
    "ALL_RWARE_GAME_IDS",
]
```

**Step 2: Verify adapter import**

Run: `source .venv/bin/activate && python -c "from gym_gui.core.adapters.rware import RWAREAdapter, RWARE_ADAPTERS; print(f'{len(RWARE_ADAPTERS)} adapters registered'); print(list(RWARE_ADAPTERS.keys())[:3])"`
Expected: `12 adapters registered` and first 3 GameIds printed

**Step 3: Commit**

```bash
git add gym_gui/core/adapters/rware.py
git commit -m "feat(rware): create RWARE adapter with 12 warehouse variants"
```

---

### Task 6: Wire adapter into factory

**Files:**
- Modify: `gym_gui/core/factories/adapters.py` (lines 206-213 for import, 254-256 for registry)

**Step 1: Add RWARE try/except import block**

Insert after the SMACv2 try/except block (after line 213):

```python
try:  # Optional dependency - RWARE (Robotic Warehouse)
    from gym_gui.core.adapters.rware import (  # pragma: no cover - optional
        RWARE_ADAPTERS,
        RWAREAdapter,
    )
except Exception:  # pragma: no cover - rware optional
    RWARE_ADAPTERS: dict[Any, Any] = {}
    RWAREAdapter = None  # type: ignore[misc, assignment]
```

**Step 2: Add RWARE_ADAPTERS to _registry() return**

Insert `**RWARE_ADAPTERS,` after `**SMACV2_ADAPTERS,` in the _registry() return dict (after line 255):

```python
        **RWARE_ADAPTERS,
```

**Step 3: Verify factory wiring**

Run: `source .venv/bin/activate && python -c "from gym_gui.core.factories.adapters import available_games; games = available_games(); rware = [g for g in games if 'rware' in g.value.lower()]; print(f'{len(rware)} RWARE games available')"`
Expected: `12 RWARE games available`

**Step 4: Commit**

```bash
git add gym_gui/core/factories/adapters.py
git commit -m "feat(rware): wire RWARE adapters into factory registry"
```

---

### Task 7: Add dependency detection

**Files:**
- Modify: `gym_gui/app.py` (line ~195 in checks dict)

**Step 1: Add rware to dependency checks**

Insert before the closing `}` of the `checks` dict (before line 196):

```python
        # RWARE: Robotic Warehouse multi-agent cooperative environment
        "rware": "rware",
```

**Step 2: Verify detection**

Run: `source .venv/bin/activate && python -c "from gym_gui.app import _detect_optional_dependencies; deps = _detect_optional_dependencies(); print(f'rware: {deps.get(\"rware\", False)}')"`
Expected: `rware: True`

**Step 3: Commit**

```bash
git add gym_gui/app.py
git commit -m "feat(rware): add rware to optional dependency detection"
```

---

### Task 8: Add config builder branch

**Files:**
- Modify: `gym_gui/config/game_config_builder.py`

**Step 1: Add RWARE game IDs constant**

At the top of the file, after other game ID tuples (around line 38):

```python
_RWARE_GAME_IDS = (
    GameId.RWARE_TINY_2AG,
    GameId.RWARE_TINY_4AG,
    GameId.RWARE_SMALL_2AG,
    GameId.RWARE_SMALL_4AG,
    GameId.RWARE_MEDIUM_2AG,
    GameId.RWARE_MEDIUM_4AG,
    GameId.RWARE_MEDIUM_4AG_EASY,
    GameId.RWARE_MEDIUM_4AG_HARD,
    GameId.RWARE_LARGE_4AG,
    GameId.RWARE_LARGE_4AG_HARD,
    GameId.RWARE_LARGE_8AG,
    GameId.RWARE_LARGE_8AG_HARD,
)
```

**Step 2: Add RWARE branch in build_config()**

Insert a new elif branch before the final return None (before line ~435):

```python
        # RWARE environments (Robotic Warehouse)
        if game_id in _RWARE_GAME_IDS:
            from gym_gui.config.game_configs import RWAREConfig

            observation_type = overrides.get("observation_type", "flattened")
            sensor_range = int(overrides.get("sensor_range", 1))
            reward_type = overrides.get("reward_type", "individual")
            msg_bits = int(overrides.get("msg_bits", 0))
            max_steps = int(overrides.get("max_steps", 500))
            seed_val = overrides.get("seed")
            seed = int(seed_val) if seed_val is not None and int(seed_val) >= 0 else None

            return RWAREConfig(
                observation_type=observation_type,
                sensor_range=sensor_range,
                reward_type=reward_type,
                msg_bits=msg_bits,
                max_steps=max_steps,
                seed=seed,
            )
```

**Step 3: Commit**

```bash
git add gym_gui/config/game_config_builder.py
git commit -m "feat(rware): add RWARE branch in GameConfigBuilder"
```

---

### Task 9: Create game documentation

**Files:**
- Create: `gym_gui/game_docs/RWARE/__init__.py`
- Create: `gym_gui/game_docs/RWARE/_shared.py`
- Create: `gym_gui/game_docs/RWARE/RWARE_Tiny/__init__.py`
- Create: `gym_gui/game_docs/RWARE/RWARE_Small/__init__.py`
- Create: `gym_gui/game_docs/RWARE/RWARE_Medium/__init__.py`
- Create: `gym_gui/game_docs/RWARE/RWARE_Large/__init__.py`
- Modify: `gym_gui/game_docs/__init__.py` (after line 333)

**Step 1: Create _shared.py with common HTML fragments**

Create `gym_gui/game_docs/RWARE/_shared.py` with shared descriptions for actions,
observations, rewards, and environment mechanics. See SMAC's `_shared.py` for pattern.

**Step 2: Create per-size doc modules**

Each module exports HTML constants per GameId (e.g., `RWARE_TINY_2AG_HTML`).
Group by warehouse size since variants within a size share most content.

**Step 3: Create __init__.py re-export module**

`gym_gui/game_docs/RWARE/__init__.py` re-exports all 12 HTML constants.

**Step 4: Register docs in game_docs/__init__.py**

Insert after line 333 (after SMACv2 GAME_INFO block):

```python
# RWARE (Robotic Warehouse) mappings
try:
    from gym_gui.game_docs.RWARE import (
        RWARE_TINY_2AG_HTML,
        RWARE_TINY_4AG_HTML,
        RWARE_SMALL_2AG_HTML,
        RWARE_SMALL_4AG_HTML,
        RWARE_MEDIUM_2AG_HTML,
        RWARE_MEDIUM_4AG_HTML,
        RWARE_MEDIUM_4AG_EASY_HTML,
        RWARE_MEDIUM_4AG_HARD_HTML,
        RWARE_LARGE_4AG_HTML,
        RWARE_LARGE_4AG_HARD_HTML,
        RWARE_LARGE_8AG_HTML,
        RWARE_LARGE_8AG_HARD_HTML,
    )

    GAME_INFO.update({
        GameId.RWARE_TINY_2AG: RWARE_TINY_2AG_HTML,
        GameId.RWARE_TINY_4AG: RWARE_TINY_4AG_HTML,
        GameId.RWARE_SMALL_2AG: RWARE_SMALL_2AG_HTML,
        GameId.RWARE_SMALL_4AG: RWARE_SMALL_4AG_HTML,
        GameId.RWARE_MEDIUM_2AG: RWARE_MEDIUM_2AG_HTML,
        GameId.RWARE_MEDIUM_4AG: RWARE_MEDIUM_4AG_HTML,
        GameId.RWARE_MEDIUM_4AG_EASY: RWARE_MEDIUM_4AG_EASY_HTML,
        GameId.RWARE_MEDIUM_4AG_HARD: RWARE_MEDIUM_4AG_HARD_HTML,
        GameId.RWARE_LARGE_4AG: RWARE_LARGE_4AG_HTML,
        GameId.RWARE_LARGE_4AG_HARD: RWARE_LARGE_4AG_HARD_HTML,
        GameId.RWARE_LARGE_8AG: RWARE_LARGE_8AG_HTML,
        GameId.RWARE_LARGE_8AG_HARD: RWARE_LARGE_8AG_HARD_HTML,
    })
except ImportError:
    pass  # rware docs not available
```

**Step 5: Commit**

```bash
git add gym_gui/game_docs/RWARE/ gym_gui/game_docs/__init__.py
git commit -m "feat(rware): add game documentation for 12 RWARE variants"
```

---

### Task 10: Create config panel

**Files:**
- Create: `gym_gui/ui/config_panels/multi_agent/rware/__init__.py`
- Create: `gym_gui/ui/config_panels/multi_agent/rware/config_panel.py`

**Step 1: Create config_panel.py**

Create `gym_gui/ui/config_panels/multi_agent/rware/config_panel.py` with:
- `ALL_RWARE_GAME_IDS` tuple (all 12 GameIds)
- `_MAP_INFO` dict mapping GameId -> (n_agents, size_label, difficulty)
- `build_rware_controls()` function that creates:
  - Map Info label (auto-populated from GameId)
  - Observation Type: QComboBox (Flattened / Dict / Image / Image+Dict)
  - Sensor Range: QSpinBox (1-5, default 1)
  - Reward Type: QComboBox (Global / Individual / Two-Stage)
  - Communication Bits: QSpinBox (0-8, default 0)
  - Max Steps: QSpinBox (100-10000, default 500)
  - Seed: QSpinBox (-1 for random, 0-99999)
- All widgets emit changes via `callbacks.on_change(overrides_dict)`

Follow the exact pattern from `gym_gui/ui/config_panels/multi_agent/smac/config_panel.py`.

**Step 2: Create __init__.py**

Create `gym_gui/ui/config_panels/multi_agent/rware/__init__.py`:

```python
"""RWARE (Robotic Warehouse) environment configuration panel."""

from gym_gui.ui.config_panels.multi_agent.rware.config_panel import (
    ALL_RWARE_GAME_IDS,
    build_rware_controls,
)

__all__ = [
    "ALL_RWARE_GAME_IDS",
    "build_rware_controls",
]
```

**Step 3: Commit**

```bash
git add gym_gui/ui/config_panels/multi_agent/rware/
git commit -m "feat(rware): create RWARE config panel with obs/reward/sensor controls"
```

---

### Task 11: Wire config panel into control_panel.py

**Files:**
- Modify: `gym_gui/ui/widgets/control_panel.py` (lines 1895-1907)

**Step 1: Add RWARE imports at top of file**

Add with the other config panel imports:

```python
from gym_gui.ui.config_panels.multi_agent.rware import (
    ALL_RWARE_GAME_IDS,
    build_rware_controls,
)
from gym_gui.config.game_configs import RWAREConfig
```

**Step 2: Add RWARE elif branch in _populate_game_config()**

Insert after the SMAC elif block (after line 1907), before the `else:` block:

```python
        elif self._current_game is not None and self._current_game in ALL_RWARE_GAME_IDS:
            current_game = self._current_game
            overrides = self._game_overrides.setdefault(current_game, {})
            build_rware_controls(
                parent=self._config_group,
                layout=self._config_layout,
                game_id=current_game,
                overrides=overrides,
                on_change=self._on_rware_config_changed,
            )
```

**Step 3: Add _on_rware_config_changed callback method**

Add alongside the existing `_on_smac_config_changed`:

```python
    def _on_rware_config_changed(self, overrides: dict[str, Any]) -> None:
        """Handle RWARE config panel changes."""
        if self._current_game is not None:
            self._game_overrides[self._current_game] = overrides
            self._emit_game_changed(self._current_game)
```

**Step 4: Commit**

```bash
git add gym_gui/ui/widgets/control_panel.py
git commit -m "feat(rware): wire RWARE config panel into control_panel.py"
```

---

### Task 12: Add human play keyboard mapping

**Files:**
- Modify: `gym_gui/controllers/human_input.py`

**Step 1: Add _KEY_P and _KEY_L constants if not present**

Check if `_KEY_P` and `_KEY_L` exist. If not, add alongside existing key constants:

```python
_KEY_P = _get_qt_key("Key_P")
_KEY_L = _get_qt_key("Key_L")
_KEY_TAB = _get_qt_key("Key_Tab")
```

**Step 2: Add RWARE key resolver class**

Add a new resolver class following the existing pattern:

```python
class RWAREKeyCombinationResolver(KeyCombinationResolver):
    """Key resolver for RWARE (Robotic Warehouse) environments.

    Actions: NOOP=0, FORWARD=1, LEFT=2, RIGHT=3, TOGGLE_LOAD=4
    Tab: switch active agent
    """

    def resolve(self, pressed_keys: set[int]) -> int | None:
        if _KEY_UP in pressed_keys or _KEY_W in pressed_keys:
            return 1  # FORWARD
        if _KEY_LEFT in pressed_keys or _KEY_A in pressed_keys:
            return 2  # LEFT (rotate)
        if _KEY_RIGHT in pressed_keys or _KEY_D in pressed_keys:
            return 3  # RIGHT (rotate)
        if _KEY_P in pressed_keys or _KEY_L in pressed_keys:
            return 4  # TOGGLE_LOAD
        if _KEY_SPACE in pressed_keys:
            return 0  # NOOP
        return None
```

**Step 3: Register the resolver for RWARE family**

In the resolver factory/dispatch (wherever new families are registered), add RWARE mapping.

**Step 4: Commit**

```bash
git add gym_gui/controllers/human_input.py
git commit -m "feat(rware): add keyboard mapping for RWARE human play"
```

---

### Task 13: Integration test

**Files:**
- Create: `gym_gui/tests/test_rware_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for RWARE adapter in MOSAIC."""

import numpy as np
import pytest


def _rware_available() -> bool:
    try:
        import rware  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _rware_available(), reason="rware not installed"
)


class TestRWAREAdapter:
    """Test RWARE adapter lifecycle."""

    def test_load_and_reset(self):
        from gym_gui.core.adapters.rware import RWARETiny2AgAdapter

        adapter = RWARETiny2AgAdapter()
        adapter.load()

        result = adapter.reset()
        assert "observations" in result
        assert len(result["observations"]) == 2  # 2 agents
        assert result["terminated"] is False
        adapter.close()

    def test_step_returns_correct_shape(self):
        from gym_gui.core.adapters.rware import RWARETiny2AgAdapter

        adapter = RWARETiny2AgAdapter()
        adapter.load()
        adapter.reset()

        # All agents do NOOP
        result = adapter.step([0, 0])
        assert len(result["observations"]) == 2
        assert len(result["rewards"]) == 2
        assert isinstance(result["terminated"], bool)
        assert isinstance(result["truncated"], bool)
        adapter.close()

    def test_render_returns_rgb_array(self):
        from gym_gui.core.adapters.rware import RWARETiny2AgAdapter

        adapter = RWARETiny2AgAdapter()
        adapter.load()
        adapter.reset()

        frame = adapter.render()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        adapter.close()

    def test_all_12_variants_registered(self):
        from gym_gui.core.adapters.rware import RWARE_ADAPTERS

        assert len(RWARE_ADAPTERS) == 12

    def test_factory_can_find_rware(self):
        from gym_gui.core.factories.adapters import available_games

        games = available_games()
        rware_games = [g for g in games if "rware" in g.value.lower()]
        assert len(rware_games) == 12


class TestRWAREEnums:
    """Test RWARE enum registration."""

    def test_environment_family_exists(self):
        from gym_gui.core.enums import EnvironmentFamily

        assert EnvironmentFamily.RWARE.value == "rware"

    def test_all_game_ids_mapped(self):
        from gym_gui.core.enums import (
            ENVIRONMENT_FAMILY_BY_GAME,
            EnvironmentFamily,
            GameId,
        )

        rware_ids = [g for g in GameId if g.value.startswith("rware-")]
        assert len(rware_ids) == 12
        for gid in rware_ids:
            assert ENVIRONMENT_FAMILY_BY_GAME[gid] == EnvironmentFamily.RWARE
```

**Step 2: Run tests**

Run: `source .venv/bin/activate && python -m pytest gym_gui/tests/test_rware_integration.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add gym_gui/tests/test_rware_integration.py
git commit -m "test(rware): add integration tests for RWARE adapter"
```

---

## Execution Order

Tasks 1-4 are foundation (enums, config, log constants) -- do these first.
Task 5 (adapter) depends on Tasks 1-4.
Task 6 (factory) depends on Task 5.
Task 7 (app.py) is independent -- can run in parallel with Task 5.
Task 8 (config builder) depends on Task 4.
Task 9 (game docs) is independent -- can run in parallel.
Task 10 (config panel) depends on Task 4.
Task 11 (control_panel wiring) depends on Task 10.
Task 12 (human input) is independent.
Task 13 (integration test) depends on all prior tasks.

```
Task 1 (requirements/pyproject) ─┐
Task 2 (enums)                   ├─► Task 5 (adapter) ─► Task 6 (factory)
Task 3 (log constants)           │
Task 4 (config)                  ─┼─► Task 8 (config builder)
                                  ├─► Task 10 (config panel) ─► Task 11 (control_panel)
Task 7 (app.py)         ─────────┘
Task 9 (game docs)      ─────────── (independent)
Task 12 (human input)   ─────────── (independent)
Task 13 (integration test) ◄─────── (depends on all)
```
