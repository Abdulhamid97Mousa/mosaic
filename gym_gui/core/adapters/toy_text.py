"""Toy-text environment adapters for the Gym GUI."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any, List, Sequence, Tuple, Type

import gymnasium as gym
import random

from gym_gui.config.game_configs import (
    FrozenLakeConfig,
    TaxiConfig,
    CliffWalkingConfig,
    BlackjackConfig,
    DEFAULT_FROZEN_LAKE_CONFIG,
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
    DEFAULT_TAXI_CONFIG,
    DEFAULT_CLIFF_WALKING_CONFIG,
    DEFAULT_BLACKJACK_CONFIG,
)
from gym_gui.config.paths import VAR_DATA_DIR
from gym_gui.core.adapters.base import AdapterContext, EnvironmentAdapter, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.constants.constants_game import (
    ToyTextDefaults,
    TOY_TEXT_DEFAULTS,
    FROZEN_LAKE_DEFAULTS,
    FROZEN_LAKE_V2_DEFAULTS,
    CLIFF_WALKING_DEFAULTS,
    TAXI_DEFAULTS,
    BLACKJACK_DEFAULTS,
)
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ADAPTER_INIT_ERROR,
    LOG_ADAPTER_STEP_ERROR,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_STATE_INVALID,
    LOG_ADAPTER_MAP_GENERATION,
    LOG_ADAPTER_HOLE_PLACEMENT,
    LOG_ADAPTER_GOAL_OVERRIDE,
    LOG_ADAPTER_RENDER_PAYLOAD,
)

_TOY_TEXT_DATA_DIR = (VAR_DATA_DIR / "toy_text").resolve()
_TOY_TEXT_DATA_DIR.mkdir(parents=True, exist_ok=True)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[(?P<codes>[0-9;]*)m")
_AGENT_TOKEN_HINTS = {"x"}



def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _ansi_to_grid(ansi: str) -> Tuple[List[List[str]], Tuple[int, int] | None]:
    rows: List[List[str]] = []
    agent_pos: Tuple[int, int] | None = None

    for raw_line in ansi.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("("):
            continue

        row_chars: List[str] = []
        i = 0
        current_bg: int | None = None
        while i < len(raw_line):
            char = raw_line[i]
            if char == "\x1b":
                match = _ANSI_ESCAPE_RE.match(raw_line, i)
                if match:
                    codes = match.group("codes") or ""
                    if not codes:
                        current_bg = None
                    else:
                        for code in codes.split(";"):
                            if not code:
                                continue
                            if code == "0":
                                current_bg = None
                                continue
                            try:
                                value = int(code)
                            except ValueError:
                                continue
                            if 40 <= value <= 47 or 100 <= value <= 107:
                                current_bg = value
                    i = match.end()
                    continue
            if char == " ":
                # Preserve spacing so the grid matches the rendered layout and can be highlighted.
                row_chars.append(" ")
                if current_bg is not None:
                    agent_pos = (len(rows), len(row_chars) - 1)
            else:
                row_chars.append(char)
                if current_bg is not None:
                    agent_pos = (len(rows), len(row_chars) - 1)
            i += 1
        if row_chars:
            rows.append(row_chars)

    if agent_pos is None:
        hint = _find_agent_token(rows)
        if hint is not None:
            agent_pos = hint

    return rows, agent_pos


def _find_agent_token(grid: List[List[str]]) -> Tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if not value:
                continue
            if value.strip() == "":
                continue
            if value.lower() in _AGENT_TOKEN_HINTS:
                return r, c
    return None


def _coalesce(value: Any, fallback: Any) -> Any:
    """Return fallback only when value is None; respects falsy but valid values."""
    return fallback if value is None else value


class ToyTextAdapter(EnvironmentAdapter[int, int]):
    """Base adapter for Gymnasium toy-text environments."""

    default_render_mode = RenderMode.GRID
    supported_render_modes = (RenderMode.GRID,)
    _gym_render_mode = "ansi"

    # Subclasses can override with their canonical defaults.
    toy_text_defaults: ToyTextDefaults | None = None

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        defaults: ToyTextDefaults | None = None,
    ) -> None:
        """Initialize toy-text adapter with step tracking."""
        super().__init__(context)
        self._defaults = self._resolve_defaults(defaults)
        self._last_terminated: bool = False
        self._last_truncated: bool = False

    # ------------------------------------------------------------------
    def _resolve_defaults(self, override: ToyTextDefaults | None) -> ToyTextDefaults:
        if override is not None:
            return override
        if self.toy_text_defaults is not None:
            return self.toy_text_defaults
        try:
            game_id = GameId(self.id)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"No GameId match for adapter id '{self.id}'") from exc
        try:
            return TOY_TEXT_DEFAULTS[game_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"No toy-text defaults registered for {game_id}") from exc

    @property
    def defaults(self) -> ToyTextDefaults:
        return self._defaults

    def _get_grid_width(self) -> int:
        """Get grid width from environment.
        
        Returns the number of columns in the grid. Override in subclasses
        if special handling is needed (e.g., CliffWalking always 12).
        
        Returns:
            Grid width (ncol). Defaults to 8 if unable to determine.
        """
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        
        # Try ncol attribute first (FrozenLake)
        if hasattr(unwrapped, "ncol"):
            return int(getattr(unwrapped, "ncol"))
        
        # Try desc attribute (map descriptor)
        if hasattr(unwrapped, "desc"):
            desc = getattr(unwrapped, "desc")
            if desc and len(desc) > 0:
                return len(desc[0])
        
        # Fallback to canonical defaults
        return self.defaults.grid_width

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """Convert state (single integer) to grid position (row, col).
        
        Args:
            state: Integer state from environment (0 to ncol*nrow - 1)
            
        Returns:
            Tuple of (row, col) grid indices (0-indexed)
        """
        width = self._get_grid_width()
        row = int(state) // width
        col = int(state) % width
        return (row, col)

    def pos_to_state(self, x: int, y: int) -> int:
        """Convert grid position (col, row) to state (single integer).
        
        Args:
            x: Column index (0-indexed)
            y: Row index (0-indexed)
            
        Returns:
            Integer state representation
        """
        width = self._get_grid_width()
        return y * width + x

    def load(self) -> None:
        kwargs = self.gym_kwargs()
        env = gym.make(self.id, render_mode=self._gym_render_mode, **kwargs)
        self.log_constant(
            LOG_ADAPTER_ENV_CREATED,
            message="toy_text_env_loaded",
            extra={
                "env_id": self.id,
                "render_mode": self._gym_render_mode,
                "kwargs": ",".join(sorted(kwargs.keys())) if kwargs else "-",
            },
        )
        self._set_env(env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear step tracking."""
        self._last_terminated = False
        self._last_truncated = False
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Step and track terminated/truncated states."""
        result = super().step(action)
        self._last_terminated = bool(result.terminated)
        self._last_truncated = bool(result.truncated)

        payload = result.render_payload
        if isinstance(payload, dict):
            enriched = dict(payload)
            enriched["terminated"] = self._last_terminated
            enriched["truncated"] = self._last_truncated
            enriched.setdefault("game_id", self.id)
            result.render_payload = enriched

        return result

    def render(self) -> dict[str, Any]:
        env = self._require_env()
        ansi_raw = env.render()
        ansi = str(ansi_raw)
        grid, agent_pos = _ansi_to_grid(ansi)
        env_position = self._agent_position_from_state(grid)
        if env_position is not None:
            agent_pos = env_position
        snapshot_path = _TOY_TEXT_DATA_DIR / f"{self.id}.txt"
        snapshot_path.write_text(ansi, encoding="utf-8")
        
        payload = {
            "mode": RenderMode.GRID.value,
            "grid": grid,
            "ansi": ansi,
            "snapshot_path": str(snapshot_path),
            "agent_position": agent_pos,
            "game_id": self.id,
        }

        # Add FrozenLake-specific info (holes and goal positions)
        if self.id in (GameId.FROZEN_LAKE.value, GameId.FROZEN_LAKE_V2.value):
            unwrapped = getattr(env, "unwrapped", env)
            desc = getattr(unwrapped, "desc", None)
            if desc is not None:
                # Extract holes and goal positions from the environment descriptor
                holes = []
                goal = None
                try:
                    for r, row in enumerate(desc):
                        for c, cell in enumerate(row):
                            if isinstance(cell, bytes):
                                cell_char = cell.decode('utf-8')
                            else:
                                cell_char = str(cell)
                            if cell_char == 'H':
                                holes.append({"row": int(r), "col": int(c)})
                            elif cell_char == 'G':
                                goal = {"row": int(r), "col": int(c)}
                except Exception:
                    pass

                if holes:
                    payload["holes"] = holes
                if goal:
                    payload["goal"] = goal
                # Add grid dimensions
                payload["grid_size"] = {"height": len(desc), "width": len(desc[0]) if len(desc) > 0 else 0}

        # Add Taxi-specific state info
        if self.id == GameId.TAXI.value:
            unwrapped = getattr(env, "unwrapped", env)
            state = getattr(unwrapped, "s", None)
            decode = getattr(unwrapped, "decode", None)
            if state is not None and callable(decode):
                raw_decoded = decode(int(state))
                decoded = _ensure_sequence(raw_decoded)
                if len(decoded) >= 4:
                    taxi_row = decoded[0]
                    taxi_col = decoded[1]
                    pass_idx = decoded[2]
                    dest_idx = decoded[3]
                    payload["taxi_state"] = {
                        "taxi_position": (int(taxi_row), int(taxi_col)),
                        "passenger_index": int(pass_idx),  # 0-3 = at depot R/G/Y/B, 4 = in taxi
                        "destination_index": int(dest_idx),  # 0-3 = depot R/G/Y/B
                    }
        
        return payload

    def build_frame_reference(self, render_payload: Any | None, state: StepState) -> str | None:
        """Generate timestamped frame reference for media storage.

        Args:
            render_payload: The render payload (unused for toy-text)
            state: The step state (unused for toy-text)

        Returns:
            Timestamped frame reference string or None if payload is invalid
        """
        if render_payload is None:
            return None

        # Generate timestamp: YYYY-MM-DD_HH-MM-SS_NNN
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        microseconds = now.microsecond // 1000  # Convert to milliseconds

        return f"frames/{timestamp}_{microseconds:03d}.png"

    def gym_kwargs(self) -> dict[str, Any]:
        return {}

    def _agent_position_from_state(self, grid: List[List[str]]) -> Tuple[int, int] | None:
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        try:
            state = getattr(unwrapped, "s", None)
            width: int | None = None
            height: int | None = None
            row: int | None = None
            col: int | None = None

            if self.id == GameId.TAXI.value:
                decode = getattr(unwrapped, "decode", None)
                if state is None or not callable(decode):
                    return None
                raw_decoded = decode(int(state))
                decoded = _ensure_sequence(raw_decoded)
                if len(decoded) < 2:
                    return None
                taxi_row = decoded[0]
                taxi_col = decoded[1]
                row = int(taxi_row)
                col = int(taxi_col)
                height = self.defaults.grid_height
                width = self.defaults.grid_width
            else:
                if state is None:
                    return None
                if hasattr(unwrapped, "ncol"):
                    width = int(getattr(unwrapped, "ncol"))
                if hasattr(unwrapped, "nrow"):
                    height = int(getattr(unwrapped, "nrow"))
                if (width is None or height is None) and hasattr(unwrapped, "desc"):
                    desc = getattr(unwrapped, "desc")
                    try:
                        height = height or len(desc)
                        width = width or (len(desc[0]) if height else None)
                    except Exception:
                        pass
                if (width is None or height is None) and hasattr(unwrapped, "shape"):
                    shape = getattr(unwrapped, "shape")
                    if shape and len(shape) >= 2:
                        if height is None:
                            height = int(shape[0])
                        if width is None:
                            width = int(shape[1])
                if width is None or height is None:
                    return None
                row = int(state) // width
                col = int(state) % width

            if row is None or col is None or not grid:
                return None
            grid_row = min(max(row, 0), len(grid) - 1)
            row_chars = grid[grid_row]
            if not row_chars:
                return None
            if width <= 1:
                grid_col = 0
            else:
                span = len(row_chars) - 1
                base = width - 1
                if base <= 0:
                    grid_col = 0
                else:
                    ratio = col / base
                    grid_col = int(round(ratio * span))
            grid_col = min(max(grid_col, 0), len(row_chars) - 1)
            return grid_row, grid_col
        except Exception:
            return None


class FrozenLakeAdapter(ToyTextAdapter):
    """Adapter for FrozenLake environment with game-specific configuration."""
    
    id = GameId.FROZEN_LAKE.value
    toy_text_defaults = FROZEN_LAKE_DEFAULTS

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: FrozenLakeConfig | dict | None = None,
        map_size: str | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context, defaults=self.toy_text_defaults)
        # Handle dict game_config (from worker) by converting to FrozenLakeConfig
        if isinstance(game_config, dict):
            self._game_config = FrozenLakeConfig(**game_config)
        else:
            self._game_config = game_config or DEFAULT_FROZEN_LAKE_CONFIG
        self._last_action: int | None = None
        self.map_size = map_size

        if map_size:
            normalized = map_size.strip().lower()
            if normalized not in {"4x4", "8x8"}:
                raise ValueError(f"Unsupported map_size '{map_size}'. Expected '4x4' or '8x8'.")
            grid_dim = 4 if normalized == "4x4" else 8
            # Rebuild config with standardized dimensions and positions
            base = self._game_config
            self._game_config = FrozenLakeConfig(
                is_slippery=base.is_slippery,
                success_rate=base.success_rate,
                reward_schedule=base.reward_schedule,
                grid_height=grid_dim,
                grid_width=grid_dim,
                start_position=(0, 0),
                goal_position=(grid_dim - 1, grid_dim - 1),
                hole_count=base.hole_count,
                random_holes=base.random_holes,
            )

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset state and clear the last action tracker."""
        self._last_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Record the most recent action for orientation-aware rendering."""
        self._last_action = int(action)
        return super().step(action)

    def render(self) -> dict[str, Any]:
        """Render with FrozenLake-specific terminated state."""
        payload = super().render()
        
        # Add terminated state for cracked_hole visualization
        payload["terminated"] = self._last_terminated
        payload["truncated"] = self._last_truncated
        payload["last_action"] = self._last_action
        
        return payload

    def set_goal(self, x: int, y: int) -> None:
        """Store goal position for BDI planning.
        
        For dynamic goal scenarios in BDI. Stores the goal but may not
        reconfigure the environment (depends on environment implementation).
        
        Args:
            x: Goal column (0-indexed)
            y: Goal row (0-indexed)
        """
        self._goal_position = (x, y)
        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            message="frozenlake_goal_set",
            extra={"env_id": self.id, "goal_x": x, "goal_y": y},
        )

    def reset_q_table(self) -> None:
        """Reset Q-table for goal-switching scenarios.
        
        Called by BDI when switching goals to clear learned policy.
        The actual Q-table is managed by the RL agent/trainer, not the adapter.
        This is a hook for coordination.
        """
        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            message="frozenlake_qtable_reset_requested",
            extra={"env_id": self.id},
        )

    def get_grid_size(self) -> tuple[int, int]:
        """Get (height, width) of the FrozenLake grid.
        
        Required for BDI pathfinding and goal validation.
        
        Returns:
            Tuple of (height, width) dimensions
        """
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        height, width = None, None
        
        if hasattr(unwrapped, "nrow"):
            height = int(getattr(unwrapped, "nrow"))
        if hasattr(unwrapped, "ncol"):
            width = int(getattr(unwrapped, "ncol"))
        
        if hasattr(unwrapped, "desc"):
            desc = getattr(unwrapped, "desc")
            if desc:
                height = height or len(desc)
                width = width or (len(desc[0]) if len(desc) > 0 else 0)
        
        fallback_height = self.defaults.grid_height
        fallback_width = self.defaults.grid_width
        return (height or fallback_height, width or fallback_width)


class FrozenLakeV2Adapter(ToyTextAdapter):
    """Adapter for FrozenLake-v2 with configurable grid, start, goal, and hole count."""
    
    id = GameId.FROZEN_LAKE_V2.value
    toy_text_defaults = FROZEN_LAKE_V2_DEFAULTS

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: FrozenLakeConfig | dict | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context, defaults=self.toy_text_defaults)
        
        # Convert dictionary to FrozenLakeConfig if needed
        if isinstance(game_config, dict):
            game_config = FrozenLakeConfig(**game_config)
        
        self._game_config = game_config or DEFAULT_FROZEN_LAKE_V2_CONFIG
        self._last_action: int | None = None
        self._custom_desc: list[str] | None = None
        self._resolved_start: tuple[int, int] | None = None
        self._resolved_goal: tuple[int, int] | None = None

    def _generate_map_descriptor(self) -> list[str]:
        """Generate custom map descriptor based on configuration.
        
        If random_holes=False and using standard 4×4 or 8×8 grid with default positions,
        returns the official Gymnasium default map. Otherwise generates a custom map.
        """

        defaults = self.defaults
        height = _coalesce(self._game_config.grid_height, defaults.grid_height)
        width = _coalesce(self._game_config.grid_width, defaults.grid_width)
        height = max(1, int(height))
        width = max(1, int(width))

        start_pos = _coalesce(self._game_config.start_position, defaults.start)
        goal_pos = _coalesce(self._game_config.goal_position, defaults.goal)

        def _normalize(position: tuple[int, int] | None, fallback: tuple[int, int]) -> tuple[int, int]:
            base = fallback
            if position is not None:
                try:
                    row = int(position[0])
                    col = int(position[1])
                except (TypeError, ValueError, IndexError):
                    row, col = base
                else:
                    row = max(0, min(height - 1, row))
                    col = max(0, min(width - 1, col))
                    return (row, col)
            row, col = base
            row = max(0, min(height - 1, int(row)))
            col = max(0, min(width - 1, int(col)))
            return (row, col)

        start_pos = _normalize(start_pos, (0, 0))
        goal_pos = _normalize(goal_pos, (height - 1, width - 1))

        hole_count = _coalesce(self._game_config.hole_count, defaults.hole_count)
        random_holes = (
            self._game_config.random_holes
            if self._game_config.random_holes is not None
            else defaults.random_holes
        )

        # Use official Gymnasium maps if conditions match exactly
        if (
            not random_holes
            and defaults.official_map
            and height == defaults.grid_height
            and width == defaults.grid_width
            and start_pos == defaults.start
            and goal_pos == defaults.goal
        ):
            self._resolved_start = start_pos
            self._resolved_goal = goal_pos
            return list(defaults.official_map)
        
        # Generate custom map (random holes or custom start/goal positions)
        # Default hole count if not specified
        if hole_count is None:
            total_tiles = height * width
            if defaults.hole_count is not None:
                hole_count = defaults.hole_count
            else:
                # Scale holes proportionally
                hole_count = max(1, int(total_tiles * 0.15))  # ~15% holes
        
        # Initialize grid with frozen tiles
        self._resolved_start = start_pos
        self._resolved_goal = goal_pos

        grid = [['F' for _ in range(width)] for _ in range(height)]

        # Place start and goal
        grid[start_pos[0]][start_pos[1]] = 'S'
        grid[goal_pos[0]][goal_pos[1]] = 'G'
        
        # Collect available positions for holes (exclude start and goal)
        available_positions = [
            (r, c) for r in range(height) for c in range(width)
            if (r, c) != start_pos and (r, c) != goal_pos
        ]
        
        hole_count = min(hole_count, len(available_positions))
        
        if random_holes:
            # RANDOM: Randomly place holes across the entire grid
            hole_positions = random.sample(available_positions, hole_count)
        else:
            # DETERMINISTIC: Use official map hole pattern (if same grid size)
            hole_positions = []
            if defaults.official_map and height == defaults.grid_height and width == defaults.grid_width:
                # Extract hole positions from official Gymnasium map
                for r, row in enumerate(defaults.official_map):
                    for c, cell in enumerate(row):
                        if cell == 'H':
                            # Only use this hole if it doesn't conflict with custom start/goal
                            if (r, c) not in [start_pos, goal_pos]:
                                hole_positions.append((r, c))
                # Use exactly the hole_count requested (trim or keep all official holes)
                hole_positions = hole_positions[:hole_count]
            else:
                # Different grid size or no official map - use first N positions as fallback
                # This is only used for non-standard grids (e.g., 6x6, 10x10)
                hole_positions = available_positions[:hole_count]
        
        # Log detailed hole placement configuration
        self.log_constant(
            LOG_ADAPTER_HOLE_PLACEMENT,
            message="FrozenLake hole placement configuration",
            extra={
                "random_holes": random_holes,
                "hole_count": hole_count,
                "grid_size": f"{height}x{width}",
                "total_available_positions": len(available_positions),
                "hole_positions": hole_positions,
                "start_pos": start_pos,
                "goal_pos": goal_pos,
            },
        )
        
        for r, c in hole_positions:
            grid[r][c] = 'H'
        
        # Convert to list of strings
        return [''.join(row) for row in grid]

    def load(self) -> None:
        """Load with custom map descriptor."""
        self._custom_desc = self._generate_map_descriptor()
        kwargs = {
            "is_slippery": self._game_config.is_slippery,
            "success_rate": self._game_config.success_rate,
            "reward_schedule": self._game_config.reward_schedule,
            "desc": self._custom_desc,
        }
        # Use FrozenLake8x8-v1 (the 8x8 variant available in Gymnasium)
        # This adapter is designed for larger customizable grids, hence the v2 naming in our code
        env = gym.make("FrozenLake8x8-v1", render_mode=self._gym_render_mode, **kwargs)
        
        # Log complete map configuration
        self.log_constant(
            LOG_ADAPTER_MAP_GENERATION,
            message="FrozenLake-v2 map loaded with custom configuration",
            extra={
                "env_id": "FrozenLake8x8-v1",
                "grid_height": self._game_config.grid_height,
                "grid_width": self._game_config.grid_width,
                "start_position": self._game_config.start_position,
                "goal_position": self._game_config.goal_position,
                "hole_count": self._game_config.hole_count,
                "random_holes": self._game_config.random_holes,
                "is_slippery": self._game_config.is_slippery,
                "map_descriptor": self._custom_desc,
            },
        )
        self._set_env(env)

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs (handled in load)."""
        return {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset state and clear the last action tracker."""
        self._last_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        """Record the most recent action for orientation-aware rendering."""
        self._last_action = int(action)
        return super().step(action)

    def render(self) -> dict[str, Any]:
        """Render with FrozenLake-specific terminated state."""
        payload = super().render()
        
        # Add terminated state for cracked_hole visualization
        payload["terminated"] = self._last_terminated
        payload["truncated"] = self._last_truncated
        payload["last_action"] = self._last_action
        
        # Log render payload details (for debugging visualization issues)
        self.log_constant(
            LOG_ADAPTER_RENDER_PAYLOAD,
            message="FrozenLake-v2 render payload generated",
            extra={
                "has_holes": "holes" in payload,
                "hole_count": len(payload.get("holes", [])),
                "has_goal": "goal" in payload,
                "goal_position": payload.get("goal"),
                "agent_position": payload.get("agent_position"),
                "grid_size": payload.get("grid_size"),
                "terminated": self._last_terminated,
            },
        )
        
        return payload

    def goal_pos(self) -> tuple[int, int] | None:
        """Return the resolved goal position for worker compatibility."""
        return self._resolved_goal

    @staticmethod
    def get_available_positions(grid_height: int, grid_width: int, start_position: tuple[int, int], exclude_holes: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        """Get list of valid positions for goal selection (excludes start and existing holes)."""
        exclude_holes = exclude_holes or []
        excluded_set = {start_position} | set(exclude_holes)
        
        return [
            (r, c) for r in range(grid_height) for c in range(grid_width)
            if (r, c) not in excluded_set
        ]


class CliffWalkingAdapter(ToyTextAdapter):
    """Adapter for CliffWalking environment with game-specific configuration."""
    
    id = GameId.CLIFF_WALKING.value
    toy_text_defaults = CLIFF_WALKING_DEFAULTS

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: CliffWalkingConfig | dict | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context, defaults=self.toy_text_defaults)
        
        # Convert dictionary to CliffWalkingConfig if needed
        if isinstance(game_config, dict):
            game_config = CliffWalkingConfig(**game_config)
        
        self._game_config = game_config or DEFAULT_CLIFF_WALKING_CONFIG
        self._last_action: int | None = None

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear last action."""
        self._last_action = None
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int):
        """Step and track the action for rendering."""
        self._last_action = int(action)
        return super().step(action)
    
    def _agent_position_from_state(self, grid: List[List[str]]) -> Tuple[int, int] | None:
        """Override to correctly calculate position for CliffWalking's 4x12 grid."""
        env = self._require_env()
        unwrapped = getattr(env, "unwrapped", env)
        try:
            state = getattr(unwrapped, "s", None)
            if state is None:
                return None
            
            # CliffWalking is 4 rows × 12 columns
            width = self.defaults.grid_width
            
            # State is a single integer from 0-47
            row = int(state) // width
            col = int(state) % width
            
            return (row, col)
        except Exception:
            return None
    
    def render(self) -> dict[str, Any]:
        """Override render to strip spacing columns from CliffWalking ANSI grid.
        
        CliffWalking ANSI format includes spacing between cells: 'o  o  o...'
        This creates a 34-column grid (['o', ' ', ' ', 'o', ' ', ' ', ...])
        We need to remove the spacing columns to get a clean 12-column grid.
        """
        # Get the base render payload with the raw grid
        payload = super().render()
        
        # Extract the grid with spacing
        grid_with_spacing = payload.get("grid", [])
        if not grid_with_spacing:
            return payload
        
        # Strip spacing columns: keep only columns 0, 3, 6, 9, ... (every 3rd)
        clean_grid = []
        for row in grid_with_spacing:
            clean_row = [row[col] for col in range(len(row)) if col % 3 == 0]
            clean_grid.append(clean_row)
        
        # Update payload with clean grid
        payload["grid"] = clean_grid
        
        # Agent position was already calculated correctly by our override above
        # (using 4x12 dimensions), so no adjustment needed
        
        # Add last action for directional elf rendering
        payload["last_action"] = self._last_action
        
        return payload


class TaxiAdapter(ToyTextAdapter):
    """Adapter for Taxi-v3 environment with game-specific configuration."""
    
    id = GameId.TAXI.value
    toy_text_defaults = TAXI_DEFAULTS
    
    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: TaxiConfig | dict | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context, defaults=self.toy_text_defaults)
        
        # Convert dictionary to TaxiConfig if needed
        if isinstance(game_config, dict):
            game_config = TaxiConfig(**game_config)
        
        self._game_config = game_config or DEFAULT_TAXI_CONFIG
        self._last_action: int | None = None

    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset and clear last action."""
        self._last_action = None
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int):
        """Step and track the action for rendering."""
        self._last_action = int(action)
        return super().step(action)
    
    def render(self) -> dict[str, Any]:
        """Override render to convert taxi position from 5×5 logical to 11×11 grid coordinates."""
        # Get the base render payload with full 11×11 ANSI grid
        payload = super().render()
        
        # Taxi uses an 11×11 character grid with borders
        # The taxi_state has taxi_position in 5×5 logical coordinates
        # We need to convert it to 11×11 grid coordinates
        
        taxi_state = payload.get("taxi_state")
        if taxi_state and "taxi_position" in taxi_state:
            logical_row, logical_col = taxi_state["taxi_position"]
            
            # Convert 5×5 logical coordinates to 11×11 grid coordinates
            # Logical row N → grid row (N + 1)
            # Logical col N → grid col (N * 2 + 1)
            # This accounts for borders and spacing in the ANSI grid
            grid_row = logical_row + 1  # Skip top border row
            grid_col = logical_col * 2 + 1  # Account for spacing (each cell takes 2 chars)
            
            payload["agent_position"] = (grid_row, grid_col)
            # Add last action for directional cab rendering
            taxi_state["last_action"] = self._last_action
        
        return payload


class BlackjackAdapter(ToyTextAdapter):
    """Adapter for Blackjack environment with pygame-based card rendering."""
    
    id = GameId.BLACKJACK.value
    toy_text_defaults = BLACKJACK_DEFAULTS
    _gym_render_mode = "rgb_array"  # Override: use pygame rendering instead of ANSI
    
    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )
    
    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        game_config: BlackjackConfig | dict | None = None,
    ) -> None:
        """Initialize with optional game-specific configuration."""
        super().__init__(context, defaults=self.toy_text_defaults)
        
        # Convert dictionary to BlackjackConfig if needed
        if isinstance(game_config, dict):
            game_config = BlackjackConfig(**game_config)
        
        self._game_config = game_config or DEFAULT_BLACKJACK_CONFIG
    
    def gym_kwargs(self) -> dict[str, Any]:
        """Return Gymnasium environment kwargs from game configuration."""
        return self._game_config.to_gym_kwargs()
    
    def render(self) -> dict[str, Any]:
        """Render Blackjack game state using pygame card display.
        
        Returns RGB array from pygame renderer along with game state information.
        """
        env = self._require_env()
        
        # Get RGB array from pygame renderer (returns numpy array H×W×3)
        rgb_array = env.render()
        
        # Extract current game state from environment
        unwrapped = getattr(env, "unwrapped", env)
        player_sum, dealer_card, usable_ace = None, None, None
        
        if hasattr(unwrapped, 'player') and hasattr(unwrapped, 'dealer'):
            # Import helper functions from blackjack module
            try:
                from gymnasium.envs.toy_text.blackjack import sum_hand, usable_ace as check_usable_ace
                # Type checker doesn't know about BlackjackEnv's player/dealer attributes
                player_hand = getattr(unwrapped, 'player')  # type: ignore[attr-defined]
                dealer_hand = getattr(unwrapped, 'dealer')  # type: ignore[attr-defined]
                player_sum = sum_hand(player_hand)
                dealer_card = dealer_hand[0] if dealer_hand else None
                usable_ace = check_usable_ace(player_hand)
            except (ImportError, AttributeError):
                pass
        
        # Build formatted state description for UI display
        state_lines = []
        if player_sum is not None:
            state_lines.append(f"Player Sum: {player_sum}")
        if dealer_card is not None:
            state_lines.append(f"Dealer Showing: {dealer_card}")
        if usable_ace is not None:
            state_lines.append(f"Usable Ace: {'Yes' if usable_ace else 'No'}")
        
        payload = {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": rgb_array,  # Use "rgb" key to match RgbRendererStrategy
            "game_id": self.id,
            "player_sum": player_sum,
            "dealer_card": dealer_card,
            "usable_ace": bool(usable_ace) if usable_ace is not None else None,
            "terminated": self._last_terminated,
            "truncated": self._last_truncated,
            "state_description": "\n".join(state_lines) if state_lines else None,
        }
        
        return payload


TOY_TEXT_ADAPTERS: dict[GameId, Type[ToyTextAdapter]] = {
    GameId.FROZEN_LAKE: FrozenLakeAdapter,
    GameId.FROZEN_LAKE_V2: FrozenLakeV2Adapter,
    GameId.CLIFF_WALKING: CliffWalkingAdapter,
    GameId.TAXI: TaxiAdapter,
    GameId.BLACKJACK: BlackjackAdapter,
}

__all__ = [
    "ToyTextAdapter",
    "FrozenLakeAdapter",
    "FrozenLakeV2Adapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "BlackjackAdapter",
    "TOY_TEXT_ADAPTERS",
]
