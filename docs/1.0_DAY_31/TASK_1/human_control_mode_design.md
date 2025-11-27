# ViZDoom Human Control Mode Design

## Status: IMPLEMENTED

This document details the implemented design for Human Control Mode in ViZDoom through gym_gui.

## Architecture Overview

### Qt-Based Input (Implemented Solution)

Instead of using ViZDoom's native window with SPECTATOR mode, we implemented a Qt-based input system that:
- Renders ViZDoom in gym_gui's Video tab (no popup window)
- Captures keyboard via Qt QShortcuts
- Captures mouse via Qt mouse events with cursor grabbing
- Provides true FPS-style 360° mouse control

```
┌─────────────────────────────────────────────────────────────────┐
│                      gym_gui Window                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Video Tab (RGB Renderer)                    │    │
│  │                                                          │    │
│  │         [ViZDoom renders here - NO popup window]         │    │
│  │                                                          │    │
│  │   Click to capture mouse → 360° FPS control              │    │
│  │   ESC to release mouse → normal cursor                   │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Keyboard: Qt QShortcuts → SessionController → Adapter.step()    │
│  Mouse:    Qt Events → Delta callback → Adapter.apply_mouse()   │
└─────────────────────────────────────────────────────────────────┘
```

## Keyboard Controls by Scenario

Each ViZDoom scenario has a specific set of available buttons. The keyboard mappings are:

### Basic (3 actions)
| Key | Action |
|-----|--------|
| Space / Ctrl | ATTACK |
| A / ← | MOVE_LEFT |
| D / → | MOVE_RIGHT |

*Note: No W/S (forward/backward) in this scenario*

### Deadly Corridor (6 actions)
| Key | Action |
|-----|--------|
| Space / Ctrl | ATTACK |
| A | MOVE_LEFT (strafe) |
| D | MOVE_RIGHT (strafe) |
| W / ↑ | MOVE_FORWARD |
| Q / ← | TURN_LEFT |
| E / → | TURN_RIGHT |

*Note: No backward movement (S)*

### Defend The Center / Defend The Line (3 actions)
| Key | Action |
|-----|--------|
| Space / Ctrl | ATTACK |
| A / ← | TURN_LEFT |
| D / → | TURN_RIGHT |

*Note: Stationary - no movement keys*

### Health Gathering / Health Gathering Supreme / My Way Home (3 actions)
| Key | Action |
|-----|--------|
| W / ↑ | MOVE_FORWARD |
| A / ← | TURN_LEFT |
| D / → | TURN_RIGHT |

*Note: No attack or backward movement*

### Predict Position (3 actions)
| Key | Action |
|-----|--------|
| Space / Ctrl | ATTACK |
| A / ← | TURN_LEFT |
| D / → | TURN_RIGHT |

*Note: Rotation only - no movement*

### Take Cover (2 actions)
| Key | Action |
|-----|--------|
| A / ← | MOVE_LEFT |
| D / → | MOVE_RIGHT |

*Note: Only lateral movement - no attack, turning, or forward/backward*

### Deathmatch (8 actions)
| Key | Action |
|-----|--------|
| Space / Ctrl | ATTACK |
| E / Enter | USE (doors, interact) |
| W / ↑ | MOVE_FORWARD |
| S / ↓ | MOVE_BACKWARD |
| A | MOVE_LEFT (strafe) |
| D | MOVE_RIGHT (strafe) |
| Q / ← | TURN_LEFT |
| → | TURN_RIGHT |

*This is the only scenario with full WASD + backward movement*

## Mouse Control (360° FPS-Style)

### How It Works

1. **Click** on Video tab to capture mouse
2. **Move mouse** horizontally → camera turns left/right
3. **Move mouse** vertically → camera looks up/down
4. **Press ESC** to release mouse and restore cursor

### Technical Implementation

**Delta Mode** uses ViZDoom's delta buttons for smooth continuous rotation:

| Button | Index | Description |
|--------|-------|-------------|
| `TURN_LEFT_RIGHT_DELTA` | 39 | Horizontal rotation (degrees) |
| `LOOK_UP_DOWN_DELTA` | 38 | Vertical look (degrees) |

**Code Flow:**
```
_RgbView.mouseMoveEvent()
  → calculate delta_x, delta_y (pixels)
  → apply sensitivity scaling (0.5 degrees/pixel)
  → call delta_callback(degrees_x, degrees_y)
    → ViZDoomAdapter.apply_mouse_delta(delta_x, delta_y)
      → store in _pending_mouse_delta
        → applied in next step() via delta button slots
```

**Key Files:**
- `gym_gui/rendering/strategies/rgb.py` - Mouse capture and delta calculation
- `gym_gui/ui/widgets/render_tabs.py` - Delta callback configuration
- `gym_gui/ui/main_window.py` - Wiring delta mode detection
- `gym_gui/core/adapters/vizdoom.py` - Delta button setup and application

### Cursor Management

- `grabMouse()` hides cursor and captures input
- `releaseMouse()` restores cursor
- Cursor is re-centered each frame to prevent edge accumulation
- Auto-release on window focus loss

## Continuous Gameplay (Idle Tick)

### Problem
In the original implementation, ViZDoom would freeze between player inputs - enemies wouldn't move, projectiles would stop mid-air.

### Solution: ViZDoomInteractionController

Created `ViZDoomInteractionController` in `gym_gui/controllers/interaction.py`:

```python
class ViZDoomInteractionController(InteractionController):
    """Idle controller for ViZDoom: step continuously with NOOP when idle."""

    def __init__(self, owner, target_hz: int = 35):
        self._owner = owner
        self._interval_ms = max(1, int(1000 / float(target_hz)))  # ~28ms

    def idle_interval_ms(self) -> Optional[int]:
        return self._interval_ms

    def should_idle_tick(self) -> bool:
        o = self._owner
        if o._adapter is None or o._game_id is None:
            return False
        if not getattr(o, "_game_started", False):
            return False
        if o._game_paused:
            return False
        if getattr(o._control_mode, "name", "") != "HUMAN_ONLY":
            return False
        if o._last_step is not None and (o._last_step.terminated or o._last_step.truncated):
            return False
        return True

    def maybe_passive_action(self) -> Optional[Any]:
        return -1  # NOOP sentinel (all buttons = 0)

    def step_dt(self) -> float:
        return 0.0
```

**NOOP Action Handling:**
- Return `-1` as special sentinel meaning "no buttons pressed"
- Adapter recognizes `-1` and keeps all button values at 0
- This allows mouse delta to still be applied even during idle ticks

## Game Configuration

### Config Panel Options

The ViZDoom configuration panel (`gym_gui/ui/environments/single_agent_env/vizdoom/config_panel.py`) provides:

**Display Settings:**
- Resolution: 320x240, 640x480, 800x600, 1024x768
- Screen format: RGB24, RGBA32, GRAY8

**Render Toggles:**
- Show HUD (health, ammo, face)
- Show weapon sprite
- Show crosshair
- Particles effects
- Blood decals
- Sound enabled
- Depth buffer
- Labels buffer
- Automap buffer

**Episode Settings:**
- Episode timeout (100-5000 tics)
- Living reward (-10.0 to +10.0)
- Death penalty (0.0 to 500.0)

### Configuration Application

Configurations are built via `GameConfigBuilder.build_config()` and passed to the adapter:

```python
# In gym_gui/config/game_config_builder.py
elif game_id in _VIZDOOM_GAME_IDS and ViZDoomConfig is not None:
    return ViZDoomConfig(
        screen_resolution=str(overrides.get("screen_resolution", "RES_640X480")),
        screen_format=str(overrides.get("screen_format", "RGB24")),
        render_hud=bool(overrides.get("render_hud", True)),
        # ... etc
    )
```

## Telemetry and Game Variables

The adapter extracts game variables per scenario:

| Scenario | Available Variables |
|----------|-------------------|
| Basic | AMMO2 |
| Deadly Corridor | HEALTH, AMMO2 |
| Defend The Center/Line | HEALTH, AMMO2 |
| Health Gathering | HEALTH |
| Deathmatch | HEALTH, AMMO2, ARMOR, KILLCOUNT |

These are exposed in `info` dict and displayed in the telemetry panel.

## Known Limitations

1. **Sound**: ViZDoom audio may conflict with Qt; disabled by default
2. **Performance**: RGB rendering at 35 FPS is computationally intensive
3. **Thread Safety**: ViZDoom is single-threaded; use QTimer for updates

## Files Modified/Created

### Created
- `gym_gui/controllers/interaction.py` - ViZDoomInteractionController
- `gym_gui/game_docs/ViZDoom/controls.py` - Mouse capture instructions
- `gym_gui/ui/environments/single_agent_env/vizdoom/config_panel.py`

### Modified
- `gym_gui/core/adapters/vizdoom.py` - Delta buttons, mouse delta, NOOP handling
- `gym_gui/rendering/strategies/rgb.py` - 2D mouse delta capture
- `gym_gui/ui/widgets/render_tabs.py` - Delta callback support
- `gym_gui/ui/main_window.py` - Delta mode wiring
- `gym_gui/controllers/session.py` - ViZDoomInteractionController registration
- `gym_gui/controllers/human_input.py` - ViZDoom keyboard shortcuts
