"""UI-level defaults and ranges for widgets and renderers.

Consolidated UI constants from:
- Original: gym_gui/ui/constants.py (simple values)
- Original: gym_gui/ui/constants_ui.py (dataclass structures)

These values are internal implementation details. Prefer reading them via
`gym_gui.constants` rather than sprinkling literals across widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ================================================================
# Render delay slider configuration (Agent Train form & live telemetry)
# ================================================================

RENDER_DELAY_MIN_MS = 10
RENDER_DELAY_MAX_MS = 500
RENDER_DELAY_TICK_INTERVAL_MS = 50
DEFAULT_RENDER_DELAY_MS = 100  # 10 FPS

# ================================================================
# UI training speed slider (maps value→ms = value * 10)
# ================================================================

UI_TRAINING_SPEED_MIN = 0
UI_TRAINING_SPEED_MAX = 100

# ================================================================
# Telemetry throttles (train form sliders)
# ================================================================

TRAINING_TELEMETRY_THROTTLE_MIN = 1
TRAINING_TELEMETRY_THROTTLE_MAX = 10
UI_RENDERING_THROTTLE_MIN = 1
UI_RENDERING_THROTTLE_MAX = 10

# ================================================================
# Buffer spin boxes (train form defaults)
# ================================================================

TELEMETRY_BUFFER_MIN = 256
TELEMETRY_BUFFER_MAX = 10_000
DEFAULT_TELEMETRY_BUFFER_SIZE = 512

# Backwards compatibility alias kept for legacy imports expecting this typo.
BUFFER_BUFFER_MIN = TELEMETRY_BUFFER_MIN

EPISODE_BUFFER_MIN = 10
EPISODE_BUFFER_MAX = 1_000
DEFAULT_EPISODE_BUFFER_SIZE = 100

# ================================================================
# Structured UI Defaults (Dataclasses for organized access)
# ================================================================


@dataclass(frozen=True)
class RenderDefaults:
    """Rendering cadence options shared across UI widgets."""

    min_delay_ms: int = 10
    max_delay_ms: int = 500
    tick_interval_ms: int = 50
    default_delay_ms: int = 100
    queue_size: int = 32
    bootstrap_timeout_ms: int = 500


@dataclass(frozen=True)
class SliderDefaults:
    """Slider ranges for training speed and throttling controls."""

    training_speed_min: int = 0
    training_speed_max: int = 100
    telemetry_throttle_min: int = 1
    telemetry_throttle_max: int = 10
    rendering_throttle_min: int = 1
    rendering_throttle_max: int = 10


@dataclass(frozen=True)
class BufferDefaults:
    """Buffer sizing for live telemetry pre-tab and queue limits."""

    step_pre_tab: int = 64
    episode_pre_tab: int = 32
    telemetry_buffer_min: int = 256
    telemetry_buffer_max: int = 10_000
    telemetry_buffer_default: int = 512
    episode_buffer_min: int = 10
    episode_buffer_max: int = 1_000
    episode_buffer_default: int = 100
    live_step_queue: int = 64
    live_episode_queue: int = 64
    live_control_queue: int = 32


@dataclass(frozen=True)
class LayoutDefaults:
    """Splitter column sizing for the main window layout."""

    control_panel_min_width: int = 320
    control_panel_default_width: int = 360
    render_min_width: int = 640
    render_default_width: int = 720
    render_max_width: int | None = None
    info_min_width: int = 260
    info_default_width: int = 280
    log_min_width: int = 320
    log_default_width: int = 360


@dataclass(frozen=True)
class UIDefaults:
    """Aggregated UI defaults."""

    render: RenderDefaults = field(default_factory=RenderDefaults)
    sliders: SliderDefaults = field(default_factory=SliderDefaults)
    buffers: BufferDefaults = field(default_factory=BufferDefaults)
    layout: LayoutDefaults = field(default_factory=LayoutDefaults)


UI_DEFAULTS = UIDefaults()

__all__ = [
    # Simple constants
    "RENDER_DELAY_MIN_MS",
    "RENDER_DELAY_MAX_MS",
    "RENDER_DELAY_TICK_INTERVAL_MS",
    "DEFAULT_RENDER_DELAY_MS",
    "UI_TRAINING_SPEED_MIN",
    "UI_TRAINING_SPEED_MAX",
    "TRAINING_TELEMETRY_THROTTLE_MIN",
    "TRAINING_TELEMETRY_THROTTLE_MAX",
    "UI_RENDERING_THROTTLE_MIN",
    "UI_RENDERING_THROTTLE_MAX",
    "TELEMETRY_BUFFER_MIN",
    "TELEMETRY_BUFFER_MAX",
    "DEFAULT_TELEMETRY_BUFFER_SIZE",
    "BUFFER_BUFFER_MIN",
    "EPISODE_BUFFER_MIN",
    "EPISODE_BUFFER_MAX",
    "DEFAULT_EPISODE_BUFFER_SIZE",
    # Structured defaults
    "UI_DEFAULTS",
    "UIDefaults",
    "RenderDefaults",
    "SliderDefaults",
    "BufferDefaults",
    "LayoutDefaults",
]
