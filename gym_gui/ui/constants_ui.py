from __future__ import annotations

"""UI-facing defaults for telemetry controls and rendering."""

from dataclasses import dataclass, field


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
class UIDefaults:
    """Aggregated UI defaults."""

    render: RenderDefaults = field(default_factory=RenderDefaults)
    sliders: SliderDefaults = field(default_factory=SliderDefaults)
    buffers: BufferDefaults = field(default_factory=BufferDefaults)


UI_DEFAULTS = UIDefaults()

__all__ = [
    "UI_DEFAULTS",
    "UIDefaults",
    "RenderDefaults",
    "SliderDefaults",
    "BufferDefaults",
]
