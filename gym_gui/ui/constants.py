"""UI-level defaults and ranges for widgets and renderers.

These values are internal implementation details. Prefer reading them via
`gym_gui.ui.constants` rather than sprinkling literals across widgets.
"""

# Render delay slider configuration (Agent Train form & live telemetry)
RENDER_DELAY_MIN_MS = 10
RENDER_DELAY_MAX_MS = 500
RENDER_DELAY_TICK_INTERVAL_MS = 50
DEFAULT_RENDER_DELAY_MS = 100  # 10 FPS

# UI training speed slider (maps value→ms = value * 10)
UI_TRAINING_SPEED_MIN = 0
UI_TRAINING_SPEED_MAX = 100

# Telemetry throttles (train form sliders)
TRAINING_TELEMETRY_THROTTLE_MIN = 1
TRAINING_TELEMETRY_THROTTLE_MAX = 10
UI_RENDERING_THROTTLE_MIN = 1
UI_RENDERING_THROTTLE_MAX = 10

# Buffer spin boxes (train form defaults)
TELEMETRY_BUFFER_MIN = 256
TELEMETRY_BUFFER_MAX = 10_000
DEFAULT_TELEMETRY_BUFFER_SIZE = 512

EPISODE_BUFFER_MIN = 10
EPISODE_BUFFER_MAX = 1_000
DEFAULT_EPISODE_BUFFER_SIZE = 100

__all__ = [
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
    "EPISODE_BUFFER_MIN",
    "EPISODE_BUFFER_MAX",
    "DEFAULT_EPISODE_BUFFER_SIZE",
]
