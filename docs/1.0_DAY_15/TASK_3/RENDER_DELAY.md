# Render Delay Slider

## Current Behaviour

- Slider defined in `gym_gui/ui/widgets/spade_bdi_train_form.py` (`_render_delay_slider`).
- `_on_render_delay_changed` updates label to “X ms (Y FPS)”.
- Outputs:
  - `metadata.ui.render_delay_ms`
  - `environment["UI_RENDER_DELAY_MS"]` (set to `0` when live rendering disabled)
- `LiveTelemetryController` stores delay via `set_render_delay_for_run`, and
  `LiveTelemetryTab` passes it to `RenderingSpeedRegulator`, controlling the timer cadence.
- When live rendering is disabled, the tab shows a notice and no regulator is created.

## Problem

- Operators may think the slider influences dropped telemetry; it only affects regulator cadence.
- Metadata vs. environment values diverge when rendering is disabled (metadata keeps original value).

## Files / Modules Impacted

- `gym_gui/ui/widgets/spade_bdi_train_form.py`
- `gym_gui/ui/main_window.py`
- `gym_gui/controllers/live_telemetry_controllers.py`
- `gym_gui/ui/widgets/live_telemetry_tab.py`

## Proposal

- Surface active render delay in UI status or log when tabs start.
- When rendering disabled, consider zeroing metadata or annotating config for clarity.

