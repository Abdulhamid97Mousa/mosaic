# Implementation Details — Day 16 Task 4

## Summary

- Introduced domain-scoped constant modules:
  - `gym_gui/telemetry/constants.py`: queue sizes, hub defaults, DB sink settings, health monitor cadence, credit defaults.
  - `gym_gui/ui/constants.py`: UI slider ranges, render delay defaults, buffer bounds.
  - `spade_bdi_rl/constants.py`: worker credentials, start timeouts, telemetry buffer defaults, epsilon/map defaults.
    - Applied in `core/agent.py`, `core/config.py`, `core/bdi_agent.py`, `core/bdi_actions.py`, and `core/runtime.py`.
- Centralized trainer defaults in `gym_gui/services/trainer/constants.py` and refactored the trainer client/config validation to consume them instead of scattered literals.
- Refactored telemetry components (`TelemetryAsyncHub`, `TelemetryService`, `TelemetryDBSink`, `HealthMonitor`, `LiveTelemetryController`, bootstrap wiring) to consume values from `telemetry.constants` instead of literals.
- Updated UI components (`SpadeBdiTrainForm`, `LiveTelemetryTab`, `MainWindow`) and tests to read defaults from `ui.constants` & `telemetry.constants`.
- Rendering regulator now defaults to `DEFAULT_RENDER_DELAY_MS`.

## Key Changes

### New/Updated Files

- `gym_gui/telemetry/constants.py` (expanded)
- `gym_gui/ui/constants.py` (new)
- `gym_gui/services/trainer/constants.py`
- `gym_gui/services/trainer/client.py`
- `gym_gui/services/trainer/config.py`
- `gym_gui/services/trainer/streams.py`
- `gym_gui/services/telemetry.py`
- `gym_gui/telemetry/db_sink.py`
- `gym_gui/telemetry/health.py`
- `gym_gui/controllers/live_telemetry_controllers.py`
- `gym_gui/services/bootstrap.py`
- `gym_gui/telemetry/rendering_speed_regulator.py`
- `gym_gui/ui/widgets/spade_bdi_train_form.py`
- `gym_gui/ui/widgets/live_telemetry_tab.py`
- `gym_gui/ui/main_window.py`
- `gym_gui/tests/test_telemetry_reliability_fixes.py`
- `spade_bdi_rl/constants.py`
- `spade_bdi_rl/core/agent.py`
- `spade_bdi_rl/core/bdi_agent.py`
- `spade_bdi_rl/core/bdi_actions.py`
- `spade_bdi_rl/core/config.py`
- `spade_bdi_rl/core/runtime.py`
- TODO (next step): refactor `gym_gui/core/adapters/toy_text.py` to consume shared defaults.

### Notable Constants

- Telemetry hub: `TELEMETRY_HUB_MAX_QUEUE`, `TELEMETRY_HUB_BUFFER_SIZE`.
- DB sink: `DB_SINK_BATCH_SIZE`, `DB_SINK_CHECKPOINT_INTERVAL`, `DB_SINK_WRITER_QUEUE_SIZE`.
- Health: `HEALTH_MONITOR_HEARTBEAT_INTERVAL_S`.
- Live controller queue sizes: `LIVE_STEP_QUEUE_SIZE`, `LIVE_EPISODE_QUEUE_SIZE`, `LIVE_CONTROL_QUEUE_SIZE`.
- UI sliders/buffers: `TRAINING_TELEMETRY_THROTTLE_*`, `UI_RENDERING_THROTTLE_*`, `RENDER_DELAY_*`, `DEFAULT_TELEMETRY_BUFFER_SIZE`, `DEFAULT_EPISODE_BUFFER_SIZE`.
- Trainer service: `TRAINER_DEFAULTS.client.*`, `TRAINER_DEFAULTS.daemon.*`, and `TRAINER_DEFAULTS.retry.*` keep the gRPC bridge, daemon lifecycle, and reconnect cadence consistent.
- Train config validation: `TRAINER_DEFAULTS.schema.*` aligns JSON schema limits, canonicalization, and UI defaults.

## Tests

`python -m pytest gym_gui/tests/test_credit_manager_integration.py gym_gui/tests/test_logging_ui_path_config.py gym_gui/tests/test_telemetry_reliability_fixes.py`

- Skipped/errored due to optional deps (`grpc`, `qtpy`) not installed.

## Follow-up

- Optional: add env override helpers (similar to Ray) for constants.
- Consider documenting env keys in README once loader supports overrides.
- Monitor downstream modules for remaining literals.
- **Toy-text adapter alignment (FrozenLake/Cliff/Taxi complete):**
  - `gym_gui/constants/toy_text.py` carries canonical grid dimensions, hole counts, and official map descriptors straight from Gymnasium (stored via the new `official_map` field).
  - `gym_gui/core/adapters/toy_text.py` now resolves those defaults in the base class, removing hard-coded 4×4 / 8×8 / 4×12 / 5×5 fallbacks and reusing the official layouts when available.
  - Config overrides flow through a small `_coalesce()` helper so we only fall back when a field is `None`; explicit values like `hole_count=0` or `random_holes=False` are preserved.
  - `gym_gui/constants/loader.py` exposes `get_toy_text_defaults(GameId)` so headless code (SPADE worker) can import the same defaults without pulling the whole UI stack.
  - Worker adapters (`spade_bdi_rl/adapters/*.py`) now hydrate their map geometry from that loader and mirror the GUI’s deterministic map generation (official maps when possible, stable hole placement otherwise).
  - `spade_bdi_rl/tests/test_toy_text_alignment.py` exercises a non-default goal override end-to-end, asserting the CLI adapter factory and GUI adapter generate identical descriptors.
  - `gym_gui/config/game_configs.py` consumes the same defaults, keeping UI sliders, worker overrides, and adapter behaviour aligned without duplicate literals.
  - Stretch goal: snapshot upstream ANSI layouts to disk so we can diff Gymnasium updates automatically.

## Config Package Renames (Contrarian Take)

- Proposed rename: keep `gym_gui/config/` as the user-facing entrypoint but split internal loaders into a `gym_gui/configuration/` package. Rationale:
  - Avoid a grab-bag `config` directory where dataclasses, env parsing, and constants mingle.
  - `configuration/loader.py` can expose `load_settings()` / `get_constant()` APIs while `config/` retains declarative schemas (`game_configs.py`, `paths.py`).
  - Minimal disruption: add re-export shims in `gym_gui/config/__init__.py` for backwards compatibility and migrate call sites gradually.
- If the rename feels heavy, a lighter alternative is to rename `settings.py` → `runtime_settings.py` and `paths.py` → `filesystem_paths.py` to clarify intent. Either way, the contrarian stance is to make intent explicit so we resist the temptation to dump unrelated knobs into a single namespace.

## Agent Train Form Overrides & Worker Constants

- Surface every slider/spin-box default via the constants loader so the train form simply calls `get_constant("ui.render.delay_ms.default")`.
- When the UI submits a run, inject the selected overrides into `path_config` exactly as we log today; the worker side (`RunConfig.from_dict`) already records the same payload, so wiring the loader there ensures env-based overrides and GUI inputs converge.
- For worker-exclusive knobs (credentials, telemetry buffers), keep them in `spade_bdi_rl/constants.py` but expose read-only proxies in the loader. This keeps UI tests type-safe while letting operators change values via `.env`.

## Toy-Text Adapter Legacy Remediation (Least-Boilerplate Plan)

- `ToyTextDefaults` acts as the single source of truth for grid geometry, slippery defaults, and hole counts. The adapter can now import these values directly rather than cloning logic from Gymnasium.
- Minimal code changes required:
  1. Inject `ToyTextDefaults` into `FrozenLakeAdapter` / `FrozenLakeV2Adapter` when computing fallback grid sizes and when generating deterministic maps.
  2. Replace `_TOY_TEXT_DATA_DIR` with `paths.VAR_DATA_DIR / "toy_text"` (done) so assets respect the shared runtime directory.
  3. Guard against Gymnasium updates by snapshotting upstream MAP descriptors under `gym_gui/runtime/data/toy_text/official/` and referencing them when available.
- This approach keeps override hooks intact (UI-provided map metadata still wins) while trimming duplicate literals and making the adapter trustworthy for multi-agent rollouts.
