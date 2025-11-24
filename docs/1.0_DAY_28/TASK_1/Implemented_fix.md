# Recommended Fix: Telemetry Naming Contracts & UI Wiring

This note explains the naming/enforcement changes we just implemented (Control Panel telemetry widget + updated docs) and outlines why stricter naming conventions matter. Treat it as a guide for junior engineers who want to reason about future fixes.

## 1. UI change recap – telemetry mode widget

- We moved the interactive “Fast Lane Only” toggle out of the Status panel and into a dedicated **Telemetry Mode** group between “Game Control Flow” and “Status”.
- The widget exposes radio buttons:
  - **Fast Lane Only** – fast lane shared-memory visuals only, no RunBus durable persistence.
  - **Dual Path (RunBus UI + SQLite)** – both RunBus fast path and durable path remain active.
- Status now shows a read-only “Telemetry Mode” label so metrics stay separate from controls.
- Layout remains single-column to keep the Human Control tab predictable.

## 2. Why naming conventions matter (the “why” for enforcement)

- **Unique meaning per term:** When “fast path,” “fast lane,” or “slow lane” are reused for different parts of the stack, on-call engineers waste time guessing which toggle affects what. A shared vocabulary makes questions like “Is the durable path up?” unambiguous.
- **Self-service onboarding:** New contributors should be able to read docs and see *exactly* which modules implement each path. If the same phrase describes multiple components, we lose the ability to delegate work confidently.
- **Instrumentation & observability:** Metrics/logs only make sense when their names are stable. If `LOG_FASTLANE_QUEUE_DEPTH` means one thing in UI logs and another in worker logs, dashboards become unreliable.
- **Future refactors:** Today we have two RunBus subscribers; tomorrow we may add more. Without strong names, adding a third lane turns “dual path” from a design into a lie. Better to use descriptive names (e.g., `runbus.ui`, `runbus.db`) and keep “dual path” only as historical context.

## 3. Recommended naming contract (what to enforce)

| Object | Name to use | Reason |
| --- | --- | --- |
| Fast lane shared-memory visuals | `fastlane` | Matches `FastLaneWriter`/`FastLaneReader`; unambiguous. |
| RunBus UI queue | `runbus.ui` | Describes subscriber intent (UI) and location (RunBus). |
| RunBus durable queue | `runbus.db` | Same pattern; ready for more subscribers later. |
| Telemetry modes | `fastlane_only`, `ui_and_db` (dual path), `db_only` (future) | Encodes the combination of active paths; easy to store in config/env vars. |
| Log IDs | `LOG_FASTLANE_*`, `LOG_RUNBUS_UI_*`, `LOG_RUNBUS_DB_*` | Structured names so dashboards can filter by subsystem. |

## 4. How to enforce (next steps for junior engineers)

1. **Create a semantic-conventions helper** (`gym_gui/telemetry/semconv.py`) defining the constants above. Import it everywhere instead of hardcoding strings.
2. **Standardize metrics/log attribute names** using dot-separated names (e.g., `fastlane.frame_rate_hz`, `runbus.ui.queue_depth`). This mirrors OpenTelemetry best practices and keeps data queryable.
3. **Freeze worker toggles** through a helper (e.g., `FASTLANE_ONLY_ENV = "MOSAIC_FASTLANE_ONLY"`) so CleanRL, Coach, SPADE, etc., never re-invent names.
4. **Add lightweight tests/linters**: e.g., assert every `LOG_FASTLANE_*` constant is declared centrally; fail builds if new “fast lane” strings appear outside approved modules.

### Bonus: OTel-style metric/log names

- Adopt dot-separated, lowercase names for metrics and structured log attributes, e.g., `fastlane.frame_rate_hz`, `fastlane.reader_backlog`, `runbus.ui.queue_depth`, `runbus.db.dropped_events_total`.
- Mirrors OpenTelemetry semantic conventions, so searching/dashboards stay consistent and exporting to an OTel backend later requires zero renaming.
- Action item: define these names in the semantic convention helper and use them wherever we emit stats (FastLaneWriter, RunBus overflow reporting, DB sink metrics).

### Impacted & future files/modules

- `gym_gui/constants/constants_telemetry.py` – remains the source for queue sizes, credit limits, and will reference the semantic conventions for naming.
- `gym_gui/fastlane/*` – FastLaneWriter/Reader code will emit metrics/logs using the new OTel-style names and `LOG_FASTLANE_*` IDs.
- `gym_gui/fastlane/buffer.py` – reader now guards against zero-capacity headers so the GUI no longer crashes if it attaches before the worker finishes initializing the shared-memory ring (fixes the `ZeroDivisionError` seen when runs terminate mid-stream).
- `gym_gui/logging_config/log_constants.py` – new log IDs (e.g., `LOG_FASTLANE_HEADER_INVALID`, `LOG_FASTLANE_FRAME_READ_ERROR`, `LOG_RUNBUS_UI_QUEUE_DEPTH`) live here so everything stays centralized.
- `gym_gui/ui/fastlane_consumer.py` – now emits those FastLane log codes when it reconnects, sees an invalid header, or hits a read error, so ops can correlate UI failures with worker lifecycle.
- `gym_gui/controllers/live_telemetry_controllers.py` – likely consumer of `runbus.*` metrics/log IDs; needs updates to emit overflow stats via the semconv names.
- `gym_gui/services/trainer/trainer_telemetry_proxy.py` – bridge that currently logs FastLane events; update it to use the shared log/metric names.
- `docs/1.0_DAY_28/TASK_1/telemetry_standardization_plan.md` & `Dual Path, Fast Lane, and Slow Lane Briefing.md` – already reference constants/log IDs; keep them in sync with the semantic module.
- `gym_gui/telemetry/semconv.py` (new) centralizes path names, telemetry modes, log IDs, and metric names so all modules import a single source of truth.
- `gym_gui/fastlane/worker_helpers.py` (new) applies the canonical environment variables when launching workers.
- `gym_gui/ui/widgets/control_panel.py` – added the Environment Family filter and removed the redundant inline “CleanRL Environment” selector; only the CleanRL Agent Train Form now controls its environment list.
- `gym_gui/ui/widgets/cleanrl_train_form.py` – now owns the family+environment picker plus telemetry video mode controls.
- **Future linters/tests:** new files under `gym_gui/tests/` or `scripts/linting/` to ensure no rogue strings/log IDs slip in once the conventions are defined.
- **Multi-env Fast Lane support:** `cleanrl_worker/MOSAIC_CLEANRL_WORKER/fastlane.py` now understands the `GYM_GUI_FASTLANE_VIDEO_MODE` / `GYM_GUI_FASTLANE_GRID_LIMIT` env vars. When the mode is `grid`, the first *N* vectorized envs feed a tiler (`gym_gui/fastlane/tiling.py`) and slot `0` streams the composite image. A new pytest (`test_fastlane_grid_mode_tiles_frames`) keeps this path exercised without needing shared memory at test time.

## Upcoming implementation plan (multi-env & environment selector)

1. **FastLane video modes (SB3-style grid)** *(DONE)*
   - Worker-side tiling now exists: the FastLane wrapper batches frames from the first `grid_limit` envs, tiles them via `gym_gui/fastlane/tiling.py`, and streams the composite when `GYM_GUI_FASTLANE_VIDEO_MODE=grid`.
   - The CleanRL train form exposes the video-mode selector plus grid limit, and the worker honors those env vars. (Single mode still uses a probe index; grid mode ignores additional slots.)
   - Telemetry semantic conventions capture the new video-mode enum so UI + workers share names.
   - Contract remains one FastLane ring per run.
   - Regression coverage: `pytest gym_gui/tests/test_cleanrl_fastlane_wrapper.py` now includes a stub-writer grid test so the tiler path stays green even when shared memory is unavailable in CI.

2. **Environment family selector** *(DONE – see ControlPanel + CleanRL form updates above)*
   - Introduce a shared helper that maps `EnvironmentFamily → [(label, GameId)]`.
   - Update the Human Control tab to show a “Environment Family” combo above the existing “Environment” combo, filtering the game list dynamically.
   - Apply the same family+game picker inside the CleanRL train form (replace the single massive combo).

3. **Docs & validation** *(DONE for current phase)*
   - Document the new selectors/video modes in both `TASK_1` briefs and the new `TASK_2` folder.
   - Add basic tests for the tiling helper and environment index logic; expand existing FastLane wrapper tests to cover grid mode.
- **Possible linters/tests:** new files under `gym_gui/tests/` or `scripts/linting/` to ensure no rogue strings/log IDs slip in once the conventions are defined.

By writing these conventions into code (constants, helpers, tests) instead of prose only, we make it much harder to regress—even as the system grows beyond the current “dual path.”
