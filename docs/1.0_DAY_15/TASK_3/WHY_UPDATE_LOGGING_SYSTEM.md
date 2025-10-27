# Why We Updated the Logging System (Day 16 Follow-Up)

## Problem Signals

- **Dual-path ambiguity.** Operators could not tell whether slow UI updates were caused by rendering throttles or durable telemetry backpressure. UI sliders surfaced intent, but the worker silently altered or ignored those values.
- **Missing provenance.** We lacked structured logs that tied a run’s UI settings to the worker’s applied configuration, making regression triage guesswork.
- **Tests failed to guard behavior.** Prior suites verified dispatcher parsing but never asserted that UI submissions and worker bootstrap emitted traceable artifacts.

## Objectives

1. **Expose the UI/telemetry split** directly in structured logs so filters and dashboards can group incidents by path.
2. **Detect drift at bootstrap** (e.g., step delay coercion, buffer overrides) instead of letting mismatches leak into runtime.
3. **Back the catalog with tests** to ensure future refactors keep emitting the same diagnostics.

## Changes Delivered

- Added `LOG_UI_TRAIN_FORM_UI_PATH` and `LOG_UI_TRAIN_FORM_TELEMETRY_PATH` so the train form records the UI-only and durable-path payloads when a run is submitted.
- Added `LOG_WORKER_CONFIG_UI_PATH` and `LOG_WORKER_CONFIG_DURABLE_PATH`, plus a `LOG_WORKER_CONFIG_WARNING` mismatch check, to announce the worker’s applied settings.
- Persisted a shared `path_config` block in both UI metadata and worker config so tooling can reconcile intent vs. application.
- Authored focused tests:
  - `gym_gui/tests/test_logging_ui_path_config.py` (QtPy-dependent) validates the UI emissions and metadata.
  - `spade_bdi_rl/tests/test_logging_path_config.py` ensures the worker emits the new constants and surfaces mismatches.

## What To Do Next

- Wire these new log codes into the runtime log viewer filters so operators can isolate “UI-only” vs. “Telemetry-durable” events quickly.
- Add CI support for QtPy (or a headless Qt backend) so the UI logging test runs consistently instead of skipping in bare environments.
- Consider enforcing the `path_config` contract at schema-validation time to catch malformed submissions before they reach the dispatcher.
