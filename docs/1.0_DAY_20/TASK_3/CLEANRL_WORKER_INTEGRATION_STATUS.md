# Task 3 — CleanRL Worker Integration Status (2025-11-03)

## Objective

Document the current state of the `cleanrl_worker` integration, confirm which follow-up items remain outstanding, and outline the concrete engineering steps required to expose CleanRL as a first-class worker alongside the existing SPADE-BDI path.

## Repository Findings

- `cleanrl_worker/` now exposes a proper shim (config loader, runtime, telemetry, analytics) under `MOSAIC_CLEANRL_WORKER/`. Thin wrappers remain at the legacy import paths (`cleanrl_worker.cli`, etc.) for compatibility.
- `cleanrl_worker/pyproject.toml` has been replaced with a local build definition that installs the shim and vendored modules together.
- The upstream test suite under `cleanrl_worker/tests/` has been replaced by focused unit coverage (CLI, config parsing, runtime dry-run/execution). GUI-side analytics integration now lives in `gym_gui/tests`.
- `gym_gui/ui/main_window.py` delegates analytics-tab wiring to `AnalyticsTabManager`, which creates both TensorBoard and W&B tabs when the worker uploads manifests.

## Validation of Follow-up Items

| Item | Current Status | Evidence |
| --- | --- | --- |
| TelemetryAsyncHub credit/backpressure loop | **Implemented**. The drain loop initializes the credit manager, consumes credits, and emits STARVED/RESUMED control events before publishing to the RunBus. Regression tests document the behaviour. | `gym_gui/services/trainer/streams.py:543-706`; `gym_gui/tests/test_telemetry_credit_backpressure_gap.py` |
| CleanRL runtime & manifest wiring | **Implemented**. The shim now spawns CleanRL scripts via subprocess, emits lifecycle events/heartbeats, writes `analytics.json`, and surfaces TensorBoard/W&B tabs via `AnalyticsTabManager`. | `cleanrl_worker/runtime.py`, `cleanrl_worker/cli.py`, `gym_gui/ui/panels/analytics_tabs.py`, runtime/CLI pytest suites |
| Lightweight metric bridge for analytics-first workers | **Missing**. The GUI analytics path has not been implemented, and the telemetry hub only emits STEP/EPISODE events. | No references to a metric bridge under `gym_gui/ui/widgets` or `cleanrl_worker/` |
| Operator runbooks (dependencies, ROM management, analytics access) | **Missing**. Documentation only records the intention in Task 1; there are no operator-facing runbooks yet. | `docs/1.0_DAY_20/TASK_1/INTRODUCE_CLEANRL_WORKER.md` (follow-up list), no runbook files under `docs/` |

## Restructure & Integration Plan

1. **Package Layout** — ✅
   - Shim modules (`cleanrl_worker/__init__.py`, `cli.py`, `runtime.py`, `telemetry.py`, `analytics.py`, `config.py`) now live alongside the vendored CleanRL sources.
   - Local `pyproject.toml` installs the shim while exposing `cleanrl`/`cleanrl_utils` for imports.

2. **Runtime & Telemetry Shim** — ✅
   - CLI loads trainer configs, applies overrides, and emits lifecycle JSONL.
   - Runtime spawns the requested CleanRL algorithm as a subprocess, emits heartbeats, captures logs, and writes `analytics.json` into `var/trainer/runs/<run_id>/`.

3. **Analytics Manifest** — ✅
   - `cleanrl_worker/analytics.py` assembles manifests; the runtime persists them; `AnalyticsTabManager` reads metadata to create TensorBoard/W&B tabs.

4. **Tests** — ✅ (unit coverage in place, integration suite still to come)
   - Targeted pytest modules cover CLI parsing, runtime execution (stubbed subprocess), and form/presenter wiring.
   - A dedicated trainer integration test remains on the backlog once the dispatcher flow is exercised end-to-end.

5. **Trainer & GUI Wiring** — ✅
   - Worker catalog, forms, presenters, and `AnalyticsTabManager` surface CleanRL runs through analytics tabs. Trainer promotes runs to `EXECUTING` when the first episode arrives, even without step telemetry.

6. **Documentation & Runbooks** — 🚧
   - Operator guidance (dependencies, ROM management, analytics access) is still pending.

7. **Validation** — ⏳
   - Smoke: `source .venv/bin/activate && pytest cleanrl_worker/tests`
   - GUI wiring: `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py gym_gui/tests/test_worker_presenter_and_tabs.py`
   - Integration test for dispatcher ↔ CleanRL worker remains to be authored.

## Immediate Next Actions

1. Add an end-to-end trainer integration test (`pytest gym_gui/tests/test_trainer_cleanrl_worker.py`) that exercises the telemetry proxy and manifest ingestion.
2. Wire the CleanRL worker to call `RegisterWorker` directly (bypassing the telemetry proxy handshake) and capture session capabilities.
3. Design the lightweight metric bridge so analytics-first workers can stream headline numbers to the GUI without full telemetry.
4. Draft operator runbooks (dependencies, ROM/W&B access, troubleshooting) and link them from Task 1.
5. Scope checkpoint/resume hooks once CleanRL exposes stable CLI flags.

This document will be updated after each milestone with the commands executed and pytest results to maintain traceability.

## Validation (2025-11-03)

- `source .venv/bin/activate && pytest cleanrl_worker/tests` → ✅ 9 passed (config parsing, runtime execution path with mocked subprocess, CLI entry point).
- `source .venv/bin/activate && pytest gym_gui/tests/test_cleanrl_train_form.py gym_gui/tests/test_worker_presenter_and_tabs.py` → ✅ 31 passed (train form payload, presenter registry integration, analytics tab wiring).

CleanRL runs now execute via the shim, emit lifecycle events/heartbeats, and write manifests that the GUI renders through TensorBoard/W&B tabs.
