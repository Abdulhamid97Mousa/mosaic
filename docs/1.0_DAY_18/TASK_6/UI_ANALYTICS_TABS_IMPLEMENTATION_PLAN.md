# UI Analytics Tabs Implementation Plan

_Intent: document the concrete work needed to surface TensorBoard and Weights & Biases dashboards inside the GUI for analytics-first workers (initially `cleanrl_worker`)._

## 1. Objectives

- Add dedicated analytics tabs labelled `TensorBoard-{worker_id}` and `WAB-{worker_id}` when the selected worker advertises fast-path analytics output.
- Keep live telemetry tabs disabled for analytics-only runs, but retain lifecycle status and log streaming.
- Ensure tab creation is fully dynamic so future workers can reuse the same capability without touching `MainWindow`.

## 2. Current Gaps

- `gym_gui/ui/main_window.py` still assumes SPADE as the sole worker and never instantiates analytics tabs.
- `SpadeBdiWorkerPresenter` owns tab wiring; there is no presenter for CleanRL or other analytics-focused workers.
- Worker metadata (`metadata["capabilities"]`) is not yet interpreted to drive analytics tab creation.
- No manifest parser exists on the GUI side to locate TensorBoard directories or W&B run descriptors.

## 3. Proposed UI Wiring

- Introduce a `CleanRlWorkerPresenter` (or generic `AnalyticsWorkerPresenter`) registered in `ui/presenters/workers/registry.py`.
- Extend the presenter interface with `build_analytics_tabs(run_id, worker_id, manifest)` returning descriptors for TensorBoard and W&B tabs.
- Update `MainWindow._create_agent_tabs_for` to:
  - Resolve the presenter via worker ID (no hard-coded SPADE check).
  - Ask the presenter for both live telemetry tabs and analytics tabs; fall back gracefully when a presenter does not implement one lane.
  - Use the naming convention `TensorBoard-{worker_id}` and `WAB-{worker_id}` when instantiating Qt widgets.
- Add lightweight Qt widgets:
  - `TensorBoardAnalyticsTab` → wraps TensorBoard process launcher + `QWebEngineView`.
  - `WeightsBiasesAnalyticsTab` → decides between embedded web view (online) and offline summary panel.

## 4. Worker Metadata Contract

- `cleanrl_worker` launch request should include:

  ```json
  {
    "worker_id": "cleanrl_worker",
    "capabilities": ["analytics.manifest"],
    "artifacts": {
      "tensorboard_dir": "tensorboard",
      "wandb_run": "<entity>/<project>/<run>",
      "wandb_mode": "offline"
    }
  }

  ```
- Presenter reads the manifest from `var/trainer/runs/{run_name}/artifacts.json`.
- If `tensorboard_dir` is missing, display a guidance banner instead of attempting to launch TensorBoard.


## 5. Tab Behaviour Rules

- Tabs appear only after the run enters `STARTED`; they persist after completion for post-mortem review.
- Titles follow the exact casing `TensorBoard-{worker_id}` and `WAB-{worker_id}`.
- TensorBoard tab lazily spawns `tensorboard --logdir <dir>` when first activated; shutdown occurs on tab close or app exit.
- W&B tab logic:
  - **Online**: embed the dashboard URL; show reconnect button if credentials invalid.
  - **Offline**: parse `wandb/latest-run.json`, show metrics table, and expose "Sync to W&B" button (future work).

## 6. Implementation Steps

1. Generalise presenter lookup in `MainWindow` to use worker registry and return both live + analytics descriptors.
2. Scaffold `cleanrl_worker` UI presenter, train form stub, and register with worker catalog.
3. Introduce analytics tab widgets and supporting services (TensorBoard launcher, W&B manifest reader).
4. Add manifest loading utility under `gym_gui/services/artifacts/` with unit tests covering happy/sad paths.
5. Wire presenter to call analytics widgets with manifest-derived paths, handling errors via status banners.
6. Update control panel metadata to flag workers that require analytics tabs; ensure selection dialog surfaces this.
7. Capture run lifecycle events to update tab headers with status badges (e.g., FAILED, COMPLETED) without assuming telemetry.

## 7. Testing Strategy

- Unit tests for manifest parsing and presenter tab selection logic (`pytest -k analytics_tabs`).
- Qt widget smoke tests using `QT_QPA_PLATFORM=offscreen` to verify TensorBoard/W&B tabs render placeholder states.
- End-to-end harness spawning a dummy worker that writes TensorBoard event files and mock W&B JSON, confirming GUI tabs detect artifacts.

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| TensorBoard process leaks | Track PID in the widget, terminate on tab close and app shutdown hooks. |
| Missing TensorBoard binary | Detect via `shutil.which`, surface actionable error message in tab. |
| W&B credentials unavailable | Default to offline summary and provide instructions to export or sync later. |
| Worker emits telemetry unexpectedly | Presenter should still attach live tabs when `capabilities` includes telemetry support. |

## 9. Open Questions

- Should analytics tabs be lazy-loaded per run or shared globally with reconfiguration? (Leaning toward per-run to simplify resource cleanup.)
- Do we store per-run TensorBoard ports in state to allow reconnect after crash? Needs exploration.
- What is the UX for multiple CleanRL runs in parallel—do we allow multiple TensorBoard servers? Consider port allocation pool.

## 10. Next Actions

1. Review with UI/UX stakeholders to confirm tab layout and naming.
2. Implement presenter + main window wiring changes.
3. Build analytics widgets and supporting services.
4. Write tests and run targeted pytest suites.
5. Update Day 18 documentation once implementation lands.
