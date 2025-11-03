# Task 5 — WAB Analytics Tab (Weights & Biases)

## Overview

The **WAB tab** is the Weights & Biases (“W&B”) analytics surface embedded inside the GUI. After a CleanRL run completes, the worker emits a manifest (`var/trainer/runs/<run_id>/analytics.json`) that can include the key `wandb_run_path`. When present, the GUI’s `AnalyticsTabManager` instantiates a tab titled **`WAB-Agent-{agent_id}`** for each agent with W&B data.

- In the SPADE-BDI Train Form, the analytics checkboxes are unlocked only when **Fast Training (Disable Telemetry)** is enabled, preventing live-telemetry runs from incurring analytics overhead by accident.

## Lifecycle

1. **Manifest creation** – `cleanrl_worker/runtime.py` writes `analytics.json` with fields such as:
   ```json
   {
     "tensorboard_dir": "var/trainer/runs/<run_id>/tensorboard",
     "wandb_run_path": "entity/project/runs/<run_key>",
     "optuna_db_path": null
   }
   ```
2. **GUI detection** – On training completion, `AnalyticsTabManager.ensure_wandb_tab` reads the manifest and, if `wandb_run_path` is a non-empty string, adds/refreshes the corresponding WAB tab.
3. **Tab creation** – The tab is constructed by `gym_gui.ui.widgets.wandb_artifact_tab.WandbArtifactTab`, providing:
   - A copyable W&B run URL (`https://wandb.ai/<run_path>` by default).
   - “Open in Browser” action (launches via `QtGui.QDesktopServices`).
   - “Open Embedded View” (when Qt WebEngine is available).
   - Status logging via `LOG_UI_RENDER_TABS_WANDB_*` constants.

## Authentication Requirements

- W&B dashboards require user authentication. Operators must run `wandb login` (or export `WANDB_API_KEY`) on the host running the GUI prior to launching embedded views.
- Without authentication, the “Open in Browser” action will still attempt to load the URL but W&B may prompt for login.
- The tab does not store credentials; it simply helps navigate to the hosted dashboard.

## Tab Naming Convention

- Tabs follow the pattern **`WAB-Agent-{agent_id}`**, aligning with the existing TensorBoard tab naming (`TensorBoard-Agent-{agent_id}`).
- This convention keeps analytics tabs grouped by agent while distinguishing the analytics provider.

## Logging & Telemetry

- Status changes emit structured logs using the newly-added constants:
  - `LOG_UI_RENDER_TABS_WANDB_STATUS`
  - `LOG_UI_RENDER_TABS_WANDB_WARNING`
  - `LOG_UI_RENDER_TABS_WANDB_ERROR`
- These logs include `run_id`, `agent_id`, and the relevant URL to aid troubleshooting.

## Future Enhancements

- Fetching W&B summaries via the REST API (subject to authentication) for lightweight metric bridging.
- Handling offline runs by pointing to locally-exported reports when no hosted dashboard exists.
