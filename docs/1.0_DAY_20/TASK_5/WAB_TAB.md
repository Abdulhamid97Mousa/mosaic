# Task 5 — WAB Analytics Tab (Weights & Biases)

## Overview

The **WANDB tab** is the Weights & Biases ("W&B") analytics surface embedded inside the GUI. After a CleanRL run completes, the worker emits a manifest (`var/trainer/runs/<run_id>/analytics.json`) that can include the key `wandb_run_path`. When present, the GUI's `AnalyticsTabManager` instantiates a tab titled **`WANDB-Agent-{agent_id}`** for each agent with W&B data.

- In the SPADE-BDI Train Form, the analytics checkboxes are unlocked only when **Fast Training (Disable Telemetry)** is enabled, preventing live-telemetry runs from incurring analytics overhead by accident.

## Lifecycle

1. **Manifest creation** – Both `cleanrl_worker/runtime.py` and `spade_bdi_worker/core/runtime.py` write `analytics.json` with fields such as:
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
   - "Open in Browser" action (launches via `QtGui.QDesktopServices`).
   - "Open Embedded View" (when Qt WebEngine is available).
   - Status logging via `LOG_UI_RENDER_TABS_WANDB_*` constants.
   - Clipboard null-safety check to handle platforms where clipboard may be unavailable.

## Worker Configuration & Analytics Support

### SPADE-BDI Worker Integration (Day 20)

The SPADE-BDI worker now fully supports W&B analytics through a critical configuration fix in `spade_bdi_worker/core/config.py`:

**Problem**: The UI nests worker-specific extras under `extra.extra`, which meant W&B flags (`track_wandb`, `wandb_project_name`, `wandb_entity`) were never accessible at runtime, preventing analytics initialization.

**Solution**: `RunConfig.from_json()` now merges nested extras into the top-level `extra` dict:

```python
# Lines 90-99 in spade_bdi_worker/core/config.py
nested_extra = data.pop("extra", {})
if isinstance(nested_extra, dict):
    # Only populate keys that are not already present in the outer dict so
    # explicit overrides (e.g. telemetry buffer sizes) continue to win.
    for key, value in nested_extra.items():
        data.setdefault(key, value)
```

This allows `HeadlessTrainer` in `spade_bdi_worker/core/runtime.py` to correctly read W&B configuration:

```python
# Lines 139-141
self._wandb_enabled = bool(self.config.extra.get("track_wandb"))
self._wandb_project = self.config.extra.get("wandb_project_name")
self._wandb_entity = self.config.extra.get("wandb_entity")
```

**Test Coverage**: Added `test_run_config_live_render.py` to verify the nested extra flattening:

```python
"extra": {
    "track_wandb": True,
    "wandb_project_name": "MOSAIC",
    "wandb_entity": "abdulhamid-m-mousa-beijing-institute-of-technology",
}
```

### Analytics Manifest Generation

Both CleanRL and SPADE-BDI workers emit `analytics.json` at the end of training with the `wandb_run_path` field populated when W&B tracking is enabled. The manifest structure:

```json
{
  "tensorboard_dir": "var/trainer/runs/<run_id>/tensorboard",
  "wandb_run_path": "<entity>/<project>/runs/<run_key>",
  "optuna_db_path": null
}
```

## Authentication Requirements

- W&B dashboards require user authentication. Operators must run `wandb login` (or export `WANDB_API_KEY`) on the host running the GUI prior to launching embedded views.
- Without authentication, the "Open in Browser" action will still attempt to load the URL but W&B may prompt for login.
- The tab does not store credentials; it simply helps navigate to the hosted dashboard.
- For SPADE-BDI worker: W&B API key can also be provided via `wandb_api_key` in the extra configuration.

## Tab Naming Convention

- Tabs follow the pattern **`WANDB-Agent-{agent_id}`**, aligning with the existing TensorBoard tab naming (`TensorBoard-Agent-{agent_id}`).
- This convention keeps analytics tabs grouped by agent while distinguishing the analytics provider.

## Logging & Telemetry

- Status changes emit structured logs using the newly-added constants:
  - `LOG_UI_RENDER_TABS_WANDB_STATUS`
  - `LOG_UI_RENDER_TABS_WANDB_WARNING`
  - `LOG_UI_RENDER_TABS_WANDB_ERROR`
- These logs include `run_id`, `agent_id`, and the relevant URL to aid troubleshooting.

## Key Implementation Details

### WandbArtifactTab Widget (`gym_gui/ui/widgets/wandb_artifact_tab.py`)

**Recent Fix (Day 20)**: Added null-safety check for clipboard operations:

```python
def _copy_to_clipboard(self, value: str) -> None:
    clipboard = QtWidgets.QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(value)
        self._emit_status("copied_url", success=True)
    else:
        self._emit_status("clipboard_unavailable", success=False)
```

This prevents Pylance type errors and handles platforms where `QApplication.clipboard()` may return `None`.

## Analytics Manifest Structure Fix (Day 20)

### Problem: Structure Mismatch

The GUI expected analytics.json in a **nested structure**:

```json
{
  "artifacts": {
    "tensorboard": {
      "enabled": true,
      "log_dir": "/path/to/tensorboard",
      "relative_path": "var/trainer/runs/{run_id}/tensorboard"
    },
    "wandb": {
      "enabled": true,
      "run_path": "entity/project/runs/run_id"
    }
  }
}
```

But the SPADE-BDI worker was emitting a **flat structure**:

```json
{
  "tensorboard_dir": "/path/to/tensorboard",
  "wandb_run_path": "entity/project/runs/run_id"
}
```

This prevented `AnalyticsTabManager.ensure_wandb_tab()` from finding the W&B data at the expected path `metadata["artifacts"]["wandb"]["run_path"]`, causing W&B tabs to never appear.

### Solution

**1. Worker Manifest Generation** (`spade_bdi_worker/core/runtime.py`, lines 438-465):

Updated `_write_analytics_manifest()` to emit the nested structure:

```python
manifest = {
    "artifacts": {
        "tensorboard": {
            "enabled": self._tensorboard is not None,
            "log_dir": str(self._tensorboard.log_dir) if self._tensorboard else None,
            "relative_path": f"var/trainer/runs/{self.config.run_id}/tensorboard" if self._tensorboard else None,
        },
        "wandb": {
            "enabled": self._wandb_enabled and self._wandb_run_path is not None,
            "run_path": self._wandb_run_path,
        },
    }
}
```

**2. GUI Analytics Loading** (`gym_gui/ui/panels/analytics_tabs.py`, lines 29-90):

Added `load_and_create_tabs()` method to `AnalyticsTabManager` that:
- Loads `var/trainer/runs/{run_id}/analytics.json` from disk when training finishes
- Implements **retry mechanism** with QTimer to handle race conditions:
  - Worker writes analytics.json in `finally` block before process exits
  - GUI receives `training_finished` signal immediately when process exits
  - File might not be flushed to disk yet, so retry up to 3 times with 100ms delay
- Directly creates/refreshes TensorBoard and W&B tabs with the loaded analytics data
- Handles missing files gracefully with debug logging after all retries exhausted

```python
def load_and_create_tabs(self, run_id: str, agent_id: str, max_retries: int = 3, retry_delay_ms: int = 100) -> None:
    """Load analytics.json from disk and create/refresh analytics tabs."""
    # ... uses QTimer.singleShot for retry logic ...
```

Called automatically in `MainWindow._on_training_finished()` for each agent (line 1607).

**Why Retry Logic is Needed:**
- TensorBoard tabs appear immediately because path is pre-populated in training config
- W&B `run_path` is only known after W&B SDK initialization, written to analytics.json
- Race condition: signal emitted when process exits, but file write may still be buffering
- Retry mechanism ensures tab appears even if file write is slightly delayed

**3. Backward Compatibility**: Old analytics.json files with flat structure will be ignored (no tabs created), but new runs will work correctly.

## Modified Files Summary (Day 20)

- **`spade_bdi_worker/core/config.py`**: Added nested extra flattening logic (+11 lines, lines 90-100)
- **`spade_bdi_worker/core/runtime.py`**: 
  - Now correctly reads `track_wandb`, `wandb_project_name`, `wandb_entity` from flattened `config.extra`
  - **Fixed analytics.json structure** to nested format (lines 438-465)
- **`spade_bdi_worker/tests/test_run_config_live_render.py`**: Added test case for nested extra configuration
- **`gym_gui/ui/widgets/wandb_artifact_tab.py`**: Added clipboard null-safety check
- **`gym_gui/ui/panels/analytics_tabs.py`**:
  - **Added `load_and_create_tabs()` method** to `AnalyticsTabManager` (lines 29-63)
  - Loads analytics.json from disk and creates/refreshes analytics tabs
- **`gym_gui/ui/main_window.py`**:
  - Updated `_on_training_finished()` to call `analytics_tabs.load_and_create_tabs()` for each agent (line 1607)## Future Enhancements

- Fetching W&B summaries via the REST API (subject to authentication) for lightweight metric bridging.
- Handling offline runs by pointing to locally-exported reports when no hosted dashboard exists.
- Extending the nested extra flattening pattern to other worker configurations if needed.
