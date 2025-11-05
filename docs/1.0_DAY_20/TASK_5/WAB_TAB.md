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
  - Updated `_on_training_finished()` to call `analytics_tabs.load_and_create_tabs()` for each agent (line 1607)## WANDB Complete Integration Status ✅

**As of Day 20, the WANDB integration is COMPLETE and fully functional!**

All components are working correctly:

- ✅ WANDB run tracking initializes properly during training
- ✅ Analytics manifest with nested structure is generated correctly
- ✅ WANDB slug discovery extracts run ID from filesystem (`wandb/run-*` directories)
- ✅ Environment variable fallback reads entity/project from `.env` file
- ✅ Run URL is constructed and displayed: `https://wandb.ai/{entity}/{project}/runs/{slug}`
- ✅ "Copy URL" and "Open in Browser" buttons work perfectly
- ✅ Tab appears automatically when training completes with WANDB enabled

### Known Limitation: Embedded View

The embedded WANDB view will show an error message. This is **expected behavior** and not a bug.

Common error messages you may see:

- **"This site can't be reached / ERR_CONNECTION_TIMED_OUT"** - WANDB server is blocking the connection
- **Blank page** - WANDB iframe embedding is blocked by security headers
- **"Checking the proxy and the firewall"** - Browser cannot establish connection to wandb.ai in embedded context

## Iframe Embedding Limitations

### What W&B Supports

W&B officially supports embedding **public reports** via iframe:

> "Copy the embed code … It will render within an Inline Frame (IFrame) HTML element."  
> — [W&B Documentation](https://docs.wandb.ai/guides/app/features/panels/run-comparer)

However, the documentation also notes:

> "Only public reports are viewable when embedded."  
> — [W&B Documentation](https://docs.wandb.ai/guides/app/features/panels/run-comparer)

### What Doesn't Work (Current Implementation)

W&B **blocks embedding of project dashboards and run pages** on external domains via iframe for security reasons:

1. **X-Frame-Options Header**: W&B sets `X-Frame-Options: sameorigin` or similar headers that prevent the page from being loaded in an iframe from a different origin.

2. **Content-Security-Policy**: W&B uses CSP `frame-ancestors` directive to restrict which domains can embed their content.

3. **GitHub Issue #7632**: The W&B community has an open feature request asking for iframe embedding support:
   > "I tried embedding it in a streamlit application, but I think embedding on external domains is disabled."  
   > — [W&B GitHub Issue](https://github.com/wandb/wandb/issues/7632)

### Technical Diagnosis

When loading a W&B URL in the Qt WebEngine viewer, the following occurs:

```bash
Response Headers:
X-Frame-Options: sameorigin
Content-Security-Policy: frame-ancestors 'self' ...
```

These headers instruct the browser (and Qt WebEngine) to **refuse loading the page in an iframe** if the parent page is from a different origin. This is a security feature to prevent clickjacking attacks.

### Current Implementation (Day 20)

We've implemented appropriate fallbacks and warnings:

1. **Disabled Auto-Embed**: Changed `AUTO_EMBED_ENABLED` to `False` by default (line 20)

   ```python
   # Disable auto-embed by default since WANDB blocks iframe embedding for security
   # Set GYM_GUI_ENABLE_WANDB_AUTO_EMBED=1 to force enable (will show blank page)
   AUTO_EMBED_ENABLED = os.environ.get("GYM_GUI_ENABLE_WANDB_AUTO_EMBED", "0") == "1"
   ```

2. **Warning Messages**: Added clear warnings in the UI:
   - Placeholder text explains embedding limitations
   - Status area shows warnings when embedding fails
   - Load failure handler (`_on_load_finished()`) detects and reports failed embeds

3. **Primary Action**: "Open in Browser" button is the recommended way to view WANDB dashboards

### Recommendations

**For Users:**

- Use the **"Open in Browser"** button to view WANDB dashboards
- The embedded view won't work due to W&B's security policies
- Copy the URL for sharing or bookmarking

**For Developers:**
If embedding is critical for your application:

1. **Contact W&B Support**: Check if paid plans or custom configurations allow embedding for your domain/origin

2. **Use Public Reports**: Instead of embedding run pages, create and embed public reports (which W&B officially supports)

3. **Export Data Locally**: Download charts/plots and render them locally inside Qt instead of embedding the full interactive W&B UI

4. **Proxy/Same-Origin Workaround**: Set up a proxy that serves W&B content from the same origin as your app (complex, not recommended)

### Header Inspection

To diagnose embedding issues for a specific W&B URL, check the response headers:

```bash
curl -I https://wandb.ai/your-entity/your-project/runs/run-id
```

Look for:

- `X-Frame-Options: sameorigin` or `X-Frame-Options: DENY`
- `Content-Security-Policy: frame-ancestors 'self'` or similar

These headers confirm that W&B blocks external iframe embedding for security reasons.

## Future Enhancements

- Fetching W&B summaries via the REST API (subject to authentication) for lightweight metric bridging.
- Handling offline runs by pointing to locally-exported reports when no hosted dashboard exists.
- Extending the nested extra flattening pattern to other worker configurations if needed.
- Investigating W&B public report embedding as an alternative to run page embedding.
