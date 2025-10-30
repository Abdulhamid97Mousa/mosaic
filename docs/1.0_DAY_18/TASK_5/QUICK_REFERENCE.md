# QUICK REFERENCE: File Changes at a Glance

## Where to Make Changes

### 1. `spade_bdi_worker/core/telemetry_worker.py`
```
Line 22-29   → Add disabled parameter to __init__
Line 34-40   → Add if self._disabled: return at start of emit()
Line 140+    → Add new emit_lifecycle() method
```

### 2. `spade_bdi_worker/worker.py`
```
Line 74+     → Add --no-telemetry argument
Line 91-96   → Extract disable_telemetry_flag from parsed/env
Line 104     → Pass disabled=... to TelemetryEmitter()
```

### 3. `spade_bdi_worker/core/runtime.py`
```
Line 69      → Change emitter.run_started() → emitter.emit_lifecycle()
Line 130+    → Change all emitter.run_completed() → emitter.emit_lifecycle()
```

### 4. `gym_gui/ui/widgets/spade_bdi_train_form.py`
```
Line 185+    → Add _fast_training_checkbox and handler
Line 710+    → In _on_accept(), add confirmation dialog
Line 815     → In _build_base_config(), add fast_training_mode = ...
Line 870+    → Add DISABLE_TELEMETRY to environment dict
Line 935     → Add disable_telemetry to "extra" dict
```

---

## Code Snippets (Copy-Paste Ready)

### Snippet 1: TelemetryEmitter.__init__
```python
def __init__(self, stream: IO[str] | None = None, disabled: bool = False) -> None:
    self._stream: IO[str] = stream or sys.stdout
    self._disabled: bool = disabled
```

### Snippet 2: TelemetryEmitter.emit (start)
```python
def emit(self, event_type: str, **fields: Any) -> None:
    if self._disabled:
        return  # Skip when disabled
    ts_ns = _utc_timestamp_ns()
    # ... rest of existing code ...
```

### Snippet 3: TelemetryEmitter.emit_lifecycle (new method)
```python
def emit_lifecycle(self, event_type: str, **fields: Any) -> None:
    """Emit lifecycle events bypassing disabled flag."""
    ts_ns = _utc_timestamp_ns()
    payload: Dict[str, Any] = {
        "type": event_type,
        "ts": datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat(),
        "ts_unix_ns": ts_ns,
        **fields,
    }
    json.dump(payload, self._stream, separators=(",", ":"))
    self._stream.write("\n")
    self._stream.flush()
```

### Snippet 4: Worker --no-telemetry arg
```python
parser.add_argument(
    "--no-telemetry",
    action="store_true",
    help="Disable per-step telemetry emission for fast training.",
)
```

### Snippet 5: Worker config flag extraction
```python
disable_telemetry_flag = parsed.no_telemetry or os.environ.get("DISABLE_TELEMETRY", "").lower() == "true"
run_config.extra["disable_telemetry"] = disable_telemetry_flag
```

### Snippet 6: Worker emitter creation
```python
emitter = TelemetryEmitter(disabled=run_config.extra.get("disable_telemetry", False))
```

### Snippet 7: GUI checkbox
```python
self._fast_training_checkbox = QtWidgets.QCheckBox("Fast Training Mode")
self._fast_training_checkbox.setCheckState(QtCore.Qt.CheckState.Unchecked)
self._fast_training_checkbox.setToolTip(
    "When enabled:\n"
    "• Disables per-step telemetry collection\n"
    "• 30-50% faster training on GPU\n"
    "⚠ Episode replay unavailable"
)
self._fast_training_checkbox.toggled.connect(self._on_fast_training_toggled)
right_layout.addRow("Fast Training (Disable Telemetry):", self._fast_training_checkbox)
```

### Snippet 8: GUI toggle handler
```python
def _on_fast_training_toggled(self, checked: bool) -> None:
    """Handle fast training mode toggle."""
    if checked:
        self._disable_live_render_checkbox.setChecked(True)
        self._disable_live_render_checkbox.setEnabled(False)
        self._ui_rendering_throttle_slider.setEnabled(False)
        self._training_telemetry_throttle_slider.setEnabled(False)
    else:
        self._disable_live_render_checkbox.setEnabled(True)
        self._ui_rendering_throttle_slider.setEnabled(True)
        self._training_telemetry_throttle_slider.setEnabled(True)
```

### Snippet 9: GUI confirmation dialog
```python
if self._fast_training_checkbox.isChecked():
    reply = QtWidgets.QMessageBox.warning(
        self,
        "Fast Training Mode - Confirm",
        "Fast Training Mode will:\n\n"
        "❌ Disable per-step telemetry collection\n"
        "❌ Disable live UI updates\n"
        "❌ Make episode replay unavailable\n\n"
        "✅ Provide 30-50% speedup on GPU\n\n"
        "Are you sure?",
        QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
    )
    if reply == QtWidgets.QMessageBox.Cancel:
        return
```

### Snippet 10: Config flag passing
```python
# In _build_base_config()
fast_training_mode = self._fast_training_checkbox.isChecked()

# In environment dict
"DISABLE_TELEMETRY": "1" if fast_training_mode else "0",

# In "extra" dict
"disable_telemetry": fast_training_mode,
```

---

## Testing Commands

```bash
# Test 1: Verify flag exists
python -m spade_bdi_rl_worker --help | grep -A2 "no-telemetry"

# Test 2: Run with telemetry disabled
cat > /tmp/test_no_telem.json << 'EOF'
{
  "run_id": "test",
  "game_id": "CartPole-v1",
  "seed": 42,
  "max_episodes": 3,
  "max_steps_per_episode": 200,
  "extra": {"disable_telemetry": true}
}
EOF
python -m spade_bdi_rl_worker --config /tmp/test_no_telem.json 2>/dev/null | wc -l

# Test 3: Run with telemetry enabled (for comparison)
cat > /tmp/test_with_telem.json << 'EOF'
{
  "run_id": "test2",
  "game_id": "CartPole-v1",
  "seed": 42,
  "max_episodes": 3,
  "max_steps_per_episode": 200,
  "extra": {"disable_telemetry": false}
}
EOF
python -m spade_bdi_rl_worker --config /tmp/test_with_telem.json 2>/dev/null | wc -l

# Test 4: GUI
python -m gym_gui.app
# Open train form → check "Fast Training Mode" → should see warning → click Train → should see dialog
```

---

## Validation Checklist

- [ ] `--no-telemetry` flag appears in `python -m spade_bdi_rl_worker --help`
- [ ] Worker runs without error using `--no-telemetry`
- [ ] Output with disabled telemetry: ~3 lines (run_started, run_completed only)
- [ ] Output with enabled telemetry: 1000+ lines
- [ ] GUI checkbox appears in train form
- [ ] Checkbox tooltip is informative
- [ ] Toggling checkbox enables/disables other controls
- [ ] Confirmation dialog appears when submitting with fast mode ON
- [ ] Training completes without errors
- [ ] Old training (without fast mode) still works

---

## Files to Verify After Changes

```
✓ spade_bdi_worker/core/telemetry_worker.py  → Has disabled param + emit_lifecycle()
✓ spade_bdi_worker/worker.py                  → Has --no-telemetry arg
✓ spade_bdi_worker/core/runtime.py            → Uses emit_lifecycle() for lifecycle
✓ gym_gui/ui/widgets/spade_bdi_train_form.py → Has checkbox + pass flag to config
```

---

## Expected Output

### With Fast Training Disabled (normal mode)
```
{"type":"run_started","ts":"...","ts_unix_ns":...,"run_id":"...","config":{...},"worker_id":"..."}
{"type":"step","ts":"...","ts_unix_ns":...,"run_id":"...","episode":0,"step_index":0,...}
{"type":"step","ts":"...","ts_unix_ns":...,"run_id":"...","episode":0,"step_index":1,...}
... (hundreds of steps and episodes) ...
{"type":"run_completed","ts":"...","ts_unix_ns":...,"run_id":"...","status":"completed",...}
```

### With Fast Training Enabled
```
{"type":"run_started","ts":"...","ts_unix_ns":...,"run_id":"...","config":{...},"worker_id":"..."}
{"type":"run_completed","ts":"...","ts_unix_ns":...,"run_id":"...","status":"completed",...}
```

(Only 2 lines!)

---

Ready to start? Open `IMPLEMENTATION_GROUNDED_IN_CODEBASE.md` for the full details! 🚀


