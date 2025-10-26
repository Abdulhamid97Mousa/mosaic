# Logging Tests: Dual-Path Configuration Diagnostics

## Overview

Two comprehensive test suites validate the dual-path configuration logging system introduced in Day 15:

1. **UI-side test** (`gym_gui/tests/test_logging_ui_path_config.py`) — Validates SpadeBdiTrainForm emits structured logging for UI-only and telemetry-durable settings.
2. **Worker-side test** (`spade_bdi_rl/tests/test_logging_path_config.py`) — Validates RunConfig emits configuration events and detects mismatches between requested and applied settings.

---

## Test 1: UI-Side Logging (`test_logging_ui_path_config.py`)

### Purpose

Ensures `SpadeBdiTrainForm` captures and logs dual-path configuration (UI-only controls vs. durable telemetry settings) before handing the config to the worker.

### Test Case: `test_train_form_emits_dual_path_logs`

**File Location:** `gym_gui/tests/test_logging_ui_path_config.py:48–82`

**Preconditions:**
- Qt application instance available (fixture `qt_app`).
- Fixed datetime installed (deterministic run names via monkeypatch).

**Setup:**
```python
form = SpadeBdiTrainForm()
form._training_telemetry_throttle_slider.setValue(3)
form._ui_rendering_throttle_slider.setValue(4)
form._render_delay_slider.setValue(150)
form._ui_training_speed_slider.setValue(20)  # → 200 ms
form._telemetry_buffer_spin.setValue(4096)
form._episode_buffer_spin.setValue(512)
```

**Action:**
```python
caplog.set_level(logging.INFO, logger="gym_gui.ui.spade_bdi_train_form")
config = form._build_base_config()
```

**Assertions:**

1. **Log codes emitted:**
   ```python
   codes = _collect_log_codes(caplog)
   assert LOG_UI_TRAIN_FORM_UI_PATH.code in codes        # LOG734
   assert LOG_UI_TRAIN_FORM_TELEMETRY_PATH.code in codes # LOG735
   ```

2. **UI-only path persisted in metadata:**
   ```python
   ui_meta = config["metadata"]["ui"]["path_config"]
   assert ui_meta["ui_only"]["render_delay_ms"] == 150
   assert ui_meta["ui_only"]["step_delay_ms"] == 200
   ```

3. **Telemetry-durable path persisted:**
   ```python
   assert ui_meta["telemetry_durable"]["telemetry_buffer_size"] == 4096
   assert ui_meta["telemetry_durable"]["episode_buffer_size"] == 512
   ```

4. **Worker config receives both paths:**
   ```python
   worker_path = config["metadata"]["worker"]["config"]["path_config"]
   assert worker_path["telemetry_durable"]["telemetry_buffer_size"] == 4096
   assert worker_path["ui_only"]["render_delay_ms"] == 150
   ```

### Log Constants Registered

| Constant | Code | Level | Message | Component | Subcomponent |
|----------|------|-------|---------|-----------|--------------|
| `LOG_UI_TRAIN_FORM_UI_PATH` | LOG734 | INFO | Train form UI-only path configured | UI | TrainForm |
| `LOG_UI_TRAIN_FORM_TELEMETRY_PATH` | LOG735 | INFO | Train form telemetry durable path configured | UI | TrainForm |

**Registry Location:** `gym_gui/logging_config/log_constants.py:983–997`

### Test Execution

**Command:**
```bash
python -m pytest gym_gui/tests/test_logging_ui_path_config.py -v
```

**Expected Output:**
```
SKIPPED [1] gym_gui/tests/test_logging_ui_path_config.py:10: QtPy is required for UI logging tests
```

**Note:** Test skips automatically when QtPy is absent (headless environment). To execute:
```bash
pip install qtpy pyside2  # or PyQt5
export QT_QPA_PLATFORM=offscreen
python -m pytest gym_gui/tests/test_logging_ui_path_config.py -v
```

---

## Test 2: Worker-Side Logging (`test_logging_path_config.py`)

### Purpose

Validates `RunConfig.from_dict()` emits structured logging events for configuration initialization and detects mismatches between UI-requested and worker-applied settings.

### Test Case: `test_runconfig_logs_ui_vs_durable_settings`

**File Location:** `spade_bdi_rl/tests/test_logging_path_config.py:29–82`

**Input Payload:**
```python
payload = {
    "run_id": "run-test",
    "game_id": "ExampleGame",
    "seed": 5,
    "max_episodes": 3,
    "max_steps_per_episode": 20,
    "policy_strategy": "train_and_save",
    "agent_id": "agent-alpha",
    "step_delay": 0.5,  # 500 ms applied delay
    "telemetry_buffer_size": 2048,
    "episode_buffer_size": 128,
    "path_config": {
        "ui_only": {
            "live_rendering_enabled": False,
            "ui_rendering_throttle": 4,
            "render_delay_ms": 120,
            "step_delay_ms": 750,  # ← Mismatch: 750 ms requested vs 500 ms applied
        },
        "telemetry_durable": {
            "training_telemetry_throttle": 3,
            "telemetry_buffer_size": 4096,
            "episode_buffer_size": 256,
        },
    },
}
```

**Action:**
```python
caplog.set_level(logging.INFO, logger="spade_bdi_rl.core.config")
config = RunConfig.from_dict(payload)
```

**Assertions:**

1. **All four configuration log codes emitted:**
   ```python
   grouped = _collect_log_records(caplog)
   assert LOG_WORKER_CONFIG_EVENT.code in grouped       # LOG905
   assert LOG_WORKER_CONFIG_UI_PATH.code in grouped     # LOG907
   assert LOG_WORKER_CONFIG_DURABLE_PATH.code in grouped # LOG908
   assert LOG_WORKER_CONFIG_WARNING.code in grouped     # LOG906 (mismatch detected)
   ```

2. **UI-path record captures mismatch:**
   ```python
   ui_record = grouped[LOG_WORKER_CONFIG_UI_PATH.code][0]
   assert ui_record.live_rendering_enabled is False
   assert ui_record.step_delay_mismatch is True
   assert ui_record.requested_step_delay_ms == 750
   assert ui_record.applied_step_delay_ms == 500
   ```

3. **Durable-path record captures buffer discrepancies:**
   ```python
   durable_record = grouped[LOG_WORKER_CONFIG_DURABLE_PATH.code][0]
   assert durable_record.telemetry_buffer_requested == 4096
   assert durable_record.telemetry_buffer_applied == 2048
   assert durable_record.episode_buffer_requested == 256
   assert durable_record.episode_buffer_applied == 128
   ```

4. **Extra metadata persisted for downstream consumers:**
   ```python
   assert config.extra["path_config"]["ui_only"]["step_delay_ms"] == 750
   assert config.step_delay == pytest.approx(0.5)
   ```

### Log Constants Registered

| Constant | Code | Level | Message | Component | Subcomponent |
|----------|------|-------|---------|-----------|--------------|
| `LOG_WORKER_CONFIG_EVENT` | LOG905 | INFO | Worker configuration event | Worker | Config |
| `LOG_WORKER_CONFIG_WARNING` | LOG906 | WARNING | Worker configuration warning | Worker | Config |
| `LOG_WORKER_CONFIG_UI_PATH` | LOG907 | INFO | Worker UI-only path settings applied | Worker | Config |
| `LOG_WORKER_CONFIG_DURABLE_PATH` | LOG908 | INFO | Worker telemetry durable path settings applied | Worker | Config |

**Registry Location:** `gym_gui/logging_config/log_constants.py:1105–1135`

### Test Execution

**Command:**
```bash
python -m pytest spade_bdi_rl/tests/test_logging_path_config.py -v
```

**Expected Output:**
```
test_logging_path_config.py::test_runconfig_logs_ui_vs_durable_settings PASSED
```

---

## Running Both Tests

### Full Suite
```bash
python -m pytest \
  gym_gui/tests/test_logging_ui_path_config.py \
  spade_bdi_rl/tests/test_logging_path_config.py \
  -v
```

### Expected Results
```
gym_gui/tests/test_logging_ui_path_config.py::test_train_form_emits_dual_path_logs SKIPPED
spade_bdi_rl/tests/test_logging_path_config.py::test_runconfig_logs_ui_vs_durable_settings PASSED
=================== 1 passed, 1 skipped, 1 warning in 0.06s ===================
```

### With Qt Backend (Full Execution)
```bash
pip install qtpy pyside2
export QT_QPA_PLATFORM=offscreen
python -m pytest \
  gym_gui/tests/test_logging_ui_path_config.py \
  spade_bdi_rl/tests/test_logging_path_config.py \
  -v
```

**Expected Results:**
```
gym_gui/tests/test_logging_ui_path_config.py::test_train_form_emits_dual_path_logs PASSED
spade_bdi_rl/tests/test_logging_path_config.py::test_runconfig_logs_ui_vs_durable_settings PASSED
=================== 2 passed in 0.12s ===================
```

---

## Test Helpers

### `_collect_log_codes(caplog)` — UI Test

**Location:** `gym_gui/tests/test_logging_ui_path_config.py:39–45`

Extracts log codes from captured records by reading the `log_code` attribute:
```python
def _collect_log_codes(caplog: pytest.LogCaptureFixture) -> set[str]:
    """Extract log codes produced by log_constant helper."""
    return {
        record.__dict__["log_code"]
        for record in caplog.records
        if "log_code" in record.__dict__
    }
```

### `_collect_log_records(caplog)` — Worker Test

**Location:** `spade_bdi_rl/tests/test_logging_path_config.py:18–26`

Groups captured records by log code for assertion on extra attributes:
```python
def _collect_log_records(caplog: pytest.LogCaptureFixture) -> dict[str, list[logging.LogRecord]]:
    """Group captured records by log code."""
    grouped: dict[str, list[logging.LogRecord]] = {}
    for record in caplog.records:
        code = record.__dict__.get("log_code")
        if not code:
            continue
        grouped.setdefault(code, []).append(record)
    return grouped
```

### `_install_fixed_datetime(monkeypatch)` — UI Test

**Location:** `gym_gui/tests/test_logging_ui_path_config.py:28–36`

Patches `datetime.utcnow()` to return a fixed date (2025-10-26 12:00:00) for deterministic run name generation:
```python
def _install_fixed_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic timestamps so run_name assertions remain stable."""

    class FixedDatetime(real_datetime):
        @classmethod
        def utcnow(cls) -> real_datetime:
            return cls(2025, 10, 26, 12, 0, 0)

    monkeypatch.setattr(train_form_mod, "datetime", FixedDatetime)
```

---

## Coverage Summary

### What These Tests Validate

| Aspect | Test | File | Lines |
|--------|------|------|-------|
| UI form emits LOG734 (UI-only path) | UI | `test_logging_ui_path_config.py` | 48–82 |
| UI form emits LOG735 (durable path) | UI | `test_logging_ui_path_config.py` | 48–82 |
| UI metadata structured correctly | UI | `test_logging_ui_path_config.py` | 70–74 |
| Worker metadata structured correctly | UI | `test_logging_ui_path_config.py` | 76–78 |
| Worker emits LOG905 (event) | Worker | `test_logging_path_config.py` | 29–82 |
| Worker emits LOG906 (warning on mismatch) | Worker | `test_logging_path_config.py` | 29–82 |
| Worker emits LOG907 (UI-only applied) | Worker | `test_logging_path_config.py` | 29–82 |
| Worker emits LOG908 (durable applied) | Worker | `test_logging_path_config.py` | 29–82 |
| Mismatch detection (step_delay: 750 vs 500) | Worker | `test_logging_path_config.py` | 69–72 |
| Buffer mismatch detection (telemetry: 4096 vs 2048) | Worker | `test_logging_path_config.py` | 77–78 |
| Extra metadata persisted | Worker | `test_logging_path_config.py` | 80–82 |

### Integration Points

- **log_constants.py** → Registers 6 new LOG codes (LOG734, LOG735, LOG905–LOG908).
- **SpadeBdiTrainForm** → Emits LOG734 and LOG735 when `_build_base_config()` called.
- **RunConfig** → Emits LOG905–LOG908 when `from_dict()` processes configuration.
- **Dispatcher** → Re-emits these codes if worker logs contain structured format (Day 15 Phase B).

---

## Next Steps

1. **Enable CI/CD integration:** Install QtPy + Qt backend in CI pipeline so UI test executes instead of skipping.
2. **End-to-end validation:** Run dispatcher with actual worker output to confirm `LOG734`, `LOG735`, `LOG905–LOG908` codes surface in UI console.
3. **Monitor production:** Track mismatch warnings (LOG906) for configuration drift issues in deployed runs.
4. **Extend test coverage:** Add negative cases (invalid buffer sizes, out-of-range throttle values).
