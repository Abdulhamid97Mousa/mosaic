```markdown
# Task 3 Checklist

## Worker → UI Log Flow (Structured Logging Bridge)

### ✅ Phase A: Log Constants & Helpers (Completed)
- [x] Create shared logging message tokens (`logging_config/log_constants.py`).
- [x] Extend `LogConstant` schema to include component, severity (via level), subcategory, and tags for GUI filtering.
- [x] Guard `configure_logging()` against duplicate handler reinitialisation across GUI/daemon/CLI.
- [x] Introduce dictConfig-based policy applied at bootstrap (one logging setup per process).
- [x] Adapter base emits structured lifecycle/step constants (LOG510–LOG513).
- [x] Live telemetry controller emits structured lifecycle/queue/tab constants (LOG408–LOG420).
- [x] Telemetry services, DB sink, and SQLite store emit Service component constants (LOG601–LOG624).
- [x] Emit component/severity registry events during bootstrap so GUI filter dropdowns populate dynamically.
- [x] Publish log semantics via `logging_config/log_constants.py` and migrate emitters to reference codes.

### ✅ Phase B: Dispatcher Log Bridge (Completed — Day 15)
- [x] **Implement dispatcher LOG_CODE pattern recognition** (`gym_gui/services/trainer/dispatcher.py`)
  - Added `_LOG_CODE_PATTERN` regex (lines 44-46)
  - Added `_parse_structured_log_line()` for safe pattern extraction (lines 49-69)
  - Added `_re_emit_worker_log()` for LogConstant lookup & re-emission (lines 72-97)
  - Enhanced `_stream_stdout()` with defensive three-layer fallback (lines 434-457)
- [x] Worker process logs pass through dispatcher without infrastructure changes.
- [x] Structured logs re-emit with component/subcomponent metadata.
- [x] Fallback to plain DEBUG for unstructured output (backward compatible).
- [x] Syntax verification: `python -m py_compile` passed ✅

### ⏳ Phase C: UI Integration & Remaining Defaults (Pending)
- [ ] Centralize UI defaults/constants in `gym_gui/config/ui_defaults.py`.
- [ ] Define run-config schema separating UI metadata vs. resources.
- [ ] Make LiveTelemetryTab pluggable per worker (or move SPADE-specific implementation).
- [ ] Surface DB sink tunables (batch_size, checkpoint_interval, writer_queue_size) with validation.
- [ ] Expose resource controls (CPU, memory, GPU) through the form/presenter pipeline.
- [ ] Add live render status indicators to UI (throttle, delay, disabled state).
- [ ] Update documentation/tooltips to clarify visual vs. durable settings.
- [ ] Enforce `CorrelationIdAdapter` usage (fail when `run_id`/`agent_id` remain `unknown`).
- [ ] Replace static component filter map with session-derived logger prefixes.
- [ ] Align telemetry throttle implementation with documented intent (delay vs sample + counters).
- [ ] Buffer early render payloads until regulator start; surface a diagnostic log.
- [ ] Consume credits before live publish; emit CONTROL STARVED events when throttled.
- [ ] Convert pre-tab buffers to bounded deques and expose drop counters to the status bar.
- [ ] Split `MainWindow` responsibilities via a lightweight services container + injected presenter.
- [ ] Consolidate queue sizes/topic names in a single constants module and update consumers.

```
