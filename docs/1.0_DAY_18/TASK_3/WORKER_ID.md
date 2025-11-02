# TASK 3 – Worker ID Integration Plan

## Current Situation

| Layer | Behaviour Today | Consequence |
|-------|-----------------|-------------|
| Training form (`SpadeBdiTrainForm`) | Worker ID text field + normalization/validation | Users can pin workers (default slug width = 6) |
| Config validation (`validate_train_run_config`) | Preserves worker id in canonical payload & digest | Dispatcher/daemon receive deterministic worker metadata |
| Dispatcher / launcher | Writes `worker-{run_id}-{worker_id}.json`, passes `--worker-id` and `WORKER_ID` env | Worker subprocesses know their identifier |
| Worker runtime (`RunConfig`, `TelemetryEmitter`) | Dataclass + emitter include worker id in all events | Step/Episode telemetry carries worker scope |
| Session controller / counter | `set_run_context(..., worker_id=…)` drives `RunCounterManager` | Episode IDs use `{run}-w{worker}-ep{index}` |
| SQLite & rendering | Steps/episodes tables persist worker id; unique index `(run_id, worker_id, ep_index)` honored | DB queries + retention disambiguate multi-worker runs |

### Progress Update (Oct 29, 2025)

- ✅ Training form now captures a `worker_id` field (validated + normalized)
- ✅ Trainer config, dispatcher, and telemetry proxy persist/publish `worker_id`
- ✅ Worker runtime (`RunConfig`, `TelemetryEmitter`) stamps every payload with the worker id
- ✅ Telemetry service + SQLite store write `worker_id`; episode IDs use the new triplet format
- ✅ Session controller + counter operate in worker scope
- 🟡 Live telemetry controllers still index by `(run_id, agent_id)` — worker-aware buffers remain to do
- 🟡 Multi-worker/Bdi integration tests pending additions (`pytest spade_bdi_rl/tests` currently fails in legacy suites)

## Target Behaviour

1. **Every worker attached to a run must expose a unique identifier** – supplied by UI or generated (ULID suffix/sequence).
2. **Identifiers propagate end-to-end:** form → trainer config → dispatcher → worker CLI/env → telemetry emitter → DB/GUI.
3. **Episode IDs use full triple:** `format_episode_id(run_id, ep_index, worker_id)` resulting in `01ABC…-w000007-ep000042`.
4. **Queries and UI controllers use `(run_id, worker_id)` to filter buffers.**

## Constants & Log Codes

- `WORKER_ID_PREFIX = "w"`
- `WORKER_ID_WIDTH = 6`
- `WORKER_ID_COLUMN = "worker_id"`
- Log codes encountered:
  - `LOG_UI_TRAIN_FORM_INFO`, `LOG_UI_TRAIN_FORM_TELEMETRY_PATH`, `LOG_UI_TRAIN_FORM_UI_PATH`
  - `LOG_WORKER_CONFIG_UI_PATH`, `LOG_WORKER_CONFIG_DURABLE_PATH`
  - `LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED`, `LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED`
  - `LOG_SERVICE_SQLITE_DEBUG`, `LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED`

## Implementation Plan

### Step 1 – Training Form & Config ✅

- Add an optional "Worker ID" input (text field or toggle to auto-assign).
- Include `worker_id` inside:
  - `metadata.ui.path_config.ui_only["worker_id"]`
  - `metadata.worker.config["worker_id"]`
  - `environment["WORKER_ID"]`
- Update `validate_agent_train_form` to validate the field (non-empty slug if provided).

### Step 2 – Trainer Config Validation ✅

- Extend `TrainRunConfig` to capture `worker_id` from payload metadata.
- When `_canonicalize_config()` runs, ensure `metadata.worker.config.worker_id` remains intact.
- Update `_stable_digest` if needed, so digest reflects worker id.
- Include worker id in `_save_run_config` logs for audit trail.

### Step 3 – Dispatcher, Launcher, Service ✅

- When writing worker JSON (`worker-{run_id}.json`), persist `worker_id`.
- Append CLI argument `--worker-id <value>` if not already present.
- Export `WORKER_ID` environment variable for the subprocess.
- Update log codes in dispatcher to mention the worker id for traceability (e.g., `extra={"run_id":…, "worker_id":…}`).

### Step 4 – Worker Runtime (`spade_bdi_rl`) ✅

- Extend `RunConfig` dataclass to accept `worker_id` and ensure tests cover it.
- Store worker id when `RunConfig.from_dict()` parses the payload.
- Update telemetry emitter:
  - Add `worker_id` to every `step` and `episode` payload.
  - Use it when composing `EpisodeRollup` / `StepRecord` dataclasses.
- Propagate worker id into CLI argument parsing so manual runs can supply `--worker-id`.

### Step 5 – GUI Session & Counters ✅

- Modify `SessionController.set_run_context(run_id, max_episodes, db_conn, worker_id)` signature.
- Pass worker id to `RunCounterManager` so `format_episode_id` outputs the prefixed form.
- Update `LiveTelemetryController` to key tabs/buffers by `(run_id, worker_id, agent_id)` or collapse agent id into worker id when distributed.
- Adjust log constants: include worker id when logging run/episode transitions.

### Step 6 – Telemetry Store & Queries ✅

- In SQLite store, ensure inserts set `worker_id` column.
- Update `get_run_summary` / closure dialog to report counts grouped by worker.
- Verify schema uniqueness (`UNIQUE(run_id, worker_id, ep_index)`) holds in tests.

### Step 7 – Testing & Docs 🟡

- [x] Add unit tests for worker-aware train form, dispatcher wrapper, and signal exposure (`gym_gui/tests/*`)
- [x] Update documentation (constants README, TASK 2 + TASK 3 notes)
- [ ] Add multi-worker integration/regression tests (GUI + `spade_bdi_rl` suites)
- [ ] Consider a migration script if historic telemetry needs backfilling (optional)

## Integration Flow Summary

```bash
UI (SpadeBdiTrainForm)
 └─> metadata.worker.config.worker_id
       └─> TrainerService.validate_train_run_config
             └─> Dispatcher writes worker-{run}.json & CLI --worker-id
                   └─> spade_bdi_rl.worker (RunConfig)
                         └─> TelemetryEmitter.step/episode(worker_id)
                               └─> SQLite Store (episodes.worker_id / steps.worker_id)
                                     └─> LiveTelemetryController & Render Tabs (per worker buffers)
```

> **Next:** Live telemetry controllers still group by `(run_id, agent_id)`.
> Worker-aware buffers/tabs are the remaining UI integration task before we
> can run true multi-worker sessions end-to-end.

Once implemented, a single run may have `run_id=01ABC…` with workers `000001`, `000002`, etc., enabling concurrent actors without overwriting each other’s counters or telemetry streams.
