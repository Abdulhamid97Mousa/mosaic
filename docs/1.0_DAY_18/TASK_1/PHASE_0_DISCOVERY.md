# Phase 0 Discovery – Telemetry Contract Baseline

_This note captures the "as-built" data contract across the trainer, telemetry, storage, and UI layers. It complements `INTEGRATION_ROADMAP.md` by grounding Phase 0 in concrete code paths and highlighting the exact gaps the schema work must close._

## 1. Worker → Trainer Contract (`RunStep`)

Source of truth: `gym_gui/services/trainer/proto/trainer.proto`

| Tag | Field | Type | Current Producer | Notes |
|-----|-------|------|------------------|-------|
| 1 | `run_id` | string | `trainer_telemetry_proxy._mk_runstep` | Required on ingress; daemon rejects streams without it. |
| 2 | `episode_index` | uint64 | Telemetry proxy (`metadata.episode_index` fallback to `episode`) | 0-based; worker_id folds into episode_id later. |
| 3 | `step_index` | uint64 | Telemetry proxy | Monotonic within episode. |
| 4 | `action_json` | string | Telemetry proxy (`_coerce_str`) | JSON/text; daemon re-coerces to `int`. |
| 5 | `observation_json` | string | Telemetry proxy (`_coerce_str`) | Raw observation serialised as JSON. |
| 6 | `reward` | double | Worker event | Native float. |
| 7 | `terminated` | bool | Worker event | |
| 8 | `truncated` | bool | Worker event | |
| 9 | `timestamp` | google.protobuf.Timestamp | Worker event (`ts_unix_ns`) | Optional; daemon fills `utcnow()` when missing. |
| 10 | `policy_label` | string | Worker event | Copied into `StepRecord.info["policy_label"]`. |
| 11 | `backend` | string | Worker event | Copied into `StepRecord.info["backend"]`. |
| 12 | `seq_id` | uint64 | Trainer daemon assigns | Assigned in `RunTelemetryBroadcaster.publish_step`. |
| 13 | `agent_id` | string | Worker event | Required for multi-agent UI tabs. |
| 14 | `render_hint_json` | string | Worker event | JSON hint payload. |
| 15 | `frame_ref` | string | Worker event | Usually path to saved frame. |
| 16 | `payload_version` | uint32 | Worker event / adapter | Currently 0 or 1. |
| 17 | `render_payload_json` | string | Worker event | Structured grid/RGB payload stored as JSON text. |
| 18 | `episode_seed` | uint64 | Worker event | Stored inside `info`. |
| 19 | `worker_id` | string | Worker event | Propagated through StepRecord + DB. |

### Observations

- Worker proxy coerces every non-string field into minified JSON (`_coerce_str`), so the proto remains string-heavy.
- No dedicated slots for space descriptors, vector metadata, or normalization stats—these must arrive via future fields.

## 2. Trainer Service Ingestion (`TrainerService.PublishRunSteps` → `StepRecord`)

Key file: `gym_gui/services/trainer/service.py`

1. The daemon fans out the raw proto via `RunTelemetryBroadcaster.publish_step()` (live UI subscribers).
2. It converts the proto into a `StepRecord` (`_step_from_proto`) for local persistence.
3. It emits a pared-down dict onto the RunBus for the legacy telemetry path.

### Proto → `StepRecord` mapping

| `StepRecord` field | Source | Notes |
|--------------------|--------|-------|
| `episode_id` | `format_episode_id(run_id, episode_index, worker_id)` | Worker-aware IDs (`run-wX-epYYYYYY`). `legacy` prefix if `run_id` missing. |
| `step_index` | `message.step_index` | |
| `action` | `_decode_action(action_json)` | Coerces JSON back into `int` when possible. |
| `observation` | `_decode_json_field(observation_json)` | Rehydrates JSON payload. |
| `reward` | `message.reward` | |
| `terminated` / `truncated` | message flags | |
| `info` | Dict assembled from optional proto fields | Includes `policy_label`, `backend`, `episode_seed`, `worker_id` when present. |
| `timestamp` | `_timestamp_from_proto(message)` | Defaults to `utcnow()` if timestamp missing. |
| `render_payload` | `_decode_json_field(render_payload_json)` | Returns dict/None. |
| `agent_id` | `message.agent_id or None` | |
| `render_hint` | `_decode_json_field(render_hint_json)` | Only retained when decoded value is a mapping. |
| `frame_ref` | `message.frame_ref or None` | |
| `payload_version` | `int(message.payload_version)` | Defaults to 0. |
| `run_id` | `message.run_id or None` | |
| `worker_id` | `message.worker_id or None` | Stored twice (top-level field plus inside `info`). |

### Bus payload (`TelemetryEvent`)

`PublishRunSteps` also publishes a `TelemetryEvent` on `Topic.STEP_APPENDED` with a very limited payload:

```python
payload_dict = {
    "run_id": run_id,
    "agent_id": agent_id,
    "episode_index": int(message.episode_index),
    "step_index": int(message.step_index),
    "reward": float(message.reward),
    "terminated": bool(message.terminated),
    "truncated": bool(message.truncated),
    "timestamp": getattr(message, "timestamp", ""),
    "action": int(getattr(message, "action", -1)) if hasattr(message, "action") else None,
    "worker_id": getattr(message, "worker_id", ""),
}
```

**Gap:** Render payloads, observations, hints, and `payload_version` never reach the bus. Downstream consumers (DB sink, UI buffers) must reconstruct or fetch from elsewhere.

## 3. Distribution Paths

```mermaid
directive topMargin=8
flowchart LR
    subgraph Worker
        W[JSONL event]
    end
    subgraph Proxy
        P((trainer_telemetry_proxy))
    end
    subgraph Daemon
        D1[gRPC ingress]
        D2[RunTelemetry\nBroadcaster]
        D3[SQLite Store]
        D4[RunBus STEP_APPENDED]
    end
    subgraph GUI
        H[TelemetryAsyncHub]
        UI[Live tabs / presenters]
        DBS[TelemetryDBSink]
        SQL[TelemetrySQLiteStore]
    end

    W --> P --> D1 --> D2
    D1 --> D3
    D2 -->|RunStep proto| H
    D2 -->|seq_id| H
    D4 -->|TelemetryEvent| DBS
    DBS -->|StepRecord (reconstructed)| SQL
    H --> UI
    D3 --> SQL
```

- **Live path** (gRPC stream): `TelemetryAsyncHub` consumes the proto, normalises it to dicts (`_normalize_payload`), and dispatches Qt signals. UI widgets still parse JSON strings on the fly (e.g. `render_payload_json`).
- **Durable path** (RunBus + DB sink): Bus payload lacks render/observation data; `TelemetryDBSink` writes skeletal `StepRecord`s with `render_payload = payload.get("render_payload")` (typically `None`). Durable storage relies on the direct `TelemetrySQLiteStore.record_step(record)` call triggered in the daemon.
- **Credit/backpressure**: `TelemetryAsyncHub` enqueues `(run_id, stream_type, payload)` tuples and uses `TelemetryBridge` to hop into the Qt thread. `RunBus` sequence tracking logs gaps but does not expose credit control yet (`Topic.CONTROL` reserved).

## 4. Persistence (`TelemetrySQLiteStore`)

File: `gym_gui/telemetry/sqlite_store.py`

Schema (post migration):

```sql
steps(
  episode_id TEXT,
  step_index INTEGER,
  action INTEGER,
  observation BLOB,
  reward REAL,
  terminated INTEGER,
  truncated INTEGER,
  info BLOB,
  render_payload BLOB,
  timestamp TEXT,
  agent_id TEXT,
  render_hint BLOB,
  frame_ref TEXT,
  payload_version INTEGER,
  run_id TEXT,
  worker_id TEXT,
  game_id TEXT,        -- derived from render_payload["game_id"]
  episode_seed INTEGER -- added via info → column migration (Phase 4 backlog)
)
```

`_step_payload(record)` adds `game_id` opportunistically by inspecting `record.render_payload`. No dedicated columns exist for vector metadata, normalization stats, or space descriptors. `episode_seed` is captured only if adapters push it into `render_payload` or `info`.

Episodes table mirrors a subset of fields (`total_reward`, `steps`, `metadata`, `worker_id`).

## 5. UI Consumption

Relevant files: `gym_gui/services/trainer/streams.py`, `gym_gui/ui/widgets/live_telemetry_tab.py`, `gym_gui/controllers/live_telemetry_controllers.py`.

- `TelemetryAsyncHub` converts protobuf messages to dict using `google.protobuf.json_format.MessageToDict` and injects defaults (`payload_version` defaults to `1`, `episode_index` derived). JSON fields remain stringified.
- `LiveTelemetryTab._try_render_visual` first seeks `payload["render_payload"]`. When absent, it falls back to `render_payload_json` and parses it, otherwise it fabricates a payload from the observation (`_generate_render_payload_from_observation`).
- Run buffers (`LiveTelemetryController._process_step_queue`) log whether each payload contains `render_payload_json`; missing payloads trigger buffer warnings but not failures.
- SPADE/BDI tabs (`ui/widgets/spade_bdi_worker_tabs/*.py`) simply check for `"render_payload_json"` keys. No schema-driven rendering exists yet.

**Implication:** The UI still expects legacy payload structures and performs ad-hoc JSON parsing. Introducing a structured schema will require updating these parsing hotspots.

## 6. Discovered Gaps & Risks

1. **Dual path divergence:** RunBus payloads omit render data, so any consumer relying solely on the bus (e.g., analytics, future services) cannot reconstruct the UI view.
2. **JSON churn:** Action, observation, render payloads, and hints bounce between JSON strings and Python objects multiple times (worker → proxy → daemon → UI). This increases latency and creates decoding failure points (`_decode_json_field` warnings already present).
3. **Info overloading:** `StepRecord.info` now carries `policy_label`, `backend`, `episode_seed`, and `worker_id`. Without typing, future schema additions become brittle.
4. **Episode seed persistence:** Seeds never land in dedicated DB columns; they are lost unless downstream consumers decode `info`.
5. **`payload_version` semantics:** Adapters default to `1` in `EnvironmentAdapter.telemetry_payload_version()` but the daemon normalises to `0` when proto field is unset. We lack negotiation rules for incompatible versions.
6. **UI fallbacks mask absence:** `_generate_render_payload_from_observation` rebuilds grids heuristically, hiding upstream omissions. Once we enforce schema v1, this fallback should disappear to expose contract violations.
7. **Render assets:** `frame_ref` is persisted but no guarantees exist that remote workers can publish accessible paths/URLs. Schema needs a formal asset representation.
8. **Vector metadata exposure:** `TelemetryEvent` and `StepRecord` carry no autoreset/reset-mask info despite vector env support in roadmap.

## 7. Baseline Requirements for Schema v1

Grounded by the gaps above, the schema proposal should at minimum standardise the following field families:

- **Identity**: `run_id`, `agent_id`, `episode_index`, `step_index`, `episode_id`, `worker_id`, `payload_version`.
- **Temporal**: `timestamp` (UTC ISO8601), optional monotonic counters.
- **Core RL state**: `action`, `observation`, `reward`, `terminated`, `truncated`, `episode_seed` (dedicated field, not `info`).
- **Environment descriptors**: `space_signature` (observation/action spaces), `render_mode`, `game_id`, `environment_name`, adapters’ version hash.
- **Render payload**: Explicit variants (`grid`, `rgb`, `graph`, `text`) using canonical keys defined under `gym_gui/constants/telemetry_constants.py` (to be added). Avoid JSON-in-string; use structured dicts/bytes.
- **Vector metadata**: `vector_index`, `autoreset_mode`, `reset_mask`, `batch_size`, seeds per sub-env.
- **Wrapper diagnostics**: `normalization_stats` (mean/var), dtype conversions, backend tags.
- **Assets**: `frame_ref` upgraded to either embedded binary chunks (when small) or signed URLs with expiry metadata.
- **Validation envelope**: Schema version + capability flags so trainer and GUI can negotiate features (e.g., `supports_space_signature`, `supports_graph_payload`).

These fields must be modelled once (JSON Schema + protobuf + Python dataclasses) and propagated through adapters, trainer proxy, daemon, storage, and UI without ad-hoc transformations.

## 8. Compatibility Matrix & Open Questions

| Area | Legacy Behaviour | Required for v1 | Migration Notes |
|------|------------------|-----------------|-----------------|
| gRPC proto | String-heavy payloads; optional render | Add structured JSON blobs (`space_signature_json`, `vector_metadata_json`, etc.) | Keep old fields optional; gate new ones behind `payload_version >= 1`. |
| Telemetry store | Derived `game_id`, no vector columns | Add explicit columns for seeds, vector/autoreset, normalization stats | Write migration (Phase 4) + backfill script. |
| RunBus events | Minimal payload | Mirror full schema (or embed reference to durable record) | Consider `Topic.STEP_APPENDED_V2` to avoid breaking existing listeners. |
| UI parsing | Fallback heuristics | Strict schema parse + validation logs | Remove `_generate_render_payload_from_observation` once adapters compliant. |
| Session recorder | Stores observation/info only | Decide whether to persist structured render payloads for offline replay | Align with telemetry schema to avoid dual formats. |

### Outstanding Questions for Design Review

1. Should RunBus events carry fully structured payloads or just references (e.g., `step_id`) to avoid duplication?
2. How do we transport large RGB/video frames—inline base64, signed URLs, or chunked blobs?
3. Do we normalise action/observation spaces in the trainer proxy or upstream adapters?
4. What is the canonical place to persist `episode_seed`, `vector_index`, and `reset_mask` so both UI and analytics agree?
5. How should the GUI detect schema capability (gRPC reflection, metadata RPC, or handshake message)?

## 9. Next Actions (Phase 0 Deliverables)

1. **Schema draft** – Prepare JSON Schema & proto field extensions covering the requirements above; circulate for review with adapter, trainer, telemetry, UI owners.
2. **Compatibility matrix** – Document how each legacy consumer will read v0 vs v1 payloads (UI tabs, DB sink, replay engine, analytics).
3. **Adapter audit** – Catalogue current `render_payload` and `info` keys emitted by `ToyTextAdapter`, `Box2DAdapter`, and planned SPADE/BDI adapters to inform required schema fields.
4. **Risk register update** – Record the dual-path divergence and JSON churn risks in the Task 1 tracker so we budget time for run bus refactor.
5. **Decision log** – Schedule design review (see roadmap) to resolve open questions and lock the schema envelope before implementation.

---

### Shortcut References

- `trainer.proto`: `gym_gui/services/trainer/proto/trainer.proto`
- Trainer daemon: `gym_gui/services/trainer/service.py`
- Telemetry store: `gym_gui/telemetry/sqlite_store.py`
- Run bus + DB sink: `gym_gui/telemetry/run_bus.py`, `gym_gui/telemetry/db_sink.py`
- UI stream hub: `gym_gui/services/trainer/streams.py`
- Live tab renderer: `gym_gui/ui/widgets/live_telemetry_tab.py`
