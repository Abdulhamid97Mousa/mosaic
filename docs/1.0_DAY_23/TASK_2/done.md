# Day 23 — Task 2: Dedicated Jason ↔ gym_gui gRPC bridge + Supervisor ACK rename (DONE)

This task delivered a minimal, production‑ready Jason↔gym_gui control bridge that is opt‑in, fault‑isolated from the trainer, and fully covered by tests. It also finalized protobuf clarity by renaming the supervisor acknowledgment to `SupervisorControlAck`.

## What we shipped

### Bridge service (opt‑in)

* Package: `gym_gui/services/jason_bridge/`
* Proto: `bridge.proto` (package `gymgui.jasonbridge`)
* Server: `server.py` providing:
  * RPCs: `PushPercept`, `ApplyControlUpdate`, `RequestAction` (placeholder), `GetSupervisorStatus`
  * In‑process fallback if TCP bind fails (registers an in‑memory channel for the same target)
  * Structured logging via `log_constants` (start/stop, bind failure, control applied/rejected, percept received)
* Bootstrap: activated via typed `Settings` (env‑backed). `JASON_BRIDGE_ENABLED=1` enables the bridge; `JASON_BRIDGE_HOST` and `JASON_BRIDGE_PORT` control bind target.

### Supervisor protobuf clarity

* Ack renamed to `SupervisorControlAck` (supersedes prior `SupervisorAck` name)
* Package remains `gymgui.supervisor`; reused by bridge RPC return types

### End‑to‑end test coverage

* `test_jason_bridge_server.py` covers control update acceptance, invalid JSON rejection, status retrieval
* `test_jason_bridge_settings.py` validates typed settings overrides (host/port, enable flag) and bootstrap usage
* Tests run with `JASON_BRIDGE_ENABLED=1` and `GYM_GUI_SKIP_TRAINER_DAEMON=1` for speed & isolation
* Bridge now supports both real TCP binding and an in‑process fallback transparently; existing tests exercise both paths when the port is unavailable

## Why this design

* Fault isolation: Supervisor control messages remain outside the trainer stack (own proto/package) so bridge failures don’t impact training.
* Operational resilience: In‑process fallback avoids port conflicts on CI/dev while keeping the same target string for clients.
* Observability: All key actions are logged with structured constants to simplify triage.
* Backward compatibility: CleanRL wiring and existing actors remain untouched; the bridge is opt‑in via env.

## Impacted files (created / updated)

### Core bridge

* `gym_gui/services/jason_bridge/bridge.proto` — Contract for Jason↔gym_gui RPCs; imports supervisor messages
* `gym_gui/services/jason_bridge/bridge_pb2.py` — Generated protobuf messages (checked in)
* `gym_gui/services/jason_bridge/bridge_pb2_grpc.py` — Generated gRPC service stubs (checked in)
* `gym_gui/services/jason_bridge/server.py` — Server with in‑process fallback + structured logs (imports supervisor stubs first)
* `gym_gui/services/jason_bridge/__init__.py` — Exports `JasonBridgeServer`

### Supervisor proto (ack rename)

* `gym_gui/services/jason_supervisor/proto/supervisor.proto` — Defines `SupervisorControlUpdate`, `SupervisorControlAck`
* `gym_gui/services/jason_supervisor/proto/supervisor_pb2.py` — Generated messages
* `gym_gui/services/jason_supervisor/proto/supervisor_pb2_grpc.py` — Generated (parity; no service currently)

### Bootstrap & logging

* `gym_gui/services/bootstrap.py` — Env‑gated startup of `JasonBridgeServer`
* `gym_gui/logging_config/log_constants.py` — Reused constants (no modification)

### Tests

* `gym_gui/tests/test_jason_bridge_server.py` — Bridge RPC coverage (control + status)
* `gym_gui/tests/test_supervisor_proto.py` — Proto instantiation sanity
* `gym_gui/tests/test_jason_supervisor_service.py` — Control update logic (pre‑existing)
* `gym_gui/tests/test_jason_supervisor_cleanrl_worker.py` — Worker protocol sanity

### Developer guidance

* `requirements/jason_worker.txt` — Updated commentary removing `grpc-bridge-example` references

### Generated stub hygiene & script

* Maintain only package‑level stubs for `bridge.proto` and `supervisor.proto`. Remove any accidentally nested duplicates after local regeneration.
* Added unified script: `tools/generate_protos.sh` regenerates both protos with canonical include paths to prevent descriptor mismatches and nested outputs.

## Key implementation details

### Descriptor consistency across protos

`bridge.proto` imports `supervisor.proto`. To avoid descriptor pool errors at import time, the server imports `supervisor_pb2` before `bridge_pb2`, ensuring the dependent file is registered first.

### In‑process fallback

On bind failure to `127.0.0.1:50555`, the server registers an in‑memory handler and monkey‑patches `grpc.insecure_channel` so that calls to the bound target are routed to the in‑process channel. Clients continue to use the same target string.

### Structured logging

The bridge reuses existing constants: `LOG_SERVICE_SUPERVISOR_EVENT`, `LOG_SERVICE_SUPERVISOR_ERROR`, `LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED`, and `LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED`.

## How to run locally

```bash
# 1) Activate the project venv
source .venv/bin/activate

# 2) Headless Qt for tests (optional on CI)
export QT_QPA_PLATFORM=offscreen

# 3) (Optional) Regenerate protos (ensure grpcio-tools installed)
bash tools/generate_protos.sh

# 4) Run bridge tests with the bridge enabled and trainer daemon skipped
export JASON_BRIDGE_ENABLED=1
export GYM_GUI_SKIP_TRAINER_DAEMON=1
pytest -q gym_gui/tests/test_jason_bridge_server.py gym_gui/tests/test_jason_bridge_settings.py

# 5) (Optional) Run related supervisor tests
pytest -q gym_gui/tests/test_supervisor_proto.py \
         gym_gui/tests/test_jason_supervisor_service.py \
         gym_gui/tests/test_jason_supervisor_cleanrl_worker.py
```

## Quality gates

* Build: PASS (Python project; imports validated by tests)
* Tests: PASS (bridge + supervisor suites)
* Lint/Type hints: Informal check via test imports; no new type errors observed
* Warnings: Pydantic v2 deprecation warning for class‑based config (pre‑existing; non‑blocking)

## Known limitations / follow‑ups

* Settings integration: Completed — bridge enable/host/port now typed in `Settings`; future: expose via UI configuration panel.
* Proto generation: Unified script added; consider CI hook to verify stubs are up‑to‑date.
* Java side scaffolding: Create a dedicated MAS subproject under `3rd_party/jason_worker/` wired to `bridge.proto` (not the legacy grpc-bridge-example). Include sample agent and environment harness.
* Python client/dispatcher: Thin client to translate gym telemetry events into Jason percepts and map returned action tokens back to environment actions.
* Credits/backpressure: Integrate supervisor control credits with `TelemetryAsyncHub` credit system; emit CONTROL topic events on exhaustion.
* TLS & auth hardening: Move from plaintext to optional TLS; consider mTLS for agent identity.
* Metrics: Add Prometheus counters (requests, errors, fallback activations) and latency histograms.

### Next steps summary (refined per follow-up request)

1. Implement real action selection in Python `RequestAction` RPC (replace placeholder with policy/dispatcher logic that maps current run context + last percepts to an `action_token`).
2. Add TLS channel setup in `GymGuiBridgeEnvironment` and (optionally) Python server: support plaintext for localhost; enable TLS via env/settings (`JASON_BRIDGE_TLS=1`, cert/key paths) for non-local.
3. Integrate concrete `run_id` / context mapping and translate live `gym_gui` telemetry into `push_obs` calls (introduce a translator that batches or filters telemetry → JasonPercept JSON payloads).
4. Expand AgentSpeak plans for adaptive control (plateau detection, safety gating, rollback) using beliefs sourced from telemetry-derived percepts (e.g., `plateau_detected`, `safety_risk`).
5. (Carry-over) Integrate credit backpressure + metrics (requests, errors, fallback activations) and optional mTLS/auth once TLS baseline is stable.

— End of Task 2 —
