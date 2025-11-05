# Day 20 — Task 2: MOSAIC IPC FSM – Implementation Status (2025-11-02)

## Implementation Overview

The MOSAIC lifecycle is now fully represented in code. The trainer daemon, GUI client, and worker sidecars all cooperate around the `INIT → HSHK → RDY → EXEC ↔ PAUSE → FAULT → TERM` state machine. Key pillars:

1. **Protocol surface** – `trainer.proto` carries the new FSM enum and the `RegisterWorker`/`ControlEvent` messages. Regenerated stubs (`trainer_pb2`, `trainer_pb2_grpc`) expose these types to every consumer.
2. **Registry & persistence** – `RunStatus` (registry.py) stores the MOSAIC vocabulary; a migration remaps legacy rows. `_handle_run_completed_event` now records a `TERMINATED` outcome without raising `NameError`.
3. **Dispatcher orchestration** – `_dispatch_pending_runs` only drains `INIT` rows, transitions spawned workers into `HANDSHAKE`, and promotes to `EXECUTING` once telemetry begins. Heartbeat enforcement moves stalled runs into `FAULTED` instead of crashing them.
4. **Handshake enforcement** – `TrainerService.RegisterWorker` generates a session token, stamps the registry heartbeat, and advances the run to `READY`. Publishing telemetry without a successful handshake aborts the stream; the telemetry proxy now performs that handshake before forwarding stdout.
5. **Client/UI alignment** – the GUI requests the MOSAIC statuses when polling/watching and lengthened the SubmitRun deadline (6× client default) so large SPADE-BDI payloads can clear validation and GPU reservation without spurious deadline errors.
6. **Daemon delivery** – the default transport remains TCP (`127.0.0.1:50055`). A brief UNIX-socket experiment was rolled back to keep tooling and startup scripts simple. `run.sh` continues to spawn the daemon, wait for the port to open, then launch the Qt shell.

The net effect is that handshake, execution, pause/fault handling, and termination all map one-to-one with the MOSAIC memo.

### Source Files Touched

| Area | Files | Highlights |
| --- | --- | --- |
| Protocol & stubs | `gym_gui/services/trainer/proto/trainer.proto`, `trainer_pb2.py`, `trainer_pb2_grpc.py` | MOSAIC enum, handshake messages, regenerated Python bindings. |
| Registry/FSM persistence | `gym_gui/services/trainer/registry.py` | New `RunStatus` enum, status migration, fixed RUN_COMPLETED logging. |
| Dispatcher orchestration | `gym_gui/services/trainer/dispatcher.py` | `INIT` polling, `HANDSHAKE`/`EXECUTING` transitions, heartbeat → `FAULTED`. |
| Handshake & telemetry | `gym_gui/services/trainer/service.py`, `gym_gui/services/trainer/trainer_telemetry_proxy.py`, `gym_gui/services/telemetry.py` | Session token tracking, telemetry gating, optional storage helpers. |
| Client/UI | `gym_gui/services/trainer/client.py`, `gym_gui/ui/main_window.py`, `pytest.ini` | Status filters, longer SubmitRun deadline, async marker for new tests. |
| Worker compatibility | `spade_bdi_worker/core/bdi_agent.py`, `core/runtime.py`, `core/config.py`, `gym_gui/core/adapters/toy_text.py`, `spadeBDI_RL` assets, `var/data/toy_text/*.txt` | Ensure adapters load eagerly, expose goal metadata, align toy-text fixtures. |
| Requirements/docs | `requirements/base.txt`, `requirements/spade_bdi_worker.txt`, `README.md`, this doc | Dependency updates, handshake documentation. |

### Session Tokens – Why and How

- **Generation**: `TrainerService.RegisterWorker` calls `secrets.token_hex(16)` and stores the 32‑character token in `_worker_sessions`. The payload restates the worker id/kind, schema, and capability flags for auditing.
- **Enforcement today**: the first message on `PublishRunSteps/PublishRunEpisodes` checks for the presence of a session; if absent, the daemon aborts the stream with `FAILED_PRECONDITION`. This ensures only workers that completed the handshake can publish telemetry into a run.
- **Lifecycle**: tokens are cleared when a run reaches `TERMINATED` or when a handshake attempt fails. The telemetry proxy logs the accepted token for troubleshooting.
- **Future use**: once `ControlStream` is implemented, the same token can authenticate pause/resume acknowledgements or lett the GUI issue targeted control messages. Exposing the token to external tooling remains optional until control RPCs land.

## Current State

- **Trainer service** now exposes the MOSAIC lifecycle end-to-end:
  - `RegisterWorker` handshake promoted to `READY`.
  - Telemetry streams auto-transition `EXEC` on first payload.
  - `TERMINATED` becomes the only durable terminal state (cancel/complete/fault collapse).
- **Dispatcher** consumes only `INIT` runs, sets workers to `HANDSHAKE`, and moves stalled runs to `FAULTED` after heartbeat expiry.
- **Registry** persists FSM vocabulary and migrates legacy rows; run-complete events no longer crash (previous `status` reference bug removed).
- **Client/UI** request the new enum set (`INIT/HSHK/RDY/EXEC/PAUSE/FAULT/TERM`) and remain compatible with legacy daemons via proto aliases.
- **Telemetry proxy / worker** require handshake before publishing, aligning with spec §4.2 (capability negotiation). Session tokens are stored server-side but do not yet gate every subsequent RPC – a future enhancement can tie them to cancellation/control commands.

## Outstanding Items / Follow-ups

1. **ListRuns latency:** UI polls on the full FSM set every 2 s. Submit/list deadlines were stretched to avoid noisy `StatusCode.DEADLINE_EXCEEDED`, but a push-based GUI refresh would be cleaner.
2. **ControlStream** is intentionally stubbed out (returns `UNIMPLEMENTED`). Token/backpressure work is deferred per stakeholder direction.
3. **Doc Updates:** `trainer.proto` back in sync with generated stubs (FSM enum + handshake/control messages). Re-run `protoc` if additional schema fields change.

## Test Matrix

| Suite | Command | Result |
| --- | --- | --- |
| GUI services | `source .venv/bin/activate && pytest gym_gui/tests` | ✅ 173 passed |
| SPADE-BDI worker | `source .venv/bin/activate && pytest spade_bdi_worker/tests` | ✅ 105 passed, 4 skipped (ejabberd) |
| Deadline helper | `source .venv/bin/activate && pytest gym_gui/tests/test_main_window_submit_deadline.py` | ✅ ensures SubmitRun deadline tracks config |

## Logs of Interest

```bash
04:29:56 LOG702 Start run watch (statuses=INIT…TERM)
04:53:48 LOG704 SubmitRun deadline exceeded – observed before GUI deadline extension; resolved after bumping timeout
```

## Next Steps

1. Monitor GUI polling deadlines; consider longer `deadline` or caching to avoid noisy warnings.
2. Flesh out `ControlStream` once backpressure contracts finalized.
3. Keep pb2/pb2_grpc regenerated after future proto edits to avoid drift and IDE warnings.

## FSM Walkthrough (example)

1. **Launch** – GUI submits config; registry row inserted with `INIT`. Dispatcher sees run, spawns telemetry proxy + worker, flips status to `HANDSHAKE`.
2. **Handshake** – proxy invokes `RegisterWorker`, daemon records capabilities, status → `READY`.
3. **Execution** – first telemetry message arrives, daemon marks run `EXECUTING`, broadcasts update, and persists steps/episodes. Optional `PAUSE` or `FAULTED` transitions are available via dispatcher hooks/heartbeats.
4. **Termination** – worker exits or UI cancels. Dispatcher/daemon call `update_run_outcome`, status → `TERMINATED`, GPU slots released, row marked finished.

This sequence is now verified in both cleanrl/headless and SPADE-BDI flows.
