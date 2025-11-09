# Day 23 — Task 2: Jason ⇄ gym_gui integration baseline (DONE)

This task delivered a safe, incremental baseline to let Jason supervise RL runs in gym_gui without touching existing CleanRL wiring and while keeping fault isolation between trainer and supervisor messages.

## What we shipped

- New worker (no wiring changes to CleanRL)
  - `gym_gui/workers/jason_supervisor_cleanrl_worker/worker.py`
  - Actor id: `jason_supervisor_cleanrl_worker`
  - Implements the `Actor` protocol, abstains from action selection (CleanRL worker owns decisions), and emits low‑frequency, structured supervisor state samples for observability.
  - Optional auto‑registration gated by env var (no bootstrap edits):
    - `ENABLE_JASON_SUPERVISOR_CLEANRL_WORKER=1`

- Isolated supervisor protobuf (fault isolation)
  - New proto package: `gym_gui/services/jason_supervisor/proto/`
  - File: `supervisor.proto` with package `gymgui.supervisor` (separate from `gymgui.trainer`).
  - Messages:
    - `SupervisorControlUpdate { run_id, reason, source, params_json, timestamp }`
    - `SupervisorAck { accepted, message }`
  - Generated stubs: `supervisor_pb2.py`, `supervisor_pb2_grpc.py`
  - Rationale: keep Jason control messages independent of trainer RPCs to avoid coupling and enable evolution/versioning.

- Sanity tests
  - `gym_gui/tests/test_supervisor_proto.py` — imports the isolated stubs and instantiates a `SupervisorControlUpdate`.
  - `gym_gui/tests/test_jason_supervisor_cleanrl_worker.py` — protocol conformance for the new worker (id, hooks, abstain semantics).

- Doc alignment
  - Matches the plan in `docs/1.0_DAY_23/TASK_2/JASON_TO_GYM_GUI.md` and runtime control detail in `docs/1.0_DAY_23/TASK_3/JASON_CHANGE_AT_RUNTIME.md`.

## How to try it

- Use the project virtualenv (required for protoc and imports):

```bash
# Activate venv
source .venv/bin/activate

# (Optional) Enable auto‑registration of the new worker in the ActorService
export ENABLE_JASON_SUPERVISOR_CLEANRL_WORKER=1

# (Optional) Regenerate supervisor stubs later
python -m grpc_tools.protoc \
  -I gym_gui/services/jason_supervisor/proto \
  --python_out=gym_gui/services/jason_supervisor/proto \
  --grpc_python_out=gym_gui/services/jason_supervisor/proto \
  gym_gui/services/jason_supervisor/proto/supervisor.proto

# Run only the new tests (if your pytest discovery isn’t auto-wired)
pytest gym_gui/tests/test_supervisor_proto.py -q
pytest gym_gui/tests/test_jason_supervisor_cleanrl_worker.py -q
```

Notes:

- CleanRL wiring remains untouched; the new worker is additive and gated by env var.
- Supervisor messages live outside the trainer package for clear fault isolation.


## Files touched/added

- Protobuf (isolated)
  - `gym_gui/services/jason_supervisor/proto/supervisor.proto`
  - `gym_gui/services/jason_supervisor/proto/supervisor_pb2.py`
  - `gym_gui/services/jason_supervisor/proto/supervisor_pb2_grpc.py`
  - `gym_gui/services/jason_supervisor/proto/__init__.py`

- Worker
  - `gym_gui/workers/jason_supervisor_cleanrl_worker/__init__.py`
  - `gym_gui/workers/jason_supervisor_cleanrl_worker/worker.py`

- Tests
  - `gym_gui/tests/test_supervisor_proto.py`
  - `gym_gui/tests/test_jason_supervisor_cleanrl_worker.py`

## Design choices

- Fault isolation: Supervisor control messages packaged separately (`gymgui.supervisor`). Trainer proto left unchanged.
- Backwards compatibility: the worker is opt‑in; no bootstrap edits; zero impact unless explicitly enabled.
- Simple ACK: `SupervisorAck` is intentionally minimal; richer acks (e.g., applied fields, reasons) can be added later in a versioned message.

## Risks / watchouts

- A legacy copy of `supervisor.proto` may still exist under `gym_gui/services/trainer/proto/` from earlier experimentation. It is unused; remove it to avoid confusion.
- No trainer control RPC yet; today, `SupervisorControlUpdate` is for structured events/tests. A thin adapter can bridge to trainer once a control endpoint exists.
- UI overlays for supervisor status are not yet wired to live updates in this task; see next steps.

## Next steps (Task 3 / onward)

1. Remove legacy `trainer/proto/supervisor.proto` (if present) to prevent drift.
2. Wire UI overlays
   - Update Control Panel labels from `JasonSupervisorService.snapshot()` on a timer/signal.
   - Annotate `LiveTelemetryTab` rows for supervised actions/events (distinct color/icon).
3. Control delivery path
   - Implement an adapter that transforms validated `TrainerControlUpdate` (ValidationService) into `SupervisorControlUpdate` messages, and later into trainer control RPCs when available.
   - Honor backpressure/credit hints to avoid flooding.
4. Supervision tests
   - Unit tests for `JasonSupervisorService.apply_control_update()` and rollback logging.
   - UI overlay tests (Qt signal/slot or presenter polling).
5. Bridge client stubs (optional now)
   - If using the Jason gRPC bridge, add a minimal client in `services/jason_bridge/` that emits supervisor updates and consumes acks.
6. Ops & docs
   - Extend `JASON_CHANGE_AT_RUNTIME.md` with the supervisor message examples and UI annotations.
   - Record proto versioning policy (`gymgui.supervisor.v1`) if we expect frequent changes.

— End of Task 2 —
