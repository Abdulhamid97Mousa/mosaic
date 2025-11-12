# Jason ↔ gym_gui Bridge

Lightweight, opt-in gRPC service allowing JASON (Java) agents to interact with gym_gui without touching the trainer wiring. It exposes a small control plane and status surface, and supports an in-process fallback for local/CI stability.

## Enable and configure

The bridge is gated by typed settings sourced from environment variables:

- JASON_BRIDGE_ENABLED ("1" to enable)
- JASON_BRIDGE_HOST (default: 127.0.0.1)
- JASON_BRIDGE_PORT (default: 50555)

These are parsed by `gym_gui.config.settings.get_settings()` and used in `gym_gui.services.bootstrap.bootstrap_default_services()`.

Example:

```bash
# inside your venv
export JASON_BRIDGE_ENABLED=1
export JASON_BRIDGE_HOST=127.0.0.1
export JASON_BRIDGE_PORT=50555
export GYM_GUI_SKIP_TRAINER_DAEMON=1  # optional for faster dev/test
python -m gym_gui.app
```

## RPC surface

- PushPercept(JasonPercept) → SupervisorControlAck
- ApplyControlUpdate(SupervisorControlUpdate) → SupervisorControlAck
- RequestAction(ActionRequest) → ActionResponse (placeholder)
- GetSupervisorStatus(Empty) → SupervisorStatus

Protos live in this package and import the Supervisor control messages.

- Bridge proto: `gym_gui/services/jason_bridge/bridge.proto`
- Supervisor proto: `gym_gui/services/jason_supervisor/proto/supervisor.proto`

## Regenerate stubs

Use the unified script to regenerate both supervisor and bridge stubs into their canonical locations (prevents nested outputs and descriptor pool issues):

```bash
# ensure grpcio-tools installed in the venv
bash tools/generate_protos.sh
```

Generated files:

- `gym_gui/services/jason_supervisor/proto/supervisor_pb2.py`
- `gym_gui/services/jason_supervisor/proto/supervisor_pb2_grpc.py`
- `gym_gui/services/jason_bridge/bridge_pb2.py`
- `gym_gui/services/jason_bridge/bridge_pb2_grpc.py`

## Test locally

```bash
source .venv/bin/activate
export QT_QPA_PLATFORM=offscreen
export JASON_BRIDGE_ENABLED=1
export GYM_GUI_SKIP_TRAINER_DAEMON=1
pytest -q gym_gui/tests/test_jason_bridge_server.py gym_gui/tests/test_jason_bridge_settings.py
```

## Behavior notes

- If TCP bind fails, the server falls back to an in-process channel. Clients still connect to the same `host:port` string via `grpc.insecure_channel`.
- The bridge is isolated from the trainer; failures won’t affect CleanRL wiring.

## Next steps

- Java side: scaffold a MAS client in `3rd_party/jason_worker/` that consumes `bridge.proto` (not the legacy grpc-bridge-example).
- Optional Python client/dispatcher: translate telemetry→percepts and action tokens→environment actions for end-to-end loops.
- TLS/auth: add optional TLS/mTLS configuration flags.
