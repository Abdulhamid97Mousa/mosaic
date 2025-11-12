from __future__ import annotations

import os
import json
import grpc

from gym_gui.services.bootstrap import bootstrap_default_services
from gym_gui.config.settings import reload_settings
from gym_gui.services.jason_bridge import JasonBridgeServer
from gym_gui.services.jason_supervisor import JasonSupervisorService
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.jason_bridge import bridge_pb2, bridge_pb2_grpc  # type: ignore
from gym_gui.services.jason_supervisor.proto import supervisor_pb2  # type: ignore


def _ensure_server() -> JasonBridgeServer:
    # Force env so bootstrap starts server
    os.environ["JASON_BRIDGE_ENABLED"] = "1"
    # Skip spawning external trainer daemon during tests
    os.environ["GYM_GUI_SKIP_TRAINER_DAEMON"] = "1"
    # Refresh settings cache after env changes
    reload_settings()
    bootstrap_default_services()
    locator = get_service_locator()
    server = locator.require(JasonBridgeServer)
    assert server.is_running()
    return server


def _make_stub() -> bridge_pb2_grpc.JasonBridgeStub:  # type: ignore
    channel = grpc.insecure_channel("127.0.0.1:50555")
    return bridge_pb2_grpc.JasonBridgeStub(channel)  # type: ignore


def test_apply_control_update_via_bridge():
    _ensure_server()
    stub = _make_stub()

    params = json.dumps({"epsilon": 0.05})
    update = supervisor_pb2.SupervisorControlUpdate(  # type: ignore
        run_id="run_bridge",
        reason="bridge_update",
        source="jason_bridge_test",
        params_json=params,
    )
    ack = stub.ApplyControlUpdate(update)
    assert ack.accepted is True

    # Supervisor state should reflect applied action
    locator = get_service_locator()
    supervisor = locator.require(JasonSupervisorService)
    snap = supervisor.snapshot()
    assert snap["last_action"] == "bridge_update"


def test_apply_control_update_rejects_bad_json():
    _ensure_server()
    stub = _make_stub()
    # Invalid JSON in params_json
    update = supervisor_pb2.SupervisorControlUpdate(  # type: ignore
        run_id="run_bad_json",
        reason="bad_json",
        source="jason_bridge_test",
        params_json="{not_valid}",
    )
    ack = stub.ApplyControlUpdate(update)
    assert ack.accepted is False
    assert "invalid_params_json" in ack.message


def test_get_supervisor_status_round_trip():
    _ensure_server()
    stub = _make_stub()
    status = stub.GetSupervisorStatus(bridge_pb2.Empty())  # type: ignore
    assert status.active in (True, False)
    assert isinstance(status.last_action, str)
