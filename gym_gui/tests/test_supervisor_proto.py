from __future__ import annotations

from google.protobuf.timestamp_pb2 import Timestamp

# Import isolated supervisor proto definitions
from gym_gui.services.jason_supervisor.proto import supervisor_pb2  # type: ignore


def test_supervisor_control_update_instantiation():
    ts = Timestamp()
    ts.GetCurrentTime()
    # Access via getattr to satisfy static analysis when stubs are absent
    MsgCls = getattr(supervisor_pb2, "SupervisorControlUpdate", None)
    assert MsgCls is not None, "SupervisorControlUpdate message missing"
    msg = MsgCls(
        run_id="run_123",
        reason="plateau_reanneal",
        source="jason_supervisor",
        params_json='{"epsilon":0.05}',
        timestamp=ts,
    )  # type: ignore[attr-defined]
    assert msg.run_id == "run_123"
    assert msg.reason == "plateau_reanneal"
    assert msg.source == "jason_supervisor"
    assert msg.params_json.startswith("{")
    # Timestamp should be set (non-zero seconds)
    assert msg.timestamp.seconds > 0
