from __future__ import annotations

"""JasonBridgeServer: minimal gRPC server for Jason↔gym_gui.

Provides RPCs:
  - PushPercept: currently logs percepts; future: route to telemetry bus.
  - ApplyControlUpdate: validates + applies via JasonSupervisorService.
  - RequestAction: placeholder returning availability=False or dummy token.
  - GetSupervisorStatus: returns snapshot of supervisor state.

The server is opt-in; enable by setting env var JASON_BRIDGE_ENABLED=1.

Authoritative stubs: gym_gui/services/jason_bridge/{bridge_pb2, bridge_pb2_grpc}.
If you see nested duplicates under this package path, they were produced by a
misconfigured protoc invocation and are tombstoned to prevent accidental import.
"""

import json
import logging
import os
from concurrent import futures
from typing import Optional, Any

import grpc
from google.protobuf import timestamp_pb2

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_SUPERVISOR_EVENT,
    LOG_SERVICE_SUPERVISOR_ERROR,
    LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED,
    LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED,
)
from gym_gui.services.jason_supervisor import JasonSupervisorService
from gym_gui.services.service_locator import get_service_locator


class _InProcessContext:
    def __init__(self) -> None:
        self._code = grpc.StatusCode.OK
        self._details: str = ""

    def set_code(self, code: grpc.StatusCode) -> None:  # type: ignore
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details

    def code(self) -> grpc.StatusCode:  # type: ignore
        return self._code

    def details(self) -> str:
        return self._details


class _InProcessChannel:
    def __init__(self, servicer: "JasonBridgeServicer") -> None:
        self._servicer = servicer

    def unary_unary(self, method: str, **kwargs):
        rpc_name = method.rsplit("/", 1)[-1]
        if not hasattr(self._servicer, rpc_name):
            raise AttributeError(f"Servicer missing RPC {rpc_name}")
        rpc_method = getattr(self._servicer, rpc_name)

        def _call(request, *args, **kw):
            context = _InProcessContext()
            return rpc_method(request, context)

        return _call

    # Minimal stubs for other channel types that may be requested.
    def stream_unary(self, method: str, **kwargs):
        raise NotImplementedError("stream_unary not supported in-process")

    def unary_stream(self, method: str, **kwargs):
        raise NotImplementedError("unary_stream not supported in-process")

    def stream_stream(self, method: str, **kwargs):
        raise NotImplementedError("stream_stream not supported in-process")


_INPROCESS_SERVERS: dict[str, JasonBridgeServicer] = {}
_ORIGINAL_INSECURE_CHANNEL = grpc.insecure_channel


def _inprocess_insecure_channel(target: str, *args, **kwargs):
    servicer = _INPROCESS_SERVERS.get(target)
    if servicer is not None:
        return _InProcessChannel(servicer)
    return _ORIGINAL_INSECURE_CHANNEL(target, *args, **kwargs)


grpc.insecure_channel = _inprocess_insecure_channel


def _register_inprocess_server(target: str, servicer: "JasonBridgeServicer") -> None:
    _INPROCESS_SERVERS[target] = servicer


def _unregister_inprocess_server(target: str) -> None:
    _INPROCESS_SERVERS.pop(target, None)

# Load dependent descriptor FIRST to avoid descriptor pool import errors.
from gym_gui.services.jason_supervisor.proto import supervisor_pb2 as supervisor_pb2  
# Import local generated stubs (bridge) without sys.modules alias hacks
from . import bridge_pb2, bridge_pb2_grpc  

class JasonBridgeServicer(bridge_pb2_grpc.JasonBridgeServicer, LogConstantMixin):  
    def __init__(self, supervisor: JasonSupervisorService) -> None:
        self._supervisor = supervisor
        self._logger = logging.getLogger("gym_gui.jason_bridge")
        self._percepts_buffer: list[Any] = []

    # ---------------- RPCs -----------------
    def PushPercept(self, request: Any, context: grpc.ServicerContext):  
        self._percepts_buffer.append(request)
        # Future: convert into telemetry event
        self.log_constant(
            LOG_SERVICE_SUPERVISOR_EVENT,
            message="percept_received",
            extra={"name": request.name},
        )
        return supervisor_pb2.SupervisorControlAck(  # type: ignore[attr-defined]
            accepted=True, message="percept_buffered"
        )

    def ApplyControlUpdate(
        self, request: Any, context: grpc.ServicerContext
    ):  # type: ignore
        # Decode params_json
        try:
            params = json.loads(request.params_json or "{}")
        except json.JSONDecodeError:
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED,
                message="invalid_params_json",
            )
            return supervisor_pb2.SupervisorControlAck(  # type: ignore[attr-defined]
                accepted=False, message="invalid_params_json"
            )

        accepted = self._supervisor.apply_control_update(
            {
                "run_id": request.run_id,
                "reason": request.reason,
                "source": request.source or "jason_supervisor",
                "params": params,
                # credits placeholder: allow by default
                "available_credits":  self._supervisor._defaults.min_available_credits,  
            }
        )
        if accepted:
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED,
                extra={"reason": request.reason, "source": request.source},
            )
        else:
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED,
                extra={"reason": request.reason, "source": request.source},
            )
        return supervisor_pb2.SupervisorControlAck(  # type: ignore[attr-defined]
            accepted=accepted, message=("ok" if accepted else "rejected")
        )

    def RequestAction(self, request: Any, context: grpc.ServicerContext): 
        return bridge_pb2.ActionResponse(available=False, action_token="", message="not_implemented")  # type: ignore[attr-defined]

    def GetSupervisorStatus(self, request: Any, context: grpc.ServicerContext): 
        snap = self._supervisor.snapshot()
        return bridge_pb2.SupervisorStatus(  # type: ignore[attr-defined]
            active=snap["active"],
            safety_on=snap["safety_on"],
            last_action=snap["last_action"],
            actions_emitted=snap["actions_emitted"],
            last_error=snap["last_error"] or "",
        )


class JasonBridgeServer(LogConstantMixin):
    def __init__(self, host: str = "127.0.0.1", port: int = 50555) -> None:
        self._host = host
        self._port = port
        self._logger = logging.getLogger("gym_gui.jason_bridge")
        self._server: Optional[grpc.Server] = None
        self._bound_target: Optional[str] = None
        self._inprocess_target: Optional[str] = None

    def start(self) -> None:
        if self._server is not None:
            return
        locator = get_service_locator()
        supervisor = locator.require(JasonSupervisorService)
        servicer = JasonBridgeServicer(supervisor)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        bridge_pb2_grpc.add_JasonBridgeServicer_to_server(servicer, server)
        bind_addr = f"{self._host}:{self._port}"
        try:
            port = server.add_insecure_port(bind_addr)
            if port == 0:
                raise RuntimeError(f"Failed to bind to {bind_addr}")
            server.start()
            self._server = server
            self._bound_target = bind_addr
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_EVENT,
                message="jason_bridge_started",
                extra={"bind": bind_addr},
            )
            return
        except RuntimeError as exc:
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_ERROR,
                message="jason_bridge_binding_failed",
                extra={"target": bind_addr, "error": str(exc)},
            )

        _register_inprocess_server(bind_addr, servicer)
        self._inprocess_target = bind_addr
        self._bound_target = bind_addr
        self._logger.info("Jason bridge running in-process socket-less mode")

    def stop(self, grace: float = 2.0) -> None:
        if self._server is not None:
            self._server.stop(grace)
            self._server = None
        if self._inprocess_target is not None:
            _unregister_inprocess_server(self._inprocess_target)
            self._inprocess_target = None
        if self._bound_target is not None:
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_EVENT,
                message="jason_bridge_stopped",
            )
            self._bound_target = None

    def is_running(self) -> bool:
        return self._server is not None or self._inprocess_target is not None


def maybe_start_bridge_from_env() -> Optional[JasonBridgeServer]:
    if os.getenv("JASON_BRIDGE_ENABLED") != "1":
        return None
    server = JasonBridgeServer()
    server.start()
    return server
