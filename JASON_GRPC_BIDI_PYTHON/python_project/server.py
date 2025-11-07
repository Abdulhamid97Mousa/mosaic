"""Minimal gRPC server that lets Python policy talk to Jason/Java agents."""
from __future__ import annotations

import argparse
import logging
from concurrent import futures
from typing import Dict

import grpc

import agent_bridge_pb2
import agent_bridge_pb2_grpc


LOGGER = logging.getLogger("agent_bridge.server")


class AgentBridgeService(agent_bridge_pb2_grpc.AgentBridgeServicer):
    """Simple, synchronous service that stores percepts and serves actions.

    Updated to populate structured metadata (meta entries) and status codes.
    """

    def __init__(self) -> None:
        self._latest_percepts: Dict[str, str] = {}

    # RPCs -----------------------------------------------------------------
    def PushPercept(self, request: agent_bridge_pb2.Percept, context: grpc.ServicerContext):  # noqa: N802 (gRPC naming)
        LOGGER.info("<- percept %s: %s", request.agent, request.payload)
        self._latest_percepts[request.agent] = request.payload
        return agent_bridge_pb2.Ack(ok=True, detail="percept stored")

    def RequestAction(self, request: agent_bridge_pb2.ActionRequest, context: grpc.ServicerContext):  # noqa: N802
        LOGGER.info("-> action request from %s (%s)", request.agent, request.context)
        percept = self._latest_percepts.get(request.agent, "")
        action = self._decide_action(request.context, percept)
        metadata = f"ctx={request.context}|percept={percept}"  # legacy string form
        meta_entries = [
            agent_bridge_pb2.MetaEntry(key="ctx_len", value=str(len(request.context or ""))),
            agent_bridge_pb2.MetaEntry(key="percept_len", value=str(len(percept))),
        ]
        return agent_bridge_pb2.ActionResponse(
            action=action,
            metadata=metadata,
            meta=meta_entries,
            status=agent_bridge_pb2.ACTION_STATUS_OK,
        )

    # New streaming RPC example: emit a short sequence of incremental actions.
    def StreamActions(self, request: agent_bridge_pb2.ActionRequest, context: grpc.ServicerContext):  # noqa: N802
        for i in range(3):
            yield agent_bridge_pb2.ActionResponse(
                action=f"step_{i}",
                metadata="stream",  # legacy
                meta=[agent_bridge_pb2.MetaEntry(key="index", value=str(i))],
                status=agent_bridge_pb2.ACTION_STATUS_OK,
            )

    # Helpers --------------------------------------------------------------
    @staticmethod
    def _decide_action(context: str, percept: str) -> str:
        if "battery" in percept:
            return "recharge"
        if "temperature" in percept:
            return "cool_down"
        if "search" in context:
            return "explore"
        return "idle"


def serve(host: str, port: int) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_bridge_pb2_grpc.add_AgentBridgeServicer_to_server(AgentBridgeService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    LOGGER.info("Python AgentBridge server listening on %s:%d", host, port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server...")
        server.stop(grace=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: %(default)s)")
    parser.add_argument("--port", default=50051, type=int, help="Bind port (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[PY] %(message)s")
    args = parse_args()
    serve(args.host, args.port)
