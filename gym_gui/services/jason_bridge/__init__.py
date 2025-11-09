"""Jason↔gym_gui bridge server.

This package exposes a minimal gRPC service for Jason agents to send
supervisor control updates into gym_gui. It is independent of any
example projects under 3rd_party and is disabled by default unless
explicitly enabled via environment/config.
"""

from .server import JasonBridgeServer

__all__ = ["JasonBridgeServer"]
