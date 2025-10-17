"""Generated gRPC protocol bindings for the trainer daemon."""

from __future__ import annotations

import sys

from . import trainer_pb2 as _trainer_pb2

# gRPC tools emit absolute imports (``import trainer_pb2``). When this package is
# imported as part of ``gym_gui.services.trainer.proto`` the generated modules are
# still located under that namespace, so we register aliases in ``sys.modules`` to
# satisfy those absolute imports without mutating the generated files.
sys.modules.setdefault("trainer_pb2", _trainer_pb2)

from . import trainer_pb2_grpc as _trainer_pb2_grpc

sys.modules.setdefault("trainer_pb2_grpc", _trainer_pb2_grpc)

trainer_pb2 = _trainer_pb2
trainer_pb2_grpc = _trainer_pb2_grpc

__all__ = ["trainer_pb2", "trainer_pb2_grpc"]
