#!/usr/bin/env bash
set -euo pipefail

# Proto generation script for the MOSAIC trainer gRPC service.
# Usage:
#   source .venv/bin/activate  # ensure grpcio-tools installed
#   bash tools/generate_protos.sh
#
# This regenerates:
#   - gym_gui/services/trainer/proto/trainer_pb2.py
#   - gym_gui/services/trainer/proto/trainer_pb2_grpc.py
#   - gym_gui/services/trainer/proto/trainer_pb2.pyi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[protos] Root: $ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[protos] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import grpc_tools" 2>/dev/null; then
  echo "[protos] grpcio-tools not installed. Run: pip install grpcio-tools" >&2
  exit 1
fi

TRAINER_DIR="gym_gui/services/trainer/proto"

cd "$ROOT_DIR"

echo "[protos] Generating trainer stubs"
"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$TRAINER_DIR" \
  --python_out="$TRAINER_DIR" \
  --grpc_python_out="$TRAINER_DIR" \
  --pyi_out="$TRAINER_DIR" \
  "$TRAINER_DIR/trainer.proto"

echo "[protos] Done"
