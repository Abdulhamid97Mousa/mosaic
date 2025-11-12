#!/usr/bin/env bash
set -euo pipefail

# Proto generation script for gym_gui Jason supervisor + bridge stubs.
# Usage:
#   source .venv/bin/activate  # ensure grpcio-tools installed
#   bash tools/generate_protos.sh
#
# This regenerates:
#   - gym_gui/services/jason_supervisor/proto/supervisor.proto
#   - gym_gui/services/jason_bridge/bridge.proto
# into their canonical package paths (preventing nested duplicates).

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

SUPERVISOR_DIR="gym_gui/services/jason_supervisor/proto"
BRIDGE_DIR="gym_gui/services/jason_bridge"

cd "$ROOT_DIR"

echo "[protos] Generating supervisor stubs"
"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$SUPERVISOR_DIR" \
  --python_out="$SUPERVISOR_DIR" \
  --grpc_python_out="$SUPERVISOR_DIR" \
  "$SUPERVISOR_DIR/supervisor.proto"

echo "[protos] Generating bridge stubs"
"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$SUPERVISOR_DIR" -I "$BRIDGE_DIR" \
  --python_out="$BRIDGE_DIR" \
  --grpc_python_out="$BRIDGE_DIR" \
  "$BRIDGE_DIR/bridge.proto"

echo "[protos] Done"
