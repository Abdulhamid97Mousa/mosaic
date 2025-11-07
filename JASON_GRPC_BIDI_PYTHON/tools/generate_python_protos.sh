#!/usr/bin/env bash
set -euo pipefail

# Generate Python gRPC stubs for agent_bridge.proto into python_project/
# Requires: python -m pip install grpcio grpcio-tools protobuf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PROTO_DIR="$ROOT_DIR/proto"
OUT_DIR="$ROOT_DIR/python_project"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH" >&2
  exit 1
fi

python - <<'PY'
import sys
import pkgutil
missing = [m for m in ("grpc", "grpc_tools", "google.protobuf") if not pkgutil.find_loader(m)]
if missing:
    print("Missing packages:", ", ".join(missing), file=sys.stderr)
    sys.exit(2)
PY

python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/agent_bridge.proto"

echo "Python gRPC stubs generated in $OUT_DIR"