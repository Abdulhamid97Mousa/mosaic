#!/usr/bin/env bash
# Demo script: start Python bridge server then run Java grpc-bridge-example.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_VENV="$ROOT/.venv"
REQ_FILE="$ROOT/requirements.txt"
JAVA_PROJECT="$ROOT/JASON_java_project"
PY_SERVER_LOG="/tmp/py_bridge_server.log"
PY_SERVER_PID="/tmp/py_bridge_server.pid"
PORT=50051
HOST=127.0.0.1

if [ ! -d "$PY_VENV" ]; then
  python -m venv "$PY_VENV"
fi
source "$PY_VENV/bin/activate"
python -m pip install -r "$REQ_FILE"

# Regenerate Python stubs after any proto change.
chmod +x "$ROOT/tools/generate_python_protos.sh"
"$ROOT/tools/generate_python_protos.sh"

# Start Python server (insecure dev mode)
python "$ROOT/python_project/server.py" --host "$HOST" --port "$PORT" --insecure >"$PY_SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PY_SERVER_PID"
echo "Started Python server PID=$SERVER_PID (log: $PY_SERVER_LOG)"

# Build Java side and run demo MAS
pushd "$JAVA_PROJECT" >/dev/null
./gradlew :grpc-bridge-example:run --args='src/main/resources/grpc_bridge.mas2j' || true
popd >/dev/null

# Shutdown
kill "$SERVER_PID" 2>/dev/null || true
rm -f "$PY_SERVER_PID"
echo "Demo complete; server stopped."