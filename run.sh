#!/usr/bin/env bash
# Launch trainer daemon and Gym GUI in one go
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load environment variables from .env file
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p var/logs var/trainer

cleanup() {
  if [ "${REUSED_DAEMON:-0}" -eq 0 ] && ps -p "${DAEMON_PID:-0}" >/dev/null 2>&1; then
    kill "$DAEMON_PID" 2>/dev/null || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# Clean up any stale or zombie daemon processes before checking
STALE_PIDS="$(pgrep -f 'gym_gui.services.trainer_daemon' || true)"
if [ -n "$STALE_PIDS" ]; then
  for pid in $STALE_PIDS; do
    # Check if process is zombie (defunct) or not responding
    if ps -p "$pid" -o stat= 2>/dev/null | grep -q 'Z'; then
      echo "Cleaning up zombie daemon process (PID: $pid)..."
      kill -9 "$pid" 2>/dev/null || true
    elif ! timeout 1 bash -c "kill -0 $pid 2>/dev/null"; then
      echo "Cleaning up stale daemon process (PID: $pid)..."
      kill -9 "$pid" 2>/dev/null || true
    else
      echo "Trainer daemon already running (PID: $pid). Reusing."
      DAEMON_PID=$pid
      REUSED_DAEMON=1
    fi
  done
fi

# Remove stale PID file if it exists
rm -f var/trainer/trainer.pid 2>/dev/null || true

# Start daemon if not already reusing an existing one
if [ "${REUSED_DAEMON:-0}" -eq 0 ]; then
  echo "Starting trainer daemon..."
  QT_DEBUG_PLUGINS=0 \
  "$PYTHON_BIN" -m gym_gui.services.trainer_daemon > var/logs/trainer_daemon.log 2>&1 &
  DAEMON_PID=$!
  REUSED_DAEMON=0
fi

echo "Waiting for trainer daemon to accept connections..."
for attempt in {1..10}; do
  if "$PYTHON_BIN" - <<'PY'
import asyncio
import grpc

async def check():
    channel = grpc.aio.insecure_channel("127.0.0.1:50055")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=2.0)
        success = True
    except Exception:
        success = False
    await channel.close()
    return success

if not asyncio.run(check()):
    raise SystemExit(1)
PY
  then
    echo "Trainer daemon is ready."
    break
  fi
  if [ "$attempt" -eq 10 ]; then
    echo "Trainer daemon did not become ready in time. See var/logs/trainer_daemon.log." >&2
    exit 1
  fi
  sleep 1
done

echo "Launching Gym GUI..."
QT_API=pyqt6 QT_DEBUG_PLUGINS=0 "$PYTHON_BIN" -m gym_gui.app
