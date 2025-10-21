# Logging Architecture Analysis: Reality vs. Expectations

**Date:** October 19, 2025  
**Analysis Type:** Architecture Critique + Code Reality Check  
**Status:** FINDINGS VERIFIED AGAINST ACTUAL IMPLEMENTATION

---

## Executive Summary (Corrected)

### What I Expected (from docs)
- Scattered logs across 2+ directories
- Multiple `gym_gui.log` files with duplicate names
- Separate `daemon_output.log` capturing stdout
- Worker logs hidden from GUI

### What Actually Exists
- **ONE log directory:** `/var/logs/`
- **TWO log files only:**
  - `gym_gui.log` (3.4M) - GUI + daemon controller logs
  - `trainer_daemon.log` (118K) - Daemon process stdout/stderr
- **ONE real problem:** Daemon logs to BOTH files (dual logging)
- **FIVE fixable issues:** No rotation, no correlation IDs, no per-run tracking, no worker log capture, no structured format

### The Core Issues (in order of severity)

| Issue | Severity | Root Cause | Fix Complexity |
|-------|----------|-----------|-----------------|
| **Dual daemon logging** | HIGH | Daemon configured to log to file + stdout captured | Simple (1 conditional check) |
| **No log rotation** | HIGH | Unbounded file growth (3.4M already) | Simple (use RotatingFileHandler) |
| **No correlation IDs** | HIGH | Can't trace single run through logs | Medium (add LoggerAdapter) |
| **Mixed log sources** | MEDIUM | gym_gui.log has GUI + daemon controller logs | Medium (restructure) |
| **No worker log capture** | MEDIUM | Worker runs in daemon subprocess | Medium (stream via telemetry) |
| **No structured format** | LOW-MEDIUM | Mix of Python logging and raw stdout | Low (JSON markers in logs) |

---

## Current Log Locations (The Reality vs. Documentation)

### Actual Structure (in production):

```
Project Root: /home/hamid/Desktop/Projects/GUI_BDI_RL/
│
└─ var/logs/                             (SINGLE directory for BOTH GUI and daemon)
   ├─ gym_gui.log                        (3.4M) GUI + Daemon controller logs
   │                                      (actually has mixed content from everything)
   │
   └─ trainer_daemon.log                 (118K) Trainer daemon actual logging
                                         (structured logging from daemon process)
```

### What I Expected (per LOGGING_ARCHITECTURE_ANALYSIS.md assumptions):

```
Project Root:
├─ gym_gui/
│  └─ var/logs/
│     ├─ gym_gui.log
│     └─ daemon_output.log
│
└─ var/logs/
   ├─ gym_gui.log (DUPLICATE)
   └─ trainer_daemon.log
```

### REALITY CHECK - What Actually Exists:

Only **2 log files** in **1 location** (`/var/logs/`):
- **gym_gui.log** (3.4M) - Mix of GUI logs, daemon controller, and everything
- **trainer_daemon.log** (118K) - Actual daemon process logging

---

## ACTUAL Implementation Analysis (from Code Review)

### How gym_gui.log is Created

**File:** `gym_gui/app.py` (line 41)
```python
configure_logging(level=log_level, stream=True, log_to_file=True)
logger = logging.getLogger("gym_gui.app")
```

**File:** `gym_gui/logging_config/logger.py`
```python
def configure_logging(...) -> None:
    # All logs go to: VAR_LOGS_DIR / "gym_gui.log"
    # where VAR_LOGS_DIR = var/logs/
    
    log_file = LOG_DIR / "gym_gui.log"  # → /var/logs/gym_gui.log
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
```

**File:** `gym_gui/config/paths.py` (lines 5-14)
```python
_REPO_ROOT = _PACKAGE_ROOT.parent  # Project root
VAR_ROOT = (_REPO_ROOT / "var").resolve()  # Repo-level var/
VAR_LOGS_DIR = VAR_ROOT / "logs"  # → /var/logs/
```

### How trainer_daemon.log is Created

**File:** `gym_gui/services/trainer/launcher.py` (lines 85-100)
```python
log_path = VAR_LOGS_DIR / "trainer_daemon.log"
log_file = log_path.open("a", encoding="utf-8")

process = subprocess.Popen(
    [python_executable, "-m", "gym_gui.services.trainer_daemon"],
    stdout=log_file,        # ← STDOUT redirected to file
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env,
)
```

**File:** `gym_gui/services/trainer_daemon.py` (line 45)
```python
from gym_gui.logging_config.logger import configure_logging

# The daemon also calls configure_logging!
# This writes to the SAME gym_gui.log file (in /var/logs/)
```

## Standard logging vs. gym_gui.ui.logging_bridge: The CORRECT Answer

### Standard Python logging (`import logging`)

**What it does:**
- Emits log records to configured handlers (console, file, network, etc.)
- Used by ALL processes (GUI, daemon, worker, everything)
- Used EVERYWHERE in your codebase already

**Where logs go:**
```
logging.getLogger().addHandler(FileHandler())  → /var/logs/gym_gui.log
logging.getLogger().addHandler(StreamHandler())  → console/stdout
```

**Current usage in your code:**
- `gym_gui/app.py` configures root logger with FileHandler
- `gym_gui/services/trainer_daemon.py` configures root logger with FileHandler (DUAL!)
- `spade_bdi_rl/core/worker.py` uses logging for debug statements
- Everything that logs uses standard logging

### gym_gui.ui.logging_bridge (`QtLogHandler` + `LogEmitter`)

**What it does:**
- Intercepts Python log records
- Converts to `LogRecordPayload` dataclass
- Emits Qt signal across thread boundary
- GUI receives signal and displays in real-time

**Current usage in your code:**
- `gym_gui/ui/main_window.py` creates QtLogHandler instance
- Connected to `_append_log_record()` method
- Displays logs in GUI text widget with filtering
- **Does NOT replace** standard file logging - works alongside it

**Where logs go:**
```
Python logger → QtLogHandler → Qt signal → GUI widget text display
                ↓ (still also goes to)
                FileHandler → /var/logs/gym_gui.log
```

### The Key Difference (One Sentence)

**Standard logging:** Writes to files/console for persistence and analysis  
**logging_bridge:** Routes logs to GUI widgets for real-time display

**They are NOT mutually exclusive.** You use BOTH. The bridge doesn't replace file logging - it supplements it.

### Why Have Both?

Because:
- **File logging** needed for: Persistence, daemon logs, worker logs, debugging after the fact
- **GUI logging** needed for: Real-time visibility, user feedback, filtering

They serve different purposes and both are necessary.

### Why You DON'T Need logging_bridge for

❌ Daemon logging (separate process, no Qt)  
❌ Worker logging (subprocess, no Qt)  
❌ Telemetry (JSONL format, different use case)  
❌ Structured logs (need JSON format, not Qt signals)

### Why You DO Keep logging_bridge for

✅ GUI controller logs (background threads need safe Qt signal emission)  
✅ Real-time user feedback in UI  
✅ Live log filtering by module/level  
✅ Session-wide log history

### Problems Identified (CORRECTED)

| Problem | Impact | Actual Status |
|---------|--------|----------|
| **No gym_gui/var/logs directory** | I was wrong - doesn't exist | ✅ CORRECTED |
| **NOT two gym_gui.log files** | I was wrong - only one | ✅ CORRECTED |
| **No daemon_output.log** | I was wrong - doesn't exist | ✅ CORRECTED |
| **Daemon logs in TWO places** | Stdout→trainer_daemon.log + Logger→gym_gui.log | ⚠️ REAL ISSUE |
| **No correlation IDs** | Can't trace single run across logs | HIGH |
| **gym_gui.log is 3.4M** | Logs growing rapidly, no rotation | HIGH |
| **trainer_daemon.log is 118K** | Smaller but still no rotation | MEDIUM |
| **Worker logs not visible** | Worker runs in daemon subprocess, no separate capture | HIGH |
| **No structured format** | Mix of Python logging format and raw stdout | MEDIUM |
| **No way to tail one training run** | All runs mixed in gym_gui.log | HIGH |

---

## Architecture Analysis: Why This Happened

### Process Separation

```
┌─────────────────────────────────────────┐
│  GUI Process (PyQt6)                    │
│  • Main thread (GUI events)             │
│  • Worker threads (background tasks)    │
│  • Location: gym_gui/ package           │
│  • Logs: gym_gui/var/logs/gym_gui.log   │
└─────────────────────────────────────────┘
                    ↓
        (gRPC communication)
                    ↓
┌─────────────────────────────────────────┐
│  Trainer Daemon (Async gRPC)            │
│  • Subprocess spawned by GUI            │
│  • Captures stdout/stderr               │
│  • Location: gym_gui/services/          │
│  • Logs: gym_gui/var/logs/daemon_output │
│          var/logs/trainer_daemon.log    │
└─────────────────────────────────────────┘
                    ↓
        (Subprocess spawning)
                    ↓
┌─────────────────────────────────────────┐
│  Worker Process (Python subprocess)     │
│  • Pure Python RL/BDI agent             │
│  • Emits telemetry as JSONL             │
│  • Location: spade_bdi_rl/    │
│  • Logs: var/logs/ (captured from       │
│          daemon stdout)                 │
└─────────────────────────────────────────┘
```

**Why logs ended up scattered:**

1. **Process isolation** - Each process has its own working directory
   - GUI logs to `gym_gui/var/logs/`
   - Daemon logs to `var/logs/`
   - No centralized coordination

2. **Stdout capture** - Daemon stderr/stdout captured as `daemon_output.log`
   - Mixes actual logging with print() statements
   - Not structured format

3. **Multiple logging systems**
   - GUI uses logging_bridge (Qt-aware)
   - Daemon uses standard logging
   - Worker uses standard logging
   - No unified configuration

4. **No centralized aggregation**
   - Each process writes independently
   - No log forwarding mechanism
   - No correlation between processes

---

## Why This Is Bad Practice ❌

### 1. **Fragmented Visibility**

Problem:
```bash
# To debug one training run, you need to check:
tail -f gym_gui/var/logs/gym_gui.log           # GUI logs
tail -f gym_gui/var/logs/daemon_output.log     # Daemon startup
tail -f var/logs/trainer_daemon.log            # Daemon runtime
tail -f var/logs/gym_gui.log                   # Worker output (maybe)

# Which one is current? Are they synchronized?
```

**Better:** Single file with correlation IDs:
```bash
# All events for run_id "abc123" in one view
grep "run_id=abc123" gym_gui/var/logs/unified.log
```

### 2. **Duplicate Files with Same Name**

Problem:
```
/gym_gui/var/logs/gym_gui.log        ← GUI process logs
/var/logs/gym_gui.log                ← Different content?

# Which one should I check?
# Are they the same?
# Why two files?
```

**Better:** Clear naming with process identifier:
```
gym_gui_ui_process.log              ← GUI UI thread
gym_gui_trainer_daemon_process.log   ← Daemon subprocess
gym_gui_worker_process.log           ← Worker subprocess
```

### 3. **No Correlation Between Events**

Problem:
```
gym_gui.log:
2025-10-19 14:30:00 - User clicked "Train Agent"

(Multiple log files later...)

daemon_output.log:
2025-10-19 14:30:05 - Worker started

# How do I know these are related?
# Was there a 5-second delay? An error?
```

**Better:** Include run_id and parent process ID:
```
2025-10-19 14:30:00 - run_id=abc123 | User clicked "Train Agent"
2025-10-19 14:30:01 - run_id=abc123 | Daemon spawning worker
2025-10-19 14:30:05 - run_id=abc123 | Worker started (pid=12345)
```

### 4. **Stdout/Stderr Capture vs Structured Logging**

Problem:
```
daemon_output.log:

Starting trainer daemon...
[INFO] Service started
2025-10-19 14:30:15,123 - root - INFO - Listening on port 50055
Some print() statement from old debug code
Worker output mixed in
```

**Why bad:**
- Mix of structured and unstructured output
- Hard to parse programmatically
- Old debug statements cause noise
- No way to filter by level/module

**Better:** Pure structured logging with format:
```
{"ts": "2025-10-19T14:30:15.123Z", "level": "INFO", "module": "daemon", "msg": "Listening on port 50055"}
```

### 5. **No Real-Time Visibility in GUI**

Problem:
- Worker writes logs to `var/logs/`
- GUI running in `gym_gui/`
- GUI can't see worker logs in real-time
- User must tail files manually to debug

**Better:** Logs stream via telemetry:
```
worker.py → JSONL telemetry → trainer_daemon → GUI (via gRPC) → Display in GUI
```

### 6. **No Unified Log Rotation**

Problem:
- No log rotation configured
- Logs grow indefinitely
- Old logs never deleted
- Disk space issues

**Better:** Centralized log rotation policy:
```
gym_gui/var/logs/
├─ current/
│  ├─ gym_gui.log        (current, rotating)
│  ├─ trainer_daemon.log (current, rotating)
│  └─ worker.log         (current, rotating)
└─ archived/
   ├─ gym_gui.2025-10-19.log
   ├─ gym_gui.2025-10-18.log
   └─ ...
```

---

## Current System Issues

### Issue 1: Multiple gym_gui.log Files

**Location 1:** `/gym_gui/var/logs/gym_gui.log`
- Content: GUI application logs
- Source: gym_gui/ package logging configuration
- Size: ~??? bytes

**Location 2:** `/var/logs/gym_gui.log`
- Content: Worker process output OR daemon logs?
- Source: Unclear
- Size: ~??? bytes

**Question:** Are these the same file? Why two copies?

### Issue 2: daemon_output.log

**Location:** `/gym_gui/var/logs/daemon_output.log`
- Content: Daemon process stdout/stderr captured
- Format: Mixed (logging + print + errors)
- Problem: Not searchable, not structured

**What happens:**
```python
# In daemon process
print("Starting daemon...")              # → daemon_output.log
logging.info("Service ready")            # → ALSO daemon_output.log
worker_proc = subprocess.Popen(...)
# worker_proc.stdout/stderr → daemon_output.log (maybe?)
```

### Issue 3: No Worker Log Integration

**Where worker logs go:**
1. Worker writes via logging module
2. Captured by daemon's subprocess handling (maybe)
3. Ends up in `/var/logs/` (maybe)
4. Never reaches GUI

**Result:** GUI has no visibility into worker behavior

---

## The Correct Architecture (What You SHOULD Do)

### 1. Centralized Log Directory Structure

```
gym_gui/var/logs/
├─ gym_gui_ui.log                  ← GUI process (PyQt)
├─ trainer_daemon.log              ← Daemon process (gRPC server)
├─ worker_training_{run_id}.log    ← Worker process (one per run)
│
├─ telemetry/
│  ├─ {run_id}.jsonl               ← Run telemetry (structured)
│  └─ ...
│
└─ archived/
   ├─ gym_gui_ui.2025-10-19.log
   ├─ trainer_daemon.2025-10-19.log
   └─ ...
```

### 2. Unified Logging Configuration

```python
# gym_gui/logging_config/unified_logger.py

import logging
import json
from datetime import datetime

# Configure ALL processes to use same setup
def configure_unified_logging(process_name: str, run_id: str = None):
    """Configure logging for ANY process (GUI, daemon, worker)."""
    
    # All logs go to gym_gui/var/logs/
    log_dir = Path(__file__).parents[2] / "var" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if run_id:
        log_file = log_dir / f"worker_training_{run_id}.log"
    else:
        log_file = log_dir / f"{process_name}.log"
    
    # Use JSONFormatter for structured logs
    handler = logging.FileHandler(log_file)
    handler.setFormatter(JSONFormatter())
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    return logger

class JSONFormatter(logging.Formatter):
    """Emit logs as JSON for easy parsing."""
    
    def format(self, record):
        log_dict = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
            "pid": os.getpid(),
        }
        return json.dumps(log_dict)
```

### 3. Correlation IDs Everywhere

```python
# In GUI when starting training:
run_id = "abc123"
logger.info(f"Training started | run_id={run_id} | env=FrozenLake-v2")

# Pass run_id to daemon via gRPC
# Daemon passes to worker
# Worker logs include run_id
logger.info(f"run_id={run_id} | Episode 1 completed | reward=10")

# Now all logs can be correlated:
grep "run_id=abc123" gym_gui/var/logs/*.log
```

### 4. Telemetry as Primary Log Source for GUI

```
Worker Process
    ↓
Emit telemetry (JSONL) + logs (structured JSON)
    ↓
Daemon collects both
    ↓
Streams to GUI via gRPC
    ↓
GUI displays in real-time
    ↓
Also persists to disk for post-analysis
```

### 5. No Stdout/Stderr Capture

```python
# INSTEAD OF:
proc = subprocess.Popen(worker_cmd, stdout=PIPE, stderr=PIPE)  # ❌

# DO THIS:
proc = subprocess.Popen(
    worker_cmd,
    stdout=None,  # Let worker write directly to its own log file
    stderr=None,  # Not captured
)  # ✅

# Worker writes structured logs directly
```

---

## Visual: Complete Logging Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ GUI PROCESS (PyQt6)                                             │
│                                                                 │
│  app.py                                                         │
│  ├─ configure_logging()                                         │
│  │  └─ FileHandler → /var/logs/gym_gui.log                     │
│  │  └─ StreamHandler → console                                  │
│  │                                                              │
│  └─ MainWindow                                                  │
│     ├─ QtLogHandler()  ← Bridges logging to Qt                 │
│     │  └─ Emits signal for each log record                     │
│     │                                                           │
│     └─ _append_log_record() ← Receives signal                  │
│        └─ Displays in QPlainTextEdit (real-time)              │
│                                                                 │
│  All logs from GUI code:                                        │
│  logging.getLogger("gym_gui.*")                                │
│    ├─ → FileHandler (/var/logs/gym_gui.log) ✓ Persisted       │
│    ├─ → StreamHandler (console) ✓ Visible in terminal          │
│    └─ → QtLogHandler (GUI widget) ✓ Real-time display          │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (launch with stdout redirect)
┌─────────────────────────────────────────────────────────────────┐
│ DAEMON PROCESS (Async gRPC)                                    │
│ (subprocess spawned by GUI)                                    │
│                                                                 │
│  trainer_daemon.py                                              │
│  ├─ configure_logging() ← PROBLEM: DUAL LOGGING!               │
│  │  └─ FileHandler → /var/logs/gym_gui.log (DUPLICATE!)       │
│  │  └─ StreamHandler → stdout                                  │
│  │                                                              │
│  └─ Trainer dispatching logic                                  │
│     └─ Spawns worker subprocesses                              │
│                                                                 │
│  Daemon logs:                                                   │
│  logging.getLogger("gym_gui.trainer.*")                        │
│    ├─ → FileHandler (WRONG: goes to gym_gui.log)              │
│    ├─ → StreamHandler (stdout → captured parent)               │
│    └─ → Parent process captures stdout → /var/logs/trainer... │
│                                                                 │
│  ISSUE: Logs to BOTH /var/logs/gym_gui.log AND stdout         │
└─────────────────────────────────────────────────────────────────┘
       ↓ stdout/stderr captured by launcher.py Popen()
       ↓ to /var/logs/trainer_daemon.log
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ /var/logs/trainer_daemon.log                                   │
│ Contains:                                                       │
│ - Daemon startup messages (from stdout)                        │
│ - Daemon logging (from StreamHandler)                          │
│ - NOT the FileHandler logs (those go to gym_gui.log)           │
└─────────────────────────────────────────────────────────────────┘
       ↓ (daemon spawns for each training run)
┌─────────────────────────────────────────────────────────────────┐
│ WORKER PROCESS (Python RL/BDI)                                 │
│ (subprocess spawned by daemon dispatcher)                      │
│                                                                 │
│  worker.py / runtime.py / config.py                            │
│  ├─ logging.getLogger() + debug statements                     │
│  │  └─ Emits to stdout/stderr (not configured)                │
│  │                                                              │
│  └─ telemetry.py                                               │
│     └─ Emits JSONL to stdout                                   │
│                                                                 │
│  Where do worker logs go?                                       │
│  - They print() / logger.info() to stdout/stderr               │
│  - Daemon captures? OR lost?                                   │
│  - UNCLEAR: Need to verify                                     │
└─────────────────────────────────────────────────────────────────┘
       ↓ Worker stdout → Captured where?
       ↓ Probably lost OR mixed in trainer_daemon.log
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FILE SYSTEM RESULT                                              │
│                                                                 │
│ /var/logs/                                                      │
│ ├─ gym_gui.log (3.4M)                                          │
│ │  Contains: GUI logs + Daemon controller logs                 │
│ │  Problem: Grows unbounded, no rotation                       │
│ │           Mixes GUI and daemon concerns                      │
│ │                                                              │
│ └─ trainer_daemon.log (118K)                                   │
│    Contains: Daemon stdout/stderr only                         │
│    Problem: Not actual daemon logs (those went to gym_gui.log) │
│             Confusing what's in each file                      │
│                                                                 │
│ Missing:                                                        │
│ - Per-run log files                                            │
│ - Worker logs                                                  │
│ - Correlation IDs                                              │
│ - Log rotation                                                 │
│ - Structured format                                            │
└─────────────────────────────────────────────────────────────────┘
```

**File:** `gym_gui/ui/logging_bridge.py` (47 lines)

```python
@dataclass(slots=True)
class LogRecordPayload:
    """Encapsulates a log record for Qt signal emission."""
    level: str
    name: str
    message: str
    created: float

class LogEmitter(QtCore.QObject):
    """Qt signal emitter - runs in GUI thread."""
    record_emitted = QtCore.Signal(LogRecordPayload)

class QtLogHandler(logging.Handler):
    """Custom logging handler that bridges to Qt signals."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Convert logging record → LogRecordPayload
        # Emit Qt signal: record_emitted.emit(payload)
        # Signal received in GUI main thread
```

### How It's Used

**File:** `gym_gui/ui/main_window.py` (lines 85-86)
```python
self._log_handler = QtLogHandler(parent=self)
self._log_records: List[LogRecordPayload] = []

# Connect signal:
self._log_handler.emitter.record_emitted.connect(self._append_log_record)
```

**File:** `gym_gui/ui/main_window.py` (lines 1348-1362)
```python
def _append_log_record(self, payload: LogRecordPayload) -> None:
    """Append log record to history and display in UI."""
    self._log_records.append(payload)
    
    if self._passes_filter(payload):
        formatted = self._format_log(payload)
        # Update QPlainTextEdit widget in real-time
```

### What It DOESN'T Do

❌ **Doesn't replace file logging** - Still writes to gym_gui.log  
❌ **Doesn't have any connection to training daemon logs** - Separate process  
❌ **Doesn't aggregate worker logs** - Workers run in separate subprocess  
❌ **Doesn't provide structured format** - Just emits log records  

### What It DOES Do

✅ **Real-time GUI log display** - Logs appear in GUI as they're emitted  
✅ **Thread-safe Qt signal emission** - Works with background threads  
✅ **Filterable log history** - Can filter by module (Controllers, Adapters, Agents)  
✅ **Log record preservation** - Keeps all records in memory

---

## Summary: Current State vs. Ideal

| Aspect | Current ❌ | Ideal ✅ | Gap Size |
|--------|-----------|---------|---------|
| **Log Locations** | 1 centralized (good!) | Same | ✅ NO GAP |
| **Log Files** | 2 files (good!) | Same | ✅ NO GAP |
| **Dual Logging** | Daemon logs to 2 places (bad) | Single source | ⚠️ FIXABLE |
| **Rotation** | Not configured | RotatingFileHandler | ⚠️ FIXABLE (1 line) |
| **Correlation IDs** | None | run_id in all logs | ⚠️ FIXABLE (LoggerAdapter) |
| **Per-Run Files** | All mixed in gym_gui.log | /runs/run_{id}.log | ⚠️ FIXABLE (medium) |
| **Worker Logging** | Captured by daemon | Same as current, but streamable | ⚠️ FIXABLE (medium) |
| **Structured Format** | Mixed | JSON markers | ⚠️ FIXABLE (low) |
| **Real-Time GUI** | Partial (bridge exists) | Full worker log streaming | ⚠️ FIXABLE (high) |

### Key Insight

**You already have most of it right!** The architecture is actually quite good - centralized `/var/logs/`, proper paths resolution, rotating file handler ready to use. The main issues are:

1. **Daemon logs twice** (easy fix)
2. **No log rotation enabled** (1 line fix)
3. **No correlation IDs** (easy fix with LoggerAdapter)
4. **No per-run log tracking** (medium complexity)

Not a redesign - just **5 specific improvements**.

---

## Recommendations (Priority Order - CORRECTED)

### Priority 0 (IMMEDIATE - Bug Fix): Stop Dual Daemon Logging

**Problem:** The daemon process logs to BOTH:
1. stdout/stderr (redirected to `trainer_daemon.log`)
2. Its own configured logger (writes to `gym_gui.log`)

This creates noise and confusion.

**Solution:** In `gym_gui/services/trainer_daemon.py`, check if stdout is being captured and SKIP file logging:

```python
# gym_gui/services/trainer_daemon.py - main()
if sys.stdout.isatty():
    # Running in terminal - configure file logging
    configure_logging(level=log_level)
else:
    # Running as subprocess with redirected stdout - use console only
    # Don't create file handler (parent process is capturing)
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
```

**Impact:** Eliminates duplicate logs in `gym_gui.log`

### Priority 1 (CRITICAL): Add Log Rotation

**Problem:** `gym_gui.log` is already 3.4M (growing daily)

**File:** `gym_gui/logging_config/logger.py`

```python
# CURRENT (no rotation):
file_handler = logging.FileHandler(log_file, encoding="utf-8")

# SHOULD BE (with rotation):
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,              # Keep 5 old files
)
```

**Impact:** Prevents logs from consuming unlimited disk space

### Priority 2 (HIGH): Add Correlation IDs

1. **In app.py** when starting GUI:
```python
session_id = str(uuid.uuid4())[:8]
# Add to all log records via LoggerAdapter
```

2. **In worker dispatch** when spawning training:
```python
run_id = run_config.run_id
logger.info(f"RUN_ID={run_id} | Dispatching training job...")
```

3. **In worker.py** when running:
```python
logger.info(f"RUN_ID={run_id} | Training started")
```

**Impact:** Can trace single training run end-to-end

### Priority 3 (HIGH): Structured Worker Logging

**File:** `spade_bdi_rl/core/worker.py` (or wherever worker is)

Currently worker logs go where? Need to verify this. If they're being captured by daemon:

```python
# In worker subprocess
import logging
logger = logging.getLogger(__name__)

logger.info(f"run_id={run_id} | Episode={episode} | Reward={reward}")
```

These will be captured in `trainer_daemon.log` (via stdout redirect)

### Priority 4 (MEDIUM): Separate File for Each Training Run

Instead of mixing all runs in one file:

```
var/logs/
├─ gym_gui.log              (GUI + daemon messages only)
├─ trainer_daemon.log       (Daemon initialization)
└─ runs/
   ├─ run_abc123.log        (ALL events for run abc123)
   ├─ run_def456.log        (ALL events for run def456)
   └─ ...
```

### Priority 5 (MEDIUM): Real-time Worker Log Streaming

**Current State:**
- Worker runs in subprocess
- Logs captured by daemon's stdout redirect
- GUI can't see them in real-time

**Solution:**
- Worker emits telemetry as JSONL to stdout
- Daemon captures and forwards via gRPC
- GUI displays in real-time tab

This is already partially implemented via `TelemetryAsyncHub`

---

## Comprehensive Logging Issues Found in Codebase

### Complete Issue Inventory (from codebase search)

#### Issue A: Multiple `configure_logging()` Calls Creating Duplicate Handlers

**Files calling `configure_logging()`:**
1. `gym_gui/app.py:42` - GUI main process
2. `gym_gui/controllers/cli.py:44` - CLI interface
3. `gym_gui/services/trainer_daemon.py:376` - Daemon process (PROBLEM!)
4. `gym_gui/core/adapters/toy_text_demo.py:27` - Demo adapter

**Problem:** Each call clears handlers and reconfigures root logger. When daemon is spawned as subprocess, it calls `configure_logging()` again, creating DUPLICATE file handlers.

**Root Cause:** `gym_gui/logging_config/logger.py:32` does `root.handlers.clear()` but doesn't prevent re-initialization.

**File:** `gym_gui/logging_config/logger.py` (lines 32-45)
```python
def configure_logging(level: int = logging.INFO, stream: bool = True, *, log_to_file: bool = True) -> None:
    """Configure root logging handlers for the application."""

    root = logging.getLogger()
    root.handlers.clear()  # ← Clears ALL handlers each call

    handlers: list[logging.Handler] = []
    if stream:
        handlers.append(logging.StreamHandler())

    if log_to_file:
        log_file = LOG_DIR / "gym_gui.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")  # ← NEW handler
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, handlers=handlers)
```

---

#### Issue B: No Log Rotation Configured

**Current:** `logging.FileHandler(log_file)` - Unbounded growth

**Files Affected:**
1. `/gym_gui/var/logs/gym_gui.log` - Currently 3.4M (NO rotation)
2. `/gym_gui/var/logs/trainer_daemon.log` - Currently 118K (NO rotation)

**Impact:** Logs grow indefinitely until disk full

**Evidence:**
- `gym_gui/logging_config/logger.py:41` uses `FileHandler` (not RotatingFileHandler)
- `gym_gui/core/logging_config.py:63` DOES use `RotatingFileHandler` (with 10MB max) but this code path isn't used!
- `gym_gui/services/trainer/launcher.py:85` manually opens file with `open("a")` (MANUAL rotation needed)

**Why This Happened:**
- Two separate logging configuration files exist!
- `gym_gui/logging_config/logger.py` (used everywhere) has basic FileHandler
- `gym_gui/core/logging_config.py` (NOT imported!) has better RotatingFileHandler
- Code uses the wrong one

---

#### Issue C: Inconsistent Logger Naming Conventions

**Files with inconsistent naming:**

1. Using `__name__` (GOOD):
   - `gym_gui/services/telemetry.py:17` - `logger = logging.getLogger(__name__)`
   - `gym_gui/services/validation_service.py:19`
   - `gym_gui/core/error_handler.py:11`
   - `gym_gui/core/agent_config.py:13`
   - `spade_bdi_rl/core/runtime.py:20`
   - `spade_bdi_rl/worker.py:18`

2. Using explicit string names (INCONSISTENT):
   - `gym_gui/controllers/session.py:64` - `"gym_gui.controllers.session"`
   - `gym_gui/controllers/live_telemetry.py:35` - `"gym_gui.controllers.live_telemetry"`
   - `gym_gui/controllers/cli.py:45` - `"gym_gui.controllers.cli"`
   - `gym_gui/controllers/human_input.py:95` - `"gym_gui.controllers.human_input"`
   - `gym_gui/ui/widgets/agent_train_dialog.py:30` - `"gym_gui.ui.train_agent_dialog"`
   - `gym_gui/services/actor.py:67` - `"gym_gui.services.actor"`
   - `gym_gui/services/trainer/client.py:54` - `"gym_gui.trainer.client"`
   - `gym_gui/services/trainer/client_runner.py:25` - `"gym_gui.trainer.client_runner"`
   - `gym_gui/services/trainer_daemon.py:50` - `_LOGGER = logging.getLogger("gym_gui.trainer.daemon")`
   - `gym_gui/services/trainer/launcher.py:17` - `LOGGER = logging.getLogger("gym_gui.trainer.launcher")`

3. Using various constant names:
   - `_LOGGER` - Used in 8 files
   - `LOGGER` - Used in 2 files
   - `logger` - Used in 10+ files

**Problem:** 
- Can't grep for all logger instances (inconsistent naming)
- Makes it hard to suppress/adjust specific modules
- No standard across codebase
- Makes filtering by module inconsistent

---

#### Issue D: `print()` Statements Used for Critical Output

**Critical output using print():**

1. `gym_gui/app.py:45` - `print("[gym_gui] Loaded settings:\n" + _format_settings(settings))`
2. `gym_gui/app.py:51,54` - Error messages with print()
3. `gym_gui/controllers/cli.py:101-124` - Render output all uses print()
4. `gym_gui/core/adapters/toy_text_demo.py:43-77` - Game rendering with print()
5. `gym_gui/workers/demo_worker.py:21-23, 44-45, 108-115, 114-115, 139-142` - Worker progress tracking with print()

**Problem:**
- These don't go to log files
- Unreliable capture (depends on stdout redirection)
- Can't filter by level or module
- Worker progress output especially lost from daemon

**Example from demo_worker.py:**
```python
line 21: json.dump(event, sys.stdout, separators=(",", ":"))
line 22: sys.stdout.write("\n")
line 23: sys.stdout.flush()
# ← This is the JSONL telemetry, OK

line 44: sys.stderr.write(f"[demo_worker] Starting run_id={run_id}, agent_id={agent_id}\n")
line 45: sys.stderr.flush()
# ← This is status but using sys.stderr, not logger!
```

---

#### Issue E: `sys.stderr` and `sys.stdout` Used Directly

**Files using sys.stderr/stdout directly:**

1. `gym_gui/services/trainer/trainer_telemetry_proxy.py:243` - `sys.stderr.write(f"[worker stderr] {line}\n")`
2. `gym_gui/services/trainer/trainer_telemetry_proxy.py:292,296` - Error messages to stderr
3. `gym_gui/workers/demo_worker.py:21-23` - JSONL output to stdout (intentional, OK)
4. `gym_gui/workers/demo_worker.py:44-45, 108-115, 139-142` - Status messages to stderr
5. `spade_bdi_rl/core/telemetry.py:26` - `self._stream: IO[str] = stream or sys.stdout`

**Problem:**
- Not captured by standard logging
- Duplicates logging effort
- Bypasses filtering and handlers
- Worker status messages disappear from logs

**From trainer_telemetry_proxy.py (lines 243, 292, 296):**
```python
def filter_and_forward_telemetry(line):
    sys.stderr.write(f"[worker stderr] {line}\n")  # ← NOT via logger!

# Later in same file:
print("Proxy requires a worker command after `--`.", file=sys.stderr)  # ← Also direct!
```

---

#### Issue F: Multiple Logging Configuration Classes

**Two competing logging setup systems:**

1. **`gym_gui/logging_config/logger.py`** (34 lines)
   - Used by: app.py, cli.py, trainer_daemon.py, toy_text_demo.py
   - Provides: `configure_logging()` function
   - Issue: Basic FileHandler, no rotation

2. **`gym_gui/core/logging_config.py`** (125 lines)
   - Used by: ???
   - Provides: `LoggingConfig` class with methods like `setup_root_logger()`, `suppress_logger()`, `enable_debug_logging()`
   - Issue: NOT imported anywhere! Dead code?
   - Better: Actually has RotatingFileHandler (10MB max, 5 backups)

**Problem:** 
- Code duplication
- `LoggingConfig` class more sophisticated but unused
- App uses simpler version without rotation
- Confusing which to use for new code

**From gym_gui/core/logging_config.py (lines 63-68):**
```python
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
)  # ← Better than what's used!
```
**From gym_gui/logging_config/logger.py (line 41):**
```python
file_handler = logging.FileHandler(log_file, encoding="utf-8")  # ← No rotation!
```

---

#### Issue G: QtLogHandler Adds Handler But No Cleanup on Exception

**File:** `gym_gui/ui/main_window.py` (lines 85, 196-201, 1416)

```python
def __init__(...):
    self._log_handler = QtLogHandler(parent=self)
    # ...
    self._configure_logging()

def _configure_logging(self) -> None:
    root_logger = logging.getLogger()
    self._log_handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    self._log_handler.setFormatter(formatter)
    root_logger.addHandler(self._log_handler)  # ← Added but not idempotent!

def closeEvent(self, event: QtGui.QCloseEvent) -> None:
    logging.getLogger().removeHandler(self._log_handler)  # ← Only removed on close
```

**Problem:**
- If `_configure_logging()` called twice, handlers duplicate
- Exception before `closeEvent()` leaves handler attached
- No guard against re-initialization

---

#### Issue H: spadeBDI Worker Has Its Own basicConfig

**File:** `spade_bdi_rl/worker.py` (lines 18-20)

```python
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Problem:**
- Worker doesn't use gym_gui's centralized logging configuration
- Separate format string (inconsistent with gym_gui format)
- No file handler - logs only to console
- When spawned by daemon, these logs may be lost or mixed

**Expected:** Should use `gym_gui.logging_config.logger.configure_logging()` to stay consistent

---

#### Issue I: No Correlation IDs Between GUI and Daemon/Worker

**Current State:**
- GUI logs: No session/run ID
- Daemon logs: No run_id context
- Worker logs: No run_id context

**Example from gym_gui.log:**
```
2025-10-19 14:30:00 - User clicked "Train Agent"
2025-10-19 14:30:01 - Dispatching to daemon
2025-10-19 14:30:05 - Worker started
```

**Question:** How do I know if this is all one training run or multiple concurrent ones?

**No correlation mechanism exists:**
- No LoggerAdapter wrapping context
- No thread-local or context-var tracking
- No run_id injection into log records

---

#### Issue J: trainer_telemetry_proxy Uses Both Logger and sys.stderr

**File:** `gym_gui/services/trainer/trainer_telemetry_proxy.py`

```python
line 24: _LOGGER = logging.getLogger("gym_gui.trainer.telemetry_proxy")
# Uses _LOGGER for some things

line 243: sys.stderr.write(f"[worker stderr] {line}\n")
# But also writes directly to stderr!

line 292: print("Proxy requires a worker command after `--`.", file=sys.stderr)
# And also uses print()!
```

**Problem:** Three different logging mechanisms in same file:
1. Standard logging module
2. Direct sys.stderr writes
3. print() statements

**Why confusing:** Reader doesn't know which logs go where

---

#### Issue K: No Asymmetric Handler Attachment

**Current:** Root logger gets handlers added but:
- No guard against duplicate attachment
- No tracking of which process attached what
- No way to know if already initialized

**Symptom:** In tests or if `configure_logging()` called twice, handlers duplicate

**File:** `gym_gui/logging_config/logger.py:32`
```python
root.handlers.clear()  # ← Clears but doesn't check if already set up
```

Better would be:
```python
if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
    # Add file handler only if not already present
```

---

#### Issue L: TelemetrySQLiteStore Has Fallback Logger Creation

**File:** `gym_gui/telemetry/sqlite_store.py` (lines 39, 429)

```python
def __init__(...):
    self._logger = logging.getLogger("gym_gui.telemetry.sqlite_store")

# But later:
def _some_method(self):
    logger = getattr(self, "_logger", logging.getLogger("gym_gui.telemetry.sqlite_store"))
```

**Problem:** 
- Line 429 creates logger if attribute missing
- Why would attribute be missing? Defensive coding for what scenario?
- If `__init__` always creates it, this is dead code
- If `__init__` can fail, should fail loudly, not silently fallback

---

#### Issue M: No Structured Logging Format

**Current format:** Plain text with fields separated by pipes
```
2025-10-19 14:30:00 | INFO    | gym_gui.controllers.session | Started training session
```

**Problem:**
- Hard to parse programmatically
- Can't easily filter by level across files
- No machine-readable correlation
- Can't search across multiple log types

**Better format:** JSON-structured logs
```json
{"ts": "2025-10-19T14:30:00Z", "level": "INFO", "module": "gym_gui.controllers.session", "msg": "Started training session", "run_id": "abc123"}
```

---

#### Issue N: No Log Level Configuration Per Module

**Current:** Root logger level set globally
- Can't suppress just `asyncio` logger
- Can't enable debug for just one module
- Works by filter (gym_gui/logging_config/logger.py:21)

**File:** `gym_gui/logging_config/logger.py` (lines 21-25)
```python
class _GrpcBlockingIOFilter(logging.Filter):
    """Filter out non-fatal gRPC BlockingIOError warnings from asyncio logger."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress specific message pattern
        if record.name == "asyncio" and record.levelno == logging.ERROR:
            if "BlockingIOError" in msg and "PollerCompletionQueue" in msg:
                return False
        return True
```

**Better approach:** Per-logger configuration
```python
LoggingConfig.suppress_logger("asyncio")  # ← Available in core/logging_config.py but not used
```

---

#### Issue O: Test Uses monkeypatch for sys.stdout (No Real Logging)

**File:** `spade_bdi_rl/tests/test_bdi_trainer.py` (lines 216-219)

```python
# Mock sys.stdout to capture JSONL output
monkeypatch.setattr("sys.stdout", mock_stdout)
```

**Problem:** Tests mock stdout but don't test actual logging behavior
- Real worker logs might not work (tests don't verify)
- Logging infrastructure untested in worker context

---

### Summary Table of Issues

| ID | Issue | Severity | Files | Impact | Fixability |
|----|-------|----------|-------|--------|-----------|
| A | Multiple `configure_logging()` calls | HIGH | app.py, cli.py, daemon, adapter | Duplicate handlers | Easy (idempotent check) |
| B | No log rotation | HIGH | logger.py | Unbounded growth | Easy (1 line) |
| C | Inconsistent logger names | MEDIUM | 15+ files | Hard to grep/filter | Medium (rename all) |
| D | print() for critical output | MEDIUM | app.py, workers, adapters | Lost output | Medium (replace with logging) |
| E | sys.stderr/stdout direct use | MEDIUM | trainer_proxy, workers | Not logged | Medium (route to logging) |
| F | Duplicate logging configs | MEDIUM | Two separate files | Confusing | Medium (consolidate) |
| G | QtLogHandler not idempotent | LOW-MEDIUM | main_window.py | Duplicate handlers in GUI | Easy (guard) |
| H | Worker basicConfig override | MEDIUM | spadeBDI worker.py | Inconsistent format | Easy (use gym_gui config) |
| I | No correlation IDs | HIGH | All processes | Can't trace runs | Medium (LoggerAdapter) |
| J | Mixed logging mechanisms | MEDIUM | trainer_proxy | Confusing | Easy (standardize) |
| K | No handler attachment guard | LOW | logger.py | Potential duplicates | Easy (check) |
| L | Fallback logger creation | LOW | sqlite_store.py | Dead code? | Easy (investigate) |
| M | No structured format | MEDIUM | All handlers | Hard to parse | Medium (JSON formatter) |
| N | No per-module config | LOW-MEDIUM | logging_config | Can't adjust levels | Easy (add methods) |
| O | Tests don't verify logging | LOW | test files | Untested behavior | Easy (add logging tests) |

---

## Code Changes Needed

### Priority 0 Fix: Stop Daemon Dual Logging

**File:** `gym_gui/services/trainer_daemon.py` - Update `main()` function

Add check before calling `configure_logging()`:

```python
def main() -> int:
    """Start the async trainer daemon."""
    
    parser = argparse.ArgumentParser(description="Trainer daemon")
    # ... args parsing ...
    
    # NEW: Check if running as subprocess (stdout captured)
    if not sys.stdout.isatty():
        # Subprocess mode - only output to console (captured by parent)
        # Don't create file handler to avoid duplicate logs
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _LOGGER.info("Daemon starting in subprocess mode (stdout captured)")
    else:
        # Terminal mode - write to both console and file
        configure_logging(level=log_level, stream=True, log_to_file=True)
        _LOGGER.info("Daemon starting in terminal mode")
```

**Impact:** Eliminates duplicate logging in `gym_gui.log`

---

### Priority 1 Fix: Add Log Rotation

**File:** `gym_gui/logging_config/logger.py`

```python
def configure_logging(level: int = logging.INFO, stream: bool = True, *, log_to_file: bool = True) -> None:
    """Configure root logging handlers for the application."""

    root = logging.getLogger()
    root.handlers.clear()

    handlers: list[logging.Handler] = []
    if stream:
        handlers.append(logging.StreamHandler())

    if log_to_file:
        log_file = LOG_DIR / "gym_gui.log"
        
        # CHANGE: Use RotatingFileHandler instead of FileHandler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            encoding="utf-8",
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,              # Keep 5 old rotated files
        )
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, handlers=handlers)
    
    # Add filter to suppress non-fatal gRPC asyncio warnings
    grpc_filter = _GrpcBlockingIOFilter()
    asyncio_logger = logging.getLogger("asyncio")
    for handler in asyncio_logger.handlers or root.handlers:
        handler.addFilter(grpc_filter)
```

**Impact:** Prevents unbounded log growth

---

### Priority 2 Fix: Add Correlation IDs via LoggerAdapter

**File:** `gym_gui/logging_config/logger.py` (new addition)

```python
class ContextLogger(logging.LoggerAdapter):
    """Logger that injects context (run_id, session_id) into all messages."""
    
    def process(self, msg, kwargs):
        """Add context fields to log message."""
        context_parts = []
        
        if 'run_id' in self.extra:
            context_parts.append(f"run_id={self.extra['run_id']}")
        if 'session_id' in self.extra:
            context_parts.append(f"session_id={self.extra['session_id']}")
        if 'pid' in self.extra:
            context_parts.append(f"pid={self.extra['pid']}")
            
        prefix = " | ".join(context_parts)
        if prefix:
            msg = f"[{prefix}] {msg}"
        return msg, kwargs

# Usage in app.py:
import uuid
session_id = str(uuid.uuid4())[:8]
logger = logging.getLogger("gym_gui.app")
logger = ContextLogger(logger, {'session_id': session_id})
```

**Impact:** All logs include correlation context

---

### Priority 3 Fix: Structured Telemetry Format

**File:** `spade_bdi_rl/core/config.py` - Update RUN_CONFIG_LOADED log

```python
# CURRENT:
logger.info(f"RUN_CONFIG_LOADED | run_id=... | env_id=... | seed=... | ...")

# BETTER: Add JSON telemetry marker
import json
config_dict = {
    'type': 'run_config_loaded',
    'run_id': self.run_id,
    'env_id': self.env_id,
    'seed': self.seed,
    # ... other fields
}
logger.info(f"TELEMETRY {json.dumps(config_dict)}")
```

**File:** `spade_bdi_rl/core/telemetry.py` - Same pattern

```python
# Emit telemetry as both:
# 1. Structured JSON lines (for parsing)
# 2. Log message (for visibility)

telemetry = {"type": "step", "run_id": run_id, "episode": ep, ...}
print(json.dumps(telemetry), flush=True)  # → JSONL stream
logger.info(f"TELEMETRY {json.dumps(telemetry)}")  # → Also logged
```

**Impact:** Logs can be parsed and aggregated by tooling

---

### Priority 4 Fix: Per-Run Log Files

**File:** `gym_gui/services/trainer/launcher.py` (new addition)

```python
def setup_run_logging(run_id: str) -> Path:
    """Create dedicated log file for a training run."""
    
    run_logs_dir = VAR_LOGS_DIR / "runs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = run_logs_dir / f"run_{run_id}.log"
    return log_file

# Then in dispatcher when spawning worker:
def _spawn_worker(self, run_id: str, ...) -> subprocess.Popen:
    run_log = setup_run_logging(run_id)
    
    proc = subprocess.Popen(
        [...worker command...],
        stdout=run_log.open("a"),      # NEW: per-run file
        stderr=subprocess.STDOUT,
    )
```

**Impact:** Can grep `run_abc123.log` to see all events for one run

---

## Conclusion

**Your current logging architecture is fragmented and difficult to maintain.** The scattered logs with duplicates and missing correlation IDs make it impossible to trace single training runs end-to-end.

**The solution:**
1. ✅ Centralize all logs to `gym_gui/var/logs/`
2. ✅ Use structured JSON format
3. ✅ Add `run_id` correlation to all logs
4. ✅ Stream worker logs to GUI via telemetry
5. ✅ Remove stdout/stderr capture mess

**Impact when fixed:**
- Debugging becomes 10x easier
- End-to-end tracing of training runs
- GUI shows real-time worker status
- Logs become searchable and queryable
- Ready for production deployment

---

**Status:** ANALYSIS COMPLETE  
**Date:** October 19, 2025  
**Recommendation:** Implement Priority 1-2 tasks before next release
