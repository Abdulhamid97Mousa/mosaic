from __future__ import annotations

"""SQLite-backed registry for trainer runs and resource bookkeeping."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import sqlite3
import threading
from threading import RLock
from typing import Iterable, Optional, cast

from gym_gui.config.paths import VAR_TRAINER_DB, ensure_var_directories
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus, get_bus


_LOGGER = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Lifecycle states for the MOSAIC trainer FSM."""

    INIT = "init"
    HANDSHAKE = "handshake"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    FAULTED = "faulted"
    TERMINATED = "terminated"

    @classmethod
    def from_proto(cls, proto_value: int) -> "RunStatus":
        """Convert protobuf status integer to RunStatus enum (defaults to INIT)."""

        _PROTO_TO_ENUM = {
            0: cls.INIT,        # RUN_STATUS_UNSPECIFIED → treat as INIT
            1: cls.INIT,        # RUN_STATUS_INIT
            2: cls.HANDSHAKE,   # RUN_STATUS_HSHK
            3: cls.READY,       # RUN_STATUS_RDY
            4: cls.EXECUTING,   # RUN_STATUS_EXEC
            5: cls.PAUSED,      # RUN_STATUS_PAUSE
            6: cls.FAULTED,     # RUN_STATUS_FAULT
            7: cls.TERMINATED,  # RUN_STATUS_TERM
        }
        return _PROTO_TO_ENUM.get(proto_value, cls.INIT)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    status: RunStatus
    digest: str
    created_at: datetime
    updated_at: datetime
    last_heartbeat: Optional[datetime]
    gpu_slot: Optional[int]
    failure_reason: Optional[str]
    gpu_slots: list[int] = field(default_factory=list)
    finished_at: Optional[datetime] = None
    outcome: Optional[str] = None


@dataclass(slots=True)
class WALCheckpointStats:
    """Statistics returned from a WAL checkpoint operation."""

    busy: int
    log_frames: int
    checkpointed_frames: int

    def __str__(self) -> str:
        return f"busy={self.busy}, log_frames={self.log_frames}, checkpointed={self.checkpointed_frames}"


class RunRegistry:
    """Lightweight persistence layer for trainer runs."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        ensure_var_directories()
        self._db_path = db_path or str(VAR_TRAINER_DB)
        self._lock = RLock()
        self._initialize()

        # RunBus subscription for RUN_COMPLETED events
        self._bus: Optional[RunBus] = None
        self._subscriber_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start_run_bus_subscriber(self) -> None:
        """Start the RunBus subscriber thread for RUN_COMPLETED events."""
        if self._subscriber_thread is not None:
            _LOGGER.warning("RunBus subscriber already started")
            return

        self._stop_event.clear()
        self._subscriber_thread = threading.Thread(
            target=self._run_bus_subscriber,
            name="registry-run-bus-subscriber",
            daemon=True,
        )
        self._subscriber_thread.start()
        _LOGGER.info("RunRegistry RunBus subscriber started")

    def stop_run_bus_subscriber(self) -> None:
        """Stop the RunBus subscriber thread."""
        if self._subscriber_thread is None:
            return

        self._stop_event.set()
        self._subscriber_thread.join(timeout=5.0)
        if self._subscriber_thread.is_alive():
            _LOGGER.warning("RunBus subscriber thread did not stop cleanly")
        else:
            _LOGGER.info("RunRegistry RunBus subscriber stopped")
        self._subscriber_thread = None

    def _run_bus_subscriber(self) -> None:
        """Main loop for the RunBus subscriber thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._subscribe_to_run_completed())
        except Exception as e:
            _LOGGER.exception("Fatal error in RunBus subscriber", extra={"error": str(e)})
        finally:
            if self._loop is not None:
                self._loop.close()
            _LOGGER.info("RunBus subscriber loop exited")

    async def _subscribe_to_run_completed(self) -> None:
        """Subscribe to RUN_COMPLETED events and update registry."""
        try:
            self._bus = get_bus()
            queue = self._bus.subscribe(Topic.RUN_COMPLETED, "registry")

            _LOGGER.debug("Registry subscribed to RUN_COMPLETED topic")

            while not self._stop_event.is_set():
                try:
                    # Check for run completed events (non-blocking)
                    try:
                        evt = queue.get_nowait()
                        if isinstance(evt, TelemetryEvent):
                            self._handle_run_completed_event(evt)
                    except Exception as e:
                        # Catch queue.Empty (from thread-safe queue.Queue)
                        if type(e).__name__ != 'Empty':
                            raise

                    # Small sleep to avoid busy-waiting
                    await asyncio.sleep(0.1)
                except Exception as e:
                    _LOGGER.exception("Error in subscriber loop", extra={"error": str(e)})
        except Exception as e:
            _LOGGER.exception("Fatal error in subscribe_to_run_completed", extra={"error": str(e)})

    def _handle_run_completed_event(self, evt: TelemetryEvent) -> None:
        """Handle a RUN_COMPLETED event and update registry."""
        run_id = evt.run_id
        payload = evt.payload
        outcome = payload.get("outcome", "unknown")
        failure_reason = payload.get("failure_reason")

        try:
            # Persist outcome while driving FSM to TERMINATED.
            self.update_run_outcome(
                run_id=run_id,
                status=RunStatus.TERMINATED,
                outcome=outcome,
                failure_reason=failure_reason,
            )

            _LOGGER.info(
                "Updated run outcome from RUN_COMPLETED event",
                extra={
                    "run_id": run_id,
                    "outcome": outcome,
                    "status": RunStatus.TERMINATED.value,
                },
            )
        except Exception as e:
            _LOGGER.exception(
                "Error handling RUN_COMPLETED event",
                extra={"run_id": run_id, "error": str(e)},
            )

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # NOTE: WAL + synchronous=NORMAL favors throughput over crash-proof durability.
        # Hard power loss can roll back the last transaction; callers should plan
        # periodic exports for critical data.
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    digest TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_heartbeat TEXT,
                    gpu_slot INTEGER,
                    failure_reason TEXT,
                    gpu_slots_json TEXT NOT NULL DEFAULT '[]',
                    finished_at TEXT,
                    outcome TEXT
                );

                CREATE TABLE IF NOT EXISTS gpu_slots (
                    slot_id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    locked_at TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                        ON DELETE SET NULL ON UPDATE CASCADE
                );

                CREATE TRIGGER IF NOT EXISTS runs_updated_at
                AFTER UPDATE ON runs
                BEGIN
                    UPDATE runs SET updated_at = datetime('now') WHERE rowid = NEW.rowid;
                END;
                """
            )
            self._migrate_gpu_slots_table(conn)
            self._ensure_gpu_slots_json_column(conn)
            self._migrate_status_values(conn)
            # Seed GPU slots up to 8 to match schema limits.
            existing = conn.execute("SELECT COUNT(*) FROM gpu_slots").fetchone()[0]
            if existing == 0:
                conn.executemany(
                    "INSERT INTO gpu_slots(slot_id, run_id, locked_at) VALUES(?, NULL, NULL)",
                    [(slot,) for slot in range(8)],
                )

            # Log database initialization with schema details
            self._log_database_schema(conn)

    def _migrate_gpu_slots_table(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='gpu_slots'"
        ).fetchone()
        if not row:
            return
        schema_sql: str = row["sql"] if isinstance(row, sqlite3.Row) else row[0]
        if "UNIQUE" not in schema_sql.upper():
            return
        # Rebuild gpu_slots without the UNIQUE constraint.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS gpu_slots__new (
                slot_id INTEGER PRIMARY KEY,
                run_id TEXT,
                locked_at TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
                    ON DELETE SET NULL ON UPDATE CASCADE
            );
            INSERT INTO gpu_slots__new(slot_id, run_id, locked_at)
            SELECT slot_id, run_id, locked_at FROM gpu_slots;
            DROP TABLE gpu_slots;
            ALTER TABLE gpu_slots__new RENAME TO gpu_slots;
            """
        )

    def _ensure_gpu_slots_json_column(self, conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(runs)").fetchall()
        has_column = any(column[1] == "gpu_slots_json" for column in columns)
        if not has_column:
            conn.execute("ALTER TABLE runs ADD COLUMN gpu_slots_json TEXT NOT NULL DEFAULT '[]'")

    def _migrate_status_values(self, conn: sqlite3.Connection) -> None:
        """Normalize legacy run status values to the FSM vocabulary."""

        legacy_to_modern = {
            "pending": RunStatus.INIT.value,
            "dispatching": RunStatus.HANDSHAKE.value,
            "running": RunStatus.EXECUTING.value,
            "completed": RunStatus.TERMINATED.value,
            "failed": RunStatus.TERMINATED.value,
            "cancelled": RunStatus.TERMINATED.value,
        }

        try:
            # Build placeholders from count of keys - safe, not user input
            placeholders = ",".join("?" for _ in legacy_to_modern)
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
            # Safe: placeholders built from hardcoded dict keys count, all values parameterized
            query = f"SELECT run_id, status FROM runs WHERE status IN ({placeholders})"
            rows = conn.execute(
                query,
                tuple(legacy_to_modern.keys()),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table may not exist yet on first bootstrap.
            return

        updated = 0
        for row in rows:
            run_id = row["run_id"] if isinstance(row, sqlite3.Row) else row[0]
            current_raw = row["status"] if isinstance(row, sqlite3.Row) else row[1]
            current = str(current_raw or "").lower()
            new_value = legacy_to_modern.get(current)
            if new_value and new_value != current:
                conn.execute(
                    "UPDATE runs SET status = ? WHERE run_id = ?",
                    (new_value, run_id),
                )
                updated += 1

        if updated:
            _LOGGER.info(
                "Migrated run status values to FSM vocabulary",
                extra={"updated": updated},
            )

    def _log_database_schema(self, conn: sqlite3.Connection) -> None:
        """Log database initialization with complete schema details."""
        try:
            # Get all tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            _LOGGER.info(
                f"Database initialized successfully with {len(tables)} tables: {', '.join(tables)}",
                extra={
                    "db_path": self._db_path,
                    "tables": tables,
                    "table_count": len(tables),
                }
            )

            # Log schema for each table
            for table_name in tables:
                # Validate table name against alphanumeric + underscore to prevent SQL injection
                # (even though table_name comes from sqlite_master, satisfy static analysis)
                if not table_name.replace("_", "").isalnum():
                    _LOGGER.warning(
                        "Skipping table with non-alphanumeric name during schema logging",
                        extra={"table_name": table_name}
                    )
                    continue
                
                # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
                # Safe: table_name validated as alphanumeric above, comes from sqlite_master
                # SQLite PRAGMA doesn't support parameterized queries
                cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                column_info = [
                    {
                        "name": col[1],
                        "type": col[2],
                        "notnull": bool(col[3]),
                        "default": col[4],
                        "pk": bool(col[5]),
                    }
                    for col in columns
                ]

                # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
                # Safe: table_name validated as alphanumeric above, comes from sqlite_master
                # Table identifiers cannot be parameterized in SQL
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

                # Format column schema for logging
                column_schema = ", ".join(
                    f"{col['name']}({col['type']}, pk={col['pk']}, notnull={col['notnull']})"
                    for col in column_info
                )

                _LOGGER.info(
                    f"Table '{table_name}': {len(column_info)} columns, {row_count} rows | Schema: {column_schema}",
                    extra={
                        "table": table_name,
                        "columns": len(column_info),
                        "rows": row_count,
                        "schema": column_info,
                    }
                )

            # Log triggers
            cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
            triggers = cursor.fetchall()
            if triggers:
                trigger_names = [t[0] for t in triggers]
                _LOGGER.info(
                    f"Database triggers: {', '.join(trigger_names)}",
                    extra={
                        "trigger_count": len(triggers),
                        "triggers": trigger_names,
                    }
                )

            _LOGGER.info(
                "✓ Database ready to accept training runs",
                extra={
                    "status": "initialized",
                    "waiting_for": "training submissions",
                }
            )
        except Exception as e:
            _LOGGER.error(
                "Failed to log database schema",
                exc_info=e,
                extra={"db_path": self._db_path}
            )

    # ------------------------------------------------------------------
    def register_run(self, run_id: str, config_json: str, digest: str) -> Optional[str]:
        """Register a new run or return existing run_id if digest already exists.

        Returns:
            The run_id (either the new one or the existing one matching the digest).
        """
        now = datetime.now(timezone.utc).isoformat()
        attempt = 0
        while True:
            attempt += 1
            with self._lock:
                try:
                    with self._connect() as conn:
                        existing = conn.execute(
                            "SELECT run_id FROM runs WHERE digest = ?", (digest,)
                        ).fetchone()
                        if existing:
                            _LOGGER.info(
                                "Run already registered for digest",
                                extra={"run_id": existing[0], "digest": digest},
                            )
                            return existing[0]

                        conn.execute(
                            """
                            INSERT INTO runs(run_id, status, config_json, digest, created_at, updated_at, gpu_slots_json)
                            VALUES(?, ?, ?, ?, ?, ?, ?)
                            """,
                            (run_id, RunStatus.INIT.value, config_json, digest, now, now, "[]"),
                        )
                        _LOGGER.info(
                            "Registered new training run",
                            extra={"run_id": run_id, "digest": digest},
                        )
                        return run_id
                except sqlite3.OperationalError as exc:
                    if "no such table" in str(exc).lower() and attempt == 1:
                        _LOGGER.warning(
                            "Trainer registry schema missing when registering run; rebuilding",
                            extra={"error": str(exc), "db_path": self._db_path},
                        )
                        self._initialize()
                        continue
                    raise

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        failure_reason: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, failure_reason = ?, updated_at = datetime('now') WHERE run_id = ?",
                (status.value, failure_reason, run_id),
            )
        _LOGGER.info(
            "Run status updated",
            extra={"run_id": run_id, "status": status.value, "reason": failure_reason},
        )

    def update_run_outcome(
        self,
        run_id: str,
        status: RunStatus,
        outcome: str,
        *,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Update run status, outcome, and finished_at timestamp.

        Args:
            run_id: The run identifier.
            status: The new FSM state (typically :class:`RunStatus.TERMINATED`).
            outcome: The outcome string (success, failure, cancelled).
            failure_reason: Optional reason for failure.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, outcome = ?, finished_at = ?, failure_reason = ?, updated_at = datetime('now') WHERE run_id = ?",
                (status.value, outcome, now, failure_reason, run_id),
            )
        _LOGGER.info(
            "Run outcome updated",
            extra={
                "run_id": run_id,
                "status": status.value,
                "outcome": outcome,
                "failure_reason": failure_reason,
            },
        )

    def record_heartbeat(self, run_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET last_heartbeat = datetime('now') WHERE run_id = ?",
                (run_id,),
            )
        _LOGGER.debug("Heartbeat recorded", extra={"run_id": run_id})

    def update_gpu_slots(self, run_id: str, slots: Iterable[int]) -> None:
        slots_list = list(slots)
        payload = json.dumps(slots_list, separators=(",", ":"))
        primary_slot = slots_list[0] if slots_list else None
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET gpu_slot = ?, gpu_slots_json = ? WHERE run_id = ?",
                (primary_slot, payload, run_id),
            )

    def update_gpu_slot(self, run_id: str, slot: Optional[int]) -> None:
        if slot is None:
            self.update_gpu_slots(run_id, [])
        else:
            self.update_gpu_slots(run_id, [slot])

    def get_run_config_json(self, run_id: str) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT config_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        config_json = row[0]
        return str(config_json) if config_json is not None else None

    def load_runs(self, statuses: Optional[Iterable[RunStatus]] = None) -> list[RunRecord]:
        base_query = "SELECT run_id, status, digest, created_at, updated_at, last_heartbeat, gpu_slot, failure_reason, gpu_slots_json, finished_at, outcome FROM runs"
        params: tuple[object, ...] = ()
        status_list = list(statuses) if statuses else []
        
        if status_list:
            # Build placeholders safely - count-based, not user input
            placeholders = ",".join("?" for _ in status_list)
            # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
            # Safe: placeholders built from count, not user input. All values parameterized via tuple.
            # status.value comes from RunStatus enum (trusted), not external input
            query = base_query + f" WHERE status IN ({placeholders})"
            params = tuple(status.value for status in status_list)
            _LOGGER.debug(
                "Loading runs with status filter",
                extra={"statuses": [s.value for s in status_list], "query": query[:100]}
            )
        else:
            query = base_query
            _LOGGER.debug("Loading all runs (no status filter)")
        
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        
        _LOGGER.debug(
            "Loaded runs from database",
            extra={"count": len(rows), "requested_statuses": [s.value for s in status_list] if status_list else None}
        )
        
        records: list[RunRecord] = []
        for row in rows:
            gpu_slots: list[int] = []
            raw_slots = row["gpu_slots_json"]
            if raw_slots:
                try:
                    loaded = json.loads(raw_slots)
                except json.JSONDecodeError:
                    loaded = []
                if isinstance(loaded, list):
                    gpu_slots = [int(slot) for slot in loaded]
            status_raw = str(row["status"] or "")
            try:
                status_value = RunStatus(status_raw)
            except ValueError:
                _LOGGER.warning(
                    "Unknown run status encountered during load; defaulting to INIT",
                    extra={"run_id": row["run_id"], "status": status_raw},
                )
                status_value = RunStatus.INIT

            records.append(
                RunRecord(
                    run_id=row["run_id"],
                    status=status_value,
                    digest=row["digest"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]) if row["last_heartbeat"] else None,
                    gpu_slot=row["gpu_slot"],
                    failure_reason=row["failure_reason"],
                    gpu_slots=gpu_slots,
                    finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
                    outcome=row["outcome"],
                )
            )
        return records

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, status, digest, created_at, updated_at, last_heartbeat, gpu_slot, failure_reason, gpu_slots_json, finished_at, outcome FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        gpu_slots: list[int] = []
        raw_slots = row["gpu_slots_json"]
        if raw_slots:
            try:
                loaded = json.loads(raw_slots)
            except json.JSONDecodeError:
                loaded = []
            if isinstance(loaded, list):
                gpu_slots = [int(slot) for slot in loaded]
        status_raw = str(row["status"] or "")
        try:
            status_value = RunStatus(status_raw)
        except ValueError:
            _LOGGER.warning(
                "Unknown run status encountered during get_run; defaulting to INIT",
                extra={"run_id": row["run_id"], "status": status_raw},
            )
            status_value = RunStatus.INIT

        return RunRecord(
            run_id=row["run_id"],
            status=status_value,
            digest=row["digest"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]) if row["last_heartbeat"] else None,
            gpu_slot=row["gpu_slot"],
            failure_reason=row["failure_reason"],
            gpu_slots=gpu_slots,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            outcome=row["outcome"],
        )

    # ------------------------------------------------------------------
    def claim_gpu_slot(self, run_id: str, requested: int, mandatory: bool) -> list[int]:
        if requested <= 0:
            return []
        with self._lock, self._connect() as conn:
            # Use explicit transaction to prevent allocation races
            conn.execute("BEGIN IMMEDIATE")
            try:
                free_slots = conn.execute(
                    "SELECT slot_id FROM gpu_slots WHERE run_id IS NULL ORDER BY slot_id ASC LIMIT ?",
                    (requested,),
                ).fetchall()
                if len(free_slots) < requested:
                    if mandatory:
                        conn.rollback()
                        raise GPUReservationError("Insufficient GPU slots available")
                    requested = len(free_slots)
                slots = [row[0] for row in free_slots[:requested]]
                for slot_id in slots:
                    conn.execute(
                        "UPDATE gpu_slots SET run_id = ?, locked_at = datetime('now') WHERE slot_id = ?",
                        (run_id, slot_id),
                    )
                conn.execute(
                    "UPDATE runs SET gpu_slot = ?, gpu_slots_json = ? WHERE run_id = ?",
                    (
                        slots[0] if slots else None,
                        json.dumps(slots, separators=(",", ":")),
                        run_id,
                    ),
                )
                conn.commit()
                return slots
            except Exception:
                conn.rollback()
                raise

    def release_gpu_slots(self, run_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE gpu_slots SET run_id = NULL, locked_at = NULL WHERE run_id = ?",
                (run_id,),
            )
            conn.execute(
                "UPDATE runs SET gpu_slot = NULL, gpu_slots_json = '[]' WHERE run_id = ?",
                (run_id,),
            )

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> WALCheckpointStats:
        """Runs a WAL checkpoint and returns statistics.

        Args:
            mode: Checkpoint mode (PASSIVE, FULL, RESTART, TRUNCATE).

        Returns:
            A dataclass containing the checkpoint statistics.
        """
        # Validate mode against allowed values to prevent SQL injection
        # PASSIVE: Checkpoint without blocking writers
        # FULL: Wait for writers to finish before checkpoint
        # RESTART: Full checkpoint + reset WAL for new snapshot
        # TRUNCATE: Restart + truncate the -wal file
        # Use a whitelist mapping to avoid SQL injection concerns
        mode_normalized = mode.upper()
        allowed_modes = {
            "PASSIVE": "PRAGMA wal_checkpoint(PASSIVE)",
            "FULL": "PRAGMA wal_checkpoint(FULL)",
            "RESTART": "PRAGMA wal_checkpoint(RESTART)",
            "TRUNCATE": "PRAGMA wal_checkpoint(TRUNCATE)",
        }
        
        if mode_normalized not in allowed_modes:
            raise ValueError(f"Invalid checkpoint mode: {mode}. Must be one of {allowed_modes.keys()}")
        
        # Use pre-constructed SQL from whitelist dictionary to satisfy static analysis
        sql_statement = allowed_modes[mode_normalized]
        
        with self._lock, self._connect() as conn:
            # The PRAGMA returns a single row with three integers:
            # (busy, log_size, checkpointed)
            row = conn.execute(sql_statement).fetchone()

        if row is None:
            # This case should be unlikely, but handle it defensively.
            return WALCheckpointStats(busy=-1, log_frames=-1, checkpointed_frames=-1)

        stats = WALCheckpointStats(
            busy=row[0],
            log_frames=row[1],
            checkpointed_frames=row[2],
        )
        
        _LOGGER.info("WAL Checkpoint executed: %s", stats)
        return stats


class GPUReservationError(RuntimeError):
    pass
