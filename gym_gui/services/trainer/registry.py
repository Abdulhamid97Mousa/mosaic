from __future__ import annotations

"""SQLite-backed registry for trainer runs and resource bookkeeping."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import sqlite3
from threading import RLock
from typing import Iterable, Optional, cast

from gym_gui.config.paths import VAR_TRAINER_DB, ensure_var_directories


_LOGGER = logging.getLogger(__name__)


class RunStatus(str, Enum):
    PENDING = "pending"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
                    gpu_slots_json TEXT NOT NULL DEFAULT '[]'
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
            # Seed GPU slots up to 8 to match schema limits.
            existing = conn.execute("SELECT COUNT(*) FROM gpu_slots").fetchone()[0]
            if existing == 0:
                conn.executemany(
                    "INSERT INTO gpu_slots(slot_id, run_id, locked_at) VALUES(?, NULL, NULL)",
                    [(slot,) for slot in range(8)],
                )

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

    # ------------------------------------------------------------------
    def register_run(self, run_id: str, config_json: str, digest: str) -> Optional[str]:
        """Register a new run or return existing run_id if digest already exists.
        
        Returns:
            The run_id (either the new one or the existing one matching the digest).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            # Check for existing run with same digest
            existing = conn.execute(
                "SELECT run_id FROM runs WHERE digest = ?", (digest,)
            ).fetchone()
            if existing:
                return existing[0]
            
            conn.execute(
                """
                INSERT INTO runs(run_id, status, config_json, digest, created_at, updated_at, gpu_slots_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, RunStatus.PENDING.value, config_json, digest, now, now, "[]"),
            )
            return run_id

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

    def record_heartbeat(self, run_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET last_heartbeat = datetime('now') WHERE run_id = ?",
                (run_id,),
            )

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

    def load_runs(self, statuses: Optional[Iterable[RunStatus]] = None) -> list[RunRecord]:
        query = "SELECT run_id, status, digest, created_at, updated_at, last_heartbeat, gpu_slot, failure_reason, gpu_slots_json FROM runs"
        params: tuple[object, ...] = ()
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params = tuple(status.value for status in statuses)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
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
            records.append(
                RunRecord(
                    run_id=row["run_id"],
                    status=RunStatus(row["status"]),
                    digest=row["digest"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]) if row["last_heartbeat"] else None,
                    gpu_slot=row["gpu_slot"],
                    failure_reason=row["failure_reason"],
                    gpu_slots=gpu_slots,
                )
            )
        return records

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT run_id, status, digest, created_at, updated_at, last_heartbeat, gpu_slot, failure_reason, gpu_slots_json FROM runs WHERE run_id = ?",
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
        return RunRecord(
            run_id=row["run_id"],
            status=RunStatus(row["status"]),
            digest=row["digest"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]) if row["last_heartbeat"] else None,
            gpu_slot=row["gpu_slot"],
            failure_reason=row["failure_reason"],
            gpu_slots=gpu_slots,
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
        with self._lock, self._connect() as conn:
            # The PRAGMA returns a single row with three integers:
            # (busy, log_size, checkpointed)
            row = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()

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