from __future__ import annotations

"""SQLite-backed telemetry storage."""

from datetime import datetime
from pathlib import Path
import logging
import queue
import sqlite3
import sys
import threading
from typing import Any, List, Mapping, Optional, Sequence

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.constants import parse_episode_id
from gym_gui.utils import json_serialization
from gym_gui.telemetry.migrations import MigrationRunner, WALConfiguration
from gym_gui.logging_config.helpers import LogConstantMixin, log_constant
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_SQLITE_DEBUG,
    LOG_SERVICE_SQLITE_INFO,
    LOG_SERVICE_SQLITE_WARNING,
    LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED,
    LOG_SERVICE_SQLITE_WORKER_STARTED,
    LOG_SERVICE_SQLITE_WORKER_STOPPED,
    LOG_SERVICE_SQLITE_WRITE_ERROR,
)

_LOGGER = logging.getLogger("gym_gui.telemetry.sqlite_store")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class TelemetrySQLiteStore(LogConstantMixin):
    """Persist telemetry events to a SQLite database."""

    _DEFAULT_BATCH_SIZE = 32
    _DEFAULT_MAX_BUFFER_BYTES = 2 * 1024 * 1024  # 2 MiB
    _QUEUE_TIMEOUT_SECONDS = 0.1

    def __init__(
        self,
        db_path: Path,
        *,
        batch_size: int | None = None,
        max_buffer_bytes: int | None = None,
    ) -> None:
        self._db_path = db_path.expanduser().resolve()
        _ensure_parent(self._db_path)
        self._batch_size = max(1, batch_size or self._DEFAULT_BATCH_SIZE)
        self._max_buffer_bytes = max(1, max_buffer_bytes or self._DEFAULT_MAX_BUFFER_BYTES)
        self._logger = _LOGGER
        self._deserialization_failures: set[str] = set()

        self._conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        # Negative cache_size value means kibibytes; -131072 ~= 512 MiB cache budget
        self._conn.execute("PRAGMA cache_size=-131072")
        self._conn.isolation_level = None  # Use explicit transactions

        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._pending_payload_bytes = 0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="TelemetrySQLiteStore",
            daemon=True,
        )

        self._initialize()
        self._worker.start()
        self.log_constant(LOG_SERVICE_SQLITE_WORKER_STARTED)

    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                episode_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                action INTEGER,
                observation BLOB,
                reward REAL NOT NULL,
                terminated INTEGER NOT NULL,
                truncated INTEGER NOT NULL,
                info BLOB,
                render_payload BLOB,
                timestamp TEXT NOT NULL,
                agent_id TEXT,
                render_hint BLOB,
                frame_ref TEXT,
                payload_version INTEGER NOT NULL DEFAULT 0,
                run_id TEXT,
                worker_id TEXT,
                space_signature BLOB,
                vector_metadata BLOB,
                time_step INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                total_reward REAL NOT NULL,
                steps INTEGER NOT NULL,
                terminated INTEGER NOT NULL,
                truncated INTEGER NOT NULL,
                metadata BLOB,
                timestamp TEXT NOT NULL,
                agent_id TEXT,
                run_id TEXT,
                worker_id TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS run_status (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'active',
                deleted_at TEXT,
                archived_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_steps_episode
                ON steps(episode_id, step_index)
            """
        )
        self._conn.commit()

        # Apply idempotent migrations (Phase 4)
        MigrationRunner.run_telemetry_migrations(self._db_path)

        # Configure WAL mode for optimal performance
        WALConfiguration.configure_wal(self._conn)

        self._ensure_columns()
        self._ensure_indexes()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )

    def _ensure_columns(self) -> None:
        cursor = self._conn.execute("PRAGMA table_info(steps)")
        column_names = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("render_payload", "ALTER TABLE steps ADD COLUMN render_payload BLOB"),
            ("agent_id", "ALTER TABLE steps ADD COLUMN agent_id TEXT"),
            ("render_hint", "ALTER TABLE steps ADD COLUMN render_hint BLOB"),
            ("frame_ref", "ALTER TABLE steps ADD COLUMN frame_ref TEXT"),
            (
                "payload_version",
                "ALTER TABLE steps ADD COLUMN payload_version INTEGER NOT NULL DEFAULT 0",
            ),
            ("run_id", "ALTER TABLE steps ADD COLUMN run_id TEXT"),
            ("game_id", "ALTER TABLE steps ADD COLUMN game_id TEXT"),
            ("episode_seed", "ALTER TABLE steps ADD COLUMN episode_seed INTEGER"),
            ("worker_id", "ALTER TABLE steps ADD COLUMN worker_id TEXT"),
            ("space_signature", "ALTER TABLE steps ADD COLUMN space_signature BLOB"),
            ("vector_metadata", "ALTER TABLE steps ADD COLUMN vector_metadata BLOB"),
            ("time_step", "ALTER TABLE steps ADD COLUMN time_step INTEGER"),
        ]
        for name, ddl in migrations:
            if name not in column_names:
                self._conn.execute(ddl)
        cursor = self._conn.execute("PRAGMA table_info(episodes)")
        episode_columns = {row[1] for row in cursor.fetchall()}
        if "agent_id" not in episode_columns:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN agent_id TEXT")
        if "run_id" not in episode_columns:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN run_id TEXT")
        if "game_id" not in episode_columns:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN game_id TEXT")
        if "worker_id" not in episode_columns:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN worker_id TEXT")
        self._conn.commit()

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient querying of game_id and other fields."""
        cursor = self._conn.cursor()

        # Get existing indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        existing_indexes = {row[0] for row in cursor.fetchall()}

        # Define indexes to create
        indexes = [
            ("idx_steps_game_id", "CREATE INDEX IF NOT EXISTS idx_steps_game_id ON steps(game_id)"),
            ("idx_steps_run_id_agent_id", "CREATE INDEX IF NOT EXISTS idx_steps_run_id_agent_id ON steps(run_id, agent_id)"),
            ("idx_episodes_game_id", "CREATE INDEX IF NOT EXISTS idx_episodes_game_id ON episodes(game_id)"),
            ("idx_episodes_run_id", "CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id)"),
        ]

        for index_name, ddl in indexes:
            if index_name not in existing_indexes:
                try:
                    cursor.execute(ddl)
                    self.log_constant(
                        LOG_SERVICE_SQLITE_INFO,
                        message="index_created",
                        extra={"index": index_name},
                    )
                except sqlite3.OperationalError as e:
                    self.log_constant(
                        LOG_SERVICE_SQLITE_WARNING,
                        message="index_create_failed",
                        extra={"index": index_name, "error": str(e)},
                    )

        self._conn.commit()

    # ------------------------------------------------------------------
    def record_step(self, record: StepRecord) -> None:
        self._queue.put(("step", record))

    def record_episode(self, rollup: EpisodeRollup, *, wait: bool = False) -> None:
        """Queue a completed episode for persistence.

        Args:
            rollup: Episode summary to persist.
            wait: When True, block until the queued writes flush to disk.
        """

        self._queue.put(("flush", None))
        self._queue.put(("episode", rollup))
        self._queue.put(("flush", None))
        if wait:
            self._queue.join()

    def delete_episode(self, episode_id: str, *, wait: bool = True) -> None:
        self._queue.put(("delete_episode", episode_id))
        if wait:
            self._queue.join()

    def delete_all_episodes(self, *, wait: bool = True) -> None:
        self._queue.put(("delete_all", None))
        if wait:
            self._queue.join()

    def delete_run(self, run_id: str, *, wait: bool = True) -> None:
        """Mark a run as deleted and remove all its telemetry data."""
        self._queue.put(("delete_run", run_id))
        if wait:
            self._queue.join()

    def archive_run(self, run_id: str, *, wait: bool = True) -> None:
        """Mark a run as archived (read-only snapshot for replay)."""
        self._queue.put(("archive_run", run_id))
        if wait:
            self._queue.join()

    def is_run_deleted(self, run_id: str) -> bool:
        """Check if a run has been deleted."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT status FROM run_status WHERE run_id = ? AND status = 'deleted'",
            (run_id,),
        )
        return cursor.fetchone() is not None

    def is_run_archived(self, run_id: str) -> bool:
        """Check if a run has been archived."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT status FROM run_status WHERE run_id = ? AND status = 'archived'",
            (run_id,),
        )
        return cursor.fetchone() is not None

    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        """Return aggregate metrics for a run."""

        cursor = self._conn.cursor()
        legacy_prefix = f"{run_id}-ep%"

        cursor.execute(
            """
            SELECT COUNT(*) AS episode_count,
                   COALESCE(SUM(steps), 0) AS total_steps,
                   COALESCE(SUM(total_reward), 0.0) AS total_reward
            FROM episodes
            WHERE run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)
            """,
            (run_id, legacy_prefix),
        )
        episode_row = cursor.fetchone() or (0, 0, 0.0)
        episodes_collected = int(episode_row[0] or 0)
        steps_collected = int(episode_row[1] or 0)
        total_reward = float(episode_row[2] or 0.0)

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM steps
            WHERE run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)
            """,
            (run_id, legacy_prefix),
        )
        steps_row = cursor.fetchone()
        if steps_row and steps_row[0] and steps_row[0] > steps_collected:
            steps_collected = int(steps_row[0])

        cursor.execute(
            """
            SELECT MAX(timestamp)
            FROM steps
            WHERE run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)
            """,
            (run_id, legacy_prefix),
        )
        last_update = cursor.fetchone()
        last_update_ts = last_update[0] if last_update and last_update[0] else ""

        cursor.execute(
            """
            SELECT agent_id
            FROM steps
            WHERE (run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)) AND agent_id IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (run_id, legacy_prefix),
        )
        agent_row = cursor.fetchone()
        agent_id = agent_row[0] if agent_row and agent_row[0] else ""

        cursor.execute(
            "SELECT status FROM run_status WHERE run_id = ?",
            (run_id,),
        )
        status_row = cursor.fetchone()
        status = status_row[0] if status_row and status_row[0] else "unknown"

        return {
            "run_id": run_id,
            "episodes": episodes_collected,
            "steps": steps_collected,
            "total_reward": total_reward,
            "last_update": last_update_ts,
            "status": status,
            "agent_id": agent_id,
        }

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        pending_steps: List[dict[str, object]] = []
        while not self._stop_event.is_set():
            try:
                cmd, payload = self._queue.get(timeout=self._QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                if pending_steps:
                    self._flush_steps(pending_steps)
                    pending_steps = []
                continue

            try:
                if cmd == "step":
                    assert isinstance(payload, StepRecord)
                    step_payload, size = self._prepare_step_payload(payload)
                    pending_steps.append(step_payload)
                    self._pending_payload_bytes += size
                    if (
                        len(pending_steps) >= self._batch_size
                        or self._pending_payload_bytes >= self._max_buffer_bytes
                    ):
                        self._flush_steps(pending_steps)
                        pending_steps = []
                elif cmd == "episode":
                    assert isinstance(payload, EpisodeRollup)
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._write_episode(payload)
                elif cmd == "flush":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                elif cmd == "delete_episode":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    if isinstance(payload, str):
                        self._delete_episode_rows(payload)
                elif cmd == "delete_all":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._delete_all_rows()
                elif cmd == "delete_run":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    if isinstance(payload, str):
                        self._delete_run_data(payload, mark_deleted=True)
                elif cmd == "archive_run":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    if isinstance(payload, str):
                        self._delete_run_data(payload, mark_archived=True)
                elif cmd == "stop":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    break
            finally:
                self._queue.task_done()

        # Drain any remaining items after stop
        while True:
            try:
                cmd, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                if cmd == "step" and isinstance(payload, StepRecord):
                    step_payload, size = self._prepare_step_payload(payload)
                    pending_steps.append(step_payload)
                    self._pending_payload_bytes += size
                elif cmd == "episode" and isinstance(payload, EpisodeRollup):
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._write_episode(payload)
                elif cmd == "delete_episode" and isinstance(payload, str):
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._delete_episode_rows(payload)
                elif cmd == "delete_all":
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._delete_all_rows()
                elif cmd == "delete_run" and isinstance(payload, str):
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._delete_run_data(payload, mark_deleted=True)
                elif cmd == "archive_run" and isinstance(payload, str):
                    if pending_steps:
                        self._flush_steps(pending_steps)
                        pending_steps = []
                    self._delete_run_data(payload, mark_archived=True)
            finally:
                self._queue.task_done()

        if pending_steps:
            self._flush_steps(pending_steps)

    def _flush_steps(self, steps: List[dict[str, object]]) -> None:
        if not steps:
            return
        
        # Log what we're about to insert (DEBUG for individual steps to avoid spam)
        batch_size = len(steps)
        for index, step in enumerate(steps, start=1):
            self.log_constant(
                LOG_SERVICE_SQLITE_DEBUG,
                message="batch_step_prepared",
                extra={
                    "batch_position": index,
                    "batch_size": batch_size,
                    "episode_id": step.get("episode_id"),
                    "step_index": step.get("step_index"),
                    "reward": step.get("reward"),
                    "terminated": step.get("terminated"),
                    "truncated": step.get("truncated"),
                    "agent_id": step.get("agent_id"),
                },
            )
            self.log_constant(
                LOG_SERVICE_SQLITE_DEBUG,
                message="batch_step_types",
                extra={
                    "reward_type": type(step.get("reward")).__name__,
                    "terminated_type": type(step.get("terminated")).__name__,
                    "truncated_type": type(step.get("truncated")).__name__,
                },
            )
        
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.executemany(
            """
            INSERT INTO steps (
                episode_id, step_index, action, observation, reward,
                terminated, truncated, info, render_payload, timestamp,
                agent_id, render_hint, frame_ref, payload_version, run_id, game_id, worker_id,
                space_signature, vector_metadata, time_step
            ) VALUES (:episode_id, :step_index, :action, :observation, :reward,
                :terminated, :truncated, :info, :render_payload, :timestamp,
                :agent_id, :render_hint, :frame_ref, :payload_version, :run_id, :game_id, :worker_id,
                :space_signature, :vector_metadata, :time_step)
            """,
            steps,
        )
        cursor.execute("COMMIT")
        # Use INFO for batch commit (less noise than per-step ERROR logs)
        self.log_constant(
            LOG_SERVICE_SQLITE_INFO,
            message="batch_commit_success",
            extra={"step_count": batch_size},
        )
        self._pending_payload_bytes = 0

    def _delete_episode_rows(self, episode_id: str) -> None:
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM steps WHERE episode_id = ?", (episode_id,))
        cursor.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
        cursor.execute("COMMIT")

    def _delete_all_rows(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute("DELETE FROM steps")
        cursor.execute("DELETE FROM episodes")
        cursor.execute("COMMIT")

    def _delete_run_data(self, run_id: str, mark_deleted: bool = False, mark_archived: bool = False) -> None:
        """Delete all telemetry data for a run and mark its status."""
        try:
            cursor = self._conn.cursor()
            
            # Explicit transaction with BEGIN/COMMIT (required since isolation_level=None)
            cursor.execute("BEGIN")
            
            # Delete all steps and episodes for this run. Older builds stored the run identifier
            # only in the episode_id column ("{run_id}-epXXXX"), leaving run_id NULL. The
            # fallback LIKE clause ensures those legacy rows are also purged.
            legacy_episode_prefix = f"{run_id}-ep%"

            cursor.execute(
                "DELETE FROM steps WHERE run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)",
                (run_id, legacy_episode_prefix),
            )
            cursor.execute(
                "DELETE FROM episodes WHERE run_id = ? OR (run_id IS NULL AND episode_id LIKE ?)",
                (run_id, legacy_episode_prefix),
            )
            
            # Mark run status
            if mark_deleted:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT OR REPLACE INTO run_status (run_id, status, deleted_at) VALUES (?, 'deleted', ?)",
                    (run_id, now),
                )
                self.log_constant(
                    LOG_SERVICE_SQLITE_INFO,
                    message=f"Run deleted and marked in database: run_id={run_id}",
                    extra={"run_id": run_id}
                )
            elif mark_archived:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT OR REPLACE INTO run_status (run_id, status, archived_at) VALUES (?, 'archived', ?)",
                    (run_id, now),
                )
                self.log_constant(
                    LOG_SERVICE_SQLITE_INFO,
                    message=f"Run archived and marked in database: run_id={run_id}",
                    extra={"run_id": run_id}
                )
            
            # Commit the transaction
            cursor.execute("COMMIT")
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_SQLITE_WRITE_ERROR,
                message=f"_delete_run_data failed: {e}",
                exc_info=e,
                extra={"run_id": run_id}
            )

    def _serialize_field(self, value: Any, *, context: str) -> bytes | None:
        if value is None:
            return None
        try:
            return json_serialization.dumps(value)
        except json_serialization.SerializationError:
            self.log_constant(
                LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED,
                extra={"context": context, "phase": "serialize"},
                exc_info=sys.exc_info()[1],
            )
            return None

    def _deserialize_field(self, payload: Any, *, context: str, default: Any) -> Any:
        if payload is None:
            return self._clone_default(default)
        try:
            return json_serialization.loads(payload)
        except json_serialization.SerializationError:
            if context not in self._deserialization_failures:
                self._deserialization_failures.add(context)
                self.log_constant(
                    LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED,
                    extra={"context": context, "phase": "deserialize"},
                    exc_info=sys.exc_info()[1],
                )
            else:
                self.log_constant(
                    LOG_SERVICE_SQLITE_DEBUG,
                    message="repeated_deserialization_failure",
                    extra={"context": context},
                )
            return self._clone_default(default)

    @staticmethod
    def _clone_default(value: Any) -> Any:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            return list(value)
        if isinstance(value, set):
            return set(value)
        return value

    def _write_episode(self, rollup: EpisodeRollup) -> None:
        ep_index = None
        parsed_id: dict[str, Any] | None = None
        if isinstance(rollup.metadata, Mapping):
            meta_index = rollup.metadata.get("episode_index")
            if meta_index is not None:
                try:
                    ep_index = int(meta_index)
                except (TypeError, ValueError):
                    ep_index = None
        try:
            parsed_id = parse_episode_id(rollup.episode_id)
        except Exception:
            parsed_id = None
        if ep_index is None and parsed_id is not None:
            ep_index_raw = parsed_id.get("ep_index")
            if ep_index_raw is not None:
                try:
                    ep_index = int(ep_index_raw)
                except (TypeError, ValueError):
                    ep_index = None
        payload = {
            "episode_id": rollup.episode_id,
            "total_reward": rollup.total_reward,
            "steps": rollup.steps,
            "terminated": int(rollup.terminated),
            "truncated": int(rollup.truncated),
            "metadata": self._serialize_field(rollup.metadata, context="episode metadata"),
            "timestamp": rollup.timestamp.isoformat(),
            "agent_id": rollup.agent_id,
            "run_id": rollup.run_id,
            "game_id": rollup.game_id,
            "worker_id": rollup.worker_id,
            "ep_index": ep_index,
        }
        if parsed_id is not None:
            if payload.get("run_id") in (None, "") and parsed_id.get("run_id") is not None:
                payload["run_id"] = parsed_id.get("run_id")
            if payload.get("worker_id") in (None, "") and parsed_id.get("worker_id") is not None:
                payload["worker_id"] = parsed_id.get("worker_id")
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute(
            """
            INSERT INTO episodes (
                episode_id, total_reward, steps, terminated, truncated, metadata, timestamp, agent_id, run_id, game_id, worker_id, ep_index
            ) VALUES (:episode_id, :total_reward, :steps, :terminated, :truncated, :metadata, :timestamp, :agent_id, :run_id, :game_id, :worker_id, :ep_index)
            ON CONFLICT(episode_id) DO UPDATE SET
                total_reward=excluded.total_reward,
                steps=excluded.steps,
                terminated=excluded.terminated,
                truncated=excluded.truncated,
                metadata=excluded.metadata,
                timestamp=excluded.timestamp,
                agent_id=excluded.agent_id,
                run_id=excluded.run_id,
                game_id=excluded.game_id,
                worker_id=excluded.worker_id,
                ep_index=excluded.ep_index
            """,
            payload,
        )
        cursor.execute("COMMIT")

    def _step_payload(self, record: StepRecord) -> dict[str, object]:
        # Extract game_id from render_payload if available
        game_id = None
        if isinstance(record.render_payload, dict):
            game_id = record.render_payload.get("game_id")

        return {
            "episode_id": record.episode_id,
            "step_index": record.step_index,
            "action": record.action,
            "observation": self._serialize_field(record.observation, context="observation"),
            "reward": record.reward,
            "terminated": int(record.terminated),
            "truncated": int(record.truncated),
            "info": self._serialize_field(record.info, context="info"),
            "render_payload": self._serialize_field(record.render_payload, context="render payload"),
            "timestamp": record.timestamp.isoformat(),
            "agent_id": record.agent_id,
            "render_hint": self._serialize_field(record.render_hint, context="render hint"),
            "frame_ref": record.frame_ref,
            "payload_version": int(record.payload_version),
            "run_id": record.run_id,
            "game_id": game_id,
            "worker_id": record.worker_id,
            "space_signature": self._serialize_field(
                record.space_signature, context="space signature"
            ),
            "vector_metadata": self._serialize_field(
                record.vector_metadata, context="vector metadata"
            ),
            "time_step": int(record.time_step)
            if record.time_step is not None
            else None,
        }

    def _prepare_step_payload(self, record: StepRecord) -> tuple[dict[str, object], int]:
        payload = self._step_payload(record)
        size = self._payload_size(payload)
        return payload, size

    @staticmethod
    def _payload_size(payload: dict[str, object]) -> int:
        size = 0
        for key in (
            "observation",
            "info",
            "render_payload",
            "render_hint",
            "space_signature",
            "vector_metadata",
        ):
            value = payload.get(key)
            if isinstance(value, (bytes, bytearray, memoryview)):
                size += len(value)
            elif isinstance(value, str):
                size += len(value.encode("utf-8"))
        return size

    def flush(self) -> None:
        self._queue.put(("flush", None))
        self._queue.join()

    def checkpoint_wal(self, mode: str = "TRUNCATE") -> None:
        """Perform a WAL checkpoint to manage WAL file size.

        Args:
            mode: Checkpoint mode (PASSIVE, FULL, RESTART, TRUNCATE).
                  TRUNCATE is recommended for background checkpoints.
        """
        # Use whitelist mapping to avoid SQL injection concerns
        mode_normalized = mode.upper()
        allowed_modes = {
            "PASSIVE": "PRAGMA wal_checkpoint(PASSIVE)",
            "FULL": "PRAGMA wal_checkpoint(FULL)",
            "RESTART": "PRAGMA wal_checkpoint(RESTART)",
            "TRUNCATE": "PRAGMA wal_checkpoint(TRUNCATE)",
        }
        
        if mode_normalized not in allowed_modes:
            self.log_constant(
                LOG_SERVICE_SQLITE_WRITE_ERROR,
                extra={"context": "wal_checkpoint_invalid_mode", "mode": mode},
            )
            return
        
        # Use pre-constructed SQL from whitelist dictionary to satisfy static analysis
        sql_statement = allowed_modes[mode_normalized]
        
        try:
            self._conn.execute(sql_statement)
            self.log_constant(
                LOG_SERVICE_SQLITE_DEBUG,
                message="wal_checkpoint_completed",
                extra={"mode": mode},
            )
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_SQLITE_WRITE_ERROR,
                extra={"context": "wal_checkpoint", "mode": mode},
                exc_info=e,
            )

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._queue.put(("stop", None))
        self._queue.join()
        try:
            self._worker.join(timeout=2.0)
        finally:
            self._conn.close()
        self.log_constant(LOG_SERVICE_SQLITE_WORKER_STOPPED)

    def __enter__(self) -> "TelemetrySQLiteStore":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - defensive shutdown
        try:
            self.close()
        except Exception as exc:  # pragma: no cover - best-effort logging during GC
            logger = getattr(self, "_logger", _LOGGER)
            log_constant(
                logger,
                LOG_SERVICE_SQLITE_WARNING,
                message="close_failed_during_gc",
                exc_info=exc,
            )

    # ------------------------------------------------------------------
    def recent_steps(self, limit: int = 100) -> Sequence[StepRecord]:
        query = (
            "SELECT episode_id, step_index, action, observation, reward, terminated, truncated, info, "
            "render_payload, timestamp, agent_id, render_hint, frame_ref, payload_version, run_id, worker_id, "
            "space_signature, vector_metadata, time_step "
            "FROM steps ORDER BY rowid DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return tuple(self._row_to_step(row) for row in rows)

    def recent_episodes(self, limit: int = 20) -> Sequence[EpisodeRollup]:
        query = (
            "SELECT episode_id, total_reward, steps, terminated, truncated, metadata, timestamp, agent_id, run_id, game_id, worker_id "
            "FROM episodes ORDER BY timestamp DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return tuple(self._row_to_episode(row) for row in rows)

    def episode_steps(self, episode_id: str) -> Sequence[StepRecord]:
        query = (
            "SELECT episode_id, step_index, action, observation, reward, terminated, truncated, info, render_payload, "
            "timestamp, agent_id, render_hint, frame_ref, payload_version, run_id, worker_id, space_signature, vector_metadata, time_step "
            "FROM steps WHERE episode_id = ? ORDER BY step_index ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (episode_id,)).fetchall()
        return tuple(self._row_to_step(row) for row in rows)

    def episodes_for_run(
        self,
        run_id: str,
        *,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
        order_desc: bool = False,
    ) -> Sequence[EpisodeRollup]:
        """Return all episodes associated with a specific training run."""

        self.flush()
        clauses: list[str] = ["run_id = ?"]
        params: list[Any] = [run_id]
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        query = (
            "SELECT episode_id, total_reward, steps, terminated, truncated, metadata, timestamp, agent_id, run_id, game_id, worker_id "
            "FROM episodes"
        )
        query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp {}".format("DESC" if order_desc else "ASC")
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        episodes: list[EpisodeRollup] = []
        for row in rows:
            episode = self._row_to_episode(row)
            episodes.append(episode)
        return tuple(episodes)

    def purge_steps(self, keep_recent: int) -> None:
        self.flush()
        if keep_recent <= 0:
            with self._connect() as conn:
                conn.execute("DELETE FROM steps")
                conn.commit()
            return
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM steps WHERE rowid NOT IN (SELECT rowid FROM steps ORDER BY rowid DESC LIMIT ?)",
                (keep_recent,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def _row_to_step(self, row: Sequence) -> StepRecord:
        observation = self._deserialize_field(row[3], context="observation", default=None)
        info = self._deserialize_field(row[7], context="info", default={})
        render_payload = self._deserialize_field(row[8], context="render payload", default=None)
        timestamp_value = row[9] if len(row) > 9 else None
        agent_id = row[10] if len(row) > 10 else None
        render_hint = None
        if len(row) > 11:
            render_hint = self._deserialize_field(row[11], context="render hint", default=None)
        frame_ref = row[12] if len(row) > 12 else None
        payload_version = int(row[13]) if len(row) > 13 and row[13] is not None else 0
        run_id = row[14] if len(row) > 14 else None
        worker_id = row[15] if len(row) > 15 else None
        space_signature = None
        if len(row) > 16:
            space_signature = self._deserialize_field(
                row[16], context="space signature", default=None
            )
        vector_metadata = None
        if len(row) > 17:
            vector_metadata = self._deserialize_field(
                row[17], context="vector metadata", default=None
            )
        time_step = None
        if len(row) > 18 and row[18] is not None:
            try:
                time_step = int(row[18])
            except (TypeError, ValueError):
                time_step = None
        return StepRecord(
            episode_id=row[0],
            step_index=row[1],
            action=row[2],
            observation=observation,
            reward=row[4],
            terminated=bool(row[5]),
            truncated=bool(row[6]),
            info=info,
            timestamp=datetime.fromisoformat(timestamp_value) if timestamp_value else datetime.utcnow(),
            render_payload=render_payload,
            agent_id=agent_id,
            render_hint=render_hint,
            frame_ref=frame_ref,
            payload_version=payload_version,
            run_id=run_id,
            worker_id=worker_id,
            space_signature=space_signature,
            vector_metadata=vector_metadata,
            time_step=time_step,
        )

    def _row_to_episode(self, row: Sequence) -> EpisodeRollup:
        metadata = self._deserialize_field(row[5], context="episode metadata", default={})
        agent_id = row[7] if len(row) > 7 else None
        run_id = row[8] if len(row) > 8 else None
        game_id = row[9] if len(row) > 9 else None
        worker_id = row[10] if len(row) > 10 else None
        return EpisodeRollup(
            episode_id=row[0],
            total_reward=row[1],
            steps=row[2],
            terminated=bool(row[3]),
            truncated=bool(row[4]),
            metadata=metadata,
            timestamp=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
            agent_id=agent_id,
            run_id=run_id,
            game_id=game_id,
            worker_id=worker_id,
        )


__all__ = ["TelemetrySQLiteStore"]
