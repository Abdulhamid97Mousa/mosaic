from __future__ import annotations

"""SQLite-backed telemetry storage."""

from datetime import datetime
from pathlib import Path
import logging
import queue
import sqlite3
import threading
from typing import Any, List, Sequence

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.utils import json_serialization


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class TelemetrySQLiteStore:
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
        self._logger = logging.getLogger("gym_gui.telemetry.sqlite_store")
        self._deserialization_failures: set[str] = set()

        self._conn = sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
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
                timestamp TEXT NOT NULL
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
                timestamp TEXT NOT NULL
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
        self._ensure_columns()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self._db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )

    def _ensure_columns(self) -> None:
        cursor = self._conn.execute("PRAGMA table_info(steps)")
        column_names = {row[1] for row in cursor.fetchall()}
        if "render_payload" not in column_names:
            self._conn.execute("ALTER TABLE steps ADD COLUMN render_payload BLOB")
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
            finally:
                self._queue.task_done()

        if pending_steps:
            self._flush_steps(pending_steps)

    def _flush_steps(self, steps: List[dict[str, object]]) -> None:
        if not steps:
            return
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.executemany(
            """
            INSERT INTO steps (
                episode_id, step_index, action, observation, reward,
                terminated, truncated, info, render_payload, timestamp
            ) VALUES (:episode_id, :step_index, :action, :observation, :reward,
                :terminated, :truncated, :info, :render_payload, :timestamp)
            """,
            steps,
        )
        cursor.execute("COMMIT")
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

    def _serialize_field(self, value: Any, *, context: str) -> bytes | None:
        if value is None:
            return None
        try:
            return json_serialization.dumps(value)
        except json_serialization.SerializationError:
            self._logger.warning("Failed to serialize %s; dropping value", context, exc_info=True)
            return None

    def _deserialize_field(self, payload: Any, *, context: str, default: Any) -> Any:
        if payload is None:
            return self._clone_default(default)
        try:
            return json_serialization.loads(payload)
        except json_serialization.SerializationError:
            if context not in self._deserialization_failures:
                self._deserialization_failures.add(context)
                self._logger.warning(
                    "Failed to deserialize %s; using default fallback", context, exc_info=True
                )
            else:
                self._logger.debug(
                    "Repeated deserialization failure for %s; default applied", context,
                    exc_info=False,
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
        payload = {
            "episode_id": rollup.episode_id,
            "total_reward": rollup.total_reward,
            "steps": rollup.steps,
            "terminated": int(rollup.terminated),
            "truncated": int(rollup.truncated),
            "metadata": self._serialize_field(rollup.metadata, context="episode metadata"),
            "timestamp": rollup.timestamp.isoformat(),
        }
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        cursor.execute(
            """
            INSERT INTO episodes (
                episode_id, total_reward, steps, terminated, truncated, metadata, timestamp
            ) VALUES (:episode_id, :total_reward, :steps, :terminated, :truncated, :metadata, :timestamp)
            ON CONFLICT(episode_id) DO UPDATE SET
                total_reward=excluded.total_reward,
                steps=excluded.steps,
                terminated=excluded.terminated,
                truncated=excluded.truncated,
                metadata=excluded.metadata,
                timestamp=excluded.timestamp
            """,
            payload,
        )
        cursor.execute("COMMIT")

    def _step_payload(self, record: StepRecord) -> dict[str, object]:
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
        }

    def _prepare_step_payload(self, record: StepRecord) -> tuple[dict[str, object], int]:
        payload = self._step_payload(record)
        size = self._payload_size(payload)
        return payload, size

    @staticmethod
    def _payload_size(payload: dict[str, object]) -> int:
        size = 0
        for key in ("observation", "info", "render_payload"):
            value = payload.get(key)
            if isinstance(value, (bytes, bytearray, memoryview)):
                size += len(value)
            elif isinstance(value, str):
                size += len(value.encode("utf-8"))
        return size

    def flush(self) -> None:
        self._queue.put(("flush", None))
        self._queue.join()

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

    def __enter__(self) -> "TelemetrySQLiteStore":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - defensive shutdown
        try:
            self.close()
        except Exception:  # pragma: no cover - best-effort logging during GC
            logger = getattr(self, "_logger", logging.getLogger("gym_gui.telemetry.sqlite_store"))
            logger.warning("TelemetrySQLiteStore close during __del__ failed", exc_info=True)

    # ------------------------------------------------------------------
    def recent_steps(self, limit: int = 100) -> Sequence[StepRecord]:
        query = (
            "SELECT episode_id, step_index, action, observation, reward, terminated, truncated, info, "
            "render_payload, timestamp FROM steps ORDER BY rowid DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return tuple(self._row_to_step(row) for row in rows)

    def recent_episodes(self, limit: int = 20) -> Sequence[EpisodeRollup]:
        query = "SELECT episode_id, total_reward, steps, terminated, truncated, metadata, timestamp FROM episodes ORDER BY timestamp DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
        return tuple(self._row_to_episode(row) for row in rows)

    def episode_steps(self, episode_id: str) -> Sequence[StepRecord]:
        query = (
            "SELECT episode_id, step_index, action, observation, reward, terminated, truncated, info, render_payload, timestamp "
            "FROM steps WHERE episode_id = ? ORDER BY step_index ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (episode_id,)).fetchall()
        return tuple(self._row_to_step(row) for row in rows)

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
        render_payload = None
        if len(row) > 8:
            render_payload = self._deserialize_field(
                row[8], context="render payload", default=None
            )
        timestamp_value = row[9] if len(row) > 9 else None
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
        )

    def _row_to_episode(self, row: Sequence) -> EpisodeRollup:
        metadata = self._deserialize_field(row[5], context="episode metadata", default={})
        return EpisodeRollup(
            episode_id=row[0],
            total_reward=row[1],
            steps=row[2],
            terminated=bool(row[3]),
            truncated=bool(row[4]),
            metadata=metadata,
            timestamp=datetime.fromisoformat(row[6]) if row[6] else datetime.utcnow(),
        )


__all__ = ["TelemetrySQLiteStore"]
