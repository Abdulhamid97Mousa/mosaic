from __future__ import annotations

"""SQLite-backed telemetry storage."""

from datetime import datetime
from pathlib import Path
import pickle
import sqlite3
from typing import Sequence

from gym_gui.core.data_model import EpisodeRollup, StepRecord


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class TelemetrySQLiteStore:
    """Persist telemetry events to a SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.expanduser().resolve()
        _ensure_parent(self._db_path)
        self._initialize()

    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
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
            conn.commit()
        self._ensure_columns()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

    def _ensure_columns(self) -> None:
        with self._connect() as conn:
            cursor = conn.execute("PRAGMA table_info(steps)")
            column_names = {row[1] for row in cursor.fetchall()}
            if "render_payload" not in column_names:
                conn.execute("ALTER TABLE steps ADD COLUMN render_payload BLOB")
                conn.commit()

    # ------------------------------------------------------------------
    def record_step(self, record: StepRecord) -> None:
        payload = {
            "episode_id": record.episode_id,
            "step_index": record.step_index,
            "action": record.action,
            "observation": pickle.dumps(record.observation),
            "reward": record.reward,
            "terminated": int(record.terminated),
            "truncated": int(record.truncated),
            "info": pickle.dumps(dict(record.info)),
            "render_payload": pickle.dumps(record.render_payload)
            if record.render_payload is not None
            else None,
            "timestamp": record.timestamp.isoformat(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO steps (
                    episode_id, step_index, action, observation, reward,
                    terminated, truncated, info, render_payload, timestamp
                ) VALUES (:episode_id, :step_index, :action, :observation, :reward,
                    :terminated, :truncated, :info, :render_payload, :timestamp)
                """,
                payload,
            )
            conn.commit()

    def record_episode(self, rollup: EpisodeRollup) -> None:
        payload = {
            "episode_id": rollup.episode_id,
            "total_reward": rollup.total_reward,
            "steps": rollup.steps,
            "terminated": int(rollup.terminated),
            "truncated": int(rollup.truncated),
            "metadata": pickle.dumps(dict(rollup.metadata)),
            "timestamp": rollup.timestamp.isoformat(),
        }
        with self._connect() as conn:
            conn.execute(
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
            conn.commit()

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
    @staticmethod
    def _row_to_step(row: Sequence) -> StepRecord:
        observation = pickle.loads(row[3]) if row[3] is not None else None
        info = pickle.loads(row[7]) if row[7] is not None else {}
        render_payload = None
        render_index = 8 if len(row) > 8 else None
        timestamp_index = 9 if len(row) > 9 else (8 if len(row) > 8 else None)
        if render_index is not None and len(row) > render_index:
            payload_blob = row[render_index]
            if payload_blob is not None:
                try:
                    render_payload = pickle.loads(payload_blob)
                except Exception:
                    render_payload = None
        timestamp_value = row[timestamp_index] if timestamp_index is not None and len(row) > timestamp_index else None
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

    @staticmethod
    def _row_to_episode(row: Sequence) -> EpisodeRollup:
        metadata = pickle.loads(row[5]) if row[5] is not None else {}
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
