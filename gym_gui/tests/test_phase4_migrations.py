"""Tests for Phase 4: DB Migrations + Indexes."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from gym_gui.telemetry.migrations import MigrationRunner, WALConfiguration


class TestMigrations:
    """Test database migrations."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test.db"
            yield db_file

    def _create_base_schema(self, db_path: Path) -> None:
        """Create base schema without Phase 4 migrations."""
        conn = sqlite3.connect(db_path)
        conn.execute(
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
                run_id TEXT
            )
            """
        )
        conn.execute(
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
                run_id TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _get_indexes(self, db_path: Path) -> set:
        """Get all index names from database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()
        return indexes

    def _get_triggers(self, db_path: Path) -> set:
        """Get all trigger names from database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
        triggers = {row[0] for row in cursor.fetchall()}
        conn.close()
        return triggers

    def test_migrations_are_idempotent(self, db_path):
        """Test that migrations can be run multiple times safely."""
        self._create_base_schema(db_path)

        # Run migrations twice
        MigrationRunner.run_telemetry_migrations(db_path)
        indexes_after_first = self._get_indexes(db_path)

        MigrationRunner.run_telemetry_migrations(db_path)
        indexes_after_second = self._get_indexes(db_path)

        # Should have same indexes after second run
        assert indexes_after_first == indexes_after_second

    def test_phase4_indexes_created(self, db_path):
        """Test that Phase 4 indexes are created."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        indexes = self._get_indexes(db_path)

        # Phase 4 indexes should exist
        assert "idx_steps_run_id" in indexes
        assert "idx_episodes_timestamp" in indexes
        assert "idx_episodes_run_agent_ts" in indexes

    def test_phase4_trigger_created(self, db_path):
        """Test that duplicate prevention trigger is created."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        triggers = self._get_triggers(db_path)

        # Trigger should exist
        assert "trigger_steps_unique" in triggers

    def test_duplicate_prevention_trigger_works(self, db_path):
        """Test that duplicate prevention trigger prevents duplicate steps."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        conn = sqlite3.connect(db_path)

        # Insert first step
        conn.execute(
            """
            INSERT INTO steps (
                episode_id, step_index, action, reward, terminated, truncated, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ep1", 0, 1, 1.0, 0, 0, "2025-01-01T00:00:00"),
        )
        conn.commit()

        # Try to insert duplicate step (should be ignored)
        conn.execute(
            """
            INSERT INTO steps (
                episode_id, step_index, action, reward, terminated, truncated, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ep1", 0, 2, 2.0, 0, 0, "2025-01-01T00:00:01"),
        )
        conn.commit()

        # Check that only one step exists
        cursor = conn.execute(
            "SELECT COUNT(*) FROM steps WHERE episode_id = ? AND step_index = ?",
            ("ep1", 0),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_wal_configuration(self, db_path):
        """Test that WAL configuration is applied."""
        self._create_base_schema(db_path)
        conn = sqlite3.connect(db_path)

        WALConfiguration.configure_wal(conn)

        # Check WAL mode is enabled
        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        assert journal_mode.upper() == "WAL"

        # Check synchronous mode
        cursor = conn.execute("PRAGMA synchronous")
        synchronous = cursor.fetchone()[0]
        assert synchronous == 1  # NORMAL

        conn.close()

    def test_composite_index_on_episodes(self, db_path):
        """Test that composite index on episodes is created."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA index_info(idx_episodes_run_agent_ts)")
        columns = [row[2] for row in cursor.fetchall()]
        conn.close()

        # Should have run_id, agent_id, timestamp in that order
        assert columns == ["run_id", "agent_id", "timestamp"]

    def test_index_on_steps_run_id(self, db_path):
        """Test that index on steps(run_id) is created."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA index_info(idx_steps_run_id)")
        columns = [row[2] for row in cursor.fetchall()]
        conn.close()

        # Should have run_id
        assert columns == ["run_id"]

    def test_index_on_episodes_timestamp(self, db_path):
        """Test that index on episodes(timestamp) is created."""
        self._create_base_schema(db_path)
        MigrationRunner.run_telemetry_migrations(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA index_info(idx_episodes_timestamp)")
        columns = [row[2] for row in cursor.fetchall()]
        conn.close()

        # Should have timestamp
        assert columns == ["timestamp"]

