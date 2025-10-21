"""Database migrations for telemetry and trainer databases.

This module provides idempotent migrations that can be run multiple times
safely. Migrations are applied at startup to ensure schema consistency.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Callable, List, Tuple

_LOGGER = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""

    def __init__(self, name: str, sql: str) -> None:
        """Initialize a migration.
        
        Args:
            name: Unique identifier for the migration
            sql: SQL statement to execute (should be idempotent)
        """
        self.name = name
        self.sql = sql

    def apply(self, conn: sqlite3.Connection) -> bool:
        """Apply the migration to a database connection.
        
        Args:
            conn: SQLite connection
            
        Returns:
            True if migration was applied, False if already applied
        """
        try:
            conn.execute(self.sql)
            conn.commit()
            _LOGGER.debug(f"Applied migration: {self.name}")
            return True
        except sqlite3.OperationalError as e:
            if "already exists" in str(e) or "duplicate column" in str(e):
                _LOGGER.debug(f"Migration already applied: {self.name}")
                return False
            raise


class MigrationRunner:
    """Runs database migrations at startup."""

    # Telemetry DB migrations
    TELEMETRY_MIGRATIONS = [
        # Phase 1-3: Column additions and basic indexes
        Migration(
            "telemetry_add_run_id_to_steps",
            "ALTER TABLE steps ADD COLUMN run_id TEXT",
        ),
        Migration(
            "telemetry_add_run_id_to_episodes",
            "ALTER TABLE episodes ADD COLUMN run_id TEXT",
        ),
        Migration(
            "telemetry_add_agent_id_to_episodes",
            "ALTER TABLE episodes ADD COLUMN agent_id TEXT",
        ),
        Migration(
            "telemetry_create_index_steps_episode",
            "CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode_id, step_index)",
        ),
        Migration(
            "telemetry_create_index_episodes_run",
            "CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id, agent_id, episode_id)",
        ),
        Migration(
            "telemetry_create_index_steps_run",
            "CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, episode_id)",
        ),
        # Phase 4: Additional indexes for performance
        Migration(
            "telemetry_create_index_steps_run_id",
            "CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id)",
        ),
        Migration(
            "telemetry_create_index_episodes_timestamp",
            "CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp)",
        ),
        Migration(
            "telemetry_create_index_episodes_run_agent_timestamp",
            "CREATE INDEX IF NOT EXISTS idx_episodes_run_agent_ts ON episodes(run_id, agent_id, timestamp)",
        ),
        # Phase 4: Duplicate prevention trigger
        Migration(
            "telemetry_trigger_steps_unique",
            """
            CREATE TRIGGER IF NOT EXISTS trigger_steps_unique
            BEFORE INSERT ON steps
            BEGIN
                SELECT RAISE(IGNORE)
                WHERE EXISTS (
                    SELECT 1 FROM steps
                    WHERE episode_id = NEW.episode_id
                    AND step_index = NEW.step_index
                );
            END
            """,
        ),
    ]

    # Trainer DB migrations
    TRAINER_MIGRATIONS = [
        Migration(
            "trainer_add_finished_at",
            "ALTER TABLE runs ADD COLUMN finished_at TEXT",
        ),
        Migration(
            "trainer_add_outcome",
            "ALTER TABLE runs ADD COLUMN outcome TEXT",
        ),
        Migration(
            "trainer_add_failure_reason",
            "ALTER TABLE runs ADD COLUMN failure_reason TEXT",
        ),
        Migration(
            "trainer_create_index_runs_status",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, created_at DESC)",
        ),
        Migration(
            "trainer_create_index_runs_outcome",
            "CREATE INDEX IF NOT EXISTS idx_runs_outcome ON runs(outcome, finished_at DESC)",
        ),
    ]

    @staticmethod
    def run_telemetry_migrations(db_path: Path) -> None:
        """Run all telemetry database migrations.
        
        Args:
            db_path: Path to telemetry SQLite database
        """
        MigrationRunner._run_migrations(db_path, MigrationRunner.TELEMETRY_MIGRATIONS)

    @staticmethod
    def run_trainer_migrations(db_path: Path) -> None:
        """Run all trainer database migrations.
        
        Args:
            db_path: Path to trainer SQLite database
        """
        MigrationRunner._run_migrations(db_path, MigrationRunner.TRAINER_MIGRATIONS)

    @staticmethod
    def _run_migrations(db_path: Path, migrations: List[Migration]) -> None:
        """Run a list of migrations on a database.
        
        Args:
            db_path: Path to SQLite database
            migrations: List of Migration objects to apply
        """
        if not db_path.exists():
            _LOGGER.debug(f"Database does not exist yet: {db_path}")
            return

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            
            for migration in migrations:
                try:
                    migration.apply(conn)
                except Exception as e:
                    _LOGGER.warning(
                        f"Failed to apply migration {migration.name}: {e}",
                        extra={"migration": migration.name, "error": str(e)},
                    )
            
            conn.close()
            _LOGGER.info(
                f"Completed migrations for {db_path}",
                extra={"count": len(migrations)},
            )
        except Exception as e:
            _LOGGER.exception(
                f"Failed to run migrations on {db_path}",
                extra={"error": str(e)},
            )


class WALConfiguration:
    """Configure WAL mode for optimal performance."""

    @staticmethod
    def configure_wal(conn: sqlite3.Connection) -> None:
        """Configure WAL mode for a database connection.
        
        Args:
            conn: SQLite connection
        """
        try:
            # Enable WAL mode
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Synchronous mode: NORMAL is faster than FULL but still safe
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # WAL autocheckpoint: checkpoint after 1000 pages
            conn.execute("PRAGMA wal_autocheckpoint=1000")
            
            # Cache size: use more memory for better performance
            conn.execute("PRAGMA cache_size=-64000")  # 64MB
            
            # Temp store: use memory for temporary tables
            conn.execute("PRAGMA temp_store=MEMORY")
            
            _LOGGER.debug("WAL configuration applied")
        except Exception as e:
            _LOGGER.warning(f"Failed to configure WAL: {e}")


__all__ = ["Migration", "MigrationRunner", "WALConfiguration"]

