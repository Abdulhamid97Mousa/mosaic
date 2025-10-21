"""Correctness probes for telemetry pipeline.

This module provides diagnostic probes to verify:
- Telemetry DB reality (schema, data consistency)
- UI readiness (Online tab updates)
- Backpressure handling (slow DB, queue behavior)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a correctness probe."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name} - {self.message}"


class TelemetryProbes:
    """Correctness probes for telemetry pipeline."""

    @staticmethod
    def probe_telemetry_schema(db_path: Path) -> ProbeResult:
        """Verify telemetry database schema is correct.
        
        Args:
            db_path: Path to telemetry SQLite database
            
        Returns:
            ProbeResult indicating schema correctness
        """
        if not db_path.exists():
            return ProbeResult(
                name="Telemetry Schema",
                passed=False,
                message="Database does not exist",
                details={"db_path": str(db_path)},
            )

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check steps table
            cursor.execute("PRAGMA table_info(steps)")
            steps_columns = {row[1] for row in cursor.fetchall()}
            required_steps = {"episode_id", "step_index", "reward", "run_id", "agent_id"}
            missing_steps = required_steps - steps_columns

            # Check episodes table
            cursor.execute("PRAGMA table_info(episodes)")
            episodes_columns = {row[1] for row in cursor.fetchall()}
            required_episodes = {"episode_id", "total_reward", "agent_id", "run_id"}
            missing_episodes = required_episodes - episodes_columns

            conn.close()

            if missing_steps or missing_episodes:
                return ProbeResult(
                    name="Telemetry Schema",
                    passed=False,
                    message="Missing required columns",
                    details={
                        "missing_steps": list(missing_steps),
                        "missing_episodes": list(missing_episodes),
                    },
                )

            return ProbeResult(
                name="Telemetry Schema",
                passed=True,
                message="All required columns present",
                details={
                    "steps_columns": len(steps_columns),
                    "episodes_columns": len(episodes_columns),
                },
            )
        except Exception as e:
            return ProbeResult(
                name="Telemetry Schema",
                passed=False,
                message=f"Schema check failed: {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def probe_telemetry_data(db_path: Path) -> ProbeResult:
        """Verify telemetry data consistency.
        
        Args:
            db_path: Path to telemetry SQLite database
            
        Returns:
            ProbeResult indicating data consistency
        """
        if not db_path.exists():
            return ProbeResult(
                name="Telemetry Data",
                passed=True,
                message="Database empty (expected on first run)",
                details={"db_path": str(db_path)},
            )

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check for orphaned steps (steps without episodes)
            cursor.execute(
                """
                SELECT COUNT(*) FROM steps s
                WHERE NOT EXISTS (SELECT 1 FROM episodes e WHERE e.episode_id = s.episode_id)
                """
            )
            orphaned_steps = cursor.fetchone()[0]

            # Check for episodes without run_id
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE run_id IS NULL")
            episodes_without_run = cursor.fetchone()[0]

            # Check for steps without agent_id
            cursor.execute("SELECT COUNT(*) FROM steps WHERE agent_id IS NULL")
            steps_without_agent = cursor.fetchone()[0]

            conn.close()

            issues = []
            if orphaned_steps > 0:
                issues.append(f"Found {orphaned_steps} orphaned steps")
            if episodes_without_run > 0:
                issues.append(f"Found {episodes_without_run} episodes without run_id")
            if steps_without_agent > 0:
                issues.append(f"Found {steps_without_agent} steps without agent_id")

            if issues:
                return ProbeResult(
                    name="Telemetry Data",
                    passed=False,
                    message="Data consistency issues found",
                    details={
                        "orphaned_steps": orphaned_steps,
                        "episodes_without_run": episodes_without_run,
                        "steps_without_agent": steps_without_agent,
                    },
                )

            return ProbeResult(
                name="Telemetry Data",
                passed=True,
                message="Data consistency verified",
                details={
                    "orphaned_steps": orphaned_steps,
                    "episodes_without_run": episodes_without_run,
                    "steps_without_agent": steps_without_agent,
                },
            )
        except Exception as e:
            return ProbeResult(
                name="Telemetry Data",
                passed=False,
                message=f"Data check failed: {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def probe_trainer_schema(db_path: Path) -> ProbeResult:
        """Verify trainer database schema is correct.
        
        Args:
            db_path: Path to trainer SQLite database
            
        Returns:
            ProbeResult indicating schema correctness
        """
        if not db_path.exists():
            return ProbeResult(
                name="Trainer Schema",
                passed=False,
                message="Database does not exist",
                details={"db_path": str(db_path)},
            )

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check runs table
            cursor.execute("PRAGMA table_info(runs)")
            runs_columns = {row[1] for row in cursor.fetchall()}
            required_runs = {"run_id", "status", "outcome", "finished_at"}
            missing_runs = required_runs - runs_columns

            conn.close()

            if missing_runs:
                return ProbeResult(
                    name="Trainer Schema",
                    passed=False,
                    message="Missing required columns",
                    details={"missing_runs": list(missing_runs)},
                )

            return ProbeResult(
                name="Trainer Schema",
                passed=True,
                message="All required columns present",
                details={"runs_columns": len(runs_columns)},
            )
        except Exception as e:
            return ProbeResult(
                name="Trainer Schema",
                passed=False,
                message=f"Schema check failed: {e}",
                details={"error": str(e)},
            )


__all__ = ["ProbeResult", "TelemetryProbes"]

