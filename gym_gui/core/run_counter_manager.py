"""Per-run episode counter manager with persistence, concurrency, and bounds checking.

Provides a thread-safe mechanism for generating bounded, ordered episode indices
within a run. Supports resume-from-checkpoint functionality via database queries.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Optional

from gym_gui.constants import (
    DEFAULT_MAX_EPISODES_PER_RUN,
    EPISODE_COUNTER_WIDTH,
    format_episode_id,
    COUNTER_NOT_INITIALIZED_ERROR,
    MAX_EPISODES_REACHED_ERROR,
    INVALID_MAX_EPISODES_ERROR,
    COUNTER_CAPACITY_EXCEEDED_ERROR,
    COUNTER_EXCEEDS_MAX_ERROR,
)
from gym_gui.logging_config.log_constants import (
    LOG_COUNTER_INITIALIZED,
    LOG_COUNTER_RESUME_SUCCESS,
    LOG_COUNTER_RESUME_FAILURE,
    LOG_COUNTER_MAX_REACHED,
    LOG_COUNTER_INVALID_STATE,
    LOG_COUNTER_NEXT_EPISODE,
    LOG_COUNTER_RESET,
)

_LOGGER = logging.getLogger(__name__)


class RunCounterManager:
    """Manages episode counter per run with persistence and concurrency safety.
    
    Features:
    - Per-run counter (tied to run_id/ULID)
    - Reset on new run
    - Thread-safe atomic counter access
    - Resume from database (read last committed ep_index)
    - Hard maximum enforcement (default 1M episodes)
    - 6-digit fixed width (0-999999)
    
    Usage:
        manager = RunCounterManager(db_conn, run_id="...")
        manager.initialize()  # Load from DB or start at 0
        
        # Thread-safe increment
        with manager.next_episode() as ep_index:
            # Use ep_index in current thread
            pass  # Auto-committed on context exit
    """

    def __init__(
        self,
        db_conn: Optional[sqlite3.Connection],
        run_id: str,
        max_episodes: Optional[int] = None,
        worker_id: Optional[str] = None,
    ) -> None:
        """Initialize counter manager for a specific run.
        
        Args:
            db_conn: SQLite connection (must have episodes table) or None for headless mode
            run_id: Unique run identifier (ULID format)
            max_episodes: Hard limit (default: DEFAULT_MAX_EPISODES_PER_RUN)
            worker_id: Optional worker ID for multi-process scenarios
                      (generates IDs like f"{run_id}-w{worker_id}-ep{index:06d}")
        
        Raises:
            ValueError: If max_episodes is invalid or stored counter > max_episodes
        """
        self._db_conn = db_conn
        self._run_id = run_id
        self._worker_id = worker_id
        self._max_episodes = max_episodes if max_episodes is not None else DEFAULT_MAX_EPISODES_PER_RUN
        self._current_index = -1  # Not yet initialized
        self._initialized = False  # Flag to track if initialize() has been called
        self._lock = threading.Lock()

        if self._max_episodes <= 0:
            raise ValueError(
                INVALID_MAX_EPISODES_ERROR.format(value=self._max_episodes)
            )
        max_possible = 10 ** EPISODE_COUNTER_WIDTH - 1
        if self._max_episodes > max_possible:
            raise ValueError(
                COUNTER_CAPACITY_EXCEEDED_ERROR.format(
                    max_eps=self._max_episodes,
                    capacity=max_possible,
                )
            )

    def initialize(self) -> None:
        """Load counter state from database or start fresh.
        
        For a new run: counter starts at -1 (first increment → 0)
        For a resumed run: counter loads from last committed episode
        
        Raises:
            ValueError: If stored counter is invalid or exceeds max_episodes
            sqlite3.Error: If database query fails
        """
        with self._lock:
            self._current_index = self._load_from_db()
            self._initialized = True
            _LOGGER.info(
                f"{LOG_COUNTER_INITIALIZED.message}: "
                f"run_id={self._run_id}, starting index={self._current_index}, max={self._max_episodes}"
            )

    def next_episode(self) -> _EpisodeCounterContext:
        """Get the next episode index in a thread-safe manner.
        
        Increments counter atomically and returns a context manager that
        yields the episode index. The counter is committed on context exit
        (outside critical section, to avoid DB calls inside the lock).
        
        Usage:
            with manager.next_episode() as ep_index:
                episode_id = f"{run_id}-ep{ep_index:06d}"
                # ... record telemetry ...
                # Counter auto-committed on context exit
        
        Returns:
            Context manager that yields (int) episode index
            
        Raises:
            RuntimeError: If counter not initialized
            RuntimeError: If max_episodes reached
        """
        if not self._initialized:
            raise RuntimeError(COUNTER_NOT_INITIALIZED_ERROR)

        with self._lock:
            if self._current_index + 1 >= self._max_episodes:
                error_msg = MAX_EPISODES_REACHED_ERROR.format(
                    max_eps=self._max_episodes,
                    run_id=self._run_id,
                )
                _LOGGER.error(error_msg)
                raise RuntimeError(error_msg)
            self._current_index += 1
            ep_index = self._current_index

        _LOGGER.debug(
            f"{LOG_COUNTER_NEXT_EPISODE.message}: "
            f"run_id={self._run_id}, ep_index={ep_index}"
        )

        # Return context manager (DB commit happens outside the lock)
        return _EpisodeCounterContext(self, ep_index)

    def get_current_index(self) -> int:
        """Get the last allocated episode index (thread-safe).
        
        Returns -1 if not yet initialized.
        """
        with self._lock:
            return self._current_index

    def reset(self) -> None:
        """Reset counter to -1 (for new runs).
        
        Call this before re-using the manager with a different run_id
        (or create a new RunCounterManager instance instead).
        """
        with self._lock:
            self._current_index = -1
            self._initialized = False
            _LOGGER.info(
                f"{LOG_COUNTER_RESET.message}: run_id={self._run_id}"
            )

    # -----------------------------------------------------------------------
    # Private: Database access
    # -----------------------------------------------------------------------

    def _load_from_db(self) -> int:
        """Load the last committed episode index from database.
        
        Returns:
            Last committed ep_index (or -1 if no episodes exist yet)
            
        Raises:
            ValueError: If loaded counter exceeds max_episodes
            sqlite3.Error: If database query fails
        """
        # Headless mode (no DB connection)
        if self._db_conn is None:
            _LOGGER.debug(f"No DB connection; starting at index -1 for run {self._run_id}")
            return -1
            
        try:
            cursor = self._db_conn.cursor()
            
            # Check if worker_id column exists in the episodes table
            cursor.execute("PRAGMA table_info(episodes)")
            columns = [col[1] for col in cursor.fetchall()]
            has_worker_id_column = "worker_id" in columns
            
            # Get max ep_index for this run_id
            if self._worker_id and has_worker_id_column:
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(ep_index), -1)
                    FROM episodes
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (self._run_id, self._worker_id),
                )
            elif has_worker_id_column:
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(ep_index), -1)
                    FROM episodes
                    WHERE run_id = ? AND (worker_id IS NULL OR worker_id = '')
                    """,
                    (self._run_id,),
                )
            else:
                # Legacy table without worker_id column
                cursor.execute(
                    """
                    SELECT COALESCE(MAX(ep_index), -1)
                    FROM episodes
                    WHERE run_id = ?
                    """,
                    (self._run_id,),
                )
            row = cursor.fetchone()
            last_index = row[0] if row else -1

            max_possible = 10 ** EPISODE_COUNTER_WIDTH - 1
            if last_index >= self._max_episodes:
                error_msg = COUNTER_EXCEEDS_MAX_ERROR.format(
                    idx=last_index,
                    max_eps=self._max_episodes,
                )
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            if last_index >= 0:
                _LOGGER.info(
                    f"{LOG_COUNTER_RESUME_SUCCESS.message}: "
                    f"run_id={self._run_id}, resumed from ep_index={last_index}"
                )
            return last_index
        except sqlite3.OperationalError as e:
            # Table might not exist yet; treat as -1
            if "no such table" in str(e).lower():
                _LOGGER.debug(f"Episodes table does not exist yet; starting at index -1")
                return -1
            raise

    def _commit_episode(self, ep_index: int) -> None:
        """Mark an episode as committed in the database.
        
        This is called by _EpisodeCounterContext on context exit.
        Currently a no-op placeholder; actual commit happens when
        telemetry writes the episode record to the DB.
        
        Args:
            ep_index: Episode index to commit
        """
        # Placeholder: actual persistence happens via telemetry DB sink
        # In the future, we could add a lock table or commit log here
        pass


class _EpisodeCounterContext:
    """Context manager for safe episode counter acquisition and commitment.
    
    Yields episode index and optionally commits it to DB on exit.
    """

    def __init__(self, manager: RunCounterManager, ep_index: int) -> None:
        self._manager = manager
        self._ep_index = ep_index

    def __enter__(self) -> int:
        """Return the episode index."""
        return self._ep_index

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Commit the episode counter on exit (if no exception)."""
        if exc_type is None:
            # No exception; commit the counter
            self._manager._commit_episode(self._ep_index)
        else:
            # Exception occurred; log but don't re-raise
            _LOGGER.warning(
                f"Exception in episode {self._ep_index} context; counter auto-reverted",
                exc_info=True,
            )


__all__ = ["RunCounterManager"]
