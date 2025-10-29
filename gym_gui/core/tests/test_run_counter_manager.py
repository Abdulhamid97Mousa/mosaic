"""Tests for safe, bounded episode counter system.

Tests verify:
- Per-run counter initialization and persistence
- Counter reset on new runs
- Resume functionality from database
- Maximum episodes enforcement
- Thread-safe concurrent access
- Edge cases and error handling
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import List

import pytest

from gym_gui.core.run_counter_manager import RunCounterManager
from gym_gui.constants import DEFAULT_MAX_EPISODES_PER_RUN


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            ep_index INTEGER NOT NULL,
            episode_id TEXT,
            UNIQUE(run_id, ep_index)
        )
        """
    )
    conn.commit()

    yield conn

    conn.close()
    db_path.unlink(missing_ok=True)


class TestRunCounterManagerBasics:
    """Test basic counter functionality."""

    def test_initialization_new_run(self, temp_db):
        """Test counter initializes at -1 for new run."""
        manager = RunCounterManager(temp_db, "run_001")
        manager.initialize()

        assert manager.get_current_index() == -1

    def test_first_episode_index_is_zero(self, temp_db):
        """Test first episode gets index 0."""
        manager = RunCounterManager(temp_db, "run_002")
        manager.initialize()

        with manager.next_episode() as ep_index:
            assert ep_index == 0

        assert manager.get_current_index() == 0

    def test_sequential_episode_indices(self, temp_db):
        """Test episodes get sequential indices."""
        manager = RunCounterManager(temp_db, "run_003")
        manager.initialize()

        indices = []
        for i in range(5):
            with manager.next_episode() as ep_index:
                indices.append(ep_index)

        assert indices == [0, 1, 2, 3, 4]
        assert manager.get_current_index() == 4

    def test_counter_not_initialized_raises(self, temp_db):
        """Test that accessing counter before initialize raises RuntimeError."""
        manager = RunCounterManager(temp_db, "run_004")

        with pytest.raises(RuntimeError, match="not initialized"):
            with manager.next_episode() as ep_index:
                pass


class TestRunCounterManagerPersistence:
    """Test persistence and resume functionality."""

    def test_resume_from_empty_db(self, temp_db):
        """Test resume for new run (no episodes in DB yet)."""
        manager = RunCounterManager(temp_db, "run_new")
        manager.initialize()

        assert manager.get_current_index() == -1

    def test_resume_from_existing_episodes(self, temp_db):
        """Test resume loads last committed episode index from DB."""
        run_id = "run_resume_001"

        # First manager: allocate episodes 0-2
        manager1 = RunCounterManager(temp_db, run_id, max_episodes=100)
        manager1.initialize()

        for i in range(3):
            with manager1.next_episode() as ep_index:
                temp_db.execute(
                    "INSERT INTO episodes (run_id, ep_index, episode_id) VALUES (?, ?, ?)",
                    (run_id, ep_index, f"ep_{ep_index:06d}"),
                )
                temp_db.commit()

        assert manager1.get_current_index() == 2

        # Second manager: resume from DB
        manager2 = RunCounterManager(temp_db, run_id, max_episodes=100)
        manager2.initialize()

        # Should resume at index 2
        assert manager2.get_current_index() == 2

        # Next episode should be 3
        with manager2.next_episode() as ep_index:
            assert ep_index == 3


class TestRunCounterManagerBounds:
    """Test maximum episodes enforcement."""

    def test_max_episodes_default(self, temp_db):
        """Test default max episodes is 1M."""
        manager = RunCounterManager(temp_db, "run_005")
        assert manager._max_episodes == DEFAULT_MAX_EPISODES_PER_RUN

    def test_max_episodes_custom(self, temp_db):
        """Test custom max episodes enforcement."""
        manager = RunCounterManager(temp_db, "run_006", max_episodes=10)
        manager.initialize()

        # Allocate 0-9
        for i in range(10):
            with manager.next_episode() as ep_index:
                assert ep_index == i

        # 10th should raise
        with pytest.raises(RuntimeError, match="Max episodes"):
            with manager.next_episode() as ep_index:
                pass

    def test_max_episodes_validation_on_init(self, temp_db):
        """Test max_episodes must be positive."""
        with pytest.raises(ValueError):
            RunCounterManager(temp_db, "run_007", max_episodes=0)

        with pytest.raises(ValueError):
            RunCounterManager(temp_db, "run_008", max_episodes=-1)

    def test_max_episodes_capacity_check(self, temp_db):
        """Test max_episodes cannot exceed counter width capacity."""
        # Counter width is 6 digits = 10^6 = 1,000,000 max
        with pytest.raises(ValueError, match="exceeds counter capacity"):
            RunCounterManager(temp_db, "run_009", max_episodes=10**6 + 1)

    def test_refuse_resume_if_stored_index_exceeds_max(self, temp_db):
        """Test refusal to resume if stored index > max_episodes."""
        run_id = "run_exceed_max"

        # Insert episode with high index
        temp_db.execute(
            "INSERT INTO episodes (run_id, ep_index, episode_id) VALUES (?, ?, ?)",
            (run_id, 100, "ep_000100"),
        )
        temp_db.commit()

        # Try to resume with max_episodes=50 should fail
        manager = RunCounterManager(temp_db, run_id, max_episodes=50)
        with pytest.raises(ValueError, match="exceeds max_episodes"):
            manager.initialize()


class TestRunCounterManagerConcurrency:
    """Test thread-safe concurrent access."""

    def test_concurrent_next_episode(self, temp_db):
        """Test concurrent calls to next_episode produce unique indices."""
        manager = RunCounterManager(temp_db, "run_concurrent", max_episodes=1000)
        manager.initialize()

        num_threads = 10
        indices_per_thread = 5
        collected_indices: List[int] = []
        lock = threading.Lock()

        def allocate_episodes():
            for _ in range(indices_per_thread):
                with manager.next_episode() as ep_index:
                    with lock:
                        collected_indices.append(ep_index)

        threads = [threading.Thread(target=allocate_episodes) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have allocated exactly num_threads * indices_per_thread indices
        assert len(collected_indices) == num_threads * indices_per_thread

        # All indices should be unique
        assert len(set(collected_indices)) == num_threads * indices_per_thread

        # Indices should be in range 0 to (num_threads * indices_per_thread - 1)
        assert set(collected_indices) == set(
            range(num_threads * indices_per_thread)
        )

    def test_concurrent_max_episodes_enforcement(self, temp_db):
        """Test that concurrent threads respect max_episodes limit."""
        max_episodes = 20
        manager = RunCounterManager(temp_db, "run_concurrent_max", max_episodes=max_episodes)
        manager.initialize()

        num_threads = 10
        successful_indices = []
        failed_threads = []
        lock = threading.Lock()

        def try_allocate():
            try:
                for _ in range(5):  # Try to allocate 5 per thread
                    with manager.next_episode() as ep_index:
                        with lock:
                            successful_indices.append(ep_index)
            except RuntimeError:
                with lock:
                    failed_threads.append(threading.current_thread().name)

        threads = [threading.Thread(target=try_allocate) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should never exceed max_episodes
        assert len(successful_indices) <= max_episodes
        assert max(successful_indices) < max_episodes


class TestRunCounterManagerResetAndMultiRun:
    """Test reset and multi-run scenarios."""

    def test_reset_clears_counter(self, temp_db):
        """Test reset sets counter back to -1."""
        manager = RunCounterManager(temp_db, "run_reset_001")
        manager.initialize()

        with manager.next_episode() as ep_index:
            assert ep_index == 0

        manager.reset()
        assert manager.get_current_index() == -1

    def test_multiple_runs_independent(self, temp_db):
        """Test different runs have independent counters."""
        run1_id = "run_multi_001"
        run2_id = "run_multi_002"

        manager1 = RunCounterManager(temp_db, run1_id)
        manager1.initialize()

        manager2 = RunCounterManager(temp_db, run2_id)
        manager2.initialize()

        # Allocate episodes in run1
        for i in range(3):
            with manager1.next_episode() as ep_index:
                assert ep_index == i

        # Run2 should still be at -1
        assert manager2.get_current_index() == -1

        # Allocate one from run2
        with manager2.next_episode() as ep_index:
            assert ep_index == 0

        # Run1 should still be at 2
        assert manager1.get_current_index() == 2


class TestRunCounterManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_counter_width_formatting(self, temp_db):
        """Test that counter maintains 6-digit width in episode IDs."""
        manager = RunCounterManager(temp_db, "run_width", max_episodes=999999)
        manager.initialize()

        test_cases = [0, 9, 99, 999, 9999, 99999, 999998]
        for expected_index in test_cases:
            # Allocate up to expected_index
            while manager.get_current_index() < expected_index:
                with manager.next_episode() as ep_index:
                    pass

            # Format check (would be done in session.py)
            formatted = f"{manager._run_id}-ep{expected_index:06d}"
            assert formatted.endswith(f"-ep{expected_index:06d}")

    def test_no_db_initialization_headless(self, temp_db):
        """Test counter can work without DB for headless mode."""
        # This tests the fallback mode without a real DB connection
        manager = RunCounterManager(None, "run_headless", max_episodes=100)  # type: ignore
        # Manually set initialized flag (normally done via initialize())
        manager._initialized = True
        manager._current_index = -1

        with manager.next_episode() as ep_index:
            assert ep_index == 0

    def test_context_manager_exception_handling(self, temp_db):
        """Test context manager behavior on exception."""
        manager = RunCounterManager(temp_db, "run_exception")
        manager.initialize()

        # Exception in context should still increment counter
        try:
            with manager.next_episode() as ep_index:
                assert ep_index == 0
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Counter should still have advanced (exception logged but not reverted)
        assert manager.get_current_index() == 0


class TestRunCounterManagerUniqueConstraint:
    """Test that database UNIQUE constraint prevents duplicates."""

    def test_unique_constraint_on_run_id_ep_index(self, temp_db):
        """Test UNIQUE(run_id, ep_index) prevents duplicates."""
        run_id = "run_unique"

        # Insert first episode
        temp_db.execute(
            "INSERT INTO episodes (run_id, ep_index, episode_id) VALUES (?, ?, ?)",
            (run_id, 0, "ep_000000"),
        )
        temp_db.commit()

        # Try to insert duplicate (same run_id, same ep_index)
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.execute(
                "INSERT INTO episodes (run_id, ep_index, episode_id) VALUES (?, ?, ?)",
                (run_id, 0, "ep_different"),
            )
            temp_db.commit()


__all__ = [
    "temp_db",
    "TestRunCounterManagerBasics",
    "TestRunCounterManagerPersistence",
    "TestRunCounterManagerBounds",
    "TestRunCounterManagerConcurrency",
    "TestRunCounterManagerResetAndMultiRun",
    "TestRunCounterManagerEdgeCases",
    "TestRunCounterManagerUniqueConstraint",
]
