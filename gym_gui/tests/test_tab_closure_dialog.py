"""Test TabClosureDialog integration with render_tabs flow."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pytest
from PyQt6 import QtWidgets, QtCore

from gym_gui.ui.indicators.tab_closure_dialog import TabClosureDialog, TabClosureChoice
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore
from gym_gui.services.telemetry import TelemetryService
from gym_gui.ui.widgets.render_tabs import RenderTabs
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab


@pytest.fixture
def qapp():
    """Get or create QApplication."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class TestTabClosureDialogIntegration:
    """Integration tests for tab closure dialog with render_tabs flow."""

    def test_dialog_sets_delete_choice_when_user_selects_delete(self, qapp):
        """
        Simulate the exact flow from render_tabs._close_dynamic_tab:
        1. Show dialog
        2. User selects DELETE
        3. User clicks Continue
        4. Dialog._selected_choice should be TabClosureChoice.DELETE
        """
        dialog = TabClosureDialog()
        
        # Simulate user selecting DELETE
        dialog._delete_radio.setChecked(True)
        assert dialog._delete_radio.isChecked(), "DELETE radio should be checked"
        
        # Simulate user clicking Continue
        # _on_continue sets _selected_choice and calls accept()
        dialog._on_continue()
        
        # Check that _selected_choice was set correctly
        assert dialog._selected_choice == TabClosureChoice.DELETE, \
            f"Expected _selected_choice=DELETE, got {dialog._selected_choice.value}"

    def test_dialog_sets_archive_choice_when_user_selects_archive(self, qapp):
        """Test that selecting ARCHIVE and clicking Continue sets ARCHIVE."""
        dialog = TabClosureDialog()
        
        # Simulate user selecting ARCHIVE
        dialog._archive_radio.setChecked(True)
        assert dialog._archive_radio.isChecked(), "ARCHIVE radio should be checked"
        
        # Simulate clicking Continue
        dialog._on_continue()
        
        # Verify the choice
        assert dialog._selected_choice == TabClosureChoice.ARCHIVE, \
            f"Expected _selected_choice=ARCHIVE, got {dialog._selected_choice.value}"

    def test_dialog_sets_keep_choice_by_default(self, qapp):
        """Test that KEEP is the default choice if not changed."""
        dialog = TabClosureDialog()
        
        # Default is KEEP (radio button checked in __init__)
        assert dialog._keep_radio.isChecked(), "KEEP radio should be checked by default"
        
        # Simulate clicking Continue without changing selection
        dialog._on_continue()
        
        # Verify the choice
        assert dialog._selected_choice == TabClosureChoice.KEEP_AND_CLOSE, \
            f"Expected _selected_choice=KEEP_AND_CLOSE, got {dialog._selected_choice.value}"

    def test_dialog_sets_cancel_choice_when_user_clicks_cancel(self, qapp):
        """Test that clicking Cancel sets CANCEL."""
        dialog = TabClosureDialog()
        
        # Simulate user changing selection
        dialog._delete_radio.setChecked(True)
        
        # But then clicking Cancel
        dialog._on_cancel()
        
        # Verify the choice is CANCEL
        assert dialog._selected_choice == TabClosureChoice.CANCEL, \
            f"Expected _selected_choice=CANCEL, got {dialog._selected_choice.value}"

    def test_get_selected_choice_matches_radio_button_state(self, qapp):
        """Test that get_selected_choice() accurately reflects radio button state."""
        dialog = TabClosureDialog()
        
        # Test each radio button
        test_cases = [
            (dialog._keep_radio, TabClosureChoice.KEEP_AND_CLOSE),
            (dialog._archive_radio, TabClosureChoice.ARCHIVE),
            (dialog._delete_radio, TabClosureChoice.DELETE),
        ]
        
        for radio_button, expected_choice in test_cases:
            radio_button.setChecked(True)
            actual_choice = dialog.get_selected_choice()
            assert actual_choice == expected_choice, \
                f"Expected {expected_choice.value}, got {actual_choice.value}"

    def test_action_selected_signal_emitted_on_continue(self, qapp):
        """Test that action_selected signal is emitted with correct choice."""
        dialog = TabClosureDialog()
        
        # Setup signal spy
        signal_spy = []
        dialog.action_selected.connect(lambda choice: signal_spy.append(choice))
        
        # Select DELETE and click Continue
        dialog._delete_radio.setChecked(True)
        dialog._on_continue()
        
        # Verify signal was emitted with DELETE
        assert len(signal_spy) == 1, "Signal should be emitted once"
        assert signal_spy[0] == TabClosureChoice.DELETE, \
            f"Signal should emit DELETE, got {signal_spy[0].value}"

    def test_action_selected_signal_not_emitted_on_cancel(self, qapp):
        """Test that action_selected signal is NOT emitted when clicking Cancel."""
        dialog = TabClosureDialog()
        
        # Setup signal spy
        signal_spy = []
        dialog.action_selected.connect(lambda choice: signal_spy.append(choice))
        
        # Click Cancel without selecting anything
        dialog._on_cancel()
        
        # Verify signal was NOT emitted
        assert len(signal_spy) == 0, "Signal should not be emitted on cancel"

    def test_button_group_ensures_mutual_exclusivity(self, qapp):
        """Test that only one radio button can be selected at a time (via QButtonGroup)."""
        dialog = TabClosureDialog()
        
        # Test that selecting one unchecks the others
        dialog._delete_radio.setChecked(True)
        assert dialog._delete_radio.isChecked()
        assert not dialog._keep_radio.isChecked()
        assert not dialog._archive_radio.isChecked()
        
        # Select archive
        dialog._archive_radio.setChecked(True)
        assert dialog._archive_radio.isChecked()
        assert not dialog._delete_radio.isChecked()
        assert not dialog._keep_radio.isChecked()
        
        # Select keep
        dialog._keep_radio.setChecked(True)
        assert dialog._keep_radio.isChecked()
        assert not dialog._archive_radio.isChecked()
        assert not dialog._delete_radio.isChecked()


class TestTabClosureWithPersistence:
    """Test tab closure dialog with SQLite persistence layer."""

    def test_delete_run_persists_to_database(self, qapp):
        """Test that deleting a run via the dialog persists to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            
            # Create a test run
            test_run_id = "run_delete_test_001"
            conn = store._conn
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            conn.commit()
            
            # Simulate user selecting DELETE and clicking Continue
            dialog = TabClosureDialog()
            dialog._delete_radio.setChecked(True)
            dialog._on_continue()
            
            # Verify choice is DELETE
            assert dialog._selected_choice == TabClosureChoice.DELETE
            
            # Now execute the choice (as render_tabs does)
            store.delete_run(test_run_id, wait=True)
            
            # Verify the run is marked as deleted
            assert store.is_run_deleted(test_run_id), "Run should be marked as deleted"
            
            # Verify the run is NOT marked as archived
            assert not store.is_run_archived(test_run_id), "Run should NOT be marked as archived"

    def test_archive_run_persists_to_database(self, qapp):
        """Test that archiving a run via the dialog persists to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            
            # Create a test run
            test_run_id = "run_archive_test_001"
            conn = store._conn
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            conn.commit()
            
            # Simulate user selecting ARCHIVE and clicking Continue
            dialog = TabClosureDialog()
            dialog._archive_radio.setChecked(True)
            dialog._on_continue()
            
            # Verify choice is ARCHIVE
            assert dialog._selected_choice == TabClosureChoice.ARCHIVE
            
            # Now execute the choice (as render_tabs does)
            store.archive_run(test_run_id, wait=True)
            
            # Verify the run is marked as archived
            assert store.is_run_archived(test_run_id), "Run should be marked as archived"
            
            # Verify the run is NOT marked as deleted
            assert not store.is_run_deleted(test_run_id), "Run should NOT be marked as deleted"

    def test_keep_choice_does_not_modify_database(self, qapp):
        """Test that keeping a run doesn't modify its database status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            
            # Create a test run
            test_run_id = "run_keep_test_001"
            conn = store._conn
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            conn.commit()
            
            # Simulate user selecting KEEP (default) and clicking Continue
            dialog = TabClosureDialog()
            assert dialog._keep_radio.isChecked(), "KEEP should be default"
            dialog._on_continue()
            
            # Verify choice is KEEP_AND_CLOSE
            assert dialog._selected_choice == TabClosureChoice.KEEP_AND_CLOSE
            
            # When KEEP is selected, we DON'T call delete_run or archive_run
            # So the run should still be active
            assert not store.is_run_deleted(test_run_id), "Run should NOT be marked as deleted"
            assert not store.is_run_archived(test_run_id), "Run should NOT be marked as archived"

    def test_telemetry_service_delete_integrates_with_store(self, qapp):
        """Test that TelemetryService.delete_run() properly uses the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            service = TelemetryService()
            service.attach_store(store)
            
            # Create a test run
            test_run_id = "service_delete_test_001"
            conn = store._conn
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            conn.commit()
            
            # Call delete_run via service
            service.delete_run(test_run_id)
            
            # Verify it's marked as deleted
            assert service.is_run_deleted(test_run_id), "Service should report run as deleted"
            assert store.is_run_deleted(test_run_id), "Store should have marked run as deleted"

    def test_telemetry_service_archive_integrates_with_store(self, qapp):
        """Test that TelemetryService.archive_run() properly uses the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            service = TelemetryService()
            service.attach_store(store)
            
            # Create a test run
            test_run_id = "service_archive_test_001"
            conn = store._conn
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            conn.commit()
            
            # Call archive_run via service
            service.archive_run(test_run_id)
            
            # Verify it's marked as archived
            assert service.is_run_archived(test_run_id), "Service should report run as archived"
            assert store.is_run_archived(test_run_id), "Store should have marked run as archived"

    def test_delete_run_removes_all_data_from_database(self, qapp):
        """Test that delete_run actually removes steps and episodes from the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            
            # Create a test run with some dummy data
            test_run_id = "test_run_with_data"
            conn = store._conn
            cursor = conn.cursor()
            
            # Insert run status
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (test_run_id,)
            )
            
            # Insert dummy step
            cursor.execute(
                """INSERT INTO steps (episode_id, step_index, action, reward, terminated, 
                   truncated, timestamp, agent_id, run_id, payload_version) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("ep_001", 0, 1, 1.0, 0, 0, "2025-01-01T00:00:00", "agent_1", test_run_id, 0)
            )
            
            # Insert dummy episode
            cursor.execute(
                """INSERT INTO episodes (episode_id, total_reward, steps, terminated, truncated, 
                   timestamp, agent_id, run_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("ep_001", 1.0, 1, 0, 0, "2025-01-01T00:00:00", "agent_1", test_run_id)
            )
            conn.commit()
            
            # Verify data exists
            cursor.execute("SELECT COUNT(*) FROM steps WHERE run_id = ?", (test_run_id,))
            steps_before = cursor.fetchone()[0]
            assert steps_before == 1, "Should have 1 step before delete"
            
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ?", (test_run_id,))
            episodes_before = cursor.fetchone()[0]
            assert episodes_before == 1, "Should have 1 episode before delete"
            
            # Delete the run
            store.delete_run(test_run_id, wait=True)
            
            # Verify data is actually deleted
            cursor.execute("SELECT COUNT(*) FROM steps WHERE run_id = ?", (test_run_id,))
            steps_after = cursor.fetchone()[0]
            assert steps_after == 0, f"Steps should be deleted, but found {steps_after}"
            
            # Verify data is still marked as active in run_status but deleted in data tables
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE run_id = ?", (test_run_id,))
            episodes_after = cursor.fetchone()[0]
            assert episodes_after == 0, f"Episodes should be deleted, but found {episodes_after}"
            
            # Verify run is marked as deleted
            assert store.is_run_deleted(test_run_id), "Run should be marked as deleted"

    def test_delete_run_cleans_legacy_rows_without_run_id(self, qapp):
        """Delete should remove legacy rows where run_id was stored only in episode_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)

            run_id = "legacy-run-001"
            cursor = store._conn.cursor()

            # Insert legacy rows: run_id column NULL but episode_id encodes the run identifier
            cursor.execute(
                "INSERT INTO run_status (run_id, status) VALUES (?, 'active')",
                (run_id,),
            )
            cursor.execute(
                """INSERT INTO steps (
                        episode_id, step_index, action, reward, terminated,
                        truncated, timestamp, agent_id, run_id, payload_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"{run_id}-ep0001", 0, 0, 0.0, 0, 0, "2025-01-01T00:00:00", "agent-1", None, 0),
            )
            cursor.execute(
                """INSERT INTO episodes (
                        episode_id, total_reward, steps, terminated, truncated,
                        timestamp, agent_id, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"{run_id}-ep0001", 0.0, 1, 0, 0, "2025-01-01T00:00:00", "agent-1", None),
            )
            store._conn.commit()

            # Sanity check that legacy rows exist with NULL run_id
            cursor.execute("SELECT COUNT(*) FROM steps WHERE run_id IS NULL")
            assert cursor.fetchone()[0] == 1

            store.delete_run(run_id, wait=True)

            cursor.execute("SELECT COUNT(*) FROM steps WHERE episode_id LIKE ?", (f"{run_id}-ep%",))
            assert cursor.fetchone()[0] == 0, "Legacy step rows should be removed"

            cursor.execute("SELECT COUNT(*) FROM episodes WHERE episode_id LIKE ?", (f"{run_id}-ep%",))
            assert cursor.fetchone()[0] == 0, "Legacy episode rows should be removed"

            assert store.is_run_deleted(run_id), "Run should be marked deleted even for legacy rows"


class TestRenderTabsClosureFlow:
    """Verify that RenderTabs integrates the dialog and telemetry persistence."""

    def test_close_tab_delete_marks_run_deleted(self, qapp, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            service = TelemetryService()
            service.attach_store(store)

            render_tabs = RenderTabs(telemetry_service=service)

            run_id = "close_run_delete"
            agent_id = "agent-123"
            tab_name = f"Live-Agent-{agent_id}"
            live_tab = LiveTelemetryTab(run_id, agent_id, parent=render_tabs)
            render_tabs.add_dynamic_tab(run_id, tab_name, live_tab)

            def _mock_prompt(self, run_id_param, widget):
                assert run_id_param == run_id
                return TabClosureChoice.DELETE, False

            monkeypatch.setattr(RenderTabs, "_prompt_tab_closure", _mock_prompt)

            tab_index = render_tabs.indexOf(live_tab)
            render_tabs._close_dynamic_tab(run_id, tab_name, tab_index)

            assert store.is_run_deleted(run_id)
            assert run_id not in render_tabs._agent_tabs
            render_tabs.deleteLater()
            store.close()

    def test_close_tab_cancel_keeps_tab_open(self, qapp, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            service = TelemetryService()
            service.attach_store(store)

            render_tabs = RenderTabs(telemetry_service=service)

            run_id = "close_run_cancel"
            agent_id = "agent-456"
            tab_name = f"Live-Agent-{agent_id}"
            live_tab = LiveTelemetryTab(run_id, agent_id, parent=render_tabs)
            render_tabs.add_dynamic_tab(run_id, tab_name, live_tab)

            prompt_calls = []

            def _mock_prompt(self, run_id_param, widget):
                prompt_calls.append(run_id_param)
                return TabClosureChoice.CANCEL, False

            monkeypatch.setattr(RenderTabs, "_prompt_tab_closure", _mock_prompt)

            tab_index = render_tabs.indexOf(live_tab)
            render_tabs._close_dynamic_tab(run_id, tab_name, tab_index)

            # Dialog invoked but tab remains and run is not marked deleted
            assert prompt_calls == [run_id]
            assert render_tabs.indexOf(live_tab) != -1
            assert run_id in render_tabs._agent_tabs
            assert not store.is_run_deleted(run_id)

            render_tabs.deleteLater()
            store.close()

    def test_close_tab_apply_all_removes_all_tabs(self, qapp, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = TelemetrySQLiteStore(db_path)
            service = TelemetryService()
            service.attach_store(store)

            render_tabs = RenderTabs(telemetry_service=service)

            run_id = "close_run_batch"
            agent_id = "agent-789"
            tab_one = LiveTelemetryTab(run_id, agent_id, parent=render_tabs)
            tab_two = LiveTelemetryTab(run_id, f"{agent_id}-2", parent=render_tabs)

            render_tabs.add_dynamic_tab(run_id, f"Live-Agent-{agent_id}", tab_one)
            render_tabs.add_dynamic_tab(run_id, f"Live-Agent-{agent_id}-Extra", tab_two)

            def _mock_prompt(self, run_id_param, widget):
                return TabClosureChoice.DELETE, True

            monkeypatch.setattr(RenderTabs, "_prompt_tab_closure", _mock_prompt)

            tab_index = render_tabs.indexOf(tab_one)
            render_tabs._close_dynamic_tab(run_id, f"Live-Agent-{agent_id}", tab_index)

            assert store.is_run_deleted(run_id)
            assert run_id not in render_tabs._agent_tabs
            assert render_tabs.indexOf(tab_one) == -1
            assert render_tabs.indexOf(tab_two) == -1

            render_tabs.deleteLater()
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
