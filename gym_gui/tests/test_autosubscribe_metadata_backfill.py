import json
import sqlite3
from pathlib import Path

import pytest

from gym_gui.ui.main_window import MainWindow
from gym_gui.ui.widgets.render_tabs import RenderTabs
from gym_gui.ui.widgets.fastlane_tab import FastLaneTab


@pytest.mark.skip("Integration-heavy; requires Qt event loop and trainer daemon")
def test_autosubscribe_backfill_opens_fastlane_tab(qtbot, monkeypatch, tmp_path):
    # Placeholder test structure to remind us to cover this path.
    assert True
