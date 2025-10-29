"""Tests for the trainer signal singleton implementation."""

from __future__ import annotations

import os
import pytest

from PyQt6 import QtWidgets

from gym_gui.services.trainer.signals import get_trainer_signals

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_trainer_signals_singleton(qt_app):
    first = get_trainer_signals()
    second = get_trainer_signals()

    assert first is second
    # Ensure QObject base initialized
    assert hasattr(first, "training_started")


def test_trainer_signals_emitters(qt_app):
    signals = get_trainer_signals()

    received = {}

    def on_started(run_id, metadata):
        received["run_id"] = run_id
        received["metadata"] = metadata

    signals.training_started.connect(on_started)
    try:
        signals.emit_training_started("run123", {"agent_id": "agentA"})
        assert received["run_id"] == "run123"
        assert received["metadata"]["agent_id"] == "agentA"
    finally:
        signals.training_started.disconnect(on_started)
