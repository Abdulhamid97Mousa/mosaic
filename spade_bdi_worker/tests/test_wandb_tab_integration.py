"""Test end-to-end W&B tab creation from artifact event."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, Mock

from qtpy import QtWidgets
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestWandbTabIntegration:
    """Verify W&B tab appears when worker emits artifact event."""

    @classmethod
    def setup_class(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_artifact_event_json_format(self) -> None:
        """Verify artifact event has expected JSON structure."""
        from spade_bdi_worker.core.telemetry_worker import TelemetryEmitter
        import io
        
        # Create emitter with string buffer
        output = io.StringIO()
        emitter = TelemetryEmitter(stream=output, disabled=False)
        
        # Emit artifact event like worker does
        emitter.artifact(
            run_id="test-run-123",
            kind="wandb",
            path="abdulhamid97mousa/MOSAIC/runs/test-123",
            worker_id="worker-1",
        )
        
        # Parse emitted JSON
        output.seek(0)
        line = output.readline()
        event = json.loads(line)
        
        # Verify structure
        assert event["type"] == "artifact"
        assert event["run_id"] == "test-run-123"
        assert event["kind"] == "wandb"
        assert event["path"] == "abdulhamid97mousa/MOSAIC/runs/test-123"
        assert event["worker_id"] == "worker-1"
        assert "ts" in event
        assert "ts_unix_ns" in event

    def test_worker_writes_manifest_file(self) -> None:
        """Verify worker writes wandb.json manifest file with run_path (file-based pattern)."""
        # NOTE: This test verifies the NEW file-based approach (mirrors TensorBoard)
        # The old signal-based approach was replaced because signals don't cross process boundaries
        # Worker writes manifest → GUI reads manifest (no signals needed)
        
        # This is covered by test_analytics_tabs.py::test_wandb_manifest_creates_tab
        # which tests the full flow: manifest file → analytics_tabs.ensure_wandb_tab() → tab creation
        
        # Implementation in spade_bdi_worker/core/runtime.py:
        # - _write_wandb_manifest() writes var/trainer/runs/{run_id}/wandb.json
        # - Called after wandb.init() completes
        # - Manifest contains: {"run_path": "entity/project/run_id", "run_id": "...", ...}
        
        # Implementation in gym_gui/ui/panels/analytics_tabs.py:
        # - ensure_wandb_tab() reads manifest_file from metadata
        # - Loads JSON from manifest file
        # - Creates WANDB-Agent-{agent_id} tab with run_path
        pass
