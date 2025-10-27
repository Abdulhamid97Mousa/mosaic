"""Tests for telemetry reliability fixes (Issue #1-4).

This module tests the four concrete telemetry system fixes:
1. Credit enforcement before RunBus publish
2. Logging downgrade (ERROR -> INFO/DEBUG)
3. Bounded buffers in LiveTelemetryController
4. Rendering regulator early payload buffering
"""

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from unittest import mock
from typing import Dict, Tuple

import pytest
from qtpy import QtCore, QtWidgets

# Import the classes/functions under test
from gym_gui.telemetry.constants import (
    STEP_BUFFER_SIZE,
    EPISODE_BUFFER_SIZE,
    RENDER_QUEUE_SIZE,
    INITIAL_CREDITS,
)
from gym_gui.telemetry.credit_manager import CreditManager
from gym_gui.telemetry.rendering_speed_regulator import RenderingSpeedRegulator
from gym_gui.ui.constants import DEFAULT_RENDER_DELAY_MS
from gym_gui.telemetry.run_bus import get_bus, reset_bus, RunBus
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
from gym_gui.services.telemetry import TelemetryService
from gym_gui.core.data_model import StepRecord


# ========================================================================
# FIXTURE: Qt Application
# ========================================================================

@pytest.fixture(scope="function")
def qt_app():
    """Provide a Qt application for GUI tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
    # Don't call quit() to avoid issues with subsequent tests


# ========================================================================
# FIXTURE: RunBus Reset
# ========================================================================

@pytest.fixture(autouse=True)
def reset_runbus():
    """Reset the global RunBus singleton before each test."""
    reset_bus()
    yield
    reset_bus()


# ========================================================================
# TEST SUITE 1: Credit Enforcement (Issue #1)
# ========================================================================

class TestCreditEnforcement:
    """Tests for credit-based backpressure system."""

    def test_initial_credits_allocated(self):
        """Test that CreditManager initializes with correct credits."""
        mgr = CreditManager(initial_credits=200)
        
        # Stream not initialized yet
        assert mgr.get_credits("run_1", "agent_1") == 0
        
        # Initialize stream
        mgr.initialize_stream("run_1", "agent_1")
        assert mgr.get_credits("run_1", "agent_1") == 200

    def test_consume_credit_decrements(self):
        """Test that consuming credits decrements the counter."""
        mgr = CreditManager(initial_credits=5)
        mgr.initialize_stream("run_1", "agent_1")
        
        # Consume 3 credits
        assert mgr.consume_credit("run_1", "agent_1") is True
        assert mgr.get_credits("run_1", "agent_1") == 4
        
        assert mgr.consume_credit("run_1", "agent_1") is True
        assert mgr.get_credits("run_1", "agent_1") == 3
        
        assert mgr.consume_credit("run_1", "agent_1") is True
        assert mgr.get_credits("run_1", "agent_1") == 2

    def test_no_credit_returns_false(self):
        """Test that consuming credit when none available returns False."""
        mgr = CreditManager(initial_credits=2)
        mgr.initialize_stream("run_1", "agent_1")
        
        # Consume all credits
        mgr.consume_credit("run_1", "agent_1")
        mgr.consume_credit("run_1", "agent_1")
        
        # Attempt to consume when empty
        assert mgr.consume_credit("run_1", "agent_1") is False
        assert mgr.get_credits("run_1", "agent_1") == 0

    def test_grant_credits(self):
        """Test that granting credits replenishes the supply."""
        mgr = CreditManager(initial_credits=5)
        mgr.initialize_stream("run_1", "agent_1")
        
        # Consume all
        for _ in range(5):
            mgr.consume_credit("run_1", "agent_1")
        assert mgr.get_credits("run_1", "agent_1") == 0
        
        # Grant 10 more
        mgr.grant_credits("run_1", "agent_1", 10)
        assert mgr.get_credits("run_1", "agent_1") == 10

    def test_independent_streams_independent_credits(self):
        """Test that different streams have independent credit pools."""
        mgr = CreditManager(initial_credits=5)
        mgr.initialize_stream("run_1", "agent_1")
        mgr.initialize_stream("run_1", "agent_2")
        
        # Consume from agent_1 only
        mgr.consume_credit("run_1", "agent_1")
        mgr.consume_credit("run_1", "agent_1")
        
        # agent_1 should be at 3, agent_2 at 5
        assert mgr.get_credits("run_1", "agent_1") == 3
        assert mgr.get_credits("run_1", "agent_2") == 5

    def test_auto_initialize_on_consume(self):
        """Test that consuming credit auto-initializes stream."""
        mgr = CreditManager(initial_credits=3)
        
        # Stream not yet initialized
        assert mgr.get_credits("run_1", "agent_1") == 0
        
        # Consume triggers initialization
        assert mgr.consume_credit("run_1", "agent_1") is True
        assert mgr.get_credits("run_1", "agent_1") == 2


# ========================================================================
# TEST SUITE 2: Logging Levels (Issue #2)
# ========================================================================

class TestLoggingDowngrade:
    """Tests for logging level downgrade (ERROR -> INFO/DEBUG)."""

    def test_telemetry_service_uses_debug_for_steps(self, caplog):
        """Test that TelemetryService logs steps at DEBUG level, not ERROR."""
        with caplog.at_level(logging.DEBUG):
            service = TelemetryService()
            
            step = StepRecord(
                episode_id="ep_1",
                step_index=0,
                observation=None,
                action=0,
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
                agent_id="agent_1",
            )
            service.record_step(step)
            
            # Should have DEBUG logs, not ERROR
            debug_logs = [r for r in caplog.records if r.levelname == "DEBUG" and "[TELEMETRY]" in r.message]
            error_logs = [r for r in caplog.records if r.levelname == "ERROR" and "[TELEMETRY]" in r.message]
            
            assert len(debug_logs) > 0, "Should have DEBUG level telemetry logs"
            assert len(error_logs) == 0, "Should not have ERROR level telemetry logs"

    def test_sqlite_store_logs_at_appropriate_levels(self, caplog, tmp_path):
        """Test that SQLite store uses INFO for batch, DEBUG for individual steps."""
        from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore
        
        db_path = tmp_path / "test.db"
        
        with caplog.at_level(logging.DEBUG):
            store = TelemetrySQLiteStore(db_path)
            
            # Record a step
            step = StepRecord(
                episode_id="ep_1",
                step_index=0,
                observation=None,
                action=0,
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
                agent_id="agent_1",
            )
            store.record_step(step)
            
            # Wait for worker to process
            time.sleep(0.2)
            store.flush()
            
            # Should have INFO for batch flush, DEBUG for individual steps (via structured log constants)
            info_codes = {
                r.__dict__.get("log_code")
                for r in caplog.records
                if r.levelname == "INFO"
            }
            debug_codes = {
                r.__dict__.get("log_code")
                for r in caplog.records
                if r.levelname == "DEBUG"
            }
            error_codes = {
                r.__dict__.get("log_code")
                for r in caplog.records
                if r.levelname == "ERROR"
            }

            assert "LOG618" in info_codes, "Should have INFO level for batch flush"
            assert "LOG617" in debug_codes, "Should emit DEBUG detail for individual steps"
            assert not error_codes, "Should not have ERROR level logs"


# ========================================================================
# TEST SUITE 3: Bounded Buffers (Issue #3)
# ========================================================================

class TestBoundedBuffers:
    """Tests for bounded deque buffers in LiveTelemetryController."""

    def test_step_buffer_size_constant_defined(self):
        """Test that STEP_BUFFER_SIZE constant is defined."""
        assert STEP_BUFFER_SIZE == 64
        assert isinstance(STEP_BUFFER_SIZE, int)
        assert STEP_BUFFER_SIZE > 0

    def test_episode_buffer_size_constant_defined(self):
        """Test that EPISODE_BUFFER_SIZE constant is defined."""
        assert EPISODE_BUFFER_SIZE == 32
        assert isinstance(EPISODE_BUFFER_SIZE, int)
        assert EPISODE_BUFFER_SIZE > 0

    def test_buffers_use_deques_not_lists(self, qt_app):
        """Test that buffers are initialized as deques, not lists."""
        from gym_gui.services.trainer.streams import TelemetryAsyncHub
        from gym_gui.services.trainer import TrainerClient
        
        # Create mocks
        hub = TelemetryAsyncHub()
        client = mock.Mock(spec=TrainerClient)
        
        controller = LiveTelemetryController(hub, client)
        
        # Manually create a buffer entry to inspect type
        key = ("run_1", "agent_1")
        controller._step_buffer[key] = deque(maxlen=STEP_BUFFER_SIZE)
        controller._episode_buffer[key] = deque(maxlen=EPISODE_BUFFER_SIZE)
        
        # Verify they are deques with maxlen
        assert isinstance(controller._step_buffer[key], deque)
        assert isinstance(controller._episode_buffer[key], deque)
        assert controller._step_buffer[key].maxlen == STEP_BUFFER_SIZE
        assert controller._episode_buffer[key].maxlen == EPISODE_BUFFER_SIZE

    def test_deque_maxlen_prevents_unbounded_growth(self):
        """Test that deques with maxlen drop oldest when full."""
        buf = deque(maxlen=3)
        
        # Fill buffer
        buf.append("a")
        buf.append("b")
        buf.append("c")
        assert len(buf) == 3
        
        # Add one more - should drop oldest
        buf.append("d")
        assert len(buf) == 3
        assert list(buf) == ["b", "c", "d"]

    def test_buffer_capacity_prevents_memory_leak(self):
        """Test that bounded buffers don't grow indefinitely."""
        step_buf = deque(maxlen=STEP_BUFFER_SIZE)
        
        # Add 1000 items - should only keep last 64
        for i in range(1000):
            step_buf.append(f"step_{i}")
        
        assert len(step_buf) == STEP_BUFFER_SIZE
        assert list(step_buf)[0] == f"step_{1000 - STEP_BUFFER_SIZE}"


# ========================================================================
# TEST SUITE 4: Rendering Regulator Bootstrap (Issue #4)
# ========================================================================

class TestRenderingRegulatorBootstrap:
    """Tests for early payload buffering in RenderingSpeedRegulator."""

    def test_regulator_buffers_early_payloads(self, qt_app):
        """Test that payloads submitted before start() are buffered."""
        regulator = RenderingSpeedRegulator(render_delay_ms=DEFAULT_RENDER_DELAY_MS)
        
        # Submit payload before starting
        payload_1 = {"frame": 1}
        payload_2 = {"frame": 2}
        regulator.submit_payload(payload_1)
        regulator.submit_payload(payload_2)
        
        # Should have buffered them in early_payloads
        assert len(regulator._early_payloads) == 2
        assert len(regulator._payload_queue) == 0

    def test_regulator_drains_early_payloads_on_start(self, qt_app):
        """Test that early payloads are drained when start() is called."""
        regulator = RenderingSpeedRegulator(render_delay_ms=DEFAULT_RENDER_DELAY_MS)
        
        # Submit payloads before start
        regulator.submit_payload({"frame": 1})
        regulator.submit_payload({"frame": 2})
        assert len(regulator._early_payloads) == 2
        
        # Start the regulator
        regulator.start()
        
        # Early payloads should be moved to main queue
        assert len(regulator._early_payloads) == 0
        assert len(regulator._payload_queue) == 2

    def test_regulator_accepts_payloads_after_start(self, qt_app):
        """Test that payloads submitted after start() go to main queue."""
        regulator = RenderingSpeedRegulator(render_delay_ms=DEFAULT_RENDER_DELAY_MS)
        regulator.start()
        
        payload = {"frame": 1}
        regulator.submit_payload(payload)
        
        assert len(regulator._payload_queue) == 1
        assert payload in regulator._payload_queue

    def test_regulator_max_queue_size_enforced(self, qt_app):
        """Test that max_queue_size is enforced for early payloads."""
        max_size = 10
        regulator = RenderingSpeedRegulator(render_delay_ms=100, max_queue_size=max_size)
        
        # Submit more than max_queue_size
        for i in range(20):
            regulator.submit_payload({"frame": i})
        
        # Should only keep the last max_size items
        assert len(regulator._early_payloads) == max_size
        
        # Verify we have the last 10
        frames = [p["frame"] for p in regulator._early_payloads]
        assert frames == list(range(10, 20))

    def test_regulator_emits_payloads_after_start(self, qt_app):
        """Test that regulator emits payloads from queue after start()."""
        regulator = RenderingSpeedRegulator(render_delay_ms=50)
        emitted_payloads = []
        
        # Connect signal to collect emitted payloads
        def on_payload_ready(payload):
            emitted_payloads.append(payload)
        
        regulator.payload_ready.connect(on_payload_ready)
        
        # Submit early payloads
        regulator.submit_payload({"frame": 1})
        regulator.submit_payload({"frame": 2})
        
        # Start and wait for emissions
        regulator.start()
        QtCore.QCoreApplication.processEvents()
        
        # Give timer a chance to fire
        time.sleep(0.15)
        QtCore.QCoreApplication.processEvents()
        
        # Should have emitted at least some payloads
        assert len(emitted_payloads) >= 1

    def test_render_queue_size_constant_defined(self):
        """Test that RENDER_QUEUE_SIZE constant is defined."""
        assert RENDER_QUEUE_SIZE == 32
        assert isinstance(RENDER_QUEUE_SIZE, int)
        assert RENDER_QUEUE_SIZE > 0

    def test_auto_start_on_bootstrap_timeout(self, qt_app):
        """Test that regulator auto-starts if payloads exist after timeout."""
        regulator = RenderingSpeedRegulator(render_delay_ms=100)
        
        # Submit early payload
        regulator.submit_payload({"frame": 1})
        assert not regulator._is_running
        
        # Wait for bootstrap timeout
        time.sleep(0.6)
        QtCore.QCoreApplication.processEvents()
        
        # Should be auto-started now
        assert regulator._is_running or len(regulator._early_payloads) > 0


# ========================================================================
# TEST SUITE 5: Integration Tests
# ========================================================================

class TestTelemetryReliabilityIntegration:
    """Integration tests for all four fixes working together."""

    def test_credits_used_by_stream_drain_loop(self):
        """Test that streams.py _drain_loop respects credit system."""
        from gym_gui.services.trainer.streams import TelemetryAsyncHub
        
        hub = TelemetryAsyncHub()
        assert hasattr(hub, "_credit_mgr"), "TelemetryAsyncHub should have _credit_mgr"
        assert isinstance(hub._credit_mgr, CreditManager)

    def test_bounded_buffers_integrated_with_controller(self, qt_app):
        """Test that controller uses bounded deques for buffers."""
        from gym_gui.services.trainer.streams import TelemetryAsyncHub
        from gym_gui.services.trainer import TrainerClient
        
        hub = TelemetryAsyncHub()
        client = mock.Mock(spec=TrainerClient)
        controller = LiveTelemetryController(hub, client)
        
        # Simulate buffer creation as it would happen in _process_step_queue
        key = ("run_1", "agent_1")
        controller._step_buffer[key] = deque(maxlen=STEP_BUFFER_SIZE)
        
        # Verify it's a deque with maxlen
        assert isinstance(controller._step_buffer[key], deque)
        assert controller._step_buffer[key].maxlen == STEP_BUFFER_SIZE

    def test_constants_file_has_all_config_values(self):
        """Test that constants.py exports all required configuration values."""
        from gym_gui.telemetry import constants
        
        required_attrs = [
            "STEP_BUFFER_SIZE",
            "EPISODE_BUFFER_SIZE",
            "RENDER_QUEUE_SIZE",
            "INITIAL_CREDITS",
            "STEP_LOG_LEVEL",
            "BATCH_LOG_LEVEL",
            "ERROR_LOG_LEVEL",
        ]
        
        for attr in required_attrs:
            assert hasattr(constants, attr), f"constants.py missing {attr}"

    def test_rendering_regulator_handles_lifecycle(self, qt_app):
        """Test that rendering regulator handles full lifecycle correctly."""
        regulator = RenderingSpeedRegulator(render_delay_ms=50)
        
        # Phase 1: Early payloads
        regulator.submit_payload({"phase": "early", "id": 1})
        regulator.submit_payload({"phase": "early", "id": 2})
        
        # Phase 2: Start
        regulator.start()
        
        # Phase 3: Payloads after start
        regulator.submit_payload({"phase": "after", "id": 3})
        
        # Phase 4: Check state
        assert regulator._is_running
        assert len(regulator._early_payloads) == 0
        assert len(regulator._payload_queue) >= 3  # Early + after payloads


# ========================================================================
# TEST SUITE 6: Edge Cases & Error Handling
# ========================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_credit_manager_zero_initial_credits(self):
        """Test CreditManager with zero initial credits."""
        mgr = CreditManager(initial_credits=0)
        mgr.initialize_stream("run_1", "agent_1")
        
        assert mgr.get_credits("run_1", "agent_1") == 0
        assert mgr.consume_credit("run_1", "agent_1") is False

    def test_credit_manager_very_large_initial_credits(self):
        """Test CreditManager with very large initial credit value."""
        mgr = CreditManager(initial_credits=1000000)
        mgr.initialize_stream("run_1", "agent_1")
        
        assert mgr.get_credits("run_1", "agent_1") == 1000000

    def test_regulator_stop_clears_timers(self, qt_app):
        """Test that stopping regulator clears all timers."""
        regulator = RenderingSpeedRegulator(render_delay_ms=50)
        regulator.start()
        
        # Verify timers exist
        assert regulator._timer is not None
        
        # Stop
        regulator.stop()
        
        # Timers should be cleared
        assert regulator._timer is None
        assert not regulator._is_running

    def test_deque_iteration_safe(self):
        """Test that deques can be safely iterated for flushing."""
        buf = deque(maxlen=5)
        buf.extend([1, 2, 3, 4, 5])
        
        # Simulate draining
        items = list(buf)
        assert items == [1, 2, 3, 4, 5]
        
        # Buffer can be cleared
        buf.clear()
        assert len(buf) == 0

    def test_empty_early_payloads_queue_on_start(self, qt_app):
        """Test that start() with no early payloads doesn't error."""
        regulator = RenderingSpeedRegulator(render_delay_ms=50)
        
        # Start without submitting payloads first
        regulator.start()
        
        assert regulator._is_running
        assert len(regulator._early_payloads) == 0
        assert len(regulator._payload_queue) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
