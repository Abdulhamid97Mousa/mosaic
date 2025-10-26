"""Rendering speed regulator for decoupling visual rendering from table updates.

This component maintains a separate queue for visual payloads and emits them
at a controlled rate (independent of table update speed). This allows:
- Tables to update immediately (no throttle)
- Visual rendering to update at a configurable rate (e.g., 10 FPS)
- Independent control of both speeds via UI sliders
"""

import logging
from typing import Any, Optional
from collections import deque
from qtpy import QtCore, QtWidgets

from gym_gui.logging_config.log_constants import (
    LOG_RENDER_DROPPED_FRAME,
    LOG_RENDER_REGULATOR_NOT_STARTED,
)
from gym_gui.telemetry.constants import RENDER_QUEUE_SIZE, RENDER_BOOTSTRAP_TIMEOUT_MS
from gym_gui.ui.constants import DEFAULT_RENDER_DELAY_MS

_LOGGER = logging.getLogger(__name__)


class RenderingSpeedRegulator(QtCore.QObject):
    """Regulates visual rendering speed independent of table updates.
    
    Maintains a queue of render payloads and emits them at a fixed interval,
    allowing smooth visual rendering even when telemetry arrives at high frequency.
    
    Signals:
        payload_ready: Emitted when a payload is ready to render
                      (payload: dict)
    """

    # Signal emitted when payload is ready to render
    payload_ready = QtCore.Signal(dict)  # type: ignore[attr-defined]

    def __init__(
        self,
        render_delay_ms: int = DEFAULT_RENDER_DELAY_MS,
        max_queue_size: int = RENDER_QUEUE_SIZE,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the rendering speed regulator.
        
        Args:
            render_delay_ms: Delay between renders in milliseconds (default 100ms = 10 FPS)
            max_queue_size: Maximum number of payloads to queue (default 32)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self._render_delay_ms = max(1, render_delay_ms)  # Minimum 1ms
        self._max_queue_size = max(1, max_queue_size)
        self._payload_queue: deque[dict[str, Any]] = deque(maxlen=max_queue_size)
        self._early_payloads: deque[dict[str, Any]] = deque(maxlen=max_queue_size)  # Buffer before start()
        self._timer: Optional[QtCore.QTimer] = None
        self._is_running = False
        self._bootstrap_timer: Optional[QtCore.QTimer] = None
        
        _LOGGER.debug(
            "RenderingSpeedRegulator initialized",
            extra={
                "render_delay_ms": self._render_delay_ms,
                "max_queue_size": self._max_queue_size,
            },
        )

    def start(self) -> None:
        """Start the rendering timer."""
        if self._is_running:
            _LOGGER.debug("RenderingSpeedRegulator already running")
            return
        
        self._is_running = True
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._emit_next_payload)
        self._timer.start(self._render_delay_ms)
        
        # Drain early payloads that arrived before start()
        early_count = len(self._early_payloads)
        if early_count > 0:
            while self._early_payloads:
                early_payload = self._early_payloads.popleft()
                self._payload_queue.append(early_payload)
            _LOGGER.info(
                "RenderingSpeedRegulator started and drained early payloads",
                extra={"render_delay_ms": self._render_delay_ms, "early_payloads": early_count},
            )
        else:
            _LOGGER.info(
                "RenderingSpeedRegulator started",
                extra={"render_delay_ms": self._render_delay_ms},
            )

    def stop(self) -> None:
        """Stop the rendering timer."""
        if not self._is_running:
            _LOGGER.debug("RenderingSpeedRegulator already stopped")
            return
        
        self._is_running = False
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        
        if self._bootstrap_timer is not None:
            self._bootstrap_timer.stop()
            self._bootstrap_timer.deleteLater()
            self._bootstrap_timer = None
        
        _LOGGER.info("RenderingSpeedRegulator stopped")

    def submit_payload(self, payload: dict[str, Any]) -> None:
        """Submit a payload for rendering.
        
        Payloads are queued and emitted at the configured rate.
        If queue is full, oldest payload is dropped.
        Early payloads (before start() called) are buffered separately
        and drained when start() is called.
        
        Args:
            payload: The render payload to queue
        """
        if not self._is_running:
            # Buffer early payloads before start() is called
            was_full = len(self._early_payloads) >= self._max_queue_size
            self._early_payloads.append(payload)
            
            if not was_full:
                level = (
                    LOG_RENDER_REGULATOR_NOT_STARTED.level
                    if isinstance(LOG_RENDER_REGULATOR_NOT_STARTED.level, int)
                    else getattr(logging, LOG_RENDER_REGULATOR_NOT_STARTED.level)
                )
                _LOGGER.log(
                    level,
                    "%s %s",
                    LOG_RENDER_REGULATOR_NOT_STARTED.code,
                    LOG_RENDER_REGULATOR_NOT_STARTED.message,
                    extra={
                        "early_buffer_size": len(self._early_payloads),
                        "log_code": LOG_RENDER_REGULATOR_NOT_STARTED.code,
                        "component": LOG_RENDER_REGULATOR_NOT_STARTED.component,
                        "subcomponent": LOG_RENDER_REGULATOR_NOT_STARTED.subcomponent,
                        "tags": ",".join(LOG_RENDER_REGULATOR_NOT_STARTED.tags),
                    },
                )
            
            # Auto-start if we've been buffering for too long
            if not self._bootstrap_timer and len(self._early_payloads) > 0:
                self._bootstrap_timer = QtCore.QTimer(self)
                self._bootstrap_timer.singleShot(RENDER_BOOTSTRAP_TIMEOUT_MS, self._auto_start)
            
            return
        
        # Queue is automatically limited by deque(maxlen=...)
        # If full, oldest item is automatically dropped
        was_full = len(self._payload_queue) >= self._max_queue_size
        self._payload_queue.append(payload)
        
        if was_full:
            level = (
                LOG_RENDER_DROPPED_FRAME.level
                if isinstance(LOG_RENDER_DROPPED_FRAME.level, int)
                else getattr(logging, LOG_RENDER_DROPPED_FRAME.level)
            )
            _LOGGER.log(
                level,
                "%s %s",
                LOG_RENDER_DROPPED_FRAME.code,
                LOG_RENDER_DROPPED_FRAME.message,
                extra={
                    "queue_size": len(self._payload_queue),
                    "log_code": LOG_RENDER_DROPPED_FRAME.code,
                    "component": LOG_RENDER_DROPPED_FRAME.component,
                    "subcomponent": LOG_RENDER_DROPPED_FRAME.subcomponent,
                    "tags": ",".join(LOG_RENDER_DROPPED_FRAME.tags),
                },
            )
        else:
            _LOGGER.debug(
                "RenderingSpeedRegulator payload queued",
                extra={"queue_size": len(self._payload_queue)},
            )

    def set_render_delay(self, delay_ms: int) -> None:
        """Set the rendering delay (time between renders).
        
        Args:
            delay_ms: Delay in milliseconds (e.g., 100ms = 10 FPS, 50ms = 20 FPS)
        """
        self._render_delay_ms = max(1, delay_ms)
        
        if self._timer is not None and self._is_running:
            self._timer.setInterval(self._render_delay_ms)
        
        _LOGGER.info(
            "RenderingSpeedRegulator delay updated",
            extra={"render_delay_ms": self._render_delay_ms},
        )

    def get_render_delay(self) -> int:
        """Get the current rendering delay in milliseconds."""
        return self._render_delay_ms

    def get_queue_size(self) -> int:
        """Get the current number of payloads in queue."""
        return len(self._payload_queue)

    def clear_queue(self) -> None:
        """Clear all queued payloads."""
        self._payload_queue.clear()
        _LOGGER.debug("RenderingSpeedRegulator queue cleared")

    def _emit_next_payload(self) -> None:
        """Emit the next payload from the queue (called by timer)."""
        if not self._payload_queue:
            _LOGGER.debug("RenderingSpeedRegulator queue empty, skipping emit")
            return
        
        payload = self._payload_queue.popleft()
        _LOGGER.debug(
            "RenderingSpeedRegulator emitting payload",
            extra={"queue_size": len(self._payload_queue)},
        )
        self.payload_ready.emit(payload)

    def _auto_start(self) -> None:
        """Auto-start the regulator if early payloads exist and not yet started."""
        if not self._is_running and len(self._early_payloads) > 0:
            _LOGGER.info(
                "RenderingSpeedRegulator auto-starting due to early payloads timeout",
                extra={"early_payloads_buffered": len(self._early_payloads)},
            )
            self.start()
        if self._bootstrap_timer is not None:
            self._bootstrap_timer.deleteLater()
            self._bootstrap_timer = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
