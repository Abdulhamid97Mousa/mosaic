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
        render_delay_ms: int = 100,
        max_queue_size: int = 32,
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
        self._timer: Optional[QtCore.QTimer] = None
        self._is_running = False
        
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
        
        _LOGGER.info("RenderingSpeedRegulator stopped")

    def submit_payload(self, payload: dict[str, Any]) -> None:
        """Submit a payload for rendering.
        
        Payloads are queued and emitted at the configured rate.
        If queue is full, oldest payload is dropped.
        
        Args:
            payload: The render payload to queue
        """
        if not self._is_running:
            _LOGGER.debug("RenderingSpeedRegulator not running, ignoring payload")
            return
        
        # Queue is automatically limited by deque(maxlen=...)
        # If full, oldest item is automatically dropped
        was_full = len(self._payload_queue) >= self._max_queue_size
        self._payload_queue.append(payload)
        
        if was_full:
            _LOGGER.debug(
                "RenderingSpeedRegulator queue full, dropped oldest payload",
                extra={"queue_size": len(self._payload_queue)},
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

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()

