from __future__ import annotations

from collections import deque
from typing import Deque


class FpsCounter:
    """Simple moving-window FPS counter.

    Tracks tick() timestamps (in seconds, monotonic) and computes instantaneous
    FPS over a sliding window.
    """

    def __init__(self, window_s: float = 1.5) -> None:
        self._times: Deque[float] = deque()
        self._window = float(max(0.1, window_s))

    def reset(self) -> None:
        self._times.clear()

    def tick(self, now_s: float) -> float:
        self._times.append(now_s)
        cutoff = now_s - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
        if len(self._times) <= 1:
            return 0.0
        duration = max(1e-6, self._times[-1] - self._times[0])
        return float((len(self._times) - 1) / duration)
