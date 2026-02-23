"""Helpers for configuring worker processes with Fast Lane settings."""

from __future__ import annotations

from typing import Any, Dict

from gym_gui.telemetry.semconv import TelemetryEnv, VideoModes


def apply_fastlane_environment(
    env: Dict[str, Any],
    *,
    fastlane_only: bool,
    fastlane_slot: int,
    video_mode: str = VideoModes.SINGLE,
    grid_limit: int = 4,
) -> Dict[str, Any]:
    """Apply canonical Fast Lane environment variables to a dict.

    Args:
        env: Environment dict that will be passed to the worker launcher.
        fastlane_only: Whether the run should skip durable telemetry.
        fastlane_slot: Vectorized environment index feeding FastLaneWriter
            when video_mode is "single".
        video_mode: Rendering strategy for Fast Lane (single, grid, off).
        grid_limit: Number of env slots to include when video_mode is "grid".

    Returns:
        The same dict instance for chaining.
    """

    env[TelemetryEnv.FASTLANE_ONLY] = "1" if fastlane_only else "0"
    env[TelemetryEnv.FASTLANE_SLOT] = str(fastlane_slot)
    env[TelemetryEnv.FASTLANE_VIDEO_MODE] = video_mode
    env[TelemetryEnv.FASTLANE_GRID_LIMIT] = str(max(1, grid_limit))
    return env
