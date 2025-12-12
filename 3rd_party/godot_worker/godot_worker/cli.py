"""CLI entry point for Godot Worker.

Usage:
    godot-worker --project /path/to/project
    godot-worker --editor --project /path/to/project
    godot-worker --headless --project /path/to/project --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from godot_worker.config import GodotConfig, GodotRenderMode
from godot_worker.launcher import get_launcher

_LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Godot Worker for MOSAIC BDI-RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path to Godot project directory (containing project.godot)",
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene file to load (relative to project)",
    )

    parser.add_argument(
        "--editor",
        action="store_true",
        help="Launch in editor mode",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (for training)",
    )

    parser.add_argument(
        "--render-mode",
        type=str,
        choices=[m.value for m in GodotRenderMode],
        default=GodotRenderMode.VULKAN.value,
        help="Rendering backend to use",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Window resolution (WIDTHxHEIGHT)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port for RL communication",
    )

    parser.add_argument(
        "--fixed-fps",
        type=int,
        default=0,
        help="Fixed FPS for simulation (0 = variable)",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print Godot version and exit",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print launcher status and exit",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for Godot Worker CLI."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    launcher = get_launcher()

    # Handle version request
    if args.version:
        version = launcher.get_version()
        if version:
            print(f"Godot version: {version}")
            return 0
        else:
            print("Godot binary not available")
            return 1

    # Handle status request
    if args.status:
        status = launcher.get_status()
        for key, value in status.items():
            print(f"{key}: {value}")
        return 0

    _LOGGER.info("Godot Worker starting...")

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except ValueError:
        _LOGGER.error(f"Invalid resolution format: {args.resolution}")
        return 1

    # Editor mode
    if args.editor:
        process, message = launcher.launch_editor(args.project)
        if process:
            _LOGGER.info(message)
            _LOGGER.info("Waiting for Godot editor to close...")
            process.process.wait()
            return 0
        else:
            _LOGGER.error(message)
            return 1

    # Create configuration
    config = GodotConfig(
        project_path=args.project,
        scene_path=args.scene,
        render_mode=GodotRenderMode(args.render_mode),
        resolution=resolution,
        port=args.port,
        headless=args.headless,
        verbose=args.verbose,
        fixed_fps=args.fixed_fps,
    )

    _LOGGER.info(f"Configuration: {config}")

    # Launch Godot
    process, message = launcher.launch(config)
    if process:
        _LOGGER.info(message)
        _LOGGER.info("Waiting for Godot to close...")
        process.process.wait()
        return 0
    else:
        _LOGGER.error(message)
        return 1


if __name__ == "__main__":
    sys.exit(main())
