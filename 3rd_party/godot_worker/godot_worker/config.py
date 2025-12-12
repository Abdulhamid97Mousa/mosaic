"""Configuration dataclasses for Godot Worker.

This module defines configuration structures for the Godot integration,
including render modes, project settings, and runtime configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class GodotRenderMode(str, Enum):
    """Godot rendering backend options.

    These correspond to Godot's rendering drivers:
    - VULKAN: Modern Vulkan renderer (best for complex scenes)
    - OPENGL3: OpenGL 3.3 renderer (better compatibility)
    - HEADLESS: No rendering (server mode for training)
    """
    VULKAN = "vulkan"
    OPENGL3 = "opengl3"
    HEADLESS = "headless"


class GodotDisplayMode(str, Enum):
    """Godot display/window modes."""
    WINDOWED = "windowed"
    FULLSCREEN = "fullscreen"
    MAXIMIZED = "maximized"
    MINIMIZED = "minimized"


@dataclass
class GodotConfig:
    """Configuration for Godot Worker.

    Attributes:
        project_path: Path to Godot project directory (containing project.godot)
        scene_path: Optional specific scene to load (relative to project)
        render_mode: Rendering backend to use
        display_mode: Window display mode
        resolution: Window resolution as (width, height)
        port: TCP port for RL communication (None = auto-assign)
        headless: Run without display (for training)
        verbose: Enable verbose Godot output
        fixed_fps: Fixed FPS for simulation (0 = variable)
    """
    project_path: Optional[str] = None
    scene_path: Optional[str] = None
    render_mode: GodotRenderMode = GodotRenderMode.VULKAN
    display_mode: GodotDisplayMode = GodotDisplayMode.WINDOWED
    resolution: tuple[int, int] = (1280, 720)
    port: Optional[int] = None
    headless: bool = False
    verbose: bool = False
    fixed_fps: int = 0

    # Environment-specific parameters
    env_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(
                f"resolution must be positive, got {self.resolution}"
            )
        if self.port is not None and (self.port < 0 or self.port > 65535):
            raise ValueError(
                f"port must be between 0 and 65535, got {self.port}"
            )
        if self.fixed_fps < 0:
            raise ValueError(
                f"fixed_fps must be non-negative, got {self.fixed_fps}"
            )
        if self.project_path is not None:
            project_file = Path(self.project_path) / "project.godot"
            if not project_file.exists():
                raise ValueError(
                    f"No project.godot found at {self.project_path}"
                )


@dataclass
class GodotState:
    """Runtime state of a Godot session.

    Attributes:
        is_running: Whether the Godot instance is currently running
        project_path: The currently loaded project path
        scene_path: The currently loaded scene
        server_port: The TCP port for RL communication
        episode_count: Number of episodes completed
        step_count: Total steps across all episodes
    """
    is_running: bool = False
    project_path: Optional[str] = None
    scene_path: Optional[str] = None
    server_port: Optional[int] = None
    episode_count: int = 0
    step_count: int = 0
