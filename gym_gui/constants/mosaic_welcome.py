"""Constants for MOSAIC Welcome Widget.

All configurable values for the interactive space animation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# Animation Settings
# =============================================================================

# Frame rate (milliseconds between frames)
ANIMATION_FRAME_MS = 16  # ~60 FPS

# Orbit animation speeds
SATELLITE_ORBIT_SPEED = 0.15  # Degrees per frame
RING_ROTATION_MULTIPLIER = 15  # Ring rotation factor

# Camera settings
CAMERA_SMOOTHING = 0.1  # Camera movement smoothing factor
CAMERA_PAN_SENSITIVITY = 0.003  # Mouse drag sensitivity
CAMERA_MAX_OFFSET = 0.3  # Maximum camera pan offset

# Zoom settings
ZOOM_MIN = 0.8
ZOOM_MAX = 1.3
ZOOM_SENSITIVITY = 1200.0  # Higher = less sensitive

# Tooltip fade speed
TOOLTIP_FADE_SPEED = 0.15


# =============================================================================
# Star Field Configuration
# =============================================================================

@dataclass
class StarLayerConfig:
    """Configuration for a star parallax layer."""
    count: int
    size_min: float
    size_max: float
    parallax_factor: float


STAR_LAYERS = [
    StarLayerConfig(count=150, size_min=0.3, size_max=1.2, parallax_factor=0.05),   # Far
    StarLayerConfig(count=100, size_min=1.0, size_max=2.0, parallax_factor=0.15),   # Mid
    StarLayerConfig(count=50, size_min=1.5, size_max=3.0, parallax_factor=0.3),     # Near
]

# Star color temperature thresholds
STAR_COLOR_RED_THRESHOLD = 0.25
STAR_COLOR_YELLOW_THRESHOLD = 0.5
STAR_COLOR_WHITE_THRESHOLD = 0.75

# Star colors (R, G, B)
STAR_COLOR_RED = (255, 180, 130)      # Red giant
STAR_COLOR_YELLOW = (255, 240, 220)   # Yellow star
STAR_COLOR_WHITE = (255, 255, 255)    # White star
STAR_COLOR_BLUE = (180, 210, 255)     # Blue star

# Star twinkle settings
STAR_TWINKLE_SPEED_MIN = 0.5
STAR_TWINKLE_SPEED_MAX = 2.0


# =============================================================================
# Nebulae Configuration
# =============================================================================

@dataclass
class NebulaConfig:
    """Configuration for a nebula."""
    x: float
    y: float
    size: float
    color: Tuple[int, int, int, int]  # RGBA
    rotation: float = 0.0
    pulse_phase: float = 0.0


NEBULAE = [
    NebulaConfig(0.15, 0.2, 0.12, (100, 50, 150, 40), 0, 0),
    NebulaConfig(0.85, 0.7, 0.15, (50, 100, 150, 35), 45, 1.5),
    NebulaConfig(0.7, 0.15, 0.08, (150, 80, 50, 30), 20, 3.0),
    NebulaConfig(0.3, 0.8, 0.1, (80, 150, 100, 25), -30, 2.0),
]


# =============================================================================
# Orbit Ring Configuration
# =============================================================================

@dataclass
class OrbitRingConfig:
    """Configuration for a tilted orbit ring."""
    tilt_x: float           # Tilt angle on X axis (degrees)
    tilt_z: float           # Tilt angle on Z axis (degrees)
    radius: float           # Relative radius multiplier
    rotation_speed: float   # Rotation speed (positive = clockwise)


ORBIT_RINGS = [
    OrbitRingConfig(tilt_x=75, tilt_z=0, radius=1.0, rotation_speed=0.03),
    OrbitRingConfig(tilt_x=75, tilt_z=60, radius=1.05, rotation_speed=-0.02),
    OrbitRingConfig(tilt_x=75, tilt_z=120, radius=1.1, rotation_speed=0.025),
]

# Ring visual settings
RING_SEGMENTS = 72  # Number of segments to draw each ring
RING_GLOW_LAYERS = 3
RING_BASE_ALPHA = 70


# =============================================================================
# Satellite (Paradigm) Configuration
# =============================================================================

@dataclass
class SatelliteConfig:
    """Configuration for a satellite/paradigm."""
    name: str
    color: Tuple[int, int, int]  # RGB
    orbit_radius: float
    speed: float
    phase: float
    description: str
    features: List[str]
    ring_index: int


SATELLITES = [
    SatelliteConfig(
        name="Gymnasium",
        color=(91, 75, 138),
        orbit_radius=1.0,
        speed=1.0,
        phase=0,
        description="OpenAI Gymnasium - Standard RL API",
        features=["Single-agent environments", "Classic control tasks", "Toy text games", "Box2D physics"],
        ring_index=0
    ),
    SatelliteConfig(
        name="PettingZoo",
        color=(46, 125, 50),
        orbit_radius=1.0,
        speed=1.0,
        phase=180,
        description="PettingZoo - Multi-Agent RL",
        features=["AEC (turn-based) games", "Parallel environments", "Classic board games", "MPE scenarios"],
        ring_index=0
    ),
    SatelliteConfig(
        name="MiniGrid",
        color=(239, 108, 0),
        orbit_radius=1.0,
        speed=0.8,
        phase=90,
        description="MiniGrid - Grid World Environments",
        features=["Procedural generation", "Goal-oriented tasks", "Partial observability", "Curriculum learning"],
        ring_index=1
    ),
    SatelliteConfig(
        name="ViZDoom",
        color=(198, 40, 40),
        orbit_radius=1.0,
        speed=0.8,
        phase=270,
        description="ViZDoom - Doom-based RL Platform",
        features=["First-person shooter", "Visual learning", "Navigation tasks", "Combat scenarios"],
        ring_index=1
    ),
    SatelliteConfig(
        name="MuJoCo",
        color=(0, 131, 143),
        orbit_radius=1.0,
        speed=0.6,
        phase=45,
        description="MuJoCo - Physics Simulation",
        features=["Robotics control", "Continuous actions", "Contact dynamics", "Model predictive control"],
        ring_index=2
    ),
    SatelliteConfig(
        name="Godot",
        color=(123, 31, 162),
        orbit_radius=1.0,
        speed=0.6,
        phase=225,
        description="Godot Engine - 3D Game Environments",
        features=["Custom 3D worlds", "Game AI training", "Procedural content", "Cross-platform"],
        ring_index=2
    ),
]

# Satellite visual settings
SATELLITE_SIZE_NORMAL = 10
SATELLITE_SIZE_HOVERED = 14
SATELLITE_HIT_RADIUS = 25  # Hover detection radius


# =============================================================================
# Planet Configuration
# =============================================================================

PLANET_NAME = "MOSAIC"
PLANET_DESCRIPTION = "MOSAIC - Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads"
PLANET_FEATURES = [
    "Unified agent framework",
    "Human + AI collaboration",
    "Real-time visualization",
    "Policy mapping system"
]

# Planet visual settings
PLANET_RADIUS_FACTOR = 0.15  # Relative to min(width, height)
ORBIT_BASE_RADIUS_FACTOR = 0.32  # Relative to min(width, height)

# Planet colors (light side to shadow)
PLANET_COLORS_NORMAL = [
    (80, 120, 180),   # Light side
    (50, 80, 140),    # Mid tone
    (30, 50, 100),    # Darker
    (10, 20, 40),     # Shadow side
]

PLANET_COLORS_HOVERED = [
    (100, 150, 220),  # Light side (brighter)
    (70, 110, 180),   # Mid tone
    (45, 75, 140),    # Darker
    (15, 30, 60),     # Shadow side
]

# Atmosphere glow
ATMOSPHERE_GLOW_LAYERS = 6
ATMOSPHERE_COLOR = (80, 140, 255)

# Cloud bands
CLOUD_BAND_OFFSETS = [0.35, -0.15, 0.55, -0.35, 0.1]
CLOUD_COLOR = (180, 210, 255)


# =============================================================================
# Shooting Stars Configuration
# =============================================================================

SHOOTING_STAR_INTERVAL_MIN = 40
SHOOTING_STAR_INTERVAL_MAX = 120
SHOOTING_STAR_SPEED_MIN = 0.015
SHOOTING_STAR_SPEED_MAX = 0.03
SHOOTING_STAR_TRAIL_MIN = 40
SHOOTING_STAR_TRAIL_MAX = 80
SHOOTING_STAR_DECAY_RATE = 0.02


# =============================================================================
# Ambient Particles Configuration
# =============================================================================

PARTICLE_COUNT = 30
PARTICLE_SIZE_MIN = 0.5
PARTICLE_SIZE_MAX = 2.0
PARTICLE_SPEED_MIN = 0.3
PARTICLE_SPEED_MAX = 1.0
PARTICLE_COLOR = (150, 180, 255)


# =============================================================================
# Deep Space Background
# =============================================================================

# Primary gradient colors (center to edge)
DEEP_SPACE_COLORS = [
    (20, 20, 45),   # Center
    (12, 12, 30),   # Mid
    (6, 6, 18),     # Outer
    (2, 2, 8),      # Edge
]

# Secondary accent gradient
DEEP_SPACE_ACCENT_COLOR = (40, 20, 60, 30)


# =============================================================================
# Tooltip Settings
# =============================================================================

TOOLTIP_WIDTH = 280
TOOLTIP_OFFSET_X = 20
TOOLTIP_MARGIN = 10
TOOLTIP_BORDER_RADIUS = 12
TOOLTIP_FEATURE_LINE_HEIGHT = 18

# Tooltip colors
TOOLTIP_BG_TOP = (25, 30, 45)
TOOLTIP_BG_BOTTOM = (15, 20, 35)
TOOLTIP_SHADOW_COLOR = (0, 0, 0, 80)


# =============================================================================
# UI Text
# =============================================================================

SUBTITLE_TEXT = "Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads"
INSTRUCTION_TEXT = "Select an environment from the sidebar and click 'Load Environment' to begin"

HINT_TEXTS = [
    "Drag to look around",
    "Hover for info",
    "Scroll to zoom"
]


# =============================================================================
# Widget Minimum Size
# =============================================================================

WIDGET_MIN_WIDTH = 400
WIDGET_MIN_HEIGHT = 300
