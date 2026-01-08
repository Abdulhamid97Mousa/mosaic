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
    # Inner orbits (Workers/Paradigms - close to MOSAIC)
    OrbitRingConfig(tilt_x=75, tilt_z=0, radius=1.0, rotation_speed=0.03),
    OrbitRingConfig(tilt_x=75, tilt_z=60, radius=1.05, rotation_speed=-0.02),
    OrbitRingConfig(tilt_x=75, tilt_z=120, radius=1.1, rotation_speed=0.025),
    # Outer orbits (Research Environments - further from MOSAIC)
    OrbitRingConfig(tilt_x=70, tilt_z=30, radius=1.45, rotation_speed=0.015),
    OrbitRingConfig(tilt_x=70, tilt_z=90, radius=1.6, rotation_speed=-0.012),
    OrbitRingConfig(tilt_x=70, tilt_z=150, radius=1.75, rotation_speed=0.01),
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
        description="Godot Engine - Free & Open Source Game Engine",
        features=["2D and 3D game development", "GDScript & C# support", "Cross-platform export", "godotengine.org"],
        ring_index=2
    ),
    SatelliteConfig(
        name="CleanRL",
        color=(55, 71, 79),  # Blue-grey
        orbit_radius=1.0,
        speed=0.6,
        phase=135,
        description="CleanRL - Single-file RL Implementations",
        features=["Reference implementations", "PyTorch-based", "WandB integration", "github.com/vwxyzjn/cleanrl"],
        ring_index=2
    ),
    SatelliteConfig(
        name="XuanCe",
        color=(25, 118, 210),  # Blue
        orbit_radius=1.0,
        speed=0.6,
        phase=315,
        description="XuanCe - Comprehensive DRL Library",
        features=["PyTorch/TensorFlow/MindSpore", "50+ algorithms", "Single & Multi-agent", "github.com/agi-brain/xuance"],
        ring_index=2
    ),
    SatelliteConfig(
        name="Ray RLlib",
        color=(0, 188, 212),  # Cyan (matches HTML color scheme)
        orbit_radius=1.0,
        speed=0.55,
        phase=270,
        description="Ray RLlib - Scalable Reinforcement Learning",
        features=["Distributed training at scale", "Multi-agent support", "Industry-grade performance", "docs.ray.io/en/latest/rllib"],
        ring_index=2
    ),
    # Outer orbit environments (Research Environments)
    SatelliteConfig(
        name="NetHack",
        color=(139, 0, 0),  # Dark red
        orbit_radius=1.0,
        speed=0.4,
        phase=0,
        description="NetHack Learning Environment (NLE)",
        features=["Procedurally generated roguelike", "Hard exploration challenge", "NeurIPS 2020 benchmark", "github.com/facebookresearch/nle"],
        ring_index=3
    ),
    SatelliteConfig(
        name="MiniHack",
        color=(255, 102, 0),  # Orange
        orbit_radius=1.0,
        speed=0.35,
        phase=120,
        description="MiniHack - Customizable RL Sandbox",
        features=["Built on NetHack/NLE", "Custom environment design", "Level editor support", "github.com/facebookresearch/minihack"],
        ring_index=4
    ),
    SatelliteConfig(
        name="Crafter",
        color=(34, 139, 34),  # Forest green
        orbit_radius=1.0,
        speed=0.3,
        phase=240,
        description="Crafter - Open World Survival Benchmark",
        features=["22 achievement tasks", "Procedural 2D worlds", "Agent capability spectrum", "github.com/danijar/crafter"],
        ring_index=5
    ),
    SatelliteConfig(
        name="Procgen",
        color=(128, 0, 128),  # Purple
        orbit_radius=1.0,
        speed=0.32,
        phase=60,
        description="Procgen - Procedurally Generated Benchmark",
        features=["16 diverse environments", "Tests generalization", "Procedural level generation", "github.com/openai/procgen"],
        ring_index=5
    ),
    SatelliteConfig(
        name="Jumanji",
        color=(233, 30, 99),  # Pink/Magenta (matches welcome page)
        orbit_radius=1.0,
        speed=0.38,
        phase=300,  # Changed from 180 to make it visible (not behind planet)
        description="Jumanji - JAX-based Logic Puzzle Environments",
        features=["Hardware-accelerated (JAX)", "2048, Sudoku, Rubik's Cube", "Minesweeper, Graph Coloring", "github.com/instadeepai/jumanji"],
        ring_index=4
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
