"""Multi-panel heatmap renderer for SMAC / SMACv2 environments.

Produces a single RGB numpy array with a 2x2 grid of panels:

    [Tactical View]   [Health Heatmap]
    [Unit Type Map]   [Shield/Energy]

Each panel is rendered at ``panel_size`` x ``panel_size`` pixels (default 256),
producing a total output of (2*panel_size + HUD_HEIGHT) x (2*panel_size) x 3.

All drawing is pure numpy -- no PIL, OpenCV, or pygame dependency.

Colour palettes are reimplemented from pysc2.lib.colors to avoid a runtime
dependency on the full pysc2 package.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UnitData:
    """Normalized unit data extracted from SC2 protobuf."""

    x: float
    y: float
    z: float
    health: float
    health_max: float
    shield: float
    shield_max: float
    energy: float
    energy_max: float
    unit_type: int
    owner: int          # 1=ally, 2=enemy, 16=neutral
    facing: float       # radians
    radius: float
    is_alive: bool
    tag: int


@dataclass(slots=True)
class SMACFrameData:
    """All data needed to render one SMAC frame."""

    terrain_height: np.ndarray   # (map_x, map_y) float
    map_x: int
    map_y: int
    playable_min_x: float
    playable_min_y: float
    playable_max_x: float
    playable_max_y: float
    units: List[UnitData]
    step: int
    reward: float
    map_name: str
    n_agents: int
    n_enemies: int
    battle_won: Optional[bool] = None


# ---------------------------------------------------------------------------
# Palette generation  (reimplemented from pysc2.lib.colors)
# ---------------------------------------------------------------------------

def _piece_wise_linear(scale: int, points: List[Tuple[float, Tuple[float, float, float]]]) -> np.ndarray:
    """Piece-wise linear colour interpolation (matches pysc2)."""
    out = np.zeros((scale, 3), dtype=np.float64)
    p1, c1 = points[0]
    p2, c2 = points[1]
    c1a = np.array(c1)
    c2a = np.array(c2)
    next_pt = 2
    for i in range(1, scale):
        v = i / scale
        if v > p2 and next_pt < len(points):
            p1, c1 = p2, c2
            c1a = np.array(c1)
            p2, c2 = points[next_pt]
            c2a = np.array(c2)
            next_pt += 1
        frac = (v - p1) / max(p2 - p1, 1e-9)
        out[i] = c1a * (1 - frac) + c2a * frac
    return np.clip(out, 0, 255).astype(np.uint8)


def _smooth_hue_palette(scale: int) -> np.ndarray:
    """HSL hue wheel with S=1, L=0.5 (matches pysc2)."""
    arr = np.arange(scale, dtype=np.float64)
    h = arr * (6.0 / scale)
    x = 255.0 * (1 - np.abs(np.mod(h, 2) - 1))
    c = 255.0
    out = np.zeros((scale, 3), dtype=np.float64)
    r, g, b = out[:, 0], out[:, 1], out[:, 2]
    m = (0 < h) & (h < 1);  r[m] = c; g[m] = x[m]
    m = (1 <= h) & (h < 2); r[m] = x[m]; g[m] = c
    m = (2 <= h) & (h < 3); g[m] = c; b[m] = x[m]
    m = (3 <= h) & (h < 4); g[m] = x[m]; b[m] = c
    m = (4 <= h) & (h < 5); r[m] = x[m]; b[m] = c
    m = 5 <= h;             r[m] = c; b[m] = x[m]
    return out.astype(np.uint8)


def _shuffled_hue(scale: int) -> np.ndarray:
    """Deterministic shuffled hue palette (matches pysc2 seed=42)."""
    palette = list(_smooth_hue_palette(scale))
    import random as _rng
    _rng.Random(42).shuffle(palette)
    return np.array(palette, dtype=np.uint8)


# Pre-built palettes
PALETTE_HEIGHT_MAP: np.ndarray = _piece_wise_linear(256, [
    (0,        (0, 0, 0)),
    (40 / 255, (67, 109, 95)),
    (50 / 255, (168, 152, 129)),
    (60 / 255, (154, 124, 90)),
    (70 / 255, (117, 150, 96)),
    (80 / 255, (166, 98, 97)),
    (1,        (255, 255, 100)),
])

PALETTE_HOT: np.ndarray = _piece_wise_linear(256, [
    (0,   (127, 0, 0)),
    (0.2, (255, 0, 0)),
    (0.6, (255, 255, 0)),
    (1,   (255, 255, 255)),
])

PALETTE_WINTER: np.ndarray = _piece_wise_linear(256, [
    (0, (0, 127, 102)),
    (1, (255, 255, 102)),
])

PALETTE_PLAYER_RELATIVE: np.ndarray = np.array([
    [0,   0,   0],     # 0: Background / dead
    [0,   142, 0],     # 1: Self  (green)
    [255, 255, 0],     # 2: Ally  (yellow)
    [129, 166, 196],   # 3: Neutral (cyan)
    [113, 25,  34],    # 4: Enemy (red)
], dtype=np.uint8)

PALETTE_UNIT_TYPE: np.ndarray = _shuffled_hue(64)


# ---------------------------------------------------------------------------
# Minimal 5x7 bitmap font (upper-case, digits, basic punctuation)
# ---------------------------------------------------------------------------

_FONT_5X7: Dict[str, List[int]] = {
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "J": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    "K": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    "Q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    "S": [0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E],
    "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "V": [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
    "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
    "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
    "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "2": [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
    "3": [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
    "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    "7": [0x1F, 0x01, 0x02, 0x04, 0x04, 0x04, 0x04],
    "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    " ": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    ":": [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
    ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
    "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
    "+": [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
    "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
    "[": [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
    "]": [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
    "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
    ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
    "_": [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
}


def _draw_text(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: Tuple[int, int, int] = (200, 200, 200),
) -> None:
    """Render text using the embedded 5x7 bitmap font."""
    h, w = canvas.shape[:2]
    cx = x
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch)
        if glyph is None:
            cx += 4  # unknown char -> small space
            continue
        for row_idx, row_bits in enumerate(glyph):
            py = y + row_idx
            if py < 0 or py >= h:
                continue
            for col in range(5):
                if row_bits & (1 << (4 - col)):
                    px = cx + col
                    if 0 <= px < w:
                        canvas[py, px] = color
        cx += 6  # 5 pixels + 1 pixel gap


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize (pure numpy, no cv2/PIL)."""
    src_h, src_w = image.shape[:2]
    row_idx = (np.arange(target_h) * src_h // target_h).clip(0, src_h - 1)
    col_idx = (np.arange(target_w) * src_w // target_w).clip(0, src_w - 1)
    return image[row_idx][:, col_idx]


def _owner_to_relative(owner: int) -> int:
    """Map SC2 owner id to player_relative index for palette lookup."""
    if owner == 1:
        return 1   # Self
    if owner == 2:
        return 4   # Enemy
    if owner == 16:
        return 3   # Neutral
    return 0       # Background / unknown


# ---------------------------------------------------------------------------
# Frame data extraction
# ---------------------------------------------------------------------------

def extract_frame_data(
    smac_env: Any,
    step: int,
    map_name: str,
    playable_area: Tuple[float, float, float, float],
) -> Optional[SMACFrameData]:
    """Extract a ``SMACFrameData`` from a SMAC v1 or v2 environment.

    Works with both ``StarCraft2Env`` (v1) and
    ``StarCraftCapabilityEnvWrapper`` (v2) because the v2 wrapper proxies
    attribute access via ``__getattr__``.

    Parameters
    ----------
    smac_env:
        The SMAC env instance (v1 or v2 wrapper).
    step:
        Current step counter from the adapter.
    map_name:
        Map name string for display.
    playable_area:
        Cached ``(min_x, min_y, max_x, max_y)`` from ``game_info()``.
    """
    # Reach through v2 wrapper if needed
    env = getattr(smac_env, "env", smac_env)

    if env._obs is None:
        return None

    terrain = getattr(env, "terrain_height", None)
    if terrain is None:
        return None

    units: List[UnitData] = []
    try:
        for u in env._obs.observation.raw_data.units:
            units.append(UnitData(
                x=u.pos.x,
                y=u.pos.y,
                z=u.pos.z,
                health=u.health,
                health_max=u.health_max,
                shield=u.shield,
                shield_max=u.shield_max,
                energy=u.energy,
                energy_max=u.energy_max,
                unit_type=u.unit_type,
                owner=u.owner,
                facing=u.facing,
                radius=u.radius,
                is_alive=u.health > 0,
                tag=u.tag,
            ))
    except Exception:
        _LOGGER.debug("Failed to extract units from SMAC obs", exc_info=True)
        return None

    n_agents = len(getattr(env, "agents", {}))
    n_enemies = len(getattr(env, "enemies", {}))
    reward = getattr(env, "reward", 0.0)

    return SMACFrameData(
        terrain_height=terrain,
        map_x=env.map_x,
        map_y=env.map_y,
        playable_min_x=playable_area[0],
        playable_min_y=playable_area[1],
        playable_max_x=playable_area[2],
        playable_max_y=playable_area[3],
        units=units,
        step=step,
        reward=reward,
        map_name=map_name,
        n_agents=n_agents,
        n_enemies=n_enemies,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SMACHeatmapRenderer:
    """Multi-panel heatmap renderer for SMAC environments.

    Produces a 2x2 grid of panels plus a HUD strip at the top.
    """

    HUD_HEIGHT: int = 32
    _BLOB_SIGMA: float = 1.5  # gaussian sigma in world-coordinate units

    def __init__(self, panel_size: int = 256, show_hud: bool = True) -> None:
        self._ps = panel_size
        self._show_hud = show_hud
        self._hud_h = self.HUD_HEIGHT if show_hud else 0
        # Terrain cache
        self._cached_terrain: Optional[np.ndarray] = None
        self._cached_map_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, fd: SMACFrameData) -> np.ndarray:
        """Return a (H, W, 3) uint8 RGB array with all 4 panels + HUD."""
        ps = self._ps
        total_h = 2 * ps + self._hud_h
        total_w = 2 * ps

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Build panels
        tactical = self._render_tactical(fd)
        health = self._render_health_heatmap(fd)
        utype = self._render_unit_type_map(fd)
        shields = self._render_shield_energy(fd)

        y0 = self._hud_h
        canvas[y0:y0 + ps, 0:ps] = tactical
        canvas[y0:y0 + ps, ps:2 * ps] = health
        canvas[y0 + ps:y0 + 2 * ps, 0:ps] = utype
        canvas[y0 + ps:y0 + 2 * ps, ps:2 * ps] = shields

        # Separator lines (2px dark grey)
        canvas[y0 + ps - 1:y0 + ps + 1, :] = 40
        canvas[:, ps - 1:ps + 1] = 40

        # Panel labels
        _draw_text(canvas, 4, y0 + 4, "TACTICAL", (180, 180, 180))
        _draw_text(canvas, ps + 4, y0 + 4, "HEALTH", (180, 180, 180))
        _draw_text(canvas, 4, y0 + ps + 4, "UNIT TYPE", (180, 180, 180))
        _draw_text(canvas, ps + 4, y0 + ps + 4, "SHIELD/ENERGY", (180, 180, 180))

        if self._show_hud:
            self._draw_hud(canvas, fd)

        return canvas

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------

    def _render_tactical(self, fd: SMACFrameData) -> np.ndarray:
        """Terrain background + units coloured by allegiance."""
        panel = self._get_terrain(fd).copy()

        sf = self._scale_factor(fd)
        for u in fd.units:
            if not u.is_alive:
                continue
            px, py = self._w2p(u.x, u.y, fd)
            rel = _owner_to_relative(u.owner)
            color = PALETTE_PLAYER_RELATIVE[rel]
            r = max(4, int(u.radius * sf))
            # Semi-transparent circles so terrain shows through
            self._draw_aa_circle(panel, px, py, r, color, opacity=0.65)
            # Bright border ring for clarity
            self._draw_ring(panel, px, py, r, color)
            # Facing indicator (short line)
            self._draw_facing(panel, px, py, r, u.facing, color)
            # Health bar
            hr = u.health / max(u.health_max, 1)
            self._draw_health_bar(panel, px, py - r - 4, r * 2, hr)

        return panel

    def _render_health_heatmap(self, fd: SMACFrameData) -> np.ndarray:
        """Gaussian blobs coloured by health ratio (hot palette)."""
        panel = self._make_dark_grid(fd)

        sf = self._scale_factor(fd)
        for u in fd.units:
            if not u.is_alive:
                continue
            px, py = self._w2p(u.x, u.y, fd)
            hr = u.health / max(u.health_max, 1)
            idx = int(hr * 255)
            color = PALETTE_HOT[min(idx, 255)]
            sigma = max(6, int(self._BLOB_SIGMA * 1.5 * sf))
            self._add_gaussian_blob(panel, px, py, sigma, color, alpha=0.9)
            # Also draw a small solid dot at center for visibility
            self._draw_aa_circle(panel, px, py, max(2, int(u.radius * sf * 0.5)), color)

        return panel

    def _render_unit_type_map(self, fd: SMACFrameData) -> np.ndarray:
        """Dimmed terrain + circles coloured by unit type."""
        panel = self._get_terrain(fd).copy()
        panel = (panel.astype(np.float32) * 0.3).astype(np.uint8)

        sf = self._scale_factor(fd)
        for u in fd.units:
            if not u.is_alive:
                continue
            px, py = self._w2p(u.x, u.y, fd)
            color = PALETTE_UNIT_TYPE[u.unit_type % len(PALETTE_UNIT_TYPE)]
            r = max(4, int(u.radius * sf))
            self._draw_aa_circle(panel, px, py, r, color, opacity=0.75)
            self._draw_ring(panel, px, py, r, color)

        return panel

    def _render_shield_energy(self, fd: SMACFrameData) -> np.ndarray:
        """Blue blobs for shields, purple blobs for energy."""
        panel = self._make_dark_grid(fd)

        sf = self._scale_factor(fd)
        for u in fd.units:
            if not u.is_alive:
                continue
            px, py = self._w2p(u.x, u.y, fd)

            if u.shield_max > 0:
                sr = u.shield / max(u.shield_max, 1)
                idx = int(sr * 255)
                color = PALETTE_WINTER[min(idx, 255)]
                sigma = max(6, int(self._BLOB_SIGMA * 1.5 * sf))
                self._add_gaussian_blob(panel, px, py, sigma, color, alpha=0.85)
                self._draw_aa_circle(panel, px, py, max(2, int(u.radius * sf * 0.5)), color)

            if u.energy_max > 0:
                er = u.energy / max(u.energy_max, 1)
                color = np.array([
                    int(180 * er), 0, int(255 * er),
                ], dtype=np.uint8)
                sigma = max(5, int(self._BLOB_SIGMA * sf))
                self._add_gaussian_blob(panel, px, py, sigma, color, alpha=0.7)

        return panel

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, canvas: np.ndarray, fd: SMACFrameData) -> None:
        hud = canvas[: self._hud_h, :]
        hud[:] = [20, 20, 25]

        text = f"{fd.map_name}  STEP:{fd.step}  R:{fd.reward:.1f}  A:{fd.n_agents} E:{fd.n_enemies}"
        if fd.battle_won is True:
            text += "  [WIN]"
        elif fd.battle_won is False:
            text += "  [LOST]"

        _draw_text(canvas, 4, 12, text, (0, 200, 0))

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------

    def _draw_aa_circle(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray,
        opacity: float = 1.0,
    ) -> None:
        """Anti-aliased filled circle via distance field."""
        ps = canvas.shape[0]
        margin = 2
        x0 = max(0, cx - radius - margin)
        x1 = min(ps, cx + radius + margin + 1)
        y0 = max(0, cy - radius - margin)
        y1 = min(ps, cy + radius + margin + 1)
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0) * opacity

        for c in range(3):
            canvas[y0:y1, x0:x1, c] = (
                canvas[y0:y1, x0:x1, c] * (1.0 - alpha) + float(color[c]) * alpha
            ).astype(np.uint8)

    def _draw_ring(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray,
        thickness: float = 1.5,
    ) -> None:
        """Anti-aliased ring (outline only) for unit borders."""
        ps = canvas.shape[0]
        margin = 3
        x0 = max(0, cx - radius - margin)
        x1 = min(ps, cx + radius + margin + 1)
        y0 = max(0, cy - radius - margin)
        y1 = min(ps, cy + radius + margin + 1)
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Ring mask: bright at exactly radius distance
        ring_alpha = np.clip(1.0 - np.abs(dist - radius) / thickness, 0.0, 1.0)
        bright = np.minimum(color.astype(np.int16) + 80, 255).astype(np.uint8)

        for c in range(3):
            canvas[y0:y1, x0:x1, c] = (
                canvas[y0:y1, x0:x1, c] * (1.0 - ring_alpha)
                + float(bright[c]) * ring_alpha
            ).astype(np.uint8)

    def _add_gaussian_blob(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        sigma: int,
        color: np.ndarray,
        alpha: float = 1.0,
    ) -> None:
        """Additive gaussian blob at (cx, cy)."""
        ps = canvas.shape[0]
        extent = sigma * 3
        x0 = max(0, cx - extent)
        x1 = min(ps, cx + extent + 1)
        y0 = max(0, cy - extent)
        y1 = min(ps, cy + extent + 1)
        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        gauss = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / max(sigma ** 2, 1))
        gauss *= alpha

        for c in range(3):
            canvas[y0:y1, x0:x1, c] = np.clip(
                canvas[y0:y1, x0:x1, c].astype(np.float32) + float(color[c]) * gauss,
                0,
                255,
            ).astype(np.uint8)

    def _draw_facing(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        facing: float,
        color: np.ndarray,
    ) -> None:
        """Short line showing unit facing direction."""
        ps = canvas.shape[0]
        length = radius + 3
        ex = cx + int(length * math.cos(facing))
        ey = cy - int(length * math.sin(facing))  # y-flip

        # Bresenham-lite: iterate from (cx,cy) to (ex,ey)
        steps = max(abs(ex - cx), abs(ey - cy), 1)
        bright = np.minimum(color.astype(np.int16) + 60, 255).astype(np.uint8)
        for i in range(steps + 1):
            t = i / steps
            px = int(cx + t * (ex - cx))
            py = int(cy + t * (ey - cy))
            if 0 <= px < ps and 0 <= py < ps:
                canvas[py, px] = bright

    def _draw_health_bar(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        width: int,
        ratio: float,
    ) -> None:
        """Small horizontal health bar."""
        ps = canvas.shape[0]
        bar_h = 2
        y0 = max(0, y)
        y1 = min(ps, y + bar_h)
        x0 = max(0, x)
        x1 = min(ps, x + width)
        if y0 >= y1 or x0 >= x1:
            return

        # Background (dark)
        canvas[y0:y1, x0:x1] = [40, 10, 10]

        # Filled portion
        fill_w = max(1, int(ratio * (x1 - x0)))
        fx1 = min(x0 + fill_w, x1)
        # Green -> Yellow -> Red
        if ratio > 0.5:
            g = 200
            r = int((1 - ratio) * 2 * 200)
        else:
            g = int(ratio * 2 * 200)
            r = 200
        canvas[y0:y1, x0:fx1] = [r, g, 0]

    def _make_dark_grid(self, fd: SMACFrameData) -> np.ndarray:
        """Dark background with subtle grid lines for spatial reference."""
        panel = np.full((self._ps, self._ps, 3), 10, dtype=np.uint8)
        self._overlay_grid(panel, fd)
        return panel

    def _overlay_grid(self, panel: np.ndarray, fd: SMACFrameData) -> None:
        """Draw subtle grid lines every 4 world units."""
        ps = self._ps
        grid_step = 4.0  # world units between grid lines
        extent_x = max(fd.playable_max_x - fd.playable_min_x, 1)
        extent_y = max(fd.playable_max_y - fd.playable_min_y, 1)
        grid_color = np.array([30, 30, 35], dtype=np.uint8)

        # Vertical grid lines
        wx = fd.playable_min_x
        while wx <= fd.playable_max_x:
            px = int((wx - fd.playable_min_x) / extent_x * ps)
            if 0 <= px < ps:
                panel[:, px] = np.maximum(panel[:, px], grid_color)
            wx += grid_step

        # Horizontal grid lines
        wy = fd.playable_min_y
        while wy <= fd.playable_max_y:
            ny = 1.0 - (wy - fd.playable_min_y) / extent_y
            py = int(ny * ps)
            if 0 <= py < ps:
                panel[py, :] = np.maximum(panel[py, :], grid_color)
            wy += grid_step

    # ------------------------------------------------------------------
    # Coordinate transforms and caching
    # ------------------------------------------------------------------

    def _w2p(self, wx: float, wy: float, fd: SMACFrameData) -> Tuple[int, int]:
        """World coords -> panel pixel coords (y-flipped)."""
        extent_x = max(fd.playable_max_x - fd.playable_min_x, 1)
        extent_y = max(fd.playable_max_y - fd.playable_min_y, 1)
        nx = (wx - fd.playable_min_x) / extent_x
        ny = (wy - fd.playable_min_y) / extent_y
        ny = 1.0 - ny  # screen y is inverted
        px = int(np.clip(nx * self._ps, 0, self._ps - 1))
        py = int(np.clip(ny * self._ps, 0, self._ps - 1))
        return px, py

    def _scale_factor(self, fd: SMACFrameData) -> float:
        """Pixels per world-unit."""
        extent = max(
            fd.playable_max_x - fd.playable_min_x,
            fd.playable_max_y - fd.playable_min_y,
            1,
        )
        return self._ps / extent

    def _get_terrain(self, fd: SMACFrameData) -> np.ndarray:
        """Cached terrain background panel."""
        if self._cached_terrain is not None and self._cached_map_name == fd.map_name:
            return self._cached_terrain

        hmap = fd.terrain_height
        if hmap is None or hmap.size == 0:
            bg = np.full((self._ps, self._ps, 3), 20, dtype=np.uint8)
            self._cached_terrain = bg
            self._cached_map_name = fd.map_name
            return bg

        # terrain_height is (map_x, map_y) float -- transpose to (y, x) for image
        hmap_t = hmap.T
        # Normalise to [0, 255]
        hmin, hmax = float(hmap_t.min()), float(hmap_t.max())
        if hmax - hmin < 1e-6:
            hmap_u8 = np.full_like(hmap_t, 128, dtype=np.uint8)
        else:
            hmap_u8 = ((hmap_t - hmin) / (hmax - hmin) * 255).astype(np.uint8)

        # Apply height_map palette
        colored = PALETTE_HEIGHT_MAP[hmap_u8]  # (map_y, map_x, 3)
        colored = (colored.astype(np.float32) * 0.6).astype(np.uint8)  # dim

        # Resize to panel size
        panel = _resize_nearest(colored, self._ps, self._ps)

        # Add grid overlay (especially useful for flat maps with no height variation)
        self._overlay_grid(panel, fd)

        self._cached_terrain = panel
        self._cached_map_name = fd.map_name
        return panel
