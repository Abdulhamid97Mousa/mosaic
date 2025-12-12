"""MOSAIC Welcome Animation Widget.

Interactive space experience with MOSAIC planet, orbiting paradigm satellites,
mouse hover tooltips, panning/looking around, and parallax effects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from PyQt6.QtCore import (
    Qt, QTimer, QPointF, QRectF, pyqtSignal, QPoint
)
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QRadialGradient,
    QLinearGradient,
    QPen,
    QFont,
    QFontMetrics,
    QPainterPath,
    QBrush,
    QCursor,
    QMouseEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

from gym_gui.constants import mosaic_welcome as const


@dataclass
class Star:
    """A star with position, size, color, and twinkle phase."""
    x: float
    y: float
    base_size: float
    color_temp: float  # 0=red, 0.5=white, 1=blue
    phase: float
    twinkle_speed: float = field(default_factory=lambda: random.uniform(
        const.STAR_TWINKLE_SPEED_MIN, const.STAR_TWINKLE_SPEED_MAX))
    layer: int = 0  # Parallax layer (0=far, 1=mid, 2=near)


@dataclass
class Satellite:
    """An orbiting satellite representing a paradigm."""
    name: str
    color: QColor
    orbit_radius: float
    speed: float
    phase: float
    description: str
    features: List[str] = field(default_factory=list)
    ring_index: int = 0  # Which ring this satellite orbits on


@dataclass
class OrbitRing:
    """A tilted orbit ring around the planet."""
    tilt_x: float  # Tilt angle on X axis (degrees)
    tilt_z: float  # Tilt angle on Z axis (degrees)
    radius: float  # Relative radius
    color: QColor
    rotation_speed: float = 0.0  # How fast the ring rotates


@dataclass
class Nebula:
    """A distant nebula/galaxy."""
    x: float
    y: float
    size: float
    color: QColor
    rotation: float = 0.0
    pulse_phase: float = 0.0


class MosaicWelcomeWidget(QWidget):
    """Interactive space experience with MOSAIC planet and paradigm satellites."""

    satellite_clicked = pyqtSignal(str)  # Emitted when satellite is clicked

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._angle = 0.0
        self._time = 0.0

        # Camera/viewport for panning
        self._camera_x = 0.0
        self._camera_y = 0.0
        self._target_camera_x = 0.0
        self._target_camera_y = 0.0
        self._zoom = 1.0

        # Mouse interaction state
        self._is_dragging = False
        self._last_mouse_pos = QPoint()
        self._hovered_satellite: Optional[Satellite] = None
        self._hovered_planet = False
        self._mouse_pos = QPointF(0, 0)

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

        # Generate multi-layer star field for parallax
        self._stars: List[Star] = []
        random.seed(42)

        for layer_idx, layer_config in enumerate(const.STAR_LAYERS):
            for _ in range(layer_config.count):
                self._stars.append(Star(
                    x=random.uniform(-0.5, 1.5),
                    y=random.uniform(-0.5, 1.5),
                    base_size=random.uniform(layer_config.size_min, layer_config.size_max),
                    color_temp=random.random(),
                    phase=random.uniform(0, 2 * math.pi),
                    layer=layer_idx
                ))

        # Nebulae from config
        self._nebulae = [
            Nebula(
                n.x, n.y, n.size,
                QColor(*n.color),
                n.rotation, n.pulse_phase
            )
            for n in const.NEBULAE
        ]

        # Define tilted orbit rings from config
        self._orbit_rings = [
            OrbitRing(
                tilt_x=r.tilt_x,
                tilt_z=r.tilt_z,
                radius=r.radius,
                color=QColor(100, 100, 100, const.RING_BASE_ALPHA),  # Will be updated
                rotation_speed=r.rotation_speed
            )
            for r in const.ORBIT_RINGS
        ]

        # Paradigm satellites from config
        self._satellites = [
            Satellite(
                name=s.name,
                color=QColor(*s.color),
                orbit_radius=s.orbit_radius,
                speed=s.speed,
                phase=s.phase,
                description=s.description,
                features=s.features,
                ring_index=s.ring_index
            )
            for s in const.SATELLITES
        ]

        # Update ring colors to blend the colors of satellites on each ring
        self._update_ring_colors()

        # Planet info from config
        self._planet_description = const.PLANET_DESCRIPTION
        self._planet_features = const.PLANET_FEATURES

        # Shooting stars
        self._shooting_stars: List[Tuple[float, float, float, float, float, float]] = []
        self._next_shooting_star = 50

        # Ambient particles
        self._particles: List[Tuple[float, float, float, float, float]] = []
        for _ in range(const.PARTICLE_COUNT):
            self._particles.append((
                random.uniform(-0.5, 1.5),
                random.uniform(-0.5, 1.5),
                random.uniform(const.PARTICLE_SIZE_MIN, const.PARTICLE_SIZE_MAX),
                random.uniform(0, 2 * math.pi),
                random.uniform(const.PARTICLE_SPEED_MIN, const.PARTICLE_SPEED_MAX)
            ))

        # Tooltip animation
        self._tooltip_opacity = 0.0
        self._tooltip_target_opacity = 0.0

        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.start(const.ANIMATION_FRAME_MS)

        self.setMinimumSize(const.WIDGET_MIN_WIDTH, const.WIDGET_MIN_HEIGHT)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def _update_ring_colors(self) -> None:
        """Update ring colors to blend colors of satellites on each ring."""
        for ring_idx, ring in enumerate(self._orbit_rings):
            # Find all satellites on this ring
            ring_satellites = [s for s in self._satellites if s.ring_index == ring_idx]

            if ring_satellites:
                # Blend the satellite colors
                total_r, total_g, total_b = 0, 0, 0
                for sat in ring_satellites:
                    total_r += sat.color.red()
                    total_g += sat.color.green()
                    total_b += sat.color.blue()

                n = len(ring_satellites)
                ring.color = QColor(
                    total_r // n,
                    total_g // n,
                    total_b // n,
                    const.RING_BASE_ALPHA
                )

    def _update_animation(self) -> None:
        """Update animation state."""
        self._angle = (self._angle + const.SATELLITE_ORBIT_SPEED) % 360
        self._time += 0.016

        # Smooth camera movement
        self._camera_x += (self._target_camera_x - self._camera_x) * const.CAMERA_SMOOTHING
        self._camera_y += (self._target_camera_y - self._camera_y) * const.CAMERA_SMOOTHING

        # Tooltip fade animation
        self._tooltip_opacity += (self._tooltip_target_opacity - self._tooltip_opacity) * const.TOOLTIP_FADE_SPEED

        # Shooting star logic
        self._next_shooting_star -= 1
        if self._next_shooting_star <= 0:
            start_x = random.uniform(-0.2, 0.8)
            start_y = random.uniform(-0.1, 0.3)
            angle = random.uniform(0.5, 1.2)
            speed = random.uniform(const.SHOOTING_STAR_SPEED_MIN, const.SHOOTING_STAR_SPEED_MAX)
            self._shooting_stars.append((
                start_x, start_y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                1.0,
                random.uniform(const.SHOOTING_STAR_TRAIL_MIN, const.SHOOTING_STAR_TRAIL_MAX)
            ))
            self._next_shooting_star = random.randint(
                const.SHOOTING_STAR_INTERVAL_MIN,
                const.SHOOTING_STAR_INTERVAL_MAX
            )

        # Update shooting stars
        new_shooting = []
        for sx, sy, dx, dy, life, trail in self._shooting_stars:
            life -= const.SHOOTING_STAR_DECAY_RATE
            if life > 0:
                new_shooting.append((sx + dx, sy + dy, dx, dy, life, trail))
        self._shooting_stars = new_shooting

        # Update nebula rotations
        for nebula in self._nebulae:
            nebula.rotation += 0.02

        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._hovered_satellite:
                self.satellite_clicked.emit(self._hovered_satellite.name)
                return

            self._is_dragging = True
            self._last_mouse_pos = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse movement for panning and hover detection."""
        self._mouse_pos = QPointF(event.pos())

        if self._is_dragging:
            delta = event.pos() - self._last_mouse_pos
            self._target_camera_x -= delta.x() * const.CAMERA_PAN_SENSITIVITY
            self._target_camera_y -= delta.y() * const.CAMERA_PAN_SENSITIVITY

            # Clamp camera
            self._target_camera_x = max(-const.CAMERA_MAX_OFFSET,
                                        min(const.CAMERA_MAX_OFFSET, self._target_camera_x))
            self._target_camera_y = max(-const.CAMERA_MAX_OFFSET,
                                        min(const.CAMERA_MAX_OFFSET, self._target_camera_y))

            self._last_mouse_pos = event.pos()
        else:
            self._check_hover(event.pos())

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for subtle zoom effect."""
        delta = event.angleDelta().y() / const.ZOOM_SENSITIVITY
        self._zoom = max(const.ZOOM_MIN, min(const.ZOOM_MAX, self._zoom + delta))

    def _check_hover(self, pos: QPoint) -> None:
        """Check if mouse is hovering over interactive elements."""
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2

        planet_radius = min(width, height) * const.PLANET_RADIUS_FACTOR * self._zoom
        orbit_base_radius = min(width, height) * const.ORBIT_BASE_RADIUS_FACTOR * self._zoom

        # Check planet hover
        dx = pos.x() - center_x
        dy = pos.y() - center_y
        dist_to_planet = math.sqrt(dx * dx + dy * dy)

        old_hovered_planet = self._hovered_planet
        old_hovered_satellite = self._hovered_satellite

        self._hovered_planet = dist_to_planet < planet_radius
        self._hovered_satellite = None

        # Check satellite hover
        if not self._hovered_planet:
            for sat in self._satellites:
                x_3d, y_3d, z_3d = self._get_satellite_position(sat, orbit_base_radius)

                sx = center_x + x_3d
                sy = center_y + y_3d

                dist = math.sqrt((pos.x() - sx) ** 2 + (pos.y() - sy) ** 2)
                if dist < const.SATELLITE_HIT_RADIUS:
                    self._hovered_satellite = sat
                    break

        # Update tooltip visibility
        if self._hovered_planet or self._hovered_satellite:
            self._tooltip_target_opacity = 1.0
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self._tooltip_target_opacity = 0.0
            if not self._is_dragging:
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        if old_hovered_planet != self._hovered_planet or old_hovered_satellite != self._hovered_satellite:
            self.update()

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving widget."""
        self._hovered_satellite = None
        self._hovered_planet = False
        self._tooltip_target_opacity = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the interactive space scene."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        width = self.width()
        height = self.height()

        center_x = width / 2
        center_y = height / 2

        # Draw deep space background
        self._draw_deep_space(painter, width, height)

        # Draw nebulae with parallax
        self._draw_nebulae(painter, width, height)

        # Draw star field with parallax layers
        self._draw_stars(painter, width, height)

        # Draw ambient particles
        self._draw_particles(painter, width, height)

        # Draw shooting stars
        self._draw_shooting_stars(painter, width, height)

        # Calculate planet size with zoom
        planet_radius = min(width, height) * const.PLANET_RADIUS_FACTOR * self._zoom
        orbit_base_radius = min(width, height) * const.ORBIT_BASE_RADIUS_FACTOR * self._zoom

        # Draw back half of orbit rings
        self._draw_orbit_rings(painter, center_x, center_y, orbit_base_radius, front=False)

        # Draw satellites behind planet
        self._draw_satellites(painter, center_x, center_y, orbit_base_radius, behind=True)

        # Draw the MOSAIC planet
        self._draw_planet(painter, center_x, center_y, planet_radius)

        # Draw front half of orbit rings
        self._draw_orbit_rings(painter, center_x, center_y, orbit_base_radius, front=True)

        # Draw satellites in front of planet
        self._draw_satellites(painter, center_x, center_y, orbit_base_radius, behind=False)

        # Draw hover tooltip
        if self._tooltip_opacity > 0.05:
            self._draw_tooltip(painter, width, height)

        # Draw bottom text overlay
        self._draw_text_overlay(painter, width / 2, height)

        # Draw interaction hints
        self._draw_hints(painter, width, height)

        painter.end()

    def _draw_deep_space(self, painter: QPainter, width: int, height: int) -> None:
        """Draw deep space gradient background."""
        gradient = QRadialGradient(
            width * (0.3 - self._camera_x * 0.2),
            height * (0.3 - self._camera_y * 0.2),
            max(width, height) * 1.5
        )
        colors = const.DEEP_SPACE_COLORS
        gradient.setColorAt(0, QColor(*colors[0]))
        gradient.setColorAt(0.3, QColor(*colors[1]))
        gradient.setColorAt(0.6, QColor(*colors[2]))
        gradient.setColorAt(1, QColor(*colors[3]))
        painter.fillRect(0, 0, width, height, gradient)

        # Add subtle color variation
        gradient2 = QRadialGradient(
            width * (0.8 + self._camera_x * 0.1),
            height * (0.7 + self._camera_y * 0.1),
            width * 0.6
        )
        gradient2.setColorAt(0, QColor(*const.DEEP_SPACE_ACCENT_COLOR))
        gradient2.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, width, height, gradient2)

    def _draw_nebulae(self, painter: QPainter, width: int, height: int) -> None:
        """Draw distant nebulae with parallax."""
        parallax_factor = 0.1

        for nebula in self._nebulae:
            x = (nebula.x - self._camera_x * parallax_factor) * width
            y = (nebula.y - self._camera_y * parallax_factor) * height
            radius = nebula.size * max(width, height)

            pulse = 1.0 + 0.1 * math.sin(self._time * 0.5 + nebula.pulse_phase)
            radius *= pulse

            painter.save()
            painter.translate(x, y)
            painter.rotate(nebula.rotation + self._time * 2)

            for i in range(3):
                layer_radius = radius * (1.0 - i * 0.2)
                alpha = nebula.color.alpha() - i * 10

                gradient = QRadialGradient(0, 0, layer_radius)
                color = QColor(nebula.color)
                color.setAlpha(alpha)
                gradient.setColorAt(0, color)
                gradient.setColorAt(0.4, QColor(color.red(), color.green(), color.blue(), alpha // 2))
                gradient.setColorAt(1, QColor(0, 0, 0, 0))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(gradient)
                painter.drawEllipse(QPointF(0, 0), layer_radius, layer_radius * 0.6)

            painter.restore()

    def _draw_stars(self, painter: QPainter, width: int, height: int) -> None:
        """Draw stars with parallax layers."""
        painter.setPen(Qt.PenStyle.NoPen)

        parallax_factors = [layer.parallax_factor for layer in const.STAR_LAYERS]

        for star in self._stars:
            px = parallax_factors[star.layer]
            x = (star.x - self._camera_x * px) * width
            y = (star.y - self._camera_y * px) * height

            x = x % (width * 1.5) - width * 0.25
            y = y % (height * 1.5) - height * 0.25

            twinkle = 0.5 + 0.5 * math.sin(self._time * star.twinkle_speed + star.phase)
            size = star.base_size * twinkle * self._zoom

            # Star color based on temperature
            if star.color_temp < const.STAR_COLOR_RED_THRESHOLD:
                color = QColor(*const.STAR_COLOR_RED)
            elif star.color_temp < const.STAR_COLOR_YELLOW_THRESHOLD:
                color = QColor(*const.STAR_COLOR_YELLOW)
            elif star.color_temp < const.STAR_COLOR_WHITE_THRESHOLD:
                color = QColor(*const.STAR_COLOR_WHITE)
            else:
                color = QColor(*const.STAR_COLOR_BLUE)

            alpha = int(120 * twinkle + 80)
            color.setAlpha(alpha)

            if star.base_size > 1.5 and star.layer > 0:
                glow = QRadialGradient(x, y, size * 4)
                glow.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 60))
                glow.setColorAt(0.5, QColor(color.red(), color.green(), color.blue(), 20))
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(x, y), size * 4, size * 4)

            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), size, size)

    def _draw_particles(self, painter: QPainter, width: int, height: int) -> None:
        """Draw floating ambient particles."""
        painter.setPen(Qt.PenStyle.NoPen)

        for px, py, size, phase, speed in self._particles:
            x = (px - self._camera_x * 0.2 + math.sin(self._time * speed + phase) * 0.02) * width
            y = (py - self._camera_y * 0.2 + math.cos(self._time * speed * 0.7 + phase) * 0.02) * height

            alpha = int(30 + 20 * math.sin(self._time * 2 + phase))

            gradient = QRadialGradient(x, y, size * 3)
            gradient.setColorAt(0, QColor(*const.PARTICLE_COLOR, alpha))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))

            painter.setBrush(gradient)
            painter.drawEllipse(QPointF(x, y), size * 3, size * 3)

    def _draw_shooting_stars(self, painter: QPainter, width: int, height: int) -> None:
        """Draw shooting stars with trails."""
        for sx, sy, dx, dy, life, trail_len in self._shooting_stars:
            x = sx * width
            y = sy * height

            ex = x - dx * trail_len * width * life
            ey = y - dy * trail_len * height * life

            gradient = QLinearGradient(x, y, ex, ey)
            gradient.setColorAt(0, QColor(255, 255, 255, int(255 * life)))
            gradient.setColorAt(0.3, QColor(200, 220, 255, int(150 * life)))
            gradient.setColorAt(1, QColor(100, 150, 255, 0))

            pen = QPen(QBrush(gradient), 2 * life)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, y), QPointF(ex, ey))

            painter.setPen(Qt.PenStyle.NoPen)
            head_glow = QRadialGradient(x, y, 6 * life)
            head_glow.setColorAt(0, QColor(255, 255, 255, int(255 * life)))
            head_glow.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(head_glow)
            painter.drawEllipse(QPointF(x, y), 6 * life, 6 * life)

    def _draw_orbit_rings(self, painter: QPainter, cx: float, cy: float,
                          base_radius: float, front: bool = True) -> None:
        """Draw tilted orbit rings with 3D effect."""
        for ring in self._orbit_rings:
            radius = base_radius * ring.radius

            ring_rotation = self._time * ring.rotation_speed * const.RING_ROTATION_MULTIPLIER

            tilt_x_rad = math.radians(ring.tilt_x)
            tilt_z_rad = math.radians(ring.tilt_z + ring_rotation)

            num_segments = const.RING_SEGMENTS
            points_3d = []

            for i in range(num_segments + 1):
                angle = (i / num_segments) * 2 * math.pi

                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0

                y_rot = y * math.cos(tilt_x_rad) - z * math.sin(tilt_x_rad)
                z_rot = y * math.sin(tilt_x_rad) + z * math.cos(tilt_x_rad)
                y, z = y_rot, z_rot

                x_rot = x * math.cos(tilt_z_rad) - y * math.sin(tilt_z_rad)
                y_rot = x * math.sin(tilt_z_rad) + y * math.cos(tilt_z_rad)
                x, y = x_rot, y_rot

                points_3d.append((x, y, z))

            for i in range(num_segments):
                z1 = points_3d[i][2]
                z2 = points_3d[i + 1][2]
                avg_z = (z1 + z2) / 2

                is_front = avg_z > 0

                if is_front != front:
                    continue

                x1, y1 = points_3d[i][0], points_3d[i][1]
                x2, y2 = points_3d[i + 1][0], points_3d[i + 1][1]

                depth_factor = 0.5 + 0.5 * (avg_z / radius) if front else 0.5 - 0.5 * (avg_z / radius)
                alpha = int(ring.color.alpha() * (0.4 + 0.6 * depth_factor))

                for j in range(const.RING_GLOW_LAYERS):
                    pen_alpha = max(5, alpha - j * 15)
                    pen = QPen(QColor(ring.color.red(), ring.color.green(), ring.color.blue(), pen_alpha))
                    pen.setWidth(4 - j)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawLine(
                        QPointF(cx + x1, cy + y1),
                        QPointF(cx + x2, cy + y2)
                    )

    def _draw_planet(self, painter: QPainter, cx: float, cy: float, radius: float) -> None:
        """Draw the MOSAIC planet with atmosphere and interaction highlight."""
        # Atmosphere glow layers
        for i in range(const.ATMOSPHERE_GLOW_LAYERS):
            glow_radius = radius * (1.4 + i * 0.12)
            alpha = 35 - i * 5
            if self._hovered_planet:
                alpha = int(alpha * 1.5)

            glow = QRadialGradient(cx, cy, glow_radius)
            glow.setColorAt(0.6, QColor(*const.ATMOSPHERE_COLOR, alpha))
            glow.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

        # Planet base
        light_x = cx - radius * 0.4
        light_y = cy - radius * 0.4
        planet_gradient = QRadialGradient(light_x, light_y, radius * 2.5)

        colors = const.PLANET_COLORS_HOVERED if self._hovered_planet else const.PLANET_COLORS_NORMAL
        planet_gradient.setColorAt(0, QColor(*colors[0]))
        planet_gradient.setColorAt(0.3, QColor(*colors[1]))
        planet_gradient.setColorAt(0.6, QColor(*colors[2]))
        planet_gradient.setColorAt(1, QColor(*colors[3]))

        painter.setBrush(planet_gradient)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Surface details with clipping
        painter.setClipPath(self._create_circle_path(cx, cy, radius))

        # Animated cloud bands
        for i, offset in enumerate(const.CLOUD_BAND_OFFSETS):
            band_y = cy + radius * offset
            band_width = radius * (1.6 + i * 0.1)
            band_height = radius * (0.12 + (i % 2) * 0.05)
            cloud_alpha = 50 + (i % 3) * 10

            band_x = cx + math.sin(self._time * 0.2 + i * 0.7) * radius * 0.15

            painter.setBrush(QColor(*const.CLOUD_COLOR, cloud_alpha))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(band_x, band_y), band_width, band_height)

        # Surface features
        features = [(0.25, -0.25, 0.3, 0.2), (-0.35, 0.15, 0.25, 0.18),
                    (0.05, 0.45, 0.2, 0.12), (-0.15, -0.4, 0.18, 0.15)]
        for fx, fy, fw, fh in features:
            painter.setBrush(QColor(30, 55, 100, 70))
            painter.drawEllipse(
                QPointF(cx + radius * fx, cy + radius * fy),
                radius * fw, radius * fh
            )

        painter.setClipping(False)

        # Specular highlight
        highlight = QRadialGradient(cx - radius * 0.35, cy - radius * 0.35, radius * 0.6)
        highlight.setColorAt(0, QColor(255, 255, 255, 100 if self._hovered_planet else 70))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(highlight)
        painter.drawEllipse(QPointF(cx - radius * 0.35, cy - radius * 0.35), radius * 0.45, radius * 0.35)

        # MOSAIC text
        font_size = int(radius * 0.26)
        font = QFont("Arial", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        text = const.PLANET_NAME
        text_width = fm.horizontalAdvance(text)

        painter.setPen(QColor(0, 0, 0, 180))
        painter.drawText(int(cx - text_width / 2 + 2), int(cy + fm.height() / 4 + 2), text)

        painter.setPen(QColor(255, 255, 255, 250))
        painter.drawText(int(cx - text_width / 2), int(cy + fm.height() / 4), text)

    def _create_circle_path(self, cx: float, cy: float, radius: float) -> QPainterPath:
        """Create a circular clipping path."""
        path = QPainterPath()
        path.addEllipse(QPointF(cx, cy), radius, radius)
        return path

    def _get_satellite_position(self, sat: Satellite, base_radius: float) -> Tuple[float, float, float]:
        """Calculate 3D position of a satellite on its tilted ring."""
        ring = self._orbit_rings[sat.ring_index]
        radius = base_radius * ring.radius * sat.orbit_radius

        ring_rotation = self._time * ring.rotation_speed * const.RING_ROTATION_MULTIPLIER

        angle = math.radians(self._angle * sat.speed + sat.phase)

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.0

        tilt_x_rad = math.radians(ring.tilt_x)
        y_rot = y * math.cos(tilt_x_rad) - z * math.sin(tilt_x_rad)
        z_rot = y * math.sin(tilt_x_rad) + z * math.cos(tilt_x_rad)
        y, z = y_rot, z_rot

        tilt_z_rad = math.radians(ring.tilt_z + ring_rotation)
        x_rot = x * math.cos(tilt_z_rad) - y * math.sin(tilt_z_rad)
        y_rot = x * math.sin(tilt_z_rad) + y * math.cos(tilt_z_rad)
        x, y = x_rot, y_rot

        return x, y, z

    def _draw_satellites(self, painter: QPainter, cx: float, cy: float,
                         base_radius: float, behind: bool) -> None:
        """Draw orbiting satellites on tilted rings with hover effects."""
        for sat in self._satellites:
            x_3d, y_3d, z_3d = self._get_satellite_position(sat, base_radius)

            x = cx + x_3d
            y = cy + y_3d

            is_behind = z_3d < 0
            if is_behind != behind:
                continue

            max_z = base_radius
            depth_factor = 0.5 + 0.5 * (z_3d / max_z)
            depth_factor = max(0.3, min(1.0, depth_factor))

            is_hovered = sat == self._hovered_satellite

            base_sat_size = const.SATELLITE_SIZE_HOVERED if is_hovered else const.SATELLITE_SIZE_NORMAL
            sat_size = base_sat_size * depth_factor * self._zoom
            alpha = int(140 + 115 * depth_factor)

            if is_hovered:
                for i in range(4):
                    glow_size = sat_size * (2.5 + i * 0.8)
                    glow_alpha = 60 - i * 15
                    glow = QRadialGradient(x, y, glow_size)
                    glow.setColorAt(0, QColor(sat.color.red(), sat.color.green(), sat.color.blue(), glow_alpha))
                    glow.setColorAt(1, QColor(0, 0, 0, 0))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(glow)
                    painter.drawEllipse(QPointF(x, y), glow_size, glow_size)
            else:
                glow = QRadialGradient(x, y, sat_size * 2.5)
                glow.setColorAt(0, QColor(sat.color.red(), sat.color.green(), sat.color.blue(), alpha // 2))
                glow.setColorAt(1, QColor(0, 0, 0, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(glow)
                painter.drawEllipse(QPointF(x, y), sat_size * 2.5, sat_size * 2.5)

            sat_color = QColor(sat.color)
            sat_color.setAlpha(alpha)
            painter.setBrush(sat_color)
            painter.drawEllipse(QPointF(x, y), sat_size, sat_size)

            inner_highlight = QRadialGradient(x - sat_size * 0.3, y - sat_size * 0.3, sat_size)
            inner_highlight.setColorAt(0, QColor(255, 255, 255, 100))
            inner_highlight.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(inner_highlight)
            painter.drawEllipse(QPointF(x, y), sat_size * 0.8, sat_size * 0.8)

            if not behind and depth_factor > 0.5:
                font = QFont("Arial", 10 if is_hovered else 9)
                font.setBold(is_hovered)
                painter.setFont(font)

                if is_hovered:
                    painter.setPen(QColor(sat.color.red(), sat.color.green(), sat.color.blue(), 150))
                else:
                    painter.setPen(QColor(220, 230, 255, alpha))

                fm = QFontMetrics(font)
                name_width = fm.horizontalAdvance(sat.name)
                painter.drawText(int(x - name_width / 2), int(y + sat_size + 18), sat.name)

    def _draw_tooltip(self, painter: QPainter, width: int, height: int) -> None:
        """Draw tooltip for hovered element."""
        if self._tooltip_opacity < 0.05:
            return

        if self._hovered_satellite:
            title = self._hovered_satellite.name
            description = self._hovered_satellite.description
            features = self._hovered_satellite.features
            accent_color = self._hovered_satellite.color
        elif self._hovered_planet:
            title = const.PLANET_NAME
            description = self._planet_description
            features = self._planet_features
            accent_color = QColor(*const.ATMOSPHERE_COLOR)
        else:
            return

        tooltip_width = const.TOOLTIP_WIDTH
        tooltip_height = 120 + len(features) * const.TOOLTIP_FEATURE_LINE_HEIGHT

        tx = self._mouse_pos.x() + const.TOOLTIP_OFFSET_X
        ty = self._mouse_pos.y() - tooltip_height // 2

        if tx + tooltip_width > width - const.TOOLTIP_MARGIN:
            tx = self._mouse_pos.x() - tooltip_width - const.TOOLTIP_OFFSET_X
        if ty < const.TOOLTIP_MARGIN:
            ty = const.TOOLTIP_MARGIN
        if ty + tooltip_height > height - const.TOOLTIP_MARGIN:
            ty = height - tooltip_height - const.TOOLTIP_MARGIN

        alpha = int(230 * self._tooltip_opacity)

        bg_rect = QRectF(tx, ty, tooltip_width, tooltip_height)

        shadow_color = QColor(*const.TOOLTIP_SHADOW_COLOR[:3], int(const.TOOLTIP_SHADOW_COLOR[3] * self._tooltip_opacity))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow_color)
        painter.drawRoundedRect(bg_rect.adjusted(4, 4, 4, 4), const.TOOLTIP_BORDER_RADIUS, const.TOOLTIP_BORDER_RADIUS)

        bg_gradient = QLinearGradient(tx, ty, tx, ty + tooltip_height)
        bg_gradient.setColorAt(0, QColor(*const.TOOLTIP_BG_TOP, alpha))
        bg_gradient.setColorAt(1, QColor(*const.TOOLTIP_BG_BOTTOM, alpha))
        painter.setBrush(bg_gradient)

        border_color = QColor(accent_color)
        border_color.setAlpha(int(150 * self._tooltip_opacity))
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(bg_rect, const.TOOLTIP_BORDER_RADIUS, const.TOOLTIP_BORDER_RADIUS)

        accent_rect = QRectF(tx + 3, ty + 3, 4, tooltip_height - 6)
        painter.setPen(Qt.PenStyle.NoPen)
        accent_with_alpha = QColor(accent_color)
        accent_with_alpha.setAlpha(int(200 * self._tooltip_opacity))
        painter.setBrush(accent_with_alpha)
        painter.drawRoundedRect(accent_rect, 2, 2)

        title_font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255, int(255 * self._tooltip_opacity)))
        painter.drawText(int(tx + 18), int(ty + 28), title)

        desc_font = QFont("Arial", 10)
        painter.setFont(desc_font)
        painter.setPen(QColor(180, 190, 210, int(220 * self._tooltip_opacity)))
        painter.drawText(int(tx + 18), int(ty + 50), description)

        feature_font = QFont("Arial", 9)
        painter.setFont(feature_font)

        for i, feature in enumerate(features):
            bullet_color = QColor(accent_color)
            bullet_color.setAlpha(int(180 * self._tooltip_opacity))
            painter.setBrush(bullet_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(tx + 22), int(ty + 68 + i * const.TOOLTIP_FEATURE_LINE_HEIGHT), 5, 5)

            painter.setPen(QColor(160, 175, 195, int(200 * self._tooltip_opacity)))
            painter.drawText(int(tx + 34), int(ty + 75 + i * const.TOOLTIP_FEATURE_LINE_HEIGHT), feature)

    def _draw_text_overlay(self, painter: QPainter, cx: float, height: int) -> None:
        """Draw title and instructions."""
        font = QFont("Arial", 12)
        painter.setFont(font)
        painter.setPen(QColor(160, 180, 210, 200))

        text = const.SUBTITLE_TEXT
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)

        y = height - 40
        painter.drawText(int(cx - text_width / 2), y, text)

        font2 = QFont("Arial", 10)
        painter.setFont(font2)
        painter.setPen(QColor(120, 140, 170, 160))

        instruction = const.INSTRUCTION_TEXT
        fm2 = QFontMetrics(font2)
        instr_width = fm2.horizontalAdvance(instruction)
        painter.drawText(int(cx - instr_width / 2), y + 20, instruction)

    def _draw_hints(self, painter: QPainter, width: int, height: int) -> None:
        """Draw interaction hints."""
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.setPen(QColor(100, 120, 150, 120))

        for i, hint in enumerate(const.HINT_TEXTS):
            painter.drawText(15, 20 + i * 16, hint)

    def start_animation(self) -> None:
        """Start the animation."""
        if not self._timer.isActive():
            self._timer.start(const.ANIMATION_FRAME_MS)

    def stop_animation(self) -> None:
        """Stop the animation."""
        self._timer.stop()

    def showEvent(self, event) -> None:
        """Start animation when visible."""
        super().showEvent(event)
        self.start_animation()

    def hideEvent(self, event) -> None:
        """Stop animation when hidden."""
        super().hideEvent(event)
        self.stop_animation()

    def reset_camera(self) -> None:
        """Reset camera to default position."""
        self._target_camera_x = 0.0
        self._target_camera_y = 0.0
        self._zoom = 1.0


__all__ = ["MosaicWelcomeWidget"]
