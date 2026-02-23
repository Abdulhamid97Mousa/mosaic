"""SMAC/SMACv2 environment loader for mouse-driven camera panning.

Enables click-and-drag camera panning on the 3D rendered SC2 view.
Reuses the same mouse capture infrastructure built for ViZDoom FPS control:
click to capture mouse, drag to pan, ESC to release.

The mouse delta (in pixels) is converted to world-coordinate offsets using
the adapter's ``camera_width`` property and sent via ``adapter.move_camera()``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.controllers.session import SessionController

from gym_gui.core.enums import EnvironmentFamily, ENVIRONMENT_FAMILY_BY_GAME

_LOG = logging.getLogger(__name__)


class SmacCameraLoader:
    """Loader for SMAC/SMACv2 mouse-driven 3D camera panning.

    Configures the render widget's mouse capture system so that click-drag
    gestures translate into SC2 ``ActionRaw.camera_move`` commands.

    Args:
        render_tabs: The render tabs widget for configuring mouse capture.
    """

    def __init__(self, render_tabs: "RenderTabs") -> None:
        self._render_tabs = render_tabs

    def configure_mouse_capture(
        self,
        session: "SessionController",
        delta_scale: float = 0.15,
    ) -> bool:
        """Configure mouse panning for SMAC/SMACv2 3D-rendered games.

        Click on the Video tab to capture the mouse.  Drag to pan the camera.
        ESC or focus-loss releases the capture.

        Args:
            session: The session controller with the current game.
            delta_scale: Sensitivity multiplier (world units per pixel,
                         default 0.15 gives comfortable panning speed).

        Returns:
            True if mouse capture was enabled (SMAC/SMACv2 game with 3D
            renderer), False otherwise.
        """
        game_id = session.game_id
        if game_id is None:
            return False

        # Only activate for SMAC/SMACv2 families
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)
        if family not in (EnvironmentFamily.SMAC, EnvironmentFamily.SMACV2):
            return False

        adapter = session._adapter
        if adapter is None or not hasattr(adapter, "move_camera"):
            return False

        # Only for 3D renderer
        config = getattr(adapter, "_config", None)
        if config is not None and getattr(config, "renderer", "3d") != "3d":
            return False

        # Get camera width for pixel-to-world conversion
        cam_width = getattr(adapter, "camera_width", 24.0)

        # Capture render_tabs for the closure so we can push frames immediately
        render_tabs = self._render_tabs

        def mouse_delta_callback(delta_x: float, delta_y: float) -> None:
            """Convert pixel-based mouse delta to world-unit camera pan.

            After moving the camera, immediately re-render and push the
            new frame to the display so the user sees instant visual feedback.
            """
            if adapter is None or not hasattr(adapter, "move_camera"):
                return
            # delta_x/y arrive as degrees (pixel * delta_scale from _RgbView).
            # We repurpose them as direct world-unit offsets:
            #   drag right -> camera pans right  (positive world X)
            #   drag down  -> camera pans down   (negative world Y, since SC2 Y is up)
            adapter.move_camera(delta_x, -delta_y)

            # Re-render with the updated camera position and push to display
            try:
                render_data = adapter.render()
                if render_data is not None:
                    render_tabs.display_payload(render_data)
            except Exception:
                pass

        def scroll_zoom_callback(direction: int) -> None:
            """Scroll wheel -> software zoom in/out."""
            if adapter is None or not hasattr(adapter, "zoom_camera"):
                return
            adapter.zoom_camera(direction)
            # Re-render with new zoom level and push to display
            try:
                render_data = adapter.render()
                if render_data is not None:
                    render_tabs.display_payload(render_data)
            except Exception:
                pass

        self._render_tabs.configure_mouse_capture(
            enabled=True,
            delta_callback=mouse_delta_callback,
            delta_scale=delta_scale,
        )
        self._render_tabs.set_scroll_callback(scroll_zoom_callback)
        _LOG.debug("SMAC camera panning + zoom enabled for %s (width=%.1f)", game_id, cam_width)
        return True

    def disable_mouse_capture(self) -> None:
        """Disable mouse capture and scroll zoom."""
        self._render_tabs.configure_mouse_capture(enabled=False)
        self._render_tabs.set_scroll_callback(None)


__all__ = ["SmacCameraLoader"]
