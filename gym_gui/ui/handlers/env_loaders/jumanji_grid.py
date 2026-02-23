"""Grid-click mouse input for Jumanji environments.

Enables click-to-action input on the RGB render view for grid-based
Jumanji games.  Follows the same callback-wiring pattern as
:class:`VizdoomEnvLoader`.

Supported games:
- **Tetris**: Click a column to place the piece; scroll wheel to rotate.
- **Minesweeper**: Click a cell to reveal it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gymnasium.spaces as spaces

from gym_gui.core.enums import GameId

if TYPE_CHECKING:
    from gym_gui.controllers.session import SessionController
    from gym_gui.ui.widgets.render_tabs import RenderTabs

_LOG = logging.getLogger(__name__)


class JumanjiGridClickLoader:
    """Wire grid-click mouse input for Jumanji environments.

    Instantiated once in MainWindow; ``configure_grid_click`` is called
    each time a new game is loaded (from ``_configure_mouse_capture``).
    """

    # Normalised (top, left, bottom, right) sub-region of the rendered image
    # where the game grid lives.  Empirically measured from the 1000x1000
    # matplotlib output (figsize=10, dpi=100, ConstrainedLayout padding=5%).
    # Stable as long as our frozen Jumanji viewer code stays unchanged.
    _MINESWEEPER_GRID_RECT = (0.168, 0.109, 0.951, 0.892)
    _TETRIS_GRID_RECT = (0.414, 0.249, 0.916, 0.752)

    def __init__(self, render_tabs: "RenderTabs") -> None:
        self._render_tabs = render_tabs
        # Tetris cursor state (rotation adjusted via scroll wheel)
        self._rotation: int = 0

    # ── public API ─────────────────────────────────────────────────────

    def configure_grid_click(self, session: "SessionController") -> bool:
        """Set up grid-click input if the loaded game supports it.

        Returns True if grid-click was enabled, False otherwise.
        """
        game_id = session.game_id
        action_space = session.action_space

        if game_id == GameId.JUMANJI_TETRIS:
            return self._setup_tetris(session, action_space)

        if game_id == GameId.JUMANJI_MINESWEEPER:
            return self._setup_minesweeper(session)

        # Not a grid-click game — clear any previous config
        self.disable_grid_click()
        return False

    def disable_grid_click(self) -> None:
        """Disable grid-click mode."""
        self._render_tabs.configure_grid_click(None, 0, 0)

    # ── private helpers ────────────────────────────────────────────────

    def _setup_tetris(
        self, session: "SessionController", action_space: object
    ) -> bool:
        if not isinstance(action_space, spaces.MultiDiscrete):
            _LOG.warning("Tetris action_space is not MultiDiscrete, skipping grid click")
            self.disable_grid_click()
            return False

        num_cols = int(action_space.nvec[1])
        self._rotation = 0

        def on_click(_row: int, col: int) -> None:
            action = [self._rotation, col]
            _LOG.debug("Tetris mouse placement: rot=%d col=%d", self._rotation, col)
            session.perform_human_action(action, key_label="mouse_click")
            # Reset rotation after placement
            self._rotation = 0
            session.status_message.emit(
                f"Tetris: Placed at col {col} — rotation reset to 0 deg"
            )

        def on_scroll(direction: int) -> None:
            self._rotation = (self._rotation + direction) % 4
            session.status_message.emit(
                f"Tetris rotation: {self._rotation * 90} deg "
                f"[scroll=rotate, click=place]"
            )

        # Tetris grid: rows don't matter for placement, but 10 is the default
        self._render_tabs.configure_grid_click(
            on_click, rows=10, cols=num_cols,
            scroll_callback=on_scroll,
            grid_rect=self._TETRIS_GRID_RECT,
        )
        session.status_message.emit(
            "Tetris: Click column to place, scroll wheel to rotate"
        )
        _LOG.info("Tetris grid-click enabled (num_cols=%d)", num_cols)
        return True

    def _setup_minesweeper(self, session: "SessionController") -> bool:
        adapter = session._adapter  # noqa: SLF001 — matches ViZDoom loader pattern
        if adapter is None:
            self.disable_grid_click()
            return False

        try:
            obs_space = adapter.observation_space
            board_space = obs_space["board"]
            num_rows, num_cols = board_space.shape
        except (KeyError, AttributeError, TypeError):
            _LOG.warning("Cannot determine Minesweeper board dimensions")
            self.disable_grid_click()
            return False

        def on_click(row: int, col: int) -> None:
            action = [row, col]
            _LOG.debug("Minesweeper click: row=%d col=%d", row, col)
            session.perform_human_action(action, key_label="mouse_click")

        self._render_tabs.configure_grid_click(
            on_click, rows=num_rows, cols=num_cols,
            grid_rect=self._MINESWEEPER_GRID_RECT,
        )
        session.status_message.emit("Minesweeper: Click a cell to reveal it")
        _LOG.info(
            "Minesweeper grid-click enabled (%dx%d)", num_rows, num_cols,
        )
        return True


__all__ = ["JumanjiGridClickLoader"]
