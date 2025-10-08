from __future__ import annotations

"""Presenter that coordinates the main window widgets and controllers."""

from dataclasses import dataclass
from typing import Callable, Optional

from qtpy import QtCore

from gym_gui.controllers.human_input import HumanInputController
from gym_gui.controllers.session import SessionController
from gym_gui.ui.widgets.control_panel import ControlPanelWidget


@dataclass(slots=True)
class MainWindowView:
    """Typed view contract the presenter expects from the main window."""

    control_panel: ControlPanelWidget
    status_message_sink: Callable[[str, int | None], None]
    awaiting_label_setter: Callable[[bool], None]
    turn_label_setter: Callable[[str], None]
    render_adapter: Callable[[object], None]
    time_refresher: Callable[[], None]
    game_info_setter: Callable[[str], None] | None = None


class MainWindowPresenter(QtCore.QObject):
    """Owns controllers and mediates between the session and UI widgets."""

    def __init__(
        self,
        session: SessionController,
        human_input: HumanInputController,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._human_input = human_input
        self._view: MainWindowView | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def bind_view(self, view: MainWindowView) -> None:
        """Attach the concrete view implementation to the presenter."""

        self._view = view
        # Listen for UI-driven game changes so presenter can update game info
        try:
            view.control_panel.game_changed.connect(lambda gid: self._handle_game_changed(gid.value if gid is not None else None))
        except Exception:
            # If control_panel signal isn't available for some reason, ignore
            pass
        self._wire_session_signals()

    def unbind_view(self) -> None:
        """Detach the view so the presenter can outlive the widget."""

        self._view = None
        self._session.awaiting_human.disconnect(self._handle_awaiting_human)  # type: ignore[misc]
        self._session.turn_changed.disconnect(self._handle_turn_changed)  # type: ignore[misc]
        self._session.status_message.disconnect(self._handle_status_message)  # type: ignore[misc]
        self._session.step_processed.disconnect(self._handle_step_processed)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------
    def _wire_session_signals(self) -> None:
        if self._view is None:
            return
        self._session.awaiting_human.connect(self._handle_awaiting_human)
        self._session.turn_changed.connect(self._handle_turn_changed)
        self._session.status_message.connect(self._handle_status_message)
        self._session.step_processed.connect(self._handle_step_processed)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _handle_status_message(self, message: str) -> None:
        if self._view is None:
            return
        self._view.status_message_sink(message, 5000)

    def _handle_game_changed(self, game_id: str | None) -> None:
        if self._view is None or not self._view.game_info_setter:
            return
        if game_id is None:
            self._view.game_info_setter("")
        else:
            # Presenter-level minimal description; MainWindow can enrich
            self._view.game_info_setter(f"Selected: {game_id}")

    def _handle_awaiting_human(self, waiting: bool, message: str) -> None:
        if self._view is None:
            return
        self._view.awaiting_label_setter(waiting)
        if message:
            self._view.status_message_sink(message, 5000)
        # Note: Keyboard shortcut management is handled by MainWindow._on_awaiting_human()
        # which is called via awaiting_label_setter - don't duplicate logic here

    def _handle_turn_changed(self, turn: str) -> None:
        if self._view is None:
            return
        self._view.turn_label_setter(turn)

    def _handle_step_processed(self, step: object, _: int) -> None:
        if self._view is None:
            return
        self._view.render_adapter(getattr(step, "render_payload", None))
        self._view.time_refresher()


__all__ = ["MainWindowPresenter", "MainWindowView"]
