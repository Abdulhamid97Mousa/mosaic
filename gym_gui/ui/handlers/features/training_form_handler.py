"""Training form handler for MainWindow.

Extracts training form dialog handling from MainWindow.
Manages Train, Policy Load, and Resume form dialogs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QDialog, QMessageBox, QWidget

if TYPE_CHECKING:
    from gym_gui.core.enums import GameId
    from gym_gui.ui.forms.factory import WorkerFormFactory

_LOGGER = logging.getLogger(__name__)


class TrainingFormHandler:
    """Handler for training form dialogs.

    Manages:
    - Train Agent dialog
    - Load Trained Policy dialog
    - Resume Training dialog
    - Configuration building from dialogs
    """

    def __init__(
        self,
        parent: QWidget,
        get_form_factory: Callable[[], "WorkerFormFactory"],
        get_current_game: Callable[[], Optional["GameId"]],
        get_cleanrl_env_id: Callable[[], Optional[str]],
        submit_config: Callable[[Dict[str, Any]], None],
        build_policy_config: Callable[[str, Path], Optional[Dict[str, Any]]],
        log_callback: Optional[Callable[..., None]] = None,
        status_callback: Optional[Callable[[str, int], None]] = None,
    ):
        """Initialize the handler.

        Args:
            parent: Parent widget for dialogs.
            get_form_factory: Callback to get the worker form factory.
            get_current_game: Callback to get current selected game.
            get_cleanrl_env_id: Callback to get CleanRL environment ID.
            submit_config: Callback to submit training configuration.
            build_policy_config: Callback to build policy evaluation config.
            log_callback: Optional callback for structured logging.
            status_callback: Callback for status bar messages (message, duration_ms).
        """
        self._parent = parent
        self._get_form_factory = get_form_factory
        self._get_current_game = get_current_game
        self._get_cleanrl_env_id = get_cleanrl_env_id
        self._submit_config = submit_config
        self._build_policy_config = build_policy_config
        self._log = log_callback or (lambda *args, **kwargs: None)
        self._status = status_callback or (lambda msg, dur: None)

        # Store selected policy path for evaluation
        self._selected_policy_path: Optional[Path] = None

    def on_trained_agent_requested(self, worker_id: str) -> None:
        """Handle the 'Load Trained Policy' button for a specific worker."""
        if not worker_id:
            QMessageBox.information(
                self._parent,
                "Worker Required",
                "Select a worker integration before loading a trained policy.",
            )
            return

        factory = self._get_form_factory()
        try:
            dialog = factory.create_policy_form(
                worker_id,
                parent=self._parent,
                current_game=self._get_current_game(),
            )
        except KeyError:
            QMessageBox.warning(
                self._parent,
                "Policy form unavailable",
                f"No policy selection form registered for worker '{worker_id}'.",
            )
            return

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config_builder = getattr(dialog, "get_config", None)
        if callable(config_builder):
            config = config_builder()
            if not config:
                return
            self._log(
                message="CleanRL policy evaluation submitted",
                extra={"worker_id": worker_id},
            )
            self._status("Launching evaluation run...", 5000)
            self._submit_config(config)
            return

        policy_path = getattr(dialog, "selected_path", None)
        if policy_path is None:
            return

        self._selected_policy_path = policy_path
        self._log(
            message="Selected policy for evaluation",
            extra={
                "policy_path": str(policy_path),
                "worker_id": worker_id,
            },
        )

        config = self._build_policy_config(worker_id, policy_path)
        if config is None:
            return

        self._status(f"Launching evaluation run for {policy_path.name}", 5000)
        self._submit_config(config)

    def on_train_agent_requested(self, worker_id: str) -> None:
        """Handle the 'Train Agent' button - opens the training configuration form."""
        if not worker_id:
            QMessageBox.information(
                self._parent,
                "Worker Required",
                "Select a worker integration before configuring training.",
            )
            return

        factory = self._get_form_factory()
        form_kwargs: Dict[str, Any] = {
            "parent": self._parent,
            "default_game": self._get_current_game(),
        }
        if worker_id == "cleanrl_worker":
            env_id = self._get_cleanrl_env_id()
            if env_id:
                form_kwargs["default_env_id"] = env_id

        try:
            dialog = factory.create_train_form(worker_id, **form_kwargs)
        except KeyError:
            QMessageBox.warning(
                self._parent,
                "Train form unavailable",
                f"No train form registered for worker '{worker_id}'.",
            )
            return

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        get_config = getattr(dialog, "get_config", None)
        if not callable(get_config):
            QMessageBox.warning(
                self._parent,
                "Unsupported Form",
                "Selected worker form does not provide a configuration payload.",
            )
            return

        config = get_config()
        if config is None:
            return
        if not isinstance(config, dict):
            QMessageBox.warning(
                self._parent,
                "Invalid Configuration",
                "Worker form returned an unexpected payload. Expected a dictionary.",
            )
            return

        self._log(
            message="Agent training configuration submitted from dialog",
            extra={"worker_id": worker_id},
        )
        self._status("Launching training run...", 5000)
        self._submit_config(config)

    def on_resume_training_requested(self, worker_id: str) -> None:
        """Handle the 'Resume Training' button - loads checkpoint and continues training."""
        if not worker_id:
            QMessageBox.information(
                self._parent,
                "Worker Required",
                "Select a worker integration before resuming training.",
            )
            return

        factory = self._get_form_factory()
        try:
            dialog = factory.create_resume_form(
                worker_id,
                parent=self._parent,
                current_game=self._get_current_game(),
            )
        except KeyError:
            QMessageBox.warning(
                self._parent,
                "Resume form unavailable",
                f"Resume training for '{worker_id}' is not yet implemented.",
            )
            return

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        get_config = getattr(dialog, "get_config", None)
        if not callable(get_config):
            QMessageBox.warning(
                self._parent,
                "Unsupported Form",
                "Selected worker form does not provide a configuration payload.",
            )
            return

        config = get_config()
        if config is None:
            return
        if not isinstance(config, dict):
            QMessageBox.warning(
                self._parent,
                "Invalid Configuration",
                "Worker form returned an unexpected payload. Expected a dictionary.",
            )
            return

        self._log(
            message="Resume training configuration submitted from dialog",
            extra={"worker_id": worker_id},
        )
        self._status("Resuming training run...", 5000)
        self._submit_config(config)
