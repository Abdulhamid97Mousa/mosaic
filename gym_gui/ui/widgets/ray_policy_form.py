"""Dialog for loading trained Ray RLlib policies.

This is a thin wrapper around LoadPolicyDialog that:
1. Filters to show only Ray RLlib checkpoints
2. Converts the selected checkpoint to the expected config format
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from qtpy import QtWidgets

from gym_gui.ui.forms.factory import get_worker_form_factory
from gym_gui.ui.widgets.load_policy_dialog import LoadPolicyDialog
from gym_gui.policy_discovery.ray_policy_metadata import RayRLlibCheckpoint

_LOGGER = logging.getLogger(__name__)


class RayPolicyForm(QtWidgets.QDialog):
    """Dialog for loading trained Ray RLlib checkpoints.

    This wraps LoadPolicyDialog with a filter for Ray-only checkpoints
    and converts the selection to the expected config format.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_game: Optional[Any] = None,
        current_game: Optional[Any] = None,  # Alias for default_game
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Ray RLlib Policy")

        # Accept either default_game or current_game
        self._default_game = default_game or current_game
        self._result_config: Optional[Dict[str, Any]] = None

        # Create the wrapped dialog with Ray filter
        self._inner_dialog = LoadPolicyDialog(
            parent=self,
            filter_worker="ray",
        )

        # Forward the dialog result
        self._inner_dialog.finished.connect(self._on_inner_finished)

    def _on_inner_finished(self, result: int) -> None:
        """Handle inner dialog completion."""
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            checkpoint = self._inner_dialog.get_selected_checkpoint()
            if checkpoint is not None and isinstance(checkpoint, RayRLlibCheckpoint):
                self._result_config = self._build_config(checkpoint)
                self.accept()
            else:
                self.reject()
        else:
            self.reject()

    def exec(self) -> int:
        """Execute the dialog."""
        # Show the inner dialog instead of this one
        result = self._inner_dialog.exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            checkpoint = self._inner_dialog.get_selected_checkpoint()
            if checkpoint is not None and isinstance(checkpoint, RayRLlibCheckpoint):
                self._result_config = self._build_config(checkpoint)
                return QtWidgets.QDialog.DialogCode.Accepted
        return QtWidgets.QDialog.DialogCode.Rejected

    def _build_config(self, checkpoint: RayRLlibCheckpoint) -> Dict[str, Any]:
        """Build the evaluation configuration from a checkpoint.

        Args:
            checkpoint: The selected Ray RLlib checkpoint

        Returns:
            Configuration dict for evaluation
        """
        # Default to "shared" policy for parameter sharing paradigm
        policy_id = "shared" if checkpoint.paradigm == "parameter_sharing" else "main"
        if checkpoint.policy_ids:
            policy_id = checkpoint.policy_ids[0]

        return {
            "mode": "evaluate",
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "policy_id": policy_id,
            "deterministic": True,
            "metadata": {
                "ui": {
                    "worker_id": "ray_worker",
                    "mode": "evaluate",
                },
                "worker": {
                    "checkpoint_path": str(checkpoint.checkpoint_path),
                    "policy_id": policy_id,
                    "deterministic": True,
                },
                "ray_checkpoint": {
                    "run_id": checkpoint.run_id,
                    "env_id": checkpoint.env_id,
                    "env_family": checkpoint.env_family,
                    "algorithm": checkpoint.algorithm,
                    "paradigm": checkpoint.paradigm,
                    "policy_ids": checkpoint.policy_ids,
                },
            },
        }

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the configuration if dialog was accepted."""
        return self._result_config


# Register policy form with factory at module load
_factory = get_worker_form_factory()
if not _factory.has_policy_form("ray_worker"):
    _factory.register_policy_form(
        "ray_worker",
        lambda parent=None, **kwargs: RayPolicyForm(parent=parent, **kwargs),
    )


__all__ = ["RayPolicyForm"]
