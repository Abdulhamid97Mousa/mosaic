"""Helpers for creating analytics tabs (TensorBoard, W&B) in the main window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from gym_gui.config.paths import VAR_ROOT
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
)
from gym_gui.ui.widgets.tensorboard_artifact_tab import TensorboardArtifactTab
from gym_gui.ui.widgets.wandb_artifact_tab import WandbArtifactTab


class AnalyticsTabManager(LogConstantMixin):
    """Create analytics tabs (TensorBoard, W&B) for completed runs."""

    def __init__(self, render_tabs, parent) -> None:
        self._logger = logging.getLogger(__name__)
        self._render_tabs = render_tabs
        self._parent = parent

    # ------------------------------------------------------------------
    def ensure_tensorboard_tab(self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]) -> None:
        """Create or refresh the TensorBoard tab if metadata provides a log directory."""
        if not metadata:
            return

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            return

        tensorboard_meta = artifacts.get("tensorboard")
        if not isinstance(tensorboard_meta, Mapping):
            return

        if tensorboard_meta.get("enabled", True) is False:
            return

        resolved_path: Optional[Path] = None
        log_dir = tensorboard_meta.get("log_dir")
        relative_path = tensorboard_meta.get("relative_path")
        if isinstance(log_dir, str) and log_dir.strip():
            resolved_path = Path(log_dir).expanduser()
        elif isinstance(relative_path, str) and relative_path.strip():
            resolved_path = (VAR_ROOT.parent / relative_path).resolve()

        if resolved_path is None:
            self.log_constant(
                LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return

        tab_name = f"TensorBoard-Agent-{agent_id}"
        existing_tabs = self._render_tabs._agent_tabs.get(run_id, {})
        existing_widget = existing_tabs.get(tab_name)
        if existing_widget is not None:
            setter = getattr(existing_widget, "set_log_dir", None)
            if callable(setter):
                setter(resolved_path)
            refresher = getattr(existing_widget, "refresh", None)
            if callable(refresher):
                refresher()
            return

        tab = TensorboardArtifactTab(run_id, agent_id, resolved_path, parent=self._parent)
        self._render_tabs.add_dynamic_tab(run_id, tab_name, tab)
        self.log_constant(
            LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "log_dir": str(resolved_path),
            },
        )

    def ensure_wandb_tab(self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]) -> None:
        """Create or refresh the W&B tab when metadata includes a run path."""
        if not metadata:
            return

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            return

        wandb_meta = artifacts.get("wandb")
        if not isinstance(wandb_meta, Mapping):
            return

        run_path = wandb_meta.get("run_path")
        if not isinstance(run_path, str) or not run_path.strip():
            return

        tab_name = f"WAB-Agent-{agent_id}"
        existing_tabs = self._render_tabs._agent_tabs.get(run_id, {})
        existing_widget = existing_tabs.get(tab_name)
        if existing_widget is not None:
            setter = getattr(existing_widget, "set_run_path", None)
            if callable(setter):
                setter(run_path)
                refresher = getattr(existing_widget, "refresh", None)
                if callable(refresher):
                    refresher()
            return

        try:
            tab = WandbArtifactTab(run_id, agent_id, run_path, parent=self._parent)
        except Exception as exc:  # noqa: BLE001
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_ERROR,
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "error": str(exc),
                },
                exc_info=exc,
            )
            return

        self._render_tabs.add_dynamic_tab(run_id, tab_name, tab)
        self.log_constant(
            LOG_UI_RENDER_TABS_WANDB_STATUS,
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "run_path": run_path,
            },
        )


__all__ = ["AnalyticsTabManager"]
