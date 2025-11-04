"""Helpers for creating analytics tabs (TensorBoard, W&B) in the main window."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from qtpy import QtCore

from gym_gui.config.paths import VAR_ROOT, VAR_TRAINER_DIR
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
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
    def load_and_create_tabs(
        self,
        run_id: str,
        agent_id: str,
        *,
        attempt: int = 0,
        max_retries: int = 3,
        retry_delay_ms: int = 150,
    ) -> None:
        """Load analytics.json from disk and create/refresh analytics tabs.

        Retries a few times to accommodate workers that flush analytics manifests
        slightly after the trainer signals completion.
        """

        analytics_file = VAR_TRAINER_DIR / "runs" / run_id / "analytics.json"
        if not analytics_file.exists():
            self._logger.debug(
                "Analytics manifest not found for run %s at %s (attempt %s/%s)",
                run_id,
                analytics_file,
                attempt + 1,
                max_retries + 1,
            )
            self._schedule_retry(run_id, agent_id, attempt, max_retries, retry_delay_ms)
            return

        try:
            analytics_data = json.loads(analytics_file.read_text(encoding="utf-8"))
            self._logger.debug(
                "Loaded analytics manifest for run %s from %s",
                run_id,
                analytics_file,
            )
        except Exception as exc:
            self._logger.warning(
                "Failed to load analytics manifest for run %s (attempt %s/%s): %s",
                run_id,
                attempt + 1,
                max_retries + 1,
                exc,
                exc_info=exc,
            )
            self._schedule_retry(run_id, agent_id, attempt, max_retries, retry_delay_ms)
            return

        # Create/refresh tabs with the loaded analytics data
        self.ensure_tensorboard_tab(run_id, agent_id, analytics_data)
        wandb_ready = self.ensure_wandb_tab(run_id, agent_id, analytics_data)

        if not wandb_ready:
            self._schedule_retry(run_id, agent_id, attempt, max_retries, retry_delay_ms)

    # ------------------------------------------------------------------
    def _schedule_retry(
        self,
        run_id: str,
        agent_id: str,
        attempt: int,
        max_retries: int,
        retry_delay_ms: int,
    ) -> None:
        if attempt >= max_retries:
            return

        QtCore.QTimer.singleShot(
            retry_delay_ms,
            lambda: self.load_and_create_tabs(
                run_id,
                agent_id,
                attempt=attempt + 1,
                max_retries=max_retries,
                retry_delay_ms=retry_delay_ms,
            ),
        )

    # ------------------------------------------------------------------
    def ensure_tensorboard_tab(
        self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]
    ) -> bool:
        """Create or refresh the TensorBoard tab if metadata provides a log directory."""
        if not metadata:
            return False

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            return False

        tensorboard_meta = artifacts.get("tensorboard")
        if not isinstance(tensorboard_meta, Mapping):
            return False

        if tensorboard_meta.get("enabled", True) is False:
            return False

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
            return False

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
            return True

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
        return True

    def ensure_wandb_tab(
        self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]
    ) -> bool:
        """Create or refresh the W&B tab when metadata includes a run path."""
        if not metadata:
            return False

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            return False

        wandb_meta = artifacts.get("wandb")
        if not isinstance(wandb_meta, Mapping):
            return False

        run_path = wandb_meta.get("run_path")

        # If run_path is not in metadata, try reading from manifest file (like TensorBoard pattern)
        if not isinstance(run_path, str) or not run_path.strip():
            manifest_file = wandb_meta.get("manifest_file")
            if isinstance(manifest_file, str):
                try:
                    manifest_path = Path(manifest_file)
                    if manifest_path.exists():
                        import json
                        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        run_path = manifest_data.get("run_path", "")
                        self.log_constant(
                            LOG_UI_RENDER_TABS_WANDB_STATUS,
                            message="Read run_path from manifest file",
                            extra={"run_id": run_id, "agent_id": agent_id, "run_path": run_path},
                        )
                    else:
                        self.log_constant(
                            LOG_UI_RENDER_TABS_WANDB_WARNING,
                            message="Manifest file not found, W&B tab will appear when manifest is written",
                            extra={"run_id": run_id, "agent_id": agent_id, "manifest_file": manifest_file},
                        )
                        return False
                except Exception as e:
                    self.log_constant(
                        LOG_UI_RENDER_TABS_WANDB_WARNING,
                        message="Failed to read W&B manifest file",
                        extra={"run_id": run_id, "agent_id": agent_id, "error": str(e)},
                        exc_info=e,
                    )
                    return False

        if not isinstance(run_path, str) or not run_path.strip():
            return False

        tab_name = f"WANDB-Agent-{agent_id}"
        existing_tabs = self._render_tabs._agent_tabs.get(run_id, {})
        existing_widget = existing_tabs.get(tab_name)
        if existing_widget is not None:
            setter = getattr(existing_widget, "set_run_path", None)
            if callable(setter):
                setter(run_path)
                refresher = getattr(existing_widget, "refresh", None)
                if callable(refresher):
                    refresher()
            return True

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
            return False

        self._render_tabs.add_dynamic_tab(run_id, tab_name, tab)
        self.log_constant(
            LOG_UI_RENDER_TABS_WANDB_STATUS,
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "run_path": run_path,
            },
        )
        return True


__all__ = ["AnalyticsTabManager"]
