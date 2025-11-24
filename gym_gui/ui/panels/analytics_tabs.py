"""Helpers for creating analytics tabs (TensorBoard, WANDB) in the main window."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
import yaml

from qtpy import QtCore

from gym_gui.config.paths import VAR_ROOT, VAR_TRAINER_DIR
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_ARTIFACTS_MISSING,
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
)
from gym_gui.ui.widgets.tensorboard_artifact_tab import TensorboardArtifactTab
from gym_gui.ui.widgets.wandb_artifact_tab import WandbArtifactTab


class AnalyticsTabManager(LogConstantMixin):
    """Create analytics tabs (TensorBoard, WANDB) for completed runs."""

    def __init__(self, render_tabs, parent) -> None:
        self._logger = logging.getLogger(__name__)
        self._render_tabs = render_tabs
        self._parent = parent
        self._wandb_probes: dict[tuple[str, str], dict[str, Any]] = {}

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
        run_dir = analytics_file.parent
        if not run_dir.exists():
            config_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
            repo_root = VAR_ROOT.parent
            third_party = repo_root / "3rd_party"
            hint_cli = (
                f"PYTHONPATH=\"{third_party}:$PYTHONPATH\" python -m cleanrl_worker.cli --config {config_path} "
                "--grpc --grpc-target 127.0.0.1:50055"
            )
            self.log_constant(
                LOG_UI_RENDER_TABS_ARTIFACTS_MISSING,
                message="Run artifacts directory missing; worker appears not to have launched",
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "run_dir": str(run_dir),
                    "config_path": str(config_path),
                    "hint_cli": hint_cli,
                },
            )
            return
        if not analytics_file.exists():
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_WARNING,
                message="Analytics manifest not yet available",
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "attempt": attempt + 1,
                    "max_attempts": max_retries + 1,
                    "path": str(analytics_file),
                },
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
        tensorboard_ready = self.ensure_tensorboard_tab(run_id, agent_id, analytics_data)
        wandb_ready = self.ensure_wandb_tab(run_id, agent_id, analytics_data)

        self.log_constant(
            LOG_UI_RENDER_TABS_TENSORBOARD_STATUS if tensorboard_ready else LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
            message="TensorBoard analytics readiness",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "ready": tensorboard_ready,
                "attempt": attempt + 1,
                "max_attempts": max_retries + 1,
            },
        )
        self.log_constant(
            LOG_UI_RENDER_TABS_WANDB_STATUS if wandb_ready else LOG_UI_RENDER_TABS_WANDB_WARNING,
            message="WANDB analytics readiness",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "ready": wandb_ready,
                "attempt": attempt + 1,
                "max_attempts": max_retries + 1,
            },
        )

        if not wandb_ready:
            self._schedule_retry(run_id, agent_id, attempt, max_retries, retry_delay_ms)

        elif not tensorboard_ready:
            # TensorBoard path missing but WANDB ready; continue retrying for TensorBoard only.
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

        self.log_constant(
            LOG_UI_RENDER_TABS_WANDB_WARNING,
            message="Scheduling analytics tab retry",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "next_attempt": attempt + 2,
                "max_attempts": max_retries + 1,
                "delay_ms": retry_delay_ms,
            },
        )

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
    def _schedule_wandb_probe(
        self,
        run_id: str,
        agent_id: str,
        metadata: Mapping[str, Any],
        *,
        interval_ms: int = 500,
        max_attempts: int = 600,
    ) -> None:
        """Poll for wandb artifacts while the run is still active."""

        key = (run_id, agent_id)
        state = self._wandb_probes.get(key)
        if state is not None:
            state["metadata"] = metadata
            return

        timer = QtCore.QTimer(self._parent)
        timer.setSingleShot(True)

        self._wandb_probes[key] = {
            "metadata": metadata,
            "interval": max(100, interval_ms),
            "max_attempts": max_attempts,
            "attempts": 0,
            "timer": timer,
        }

        def _on_timeout(run_id: str = run_id, agent_id: str = agent_id) -> None:
            self._handle_wandb_probe(run_id, agent_id)

        timer.timeout.connect(_on_timeout)
        timer.start(interval_ms)

    # ------------------------------------------------------------------
    def _handle_wandb_probe(self, run_id: str, agent_id: str) -> None:
        key = (run_id, agent_id)
        state = self._wandb_probes.get(key)
        if not state:
            return

        timer: QtCore.QTimer = state["timer"]
        metadata = state.get("metadata")
        if not isinstance(metadata, Mapping):
            self._clear_wandb_probe(run_id, agent_id)
            return

        state["attempts"] += 1
        ready = self.ensure_wandb_tab(run_id, agent_id, metadata)
        if ready:
            self._clear_wandb_probe(run_id, agent_id)
            return

        if state["attempts"] >= state.get("max_attempts", 600):
            self._clear_wandb_probe(run_id, agent_id)
            return

        timer.start(state.get("interval", 500))

    # ------------------------------------------------------------------
    def _clear_wandb_probe(self, run_id: str, agent_id: str) -> None:
        key = (run_id, agent_id)
        state = self._wandb_probes.pop(key, None)
        if not state:
            return

        timer = state.get("timer")
        if isinstance(timer, QtCore.QTimer):
            try:
                timer.stop()
            except RuntimeError:
                pass
            timer.deleteLater()

    # ------------------------------------------------------------------
    def ensure_tensorboard_tab(
        self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]
    ) -> bool:
        """Create or refresh the TensorBoard tab if metadata provides a log directory."""
        if not metadata:
            self.log_constant(
                LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
                message="TensorBoard metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            self.log_constant(
                LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
                message="TensorBoard artifacts missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        tensorboard_meta = artifacts.get("tensorboard")
        if not isinstance(tensorboard_meta, Mapping):
            self.log_constant(
                LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
                message="TensorBoard metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        if tensorboard_meta.get("enabled", True) is False:
            self.log_constant(
                LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
                message="TensorBoard disabled",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
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
            self._logger.debug(
                "TensorBoard log dir not ready yet: run=%s agent=%s meta=%s",
                run_id,
                agent_id,
                tensorboard_meta,
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
            self._logger.debug(
                "TensorBoard tab refreshed for run=%s agent=%s path=%s",
                run_id,
                agent_id,
                resolved_path,
            )
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
        self._logger.info(
            "Created TensorBoard tab: run=%s agent=%s path=%s",
            run_id,
            agent_id,
            resolved_path,
        )
        return True

    def ensure_wandb_tab(
        self, run_id: str, agent_id: str, metadata: Optional[Mapping[str, Any]]
    ) -> bool:
        """Create or refresh the WANDB tab when metadata includes a run path."""
        if not metadata:
            self._clear_wandb_probe(run_id, agent_id)
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_WARNING,
                message="WANDB metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        artifacts = metadata.get("artifacts") if isinstance(metadata, Mapping) else None
        if not isinstance(artifacts, Mapping):
            self._clear_wandb_probe(run_id, agent_id)
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_WARNING,
                message="WANDB artifacts missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        wandb_meta = artifacts.get("wandb")
        if not isinstance(wandb_meta, Mapping):
            self._clear_wandb_probe(run_id, agent_id)
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_WARNING,
                message="WANDB metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        if wandb_meta.get("enabled", True) is False:
            self._clear_wandb_probe(run_id, agent_id)
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_WARNING,
                message="WANDB disabled for run",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return False

        run_path = wandb_meta.get("run_path")
        manifest_file = wandb_meta.get("manifest_file") if isinstance(wandb_meta.get("manifest_file"), str) else None
        entity_hint = wandb_meta.get("entity") if isinstance(wandb_meta.get("entity"), str) else None
        project_hint = wandb_meta.get("project") if isinstance(wandb_meta.get("project"), str) else None

        run_path, resolved_entity, resolved_project = self._resolve_wandb_run_path(
            run_id=run_id,
            agent_id=agent_id,
            metadata=metadata,
            manifest_file=manifest_file,
            current_run_path=run_path if isinstance(run_path, str) else "",
            entity_hint=entity_hint,
            project_hint=project_hint,
        )

        use_vpn_proxy, http_proxy, https_proxy = self._resolve_wandb_proxy_settings(metadata, wandb_meta)

        tab_name = f"WANDB-Agent-{agent_id}"
        metadata_mapping = metadata if isinstance(metadata, Mapping) else None
        existing_tabs = self._render_tabs._agent_tabs.get(run_id, {})
        existing_widget = existing_tabs.get(tab_name)
        if existing_widget is not None:
            identity_setter = getattr(existing_widget, "set_wandb_identity", None)
            if callable(identity_setter):
                identity_setter(resolved_entity, resolved_project)
            proxy_setter = getattr(existing_widget, "set_proxy_settings", None)
            if callable(proxy_setter):
                proxy_setter(use_vpn_proxy, http_proxy, https_proxy)
            setter = getattr(existing_widget, "set_run_path", None)
            if callable(setter) and run_path:
                setter(run_path)
                refresher = getattr(existing_widget, "refresh", None)
                if callable(refresher):
                    refresher()
                self._logger.debug(
                    "WANDB tab refreshed for run=%s agent=%s with run_path=%s",
                    run_id,
                    agent_id,
                    run_path,
                )
                self._clear_wandb_probe(run_id, agent_id)
                return True
            if metadata_mapping is not None:
                self._schedule_wandb_probe(run_id, agent_id, metadata_mapping)
            return False

        try:
            status_message = None if run_path else "Waiting for WANDB run metadata to become available..."
            tab = WandbArtifactTab(run_id, agent_id, run_path or None, parent=self._parent, status_message=status_message)
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
        tab.set_wandb_identity(resolved_entity, resolved_project)
        proxy_setter = getattr(tab, "set_proxy_settings", None)
        if callable(proxy_setter):
            proxy_setter(use_vpn_proxy, http_proxy, https_proxy)
        if run_path:
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_STATUS,
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "run_path": run_path,
                    "entity": resolved_entity,
                    "project": resolved_project,
                },
            )
            self._logger.info(
                "Created WANDB tab: run=%s agent=%s run_path=%s",
                run_id,
                agent_id,
                run_path,
            )
            self._clear_wandb_probe(run_id, agent_id)
            return True

        self.log_constant(
            LOG_UI_RENDER_TABS_WANDB_WARNING,
            message="Created placeholder WANDB tab; awaiting run_path",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "entity": resolved_entity,
                "project": resolved_project,
            },
        )
        if metadata_mapping is not None:
            self._schedule_wandb_probe(run_id, agent_id, metadata_mapping)
        return False

    def _resolve_wandb_run_path(
        self,
        *,
        run_id: str,
        agent_id: str,
        metadata: Optional[Mapping[str, Any]],
        manifest_file: Optional[str],
        current_run_path: str,
        entity_hint: Optional[str],
        project_hint: Optional[str],
    ) -> tuple[str, Optional[str], Optional[str]]:
        run_path = (current_run_path or "").strip()
        entity, project = self._resolve_wandb_identity(metadata, run_id, entity_hint, project_hint)
        if run_path:
            return run_path, entity, project

        if manifest_file:
            try:
                manifest_path = Path(manifest_file)
                if manifest_path.exists():
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    run_path = (manifest_data.get("run_path") or "").strip()
                    entity = entity or manifest_data.get("entity")
                    project = project or manifest_data.get("project")
                    if run_path:
                        self.log_constant(
                            LOG_UI_RENDER_TABS_WANDB_STATUS,
                            message="Read run_path from wandb manifest",
                            extra={"run_id": run_id, "agent_id": agent_id, "run_path": run_path},
                        )
                        return run_path, entity, project
                else:
                    self.log_constant(
                        LOG_UI_RENDER_TABS_WANDB_WARNING,
                        message="WANDB manifest not found yet; awaiting run_path",
                        extra={"run_id": run_id, "agent_id": agent_id, "manifest_file": manifest_file},
                    )
            except Exception as exc:  # pragma: no cover
                self.log_constant(
                    LOG_UI_RENDER_TABS_WANDB_WARNING,
                    message="Failed to read WANDB manifest file",
                    extra={"run_id": run_id, "agent_id": agent_id, "manifest_file": manifest_file, "error": str(exc)},
                    exc_info=exc,
                )

        slug = self._discover_wandb_slug(run_id)
        if slug and entity and project:
            run_path = f"{entity}/{project}/runs/{slug}"
            self.log_constant(
                LOG_UI_RENDER_TABS_WANDB_STATUS,
                message="Discovered run_path from WANDB files",
                extra={"run_id": run_id, "agent_id": agent_id, "run_path": run_path},
            )
            return run_path, entity, project

        return "", entity, project

    def _resolve_wandb_identity(
        self,
        metadata: Optional[Mapping[str, Any]],
        run_id: str,
        entity_hint: Optional[str],
        project_hint: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        entity = entity_hint.strip() if isinstance(entity_hint, str) and entity_hint.strip() else None
        project = project_hint.strip() if isinstance(project_hint, str) and project_hint.strip() else None

        def _update_from(extra: Mapping[str, Any]) -> None:
            nonlocal entity, project
            if entity is None:
                candidate = extra.get("wandb_entity") or extra.get("wandb_username")
                if isinstance(candidate, str) and candidate.strip():
                    entity = candidate.strip()
            if project is None:
                candidate = extra.get("wandb_project_name")
                if isinstance(candidate, str) and candidate.strip():
                    project = candidate.strip()

        if isinstance(metadata, Mapping):
            worker_meta = metadata.get("worker")
            if isinstance(worker_meta, Mapping):
                worker_config = worker_meta.get("config")
                if isinstance(worker_config, Mapping):
                    extra = worker_config.get("extra")
                    if isinstance(extra, Mapping):
                        _update_from(extra)
            environment_meta = metadata.get("environment")
            if isinstance(environment_meta, Mapping):
                env_entity = environment_meta.get("WANDB_ENTITY") or environment_meta.get("WANDB_USERNAME")
                env_project = environment_meta.get("WANDB_PROJECT") or environment_meta.get("WANDB_PROJECT_NAME")
                if isinstance(env_entity, str) and env_entity.strip():
                    entity = entity or env_entity.strip()
                if isinstance(env_project, str) and env_project.strip():
                    project = project or env_project.strip()

        if entity and project:
            return entity, project

        config_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
        if config_path.exists():
            try:
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                extra = (
                    config_payload.get("metadata", {})
                    .get("worker", {})
                    .get("config", {})
                    .get("extra", {})
                )
                if isinstance(extra, Mapping):
                    _update_from(extra)
            except Exception:  # pragma: no cover
                pass

        if entity and project:
            return entity, project

        # Final fallback: Try OS environment variables
        import os
        if entity is None:
            entity = (
                os.environ.get("WANDB_ENTITY")
                or os.environ.get("WANDB_ENTITY_NAME")
                or os.environ.get("WANDB_USERNAME")
            )
        if project is None:
            project = (
                os.environ.get("WANDB_PROJECT")
                or os.environ.get("WANDB_PROJECT_NAME")
            )

        return entity, project

    def _resolve_wandb_proxy_settings(
        self,
        metadata: Optional[Mapping[str, Any]],
        wandb_meta: Mapping[str, Any],
    ) -> tuple[bool, Optional[str], Optional[str]]:
        def _normalize(value: Any) -> Optional[str]:
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        use_vpn = bool(wandb_meta.get("use_vpn_proxy"))
        http_proxy = _normalize(wandb_meta.get("http_proxy"))
        https_proxy = _normalize(wandb_meta.get("https_proxy"))

        if isinstance(metadata, Mapping):
            worker_meta = metadata.get("worker")
            if isinstance(worker_meta, Mapping):
                worker_config = worker_meta.get("config")
                if isinstance(worker_config, Mapping):
                    extras = worker_config.get("extra") or worker_config.get("extras")
                    if isinstance(extras, Mapping):
                        if not use_vpn and extras.get("wandb_use_vpn_proxy"):
                            use_vpn = True
                        if http_proxy is None:
                            http_proxy = _normalize(extras.get("wandb_http_proxy"))
                        if https_proxy is None:
                            https_proxy = _normalize(extras.get("wandb_https_proxy"))

        return use_vpn, http_proxy, https_proxy

    def _discover_wandb_slug(self, run_id: str) -> str:
        """Discover WANDB run slug from the wandb directory structure.
        
        WANDB creates directories like: wandb/run-20251105_085314-sc26ki0p/
        We need to extract the slug (sc26ki0p) from the directory name.
        """
        wandb_root = VAR_TRAINER_DIR / "runs" / run_id / "wandb"
        if not wandb_root.exists():
            return ""

        # Look for run-* directories (not .wandb files)
        try:
            wandb_subdirs = list(wandb_root.glob("wandb/run-*"))
            if wandb_subdirs:
                # Sort by modification time, newest first
                candidates = sorted(
                    [p for p in wandb_subdirs if p.is_dir()],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                for candidate in candidates:
                    # Extract slug from directory name: run-20251105_085314-sc26ki0p -> sc26ki0p
                    dir_name = candidate.name
                    if dir_name.startswith("run-") and "-" in dir_name:
                        parts = dir_name.split("-")
                        if len(parts) >= 3:
                            slug = parts[-1]  # Get the last part after the final dash
                            if slug:
                                self._logger.debug(
                                    "Discovered WANDB slug from directory: %s -> %s",
                                    dir_name,
                                    slug,
                                )
                                return slug
        except Exception as exc:
            self._logger.warning(
                "Failed to discover WANDB slug for run %s: %s",
                run_id,
                exc,
                exc_info=exc,
            )
        
        return ""




__all__ = ["AnalyticsTabManager"]
