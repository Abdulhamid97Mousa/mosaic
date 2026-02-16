"""XuanCe Custom Script form for launching bash-based training scripts.

This form is purpose-built for custom script execution -- no algorithm
dropdown, no environment selector, no hyperparameters.  The script controls
all training parameters.  The form only provides:

    1. Script selection (combo + browse)
    2. Script metadata preview (phases, environments, timesteps)
    3. FastLane settings (video mode, grid limit)
    4. Tracking toggles (TensorBoard, W&B)
    5. GPU toggle
    6. Backend selection (PyTorch / TensorFlow / MindSpore)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFileDialog

from ulid import ULID

from gym_gui.config.paths import XUANCE_SCRIPTS_DIR, VAR_CUSTOM_SCRIPTS_DIR
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
)
from gym_gui.telemetry.semconv import VideoModes

_LOGGER = logging.getLogger("gym_gui.ui.xuance_script_form")


def _generate_run_id(prefix: str, slug: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = slug.replace("_", "-")
    return f"{prefix}-{slug}-{timestamp}"


# ---------------------------------------------------------------------------
# Script metadata parsing
# ---------------------------------------------------------------------------

def _parse_script_metadata(script_path: Path) -> str:
    """Parse @description metadata from a script file."""
    try:
        content = script_path.read_text(encoding="utf-8")
        for line in content.split("\n")[:30]:
            if "@description:" in line:
                return line.split("@description:")[-1].strip()
        return ""
    except Exception:
        return ""


def _parse_script_full_metadata(script_path: Path) -> Dict[str, Any]:
    """Parse all metadata from a script file including environments."""
    metadata: Dict[str, Any] = {
        "description": "",
        "env_family": None,
        "phases": None,
        "total_timesteps": None,
        "environments": [],
    }
    try:
        content = script_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        for line in lines[:30]:
            if "@description:" in line:
                metadata["description"] = line.split("@description:")[-1].strip()
            elif "@env_family:" in line:
                metadata["env_family"] = line.split("@env_family:")[-1].strip()
            elif "@phases:" in line:
                try:
                    metadata["phases"] = int(line.split("@phases:")[-1].strip())
                except ValueError:
                    pass
            elif "@total_timesteps:" in line:
                try:
                    metadata["total_timesteps"] = int(line.split("@total_timesteps:")[-1].strip())
                except ValueError:
                    pass
            elif "@environments:" in line:
                envs_str = line.split("@environments:")[-1].strip()
                for env_id in envs_str.split(","):
                    env_id = env_id.strip()
                    if env_id and env_id not in metadata["environments"]:
                        metadata["environments"].append(env_id)

        env_pattern = re.compile(r'(?:PHASE|LEVEL)\d+_ENV="([^"]+)"')
        for match in env_pattern.finditer(content):
            env_id = match.group(1)
            if env_id not in metadata["environments"]:
                metadata["environments"].append(env_id)
    except Exception:
        pass
    return metadata


# ---------------------------------------------------------------------------
# Form
# ---------------------------------------------------------------------------

class XuanCeScriptForm(QtWidgets.QDialog):
    """Configuration dialog for XuanCe custom training scripts."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, **kwargs: Any) -> None:
        super().__init__(parent)
        self.setWindowTitle("XuanCe Custom Script")
        self.setMinimumWidth(560)
        self._last_config: Optional[Dict[str, Any]] = None
        self._build_ui()
        self._populate_scripts()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Script Selection ---
        script_group = QtWidgets.QGroupBox("Script Selection")
        script_layout = QtWidgets.QFormLayout(script_group)

        self._script_combo = QtWidgets.QComboBox()
        self._script_combo.setToolTip("Select a predefined script or browse for a custom one.")
        script_layout.addRow("Script:", self._script_combo)

        self._script_info = QtWidgets.QLabel("Select a script to view details.")
        self._script_info.setWordWrap(True)
        self._script_info.setStyleSheet(
            "QLabel { color: #555; padding: 8px; background-color: #f5f5f5; border-radius: 4px; }"
        )
        script_layout.addRow(self._script_info)
        layout.addWidget(script_group)

        # --- Backend ---
        backend_group = QtWidgets.QGroupBox("XuanCe Backend")
        backend_layout = QtWidgets.QFormLayout(backend_group)

        self._backend_combo = QtWidgets.QComboBox()
        self._backend_combo.addItems(["pytorch", "tensorflow", "mindspore"])
        self._backend_combo.setCurrentIndex(0)
        self._backend_combo.setToolTip("Deep learning backend for XuanCe training.")
        backend_layout.addRow("Backend:", self._backend_combo)

        layout.addWidget(backend_group)

        # --- FastLane Settings ---
        fastlane_group = QtWidgets.QGroupBox("FastLane Settings")
        fl_layout = QtWidgets.QFormLayout(fastlane_group)

        self._fastlane_enabled_cb = QtWidgets.QCheckBox("Enable FastLane")
        self._fastlane_enabled_cb.setChecked(True)
        self._fastlane_enabled_cb.setToolTip("Enable live training visualization via Fast Lane.")
        fl_layout.addRow(self._fastlane_enabled_cb)

        self._fastlane_only_cb = QtWidgets.QCheckBox("FastLane Only")
        self._fastlane_only_cb.setChecked(True)
        self._fastlane_only_cb.setToolTip("Skip durable telemetry; stream frames only via Fast Lane.")
        fl_layout.addRow(self._fastlane_only_cb)

        self._video_mode_combo = QtWidgets.QComboBox()
        self._video_mode_combo.addItems(["grid", "single", "off"])
        self._video_mode_combo.setToolTip("Rendering strategy: grid (tiled envs), single (one env), off.")
        fl_layout.addRow("Video Mode:", self._video_mode_combo)

        self._grid_limit_spin = QtWidgets.QSpinBox()
        self._grid_limit_spin.setRange(1, 64)
        self._grid_limit_spin.setValue(8)
        self._grid_limit_spin.setToolTip("Number of environment slots to tile in grid mode.")
        fl_layout.addRow("Grid Limit:", self._grid_limit_spin)

        self._fastlane_slot_spin = QtWidgets.QSpinBox()
        self._fastlane_slot_spin.setRange(0, 63)
        self._fastlane_slot_spin.setValue(0)
        self._fastlane_slot_spin.setToolTip("Vectorized environment index for single mode (not used in grid mode).")
        self._fastlane_slot_spin.setEnabled(False)  # grid is default; slot only matters for single
        fl_layout.addRow("FastLane Slot:", self._fastlane_slot_spin)

        layout.addWidget(fastlane_group)

        # --- Tracking ---
        tracking_group = QtWidgets.QGroupBox("Tracking")
        tr_layout = QtWidgets.QFormLayout(tracking_group)

        self._tensorboard_cb = QtWidgets.QCheckBox("TensorBoard")
        self._tensorboard_cb.setChecked(True)
        tr_layout.addRow(self._tensorboard_cb)

        self._wandb_cb = QtWidgets.QCheckBox("WANDB")
        self._wandb_cb.setChecked(False)
        tr_layout.addRow(self._wandb_cb)

        self._wandb_project_input = QtWidgets.QLineEdit()
        self._wandb_project_input.setPlaceholderText("e.g. MOSAIC")
        self._wandb_project_input.setEnabled(False)
        tr_layout.addRow("WANDB Project:", self._wandb_project_input)

        self._wandb_entity_input = QtWidgets.QLineEdit()
        self._wandb_entity_input.setPlaceholderText("e.g. my-team")
        self._wandb_entity_input.setEnabled(False)
        tr_layout.addRow("WANDB Entity:", self._wandb_entity_input)

        layout.addWidget(tracking_group)

        # --- GPU ---
        gpu_group = QtWidgets.QGroupBox("Hardware")
        gpu_layout = QtWidgets.QFormLayout(gpu_group)

        self._gpu_cb = QtWidgets.QCheckBox("Use GPU (CUDA)")
        self._gpu_cb.setChecked(True)
        gpu_layout.addRow(self._gpu_cb)

        layout.addWidget(gpu_group)

        # --- Validation status (read-only, selectable/copyable) ---
        self._validation_output = QtWidgets.QTextEdit()
        self._validation_output.setReadOnly(True)
        self._validation_output.setMaximumHeight(120)
        self._validation_output.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self._validation_output)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        self._validate_btn = QtWidgets.QPushButton("Validate")
        self._validate_btn.setToolTip("Check that the script is valid before accepting.")
        btn_layout.addWidget(self._validate_btn)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_layout.addWidget(self._button_box)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Population & signals
    # ------------------------------------------------------------------

    def _populate_scripts(self) -> None:
        self._script_combo.blockSignals(True)
        self._script_combo.clear()

        if XUANCE_SCRIPTS_DIR.is_dir():
            scripts = sorted(XUANCE_SCRIPTS_DIR.glob("*.sh"))
            for script_path in scripts:
                description = _parse_script_metadata(script_path)
                label = f"{script_path.stem}"
                if description:
                    label = f"{script_path.stem} - {description}"
                self._script_combo.addItem(label, str(script_path))

        self._script_combo.addItem("Browse...", "BROWSE")
        self._script_combo.blockSignals(False)

        if self._script_combo.count() > 1:
            self._on_script_changed(0)

    def _connect_signals(self) -> None:
        self._script_combo.currentIndexChanged.connect(self._on_script_changed)
        self._video_mode_combo.currentTextChanged.connect(self._on_video_mode_changed)
        self._wandb_cb.toggled.connect(self._wandb_project_input.setEnabled)
        self._wandb_cb.toggled.connect(self._wandb_entity_input.setEnabled)
        self._fastlane_enabled_cb.toggled.connect(self._fastlane_only_cb.setEnabled)
        self._fastlane_enabled_cb.toggled.connect(self._video_mode_combo.setEnabled)
        self._fastlane_enabled_cb.toggled.connect(self._on_fastlane_toggled)
        self._validate_btn.clicked.connect(self._on_validate_clicked)
        self._button_box.accepted.connect(self._on_accept_clicked)
        self._button_box.rejected.connect(self.reject)

    def _on_video_mode_changed(self, mode: str) -> None:
        """Toggle grid/slot spinners based on video mode selection."""
        self._grid_limit_spin.setEnabled(mode == "grid")
        self._fastlane_slot_spin.setEnabled(mode == "single")

    def _on_fastlane_toggled(self, enabled: bool) -> None:
        """When FastLane is disabled, disable both grid and slot spinners."""
        if enabled:
            self._on_video_mode_changed(self._video_mode_combo.currentText())
        else:
            self._grid_limit_spin.setEnabled(False)
            self._fastlane_slot_spin.setEnabled(False)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_script_changed(self, index: int) -> None:
        data = self._script_combo.itemData(index)

        if data == "BROWSE":
            initial_dir = str(XUANCE_SCRIPTS_DIR) if XUANCE_SCRIPTS_DIR.is_dir() else str(Path.home())
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select XuanCe Script",
                initial_dir,
                "Shell Scripts (*.sh);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if path:
                script_path = Path(path)
                description = _parse_script_metadata(script_path)
                label = f"{script_path.stem}"
                if description:
                    label = f"{script_path.stem} - {description}"
                insert_index = self._script_combo.count() - 1
                self._script_combo.blockSignals(True)
                self._script_combo.insertItem(insert_index, label, str(script_path))
                self._script_combo.setCurrentIndex(insert_index)
                self._script_combo.blockSignals(False)
                self._update_script_info(script_path)
            else:
                self._script_combo.blockSignals(True)
                self._script_combo.setCurrentIndex(0)
                self._script_combo.blockSignals(False)
            return

        if data and data != "BROWSE":
            self._update_script_info(Path(data))
        else:
            self._script_info.setText("Select a script to view details.")

    def _update_script_info(self, script_path: Path) -> None:
        meta = _parse_script_full_metadata(script_path)
        parts = []
        if meta["description"]:
            parts.append(f"<b>{meta['description']}</b>")
        if meta["env_family"]:
            parts.append(f"Environment Family: {meta['env_family']}")
        if meta["phases"]:
            parts.append(f"Training Phases: {meta['phases']}")
        if meta["total_timesteps"]:
            parts.append(f"Total Timesteps: {meta['total_timesteps']:,}")
        if meta["environments"]:
            env_list = ", ".join(meta["environments"])
            parts.append(f"Environments: {env_list}")
        if parts:
            self._script_info.setText("<br>".join(parts))
        else:
            self._script_info.setText(f"Script: {script_path.name} (no metadata found)")

    def _selected_script_path(self) -> Optional[str]:
        data = self._script_combo.currentData()
        if data and data != "BROWSE":
            return str(data)
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_script(self) -> tuple[bool, str]:
        script_path_str = self._selected_script_path()
        if not script_path_str:
            return False, "No script selected."

        script_path = Path(script_path_str)
        if not script_path.exists():
            return False, f"Script not found: {script_path}"
        if not script_path.is_file():
            return False, f"Not a file: {script_path}"

        try:
            content = script_path.read_text(encoding="utf-8")
        except PermissionError:
            return False, f"Permission denied reading: {script_path}"

        warnings = []
        first_line = content.split("\n")[0] if content else ""
        if not first_line.startswith("#!"):
            warnings.append("Missing shebang (#!/bin/bash)")
        elif "bash" not in first_line and "sh" not in first_line:
            warnings.append(f"Unexpected shebang: {first_line}")

        if "MOSAIC_CONFIG_FILE" not in content:
            warnings.append("Script does not reference $MOSAIC_CONFIG_FILE -- may not read config")

        meta = _parse_script_full_metadata(script_path)

        parts = ["[PASSED] Script validation successful.\n"]
        parts.append(f"Script: {script_path.name}")
        if meta["description"]:
            parts.append(f"Description: {meta['description']}")
        if meta["env_family"]:
            parts.append(f"Environment Family: {meta['env_family']}")
        if meta["phases"]:
            parts.append(f"Training Phases: {meta['phases']}")
        if meta["total_timesteps"]:
            parts.append(f"Total Timesteps: {meta['total_timesteps']:,}")
        if meta["environments"]:
            for i, env_id in enumerate(meta["environments"], 1):
                parts.append(f"  Phase {i}: {env_id}")
        if warnings:
            parts.append("\nWarnings:")
            for w in warnings:
                parts.append(f"  - {w}")

        # Dry-run: preview artifact paths
        preview_dir = VAR_CUSTOM_SCRIPTS_DIR / "<ULID>"
        parts.append("\n--- Dry-Run Preview ---")
        parts.append(f"Run Dir:     {preview_dir}")
        parts.append(f"Config:      {preview_dir / 'config' / 'base_config.json'}")
        parts.append(f"TensorBoard: {preview_dir / 'tensorboard'}")
        parts.append(f"Checkpoints: {preview_dir / 'checkpoints'}")
        parts.append(f"Entry Point: /bin/bash {script_path}")
        parts.append(f"Backend: {self._backend_combo.currentText()}")
        parts.append(f"GPU: {'Yes' if self._gpu_cb.isChecked() else 'No'}")
        parts.append(f"TensorBoard Tracking: {'On' if self._tensorboard_cb.isChecked() else 'Off'}")
        parts.append(f"WANDB Tracking: {'On' if self._wandb_cb.isChecked() else 'Off'}")

        return True, "\n".join(parts)

    def _on_validate_clicked(self) -> None:
        ok, message = self._validate_script()
        color = "#2e7d32" if ok else "#c62828"
        self._validation_output.setStyleSheet(f"color: {color}; background: transparent; border: none;")
        self._validation_output.setPlainText(message)

    # ------------------------------------------------------------------
    # Accept & config building
    # ------------------------------------------------------------------

    def _on_accept_clicked(self) -> None:
        script_path_str = self._selected_script_path()
        if not script_path_str:
            QtWidgets.QMessageBox.warning(self, "No Script Selected", "Please select a script before accepting.")
            return

        ok, message = self._validate_script()
        if not ok:
            self._validation_output.setStyleSheet("color: #c62828; background: transparent; border: none;")
            self._validation_output.setPlainText(message)
            return

        script_name = Path(script_path_str).stem.replace("_", "-")
        run_name = _generate_run_id("xuance-script", script_name)
        run_id = str(ULID())

        config = self._build_config(script_path_str, run_id, run_name)
        self._last_config = copy.deepcopy(config)
        self.accept()

    def _build_config(self, script_path: str, run_id: str, run_name: str) -> Dict[str, Any]:
        script_name = Path(script_path).stem
        meta = _parse_script_full_metadata(Path(script_path))

        # Determine first environment from script metadata
        script_envs = meta.get("environments", [])
        script_env_family = meta.get("env_family", "")
        if script_envs:
            script_env_id = script_envs[0]
        elif script_env_family:
            script_env_id = f"{script_env_family}-curriculum"
        else:
            script_env_id = script_name

        use_gpu = self._gpu_cb.isChecked()
        track_tb = self._tensorboard_cb.isChecked()
        track_wandb = self._wandb_cb.isChecked()
        backend = self._backend_combo.currentText()
        fastlane_enabled = self._fastlane_enabled_cb.isChecked()

        # Custom script artifacts live under var/trainer/custom_scripts/{ULID}/
        # This is decoupled from standard training (var/trainer/runs/).
        # config.py detects custom scripts and skips path rewrites, so the
        # paths we set here are the final, authoritative ones.
        run_dir = VAR_CUSTOM_SCRIPTS_DIR / run_id
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = run_dir / "tensorboard"
        tb_absolute = str(tb_dir.resolve())

        # Worker config (base config JSON for the script to read)
        extras: Dict[str, Any] = {
            "cuda": use_gpu,
            "algo_params": {},  # Script controls all params
        }
        if track_tb:
            extras["tensorboard_dir"] = "tensorboard"
        if track_wandb:
            extras["track_wandb"] = True
            if self._wandb_project_input.text().strip():
                extras["wandb_project_name"] = self._wandb_project_input.text().strip()
            if self._wandb_entity_input.text().strip():
                extras["wandb_entity"] = self._wandb_entity_input.text().strip()

        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": "mappo",  # default placeholder; script overrides
            "env_id": script_env_id,
            "total_timesteps": meta.get("total_timesteps") or 1_000_000,
            "backend": backend,
            "device": "cuda:0" if use_gpu else "cpu",
            "extras": extras,
        }

        # Write base config JSON for the script
        config_file_path = config_dir / "base_config.json"
        config_file_path.write_text(json.dumps(worker_config, indent=2), encoding="utf-8")

        # Build environment variables -- these are FINAL (config.py won't rewrite)
        environment: Dict[str, Any] = {
            "XUANCE_RUN_ID": run_id,
            "XUANCE_DL_TOOLBOX": backend,
            # MOSAIC env vars -- point to custom_scripts/{ULID}/
            "MOSAIC_CONFIG_FILE": str(config_file_path),
            "MOSAIC_RUN_ID": run_id,
            "MOSAIC_RUN_DIR": str(run_dir),
            "MOSAIC_CUSTOM_SCRIPTS_DIR": str(config_dir),
            "MOSAIC_CHECKPOINT_DIR": str(run_dir / "checkpoints"),
        }

        # TensorBoard -- absolute path to custom_scripts/{ULID}/tensorboard
        if track_tb:
            environment["XUANCE_TENSORBOARD_DIR"] = tb_absolute

        # W&B
        if track_wandb:
            environment["WANDB_MODE"] = "online"
            if self._wandb_project_input.text().strip():
                environment["WANDB_PROJECT"] = self._wandb_project_input.text().strip()
            if self._wandb_entity_input.text().strip():
                environment["WANDB_ENTITY"] = self._wandb_entity_input.text().strip()
            wandb_api_key = os.environ.get("WANDB_API_KEY", "")
            if wandb_api_key:
                environment["WANDB_API_KEY"] = wandb_api_key
        else:
            environment["WANDB_MODE"] = "offline"

        # FastLane
        if fastlane_enabled:
            apply_fastlane_environment(
                environment,
                fastlane_only=self._fastlane_only_cb.isChecked(),
                fastlane_slot=self._fastlane_slot_spin.value(),
                video_mode=self._video_mode_combo.currentText(),
                grid_limit=self._grid_limit_spin.value(),
            )
            environment["MOSAIC_FASTLANE_ENABLED"] = "1"
        else:
            # Explicitly disable so shell scripts don't default to enabled
            environment["GYM_GUI_FASTLANE_ONLY"] = "0"
            environment["MOSAIC_FASTLANE_ENABLED"] = "0"
            environment["GYM_GUI_FASTLANE_VIDEO_MODE"] = "off"

        # Metadata -- NO "module" key; config.py uses this to detect custom
        # scripts and skip path rewrites.  "script" = /bin/bash tells the
        # dispatcher to run the shell script directly.
        metadata: Dict[str, Any] = {
            "ui": {
                "worker_id": "xuance_worker",
                "env_id": script_env_id,
                "custom_script": script_path,
                "custom_script_name": script_name,
                "fastlane_only": self._fastlane_only_cb.isChecked() if fastlane_enabled else False,
                "fastlane_slot": self._fastlane_slot_spin.value(),
                "fastlane_video_mode": self._video_mode_combo.currentText(),
                "fastlane_grid_limit": self._grid_limit_spin.value(),
            },
            "worker": {
                "worker_id": "xuance_worker",
                "script": "/bin/bash",
                "arguments": [script_path],
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "config": worker_config,
            },
            "artifacts": {
                "tensorboard": {
                    "enabled": track_tb,
                    # Absolute path -- analytics_tabs resolves directly
                    "relative_path": tb_absolute if track_tb else None,
                    "log_dir": tb_absolute if track_tb else None,
                },
                "wandb": {
                    "enabled": track_wandb,
                    "run_path": None,
                },
            },
        }

        return {
            "run_name": run_name,
            "entry_point": "/bin/bash",
            "arguments": [script_path],
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "gpus": {"requested": 1 if use_gpu else 0, "mandatory": False},
            },
            "metadata": metadata,
            "artifacts": {
                "output_prefix": f"custom_scripts/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration payload built by the form."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        return {}

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def log_constant(self, constant: Any, **kwargs: Any) -> None:
        log_constant(_LOGGER, constant, **kwargs)


__all__ = ["XuanCeScriptForm"]


# Late registration with the form factory.
from gym_gui.ui.forms.factory import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_script_form("xuance_worker"):
    _factory.register_script_form(
        "xuance_worker",
        lambda parent=None, **kw: XuanCeScriptForm(parent=parent, **kw),
    )
