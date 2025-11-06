"""SPADE-BDI training form (Qt dialog wrapper)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import os
import re
from typing import Any, Dict, Optional

from jsonschema import ValidationError
from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.services.trainer import validate_train_run_config
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_UI_PATH,
    LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
)
from gym_gui.validations import (
    AgentTrainFormInputs,
    validate_agent_train_form,
)
from gym_gui.core.schema import resolve_schema_for_game
from gym_gui.validations.validations_telemetry import ValidationService
from gym_gui.constants import (
    DEFAULT_RENDER_DELAY_MS,
    RENDER_DELAY_MIN_MS,
    RENDER_DELAY_MAX_MS,
    RENDER_DELAY_TICK_INTERVAL_MS,
    TRAINING_TELEMETRY_THROTTLE_MIN,
    TRAINING_TELEMETRY_THROTTLE_MAX,
    UI_RENDERING_THROTTLE_MIN,
    UI_RENDERING_THROTTLE_MAX,
    UI_TRAINING_SPEED_MIN,
    UI_TRAINING_SPEED_MAX,
    DEFAULT_TELEMETRY_BUFFER_SIZE,
    DEFAULT_EPISODE_BUFFER_SIZE,
    TELEMETRY_BUFFER_MIN,
    TELEMETRY_BUFFER_MAX,
    EPISODE_BUFFER_MIN,
    EPISODE_BUFFER_MAX,
    WORKER_ID_WIDTH,
)


_LOGGER = logging.getLogger("gym_gui.ui.spade_bdi_train_form")


class SpadeBdiTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """SPADE-BDI specific dialog to configure and submit a training run."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        default_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("Agent Train Form")
        # Set larger initial size to accommodate two-column layout
        # Dialog will resize dynamically when game configuration changes
        self.resize(800, 600)
        self.setMinimumWidth(600)

        self._selected_config: Optional[dict[str, Any]] = None
        # Track the user's live render preference so fast mode can restore it
        self._cached_live_render_disabled: Optional[bool] = None

        self._build_ui()
        self._update_analytics_controls()
        if default_game is not None:
            idx = self._game_combo.findText(default_game.value)
            if idx >= 0:
                self._game_combo.setCurrentIndex(idx)

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="SpadeBdiTrainForm opened",
            extra={"default_game": getattr(default_game, "value", None)},
        )

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(12)

        # Create scrollable content area for two-column form
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # Two-column container
        two_col_container = QtWidgets.QWidget()
        two_col_layout = QtWidgets.QHBoxLayout(two_col_container)
        two_col_layout.setSpacing(20)
        two_col_layout.setContentsMargins(0, 0, 0, 0)

        # ============ LEFT COLUMN ============
        left_layout = QtWidgets.QFormLayout()
        left_layout.setSpacing(8)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)

        # Game selection
        self._game_combo = QtWidgets.QComboBox(self)
        self._game_combo.addItems([
            "FrozenLake-v1",
            "FrozenLake-v2",
            "Taxi-v3",
            "CliffWalking-v1",
            "LunarLander-v3",
            "CartPole-v1",
        ])
        left_layout.addRow("Environment:", self._game_combo)
        self._game_combo.currentTextChanged.connect(self._on_game_selection_changed)
        
        # Algorithm selection
        self._algorithm_combo = QtWidgets.QComboBox(self)
        self._algorithm_combo.addItems([
            "Q-Learning",
            "DQN (future)",
            "PPO (future)",
            "A2C (future)",
        ])
        left_layout.addRow("Algorithm:", self._algorithm_combo)

        # Agent identifier input
        self._agent_id_edit = QtWidgets.QLineEdit(self)
        self._agent_id_edit.setPlaceholderText("e.g. agent_run_20251017")
        self._agent_id_edit.setText(self._default_agent_id())
        left_layout.addRow("Agent ID:", self._agent_id_edit)

        # Worker identifier input (supports distributed runs)
        self._worker_id_edit = QtWidgets.QLineEdit(self)
        self._worker_id_edit.setPlaceholderText("000001")
        self._worker_id_edit.setText(self._default_worker_id())
        left_layout.addRow("Worker ID:", self._worker_id_edit)

        # Episodes
        self._episodes_spin = QtWidgets.QSpinBox(self)
        self._episodes_spin.setRange(1, 10000)
        self._episodes_spin.setValue(1000)
        left_layout.addRow("Episodes:", self._episodes_spin)

        self._max_steps_spin = QtWidgets.QSpinBox(self)
        self._max_steps_spin.setRange(1, 10000)
        self._max_steps_spin.setValue(100)
        left_layout.addRow("Max Steps / Episode:", self._max_steps_spin)
        
        # Seed
        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        left_layout.addRow("Random Seed:", self._seed_spin)
        
        # Learning rate
        self._lr_edit = QtWidgets.QLineEdit("0.1")
        left_layout.addRow("Learning Rate:", self._lr_edit)
        
        # Gamma (discount)
        self._gamma_edit = QtWidgets.QLineEdit("0.99")
        left_layout.addRow("Discount (γ):", self._gamma_edit)
        
        # Epsilon decay
        self._epsilon_edit = QtWidgets.QLineEdit("0.995")
        left_layout.addRow("Epsilon Decay:", self._epsilon_edit)

        # ============ RIGHT COLUMN ============
        right_layout = QtWidgets.QFormLayout()
        right_layout.setSpacing(8)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        # Toggle: disable live rendering (telemetry only)
        self._disable_live_render_checkbox = QtWidgets.QCheckBox(
            "Disable live rendering (telemetry only)"
        )
        self._disable_live_render_checkbox.setToolTip(
            "When enabled, live grid/video views are not created; only telemetry tables update."
        )
        self._disable_live_render_checkbox.toggled.connect(self._on_live_render_toggle_changed)
        right_layout.addRow("Live Rendering:", self._disable_live_render_checkbox)

        # Toggle: fast training mode (disable telemetry)
        self._fast_training_checkbox = QtWidgets.QCheckBox(
            "Fast Training Mode"
        )
        self._fast_training_checkbox.setToolTip(
            "When enabled:\n"
            "• Disables per-step telemetry collection (no live updates)\n"
            "• Disables UI grid/chart rendering\n"
            "• No live Agent-{agent-id} tab\n"
            "• 30-50% faster training on GPU\n"
            "• TensorBoard metrics still available after training\n\n"
            "⚠ WARNING: Episode replay will not be available for this run"
        )
        self._fast_training_checkbox.toggled.connect(self._on_fast_training_toggled)
        right_layout.addRow("Fast Training (Disable Telemetry):", self._fast_training_checkbox)

        # Warning label for fast training mode
        self._fast_training_warning = QtWidgets.QLabel(
            "⚠ Disables live telemetry, UI updates, and episode replay. Use only for maximum speed."
        )
        self._fast_training_warning.setStyleSheet("color: #ff6b6b; font-size: 10px; font-weight: bold;")
        self._fast_training_warning.setVisible(False)
        right_layout.addRow("", self._fast_training_warning)

        analytics_widget = QtWidgets.QWidget()
        analytics_layout = QtWidgets.QVBoxLayout(analytics_widget)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(4)

        self._analytics_hint_label = QtWidgets.QLabel(
            "Enable Fast Training to export TensorBoard and WANDB analytics after the run completes."
        )
        self._analytics_hint_label.setStyleSheet("color: #777777; font-size: 10px;")
        analytics_layout.addWidget(self._analytics_hint_label)

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Export TensorBoard artifacts")
        self._tensorboard_checkbox.setEnabled(False)
        analytics_layout.addWidget(self._tensorboard_checkbox)

        self._wandb_checkbox = QtWidgets.QCheckBox("Export WANDB artifacts")
        self._wandb_checkbox.setEnabled(False)
        self._wandb_checkbox.toggled.connect(self._on_wandb_checkbox_toggled)
        analytics_layout.addWidget(self._wandb_checkbox)

        wandb_form = QtWidgets.QFormLayout()
        self._wandb_project_input = QtWidgets.QLineEdit()
        self._wandb_project_input.setPlaceholderText("e.g. MOSAIC")
        self._wandb_project_input.setEnabled(False)
        wandb_form.addRow("W&&B Project", self._wandb_project_input)

        self._wandb_entity_input = QtWidgets.QLineEdit()
        self._wandb_entity_input.setPlaceholderText("e.g. abdulhamid97mousa")
        self._wandb_entity_input.setEnabled(False)
        wandb_form.addRow("W&&B Entity", self._wandb_entity_input)

        self._wandb_run_name_input = QtWidgets.QLineEdit()
        self._wandb_run_name_input.setPlaceholderText("Optional run name")
        self._wandb_run_name_input.setEnabled(False)
        wandb_form.addRow("W&&B Run Name", self._wandb_run_name_input)

        self._wandb_api_key_input = QtWidgets.QLineEdit()
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._wandb_api_key_input.setEnabled(False)
        wandb_form.addRow("WANDB API Key", self._wandb_api_key_input)

        self._wandb_email_input = QtWidgets.QLineEdit()
        self._wandb_email_input.setPlaceholderText("Optional WANDB account email")
        self._wandb_email_input.setEnabled(False)
        wandb_form.addRow("WANDB Email", self._wandb_email_input)

        self._wandb_use_vpn_checkbox = QtWidgets.QCheckBox("Route WANDB traffic through VPN proxy")
        self._wandb_use_vpn_checkbox.setEnabled(False)
        self._wandb_use_vpn_checkbox.toggled.connect(self._on_wandb_vpn_checkbox_toggled)
        wandb_form.addRow("Use WANDB VPN", self._wandb_use_vpn_checkbox)

        self._wandb_http_proxy_input = QtWidgets.QLineEdit()
        self._wandb_http_proxy_input.setPlaceholderText("Optional HTTP proxy (e.g. http://127.0.0.1:7890)")
        self._wandb_http_proxy_input.setEnabled(False)
        wandb_form.addRow("WANDB HTTP Proxy", self._wandb_http_proxy_input)

        self._wandb_https_proxy_input = QtWidgets.QLineEdit()
        self._wandb_https_proxy_input.setPlaceholderText("Optional HTTPS proxy (e.g. https://127.0.0.1:7890)")
        self._wandb_https_proxy_input.setEnabled(False)
        wandb_form.addRow("WANDB HTTPS Proxy", self._wandb_https_proxy_input)

        analytics_layout.addLayout(wandb_form)

        right_layout.addRow("Analytics Export:", analytics_widget)

        # SLIDER 1: Training Telemetry Throttle (controls data collection speed)
        telemetry_throttle_layout = QtWidgets.QHBoxLayout()
        self._training_telemetry_throttle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._training_telemetry_throttle_slider.setRange(
            TRAINING_TELEMETRY_THROTTLE_MIN,
            TRAINING_TELEMETRY_THROTTLE_MAX,
        )
        self._training_telemetry_throttle_slider.setValue(TRAINING_TELEMETRY_THROTTLE_MIN)
        self._training_telemetry_throttle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._training_telemetry_throttle_slider.setTickInterval(1)
        self._training_telemetry_throttle_slider.valueChanged.connect(self._on_training_telemetry_throttle_changed)
        telemetry_throttle_layout.addWidget(self._training_telemetry_throttle_slider)

        self._training_telemetry_throttle_label = QtWidgets.QLabel("100%")
        self._training_telemetry_throttle_label.setMinimumWidth(40)
        telemetry_throttle_layout.addWidget(self._training_telemetry_throttle_label)

        right_layout.addRow("Telemetry Throttle:", telemetry_throttle_layout)

        # Warning label for training telemetry throttle
        self._training_telemetry_warning_label = QtWidgets.QLabel(
            "⚠ Values > 1 will skip telemetry collection."
        )
        self._training_telemetry_warning_label.setStyleSheet("color: #ff9800; font-size: 9px; font-style: italic;")
        self._training_telemetry_warning_label.setVisible(False)
        right_layout.addRow("", self._training_telemetry_warning_label)

        # SLIDER 2: UI Live Rendering Throttle
        ui_rendering_layout = QtWidgets.QHBoxLayout()
        self._ui_rendering_throttle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._ui_rendering_throttle_slider.setRange(
            UI_RENDERING_THROTTLE_MIN,
            UI_RENDERING_THROTTLE_MAX,
        )
        self._ui_rendering_throttle_slider.setValue(UI_RENDERING_THROTTLE_MIN)
        self._ui_rendering_throttle_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._ui_rendering_throttle_slider.setTickInterval(1)
        self._ui_rendering_throttle_slider.valueChanged.connect(self._on_ui_rendering_throttle_changed)
        ui_rendering_layout.addWidget(self._ui_rendering_throttle_slider)

        self._ui_rendering_throttle_label = QtWidgets.QLabel("100%")
        self._ui_rendering_throttle_label.setMinimumWidth(40)
        ui_rendering_layout.addWidget(self._ui_rendering_throttle_label)

        right_layout.addRow("UI Rendering Throttle:", ui_rendering_layout)

        # Warning label for UI rendering throttle
        self._ui_rendering_warning_label = QtWidgets.QLabel(
            "⚠ Values > 1 will skip frames."
        )
        self._ui_rendering_warning_label.setStyleSheet("color: #ff9800; font-size: 9px; font-style: italic;")
        self._ui_rendering_warning_label.setVisible(False)
        right_layout.addRow("", self._ui_rendering_warning_label)

        # SLIDER 2b: Render Delay
        render_delay_layout = QtWidgets.QHBoxLayout()
        self._render_delay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._render_delay_slider.setRange(RENDER_DELAY_MIN_MS, RENDER_DELAY_MAX_MS)
        self._render_delay_slider.setValue(DEFAULT_RENDER_DELAY_MS)
        self._render_delay_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._render_delay_slider.setTickInterval(RENDER_DELAY_TICK_INTERVAL_MS)
        self._render_delay_slider.valueChanged.connect(self._on_render_delay_changed)
        render_delay_layout.addWidget(self._render_delay_slider)

        default_fps = max(1, round(1000 / DEFAULT_RENDER_DELAY_MS))
        self._render_delay_label = QtWidgets.QLabel(
            f"{DEFAULT_RENDER_DELAY_MS}ms ({default_fps} FPS)"
        )
        self._render_delay_label.setMinimumWidth(100)
        render_delay_layout.addWidget(self._render_delay_label)

        right_layout.addRow("Render Delay:", render_delay_layout)

        # Info label for render delay
        self._render_delay_info_label = QtWidgets.QLabel(
            "ℹ Controls visual grid update speed."
        )
        self._render_delay_info_label.setStyleSheet("color: #1976d2; font-size: 9px; font-style: italic;")
        right_layout.addRow("", self._render_delay_info_label)

        # SLIDER 3: UI Training Speed
        ui_speed_layout = QtWidgets.QHBoxLayout()
        self._ui_training_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._ui_training_speed_slider.setRange(UI_TRAINING_SPEED_MIN, UI_TRAINING_SPEED_MAX)
        self._ui_training_speed_slider.setValue(UI_TRAINING_SPEED_MAX)
        self._ui_training_speed_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._ui_training_speed_slider.setTickInterval(10)
        self._ui_training_speed_slider.valueChanged.connect(self._on_ui_training_speed_changed)
        ui_speed_layout.addWidget(self._ui_training_speed_slider)

        self._ui_training_speed_label = QtWidgets.QLabel(f"{UI_TRAINING_SPEED_MAX * 10}ms")
        self._ui_training_speed_label.setMinimumWidth(50)
        ui_speed_layout.addWidget(self._ui_training_speed_label)

        right_layout.addRow("Step Delay (ms):", ui_speed_layout)

        # Warning label for UI training speed
        self._ui_training_speed_warning_label = QtWidgets.QLabel(
            "⚠ 0ms will make training too fast."
        )
        self._ui_training_speed_warning_label.setStyleSheet("color: #ff9800; font-size: 9px; font-style: italic;")
        self._ui_training_speed_warning_label.setVisible(False)
        right_layout.addRow("", self._ui_training_speed_warning_label)

        # Telemetry Buffer Size
        buffer_layout = QtWidgets.QHBoxLayout()
        self._telemetry_buffer_spin = QtWidgets.QSpinBox(self)
        self._telemetry_buffer_spin.setRange(TELEMETRY_BUFFER_MIN, TELEMETRY_BUFFER_MAX)
        self._telemetry_buffer_spin.setValue(DEFAULT_TELEMETRY_BUFFER_SIZE)
        self._telemetry_buffer_spin.setSingleStep(256)
        buffer_layout.addWidget(self._telemetry_buffer_spin)

        self._telemetry_buffer_label = QtWidgets.QLabel("steps (UI)")
        buffer_layout.addWidget(self._telemetry_buffer_label)
        buffer_layout.addStretch()

        right_layout.addRow("Telemetry Buffer:", buffer_layout)

        # Episode Buffer Size
        episode_buffer_layout = QtWidgets.QHBoxLayout()
        self._episode_buffer_spin = QtWidgets.QSpinBox(self)
        self._episode_buffer_spin.setRange(EPISODE_BUFFER_MIN, EPISODE_BUFFER_MAX)
        self._episode_buffer_spin.setValue(DEFAULT_EPISODE_BUFFER_SIZE)
        self._episode_buffer_spin.setSingleStep(10)
        episode_buffer_layout.addWidget(self._episode_buffer_spin)
        
        # Connect episodes/max_steps changes to auto-calculate buffers
        self._episodes_spin.valueChanged.connect(self._auto_calculate_buffers)
        self._max_steps_spin.valueChanged.connect(self._auto_calculate_buffers)

        self._episode_buffer_label = QtWidgets.QLabel("episodes")
        episode_buffer_layout.addWidget(self._episode_buffer_label)
        episode_buffer_layout.addStretch()

        right_layout.addRow("Episode Buffer:", episode_buffer_layout)

        # Add left and right columns to the two-column container
        two_col_layout.addWidget(left_widget, 1)
        two_col_layout.addWidget(right_widget, 1)
        
        scroll_layout.addWidget(two_col_container)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Game-specific configuration (dynamic)
        self._game_config_group = QtWidgets.QGroupBox("Game Configuration", self)
        self._game_config_layout = QtWidgets.QFormLayout(self._game_config_group)
        self._game_config_widgets: Dict[str, QtWidgets.QWidget] = {}
        self._build_game_config_widgets()
        main_layout.addWidget(self._game_config_group)

        # BDI Agent Settings (collapsible)
        self._bdi_group = QtWidgets.QGroupBox("BDI Agent Settings (SPADE)", self)
        self._bdi_group.setCheckable(True)
        self._bdi_group.setChecked(False)
        bdi_layout = QtWidgets.QFormLayout(self._bdi_group)
        
        self._bdi_jid_edit = QtWidgets.QLineEdit(self._bdi_group)
        self._bdi_jid_edit.setPlaceholderText("XMPP JID for agent")
        self._bdi_jid_edit.setText("agent@localhost")
        bdi_layout.addRow("XMPP JID:", self._bdi_jid_edit)
        
        self._bdi_password_edit = QtWidgets.QLineEdit(self._bdi_group)
        self._bdi_password_edit.setPlaceholderText("XMPP password")
        self._bdi_password_edit.setText("secret")
        self._bdi_password_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        bdi_layout.addRow("XMPP Password ~admin~:", self._bdi_password_edit)
        
        # ASL file path (optional)
        asl_layout = QtWidgets.QHBoxLayout()
        self._bdi_asl_file_edit = QtWidgets.QLineEdit(self._bdi_group)
        self._bdi_asl_file_edit.setPlaceholderText("Optional: path to ASL beliefs file")
        asl_layout.addWidget(self._bdi_asl_file_edit)
        
        asl_browse_btn = QtWidgets.QPushButton("Browse...", self._bdi_group)
        asl_browse_btn.clicked.connect(self._on_browse_asl_file)
        asl_layout.addWidget(asl_browse_btn)
        
        bdi_layout.addRow("ASL File (optional):", asl_layout)
        
        main_layout.addWidget(self._bdi_group)
        
        # Advanced options (collapsible)
        self._advanced_group = QtWidgets.QGroupBox("Advanced Options", self)
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        advanced_layout = QtWidgets.QVBoxLayout(self._advanced_group)
        
        self._custom_config = QtWidgets.QPlainTextEdit(self._advanced_group)
        self._custom_config.setPlaceholderText(
            "Optional: Provide trainer-schema overrides (see README).\n\n"
            "Example:\n"
            '{\n'
            '  "resources": {\n'
            '    "cpus": 4,\n'
            '    "memory_mb": 4096,\n'
            '    "gpus": {"requested": 1, "mandatory": true}\n'
            '  },\n'
            '  "environment": {"EXTRA_FLAG": "1"},\n'
            '  "metadata": {"notes": "custom run"}\n'
            '}'
        )
        advanced_layout.addWidget(self._custom_config)
        main_layout.addWidget(self._advanced_group)

        # Initialize control states based on live rendering toggle
        self._update_render_control_states()
        self._update_step_delay_state()

        # Info label
        info_label = QtWidgets.QLabel(
            '<i>Training will run in the background. '
            'Watch progress in the "Live Telemetry" tab.</i>'
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Trigger initial buffer calculation with default values
        QtCore.QTimer.singleShot(0, self._auto_calculate_buffers)

    def _on_browse_asl_file(self) -> None:
        """Handle ASL file browser button click."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ASL Beliefs File",
            "",
            "ASL Files (*.asl);;All Files (*)",
        )
        if path:
            self._bdi_asl_file_edit.setText(path)

    def _build_game_config_widgets(self) -> None:
        """Build game-specific configuration widgets."""
        # Clear existing widgets - properly remove all items from layout
        while self._game_config_layout.count() > 0:
            item = self._game_config_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                # Also handle layout items (for nested layouts)
                layout = item.layout()
                if layout is not None:
                    # Recursively delete all widgets in nested layouts
                    while layout.count() > 0:
                        child_item = layout.takeAt(0)
                        if child_item is not None:
                            child_widget = child_item.widget()
                            if child_widget is not None:
                                child_widget.deleteLater()
        self._game_config_widgets.clear()

        game_name = self._game_combo.currentText()

        if game_name.startswith("FrozenLake"):
            self._build_frozen_lake_config(game_name)
        elif game_name == "Taxi-v3":
            self._build_taxi_config()
        elif game_name == "CliffWalking-v1":
            self._build_cliff_walking_config()
        else:
            # No game-specific config for other games
            label = QtWidgets.QLabel("No additional configuration available")
            label.setStyleSheet("color: #999; font-style: italic;")
            self._game_config_layout.addRow(label)

    def _build_frozen_lake_config(self, game_name: str) -> None:
        """Build FrozenLake-specific configuration widgets.

        Both FrozenLake-v1 and FrozenLake-v2 have configurable options.
        """
        # Show is_slippery option for both FrozenLake-v1 and FrozenLake-v2
        self._frozen_lake_slippery = QtWidgets.QCheckBox("Slippery (stochastic movement)")
        self._frozen_lake_slippery.setChecked(True)  # Default is slippery for FrozenLake
        self._game_config_widgets["is_slippery"] = self._frozen_lake_slippery
        self._game_config_layout.addRow("Is Slippery:", self._frozen_lake_slippery)

        # Additional configuration only for FrozenLake-v2
        if game_name == "FrozenLake-v2":
            # Grid size
            grid_layout = QtWidgets.QHBoxLayout()
            self._frozen_lake_height = QtWidgets.QSpinBox()
            self._frozen_lake_height.setRange(4, 16)
            self._frozen_lake_height.setValue(8)
            grid_layout.addWidget(QtWidgets.QLabel("Height:"))
            grid_layout.addWidget(self._frozen_lake_height)
            grid_layout.addWidget(QtWidgets.QLabel("Width:"))
            self._frozen_lake_width = QtWidgets.QSpinBox()
            self._frozen_lake_width.setRange(4, 16)
            self._frozen_lake_width.setValue(8)
            grid_layout.addWidget(self._frozen_lake_width)
            grid_layout.addStretch()
            # Store the spinboxes for later retrieval
            self._game_config_widgets["grid_height"] = self._frozen_lake_height
            self._game_config_widgets["grid_width"] = self._frozen_lake_width
            self._game_config_layout.addRow("Grid Size:", grid_layout)

            # Start position (row, col)
            start_layout = QtWidgets.QHBoxLayout()
            self._frozen_lake_start_row = QtWidgets.QSpinBox()
            self._frozen_lake_start_row.setRange(0, 15)
            self._frozen_lake_start_row.setValue(0)
            start_layout.addWidget(QtWidgets.QLabel("Row:"))
            start_layout.addWidget(self._frozen_lake_start_row)
            self._frozen_lake_start_col = QtWidgets.QSpinBox()
            self._frozen_lake_start_col.setRange(0, 15)
            self._frozen_lake_start_col.setValue(0)
            start_layout.addWidget(QtWidgets.QLabel("Col:"))
            start_layout.addWidget(self._frozen_lake_start_col)
            start_layout.addStretch()
            self._game_config_widgets["start_row"] = self._frozen_lake_start_row
            self._game_config_widgets["start_col"] = self._frozen_lake_start_col
            self._game_config_layout.addRow("Start Position:", start_layout)

            # Goal position (row, col)
            goal_layout = QtWidgets.QHBoxLayout()
            self._frozen_lake_goal_row = QtWidgets.QSpinBox()
            self._frozen_lake_goal_row.setRange(0, 15)
            self._frozen_lake_goal_row.setValue(7)
            goal_layout.addWidget(QtWidgets.QLabel("Row:"))
            goal_layout.addWidget(self._frozen_lake_goal_row)
            self._frozen_lake_goal_col = QtWidgets.QSpinBox()
            self._frozen_lake_goal_col.setRange(0, 15)
            self._frozen_lake_goal_col.setValue(7)
            goal_layout.addWidget(QtWidgets.QLabel("Col:"))
            goal_layout.addWidget(self._frozen_lake_goal_col)
            goal_layout.addStretch()
            self._game_config_widgets["goal_row"] = self._frozen_lake_goal_row
            self._game_config_widgets["goal_col"] = self._frozen_lake_goal_col
            self._game_config_layout.addRow("Goal Position:", goal_layout)

            # Hole count
            self._frozen_lake_hole_count = QtWidgets.QSpinBox()
            self._frozen_lake_hole_count.setRange(0, 50)
            self._frozen_lake_hole_count.setValue(10)
            self._game_config_widgets["hole_count"] = self._frozen_lake_hole_count
            self._game_config_layout.addRow("Hole Count:", self._frozen_lake_hole_count)

            # Random holes checkbox
            self._frozen_lake_random_holes = QtWidgets.QCheckBox("Random hole placement")
            self._frozen_lake_random_holes.setChecked(False)
            self._game_config_widgets["random_holes"] = self._frozen_lake_random_holes
            self._game_config_layout.addRow("Random Holes:", self._frozen_lake_random_holes)

    def _build_taxi_config(self) -> None:
        """Build Taxi-specific configuration widgets."""
        info_label = QtWidgets.QLabel("Taxi-v3 uses default configuration (no additional options)")
        info_label.setStyleSheet("color: #999; font-style: italic;")
        self._game_config_layout.addRow(info_label)

    def _build_cliff_walking_config(self) -> None:
        """Build CliffWalking-specific configuration widgets."""
        # is_slippery checkbox
        self._cliff_walking_slippery = QtWidgets.QCheckBox("Slippery (stochastic movement)")
        self._cliff_walking_slippery.setChecked(False)  # Default is not slippery
        self._game_config_widgets["is_slippery"] = self._cliff_walking_slippery
        self._game_config_layout.addRow("Is Slippery:", self._cliff_walking_slippery)

    def _on_game_selection_changed(self, game_name: str) -> None:
        """Handle game selection change."""
        self._build_game_config_widgets()
        # Force layout refresh to properly display new widgets
        # This prevents overlapping text and ensures proper widget sizing
        self._game_config_group.updateGeometry()
        self.updateGeometry()

    def _on_training_telemetry_throttle_changed(self, value: int) -> None:
        """Handle training telemetry throttle slider change.

        This controls how fast telemetry data is collected and written to database.
        Affects: Telemetry Recent Episodes and Telemetry Recent Steps table population speed.

        Slider value (1-10) to collection percentage:
        - value=1 means 100% (collect every step) - RECOMMENDED
        - value=2 means 50% (collect every 2nd step)
        - value=10 means 10% (collect every 10th step)
        """
        percentage = int(100 / value)
        self._training_telemetry_throttle_label.setText(f"{percentage}%")

        # Show warning if value > 1 (skipping telemetry collection)
        self._training_telemetry_warning_label.setVisible(value > 1)

    def _on_live_render_toggle_changed(self, checked: bool) -> None:
        """Handle live rendering enable/disable toggle."""
        self._update_render_control_states()

    def _on_fast_training_toggled(self, checked: bool) -> None:
        """Handle fast training mode toggle."""
        self._fast_training_warning.setVisible(checked)

        if checked:
            # Remember the current user preference before forcing the switch
            self._cached_live_render_disabled = self._disable_live_render_checkbox.isChecked()
            # When fast mode is on, force live rendering to be disabled
            if not self._disable_live_render_checkbox.isChecked():
                self._disable_live_render_checkbox.setChecked(True)
            self._disable_live_render_checkbox.setEnabled(False)
            # Also disable rendering throttles since they won't be used
            self._ui_rendering_throttle_slider.setEnabled(False)
            self._training_telemetry_throttle_slider.setEnabled(False)
        else:
            # Re-enable UI options when fast mode is off
            self._disable_live_render_checkbox.setEnabled(True)
            self._ui_rendering_throttle_slider.setEnabled(True)
            self._training_telemetry_throttle_slider.setEnabled(True)
            if self._cached_live_render_disabled is not None:
                self._disable_live_render_checkbox.setChecked(self._cached_live_render_disabled)
                self._cached_live_render_disabled = None
            # Ensure dependent controls reflect the restored state
            self._update_render_control_states()

        self._update_step_delay_state()
        self._update_analytics_controls()

    def _update_render_control_states(self) -> None:
        """Enable/disable render-related controls based on toggle."""
        enabled = not getattr(self, "_disable_live_render_checkbox", None) or not self._disable_live_render_checkbox.isChecked()
        for widget in (
            getattr(self, "_ui_rendering_throttle_slider", None),
            getattr(self, "_render_delay_slider", None),
        ):
            if widget is not None:
                widget.setEnabled(enabled)
        for widget in (
            getattr(self, "_ui_rendering_throttle_label", None),
            getattr(self, "_render_delay_label", None),
            getattr(self, "_ui_rendering_warning_label", None),
            getattr(self, "_render_delay_info_label", None),
        ):
            if widget is not None:
                widget.setEnabled(enabled)

        if enabled:
            if hasattr(self, "_ui_rendering_throttle_slider"):
                self._on_ui_rendering_throttle_changed(self._ui_rendering_throttle_slider.value())
            if hasattr(self, "_render_delay_slider"):
                self._on_render_delay_changed(self._render_delay_slider.value())
        else:
            if hasattr(self, "_ui_rendering_throttle_label"):
                self._ui_rendering_throttle_label.setText("Disabled")
            if hasattr(self, "_render_delay_label"):
                self._render_delay_label.setText("Disabled")

    def _update_step_delay_state(self) -> None:
        """Disable the step delay control when telemetry is disabled and fast training is enabled."""
        slider = getattr(self, "_ui_training_speed_slider", None)
        label = getattr(self, "_ui_training_speed_label", None)
        warning_label = getattr(self, "_ui_training_speed_warning_label", None)
        if slider is None or label is None:
            return

        fast_training_enabled = self._fast_training_checkbox.isChecked()

        telemetry_disabled = fast_training_enabled
        telemetry_spin = getattr(self, "_telemetry_buffer_spin", None)
        if isinstance(telemetry_spin, QtWidgets.QSpinBox) and telemetry_spin.value() == 0:
            telemetry_disabled = True

        should_disable = fast_training_enabled and telemetry_disabled

        slider.setEnabled(not should_disable)
        if warning_label is not None:
            warning_label.setEnabled(not should_disable)

        if should_disable:
            label.setText("Disabled (Fast Training)")
            if warning_label is not None:
                warning_label.setVisible(False)
        else:
            self._on_ui_training_speed_changed(slider.value())

    def _update_analytics_controls(self) -> None:
        """Enable analytics export controls only when fast training is active."""
        enabled = self._fast_training_checkbox.isChecked()
        wandb_checkbox = getattr(self, "_wandb_checkbox", None)
        tensorboard_checkbox = getattr(self, "_tensorboard_checkbox", None)
        if tensorboard_checkbox is not None:
            tensorboard_checkbox.setEnabled(enabled)
            if not enabled:
                tensorboard_checkbox.setChecked(False)
        if wandb_checkbox is not None:
            wandb_checkbox.setEnabled(enabled)
            if not enabled:
                wandb_checkbox.setChecked(False)

        wandb_fields = (
            getattr(self, "_wandb_project_input", None),
            getattr(self, "_wandb_entity_input", None),
            getattr(self, "_wandb_run_name_input", None),
            getattr(self, "_wandb_api_key_input", None),
            getattr(self, "_wandb_email_input", None),
        )
        wandb_proxy_fields = (
            getattr(self, "_wandb_http_proxy_input", None),
            getattr(self, "_wandb_https_proxy_input", None),
        )
        vpn_checkbox = getattr(self, "_wandb_use_vpn_checkbox", None)
        wandb_active = enabled and isinstance(wandb_checkbox, QtWidgets.QCheckBox) and wandb_checkbox.isChecked()
        for field in wandb_fields:
            if isinstance(field, QtWidgets.QLineEdit):
                field.setEnabled(wandb_active)
        if isinstance(vpn_checkbox, QtWidgets.QCheckBox):
            vpn_checkbox.setEnabled(wandb_active)
            if not wandb_active:
                vpn_checkbox.setChecked(False)
        wandb_vpn_active = wandb_active and isinstance(vpn_checkbox, QtWidgets.QCheckBox) and vpn_checkbox.isChecked()
        for field in wandb_proxy_fields:
            if isinstance(field, QtWidgets.QLineEdit):
                field.setEnabled(wandb_vpn_active)

        hint = getattr(self, "_analytics_hint_label", None)
        if hint is not None:
            if enabled:
                hint.setText("Select analytics to export after the run completes (fast training only).")
                hint.setStyleSheet("color: #2e7d32; font-size: 10px;")
            else:
                hint.setText("Enable Fast Training to export TensorBoard and WANDB analytics.")
                hint.setStyleSheet("color: #777777; font-size: 10px;")

    def _on_wandb_checkbox_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_analytics_controls()

    def _on_wandb_vpn_checkbox_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_analytics_controls()

    def _on_ui_rendering_throttle_changed(self, value: int) -> None:
        """Handle UI rendering throttle slider change.

        This controls how often the visual grid display updates on screen.
        Affects: Live Rendering grid visualization only (NOT database writes or table updates).

        Slider value (1-10) to rendering percentage:
        - value=1 means 100% (render every step) - RECOMMENDED
        - value=2 means 50% (render every 2nd step)
        - value=10 means 10% (render every 10th step)
        """
        percentage = int(100 / value)
        self._ui_rendering_throttle_label.setText(f"{percentage}%")

        # Show warning if value > 1 (skipping frames)
        self._ui_rendering_warning_label.setVisible(value > 1)

    def _on_render_delay_changed(self, value: int) -> None:
        """Handle render delay slider change.

        This controls the delay between visual grid renders (independent of table update speed).
        Affects: Live Rendering grid visualization smoothness (NOT database writes or table updates).

        Slider value (10-500) to delay in milliseconds and FPS:
        - value=10 means 10ms (100 FPS, very fast, high CPU)
        - value=50 means 50ms (20 FPS, smooth)
        - value=100 means 100ms (10 FPS, default, smooth)
        - value=200 means 200ms (5 FPS, slower)
        - value=500 means 500ms (2 FPS, very slow, low CPU)
        """
        fps = 1000 // value  # Calculate FPS from delay
        self._render_delay_label.setText(f"{value}ms ({fps} FPS)")

    def _on_ui_training_speed_changed(self, value: int) -> None:
        """Handle UI training speed slider change.

        This controls the artificial delay between training steps for human observation.
        Affects: Overall training duration and visual observation speed.

        Slider value (0-100) to delay in milliseconds:
        - value=0 means 0ms (no delay, too fast to observe)
        - value=50 means 500ms (0.5 second delay per step)
        - value=100 means 1000ms (1 second delay per step) - RECOMMENDED
        """
        delay_ms = value * 10  # Convert slider value to milliseconds
        self._ui_training_speed_label.setText(f"{delay_ms}ms")

        # Show warning if value = 0 (no delay, too fast)
        self._ui_training_speed_warning_label.setVisible(value == 0)

    def _auto_calculate_buffers(self) -> None:
        """Auto-calculate buffer sizes based on training configuration.
        
        Smart buffer sizing:
        - UI Step Buffer: min(episodes * max_steps * 0.1, TELEMETRY_BUFFER_MAX)
        - UI Episode Buffer: min(episodes, EPISODE_BUFFER_MAX)
        
        This ensures buffers are large enough to hold a reasonable portion of training data
        without excessive memory usage.
        """
        episodes = self._episodes_spin.value()
        max_steps = self._max_steps_spin.value()
        
        # Calculate total expected steps
        total_expected_steps = episodes * max_steps
        
        # UI Step buffer: hold 10% of total steps (or max limit)
        # This is reasonable for UI display without overwhelming memory
        suggested_step_buffer = min(int(total_expected_steps * 0.1), TELEMETRY_BUFFER_MAX)
        suggested_step_buffer = max(suggested_step_buffer, TELEMETRY_BUFFER_MIN)
        
        # UI Episode buffer: hold all episodes (or max limit)
        suggested_episode_buffer = min(episodes, EPISODE_BUFFER_MAX)
        suggested_episode_buffer = max(suggested_episode_buffer, EPISODE_BUFFER_MIN)
        
        # Update spin boxes
        self._telemetry_buffer_spin.setValue(suggested_step_buffer)
        self._episode_buffer_spin.setValue(suggested_episode_buffer)
        
        self.log_constant(
            LOG_UI_TRAIN_FORM_TRACE,
            message="Auto-calculated buffer sizes",
            extra={
                "episodes": episodes,
                "max_steps": max_steps,
                "total_expected_steps": total_expected_steps,
                "suggested_step_buffer": suggested_step_buffer,
                "suggested_episode_buffer": suggested_episode_buffer,
            },
        )

    def _on_accept(self) -> None:
        """Validate inputs and build config before accepting."""
        # Validate input types and ranges
        validation_errors = self._validate_inputs()
        if validation_errors:
            error_msg = "⚠ Input Validation Errors:\n\n" + "\n".join(
                [f"• {err}" for err in validation_errors]
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                error_msg,
            )
            return

        try:
            custom_overrides = self._parse_custom_overrides()
            base_payload = self._build_base_config()
            merged = _deep_merge_dict(base_payload, custom_overrides)

            # Validate against trainer schema for early feedback
            validated = validate_train_run_config(merged)
            self._selected_config = dict(validated.payload)
            self._last_metadata = validated
            self.accept()

        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Configuration",
                f"Please check your inputs:\n{e}",
            )
        else:
            if self._selected_config is not None:
                worker_meta = self._selected_config.get("metadata", {}).get("worker", {})
                config_meta = worker_meta.get("config", {}) if isinstance(worker_meta, dict) else {}
                self.log_constant(
                    LOG_UI_TRAIN_FORM_INFO,
                    message="TrainAgentDialog accepted",
                    extra={
                        "run_name": self._selected_config.get("run_name"),
                        "agent_id": worker_meta.get("agent_id"),
                        "episodes": config_meta.get("max_episodes"),
                        "max_steps": config_meta.get("max_steps_per_episode"),
                    },
                )

    def _validate_inputs(self) -> list[str]:
        """Validate all input fields and return list of error messages.
        
        Returns:
            List of error messages (empty if all inputs valid)
        """
        inputs = AgentTrainFormInputs(
            episodes=self._episodes_spin.value(),
            max_steps_per_episode=self._max_steps_spin.value(),
            seed=self._seed_spin.value(),
            learning_rate=self._lr_edit.text(),
            discount=self._gamma_edit.text(),
            epsilon_decay=self._epsilon_edit.text(),
            agent_id=self._agent_id_edit.text(),
            bdi_enabled=self._bdi_group.isChecked(),
            bdi_jid=self._bdi_jid_edit.text(),
            bdi_password=self._bdi_password_edit.text(),
            worker_id=self._worker_id_edit.text(),
        )
        return validate_agent_train_form(inputs)

    def _parse_custom_overrides(self) -> dict[str, Any]:
        if not self._advanced_group.isChecked():
            return {}
        custom_text = self._custom_config.toPlainText().strip()
        if not custom_text:
            return {}
        overrides = json.loads(custom_text)
        if not isinstance(overrides, dict):
            raise ValueError("Custom overrides must be a JSON object")
        return overrides

    def _build_base_config(self) -> dict[str, Any]:
        algorithm = self._algorithm_combo.currentText()
        game_id = self._game_combo.currentText()
        max_episodes = self._episodes_spin.value()
        max_steps = self._max_steps_spin.value()
        seed = self._seed_spin.value()
        schema = resolve_schema_for_game(game_id)
        schema_id = schema.schema_id if schema is not None else "telemetry.step.default"
        schema_version = schema.version if schema is not None else 1
        validator = ValidationService(strict_mode=False)
        schema_definition = validator.get_step_schema(game_id)
        if schema_definition is None and schema is not None:
            schema_definition = schema.as_json_schema()
        if schema_definition is None:
            schema_definition = {}
        worker_id_input = self._worker_id_edit.text().strip()
        worker_id = self._normalize_worker_id(worker_id_input)
        learning_rate = float(self._lr_edit.text())
        gamma = float(self._gamma_edit.text())
        epsilon_decay = float(self._epsilon_edit.text())
        agent_id_input = self._agent_id_edit.text().strip()
        normalized_agent_id = self._normalize_agent_id(agent_id_input)
        worker_agent_id = normalized_agent_id or f"worker_{worker_id}" if worker_id.isdigit() else (normalized_agent_id or worker_id)

        # BDI settings
        bdi_enabled = self._bdi_group.isChecked()
        bdi_config = {}
        agent_type = "Headless"  # Default
        
        if bdi_enabled:
            agent_type = "BDI"
            bdi_jid = self._bdi_jid_edit.text().strip() or "agent@localhost"
            bdi_password = self._bdi_password_edit.text() or "secret"
            bdi_asl_file = self._bdi_asl_file_edit.text().strip()
            
            bdi_config = {
                "jid": bdi_jid,
                "password": bdi_password,
            }
            if bdi_asl_file:
                bdi_config["asl_file"] = bdi_asl_file
            
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="BDI Agent configuration collected",
                extra={
                    "agent_type": agent_type,
                    "bdi_jid": bdi_jid,
                    "bdi_password": "[***]",
                    "bdi_asl_file": bdi_asl_file if bdi_asl_file else "(not provided)",
                    "config_keys": list(bdi_config.keys()),
                },
            )
        else:
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="Headless Agent configuration (BDI disabled)",
                extra={
                    "agent_type": agent_type,
                    "algorithm": algorithm,
                },
            )

        # Collect game-specific configuration
        game_config_overrides: Dict[str, Any] = {}
        if game_id in ("FrozenLake-v1", "FrozenLake-v2"):
            # Both FrozenLake-v1 and FrozenLake-v2 have is_slippery option
            if hasattr(self, "_frozen_lake_slippery"):
                game_config_overrides["is_slippery"] = self._frozen_lake_slippery.isChecked()

            # Additional configuration only for FrozenLake-v2
            if game_id == "FrozenLake-v2":
                if hasattr(self, "_frozen_lake_height"):
                    game_config_overrides["grid_height"] = self._frozen_lake_height.value()
                if hasattr(self, "_frozen_lake_width"):
                    game_config_overrides["grid_width"] = self._frozen_lake_width.value()
                # Collect start position as tuple
                if hasattr(self, "_frozen_lake_start_row") and hasattr(self, "_frozen_lake_start_col"):
                    start_row = self._frozen_lake_start_row.value()
                    start_col = self._frozen_lake_start_col.value()
                    game_config_overrides["start_position"] = (start_row, start_col)
                # Collect goal position as tuple
                if hasattr(self, "_frozen_lake_goal_row") and hasattr(self, "_frozen_lake_goal_col"):
                    goal_row = self._frozen_lake_goal_row.value()
                    goal_col = self._frozen_lake_goal_col.value()
                    game_config_overrides["goal_position"] = (goal_row, goal_col)
                # Collect hole count
                if hasattr(self, "_frozen_lake_hole_count"):
                    game_config_overrides["hole_count"] = self._frozen_lake_hole_count.value()
                # Collect random holes flag
                if hasattr(self, "_frozen_lake_random_holes"):
                    game_config_overrides["random_holes"] = self._frozen_lake_random_holes.isChecked()
        elif game_id == "CliffWalking-v1":
            if hasattr(self, "_cliff_walking_slippery"):
                game_config_overrides["is_slippery"] = self._cliff_walking_slippery.isChecked()

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{game_id.lower()}-{algorithm.split()[0].lower()}-{timestamp}"

        # Get fast training mode flag
        fast_training_mode = self._fast_training_checkbox.isChecked()

        tensorboard_enabled = False
        wandb_enabled = False
        tensorboard_checkbox = getattr(self, "_tensorboard_checkbox", None)
        wandb_checkbox = getattr(self, "_wandb_checkbox", None)
        if fast_training_mode and isinstance(tensorboard_checkbox, QtWidgets.QCheckBox):
            tensorboard_enabled = tensorboard_checkbox.isChecked()
        if fast_training_mode and isinstance(wandb_checkbox, QtWidgets.QCheckBox):
            wandb_enabled = wandb_checkbox.isChecked()

        wandb_project = ""
        wandb_entity = ""
        wandb_run_name = ""
        wandb_api_key = ""
        wandb_email = ""
        wandb_http_proxy = ""
        wandb_https_proxy = ""
        wandb_use_vpn_proxy = False
        if isinstance(getattr(self, "_wandb_project_input", None), QtWidgets.QLineEdit):
            wandb_project = self._wandb_project_input.text().strip()
        if isinstance(getattr(self, "_wandb_entity_input", None), QtWidgets.QLineEdit):
            wandb_entity = self._wandb_entity_input.text().strip()
        if isinstance(getattr(self, "_wandb_run_name_input", None), QtWidgets.QLineEdit):
            wandb_run_name = self._wandb_run_name_input.text().strip()
        if isinstance(getattr(self, "_wandb_api_key_input", None), QtWidgets.QLineEdit):
            wandb_api_key = self._wandb_api_key_input.text().strip()
        if isinstance(getattr(self, "_wandb_email_input", None), QtWidgets.QLineEdit):
            wandb_email = self._wandb_email_input.text().strip()
        if isinstance(getattr(self, "_wandb_use_vpn_checkbox", None), QtWidgets.QCheckBox):
            wandb_use_vpn_proxy = self._wandb_use_vpn_checkbox.isChecked()
        raw_http_proxy = ""
        raw_https_proxy = ""
        if isinstance(getattr(self, "_wandb_http_proxy_input", None), QtWidgets.QLineEdit):
            raw_http_proxy = self._wandb_http_proxy_input.text().strip()
        if isinstance(getattr(self, "_wandb_https_proxy_input", None), QtWidgets.QLineEdit):
            raw_https_proxy = self._wandb_https_proxy_input.text().strip()
        if wandb_use_vpn_proxy:
            wandb_http_proxy = raw_http_proxy or os.environ.get("WANDB_VPN_HTTP_PROXY", "").strip()
            wandb_https_proxy = raw_https_proxy or os.environ.get("WANDB_VPN_HTTPS_PROXY", "").strip()
        else:
            wandb_http_proxy = ""
            wandb_https_proxy = ""

        # Get training telemetry throttle from slider
        # This controls how fast telemetry data is collected and written to database
        # Affects: Telemetry Recent Episodes and Telemetry Recent Steps table population speed
        training_telemetry_throttle = self._training_telemetry_throttle_slider.value()

        # Get UI rendering throttle from slider
        # This controls how often the visual grid display updates on screen
        ui_rendering_throttle = self._ui_rendering_throttle_slider.value()

        # Get render delay from slider
        render_delay_ms = self._render_delay_slider.value()

        live_rendering_enabled = not self._disable_live_render_checkbox.isChecked()

        # This controls the artificial delay between training steps for human observation
        # Slider value (0-100) maps to delay in seconds (0.0 to 1.0)
        ui_training_speed_value = self._ui_training_speed_slider.value()
        step_delay = ui_training_speed_value / 100.0  # Convert to seconds (0.0 to 1.0)

        # Get telemetry buffer size from spin box
        # This controls the in-memory ring buffer size for durable telemetry persistence
        telemetry_buffer_size = self._telemetry_buffer_spin.value()

        # Get episode buffer size from spin box
        # This controls the in-memory ring buffer size for episodes in durable storage
        episode_buffer_size = self._episode_buffer_spin.value()
        
        # Calculate hub buffer size (shared telemetry hub buffer)
        # Hub needs to be larger than UI buffer since it feeds multiple consumers
        # Use 2x the UI buffer size or the total expected steps, whichever is larger
        total_expected_steps = max_episodes * max_steps
        hub_buffer_size = max(
            telemetry_buffer_size * 2,  # At least 2x UI buffer
            min(total_expected_steps, 50000)  # Cap at 50k to prevent excessive memory
        )

        # Fast training mode completely disables UI rendering and telemetry buffers
        if fast_training_mode:
            live_rendering_enabled = False
            ui_rendering_throttle = 1
            render_delay_ms = 0
            ui_training_speed_value = 0
            step_delay = 0.0
            telemetry_buffer_size = 0
            episode_buffer_size = 0
            hub_buffer_size = 0
            training_telemetry_throttle = TRAINING_TELEMETRY_THROTTLE_MIN

        step_delay_ms = int(round(step_delay * 1000))
        env_render_delay_ms = render_delay_ms if live_rendering_enabled else 0

        ui_path_settings = {
            "live_rendering_enabled": live_rendering_enabled,
            "ui_rendering_throttle": ui_rendering_throttle,
            "render_delay_ms": render_delay_ms,
            "step_delay_ms": step_delay_ms,
            "worker_id": worker_id,
            "headless_only": fast_training_mode,
        }

        telemetry_path_settings = {
            "training_telemetry_throttle": training_telemetry_throttle,
            "telemetry_buffer_size": telemetry_buffer_size,
            "episode_buffer_size": episode_buffer_size,
            "hub_buffer_size": hub_buffer_size,  # Add hub buffer size to config
            "disabled": fast_training_mode,
        }

        self.log_constant(
            LOG_UI_TRAIN_FORM_UI_PATH,
            extra={
                "run_name": run_name,
                "agent_type": agent_type,
                "live_rendering_enabled": ui_path_settings["live_rendering_enabled"],
                "ui_rendering_throttle": ui_path_settings["ui_rendering_throttle"],
                "render_delay_ms": ui_path_settings["render_delay_ms"],
                "step_delay_ms": ui_path_settings["step_delay_ms"],
                "worker_id": worker_id,
            },
        )
        self.log_constant(
            LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
            extra={
                "run_name": run_name,
                "agent_type": agent_type,
                "training_telemetry_throttle": telemetry_path_settings["training_telemetry_throttle"],
                "telemetry_buffer_size": telemetry_path_settings["telemetry_buffer_size"],
                "episode_buffer_size": telemetry_path_settings["episode_buffer_size"],
                "hub_buffer_size": telemetry_path_settings["hub_buffer_size"],
            },
        )

        environment = {
            "GYM_ENV_ID": game_id,
            "TRAIN_SEED": str(seed),
            "TRAIN_MAX_EPISODES": str(max_episodes),
            "TRAIN_ALGORITHM": algorithm,
            "TRAIN_LEARNING_RATE": f"{learning_rate}",
            "TRAIN_DISCOUNT": f"{gamma}",
            "TRAIN_EPSILON_DECAY": f"{epsilon_decay}",
            "TRAIN_AGENT_ID": worker_agent_id,
            # Training telemetry throttle: controls data collection speed
            "TRAINING_TELEMETRY_THROTTLE": str(training_telemetry_throttle),
            # UI rendering throttle: controls visual grid update speed (backward compatible name)
            "TELEMETRY_SAMPLING_INTERVAL": str(ui_rendering_throttle),
            "UI_LIVE_RENDERING_ENABLED": "1" if live_rendering_enabled else "0",
            "UI_RENDER_DELAY_MS": str(env_render_delay_ms),
            "WORKER_ID": worker_id,
            "DISABLE_TELEMETRY": "1" if fast_training_mode else "0",
            "TRACK_TENSORBOARD": "1" if tensorboard_enabled else "0",
            "TRACK_WANDB": "1" if wandb_enabled else "0",
        }
        if wandb_api_key:
            environment["WANDB_API_KEY"] = wandb_api_key
        if wandb_email:
            environment["WANDB_EMAIL"] = wandb_email
        if wandb_use_vpn_proxy and wandb_http_proxy:
            environment["WANDB_HTTP_PROXY"] = wandb_http_proxy
            environment["HTTP_PROXY"] = wandb_http_proxy
            environment["http_proxy"] = wandb_http_proxy
        if wandb_use_vpn_proxy and wandb_https_proxy:
            environment["WANDB_HTTPS_PROXY"] = wandb_https_proxy
            environment["HTTPS_PROXY"] = wandb_https_proxy
            environment["https_proxy"] = wandb_https_proxy

        # Add BDI-specific environment variables if enabled
        if bdi_enabled:
            environment["BDI_ENABLED"] = "1"
            environment["BDI_JID"] = bdi_config["jid"]
            environment["BDI_PASSWORD"] = bdi_config["password"]
            if "asl_file" in bdi_config:
                environment["BDI_ASL_FILE"] = bdi_config["asl_file"]

        metadata: Dict[str, Any] = {
            "ui": {
                "algorithm": algorithm,
                "agent_type": agent_type,
                "training_telemetry_throttle": training_telemetry_throttle,
                "ui_rendering_throttle": ui_rendering_throttle,
                "render_delay_ms": render_delay_ms,
                "live_rendering_enabled": live_rendering_enabled,
                "disable_telemetry": fast_training_mode,
                "worker_id": worker_id,
                "ui_training_speed_ms": step_delay_ms,
                "telemetry_buffer_size": telemetry_buffer_size,
                "episode_buffer_size": episode_buffer_size,
                "schema_id": schema_id,
                "schema_version": schema_version,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "max_episodes": max_episodes,
                    "seed": seed,
                    "max_steps_per_episode": max_steps,
                },
                "path_config": {
                    "ui_only": ui_path_settings,
                    "telemetry_durable": telemetry_path_settings,
                },
            },
            "worker": {
                "module": "spade_bdi_worker.worker",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "agent_id": worker_agent_id,
                "agent_type": agent_type,
                "worker_id": worker_id,
                "bdi_enabled": bdi_enabled,
                "bdi_config": bdi_config,
                "schema_id": schema_id,
                "schema_version": schema_version,
                "config": {
                    "run_id": run_name,
                    "game_id": game_id,
                    "seed": seed,
                    "max_episodes": max_episodes,
                    "max_steps_per_episode": max_steps,
                    "policy_strategy": "train_and_save",
                    "policy_path": None,
                    "agent_id": worker_agent_id,
                    "capture_video": False,
                    "headless": True,
                    "step_delay": step_delay,  # Delay between training steps for real-time observation
                    "game_config": game_config_overrides,  # Game-specific configuration
                    "telemetry_buffer_size": telemetry_buffer_size,  # In-memory ring buffer size for steps
                    "episode_buffer_size": episode_buffer_size,  # In-memory ring buffer size for episodes
                    "schema_id": schema_id,
                    "schema_version": schema_version,
                    "schema_definition": schema_definition,
                    "extra": {
                        "algorithm": algorithm,
                        "learning_rate": learning_rate,
                        "gamma": gamma,
                        "epsilon_decay": epsilon_decay,
                        "disable_telemetry": fast_training_mode,
                        "track_tensorboard": tensorboard_enabled,
                        "track_wandb": wandb_enabled,
                        "wandb_use_vpn_proxy": wandb_use_vpn_proxy,
                        **({"wandb_project_name": wandb_project} if wandb_project else {}),
                        **({"wandb_entity": wandb_entity} if wandb_entity else {}),
                        **({"wandb_run_name": wandb_run_name} if wandb_run_name else {}),
                        **({"wandb_email": wandb_email} if wandb_email else {}),
                        **({"wandb_api_key": wandb_api_key} if wandb_api_key else {}),
                        **({"wandb_http_proxy": wandb_http_proxy} if wandb_use_vpn_proxy and wandb_http_proxy else {}),
                        **({"wandb_https_proxy": wandb_https_proxy} if wandb_use_vpn_proxy and wandb_https_proxy else {}),
                    },
                    "path_config": {
                        "ui_only": ui_path_settings,
                        "telemetry_durable": telemetry_path_settings,
                    },
                    "worker_id": worker_id,
                },
            },
        }

        tensorboard_relative = f"var/trainer/runs/{run_name}/tensorboard"
        metadata["artifacts"] = {
            "tensorboard": {
                "enabled": tensorboard_enabled,
                "relative_path": tensorboard_relative,
            },
            "wandb": {
                "enabled": wandb_enabled,
                "run_path": None,
                "use_vpn_proxy": wandb_use_vpn_proxy,
                "http_proxy": wandb_http_proxy or None,
                "https_proxy": wandb_https_proxy or None,
            },
        }

        config: dict[str, Any] = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": [
                "-m",
                "spade_bdi_worker.worker",
            ],
            "environment": environment,
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "gpus": {
                    "requested": 0,
                    "mandatory": False,
                },
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
            "metadata": metadata,
        }

        return config

    def get_config(self) -> Optional[dict]:
        """Return the configured training parameters."""
        return self._selected_config

    def _default_agent_id(self) -> str:
        return f"agent_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _normalize_agent_id(value: str) -> str:
        """Normalize agent identifier to safe slug (letters, digits, underscore, hyphen)."""
        if not value:
            return ""
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug.lower()

    def _default_worker_id(self) -> str:
        return "000001"

    @staticmethod
    def _normalize_worker_id(value: str) -> str:
        if not value:
            return "000001"
        slug = re.sub(r"[^A-Za-z0-9_-]+", "", value.strip())
        if not slug:
            return "000001"
        slug = slug[:WORKER_ID_WIDTH]
        if slug.isdigit():
            return slug.zfill(WORKER_ID_WIDTH)
        return slug.lower()


def _deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))  # deep copy via JSON for primitives

    def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                _merge(target[key], value)
            else:
                target[key] = value

    _merge(result, overrides)
    return result


__all__ = ["SpadeBdiTrainForm"]
