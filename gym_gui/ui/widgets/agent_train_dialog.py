"""Dialog for configuring and submitting headless training runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional

from jsonschema import ValidationError
from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import GameId
from gym_gui.services.trainer import validate_train_run_config


class TrainAgentDialog(QtWidgets.QDialog):
    """Dialog to configure and submit a training run."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        default_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Agent Train Form")
        # Set larger initial size to accommodate all content without overlapping
        # Dialog will resize dynamically when game configuration changes
        self.resize(600, 900)

        self._logger = logging.getLogger("gym_gui.ui.train_agent_dialog")
        self._selected_config: Optional[dict[str, Any]] = None

        self._build_ui()
        if default_game is not None:
            idx = self._game_combo.findText(default_game.value)
            if idx >= 0:
                self._game_combo.setCurrentIndex(idx)

        self._logger.info(
            "TrainAgentDialog opened",
            extra={"default_game": getattr(default_game, "value", None)},
        )

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Form layout for configuration
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

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
        form.addRow("Environment:", self._game_combo)
        self._game_combo.currentTextChanged.connect(self._on_game_selection_changed)
        
        # Algorithm selection
        self._algorithm_combo = QtWidgets.QComboBox(self)
        self._algorithm_combo.addItems([
            "Q-Learning",
            "DQN (future)",
            "PPO (future)",
            "A2C (future)",
        ])
        form.addRow("Algorithm:", self._algorithm_combo)

        # Agent identifier input
        self._agent_id_edit = QtWidgets.QLineEdit(self)
        self._agent_id_edit.setPlaceholderText("e.g. agent_run_20251017")
        self._agent_id_edit.setText(self._default_agent_id())
        form.addRow("Agent ID:", self._agent_id_edit)

        # Episodes
        self._episodes_spin = QtWidgets.QSpinBox(self)
        self._episodes_spin.setRange(1, 10000)
        self._episodes_spin.setValue(1000)
        form.addRow("Episodes:", self._episodes_spin)

        self._max_steps_spin = QtWidgets.QSpinBox(self)
        self._max_steps_spin.setRange(1, 10000)
        self._max_steps_spin.setValue(100)
        form.addRow("Max Steps / Episode:", self._max_steps_spin)
        
        # Seed
        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        form.addRow("Random Seed:", self._seed_spin)
        
        # Learning rate
        self._lr_edit = QtWidgets.QLineEdit("0.1")
        form.addRow("Learning Rate:", self._lr_edit)
        
        # Gamma (discount)
        self._gamma_edit = QtWidgets.QLineEdit("0.99")
        form.addRow("Discount (γ):", self._gamma_edit)
        
        # Epsilon decay
        self._epsilon_edit = QtWidgets.QLineEdit("0.995")
        form.addRow("Epsilon Decay:", self._epsilon_edit)

        # UI Rendering Throttle (controls how often the visual display updates)
        # Note: This controls UI rendering frequency, NOT database writes.
        # All telemetry events are always written to the database for complete training records.
        # The database uses efficient batching (batch_size=64) and WAL mode for performance.
        telemetry_layout = QtWidgets.QHBoxLayout()
        self._telemetry_sampling_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._telemetry_sampling_slider.setRange(1, 10)
        self._telemetry_sampling_slider.setValue(2)  # Default: render every 2nd step (50% UI update rate)
        self._telemetry_sampling_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._telemetry_sampling_slider.setTickInterval(1)
        self._telemetry_sampling_slider.valueChanged.connect(self._on_telemetry_sampling_changed)
        telemetry_layout.addWidget(self._telemetry_sampling_slider)

        self._telemetry_sampling_label = QtWidgets.QLabel("50%")
        self._telemetry_sampling_label.setMinimumWidth(40)
        telemetry_layout.addWidget(self._telemetry_sampling_label)

        form.addRow("UI Rendering Throttle:", telemetry_layout)

        # Training Speed Control (delays between steps for real-time observation)
        # This introduces actual delays in the training loop, allowing users to observe
        # the agent's actions in real-time through the Live Rendering panel
        speed_layout = QtWidgets.QHBoxLayout()
        self._training_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._training_speed_slider.setRange(0, 100)  # 0 = no delay, 100 = 1 second delay per step
        self._training_speed_slider.setValue(0)  # Default: no delay (fast training)
        self._training_speed_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self._training_speed_slider.setTickInterval(10)
        self._training_speed_slider.valueChanged.connect(self._on_training_speed_changed)
        speed_layout.addWidget(self._training_speed_slider)

        self._training_speed_label = QtWidgets.QLabel("0ms")
        self._training_speed_label.setMinimumWidth(50)
        speed_layout.addWidget(self._training_speed_label)

        form.addRow("Training Speed (Delay):", speed_layout)

        # Telemetry Buffer Size (in-memory ring buffer for UI display)
        # Note: All steps are persisted to SQLite database; this only controls UI display buffer
        buffer_layout = QtWidgets.QHBoxLayout()
        self._telemetry_buffer_spin = QtWidgets.QSpinBox(self)
        self._telemetry_buffer_spin.setRange(256, 10000)
        self._telemetry_buffer_spin.setValue(512)
        self._telemetry_buffer_spin.setSingleStep(256)
        self._telemetry_buffer_spin.setToolTip(
            "Ring buffer size for in-memory telemetry. All steps are persisted to SQLite database; "
            "UI only displays the last N steps in memory. Increase for longer training sessions."
        )
        buffer_layout.addWidget(self._telemetry_buffer_spin)

        self._telemetry_buffer_label = QtWidgets.QLabel("steps")
        buffer_layout.addWidget(self._telemetry_buffer_label)
        buffer_layout.addStretch()

        form.addRow("Telemetry Buffer Size:", buffer_layout)

        # Episode Buffer Size (in-memory ring buffer for UI display)
        # Note: All episodes are persisted to SQLite database; this only controls UI display buffer
        episode_buffer_layout = QtWidgets.QHBoxLayout()
        self._episode_buffer_spin = QtWidgets.QSpinBox(self)
        self._episode_buffer_spin.setRange(10, 1000)
        self._episode_buffer_spin.setValue(100)
        self._episode_buffer_spin.setSingleStep(10)
        self._episode_buffer_spin.setToolTip(
            "Ring buffer size for in-memory episodes. All episodes are persisted to SQLite database; "
            "UI only displays the last N episodes in memory. Increase for longer training sessions."
        )
        episode_buffer_layout.addWidget(self._episode_buffer_spin)

        self._episode_buffer_label = QtWidgets.QLabel("episodes")
        episode_buffer_layout.addWidget(self._episode_buffer_label)
        episode_buffer_layout.addStretch()

        form.addRow("Episode Buffer Size:", episode_buffer_layout)

        layout.addLayout(form)

        # Game-specific configuration (dynamic) - add to main layout for proper display
        self._game_config_group = QtWidgets.QGroupBox("Game Configuration", self)
        self._game_config_layout = QtWidgets.QFormLayout(self._game_config_group)
        self._game_config_widgets: Dict[str, QtWidgets.QWidget] = {}
        self._build_game_config_widgets()
        layout.addWidget(self._game_config_group)

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
        
        layout.addWidget(self._bdi_group)
        
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
        layout.addWidget(self._advanced_group)
        
        # Info label
        info_label = QtWidgets.QLabel(
            '<i>Training will run in the background. '
            'Watch progress in the "Live Telemetry" tab.</i>'
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

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

    def _on_telemetry_sampling_changed(self, value: int) -> None:
        """Handle UI rendering throttle slider change.

        This controls how often the visual display updates during training.
        All telemetry events are always written to the database (no data loss).

        Slider value (1-10) to UI update percentage:
        - value=1 means 100% (update every step)
        - value=2 means 50% (update every 2nd step)
        - value=10 means 10% (update every 10th step)
        """
        percentage = int(100 / value)
        self._telemetry_sampling_label.setText(f"{percentage}%")

    def _on_training_speed_changed(self, value: int) -> None:
        """Handle training speed slider change.

        This controls the delay between training steps for real-time observation.
        Slider value (0-100) to delay in milliseconds:
        - value=0 means 0ms (no delay, fast training)
        - value=50 means 500ms (0.5 second delay per step)
        - value=100 means 1000ms (1 second delay per step)
        """
        delay_ms = value * 10  # Convert slider value to milliseconds
        self._training_speed_label.setText(f"{delay_ms}ms")

    def _on_accept(self) -> None:
        """Validate and build config before accepting."""
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
                self._logger.info(
                    "TrainAgentDialog accepted",
                    extra={
                        "run_name": self._selected_config.get("run_name"),
                        "agent_id": worker_meta.get("agent_id"),
                        "episodes": config_meta.get("max_episodes"),
                        "max_steps": config_meta.get("max_steps_per_episode"),
                    },
                )

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
        learning_rate = float(self._lr_edit.text())
        gamma = float(self._gamma_edit.text())
        epsilon_decay = float(self._epsilon_edit.text())
        agent_id_input = self._agent_id_edit.text().strip()
        worker_agent_id = self._normalize_agent_id(agent_id_input) or self._default_agent_id()

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
            
            self._logger.info(
                "BDI Agent configuration collected",
                extra={
                    "agent_type": agent_type,
                    "bdi_jid": bdi_jid,
                    "bdi_password": "[***]",
                    "bdi_asl_file": bdi_asl_file if bdi_asl_file else "(not provided)",
                    "config_keys": list(bdi_config.keys()),
                },
            )
        else:
            self._logger.info(
                "Headless Agent configuration (BDI disabled)",
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

        # Get UI rendering throttle from slider
        # This controls how often the visual display updates (not database writes)
        # All telemetry events are always written to the database for complete training records
        ui_rendering_throttle = self._telemetry_sampling_slider.value()

        # Get training speed (step delay) from slider
        # This controls the delay between training steps for real-time observation
        # Slider value (0-100) maps to delay in seconds (0.0 to 1.0)
        training_speed_value = self._training_speed_slider.value()
        step_delay = training_speed_value / 100.0  # Convert to seconds (0.0 to 1.0)

        # Get telemetry buffer size from spin box
        # This controls the in-memory ring buffer size for UI display
        telemetry_buffer_size = self._telemetry_buffer_spin.value()

        # Get episode buffer size from spin box
        # This controls the in-memory ring buffer size for episodes in UI display
        episode_buffer_size = self._episode_buffer_spin.value()

        environment = {
            "GYM_ENV_ID": game_id,
            "TRAIN_SEED": str(seed),
            "TRAIN_MAX_EPISODES": str(max_episodes),
            "TRAIN_ALGORITHM": algorithm,
            "TRAIN_LEARNING_RATE": f"{learning_rate}",
            "TRAIN_DISCOUNT": f"{gamma}",
            "TRAIN_EPSILON_DECAY": f"{epsilon_decay}",
            "TRAIN_AGENT_ID": worker_agent_id,
            # Note: TELEMETRY_SAMPLING_INTERVAL is kept for backward compatibility
            # but it now controls UI rendering frequency, not database writes
            "TELEMETRY_SAMPLING_INTERVAL": str(ui_rendering_throttle),
        }
        
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
                "ui_rendering_throttle": ui_rendering_throttle,
                "telemetry_buffer_size": telemetry_buffer_size,
                "episode_buffer_size": episode_buffer_size,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "max_episodes": max_episodes,
                    "seed": seed,
                    "max_steps_per_episode": max_steps,
                },
            },
            "worker": {
                "module": "spade_bdi_rl.worker",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "agent_id": worker_agent_id,
                "agent_type": agent_type,
                "bdi_enabled": bdi_enabled,
                "bdi_config": bdi_config,
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
                    "extra": {
                        "algorithm": algorithm,
                        "learning_rate": learning_rate,
                        "gamma": gamma,
                        "epsilon_decay": epsilon_decay,
                    },
                },
            },
        }

        config: dict[str, Any] = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": [
                "-m",
                "spade_bdi_rl.worker",
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


__all__ = ["TrainAgentDialog"]
