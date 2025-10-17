"""Dialog for configuring and submitting headless training runs."""

from __future__ import annotations

import json
from datetime import datetime
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
        self.setWindowTitle("Train Agent")
        self.resize(500, 400)

        self._selected_config: Optional[dict[str, Any]] = None

        self._build_ui()
        if default_game is not None:
            idx = self._game_combo.findText(default_game.value)
            if idx >= 0:
                self._game_combo.setCurrentIndex(idx)

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
            "CliffWalking-v0",
            "LunarLander-v3",
            "CartPole-v1",
        ])
        form.addRow("Environment:", self._game_combo)
        
        # Algorithm selection
        self._algorithm_combo = QtWidgets.QComboBox(self)
        self._algorithm_combo.addItems([
            "Q-Learning",
            "DQN (future)",
            "PPO (future)",
            "A2C (future)",
        ])
        form.addRow("Algorithm:", self._algorithm_combo)
        
        # Episodes
        self._episodes_spin = QtWidgets.QSpinBox(self)
        self._episodes_spin.setRange(1, 10000)
        self._episodes_spin.setValue(100)
        form.addRow("Episodes:", self._episodes_spin)
        
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
        
        layout.addLayout(form)
        
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
        env_id = self._game_combo.currentText()
        max_episodes = self._episodes_spin.value()
        seed = self._seed_spin.value()
        learning_rate = float(self._lr_edit.text())
        gamma = float(self._gamma_edit.text())
        epsilon_decay = float(self._epsilon_edit.text())

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{env_id.lower()}-{algorithm.split()[0].lower()}-{timestamp}"

        environment = {
            "GYM_ENV_ID": env_id,
            "TRAIN_SEED": str(seed),
            "TRAIN_MAX_EPISODES": str(max_episodes),
            "TRAIN_ALGORITHM": algorithm,
            "TRAIN_LEARNING_RATE": f"{learning_rate}",
            "TRAIN_DISCOUNT": f"{gamma}",
            "TRAIN_EPSILON_DECAY": f"{epsilon_decay}",
        }

        metadata: Dict[str, Any] = {
            "ui": {
                "algorithm": algorithm,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "max_episodes": max_episodes,
                    "seed": seed,
                },
                "worker": {
                    "module": "spadeBDI_RL_refactored.worker",
                    "class": "HeadlessTrainer",
                },
            }
        }

        config: dict[str, Any] = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": [
                "-m",
                "spadeBDI_RL_refactored.worker",
                "--run-name",
                run_name,
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
