"""Dynamic worker-specific configuration panel.

Step 3 of the Unified Flow: Configure worker-specific parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from .agent_config_table import AgentRowConfig

_LOGGER = logging.getLogger(__name__)


# Worker configuration schemas (simplified)
WORKER_CONFIG_SCHEMAS = {
    "local": {
        "display_name": "Local Execution",
        "description": "Run in main process without external worker",
        "fields": [
            {
                "name": "render_mode",
                "label": "Render Mode",
                "type": "choice",
                "choices": ["human", "rgb_array", "ansi", "none"],
                "default": "human",
            },
            {
                "name": "record_video",
                "label": "Record Video",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "cleanrl": {
        "display_name": "CleanRL Worker",
        "description": "Single-agent RL training with CleanRL algorithms",
        "fields": [
            {
                "name": "algorithm",
                "label": "Algorithm",
                "type": "choice",
                "choices": ["PPO", "DQN", "A2C", "SAC", "TD3", "DDPG"],
                "default": "PPO",
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "float",
                "default": 0.0003,
                "min": 0.000001,
                "max": 1.0,
            },
            {
                "name": "total_timesteps",
                "label": "Total Timesteps",
                "type": "int",
                "default": 100000,
                "min": 1000,
                "max": 10000000,
            },
            {
                "name": "num_envs",
                "label": "Parallel Envs",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 64,
            },
            {
                "name": "capture_video",
                "label": "Capture Video",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "rllib": {
        "display_name": "Ray RLlib Worker",
        "description": "Distributed RL training with Ray RLlib",
        "fields": [
            {
                "name": "algorithm",
                "label": "Algorithm",
                "type": "choice",
                "choices": ["PPO", "DQN", "A2C", "IMPALA", "APPO", "SAC"],
                "default": "PPO",
            },
            {
                "name": "num_workers",
                "label": "Num Workers",
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 64,
            },
            {
                "name": "num_envs_per_worker",
                "label": "Envs per Worker",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 16,
            },
            {
                "name": "framework",
                "label": "Framework",
                "type": "choice",
                "choices": ["torch", "tf2"],
                "default": "torch",
            },
        ],
    },
    "xuance": {
        "display_name": "XuanCe Worker",
        "description": "Multi-agent RL training with XuanCe MARL algorithms",
        "fields": [
            {
                "name": "algorithm",
                "label": "Algorithm",
                "type": "choice",
                "choices": ["MAPPO", "MADDPG", "QMIX", "VDN", "COMA", "IPPO", "IQL"],
                "default": "MAPPO",
            },
            {
                "name": "learning_rate",
                "label": "Learning Rate",
                "type": "float",
                "default": 0.0005,
                "min": 0.000001,
                "max": 1.0,
            },
            {
                "name": "batch_size",
                "label": "Batch Size",
                "type": "int",
                "default": 256,
                "min": 32,
                "max": 4096,
            },
            {
                "name": "backend",
                "label": "Backend",
                "type": "choice",
                "choices": ["torch", "tensorflow", "mindspore"],
                "default": "torch",
            },
        ],
    },
    "llm": {
        "display_name": "LLM Worker",
        "description": "Language model-based decision making",
        "fields": [
            {
                "name": "model",
                "label": "Model",
                "type": "choice",
                "choices": ["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-3", "ollama-local"],
                "default": "gpt-4",
            },
            {
                "name": "temperature",
                "label": "Temperature",
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 2.0,
            },
            {
                "name": "max_tokens",
                "label": "Max Tokens",
                "type": "int",
                "default": 256,
                "min": 16,
                "max": 4096,
            },
            {
                "name": "system_prompt",
                "label": "System Prompt",
                "type": "text",
                "default": "You are an RL agent making decisions in an environment.",
            },
        ],
    },
    "jason": {
        "display_name": "Jason BDI Worker",
        "description": "Belief-Desire-Intention agent with AgentSpeak",
        "fields": [
            {
                "name": "agent_file",
                "label": "Agent File (.asl)",
                "type": "text",
                "default": "agent.asl",
            },
            {
                "name": "mas_file",
                "label": "MAS File (.mas2j)",
                "type": "text",
                "default": "project.mas2j",
            },
            {
                "name": "debug_mode",
                "label": "Debug Mode",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "spade_bdi": {
        "display_name": "SPADE BDI Worker",
        "description": "Python-based BDI agents with SPADE framework",
        "fields": [
            {
                "name": "xmpp_server",
                "label": "XMPP Server",
                "type": "text",
                "default": "localhost",
            },
            {
                "name": "agent_jid",
                "label": "Agent JID",
                "type": "text",
                "default": "agent@localhost",
            },
            {
                "name": "debug_mode",
                "label": "Debug Mode",
                "type": "bool",
                "default": False,
            },
        ],
    },
}


class WorkerConfigPanel(QtWidgets.QGroupBox):
    """Step 3: Dynamic worker-specific configuration.

    Generates configuration forms based on the selected workers.
    Shows separate sections for each unique worker type in use.

    Signals:
        config_changed: Emitted when any configuration changes

    Example:
        panel = WorkerConfigPanel()
        panel.update_from_bindings(bindings)
        panel.config_changed.connect(on_config_change)
    """

    # Signals
    config_changed = pyqtSignal(dict)  # Dict[worker_id, Dict[str, Any]]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Step 3: Worker Configuration", parent)
        self._active_workers: Dict[str, List[str]] = {}  # worker_id -> [agent_ids]
        self._config_widgets: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Info label
        info = QtWidgets.QLabel(
            "Configure worker-specific parameters. "
            "Each worker type appears once, even if used by multiple agents."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Scrollable area for worker configs
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(150)

        self._configs_container = QtWidgets.QWidget()
        self._configs_layout = QtWidgets.QVBoxLayout(self._configs_container)
        self._configs_layout.setSpacing(12)
        self._configs_layout.setContentsMargins(0, 0, 0, 0)
        self._configs_layout.addStretch(1)

        scroll.setWidget(self._configs_container)
        layout.addWidget(scroll)

        # Placeholder when no workers need config
        self._placeholder = QtWidgets.QLabel(
            "<i>Select workers that need configuration (e.g., CleanRL, RLlib).</i>"
        )
        self._placeholder.setStyleSheet("color: #888; padding: 20px;")
        layout.addWidget(self._placeholder)

    def update_from_bindings(self, bindings: Dict[str, AgentRowConfig]) -> None:
        """Update worker config panels based on agent bindings.

        Args:
            bindings: Dictionary mapping agent_id to AgentRowConfig.
        """
        # Group agents by worker
        workers_to_agents: Dict[str, List[str]] = {}
        for agent_id, config in bindings.items():
            worker_id = config.worker_id
            if worker_id not in workers_to_agents:
                workers_to_agents[worker_id] = []
            workers_to_agents[worker_id].append(agent_id)

        # Filter to workers that have config schemas (exclude "local")
        configurable_workers = {
            wid: agents
            for wid, agents in workers_to_agents.items()
            if wid in WORKER_CONFIG_SCHEMAS
        }

        # Check if workers changed
        if configurable_workers == self._active_workers:
            return  # No change needed

        self._active_workers = configurable_workers
        self._rebuild_config_panels()

    def _rebuild_config_panels(self) -> None:
        """Rebuild the configuration panels for active workers."""
        # Clear existing panels
        self._clear_panels()

        if not self._active_workers:
            self._placeholder.show()
            return

        self._placeholder.hide()

        # Create panel for each worker type
        for worker_id, agent_ids in self._active_workers.items():
            self._create_worker_panel(worker_id, agent_ids)

    def _clear_panels(self) -> None:
        """Clear all worker panels."""
        # Remove all widgets except the stretch
        while self._configs_layout.count() > 1:
            item = self._configs_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                item_layout = item.layout()
                if item_layout is not None:
                    self._clear_layout(item_layout)

        self._config_widgets.clear()

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        """Recursively clear a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                item_layout = item.layout()
                if item_layout is not None:
                    self._clear_layout(item_layout)

    def _create_worker_panel(self, worker_id: str, agent_ids: List[str]) -> None:
        """Create a configuration panel for a worker type."""
        schema = WORKER_CONFIG_SCHEMAS.get(worker_id)
        if not schema:
            return

        # Create frame for this worker
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #f8f8f8; border-radius: 4px; padding: 8px; }"
        )
        frame_layout = QtWidgets.QVBoxLayout(frame)

        # Header with worker name and agent list
        header = QtWidgets.QLabel(
            f"<b>{schema['display_name']}</b> "
            f"<span style='color: #666;'>(used by: {', '.join(agent_ids)})</span>"
        )
        frame_layout.addWidget(header)

        # Initialize config with defaults
        if worker_id not in self._configs:
            self._configs[worker_id] = {}
            for field in schema["fields"]:
                self._configs[worker_id][field["name"]] = field["default"]

        # Create form for each field
        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(6)

        widgets = {}
        for field in schema["fields"]:
            widget = self._create_field_widget(worker_id, field)
            form_layout.addRow(f"{field['label']}:", widget)
            widgets[field["name"]] = widget

        frame_layout.addLayout(form_layout)

        self._config_widgets[worker_id] = widgets

        # Insert before stretch
        insert_idx = self._configs_layout.count() - 1
        self._configs_layout.insertWidget(insert_idx, frame)

    def _create_field_widget(
        self, worker_id: str, field: Dict[str, Any]
    ) -> QtWidgets.QWidget:
        """Create a widget for a config field."""
        field_type = field["type"]
        field_name = field["name"]
        default = field.get("default")

        if field_type == "choice":
            widget = QtWidgets.QComboBox()
            for choice in field["choices"]:
                widget.addItem(choice, choice)
            # Set default
            idx = widget.findData(default)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            widget.currentIndexChanged.connect(
                lambda _, wid=worker_id, fn=field_name: self._on_field_changed(wid, fn)
            )
            return widget

        elif field_type == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(field.get("min", 0), field.get("max", 1000000))
            widget.setValue(default or 0)
            widget.valueChanged.connect(
                lambda _, wid=worker_id, fn=field_name: self._on_field_changed(wid, fn)
            )
            return widget

        elif field_type == "float":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(field.get("min", 0.0), field.get("max", 1.0))
            widget.setDecimals(6)
            widget.setValue(default or 0.0)
            widget.valueChanged.connect(
                lambda _, wid=worker_id, fn=field_name: self._on_field_changed(wid, fn)
            )
            return widget

        elif field_type == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(default or False)
            widget.stateChanged.connect(
                lambda _, wid=worker_id, fn=field_name: self._on_field_changed(wid, fn)
            )
            return widget

        elif field_type == "text":
            widget = QtWidgets.QLineEdit()
            widget.setText(default or "")
            widget.textChanged.connect(
                lambda _, wid=worker_id, fn=field_name: self._on_field_changed(wid, fn)
            )
            return widget

        else:
            # Fallback to text
            widget = QtWidgets.QLineEdit()
            widget.setText(str(default) if default else "")
            return widget

    def _on_field_changed(self, worker_id: str, field_name: str) -> None:
        """Handle a config field change."""
        widgets = self._config_widgets.get(worker_id, {})
        widget = widgets.get(field_name)
        if not widget:
            return

        # Get value based on widget type
        if isinstance(widget, QtWidgets.QComboBox):
            value = widget.currentData()
        elif isinstance(widget, QtWidgets.QSpinBox):
            value = widget.value()
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            value = widget.value()
        elif isinstance(widget, QtWidgets.QCheckBox):
            value = widget.isChecked()
        elif isinstance(widget, QtWidgets.QLineEdit):
            value = widget.text()
        else:
            value = None

        if worker_id not in self._configs:
            self._configs[worker_id] = {}
        self._configs[worker_id][field_name] = value

        self._emit_configs()

    def _emit_configs(self) -> None:
        """Emit the current configurations."""
        self.config_changed.emit(dict(self._configs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all worker configurations.

        Returns:
            Dictionary mapping worker_id to config dict.
        """
        return dict(self._configs)

    def get_config(self, worker_id: str) -> Dict[str, Any]:
        """Get configuration for a specific worker.

        Args:
            worker_id: The worker identifier.

        Returns:
            Configuration dict, or empty dict if not found.
        """
        return self._configs.get(worker_id, {})
