"""Script Experiment Widget - Clean interface for automatic baseline execution."""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from gym_gui.services.operator import OperatorConfig, WorkerAssignment
from gym_gui.services.operator_script_execution_manager import OperatorScriptExecutionManager


_LOGGER = logging.getLogger(__name__)


class ScriptExperimentWidget(QtWidgets.QWidget):
    """Widget for loading and running scripted baseline experiments.

    This is a pure UI widget. Execution logic is handled by OperatorScriptExecutionManager.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._script_path: Optional[Path] = None
        self._operator_configs: List[OperatorConfig] = []
        self._execution_config: Dict[str, Any] = {}

        # Execution manager handles all script execution logic
        self._execution_manager = OperatorScriptExecutionManager(self)

        # Connect to execution manager's progress signals
        self._execution_manager.progress_updated.connect(self._on_progress_updated)
        self._execution_manager.experiment_completed.connect(self._on_experiment_completed)

        self._setup_ui()

    def _setup_ui(self):
        """Build the UI."""
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Script-Based Experiments")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        description = QtWidgets.QLabel(
            "Load and run baseline operator experiments from Python scripts.\n"
            "Scripts define operators, seeds, and episode counts for reproducible experiments."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; padding: 5px;")
        layout.addWidget(description)

        layout.addSpacing(10)

        # Script selection
        script_group = QtWidgets.QGroupBox("Experiment Script")
        script_layout = QtWidgets.QVBoxLayout(script_group)

        self._script_label = QtWidgets.QLabel("No script loaded")
        self._script_label.setStyleSheet("color: gray; font-style: italic;")
        script_layout.addWidget(self._script_label)

        self._browse_button = QtWidgets.QPushButton("Browse and Load Script...")
        self._browse_button.clicked.connect(self._on_browse_clicked)
        script_layout.addWidget(self._browse_button)

        layout.addWidget(script_group)

        # Experiment info
        self._info_group = QtWidgets.QGroupBox("Experiment Configuration")
        info_layout = QtWidgets.QVBoxLayout(self._info_group)

        self._info_text = QtWidgets.QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(120)
        info_layout.addWidget(self._info_text)

        layout.addWidget(self._info_group)
        self._info_group.setVisible(False)

        # Execution settings
        self._settings_group = QtWidgets.QGroupBox("Execution Settings")
        settings_layout = QtWidgets.QGridLayout(self._settings_group)

        # Max episodes
        settings_layout.addWidget(QtWidgets.QLabel("Max episodes:"), 0, 0)
        self._max_episodes_spin = QtWidgets.QSpinBox()
        self._max_episodes_spin.setRange(1, 10000)
        self._max_episodes_spin.setValue(10)
        self._max_episodes_spin.setToolTip("Number of episodes to run (overrides script value)")
        settings_layout.addWidget(self._max_episodes_spin, 0, 1)

        # Max steps per episode
        settings_layout.addWidget(QtWidgets.QLabel("Max steps/episode:"), 0, 2)
        self._max_steps_spin = QtWidgets.QSpinBox()
        self._max_steps_spin.setRange(1, 100000)
        self._max_steps_spin.setValue(500)
        self._max_steps_spin.setToolTip("Max steps before episode truncation (higher = more time to reach goal)")
        settings_layout.addWidget(self._max_steps_spin, 0, 3)

        # Step delay (pacing)
        settings_layout.addWidget(QtWidgets.QLabel("Step delay (ms):"), 1, 0)
        self._step_delay_spin = QtWidgets.QSpinBox()
        self._step_delay_spin.setRange(0, 1000)
        self._step_delay_spin.setValue(50)
        self._step_delay_spin.setSuffix(" ms")
        self._step_delay_spin.setToolTip("0 = fastest (may skip frames), 50 = smooth, 200 = slow-motion")
        self._step_delay_spin.valueChanged.connect(self._on_step_delay_changed)
        settings_layout.addWidget(self._step_delay_spin, 1, 1)

        # Environment mode (Fixed vs Procedural generation)
        settings_layout.addWidget(QtWidgets.QLabel("Environment:"), 1, 2)
        self._env_mode_combo = QtWidgets.QComboBox()
        self._env_mode_combo.addItems(["Procedural", "Fixed"])
        self._env_mode_combo.setCurrentIndex(0)  # Default: Procedural
        self._env_mode_combo.setToolTip(
            "Procedural: different layout each episode (generalization test)\n"
            "Fixed: same layout every episode (isolate agent behavior)"
        )
        settings_layout.addWidget(self._env_mode_combo, 1, 3)

        layout.addWidget(self._settings_group)

        # Progress display
        self._progress_group = QtWidgets.QGroupBox("Execution Progress")
        progress_layout = QtWidgets.QVBoxLayout(self._progress_group)

        self._progress_label = QtWidgets.QLabel("Ready to run")
        self._progress_label.setStyleSheet("font-size: 12pt;")
        progress_layout.addWidget(self._progress_label)

        self._episode_label = QtWidgets.QLabel("")
        progress_layout.addWidget(self._episode_label)

        layout.addWidget(self._progress_group)
        self._progress_group.setVisible(False)

        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()

        self._run_button = QtWidgets.QPushButton("Run Experiment")
        self._run_button.setEnabled(False)
        self._run_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        self._run_button.clicked.connect(self._on_run_clicked)
        button_layout.addWidget(self._run_button)

        self._stop_button = QtWidgets.QPushButton("Stop Experiment")
        self._stop_button.setEnabled(False)
        self._stop_button.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold; padding: 10px;"
        )
        self._stop_button.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self._stop_button)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _on_browse_clicked(self):
        """Open file dialog to select experiment script."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Experiment Script",
            str(Path.cwd() / "experiments" / "operator_configs"),
            "Python Scripts (*.py)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )

        if not file_path:
            return

        self._script_path = Path(file_path)
        self._load_script()

    def _load_script(self):
        """Load and parse the experiment script."""
        if not self._script_path or not self._script_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Script Not Found", f"Script file not found: {self._script_path}"
            )
            return

        try:
            result = self._parse_script(self._script_path)

            if not result.get("success"):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Script Parse Error",
                    f"Failed to parse script:\n\n{result.get('error', 'Unknown error')}"
                )
                return

            self._operator_configs = result["operators"]
            self._execution_config = result.get("execution", {})

            self._script_label.setText(f"Loaded: {self._script_path.name}")
            self._script_label.setStyleSheet("color: green; font-weight: bold;")

            # Populate spinboxes from script values
            num_episodes = self._execution_config.get("num_episodes", 10)
            self._max_episodes_spin.setValue(num_episodes)

            script_max_steps = None
            if self._operator_configs and self._operator_configs[0].max_steps:
                script_max_steps = self._operator_configs[0].max_steps
            self._max_steps_spin.setValue(script_max_steps or 500)

            # Set environment mode from script
            env_mode = self._execution_config.get("environment_mode", "procedural")
            self._env_mode_combo.setCurrentIndex(1 if env_mode == "fixed" else 0)

            # Build info display
            num_operators = len(self._operator_configs)
            seeds = self._execution_config.get("seeds", range(1000, 1000 + num_episodes))
            seed_list = list(seeds)[:num_episodes]

            info_text = (
                f"<b>Operators:</b> {num_operators}<br>"
                f"<b>Episodes:</b> {num_episodes}<br>"
                f"<b>Seeds:</b> {seed_list[0]} - {seed_list[-1]}"
            )

            if script_max_steps:
                info_text += f"<br><b>Max Steps/Episode:</b> {script_max_steps}"

            env_mode_display = self._env_mode_combo.currentText()
            info_text += f"<br><b>Environment Mode:</b> {env_mode_display}"

            info_text += "<br><br><b>Operators:</b><br>"
            for config in self._operator_configs:
                worker_type = list(config.workers.values())[0].worker_type if config.workers else "unknown"
                behavior = ""
                if worker_type == "baseline":
                    behavior = list(config.workers.values())[0].settings.get("behavior", "")
                    behavior = f" ({behavior})"
                info_text += f"  • {config.display_name} [{worker_type}{behavior}]<br>"

            self._info_text.setHtml(info_text)
            self._info_group.setVisible(True)
            self._progress_group.setVisible(True)
            self._run_button.setEnabled(True)

            QtWidgets.QMessageBox.information(
                self,
                "Script Loaded",
                f"Successfully loaded experiment script!\n\n"
                f"{num_operators} operator(s) configured for {num_episodes} episodes.\n\n"
                f"Click 'Run Experiment' to start automatic execution."
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error Loading Script", f"Failed to load script:\n\n{str(e)}"
            )
            _LOGGER.exception("Failed to load experiment script")

    def _parse_script(self, script_path: Path) -> Dict[str, Any]:
        """Parse experiment script by executing in a sandboxed namespace.

        Instead of using importlib (which can hang due to module import machinery),
        we read the file as text and exec() in a controlled namespace.
        Experiment scripts only define dicts/lists, so this is safe and fast.
        """
        try:
            # Read script as text (instant, no import machinery)
            script_text = script_path.read_text(encoding="utf-8")

            # Execute in a sandboxed namespace with only builtins
            namespace: Dict[str, Any] = {"__builtins__": __builtins__}
            exec(compile(script_text, str(script_path), "exec"), namespace)

            operators_data = namespace.get("operators", [])
            execution_config = namespace.get("execution", {})

            if not operators_data:
                return {"success": False, "error": "No 'operators' list found in script"}

            operator_configs = []
            for i, op_data in enumerate(operators_data):
                if not isinstance(op_data, dict):
                    return {"success": False, "error": f"Operator {i} must be a dict"}

                operator_id = op_data.get("id")
                display_name = op_data.get("name")
                workers_data = op_data.get("workers", {})

                if not operator_id or not display_name or not workers_data:
                    return {"success": False, "error": f"Operator {i} missing required fields"}

                env_name = op_data.get("env_name", "minigrid")
                task = op_data.get("task", "MiniGrid-Empty-8x8-v0")
                max_steps = op_data.get("max_steps")

                workers: Dict[str, WorkerAssignment] = {}
                for agent_id, worker_data in workers_data.items():
                    if not isinstance(worker_data, dict):
                        return {"success": False, "error": f"Operator {i}, agent {agent_id}: worker must be dict"}

                    worker_type = worker_data.get("type")
                    if not worker_type:
                        return {"success": False, "error": f"Operator {i}, agent {agent_id}: missing 'type'"}

                    worker_id_map = {
                        "baseline": "operators_worker",
                        "rl": "cleanrl_worker",
                        "llm": "balrog_worker",
                        "vlm": "balrog_worker",
                        "human": "human_worker",
                    }
                    worker_id = worker_id_map.get(worker_type)
                    if not worker_id:
                        return {"success": False, "error": f"Unknown worker type: {worker_type}"}

                    settings = {k: v for k, v in worker_data.items() if k != "type"}
                    workers[agent_id] = WorkerAssignment(
                        worker_id=worker_id, worker_type=worker_type, settings=settings
                    )

                if len(workers) == 1:
                    config = OperatorConfig.single_agent(
                        operator_id=operator_id,
                        display_name=display_name,
                        worker_id=list(workers.values())[0].worker_id,
                        worker_type=list(workers.values())[0].worker_type,
                        env_name=env_name,
                        task=task,
                        settings=list(workers.values())[0].settings,
                        max_steps=max_steps,
                    )
                else:
                    config = OperatorConfig.multi_agent(
                        operator_id=operator_id,
                        display_name=display_name,
                        player_workers=workers,
                        env_name=env_name,
                        task=task,
                        max_steps=max_steps,
                    )

                operator_configs.append(config)

            return {"success": True, "operators": operator_configs, "execution": execution_config}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _on_step_delay_changed(self, value: int):
        """Update execution manager step delay in real-time."""
        self._execution_manager.step_delay_ms = value

    def _on_run_clicked(self):
        """Start automatic experiment execution."""
        if not self._operator_configs:
            return

        self._run_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._browse_button.setEnabled(False)
        self._progress_label.setText("Starting experiment...")

        # Apply UI overrides to execution config
        self._execution_config["step_delay_ms"] = self._step_delay_spin.value()
        self._execution_config["num_episodes"] = self._max_episodes_spin.value()
        self._execution_config["environment_mode"] = (
            "fixed" if self._env_mode_combo.currentIndex() == 1 else "procedural"
        )

        # Apply max_steps override to all operator configs
        max_steps = self._max_steps_spin.value()
        for config in self._operator_configs:
            config.max_steps = max_steps

        # Delegate to execution manager
        self._execution_manager.start_experiment(
            self._operator_configs,
            self._execution_config
        )

    def _on_stop_clicked(self):
        """Stop running experiment."""
        self._run_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._browse_button.setEnabled(True)
        self._progress_label.setText("Stopped by user")

        # Delegate to execution manager
        self._execution_manager.stop_experiment()

    def _on_progress_updated(self, episode_num: int, total_episodes: int, seed: int):
        """Handle progress update from execution manager."""
        self._progress_label.setText("<span style='font-size: 14pt; color: #4CAF50;'>Running...</span>")
        env_mode = self._env_mode_combo.currentText()
        self._episode_label.setText(
            f"<b>Episode:</b> {episode_num}/{total_episodes} &nbsp;&nbsp; "
            f"<b>Seed:</b> {seed} &nbsp;&nbsp; "
            f"<b>Mode:</b> {env_mode}"
        )

    def _on_experiment_completed(self, num_episodes: int):
        """Handle experiment completion from execution manager."""
        self._run_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._browse_button.setEnabled(True)
        self._progress_label.setText("<span style='font-size: 14pt; color: #4CAF50;'>✓ Completed!</span>")
        self._episode_label.setText(f"Successfully completed {num_episodes} episodes")

        QtWidgets.QMessageBox.information(
            self,
            "Experiment Complete",
            f"Successfully completed {num_episodes} episodes!\n\nTelemetry saved to: var/operators/telemetry/"
        )

    @property
    def execution_manager(self) -> OperatorScriptExecutionManager:
        """Get the execution manager (for MainWindow to connect signals)."""
        return self._execution_manager
