"""Advanced configuration tab - main container for the Unified Flow.

Combines all four steps into a single coherent configuration interface:
1. Environment Selection
2. Agent Configuration
3. Worker Configuration
4. Run Mode Selection

See Also:
    - docs/1.0_DAY_41/TASK_3/01_ui_migration_plan.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.core.enums import SteppingParadigm
from gym_gui.services.policy_mapping import AgentPolicyBinding, PolicyMappingService
from gym_gui.services.service_locator import get_service_locator

from .environment_selector import EnvironmentSelector
from .agent_config_table import AgentConfigTable, AgentRowConfig
from .worker_config_panel import WorkerConfigPanel
from .run_mode_selector import RunModeSelector, RunMode

_LOGGER = logging.getLogger(__name__)


@dataclass
class LaunchConfig:
    """Complete configuration for launching a session.

    Attributes:
        env_id: The environment identifier
        seed: Random seed
        paradigm: Detected stepping paradigm
        agent_bindings: Per-agent policy/worker bindings
        worker_configs: Worker-specific configurations
        run_mode: Interactive, headless, or evaluation
    """

    env_id: str
    seed: int
    paradigm: SteppingParadigm
    agent_bindings: Dict[str, AgentRowConfig] = field(default_factory=dict)
    worker_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_mode: RunMode = RunMode.INTERACTIVE

    @property
    def primary_worker_id(self) -> Optional[str]:
        """Get the primary worker ID (first non-local worker)."""
        for binding in self.agent_bindings.values():
            if binding.worker_id != "local":
                return binding.worker_id
        return None


class AdvancedConfigTab(QtWidgets.QWidget):
    """Advanced configuration tab with the Unified Flow.

    Provides full flexibility for configuring:
    - Any environment (Gymnasium, PettingZoo, ViZDoom, etc.)
    - Per-agent policy/worker bindings
    - Worker-specific parameters
    - Run mode (interactive/headless/evaluation)

    Signals:
        launch_requested: Emitted when user clicks Launch
        environment_load_requested: Emitted when user wants to load environment

    Example:
        tab = AdvancedConfigTab()
        tab.launch_requested.connect(on_launch)
    """

    # Signals
    launch_requested = pyqtSignal(object)  # LaunchConfig
    environment_load_requested = pyqtSignal(str, int)  # env_id, seed

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._current_config: Optional[LaunchConfig] = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QtWidgets.QLabel(
            "<b>Advanced Configuration</b><br>"
            "<span style='color: #666; font-size: 11px;'>"
            "Full control over environment, agents, workers, and run mode. "
            "Use this for scenarios the other tabs can't express."
            "</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Scrollable content area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setSpacing(16)

        # Step 1: Environment Selection
        self._env_selector = EnvironmentSelector()
        content_layout.addWidget(self._env_selector)

        # Step 2: Agent Configuration
        self._agent_table = AgentConfigTable()
        content_layout.addWidget(self._agent_table)

        # Step 3: Worker Configuration
        self._worker_panel = WorkerConfigPanel()
        content_layout.addWidget(self._worker_panel)

        # Step 4: Run Mode
        self._run_mode_selector = RunModeSelector()
        content_layout.addWidget(self._run_mode_selector)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Launch button section
        launch_frame = QtWidgets.QFrame()
        launch_frame.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border-radius: 4px; padding: 8px; }"
        )
        launch_layout = QtWidgets.QHBoxLayout(launch_frame)

        # Config summary
        self._summary_label = QtWidgets.QLabel("<i>Configure above to enable launch.</i>")
        self._summary_label.setStyleSheet("color: #666;")
        launch_layout.addWidget(self._summary_label, 1)

        # Launch button
        self._launch_btn = QtWidgets.QPushButton("Launch")
        self._launch_btn.setMinimumWidth(120)
        self._launch_btn.setEnabled(False)
        self._launch_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        launch_layout.addWidget(self._launch_btn)

        layout.addWidget(launch_frame)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Environment changes trigger agent list update
        self._env_selector.agents_detected.connect(self._on_agents_detected)
        self._env_selector.environment_changed.connect(self._on_environment_changed)

        # Agent bindings update worker panel
        self._agent_table.bindings_changed.connect(self._on_bindings_changed)

        # Run mode changes
        self._run_mode_selector.mode_changed.connect(self._update_summary)

        # Launch button
        self._launch_btn.clicked.connect(self._on_launch)

    def _on_agents_detected(self, agent_ids: List[str]) -> None:
        """Handle agents detected from environment."""
        self._agent_table.set_agents(agent_ids)
        self._update_summary()

    def _on_environment_changed(self, env_id: str) -> None:
        """Handle environment selection change."""
        seed = self._env_selector.seed
        self.environment_load_requested.emit(env_id, seed)
        self._update_summary()

    def _on_bindings_changed(self, bindings: Dict[str, AgentRowConfig]) -> None:
        """Handle agent bindings change."""
        self._worker_panel.update_from_bindings(bindings)
        self._update_summary()

    def _update_summary(self, *args: Any) -> None:
        """Update the configuration summary."""
        env_id = self._env_selector.selected_env_id
        if not env_id:
            self._summary_label.setText("<i>Select an environment to begin.</i>")
            self._launch_btn.setEnabled(False)
            return

        bindings = self._agent_table.get_bindings()
        run_mode = self._run_mode_selector.selected_mode

        # Count agents by type
        human_count = sum(1 for b in bindings.values() if b.actor_id == "human_keyboard")
        ai_count = len(bindings) - human_count

        # Determine primary worker
        workers = set(b.worker_id for b in bindings.values() if b.worker_id != "local")
        worker_str = ", ".join(workers) if workers else "Local only"

        summary = (
            f"<b>{env_id}</b> | "
            f"{human_count} human, {ai_count} AI | "
            f"Workers: {worker_str} | "
            f"Mode: {run_mode.name.title()}"
        )
        self._summary_label.setText(summary)
        self._launch_btn.setEnabled(True)

    def _on_launch(self) -> None:
        """Handle launch button click."""
        env_id = self._env_selector.selected_env_id
        if not env_id:
            return

        # Build launch config
        config = LaunchConfig(
            env_id=env_id,
            seed=self._env_selector.seed,
            paradigm=self._env_selector.paradigm,
            agent_bindings=self._agent_table.get_bindings(),
            worker_configs=self._worker_panel.get_configs(),
            run_mode=self._run_mode_selector.selected_mode,
        )

        # Configure PolicyMappingService
        self._configure_policy_mapping(config)

        self._current_config = config
        self.launch_requested.emit(config)

        _LOGGER.info(
            "Launch requested: env=%s, paradigm=%s, mode=%s, agents=%d",
            config.env_id,
            config.paradigm.value,
            config.run_mode.name,
            len(config.agent_bindings),
        )

    def _configure_policy_mapping(self, config: LaunchConfig) -> None:
        """Configure PolicyMappingService from launch config."""
        policy_mapping = get_service_locator().resolve(PolicyMappingService)
        if policy_mapping is None:
            _LOGGER.warning("PolicyMappingService not registered, skipping configuration")
            return

        # Reset and configure
        policy_mapping.reset()
        policy_mapping.set_paradigm(config.paradigm)
        policy_mapping.set_agents(list(config.agent_bindings.keys()))

        # Bind each agent
        for agent_id, row_config in config.agent_bindings.items():
            worker_config = config.worker_configs.get(row_config.worker_id, {})
            policy_mapping.bind_agent_policy(
                agent_id,
                row_config.actor_id,
                worker_id=row_config.worker_id if row_config.worker_id != "local" else None,
                config=worker_config,
            )

        _LOGGER.debug(
            "PolicyMappingService configured: %d bindings",
            len(config.agent_bindings),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_config(self) -> Optional[LaunchConfig]:
        """Get the current launch configuration."""
        return self._current_config

    def get_env_id(self) -> Optional[str]:
        """Get the selected environment ID."""
        return self._env_selector.selected_env_id

    def get_seed(self) -> int:
        """Get the current seed."""
        return self._env_selector.seed

    def get_run_mode(self) -> RunMode:
        """Get the selected run mode."""
        return self._run_mode_selector.selected_mode
