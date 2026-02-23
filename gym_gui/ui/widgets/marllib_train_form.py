"""MARLlib worker training form.

Provides a configuration dialog for MARLlib multi-agent RL training runs with:
- Algorithm selection grouped by paradigm (IL / CC / VD)
- Environment family and map name selection (linked dropdowns)
- Policy sharing, network architecture, and encode layer configuration
- Ray/RLlib resource allocation (GPUs, rollout workers, local mode)
- Stop conditions (timesteps, reward, iterations)
- Dry-run validation via marllib_worker CLI
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
)

from marllib_worker.registries import (
    ALGO_TYPE_DICT,
    ALL_ALGORITHMS,
    ALL_ENVIRONMENTS,
    SHARE_POLICY_OPTIONS,
    CORE_ARCH_OPTIONS,
    DISCRETE_ONLY_ALGOS,
    CONTINUOUS_ONLY_ALGOS,
    get_algo_type,
)


_LOGGER = logging.getLogger("gym_gui.ui.marllib_train_form")

# ---------------------------------------------------------------------------
# Map name suggestions per environment family
# Source: MARLlib documentation, environment YAML configs, and MARLlib papers
# ---------------------------------------------------------------------------

_MAP_SUGGESTIONS: Dict[str, List[str]] = {
    "mpe": [
        "simple_spread",
        "simple_reference",
        "simple_speaker_listener",
        "simple_adversary",
        "simple_crypto",
        "simple_push",
        "simple_tag",
        "simple_world_comm",
    ],
    "smac": [
        "3m",
        "8m",
        "2s3z",
        "25m",
        "3s5z",
        "5m_vs_6m",
        "8m_vs_9m",
        "10m_vs_11m",
        "27m_vs_30m",
        "3s5z_vs_3s6z",
        "MMM",
        "MMM2",
        "2c_vs_64zg",
        "corridor",
        "6h_vs_8z",
        "3s_vs_5z",
        "1c3s5z",
        "bane_vs_bane",
    ],
    "football": [
        "academy_empty_goal_close",
        "academy_empty_goal",
        "academy_run_to_score",
        "academy_run_to_score_with_keeper",
        "academy_pass_and_shoot_with_keeper",
        "academy_3_vs_1_with_keeper",
        "academy_corner",
        "academy_counterattack_easy",
        "academy_counterattack_hard",
        "academy_single_goal_versus_lazy",
    ],
    "mamujoco": [
        "2x3-Ant",
        "2x4-Ant",
        "4x2-Ant",
        "2x3-HalfCheetah",
        "6x1-HalfCheetah",
    ],
    "gymnasium_mamujoco": [
        "Ant-v2_2x4",
        "Ant-v2_4x2",
        "HalfCheetah-v2_2x3",
        "HalfCheetah-v2_6x1",
    ],
    "gymnasium_mpe": [
        "simple_spread_v3",
        "simple_reference_v3",
        "simple_speaker_listener_v4",
        "simple_adversary_v3",
        "simple_push_v3",
        "simple_tag_v3",
    ],
    "magent": [
        "adversarial_pursuit",
        "battle",
        "battlefield",
        "combined_arms",
        "gather",
        "tiger_deer",
    ],
    "pommerman": [
        "PommeFFACompetition-v0",
        "PommeTeamCompetition-v0",
        "PommeTeam-v0",
    ],
    "rware": [
        "rware-tiny-2ag-v1",
        "rware-tiny-4ag-v1",
        "rware-small-4ag-v1",
    ],
    "lbf": [
        "Foraging-8x8-2p-1f-v2",
        "Foraging-8x8-2p-2f-v2",
        "Foraging-10x10-3p-3f-v2",
    ],
    "hanabi": [
        "Hanabi-Full",
        "Hanabi-Full-CardKnowledge",
        "Hanabi-Small",
        "Hanabi-Very-Small",
    ],
    "metadrive": [
        "Bottleneck",
        "ParkingLot",
        "Intersection",
        "Roundabout",
        "Tollgate",
    ],
    "overcooked": [
        "cramped_room",
        "asymmetric_advantages",
        "coordination_ring",
        "forced_coordination",
        "counter_circuit",
    ],
    "sisl": [
        "multiwalker",
        "pursuit",
        "waterworld",
    ],
    "hns": [
        "BoxLocking",
        "BlueprintConstruction",
    ],
    "gobigger": [
        "st_t1p2",
    ],
    "mate": [
        "MATE-4v2-9-v0",
        "MATE-4v2-32-v0",
    ],
    "aircombat": [
        "MultipleCombat",
    ],
    "voltage": [
        "voltage_control",
    ],
}

# Paradigm display labels
_PARADIGM_LABELS: Dict[str, str] = {
    "IL": "Independent Learning",
    "CC": "Centralized Critic",
    "VD": "Value Decomposition",
}


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------


def _generate_run_id(prefix: str, algo: str) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = algo.replace("_", "-").lower()
    short_uuid = str(uuid.uuid4())[:6]
    return f"{prefix}-{slug}-{timestamp}-{short_uuid}"


# ---------------------------------------------------------------------------
# Form state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FormState:
    """Captured state from the form."""

    algo: str
    algo_type: str  # IL, CC, or VD
    environment_name: str
    map_name: str
    share_policy: str
    core_arch: str
    encode_layer: str
    force_coop: bool
    hyperparam_source: str
    framework: str
    num_gpus: int
    num_workers: int
    local_mode: bool
    stop_timesteps: int
    stop_reward: float
    stop_iters: int
    checkpoint_freq: int
    checkpoint_end: bool
    seed: Optional[int]
    notes: Optional[str]


# ---------------------------------------------------------------------------
# Main form widget
# ---------------------------------------------------------------------------


class MARLlibTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Configuration dialog for MARLlib multi-agent training."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("MARLlib Training Configuration")
        self.setModal(True)
        self.resize(750, 720)

        self._last_config: Optional[Dict[str, Any]] = None

        self._build_ui()

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="MARLlib training form opened",
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        self._build_header(layout)
        self._build_algorithm_section(layout)
        self._build_environment_section(layout)
        self._build_architecture_section(layout)
        self._build_training_section(layout)
        self._build_resource_section(layout)
        self._build_notes_section(layout)

        layout.addStretch()

        # Button bar
        btn_layout = QtWidgets.QHBoxLayout()
        self._validate_btn = QtWidgets.QPushButton("Validate (Dry Run)")
        self._validate_btn.setToolTip("Run marllib_worker CLI in --dry-run mode to check config")
        self._validate_btn.clicked.connect(self._handle_validate)
        btn_layout.addWidget(self._validate_btn)
        btn_layout.addStretch()

        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._handle_accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        root.addLayout(btn_layout)

    def _build_header(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        header = QtWidgets.QLabel(
            "<b>MARLlib Multi-Agent RL Training</b><br>"
            "<small>18 algorithms across 3 paradigms (IL, CC, VD) | "
            "19 environment families | Ray/RLlib backend</small>"
        )
        header.setWordWrap(True)
        parent_layout.addWidget(header)

    # --- Algorithm ---

    def _build_algorithm_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Algorithm")
        grid = QtWidgets.QGridLayout(group)
        grid.setColumnStretch(1, 1)

        # Paradigm filter
        grid.addWidget(QtWidgets.QLabel("Paradigm:"), 0, 0)
        self._paradigm_combo = QtWidgets.QComboBox()
        self._paradigm_combo.addItems(["All", "IL", "CC", "VD"])
        self._paradigm_combo.setToolTip(
            "IL = Independent Learning, CC = Centralized Critic, "
            "VD = Value Decomposition"
        )
        self._paradigm_combo.currentTextChanged.connect(self._on_paradigm_changed)
        grid.addWidget(self._paradigm_combo, 0, 1)

        # Algorithm selector
        grid.addWidget(QtWidgets.QLabel("Algorithm:"), 1, 0)
        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.setToolTip("Select the MARL algorithm to train")
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        grid.addWidget(self._algo_combo, 1, 1)

        # Algo type display (read-only label)
        grid.addWidget(QtWidgets.QLabel("Type:"), 2, 0)
        self._algo_type_label = QtWidgets.QLabel("")
        self._algo_type_label.setStyleSheet("color: #666; font-size: 11px;")
        grid.addWidget(self._algo_type_label, 2, 1)

        parent_layout.addWidget(group)

        # Populate initial list
        self._populate_algo_combo("All")

    def _populate_algo_combo(self, paradigm_filter: str) -> None:
        """Populate algorithm dropdown, optionally filtered by paradigm."""
        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()

        if paradigm_filter == "All":
            # Show all algorithms grouped by paradigm via display text
            for paradigm in ("IL", "CC", "VD"):
                for algo in sorted(ALGO_TYPE_DICT[paradigm]):
                    self._algo_combo.addItem(f"{algo}  [{paradigm}]", algo)
        else:
            for algo in sorted(ALGO_TYPE_DICT.get(paradigm_filter, [])):
                self._algo_combo.addItem(algo, algo)

        self._algo_combo.blockSignals(False)

        if self._algo_combo.count() > 0:
            self._algo_combo.setCurrentIndex(0)
            self._on_algo_changed(self._algo_combo.currentText())

    def _on_paradigm_changed(self, paradigm: str) -> None:
        self._populate_algo_combo(paradigm)
        self.log_constant(
            LOG_UI_TRAIN_FORM_TRACE,
            message="Paradigm filter changed",
            extra={"paradigm": paradigm},
        )

    def _on_algo_changed(self, _text: str) -> None:
        algo = self._algo_combo.currentData()
        if algo and algo in ALL_ALGORITHMS:
            algo_type = get_algo_type(algo)
            label = _PARADIGM_LABELS.get(algo_type, algo_type)
            constraint = ""
            if algo in DISCRETE_ONLY_ALGOS:
                constraint = " (discrete action spaces only)"
            elif algo in CONTINUOUS_ONLY_ALGOS:
                constraint = " (continuous action spaces only)"
            self._algo_type_label.setText(f"{label}{constraint}")
        else:
            self._algo_type_label.setText("")

    # --- Environment ---

    def _build_environment_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Environment")
        grid = QtWidgets.QGridLayout(group)
        grid.setColumnStretch(1, 1)

        # Environment family
        grid.addWidget(QtWidgets.QLabel("Environment:"), 0, 0)
        self._env_combo = QtWidgets.QComboBox()
        for env in ALL_ENVIRONMENTS:
            self._env_combo.addItem(env)
        self._env_combo.setToolTip("MARLlib environment family")
        self._env_combo.currentTextChanged.connect(self._on_env_changed)
        grid.addWidget(self._env_combo, 0, 1)

        # Map name (editable combo with per-environment suggestions)
        grid.addWidget(QtWidgets.QLabel("Map / Scenario:"), 1, 0)
        self._map_combo = QtWidgets.QComboBox()
        self._map_combo.setEditable(True)
        self._map_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self._map_combo.setToolTip(
            "Map or scenario name within the environment. "
            "Select from suggestions or type a custom name."
        )
        grid.addWidget(self._map_combo, 1, 1)

        # Force cooperative
        self._force_coop_check = QtWidgets.QCheckBox(
            "Force cooperative (global shared reward)"
        )
        self._force_coop_check.setToolTip(
            "When enabled, all agents share a single global reward signal "
            "regardless of environment default."
        )
        grid.addWidget(self._force_coop_check, 2, 0, 1, 2)

        parent_layout.addWidget(group)

        # Populate initial map suggestions
        if self._env_combo.currentText():
            self._on_env_changed(self._env_combo.currentText())

    def _on_env_changed(self, env_name: str) -> None:
        """Update map suggestions when environment changes."""
        self._map_combo.blockSignals(True)
        self._map_combo.clear()
        suggestions = _MAP_SUGGESTIONS.get(env_name, [])
        for s in suggestions:
            self._map_combo.addItem(s)
        if suggestions:
            self._map_combo.setCurrentIndex(0)
        else:
            self._map_combo.setEditText("")
        self._map_combo.blockSignals(False)

        self.log_constant(
            LOG_UI_TRAIN_FORM_TRACE,
            message="Environment changed",
            extra={"env": env_name, "map_suggestions": len(suggestions)},
        )

    # --- Architecture ---

    def _build_architecture_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Policy Architecture")
        grid = QtWidgets.QGridLayout(group)
        grid.setColumnStretch(1, 1)

        # Share policy
        grid.addWidget(QtWidgets.QLabel("Policy Sharing:"), 0, 0)
        self._share_policy_combo = QtWidgets.QComboBox()
        for opt in SHARE_POLICY_OPTIONS:
            label = {
                "all": "all  --  single shared policy for all agents",
                "group": "group  --  one policy per agent group",
                "individual": "individual  --  separate policy per agent",
            }.get(opt, opt)
            self._share_policy_combo.addItem(label, opt)
        self._share_policy_combo.setToolTip("How policies are shared among agents")
        grid.addWidget(self._share_policy_combo, 0, 1)

        # Core architecture
        grid.addWidget(QtWidgets.QLabel("Core Network:"), 1, 0)
        self._core_arch_combo = QtWidgets.QComboBox()
        for opt in CORE_ARCH_OPTIONS:
            label = {
                "mlp": "mlp  --  feedforward",
                "gru": "gru  --  recurrent (GRU)",
                "lstm": "lstm  --  recurrent (LSTM)",
            }.get(opt, opt)
            self._core_arch_combo.addItem(label, opt)
        self._core_arch_combo.setToolTip(
            "Network architecture backbone: MLP, GRU, or LSTM"
        )
        grid.addWidget(self._core_arch_combo, 1, 1)

        # Encode layer
        grid.addWidget(QtWidgets.QLabel("Encode Layer:"), 2, 0)
        self._encode_layer_input = QtWidgets.QLineEdit("128-256")
        self._encode_layer_input.setToolTip(
            "Hidden layer sizes separated by hyphens (e.g. '128-256')"
        )
        grid.addWidget(self._encode_layer_input, 2, 1)

        # Hyperparam source
        grid.addWidget(QtWidgets.QLabel("Hyperparam Source:"), 3, 0)
        self._hyperparam_combo = QtWidgets.QComboBox()
        self._hyperparam_combo.setEditable(True)
        self._hyperparam_combo.addItems(["common", "test"])
        self._hyperparam_combo.setToolTip(
            "'common' = default hyperparams, 'test' = small/fast, "
            "or type an environment name for environment-tuned hyperparams."
        )
        grid.addWidget(self._hyperparam_combo, 3, 1)

        parent_layout.addWidget(group)

    # --- Training ---

    def _build_training_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Training")
        grid = QtWidgets.QGridLayout(group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Stop conditions row 0
        grid.addWidget(QtWidgets.QLabel("Max Timesteps:"), 0, 0)
        self._stop_timesteps_spin = QtWidgets.QSpinBox()
        self._stop_timesteps_spin.setRange(1_000, 100_000_000)
        self._stop_timesteps_spin.setValue(1_000_000)
        self._stop_timesteps_spin.setSingleStep(100_000)
        self._stop_timesteps_spin.setToolTip("Training stops after this many timesteps")
        grid.addWidget(self._stop_timesteps_spin, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Max Iterations:"), 0, 2)
        self._stop_iters_spin = QtWidgets.QSpinBox()
        self._stop_iters_spin.setRange(1, 99_999_999)
        self._stop_iters_spin.setValue(9_999_999)
        self._stop_iters_spin.setSingleStep(1_000)
        self._stop_iters_spin.setToolTip("Training stops after this many iterations")
        grid.addWidget(self._stop_iters_spin, 0, 3)

        # Stop conditions row 1
        grid.addWidget(QtWidgets.QLabel("Target Reward:"), 1, 0)
        self._stop_reward_spin = QtWidgets.QDoubleSpinBox()
        self._stop_reward_spin.setRange(-1e9, 1e9)
        self._stop_reward_spin.setValue(999_999.0)
        self._stop_reward_spin.setDecimals(1)
        self._stop_reward_spin.setToolTip("Training stops if mean reward exceeds this")
        grid.addWidget(self._stop_reward_spin, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Seed:"), 1, 2)
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(-1, 999_999)
        self._seed_spin.setValue(-1)
        self._seed_spin.setSpecialValueText("None (random)")
        self._seed_spin.setToolTip("Random seed (-1 = random)")
        grid.addWidget(self._seed_spin, 1, 3)

        # Checkpoint row 2
        grid.addWidget(QtWidgets.QLabel("Checkpoint Every:"), 2, 0)
        self._checkpoint_freq_spin = QtWidgets.QSpinBox()
        self._checkpoint_freq_spin.setRange(0, 100_000)
        self._checkpoint_freq_spin.setValue(100)
        self._checkpoint_freq_spin.setToolTip(
            "Save checkpoint every N iterations (0 = disabled)"
        )
        grid.addWidget(self._checkpoint_freq_spin, 2, 1)

        self._checkpoint_end_check = QtWidgets.QCheckBox("Save checkpoint at training end")
        self._checkpoint_end_check.setChecked(True)
        grid.addWidget(self._checkpoint_end_check, 2, 2, 1, 2)

        # Framework row 3
        grid.addWidget(QtWidgets.QLabel("Framework:"), 3, 0)
        self._framework_combo = QtWidgets.QComboBox()
        self._framework_combo.addItems(["torch", "tf"])
        self._framework_combo.setToolTip("Deep learning framework for RLlib")
        grid.addWidget(self._framework_combo, 3, 1)

        parent_layout.addWidget(group)

    # --- Resources ---

    def _build_resource_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Ray / RLlib Resources")
        grid = QtWidgets.QGridLayout(group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        grid.addWidget(QtWidgets.QLabel("GPUs:"), 0, 0)
        self._num_gpus_spin = QtWidgets.QSpinBox()
        self._num_gpus_spin.setRange(0, 16)
        self._num_gpus_spin.setValue(1)
        self._num_gpus_spin.setToolTip("Number of GPUs for the trainer process")
        grid.addWidget(self._num_gpus_spin, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Rollout Workers:"), 0, 2)
        self._num_workers_spin = QtWidgets.QSpinBox()
        self._num_workers_spin.setRange(0, 64)
        self._num_workers_spin.setValue(2)
        self._num_workers_spin.setToolTip("Number of Ray rollout worker processes")
        grid.addWidget(self._num_workers_spin, 0, 3)

        self._local_mode_check = QtWidgets.QCheckBox(
            "Ray local mode (single-process, for debugging)"
        )
        self._local_mode_check.setToolTip(
            "Run Ray in local mode (no parallelism). Useful for debugging."
        )
        grid.addWidget(self._local_mode_check, 1, 0, 1, 4)

        parent_layout.addWidget(group)

    # --- Notes ---

    def _build_notes_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Notes")
        vbox = QtWidgets.QVBoxLayout(group)
        self._notes_input = QtWidgets.QTextEdit()
        self._notes_input.setMaximumHeight(60)
        self._notes_input.setPlaceholderText("Optional notes for this training run...")
        vbox.addWidget(self._notes_input)
        parent_layout.addWidget(group)

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def _collect_state(self) -> _FormState:
        """Capture all form widget values into an immutable state snapshot."""
        algo = self._algo_combo.currentData() or ""
        if not algo:
            # Fallback: parse algo name from display text ("mappo  [CC]" -> "mappo")
            raw = self._algo_combo.currentText()
            algo = raw.split()[0] if raw else ""

        try:
            algo_type = get_algo_type(algo)
        except ValueError:
            algo_type = "IL"

        env_name = self._env_combo.currentText()
        map_name = self._map_combo.currentText().strip()
        seed_val = self._seed_spin.value()

        return _FormState(
            algo=algo,
            algo_type=algo_type,
            environment_name=env_name,
            map_name=map_name,
            share_policy=self._share_policy_combo.currentData() or "all",
            core_arch=self._core_arch_combo.currentData() or "mlp",
            encode_layer=self._encode_layer_input.text().strip() or "128-256",
            force_coop=self._force_coop_check.isChecked(),
            hyperparam_source=self._hyperparam_combo.currentText().strip() or "common",
            framework=self._framework_combo.currentText(),
            num_gpus=self._num_gpus_spin.value(),
            num_workers=self._num_workers_spin.value(),
            local_mode=self._local_mode_check.isChecked(),
            stop_timesteps=self._stop_timesteps_spin.value(),
            stop_reward=self._stop_reward_spin.value(),
            stop_iters=self._stop_iters_spin.value(),
            checkpoint_freq=self._checkpoint_freq_spin.value(),
            checkpoint_end=self._checkpoint_end_check.isChecked(),
            seed=seed_val if seed_val >= 0 else None,
            notes=self._notes_input.toPlainText().strip() or None,
        )

    # ------------------------------------------------------------------
    # Config building
    # ------------------------------------------------------------------

    def _build_config(
        self,
        state: _FormState,
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the trainer payload for the MARLlib worker.

        This dict is consumed by the MOSAIC trainer daemon and matches the
        structure used by CleanRL, XuanCe, and other worker forms.
        """
        run_id = run_id or _generate_run_id("marllib", state.algo)

        # Worker-specific config (passed to marllib_worker.cli via --config)
        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": state.algo,
            "environment_name": state.environment_name,
            "map_name": state.map_name,
            "force_coop": state.force_coop,
            "hyperparam_source": state.hyperparam_source,
            "share_policy": state.share_policy,
            "core_arch": state.core_arch,
            "encode_layer": state.encode_layer,
            "num_gpus": state.num_gpus,
            "num_workers": state.num_workers,
            "local_mode": state.local_mode,
            "framework": state.framework,
            "stop_timesteps": state.stop_timesteps,
            "stop_reward": state.stop_reward,
            "stop_iters": state.stop_iters,
            "checkpoint_freq": state.checkpoint_freq,
            "checkpoint_end": state.checkpoint_end,
        }
        if state.seed is not None:
            worker_config["seed"] = state.seed

        # Metadata for the trainer daemon and UI tab system
        metadata: Dict[str, Any] = {
            "ui": {
                "worker_id": "marllib_worker",
                "algo": state.algo,
                "algo_type": state.algo_type,
                "environment_name": state.environment_name,
                "map_name": state.map_name,
                "share_policy": state.share_policy,
                "core_arch": state.core_arch,
            },
            "worker": {
                "worker_id": "marllib_worker",
                "module": "marllib_worker.cli",
                "config": worker_config,
            },
        }

        # Environment variables for the subprocess
        environment: Dict[str, str] = {
            "MARLLIB_RUN_ID": run_id,
            "MARLLIB_ALGO": state.algo,
            "MARLLIB_ENV": state.environment_name,
            "MARLLIB_MAP": state.map_name,
        }
        if state.seed is not None:
            environment["MARLLIB_SEED"] = str(state.seed)

        # Top-level trainer config dict
        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "marllib_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": max(2, state.num_workers + 1),
                "memory_mb": 4096,
                "gpus": {
                    "requested": state.num_gpus,
                    "mandatory": False,
                },
            },
            "metadata": metadata,
            "artifacts": {
                "output_prefix": f"runs/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": True,
            },
        }

        return config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(
        self,
        state: _FormState,
        *,
        run_id: str,
        persist_config: bool,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Run dry-run validation via marllib_worker CLI subprocess."""
        try:
            from gym_gui.validations.validation_marllib_worker_form import (
                run_marllib_dry_run,
            )
        except ImportError:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="MARLlib validation module not available, skipping dry-run",
            )
            config = self._build_config(state, run_id=run_id)
            return True, config

        config = self._build_config(state, run_id=run_id)
        worker_config = config["metadata"]["worker"]["config"]

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="Running MARLlib dry-run validation",
            extra={
                "algo": state.algo,
                "env": state.environment_name,
                "map": state.map_name,
            },
        )

        success, output = run_marllib_dry_run(worker_config)

        if success:
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="MARLlib dry-run validation passed",
                extra={"run_id": run_id},
            )
            if persist_config:
                self._last_config = config
            return True, config
        else:
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message="MARLlib dry-run validation failed",
                extra={"output": output[:500]},
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Failed",
                f"MARLlib dry-run failed:\n\n{output[:1000]}",
            )
            return False, None

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _handle_validate(self) -> None:
        """Handle Validate button click."""
        state = self._collect_state()
        if not state.algo or not state.map_name:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Validation rejected: incomplete configuration",
                extra={"has_algo": bool(state.algo), "has_map": bool(state.map_name)},
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Configuration",
                "Select an algorithm and specify a map name before validation.",
            )
            return
        run_id = _generate_run_id("marllib", state.algo)
        self._run_validation(state, run_id=run_id, persist_config=False)

    def _handle_accept(self) -> None:
        """Handle OK button click -- validates before accepting."""
        state = self._collect_state()

        if not state.algo:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Accept rejected: algorithm not selected",
            )
            QtWidgets.QMessageBox.warning(
                self, "Algorithm Required", "Please select an algorithm."
            )
            return

        if not state.map_name:
            self.log_constant(
                LOG_UI_TRAIN_FORM_WARNING,
                message="Accept rejected: map name not specified",
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Map Name Required",
                "Please specify a map or scenario name for the environment.",
            )
            return

        run_id = _generate_run_id("marllib", state.algo)

        success, config = self._run_validation(
            state, run_id=run_id, persist_config=True
        )
        if not success:
            self._last_config = None
            return

        self._last_config = config

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="MARLlib training config accepted",
            extra={
                "run_id": run_id,
                "algo": state.algo,
                "algo_type": state.algo_type,
                "env": state.environment_name,
                "map": state.map_name,
                "share_policy": state.share_policy,
                "core_arch": state.core_arch,
            },
        )

        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the trainer payload generated by the form."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        state = self._collect_state()
        return self._build_config(state)


__all__ = ["MARLlibTrainForm"]


# ---------------------------------------------------------------------------
# Self-registration with WorkerFormFactory
# ---------------------------------------------------------------------------
try:
    from gym_gui.ui.forms.factory import get_worker_form_factory

    _factory = get_worker_form_factory()
    if not _factory.has_train_form("marllib_worker"):
        _factory.register_train_form(
            "marllib_worker",
            lambda parent=None, **kwargs: MARLlibTrainForm(parent=parent, **kwargs),
        )
except Exception:
    pass  # Form factory not available or other forms have import errors
