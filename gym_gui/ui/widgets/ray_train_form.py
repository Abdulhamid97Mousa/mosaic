"""Ray RLlib worker training form.

This form provides configuration options for training multi-agent policies
using Ray RLlib. Supports multiple training paradigms and PettingZoo environments.
"""

from __future__ import annotations

import copy
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.pettingzoo_enums import (
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    get_description,
    get_display_name,
    get_envs_by_family,
    is_aec_env,
)

# Import algorithm parameter schema functions
# Type stubs for when imports fail
get_available_algorithms: Any = None
get_algorithm_info: Any = None
get_algorithm_fields: Any = None
get_default_params: Any = None
filter_params_for_algorithm: Any = None
_HAS_ALGO_PARAMS = False

try:
    from ray_worker.algo_params import (  # type: ignore[import-not-found]
        get_available_algorithms,
        get_algorithm_info,
        get_algorithm_fields,
        get_default_params,
        filter_params_for_algorithm,
    )
    _HAS_ALGO_PARAMS = True
except ImportError:
    pass

_LOGGER = logging.getLogger(__name__)


# Default agent counts for PettingZoo environments
# Used for auto-calculating optimal rollout workers
_DEFAULT_AGENT_COUNTS: Dict[str, int] = {
    # SISL (cooperative)
    "pursuit_v4": 8,
    "waterworld_v4": 5,  # 5 pursuers by default
    "multiwalker_v9": 3,
    # MPE
    "simple_v3": 1,
    "simple_adversary_v3": 3,
    "simple_crypto_v3": 2,
    "simple_push_v3": 2,
    "simple_reference_v3": 2,
    "simple_speaker_listener_v4": 2,
    "simple_spread_v3": 3,
    "simple_tag_v3": 4,
    "simple_world_comm_v3": 6,
    # Butterfly
    "knights_archers_zombies_v10": 4,
    "pistonball_v6": 20,
    "cooperative_pong_v5": 2,
    # Classic (turn-based, usually 2 players)
    "chess_v6": 2,
    "go_v5": 2,
    "connect_four_v3": 2,
    "tictactoe_v3": 2,
    "backgammon_v3": 2,
    "checkers_v3": 2,
    "hanabi_v5": 2,
    "leduc_holdem_v4": 2,
    "texas_holdem_v4": 2,
    "texas_holdem_no_limit_v6": 2,
    "gin_rummy_v4": 2,
}


def _get_default_agent_count(env_id: str) -> int:
    """Get the default agent count for an environment.

    Args:
        env_id: PettingZoo environment ID

    Returns:
        Default number of agents (falls back to 2 if unknown)
    """
    return _DEFAULT_AGENT_COUNTS.get(env_id, 2)


def _get_available_cpus() -> int:
    """Get the number of available CPU cores.

    Returns:
        Number of available CPUs (minimum 1)
    """
    import os
    try:
        # Try multiprocessing first
        import multiprocessing
        return max(1, multiprocessing.cpu_count())
    except Exception:
        pass

    try:
        # Fallback to os.cpu_count()
        count = os.cpu_count()
        if count:
            return max(1, count)
    except Exception:
        pass

    return 4  # Safe default


def _calculate_optimal_workers(env_id: str, available_cpus: int) -> int:
    """Calculate optimal number of rollout workers.

    Heuristics:
    - Leave 2 CPUs for learner/driver process
    - Scale with number of agents (more agents = more parallel samples help)
    - Cap at reasonable maximum

    Args:
        env_id: PettingZoo environment ID
        available_cpus: Number of available CPU cores

    Returns:
        Recommended number of rollout workers
    """
    num_agents = _get_default_agent_count(env_id)

    # Reserve CPUs for learner (1-2 depending on total)
    reserved_cpus = 2 if available_cpus > 4 else 1

    # Available for workers
    worker_cpus = max(1, available_cpus - reserved_cpus)

    # For multi-agent envs, having more workers helps parallelize sampling
    # Rule: workers ≈ max(2, min(worker_cpus, num_agents))
    if num_agents >= 8:
        # Large multi-agent (pursuit): use more workers
        optimal = min(worker_cpus, max(4, num_agents // 2))
    elif num_agents >= 4:
        # Medium multi-agent: moderate workers
        optimal = min(worker_cpus, max(2, num_agents))
    else:
        # Small (2-3 agents) or single agent
        optimal = min(worker_cpus, 2)

    # Ensure at least 1 worker, cap at 16
    return max(1, min(optimal, 16))


# Training paradigms for Ray RLlib multi-agent
_TRAINING_PARADIGMS = (
    ("parameter_sharing", "Parameter Sharing", "All agents share one policy (cooperative)"),
    ("independent", "Independent Learning", "Each agent has its own policy"),
    ("self_play", "Self-Play", "Agent plays against copies of itself (competitive)"),
    ("shared_value_function", "Shared Value Function", "CTDE - shared critic, separate actors"),
)

# Supported algorithms with descriptions
_ALGORITHMS = {
    "PPO": {
        "name": "Proximal Policy Optimization",
        "description": """<b>PPO (Proximal Policy Optimization)</b><br><br>
A policy gradient method that uses clipped surrogate objectives to ensure stable updates.
Well-suited for multi-agent cooperative scenarios with continuous action spaces.<br><br>
<b>Best for:</b> SISL environments (Waterworld, Multiwalker, Pursuit)<br>
<b>Strengths:</b> Sample efficient, stable training, good for parameter sharing<br>
<b>Limitations:</b> May require tuning clip parameter for competitive scenarios""",
    },
    "APPO": {
        "name": "Asynchronous PPO",
        "description": """<b>APPO (Asynchronous Proximal Policy Optimization)</b><br><br>
A distributed version of PPO that uses asynchronous workers for faster training.
Combines PPO's stability with IMPALA's scalability.<br><br>
<b>Best for:</b> Large-scale training with many workers<br>
<b>Strengths:</b> High throughput, scalable to many CPUs<br>
<b>Limitations:</b> Slightly less stable than synchronous PPO""",
    },
    "IMPALA": {
        "name": "Importance Weighted Actor-Learner",
        "description": """<b>IMPALA (Importance Weighted Actor-Learner Architecture)</b><br><br>
A highly scalable algorithm using V-trace for off-policy correction.
Designed for distributed training across many actors.<br><br>
<b>Best for:</b> Environments requiring high sample throughput<br>
<b>Strengths:</b> Excellent scalability, decoupled acting and learning<br>
<b>Limitations:</b> May need more samples for complex tasks""",
    },
    "DQN": {
        "name": "Deep Q-Network",
        "description": """<b>DQN (Deep Q-Network)</b><br><br>
Value-based algorithm using experience replay and target networks.
Only works with discrete action spaces.<br><br>
<b>Best for:</b> Classic games (Chess, Connect Four, Go) with discrete actions<br>
<b>Strengths:</b> Simple, well-understood, good for turn-based games<br>
<b>Limitations:</b> Cannot handle continuous actions, struggles with high-dimensional action spaces""",
    },
    "SAC": {
        "name": "Soft Actor-Critic",
        "description": """<b>SAC (Soft Actor-Critic)</b><br><br>
An off-policy algorithm that maximizes both expected reward and entropy.
Excellent for continuous control tasks.<br><br>
<b>Best for:</b> Continuous control (Multiwalker, physics-based envs)<br>
<b>Strengths:</b> Sample efficient, automatic entropy tuning, stable<br>
<b>Limitations:</b> Higher computational cost per update""",
    },
}


class RayRLlibTrainForm(QtWidgets.QDialog):
    """Training configuration form for Ray RLlib multi-agent environments.

    Provides UI for:
    - Environment family and game selection (PettingZoo)
    - Training paradigm selection (parameter sharing, independent, self-play, CTDE)
    - Algorithm selection (PPO, APPO, IMPALA, etc.)
    - Hyperparameter configuration
    - Resource allocation (workers, GPUs)
    - Logging and checkpoint settings
    - FastLane live visualization configuration
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        initial_env_id: Optional[str] = None,
        initial_family: Optional[str] = None,
        default_game: Optional[Any] = None,
    ) -> None:
        """Initialize the training form.

        Args:
            parent: Parent widget
            initial_env_id: Pre-selected environment ID
            initial_family: Pre-selected environment family
            default_game: Default game from control panel (used if initial_env_id not set)
        """
        super().__init__(parent)
        self.setWindowTitle("Ray RLlib Training Configuration")
        self.setMinimumSize(800, 850)

        # Use default_game if initial_env_id not provided
        if initial_env_id is None and default_game is not None:
            initial_env_id = str(default_game) if default_game else None

        self._initial_env_id = initial_env_id
        self._initial_family = initial_family
        self._last_config: Optional[Dict[str, Any]] = None
        self._last_validation_output: str = ""

        self._build_ui()
        self._connect_signals()
        self._populate_families()

        # Apply initial selections if provided
        if initial_family:
            self._select_family(initial_family)
        if initial_env_id:
            self._select_env(initial_env_id)

        # Trigger initial algorithm notes update
        self._on_algorithm_changed(0)

        # Initialize auto workers (disabled spinbox since Auto is checked by default)
        self._workers_spin.setEnabled(False)
        self._update_auto_workers()

    def _build_ui(self) -> None:
        """Build the form UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Scroll area for main content
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(12)

        # Top section with environment and algorithm side by side
        top_section = QtWidgets.QHBoxLayout()
        top_section.setSpacing(12)

        # Left column: Environment + Paradigm
        left_column = QtWidgets.QVBoxLayout()
        left_column.addWidget(self._create_env_group())
        left_column.addWidget(self._create_paradigm_group())
        left_column.addStretch(1)

        # Right column: Algorithm + Notes
        right_column = QtWidgets.QVBoxLayout()
        right_column.addWidget(self._create_algorithm_group())
        right_column.addWidget(self._create_algorithm_notes_group())
        right_column.addStretch(1)

        top_section.addLayout(left_column, 1)
        top_section.addLayout(right_column, 1)
        content_layout.addLayout(top_section)

        # Training Parameters Group
        content_layout.addWidget(self._create_training_params_group())

        # Resources Group
        content_layout.addWidget(self._create_resources_group())

        # Analytics & Tracking Group
        content_layout.addWidget(self._create_analytics_group())

        # FastLane Configuration Group
        content_layout.addWidget(self._create_fastlane_group())

        # Checkpoint Group
        content_layout.addWidget(self._create_checkpoint_group())

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Validation status label
        self._validation_status_label = QtWidgets.QLabel(
            "Configuration ready. Click Validate to run a dry-run check."
        )
        self._validation_status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self._validation_status_label)

        # Dialog buttons with Validate
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        # Add Validate button
        validate_btn = button_box.addButton(
            "Validate", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        if validate_btn is not None:
            validate_btn.clicked.connect(self._on_validate_clicked)

        layout.addWidget(button_box)

    def _create_env_group(self) -> QtWidgets.QGroupBox:
        """Create environment selection group."""
        group = QtWidgets.QGroupBox("Environment Selection", self)
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        # Family dropdown
        self._family_combo = QtWidgets.QComboBox(group)
        self._family_combo.setMinimumWidth(200)
        layout.addRow("Family:", self._family_combo)

        # Environment dropdown
        self._env_combo = QtWidgets.QComboBox(group)
        self._env_combo.setMinimumWidth(200)
        layout.addRow("Environment:", self._env_combo)

        # Environment info
        self._env_info_label = QtWidgets.QLabel("", group)
        self._env_info_label.setWordWrap(True)
        self._env_info_label.setStyleSheet(
            "color: #666; font-size: 10px; padding: 4px;"
        )
        self._env_info_label.setMaximumHeight(60)
        layout.addRow("", self._env_info_label)

        # API type indicator
        self._api_type_label = QtWidgets.QLabel("", group)
        self._api_type_label.setStyleSheet("font-weight: bold;")
        layout.addRow("API Type:", self._api_type_label)

        return group

    def _create_paradigm_group(self) -> QtWidgets.QGroupBox:
        """Create training paradigm selection group."""
        group = QtWidgets.QGroupBox("Multi-Agent Training Paradigm", self)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(4)

        self._paradigm_group = QtWidgets.QButtonGroup(group)

        for i, (paradigm_id, paradigm_name, paradigm_desc) in enumerate(_TRAINING_PARADIGMS):
            radio = QtWidgets.QRadioButton(f"{paradigm_name}", group)
            radio.setToolTip(paradigm_desc)
            radio.setProperty("paradigm_id", paradigm_id)
            self._paradigm_group.addButton(radio, i)
            layout.addWidget(radio)

            # Add description label
            desc_label = QtWidgets.QLabel(f"  {paradigm_desc}", group)
            desc_label.setStyleSheet("color: #666; font-size: 9px; margin-left: 20px;")
            layout.addWidget(desc_label)

        # Default to parameter sharing
        first_button = self._paradigm_group.button(0)
        if first_button is not None:
            first_button.setChecked(True)

        return group

    def _create_algorithm_group(self) -> QtWidgets.QGroupBox:
        """Create algorithm selection group."""
        group = QtWidgets.QGroupBox("Training Algorithm", self)
        layout = QtWidgets.QFormLayout(group)

        # Algorithm dropdown
        self._algo_combo = QtWidgets.QComboBox(group)
        for algo_id, algo_info in _ALGORITHMS.items():
            self._algo_combo.addItem(f"{algo_id} - {algo_info['name']}", algo_id)
        layout.addRow("Algorithm:", self._algo_combo)

        return group

    def _create_algorithm_notes_group(self) -> QtWidgets.QGroupBox:
        """Create algorithm notes/documentation group."""
        group = QtWidgets.QGroupBox("Algorithm Notes", self)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)

        self._algo_notes_text = QtWidgets.QTextEdit(group)
        self._algo_notes_text.setReadOnly(True)
        self._algo_notes_text.setMinimumHeight(150)
        self._algo_notes_text.setMaximumHeight(200)
        self._algo_notes_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)
        layout.addWidget(self._algo_notes_text)

        return group

    def _create_training_params_group(self) -> QtWidgets.QGroupBox:
        """Create training parameters group with dynamic algorithm-specific fields."""
        group = QtWidgets.QGroupBox("Training Parameters", self)
        main_layout = QtWidgets.QVBoxLayout(group)

        # Fixed parameters section (always shown)
        fixed_layout = QtWidgets.QGridLayout()
        fixed_layout.setColumnStretch(1, 1)
        fixed_layout.setColumnStretch(3, 1)

        # Total timesteps
        fixed_layout.addWidget(QtWidgets.QLabel("Total Timesteps:"), 0, 0)
        self._timesteps_spin = QtWidgets.QSpinBox(group)
        self._timesteps_spin.setRange(1000, 100_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10000)
        fixed_layout.addWidget(self._timesteps_spin, 0, 1)

        # Seed
        fixed_layout.addWidget(QtWidgets.QLabel("Random Seed:"), 0, 2)
        self._seed_spin = QtWidgets.QSpinBox(group)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setSpecialValueText("Random")
        fixed_layout.addWidget(self._seed_spin, 0, 3)

        main_layout.addLayout(fixed_layout)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet("color: #ccc;")
        main_layout.addWidget(separator)

        # Dynamic algorithm parameters section
        self._algo_params_label = QtWidgets.QLabel("Algorithm Parameters (PPO):")
        self._algo_params_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        main_layout.addWidget(self._algo_params_label)

        # Container for dynamic parameter widgets
        self._algo_params_container = QtWidgets.QWidget(group)
        self._algo_params_layout = QtWidgets.QGridLayout(self._algo_params_container)
        self._algo_params_layout.setColumnStretch(1, 1)
        self._algo_params_layout.setColumnStretch(3, 1)
        self._algo_params_layout.setContentsMargins(0, 4, 0, 0)
        main_layout.addWidget(self._algo_params_container)

        # Dictionary to hold dynamic parameter widgets
        self._algo_param_widgets: Dict[str, QtWidgets.QWidget] = {}

        # Initialize with PPO parameters (will be rebuilt on algorithm change)
        self._rebuild_algo_params("PPO")

        return group

    def _rebuild_algo_params(self, algorithm: str) -> None:
        """Rebuild the algorithm parameter widgets based on selected algorithm.

        Args:
            algorithm: Selected algorithm name (PPO, APPO, etc.)
        """
        # Clear existing widgets - must use setParent(None) to immediately
        # remove from layout, then deleteLater() for proper cleanup
        while self._algo_params_layout.count():
            item = self._algo_params_layout.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.setParent(None)  # Immediately remove from parent
                widget.deleteLater()    # Schedule for deletion
        self._algo_param_widgets.clear()

        # Process events to ensure widgets are fully removed before adding new ones
        QtWidgets.QApplication.processEvents()

        # Update section label
        self._algo_params_label.setText(f"Algorithm Parameters ({algorithm}):")

        # Get fields for this algorithm from schema
        fields: List[Dict[str, Any]] = []
        defaults: Dict[str, Any] = {}
        if _HAS_ALGO_PARAMS:
            try:
                fields = get_algorithm_fields(algorithm)
                defaults = get_default_params(algorithm)
            except Exception as e:
                _LOGGER.warning(f"Failed to get algo params for {algorithm}: {e}")
                fields = self._get_fallback_fields(algorithm)
        else:
            # Fallback if algo_params not available
            fields = self._get_fallback_fields(algorithm)

        # Create widgets for each field
        row = 0
        col = 0
        for field in fields:
            name = field["name"]
            field_type = field.get("type", "float")
            default = defaults.get(name, field.get("default"))
            min_val = field.get("min", 0)
            max_val = field.get("max", 1000000)
            help_text = field.get("help", "")

            # Create label
            display_name = self._format_field_name(name)
            label = QtWidgets.QLabel(f"{display_name}:")
            label.setToolTip(help_text)
            self._algo_params_layout.addWidget(label, row, col * 2)

            # Create appropriate widget based on type
            if field_type == "int":
                widget = QtWidgets.QSpinBox(self._algo_params_container)
                widget.setRange(int(min_val), int(max_val))
                widget.setValue(int(default) if default is not None else 0)
                widget.setSingleStep(max(1, int((max_val - min_val) / 100)))
            elif field_type == "float":
                widget = QtWidgets.QDoubleSpinBox(self._algo_params_container)
                widget.setDecimals(6 if max_val < 1 else 4)
                widget.setRange(float(min_val), float(max_val))
                widget.setValue(float(default) if default is not None else 0.0)
                widget.setSingleStep(max(0.0001, (max_val - min_val) / 100))
            elif field_type == "bool":
                widget = QtWidgets.QCheckBox(self._algo_params_container)
                widget.setChecked(bool(default) if default is not None else False)
            else:
                # Default to line edit for unknown types
                widget = QtWidgets.QLineEdit(self._algo_params_container)
                widget.setText(str(default) if default is not None else "")

            widget.setToolTip(help_text)
            self._algo_params_layout.addWidget(widget, row, col * 2 + 1)
            self._algo_param_widgets[name] = widget

            # Move to next column, or next row
            col += 1
            if col >= 2:  # 2 columns
                col = 0
                row += 1

    def _format_field_name(self, name: str) -> str:
        """Format a field name for display.

        Args:
            name: Field name (e.g., "sgd_minibatch_size")

        Returns:
            Display name (e.g., "SGD Minibatch Size")
        """
        # Special cases
        name_map = {
            "lr": "Learning Rate",
            "gamma": "Discount (γ)",
            "lambda_": "GAE Lambda (λ)",
            "clip_param": "Clip Param",
            "vf_loss_coeff": "VF Loss Coeff",
            "entropy_coeff": "Entropy Coeff",
            "train_batch_size": "Train Batch",
            "sgd_minibatch_size": "SGD Minibatch",
            "num_sgd_iter": "SGD Epochs",
            "vtrace": "V-Trace",
            "use_kl_loss": "Use KL Loss",
            "tau": "Soft Update (τ)",
            "initial_alpha": "Initial Alpha",
            "n_step": "N-Step",
            "target_network_update_freq": "Target Update Freq",
            "double_q": "Double DQN",
            "dueling": "Dueling",
        }
        if name in name_map:
            return name_map[name]
        # Default: replace underscores with spaces and title case
        return name.replace("_", " ").title()

    def _get_fallback_fields(self, algorithm: str) -> List[Dict[str, Any]]:
        """Get fallback field definitions when algo_params module is unavailable."""
        # Basic PPO fields as fallback
        common_fields = [
            {"name": "lr", "type": "float", "default": 0.0003, "min": 1e-8, "max": 0.1, "help": "Learning rate"},
            {"name": "gamma", "type": "float", "default": 0.99, "min": 0.0, "max": 1.0, "help": "Discount factor"},
            {"name": "train_batch_size", "type": "int", "default": 4000, "min": 128, "max": 65536, "help": "Train batch size"},
        ]

        if algorithm == "PPO":
            return common_fields + [
                {"name": "sgd_minibatch_size", "type": "int", "default": 128, "min": 16, "max": 8192, "help": "SGD minibatch size"},
                {"name": "num_sgd_iter", "type": "int", "default": 30, "min": 1, "max": 100, "help": "Number of SGD epochs"},
                {"name": "clip_param", "type": "float", "default": 0.3, "min": 0.01, "max": 1.0, "help": "PPO clip parameter"},
            ]
        elif algorithm == "APPO":
            return common_fields + [
                {"name": "clip_param", "type": "float", "default": 0.4, "min": 0.01, "max": 1.0, "help": "PPO clipping parameter"},
                {"name": "vf_loss_coeff", "type": "float", "default": 0.5, "min": 0.0, "max": 10.0, "help": "Value function loss coefficient"},
                {"name": "entropy_coeff", "type": "float", "default": 0.01, "min": 0.0, "max": 1.0, "help": "Entropy coefficient"},
                {"name": "vtrace", "type": "bool", "default": True, "help": "Use V-trace (replaces GAE)"},
                {"name": "use_kl_loss", "type": "bool", "default": False, "help": "Use KL divergence loss"},
            ]
        elif algorithm == "IMPALA":
            return common_fields + [
                {"name": "vf_loss_coeff", "type": "float", "default": 0.5, "min": 0.0, "max": 10.0, "help": "Value function loss coefficient"},
                {"name": "entropy_coeff", "type": "float", "default": 0.01, "min": 0.0, "max": 1.0, "help": "Entropy coefficient"},
                {"name": "vtrace", "type": "bool", "default": True, "help": "Use V-trace"},
            ]
        elif algorithm == "DQN":
            return common_fields + [
                {"name": "n_step", "type": "int", "default": 3, "min": 1, "max": 10, "help": "N-step returns"},
                {"name": "double_q", "type": "bool", "default": True, "help": "Use Double DQN"},
            ]
        elif algorithm == "SAC":
            return common_fields + [
                {"name": "tau", "type": "float", "default": 0.005, "min": 0.001, "max": 1.0, "help": "Soft update coefficient"},
                {"name": "initial_alpha", "type": "float", "default": 1.0, "min": 0.01, "max": 10.0, "help": "Initial entropy coefficient"},
            ]
        return common_fields

    def _get_algo_params_values(self) -> Dict[str, Any]:
        """Get current values from all algorithm parameter widgets.

        Returns:
            Dictionary of parameter name -> value
        """
        params = {}
        for name, widget in self._algo_param_widgets.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QtWidgets.QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text()
                # Try to convert to appropriate type
                try:
                    params[name] = float(text) if "." in text else int(text)
                except ValueError:
                    params[name] = text
        return params

    def _create_resources_group(self) -> QtWidgets.QGroupBox:
        """Create resources configuration group."""
        group = QtWidgets.QGroupBox("Ray Resources", self)
        layout = QtWidgets.QGridLayout(group)

        # Detect available resources
        gpu_count, gpu_name = self._detect_gpus()
        self._available_cpus = _get_available_cpus()

        # Num workers with Auto checkbox
        layout.addWidget(QtWidgets.QLabel("Rollout Workers:"), 0, 0)

        workers_layout = QtWidgets.QHBoxLayout()
        workers_layout.setSpacing(4)

        self._workers_spin = QtWidgets.QSpinBox(group)
        self._workers_spin.setRange(0, 32)
        self._workers_spin.setValue(2)
        self._workers_spin.setToolTip(
            "Number of remote rollout workers (active samplers).\n"
            "Worker 0 is the coordinator, Workers 1..N do sampling.\n"
            "Example: num_workers=2 → W1, W2 active (W0 coordinates)"
        )
        workers_layout.addWidget(self._workers_spin)

        self._auto_workers_checkbox = QtWidgets.QCheckBox("Auto", group)
        self._auto_workers_checkbox.setChecked(True)
        self._auto_workers_checkbox.setToolTip(
            f"Auto-calculate based on environment agents and CPUs.\n"
            f"Detected {self._available_cpus} CPU cores."
        )
        self._auto_workers_checkbox.toggled.connect(self._on_auto_workers_toggled)
        workers_layout.addWidget(self._auto_workers_checkbox)

        layout.addLayout(workers_layout, 0, 1)

        # Num GPUs - now an integer with detected count
        layout.addWidget(QtWidgets.QLabel("GPUs:"), 0, 2)
        self._gpu_spin = QtWidgets.QSpinBox(group)
        self._gpu_spin.setRange(0, max(gpu_count, 8))
        self._gpu_spin.setValue(1 if gpu_count > 0 else 0)
        if gpu_name:
            self._gpu_spin.setToolTip(f"Detected: {gpu_name} ({gpu_count} available)")
        else:
            self._gpu_spin.setToolTip("No GPU detected - CPU only")
        layout.addWidget(self._gpu_spin, 0, 3)

        # CPUs per worker
        layout.addWidget(QtWidgets.QLabel("CPUs/Worker:"), 1, 0)
        self._cpus_per_worker_spin = QtWidgets.QSpinBox(group)
        self._cpus_per_worker_spin.setRange(1, 16)
        self._cpus_per_worker_spin.setValue(1)
        layout.addWidget(self._cpus_per_worker_spin, 1, 1)

        # GPU info label
        if gpu_count > 0:
            gpu_info = QtWidgets.QLabel(f"✓ {gpu_name}", group)
            gpu_info.setStyleSheet("color: green; font-size: 10px;")
        else:
            gpu_info = QtWidgets.QLabel("No GPU detected", group)
            gpu_info.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(gpu_info, 1, 2, 1, 2)

        # CPU info label
        cpu_info = QtWidgets.QLabel(f"({self._available_cpus} CPUs available)", group)
        cpu_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(cpu_info, 2, 0, 1, 2)

        return group

    def _on_auto_workers_toggled(self, checked: bool) -> None:
        """Handle Auto workers checkbox toggle."""
        self._workers_spin.setEnabled(not checked)
        if checked:
            self._update_auto_workers()

    def _create_analytics_group(self) -> QtWidgets.QGroupBox:
        """Create analytics & tracking configuration group."""
        group = QtWidgets.QGroupBox("Analytics & Tracking", self)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Hint label
        hint_label = QtWidgets.QLabel(
            "Select analytics to export during and after training.",
            group,
        )
        hint_label.setStyleSheet("color: #777777; font-size: 11px;")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        # Checkbox row
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Export TensorBoard", group)
        self._tensorboard_checkbox.setChecked(True)
        self._tensorboard_checkbox.setToolTip(
            "Write TensorBoard event files to var/trainer/runs/<run_id>/tensorboard"
        )
        checkbox_layout.addWidget(self._tensorboard_checkbox)

        self._wandb_checkbox = QtWidgets.QCheckBox("Export WandB", group)
        self._wandb_checkbox.setChecked(False)
        self._wandb_checkbox.setToolTip("Requires wandb login on the trainer host")
        self._wandb_checkbox.toggled.connect(self._on_wandb_toggled)
        checkbox_layout.addWidget(self._wandb_checkbox)

        checkbox_layout.addStretch(1)
        layout.addLayout(checkbox_layout)

        # WandB configuration section
        wandb_container = QtWidgets.QWidget(group)
        wandb_layout = QtWidgets.QFormLayout(wandb_container)
        wandb_layout.setContentsMargins(0, 4, 0, 0)
        wandb_layout.setSpacing(4)

        self._wandb_project_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_project_input.setPlaceholderText("e.g. ray-marl")
        self._wandb_project_input.setToolTip(
            "Project name inside wandb.ai where runs will be grouped."
        )
        wandb_layout.addRow("Project:", self._wandb_project_input)

        self._wandb_entity_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_entity_input.setPlaceholderText("e.g. your-username")
        self._wandb_entity_input.setToolTip(
            "WandB entity (team or user namespace) to publish to."
        )
        wandb_layout.addRow("Entity:", self._wandb_entity_input)

        self._wandb_run_name_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_run_name_input.setPlaceholderText("Optional custom run name")
        self._wandb_run_name_input.setToolTip(
            "Custom run name (defaults to run_id if not specified)."
        )
        wandb_layout.addRow("Run Name:", self._wandb_run_name_input)

        self._wandb_api_key_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._wandb_api_key_input.setToolTip(
            "Override the default WandB API key (from wandb login)."
        )
        wandb_layout.addRow("API Key:", self._wandb_api_key_input)

        # VPN proxy settings
        self._wandb_use_vpn_checkbox = QtWidgets.QCheckBox(
            "Route WandB traffic through VPN proxy", wandb_container
        )
        self._wandb_use_vpn_checkbox.toggled.connect(self._on_wandb_vpn_toggled)
        wandb_layout.addRow("", self._wandb_use_vpn_checkbox)

        self._wandb_http_proxy_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_http_proxy_input.setPlaceholderText(
            "e.g. http://127.0.0.1:7890"
        )
        self._wandb_http_proxy_input.setToolTip("HTTP proxy for WandB traffic.")
        wandb_layout.addRow("HTTP Proxy:", self._wandb_http_proxy_input)

        self._wandb_https_proxy_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_https_proxy_input.setPlaceholderText(
            "e.g. http://127.0.0.1:7890"
        )
        self._wandb_https_proxy_input.setToolTip("HTTPS proxy for WandB traffic.")
        wandb_layout.addRow("HTTPS Proxy:", self._wandb_https_proxy_input)

        layout.addWidget(wandb_container)
        self._wandb_container = wandb_container

        # Initialize WandB control states
        self._update_wandb_controls()

        return group

    def _on_wandb_toggled(self, checked: bool) -> None:
        """Handle WandB checkbox toggle."""
        _ = checked
        self._update_wandb_controls()

    def _on_wandb_vpn_toggled(self, checked: bool) -> None:
        """Handle WandB VPN checkbox toggle."""
        _ = checked
        self._update_wandb_controls()

    def _update_wandb_controls(self) -> None:
        """Update WandB control enabled states based on checkbox state."""
        wandb_enabled = self._wandb_checkbox.isChecked()

        # Base WandB fields
        base_fields = (
            self._wandb_project_input,
            self._wandb_entity_input,
            self._wandb_run_name_input,
            self._wandb_api_key_input,
        )
        for field in base_fields:
            field.setEnabled(wandb_enabled)

        # VPN checkbox
        self._wandb_use_vpn_checkbox.setEnabled(wandb_enabled)
        if not wandb_enabled:
            self._wandb_use_vpn_checkbox.setChecked(False)

        # Proxy fields (only enabled if VPN is checked)
        vpn_enabled = wandb_enabled and self._wandb_use_vpn_checkbox.isChecked()
        self._wandb_http_proxy_input.setEnabled(vpn_enabled)
        self._wandb_https_proxy_input.setEnabled(vpn_enabled)

    def _create_fastlane_group(self) -> QtWidgets.QGroupBox:
        """Create FastLane live visualization configuration group.

        Each Ray worker gets its own FastLane tile showing its environment state.
        With num_workers=2, you'll see 3 tiles (workers 0, 1, 2) tiled together.
        """
        group = QtWidgets.QGroupBox("Live Visualization (FastLane)", self)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Description label
        desc_label = QtWidgets.QLabel(
            "Each Ray worker gets its own display tile showing live training frames. "
            "Disable FastLane for faster training (no rendering overhead).",
            group,
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(desc_label)

        # Checkbox row
        checkbox_row = QtWidgets.QHBoxLayout()

        # FastLane enabled checkbox
        self._fastlane_checkbox = QtWidgets.QCheckBox("Enable FastLane streaming", group)
        self._fastlane_checkbox.setChecked(True)
        self._fastlane_checkbox.setToolTip(
            "Stream live training frames to Render view via shared memory.\n"
            "Each worker gets its own tile (num_workers=2 → 3 tiles).\n"
            "Disable for faster training without rendering overhead."
        )
        self._fastlane_checkbox.toggled.connect(self._on_fastlane_toggled)
        checkbox_row.addWidget(self._fastlane_checkbox)

        checkbox_row.addSpacing(20)

        # FastLane Only (skip persistence)
        self._fastlane_only_checkbox = QtWidgets.QCheckBox("FastLane Only (skip gRPC telemetry)", group)
        self._fastlane_only_checkbox.setChecked(True)
        self._fastlane_only_checkbox.setToolTip(
            "Disable slow lane (gRPC/SQLite) so only shared-memory FastLane runs"
        )
        checkbox_row.addWidget(self._fastlane_only_checkbox)

        checkbox_row.addStretch(1)
        layout.addLayout(checkbox_row)

        # Throttle row
        throttle_row = QtWidgets.QHBoxLayout()

        throttle_label = QtWidgets.QLabel("Frame throttle:")
        throttle_row.addWidget(throttle_label)

        self._throttle_spin = QtWidgets.QSpinBox(group)
        self._throttle_spin.setRange(16, 1000)
        self._throttle_spin.setValue(33)  # ~30 FPS
        self._throttle_spin.setToolTip("Minimum milliseconds between frame updates (~30 FPS at 33ms)")
        self._throttle_spin.setSuffix(" ms")
        throttle_row.addWidget(self._throttle_spin)

        fps_hint = QtWidgets.QLabel("(33ms ≈ 30 FPS)")
        fps_hint.setStyleSheet("color: #888; font-size: 10px;")
        throttle_row.addWidget(fps_hint)

        throttle_row.addStretch(1)
        layout.addLayout(throttle_row)

        # Initialize control states
        self._update_fastlane_controls()

        return group

    def _create_checkpoint_group(self) -> QtWidgets.QGroupBox:
        """Create checkpoint configuration group."""
        group = QtWidgets.QGroupBox("Model Checkpoints", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Checkpoint options row
        checkbox_row = QtWidgets.QHBoxLayout()

        # Save model checkbox
        self._save_model_checkbox = QtWidgets.QCheckBox("Save trained model", group)
        self._save_model_checkbox.setChecked(True)
        self._save_model_checkbox.setToolTip(
            "Save the trained model checkpoint when training completes"
        )
        checkbox_row.addWidget(self._save_model_checkbox)

        checkbox_row.addSpacing(20)

        # Checkpoint at end only
        self._checkpoint_at_end_checkbox = QtWidgets.QCheckBox("Save at end only", group)
        self._checkpoint_at_end_checkbox.setChecked(True)
        self._checkpoint_at_end_checkbox.setToolTip("Only save final policy, no intermediate checkpoints")
        self._checkpoint_at_end_checkbox.stateChanged.connect(self._on_checkpoint_mode_changed)
        checkbox_row.addWidget(self._checkpoint_at_end_checkbox)

        checkbox_row.addStretch(1)
        layout.addLayout(checkbox_row)

        # Interval checkpoint row
        interval_row = QtWidgets.QHBoxLayout()

        self._checkpoint_interval_label = QtWidgets.QLabel("Also save every:")
        self._checkpoint_interval_label.setEnabled(False)
        interval_row.addWidget(self._checkpoint_interval_label)

        self._checkpoint_freq_spin = QtWidgets.QSpinBox(group)
        self._checkpoint_freq_spin.setRange(1, 1000)
        self._checkpoint_freq_spin.setValue(10)
        self._checkpoint_freq_spin.setEnabled(False)
        self._checkpoint_freq_spin.setToolTip("Save intermediate checkpoints every N training iterations")
        interval_row.addWidget(self._checkpoint_freq_spin)

        self._checkpoint_unit_label = QtWidgets.QLabel("iterations")
        self._checkpoint_unit_label.setEnabled(False)
        interval_row.addWidget(self._checkpoint_unit_label)

        interval_row.addStretch(1)
        layout.addLayout(interval_row)

        return group

    def _detect_gpus(self) -> tuple[int, str]:
        """Detect available GPUs.

        Returns:
            Tuple of (gpu_count, gpu_name)
        """
        # Try torch first
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0) if count > 0 else ""
                return count, name
        except ImportError:
            pass

        # Try ray
        try:
            import ray
            if ray.is_initialized():
                resources = ray.available_resources()
                gpu_count = int(resources.get("GPU", 0))
                if gpu_count > 0:
                    return gpu_count, "GPU"
        except Exception:
            pass

        # Fallback: check nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpus = [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]
                if gpus:
                    return len(gpus), gpus[0]
        except Exception:
            pass

        return 0, ""

    def _on_checkpoint_mode_changed(self, state: int) -> None:
        """Handle checkpoint mode checkbox change."""
        at_end_only = state == QtCore.Qt.CheckState.Checked.value
        self._checkpoint_interval_label.setEnabled(not at_end_only)
        self._checkpoint_freq_spin.setEnabled(not at_end_only)
        self._checkpoint_unit_label.setEnabled(not at_end_only)

    def _on_fastlane_toggled(self, checked: bool) -> None:
        """Handle FastLane enable/disable toggle."""
        self._update_fastlane_controls()

    def _update_fastlane_controls(self) -> None:
        """Update FastLane control enable states."""
        fastlane_enabled = self._fastlane_checkbox.isChecked()

        # Disable dependent controls if FastLane is off
        self._fastlane_only_checkbox.setEnabled(fastlane_enabled)
        self._throttle_spin.setEnabled(fastlane_enabled)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)

    def _populate_families(self) -> None:
        """Populate the family dropdown."""
        self._family_combo.clear()
        # Prioritize SISL for cooperative training
        self._family_combo.addItem("SISL (Continuous Cooperative)", PettingZooFamily.SISL.value)
        self._family_combo.addItem("Classic (Turn-based)", PettingZooFamily.CLASSIC.value)
        self._family_combo.addItem("MPE (Particle)", PettingZooFamily.MPE.value)
        self._family_combo.addItem("Butterfly (Visual)", PettingZooFamily.BUTTERFLY.value)
        self._family_combo.addItem("Atari (2-Player)", PettingZooFamily.ATARI.value)

    def _select_family(self, family: str) -> None:
        """Select a family by value."""
        for i in range(self._family_combo.count()):
            if self._family_combo.itemData(i) == family:
                self._family_combo.setCurrentIndex(i)
                break

    def _select_env(self, env_id: str) -> None:
        """Select an environment by ID."""
        for i in range(self._env_combo.count()):
            if self._env_combo.itemData(i) == env_id:
                self._env_combo.setCurrentIndex(i)
                break

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_value = self._family_combo.currentData()
        if not family_value:
            return

        try:
            family = PettingZooFamily(family_value)
        except ValueError:
            return

        # Populate environments
        self._env_combo.clear()
        envs = get_envs_by_family(family)

        for env_id in envs:
            display_name = get_display_name(env_id)
            self._env_combo.addItem(display_name, env_id.value)

        if envs:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info_label.setText("")
            self._api_type_label.setText("")
            return

        try:
            env_id = PettingZooEnvId(env_value)
        except ValueError:
            return

        # Update description
        description = get_description(env_id)
        self._env_info_label.setText(description)

        # Update API type
        api_type = get_api_type(env_id)
        if api_type == PettingZooAPIType.AEC:
            self._api_type_label.setText("AEC (Turn-based)")
            self._api_type_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        else:
            self._api_type_label.setText("Parallel (Simultaneous)")
            self._api_type_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

        # Auto-update rollout workers if Auto is checked
        if hasattr(self, '_auto_workers_checkbox') and self._auto_workers_checkbox.isChecked():
            self._update_auto_workers()

    def _update_auto_workers(self) -> None:
        """Update the rollout workers count based on environment and available CPUs."""
        env_id = self._env_combo.currentData()
        if not env_id:
            return

        optimal_workers = _calculate_optimal_workers(env_id, self._available_cpus)
        self._workers_spin.setValue(optimal_workers)

        # Update tooltip with explanation
        num_agents = _get_default_agent_count(env_id)
        self._workers_spin.setToolTip(
            f"Auto-calculated: {optimal_workers} workers\n"
            f"Environment: {env_id} ({num_agents} agents)\n"
            f"Available CPUs: {self._available_cpus}\n\n"
            f"Workers 1..{optimal_workers} will do sampling.\n"
            f"Worker 0 coordinates (doesn't sample)."
        )

    def _on_algorithm_changed(self, index: int) -> None:
        """Handle algorithm selection change."""
        algo_id = self._algo_combo.currentData()
        if algo_id and algo_id in _ALGORITHMS:
            algo_info = _ALGORITHMS[algo_id]
            self._algo_notes_text.setHtml(algo_info["description"])

            # Rebuild the dynamic algorithm parameters panel
            if hasattr(self, "_algo_params_layout"):
                self._rebuild_algo_params(algo_id)
        else:
            self._algo_notes_text.setHtml("")

    def _get_selected_paradigm(self) -> str:
        """Get the selected training paradigm ID."""
        checked_button = self._paradigm_group.checkedButton()
        if checked_button:
            return checked_button.property("paradigm_id")
        return "parameter_sharing"

    def _on_validate_clicked(self) -> None:
        """Handle Validate button click - run dry-run validation."""
        self._validation_status_label.setText("Running validation...")
        self._validation_status_label.setStyleSheet("color: #1565c0;")
        QtWidgets.QApplication.processEvents()

        try:
            # Build the config
            config = self._build_config()

            # For now, do basic validation checks
            env_id = self._env_combo.currentData()
            algorithm = self._algo_combo.currentData()

            errors = []

            if not env_id:
                errors.append("No environment selected")

            if not algorithm:
                errors.append("No algorithm selected")

            # Check for incompatible combinations
            if algorithm == "DQN":
                family = self._family_combo.currentData()
                if family == "sisl":
                    errors.append("DQN requires discrete actions - SISL environments have continuous actions")

            if errors:
                self._validation_status_label.setText(
                    f"✖ Validation failed: {'; '.join(errors)}"
                )
                self._validation_status_label.setStyleSheet("color: #c62828;")
            else:
                self._validation_status_label.setText(
                    "✔ Validation passed. Configuration is valid."
                )
                self._validation_status_label.setStyleSheet("color: #2e7d32;")

        except Exception as e:
            self._validation_status_label.setText(f"✖ Validation error: {e}")
            self._validation_status_label.setStyleSheet("color: #c62828;")

    def _on_accept(self) -> None:
        """Handle OK button click."""
        # Run validation first
        self._on_validate_clicked()

        if "✖" in self._validation_status_label.text():
            # Validation failed, don't accept
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Failed",
                "Please fix the configuration issues before starting training.",
            )
            return

        self._last_config = self._build_config()
        self.accept()

    def _build_config(self) -> Dict[str, Any]:
        """Build the training configuration dictionary."""
        env_id = self._env_combo.currentData() or "waterworld_v4"
        family = self._family_combo.currentData() or "sisl"
        algorithm = self._algo_combo.currentData() or "PPO"
        paradigm = self._get_selected_paradigm()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"ray-{env_id.replace('_', '-')}-{timestamp}"

        # Determine API type
        api_type = "parallel"
        try:
            pz_env_id = PettingZooEnvId(env_id)
            api_type = "aec" if is_aec_env(pz_env_id) else "parallel"
        except ValueError:
            pass

        # Get FastLane settings (multi-agent composite view is automatic)
        fastlane_enabled = self._fastlane_checkbox.isChecked()
        fastlane_only = self._fastlane_only_checkbox.isChecked()
        throttle_ms = self._throttle_spin.value()

        # Get WandB settings
        track_wandb = self._wandb_checkbox.isChecked()
        wandb_project = self._wandb_project_input.text().strip() or None
        wandb_entity = self._wandb_entity_input.text().strip() or None
        wandb_run_name = self._wandb_run_name_input.text().strip() or None
        wandb_api_key = self._wandb_api_key_input.text().strip() or None
        use_wandb_vpn = self._wandb_use_vpn_checkbox.isChecked()
        wandb_http_proxy = self._wandb_http_proxy_input.text().strip() or None
        wandb_https_proxy = self._wandb_https_proxy_input.text().strip() or None

        # Build environment variables for multi-agent FastLane
        # NOTE: RAY_FASTLANE_RUN_ID is NOT set here - it will be set by dispatcher
        # using the actual ULID run_id, not the human-readable run_name
        environment: Dict[str, str] = {
            "RAY_FASTLANE_ENABLED": "1" if fastlane_enabled else "0",
            "RAY_FASTLANE_ENV_NAME": env_id,
            "RAY_FASTLANE_THROTTLE_MS": str(throttle_ms),
            "GYM_GUI_FASTLANE_ONLY": "1" if fastlane_only else "0",
        }

        # Add WandB environment variables
        if track_wandb:
            environment["WANDB_MODE"] = "online"
            if wandb_api_key:
                environment["WANDB_API_KEY"] = wandb_api_key
            if use_wandb_vpn and wandb_http_proxy:
                environment["WANDB_HTTP_PROXY"] = wandb_http_proxy
                environment["HTTP_PROXY"] = wandb_http_proxy
                environment["http_proxy"] = wandb_http_proxy
            if use_wandb_vpn and wandb_https_proxy:
                environment["WANDB_HTTPS_PROXY"] = wandb_https_proxy
                environment["HTTPS_PROXY"] = wandb_https_proxy
                environment["https_proxy"] = wandb_https_proxy
        else:
            environment["WANDB_MODE"] = "offline"

        config = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "ray_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": max(1, self._workers_spin.value() * self._cpus_per_worker_spin.value()),
                "memory_mb": 4096,
                "gpus": {
                    "requested": int(self._gpu_spin.value()),
                    "mandatory": False,
                },
            },
            "artifacts": {
                "output_prefix": f"runs/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": self._save_model_checkbox.isChecked(),
            },
            "metadata": {
                "ui": {
                    "worker_id": "ray_worker",
                    "env_id": env_id,
                    "family": family,
                    "algorithm": algorithm,
                    "paradigm": paradigm,
                    "mode": "training",
                    "fastlane_enabled": fastlane_enabled,
                    "fastlane_only": fastlane_only,
                },
                "worker": {
                    "worker_id": "ray_worker",
                    "module": "ray_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "environment": {
                            "family": family,
                            "env_id": env_id,
                            "api_type": api_type,
                        },
                        "paradigm": paradigm,
                        "training": {
                            "algorithm": algorithm,
                            "total_timesteps": self._timesteps_spin.value(),
                            "algo_params": self._get_algo_params_values(),
                        },
                        "resources": {
                            "num_workers": self._workers_spin.value(),
                            "num_gpus": self._gpu_spin.value(),
                            "num_cpus_per_worker": self._cpus_per_worker_spin.value(),
                        },
                        "checkpoint": {
                            "checkpoint_freq": self._checkpoint_freq_spin.value() if not self._checkpoint_at_end_checkbox.isChecked() else 0,
                            "checkpoint_at_end": self._save_model_checkbox.isChecked(),
                            "export_policy": self._save_model_checkbox.isChecked(),
                        },
                        "fastlane_enabled": fastlane_enabled,
                        "fastlane_throttle_ms": throttle_ms,
                        "seed": self._seed_spin.value() if self._seed_spin.value() > 0 else None,
                        "tensorboard": self._tensorboard_checkbox.isChecked(),
                        "wandb": track_wandb,
                        "wandb_project": wandb_project,
                        "wandb_entity": wandb_entity,
                        "wandb_run_name": wandb_run_name,
                        "extras": {
                            "fastlane_only": fastlane_only,
                            "save_model": self._save_model_checkbox.isChecked(),
                            "track_wandb": track_wandb,
                            "wandb_project": wandb_project,
                            "wandb_entity": wandb_entity,
                            "wandb_run_name": wandb_run_name,
                            "wandb_use_vpn_proxy": use_wandb_vpn,
                        },
                    },
                },
                "artifacts": {
                    "tensorboard": {
                        "enabled": self._tensorboard_checkbox.isChecked(),
                        "relative_path": "tensorboard",
                    },
                    "wandb": {
                        "enabled": track_wandb,
                        "project": wandb_project or "ray-marl",
                        "entity": wandb_entity,
                        "run_name": wandb_run_name,
                        "use_vpn_proxy": use_wandb_vpn,
                        "http_proxy": wandb_http_proxy if use_wandb_vpn else None,
                        "https_proxy": wandb_https_proxy if use_wandb_vpn else None,
                    },
                    "fastlane": {
                        "enabled": fastlane_enabled,
                        "mode": "composite",  # Multi-agent composite view
                        "throttle_ms": throttle_ms,
                    },
                },
            },
        }

        _LOGGER.info(
            "Built Ray RLlib training config: env=%s, algo=%s, paradigm=%s, timesteps=%d, wandb=%s",
            env_id,
            algorithm,
            paradigm,
            self._timesteps_spin.value(),
            track_wandb,
        )

        return config

    def get_config(self) -> Dict[str, Any]:
        """Return the training configuration.

        Returns:
            dict: Configuration dictionary for TrainerClient submission
        """
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        return self._build_config()


# Register form with factory at module load
def _register_ray_train_form() -> None:
    """Register Ray RLlib train form with factory (deferred to avoid circular import)."""
    from gym_gui.ui.forms import get_worker_form_factory
    factory = get_worker_form_factory()
    if not factory.has_train_form("ray_worker"):
        factory.register_train_form(
            "ray_worker",
            lambda parent=None, **kwargs: RayRLlibTrainForm(parent=parent, **kwargs),
        )

_register_ray_train_form()


__all__ = ["RayRLlibTrainForm"]
