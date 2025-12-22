"""Multi-operator configuration widget for side-by-side agent comparison.

This module provides UI widgets for configuring N operators (LLM or RL workers)
that can run in parallel, each with its own render container.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_MODELS_HF_CACHE
from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.worker_catalog.catalog import get_worker_catalog, WorkerDefinition
from gym_gui.constants.constants_operator import (
    BARLOG_SUPPORTED_ENVS,
    BARLOG_DEFAULT_TASK,
)

_LOGGER = logging.getLogger(__name__)


def scan_local_models() -> List[Tuple[str, str]]:
    """Scan var/models/huggingface for installed local models.

    Returns:
        List of (model_id, display_name) tuples for locally installed models.
    """
    models = []
    hf_cache = VAR_MODELS_HF_CACHE

    if not hf_cache.exists():
        _LOGGER.debug("HuggingFace model cache not found: %s", hf_cache)
        return models

    for model_dir in hf_cache.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("."):
            continue

        # Convert directory name to HuggingFace model ID format
        # e.g. "Qwen--Qwen2.5-Coder-7B-Instruct" -> "Qwen/Qwen2.5-Coder-7B-Instruct"
        # e.g. "Llama-3.1-8B-Instruct" -> "meta-llama/Llama-3.1-8B-Instruct"
        dir_name = model_dir.name

        if "--" in dir_name:
            # Format: "org--model-name"
            org, model_name = dir_name.split("--", 1)
            model_id = f"{org}/{model_name}"
            display_name = model_name
        elif dir_name.startswith("Llama"):
            # Meta Llama models often stored without org prefix
            model_id = f"meta-llama/{dir_name}"
            display_name = dir_name
        else:
            # Use as-is
            model_id = dir_name
            display_name = dir_name

        models.append((model_id, display_name))
        _LOGGER.debug("Found local model: %s", model_id)

    # Sort by display name
    models.sort(key=lambda x: x[1])
    return models


def get_vllm_models() -> List[Tuple[str, str]]:
    """Get available vLLM models (scanned from local cache).

    Returns:
        List of (model_id, display_name) tuples.
        Returns default suggestions if no local models found.
    """
    local_models = scan_local_models()
    if local_models:
        return local_models

    # Fallback if no models installed
    return [
        ("(No local models found)", "(Install models to var/models/huggingface)"),
    ]


# Maximum number of operators allowed
MAX_OPERATORS = 8

# Default environments for RL workers (Gymnasium)
RL_SUPPORTED_ENVS = (
    "FrozenLake-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2",
    "BipedalWalker-v3",
    "Taxi-v3",
    "CliffWalking-v0",
)

# LLM Client configurations
# Maps client_name -> (display_name, requires_api_key, default_base_url)
LLM_CLIENTS: Dict[str, Tuple[str, bool, Optional[str]]] = {
    "openrouter": ("OpenRouter", True, "https://openrouter.ai/api/v1"),
    "vllm": ("vLLM (Local)", False, "http://localhost:8000/v1"),
    "openai": ("OpenAI (Direct)", True, None),
    "anthropic": ("Anthropic (Direct)", True, None),
    "google": ("Google (Direct)", True, None),
}

# Default models for each LLM client
# OpenRouter provides access to all major models via a unified API
LLM_CLIENT_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "openrouter": [
        # OpenAI models via OpenRouter
        ("openai/gpt-4o-mini", "GPT-4o Mini"),
        ("openai/gpt-4o", "GPT-4o"),
        ("openai/gpt-4-turbo", "GPT-4 Turbo"),
        # Anthropic models via OpenRouter
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("anthropic/claude-3.5-haiku", "Claude 3.5 Haiku"),
        ("anthropic/claude-3-opus", "Claude 3 Opus"),
        # Google models via OpenRouter
        ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash (Free)"),
        ("google/gemini-pro-1.5", "Gemini 1.5 Pro"),
        ("google/gemini-flash-1.5", "Gemini 1.5 Flash"),
        # Meta Llama models via OpenRouter
        ("meta-llama/llama-3.1-8b-instruct:free", "Llama 3.1 8B (Free)"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B"),
        ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
        # Mistral models via OpenRouter
        ("mistralai/mistral-7b-instruct:free", "Mistral 7B (Free)"),
        ("mistralai/mixtral-8x7b-instruct", "Mixtral 8x7B"),
        # DeepSeek models via OpenRouter
        ("deepseek/deepseek-chat", "DeepSeek Chat"),
        ("deepseek/deepseek-r1", "DeepSeek R1"),
    ],
    # vLLM models are dynamically scanned - see get_vllm_models()
    "vllm": [],
    "openai": [
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ],
    "anthropic": [
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
        ("claude-3-opus-20240229", "Claude 3 Opus"),
    ],
    "google": [
        ("gemini-2.0-flash-exp", "Gemini 2.0 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ],
}


def _get_llm_workers() -> List[WorkerDefinition]:
    """Get LLM workers from catalog (supports_training=False)."""
    return [w for w in get_worker_catalog() if not w.supports_training]


def _get_rl_workers() -> List[WorkerDefinition]:
    """Get RL workers from catalog (supports_training=True)."""
    return [w for w in get_worker_catalog() if w.supports_training]


class OperatorConfigRow(QtWidgets.QWidget):
    """Single row in the operator configuration list.

    Each row represents one operator with:
    - Row 1: Display name, Type selector, Worker dropdown, Remove button
    - Row 2 (LLM): Client selector, Model selector, API Key, Environment, Task
    - Row 2 (RL): Environment, Task, Policy Path with Browse
    """

    config_changed = pyqtSignal(str, object)  # operator_id, new_config
    remove_requested = pyqtSignal(str)  # operator_id
    initialize_requested = pyqtSignal(str)  # operator_id - request to preview env

    def __init__(
        self,
        operator_id: str,
        initial_config: Optional[OperatorConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._operator_id = operator_id
        self._updating = False  # Prevent signal loops

        self._build_ui()
        self._connect_signals()

        if initial_config:
            self._load_config(initial_config)

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ============================================================
        # Row 1: Identity - Index, Name, Type, Worker, Remove button
        # ============================================================
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(8)

        # Operator index label
        self._index_label = QtWidgets.QLabel(f"#{self._operator_id[-1]}", self)
        self._index_label.setFixedWidth(20)
        self._index_label.setStyleSheet("font-weight: bold; color: #666;")
        row1.addWidget(self._index_label)

        # Display name
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setPlaceholderText("Name")
        self._name_edit.setFixedWidth(120)
        row1.addWidget(self._name_edit)

        # Type selector (LLM / RL)
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.addItems(["LLM", "RL"])
        self._type_combo.setFixedWidth(60)
        row1.addWidget(self._type_combo)

        # Worker dropdown
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setMinimumWidth(130)
        row1.addWidget(self._worker_combo)

        row1.addStretch()

        # Remove button (red X)
        self._remove_btn = QtWidgets.QPushButton("✕", self)
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setToolTip("Remove this operator")
        self._remove_btn.setStyleSheet(
            "QPushButton { color: #c00; font-weight: bold; border: none; }"
            "QPushButton:hover { color: #f00; background-color: #fee; border-radius: 3px; }"
        )
        row1.addWidget(self._remove_btn)

        main_layout.addLayout(row1)

        # ============================================================
        # Row 2: Environment - Env dropdown, Task dropdown, Load button
        # ============================================================
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(8)

        row2.addWidget(QtWidgets.QLabel("Environment:", self))
        self._env_combo = QtWidgets.QComboBox(self)
        self._env_combo.setMinimumWidth(100)
        row2.addWidget(self._env_combo)

        row2.addWidget(QtWidgets.QLabel("Task:", self))
        self._task_combo = QtWidgets.QComboBox(self)
        self._task_combo.setMinimumWidth(180)
        row2.addWidget(self._task_combo)

        # Load Environment button - mandatory to initialize the environment
        self._init_btn = QtWidgets.QPushButton("Load Environment", self)
        self._init_btn.setToolTip("Load and initialize this environment for the operator")
        self._init_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
            "border-radius: 3px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        row2.addWidget(self._init_btn)

        row2.addStretch()

        main_layout.addLayout(row2)

        # ============================================================
        # Row 3: Type-specific settings (LLM or RL)
        # ============================================================

        # === LLM-specific widgets ===
        self._llm_container = QtWidgets.QWidget(self)
        llm_layout = QtWidgets.QHBoxLayout(self._llm_container)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(8)

        # LLM Client selector
        llm_layout.addWidget(QtWidgets.QLabel("Provider:", self._llm_container))
        self._client_combo = QtWidgets.QComboBox(self._llm_container)
        self._client_combo.setMinimumWidth(110)
        for client_name, (display_name, _, _) in LLM_CLIENTS.items():
            self._client_combo.addItem(display_name, client_name)
        llm_layout.addWidget(self._client_combo)

        # LLM Model selector
        llm_layout.addWidget(QtWidgets.QLabel("Model:", self._llm_container))
        self._model_combo = QtWidgets.QComboBox(self._llm_container)
        self._model_combo.setMinimumWidth(140)
        llm_layout.addWidget(self._model_combo)

        llm_layout.addStretch()

        # API Key field
        self._api_key_label = QtWidgets.QLabel("API Key:", self._llm_container)
        llm_layout.addWidget(self._api_key_label)
        self._api_key_edit = QtWidgets.QLineEdit(self._llm_container)
        self._api_key_edit.setPlaceholderText("Set env var or enter key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.setMinimumWidth(160)
        llm_layout.addWidget(self._api_key_edit)

        # Show/hide API key toggle
        self._show_key_btn = QtWidgets.QPushButton("👁", self._llm_container)
        self._show_key_btn.setFixedWidth(28)
        self._show_key_btn.setToolTip("Show/hide API key")
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        llm_layout.addWidget(self._show_key_btn)

        main_layout.addWidget(self._llm_container)

        # === RL-specific widgets ===
        self._rl_container = QtWidgets.QWidget(self)
        rl_layout = QtWidgets.QHBoxLayout(self._rl_container)
        rl_layout.setContentsMargins(0, 0, 0, 0)
        rl_layout.setSpacing(8)

        # Policy Path field
        rl_layout.addWidget(QtWidgets.QLabel("Policy:", self._rl_container))
        self._policy_path_edit = QtWidgets.QLineEdit(self._rl_container)
        self._policy_path_edit.setPlaceholderText("Path to trained policy/checkpoint")
        rl_layout.addWidget(self._policy_path_edit)

        # Browse button
        self._browse_btn = QtWidgets.QPushButton("Browse...", self._rl_container)
        self._browse_btn.setFixedWidth(70)
        self._browse_btn.clicked.connect(self._on_browse_policy)
        rl_layout.addWidget(self._browse_btn)

        main_layout.addWidget(self._rl_container)

        # Initialize dropdowns
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._update_model_dropdown()
        self._update_type_specific_visibility()

    def _connect_signals(self) -> None:
        self._name_edit.textChanged.connect(self._on_config_changed)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_config_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._task_combo.currentIndexChanged.connect(self._on_config_changed)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._operator_id))
        self._init_btn.clicked.connect(lambda: self.initialize_requested.emit(self._operator_id))

        # LLM-specific signals
        self._client_combo.currentIndexChanged.connect(self._on_client_changed)
        self._model_combo.currentIndexChanged.connect(self._on_config_changed)
        self._api_key_edit.textChanged.connect(self._on_config_changed)

        # RL-specific signals
        self._policy_path_edit.textChanged.connect(self._on_config_changed)

    def _on_type_changed(self) -> None:
        """Handle operator type change (LLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._update_type_specific_visibility()
        self._on_config_changed()

    def _on_env_changed(self) -> None:
        """Handle environment change."""
        if self._updating:
            return
        self._update_task_dropdown()
        self._on_config_changed()

    def _on_client_changed(self) -> None:
        """Handle LLM client change."""
        if self._updating:
            return
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self._on_config_changed()

    def _on_config_changed(self) -> None:
        """Emit config_changed signal with current configuration."""
        if self._updating:
            return
        config = self.get_config()
        self.config_changed.emit(self._operator_id, config)

    def _toggle_api_key_visibility(self) -> None:
        """Toggle API key visibility."""
        if self._show_key_btn.isChecked():
            self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText("Hide")
        else:
            self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self._show_key_btn.setText("Show")

    def _on_browse_policy(self) -> None:
        """Open file dialog to browse for policy file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Policy File",
            "",
            "Policy Files (*.pt *.pth *.zip *.pkl *.ckpt);;All Files (*)"
        )
        if file_path:
            self._policy_path_edit.setText(file_path)

    def _update_type_specific_visibility(self) -> None:
        """Show/hide LLM or RL specific widgets based on operator type."""
        operator_type = self._type_combo.currentText().lower()
        is_llm = operator_type == "llm"

        self._llm_container.setVisible(is_llm)
        self._rl_container.setVisible(not is_llm)

        if is_llm:
            self._update_api_key_visibility()

    def _update_api_key_visibility(self) -> None:
        """Show/hide API key field based on selected client."""
        client_name = self._client_combo.currentData()
        if client_name and client_name in LLM_CLIENTS:
            _, requires_api_key, _ = LLM_CLIENTS[client_name]
            self._api_key_label.setVisible(requires_api_key)
            self._api_key_edit.setVisible(requires_api_key)
            self._show_key_btn.setVisible(requires_api_key)

            # Update placeholder based on client
            if requires_api_key:
                env_var_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_API_KEY",
                }
                env_var = env_var_map.get(client_name, "")
                if env_var and os.environ.get(env_var):
                    self._api_key_edit.setPlaceholderText(f"Using {env_var} from env")
                else:
                    self._api_key_edit.setPlaceholderText(f"Enter key or set {env_var}")

    def _update_model_dropdown(self) -> None:
        """Update model dropdown based on selected LLM client."""
        self._updating = True
        current_model = self._model_combo.currentData()
        self._model_combo.clear()

        client_name = self._client_combo.currentData()
        if client_name:
            # vLLM uses dynamically scanned local models
            if client_name == "vllm":
                models = get_vllm_models()
            elif client_name in LLM_CLIENT_MODELS:
                models = LLM_CLIENT_MODELS[client_name]
            else:
                models = []

            for model_id, display_name in models:
                self._model_combo.addItem(display_name, model_id)

        # Restore selection if possible
        if current_model:
            idx = self._model_combo.findData(current_model)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        if operator_type == "llm":
            workers = _get_llm_workers()
        else:
            workers = _get_rl_workers()

        for worker in workers:
            self._worker_combo.addItem(worker.display_name, worker.worker_id)

        # Restore selection if possible
        if current_worker:
            idx = self._worker_combo.findData(current_worker)
            if idx >= 0:
                self._worker_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_env_dropdown(self) -> None:
        """Update environment dropdown based on selected type."""
        self._updating = True
        current_env = self._env_combo.currentText()
        self._env_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        if operator_type == "llm":
            envs = BARLOG_SUPPORTED_ENVS
        else:
            envs = RL_SUPPORTED_ENVS

        self._env_combo.addItems(envs)

        # Restore selection if possible
        if current_env:
            idx = self._env_combo.findText(current_env)
            if idx >= 0:
                self._env_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_task_dropdown(self) -> None:
        """Update task dropdown based on selected environment."""
        self._updating = True
        current_task = self._task_combo.currentText()
        self._task_combo.clear()

        env_name = self._env_combo.currentText()
        operator_type = self._type_combo.currentText().lower()

        # Define tasks for each environment
        tasks: List[str] = []
        if operator_type == "llm":
            if env_name == "babyai":
                tasks = [
                    "BabyAI-GoToRedBall-v0",
                    "BabyAI-GoToRedBallGrey-v0",
                    "BabyAI-GoToRedBallNoDists-v0",
                    "BabyAI-GoToObj-v0",
                    "BabyAI-GoToLocal-v0",
                    "BabyAI-PutNextLocal-v0",
                ]
            elif env_name == "minigrid":
                tasks = [
                    "MiniGrid-Empty-5x5-v0",
                    "MiniGrid-DoorKey-5x5-v0",
                    "MiniGrid-DoorKey-6x6-v0",
                    "MiniGrid-DoorKey-8x8-v0",
                    "MiniGrid-LavaGapS5-v0",
                    "MiniGrid-LavaGapS7-v0",
                ]
            elif env_name == "minihack":
                tasks = [
                    "MiniHack-Room-5x5-v0",
                    "MiniHack-Room-15x15-v0",
                    "MiniHack-Corridor-R5-v0",
                    "MiniHack-Quest-Easy-v0",
                ]
            elif env_name == "crafter":
                tasks = ["crafter-reward-v1", "crafter-nonreward-v1"]
            elif env_name == "textworld":
                tasks = ["tw-simple", "tw-cooking"]
            else:
                tasks = [BARLOG_DEFAULT_TASK]
        else:
            # RL environments typically have a single variant
            tasks = [env_name]

        self._task_combo.addItems(tasks if tasks else ["default"])

        # Restore selection if possible
        if current_task:
            idx = self._task_combo.findText(current_task)
            if idx >= 0:
                self._task_combo.setCurrentIndex(idx)

        # Show/hide task dropdown based on whether there are options
        self._task_combo.setVisible(len(tasks) > 1 or operator_type == "llm")

        self._updating = False

    def _load_config(self, config: OperatorConfig) -> None:
        """Load configuration into UI elements."""
        self._updating = True

        self._name_edit.setText(config.display_name)

        # Set type
        type_idx = 0 if config.operator_type == "llm" else 1
        self._type_combo.setCurrentIndex(type_idx)

        # Update dropdowns for type
        self._update_worker_dropdown()
        self._update_env_dropdown()

        # Set worker
        worker_idx = self._worker_combo.findData(config.worker_id)
        if worker_idx >= 0:
            self._worker_combo.setCurrentIndex(worker_idx)

        # Set environment
        env_idx = self._env_combo.findText(config.env_name)
        if env_idx >= 0:
            self._env_combo.setCurrentIndex(env_idx)

        # Update and set task
        self._update_task_dropdown()
        task_idx = self._task_combo.findText(config.task)
        if task_idx >= 0:
            self._task_combo.setCurrentIndex(task_idx)

        # Load LLM-specific settings
        if config.operator_type == "llm" and config.settings:
            # Set client
            client_name = config.settings.get("client_name", "openai")
            client_idx = self._client_combo.findData(client_name)
            if client_idx >= 0:
                self._client_combo.setCurrentIndex(client_idx)

            # Update and set model
            self._update_model_dropdown()
            model_id = config.settings.get("model_id")
            if model_id:
                model_idx = self._model_combo.findData(model_id)
                if model_idx >= 0:
                    self._model_combo.setCurrentIndex(model_idx)

            # Set API key
            api_key = config.settings.get("api_key", "")
            self._api_key_edit.setText(api_key)

        # Load RL-specific settings
        elif config.operator_type == "rl" and config.settings:
            policy_path = config.settings.get("policy_path", "")
            self._policy_path_edit.setText(policy_path)

        # Update visibility
        self._update_type_specific_visibility()

        self._updating = False

    def get_config(self) -> OperatorConfig:
        """Get current configuration from UI elements."""
        operator_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""
        display_name = self._name_edit.text() or f"Operator {self._operator_id[-1]}"
        env_name = self._env_combo.currentText() or "babyai"
        task = self._task_combo.currentText() or BARLOG_DEFAULT_TASK

        # Build settings based on operator type
        settings: Dict[str, Any] = {}

        if operator_type == "llm":
            # LLM-specific settings
            client_name = self._client_combo.currentData() or "openai"
            model_id = self._model_combo.currentData() or ""
            api_key = self._api_key_edit.text().strip()

            settings["client_name"] = client_name
            settings["model_id"] = model_id

            # Only include API key if provided
            if api_key:
                settings["api_key"] = api_key

            # Include base_url for vLLM
            if client_name in LLM_CLIENTS:
                _, _, default_base_url = LLM_CLIENTS[client_name]
                if default_base_url:
                    settings["base_url"] = default_base_url

        else:
            # RL-specific settings
            policy_path = self._policy_path_edit.text().strip()
            if policy_path:
                settings["policy_path"] = policy_path

        return OperatorConfig(
            operator_id=self._operator_id,
            operator_type=operator_type,
            worker_id=worker_id,
            display_name=display_name,
            env_name=env_name,
            task=task,
            settings=settings,
        )

    @property
    def operator_id(self) -> str:
        return self._operator_id


class OperatorConfigWidget(QtWidgets.QWidget):
    """Widget for managing N operator configurations.

    Provides:
    - List of OperatorConfigRow widgets
    - Add Operator button
    - Operator count limit (default MAX_OPERATORS)
    - Signals for operator list changes
    """

    operators_changed = pyqtSignal(list)  # List[OperatorConfig]
    initialize_requested = pyqtSignal(str, object)  # operator_id, config

    def __init__(
        self,
        max_operators: int = MAX_OPERATORS,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._max_operators = max_operators
        self._rows: Dict[str, OperatorConfigRow] = {}
        self._next_index = 0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QLabel("Configure Operators", self)
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Scroll area for operator rows
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(100)
        scroll.setMaximumHeight(250)

        self._rows_container = QtWidgets.QWidget(scroll)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()

        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll)

        # Add button
        self._add_btn = QtWidgets.QPushButton("+ Add Operator", self)
        self._add_btn.clicked.connect(self.add_operator)
        layout.addWidget(self._add_btn)

        # Info label
        self._info_label = QtWidgets.QLabel(f"0 / {self._max_operators} operators", self)
        self._info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._info_label)

    def add_operator(self, config: Optional[OperatorConfig] = None) -> Optional[str]:
        """Add a new operator row.

        Args:
            config: Optional initial configuration.

        Returns:
            The operator_id of the new row, or None if max reached.
        """
        if len(self._rows) >= self._max_operators:
            QtWidgets.QMessageBox.warning(
                self,
                "Maximum Operators",
                f"Maximum of {self._max_operators} operators allowed."
            )
            return None

        operator_id = f"operator_{self._next_index}"
        self._next_index += 1

        # Create default config if not provided
        if config is None:
            config = OperatorConfig(
                operator_id=operator_id,
                operator_type="llm",
                worker_id="barlog_worker",
                display_name=f"Operator {len(self._rows) + 1}",
                env_name="babyai",
                task="BabyAI-GoToRedBall-v0",
                settings={
                    "client_name": "openai",
                    "model_id": "gpt-4o-mini",
                },
            )

        # Create row widget
        row = OperatorConfigRow(operator_id, config, self._rows_container)
        row.config_changed.connect(self._on_row_config_changed)
        row.remove_requested.connect(self.remove_operator)
        row.initialize_requested.connect(self._on_initialize_requested)

        # Insert before stretch
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows[operator_id] = row

        self._update_ui_state()
        self._emit_operators_changed()

        return operator_id

    def remove_operator(self, operator_id: str) -> None:
        """Remove an operator row.

        Args:
            operator_id: ID of the operator to remove.
        """
        if operator_id not in self._rows:
            return

        row = self._rows.pop(operator_id)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

        self._update_ui_state()
        self._emit_operators_changed()

    def get_operators(self) -> List[OperatorConfig]:
        """Get all current operator configurations."""
        return [row.get_config() for row in self._rows.values()]

    def set_operators(self, configs: List[OperatorConfig]) -> None:
        """Set operator configurations, replacing any existing ones."""
        # Clear existing
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

        # Add new
        for config in configs:
            self.add_operator(config)

    def clear(self) -> None:
        """Remove all operators."""
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

    def _on_row_config_changed(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle config change from a row."""
        self._emit_operators_changed()

    def _on_initialize_requested(self, operator_id: str) -> None:
        """Handle initialize request from a row."""
        if operator_id not in self._rows:
            return
        config = self._rows[operator_id].get_config()
        _LOGGER.info(f"Initialize requested for operator {operator_id}: {config.env_name}/{config.task}")
        self.initialize_requested.emit(operator_id, config)

    def _emit_operators_changed(self) -> None:
        """Emit the operators_changed signal with current configs."""
        configs = self.get_operators()
        self.operators_changed.emit(configs)
        _LOGGER.debug(f"Operators changed: {len(configs)} operators")

    def _update_ui_state(self) -> None:
        """Update UI state based on current operator count."""
        count = len(self._rows)
        self._info_label.setText(f"{count} / {self._max_operators} operators")
        self._add_btn.setEnabled(count < self._max_operators)

        # Update index labels
        for i, (operator_id, row) in enumerate(self._rows.items()):
            row._index_label.setText(f"#{i + 1}")

    @property
    def operator_count(self) -> int:
        """Get the number of configured operators."""
        return len(self._rows)


__all__ = [
    "OperatorConfigRow",
    "OperatorConfigWidget",
    "MAX_OPERATORS",
    "LLM_CLIENTS",
    "LLM_CLIENT_MODELS",
]
