"""Multi-operator configuration widget for side-by-side agent comparison.

This module provides UI widgets for configuring N operators (LLM or RL workers)
that can run in parallel, each with its own render container.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_MODELS_HF_CACHE
from gym_gui.services.operator import OperatorConfig, WorkerAssignment
from gym_gui.ui.worker_catalog.catalog import get_worker_catalog, WorkerDefinition
from gym_gui.constants.constants_operator import (
    BALROG_SUPPORTED_ENVS,
    BALROG_DEFAULT_TASK,
)


@dataclass
class VLLMServerInfo:
    """Information about a running vLLM server for operator assignment."""

    server_id: int  # 1-indexed (Server 1, Server 2, etc.)
    port: int
    model_id: str
    base_url: str
    status: str  # "running", "stopped", etc.

    @property
    def display_name(self) -> str:
        """Display name for dropdown: 'Server 1 - ModelName @ :8000'."""
        model_short = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
        return f"Server {self.server_id} - {model_short} @ :{self.port}"

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

        # Skip non-model directories
        if model_dir.name in ("hub", "xet", "datasets"):
            continue

        # Convert directory name to HuggingFace model ID format
        # HuggingFace cache format: "models--Org--ModelName" -> "Org/ModelName"
        # Simple format: "Org--ModelName" -> "Org/ModelName"
        dir_name = model_dir.name

        # Handle HuggingFace cache format: models--Org--ModelName
        if dir_name.startswith("models--"):
            # Strip "models--" prefix and split remaining by first "--"
            remainder = dir_name[len("models--"):]
            if "--" in remainder:
                org, model_name = remainder.split("--", 1)
                model_id = f"{org}/{model_name}"
                display_name = model_name
            else:
                # Malformed, skip
                _LOGGER.debug("Skipping malformed cache dir: %s", dir_name)
                continue
        elif "--" in dir_name:
            # Simple format: "Org--ModelName"
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

# PettingZoo Classic Games - turn-based multi-agent environments
# Each game has player IDs and human-readable labels
PETTINGZOO_GAMES: Dict[str, Dict[str, Any]] = {
    # Two-player board games
    "chess_v6": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "White", "player_1": "Black"},
    },
    "connect_four_v3": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Yellow", "player_1": "Red"},
    },
    "go_v5": {
        "family": "classic",
        "players": ["black_0", "white_0"],
        "player_labels": {"black_0": "Black", "white_0": "White"},
    },
    "tictactoe_v3": {
        "family": "classic",
        "players": ["player_1", "player_2"],
        "player_labels": {"player_1": "X", "player_2": "O"},
    },
    "backgammon_v3": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "White", "player_1": "Black"},
    },
    "gin_rummy_v4": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Player 1", "player_1": "Player 2"},
    },
    "texas_holdem_v4": {
        "family": "classic",
        "players": ["player_0", "player_1"],
        "player_labels": {"player_0": "Player 1", "player_1": "Player 2"},
    },
}

# All environment families with their environments
# Used by both RL and LLM operators
ENV_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "babyai": (),  # Loaded dynamically from gymnasium
    "minigrid": (),  # Loaded dynamically from gymnasium
    "minihack": (),  # Loaded dynamically from gymnasium
    "crafter": ("crafter-reward-v1", "crafter-nonreward-v1"),
    "nle": (),  # Loaded dynamically from gymnasium
    "textworld": ("tw-simple", "tw-cooking"),
    "toytext": (
        "FrozenLake-v1",
        "Taxi-v3",
        "CliffWalking-v0",
        "Blackjack-v1",
    ),
    "classic_control": (
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "Acrobot-v1",
    ),
    "box2d": (
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
        "BipedalWalkerHardcore-v3",
        "CarRacing-v3",
    ),
    # PettingZoo multi-agent games - populated from PETTINGZOO_GAMES
    "pettingzoo": tuple(PETTINGZOO_GAMES.keys()),
}

# RL has access to ALL environment families
RL_ENV_FAMILIES = tuple(ENV_FAMILIES.keys())

# LLM has access to all families EXCEPT classic_control and box2d
LLM_ENV_FAMILIES = tuple(f for f in ENV_FAMILIES.keys() if f not in ("classic_control", "box2d"))

# LLM Client configurations
# Maps client_name -> (display_name, requires_api_key, default_base_url)
LLM_CLIENTS: Dict[str, Tuple[str, bool, Optional[str]]] = {
    "openrouter": ("OpenRouter", True, "https://openrouter.ai/api/v1"),
    "vllm": ("vLLM (Local)", False, "http://localhost:8000/v1"),
    "openai": ("OpenAI (Direct)", True, None),
    "anthropic": ("Anthropic (Direct)", True, None),
    "google": ("Google (Direct)", True, None),
}

# =============================================================================
# VLM Models (Vision Language Models) - Support image input
# Verified via OpenRouter API: models with 'image' in input_modalities
# =============================================================================
VLM_CLIENT_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "openrouter": [
        # OpenAI VLMs (verified: image modality)
        # GPT-5 Series (Latest - Vision capable)
        ("openai/gpt-5.2", "GPT-5.2"),
        ("openai/gpt-5.1", "GPT-5.1"),
        ("openai/gpt-5", "GPT-5"),
        ("openai/gpt-5-mini", "GPT-5 Mini"),
        ("openai/gpt-5-image", "GPT-5 Image"),
        # GPT-4.1 Series (Vision capable)
        ("openai/gpt-4.1", "GPT-4.1"),
        ("openai/gpt-4.1-mini", "GPT-4.1 Mini"),
        ("openai/gpt-4.1-nano", "GPT-4.1 Nano"),
        # GPT-4o Series (Vision capable)
        ("openai/gpt-4o", "GPT-4o"),
        ("openai/gpt-4o-mini", "GPT-4o Mini"),
        ("openai/gpt-4-turbo", "GPT-4 Turbo"),
        # Anthropic VLMs (verified: image modality)
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
        ("anthropic/claude-3.5-haiku", "Claude 3.5 Haiku"),
        ("anthropic/claude-3-opus", "Claude 3 Opus"),
        ("anthropic/claude-3-haiku", "Claude 3 Haiku"),
        # Google Gemini VLMs (verified: image modality)
        # Gemini 3 (Latest - Vision capable)
        ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
        ("google/gemini-3-pro-image-preview", "Gemini 3 Pro Image Preview"),
        # Gemini 2.5 (Vision capable)
        ("google/gemini-2.5-pro-preview-05-06", "Gemini 2.5 Pro Preview"),
        ("google/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro (March)"),
        ("google/gemini-2.5-flash-preview", "Gemini 2.5 Flash Preview"),
        ("google/gemini-2.5-flash-preview-05-20", "Gemini 2.5 Flash (May)"),
        ("google/gemini-2.5-flash-lite-preview-06-17", "Gemini 2.5 Flash Lite"),
        ("google/gemini-2.5-flash-preview-image-05-20", "Gemini 2.5 Flash Image"),
        # Gemini 2.0 (Vision capable)
        ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash"),
        ("google/gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite"),
        ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash Exp (Free)"),
        # Meta Llama VLMs (verified: image modality)
        # Llama 4 (Latest)
        ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
        ("meta-llama/llama-4-scout", "Llama 4 Scout"),
        # Llama 3.2 Vision
        ("meta-llama/llama-3.2-90b-vision-instruct", "Llama 3.2 90B Vision"),
        ("meta-llama/llama-3.2-11b-vision-instruct", "Llama 3.2 11B Vision"),
        # Qwen VLMs (verified: image modality)
        # Qwen 3 VL (Latest)
        ("qwen/qwen3-vl-235b-a22b-instruct", "Qwen 3 VL 235B"),
        ("qwen/qwen3-vl-32b-instruct", "Qwen 3 VL 32B"),
        ("qwen/qwen3-vl-8b-instruct", "Qwen 3 VL 8B"),
        # Qwen 2.5 VL
        ("qwen/qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B"),
        ("qwen/qwen2.5-vl-32b-instruct", "Qwen 2.5 VL 32B"),
        ("qwen/qwen-2.5-vl-7b-instruct:free", "Qwen 2.5 VL 7B (Free)"),
        ("qwen/qwen-2.5-vl-7b-instruct", "Qwen 2.5 VL 7B"),
        # Qwen VL Legacy
        ("qwen/qwen-vl-max", "Qwen VL Max"),
        ("qwen/qwen-vl-plus", "Qwen VL Plus"),
        # Google Gemma VLMs (verified: image modality - Gemma 3 has vision)
        ("google/gemma-3-27b-it", "Gemma 3 27B"),
        ("google/gemma-3-27b-it:free", "Gemma 3 27B (Free)"),
        ("google/gemma-3-12b-it", "Gemma 3 12B"),
        ("google/gemma-3-12b-it:free", "Gemma 3 12B (Free)"),
        ("google/gemma-3-4b-it", "Gemma 3 4B"),
        ("google/gemma-3-4b-it:free", "Gemma 3 4B (Free)"),
        # Mistral VLMs (verified: image modality)
        # Pixtral (Vision-first models)
        ("mistralai/pixtral-large-2411", "Pixtral Large 2411"),
        ("mistralai/pixtral-12b", "Pixtral 12B"),
        # Mistral Small 3.x (Vision capable)
        ("mistralai/mistral-small-3.2-24b-instruct-2506", "Mistral Small 3.2 24B"),
        ("mistralai/mistral-small-3.1-24b-instruct-2503", "Mistral Small 3.1 24B"),
        ("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1 24B (Free)"),
        # Ministral 3 (Vision capable)
        ("mistralai/ministral-14b-2512", "Ministral 3 14B"),
        ("mistralai/ministral-8b-2512", "Ministral 3 8B"),
        ("mistralai/ministral-3b-2512", "Ministral 3 3B"),
    ],
    "vllm": [],  # vLLM servers are dynamically scanned
    "openai": [
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o Mini"),
        ("gpt-4-turbo", "GPT-4 Turbo"),
    ],
    "anthropic": [
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
        ("claude-3-opus-20240229", "Claude 3 Opus"),
        ("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    "google": [
        ("gemini-2.0-flash-exp", "Gemini 2.0 Flash"),
        ("gemini-1.5-pro", "Gemini 1.5 Pro"),
    ],
}

# =============================================================================
# LLM Models (Text-Only Language Models) - NO image support
# Verified via OpenRouter API: models with only 'text' in input_modalities
# =============================================================================
LLM_CLIENT_MODELS: Dict[str, List[Tuple[str, str]]] = {
    "openrouter": [
        # Meta Llama text-only models (verified: text-only modality)
        # Llama 3.3
        ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (Free)"),
        ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
        # Llama 3.2
        ("meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B (Free)"),
        ("meta-llama/llama-3.2-3b-instruct", "Llama 3.2 3B"),
        ("meta-llama/llama-3.2-1b-instruct", "Llama 3.2 1B"),
        # Llama 3.1
        ("meta-llama/llama-3.1-405b-instruct:free", "Llama 3.1 405B (Free)"),
        ("meta-llama/llama-3.1-405b-instruct", "Llama 3.1 405B"),
        ("meta-llama/llama-3.1-70b-instruct", "Llama 3.1 70B"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B"),
        # Llama 3.0
        ("meta-llama/llama-3-70b-instruct", "Llama 3 70B"),
        ("meta-llama/llama-3-8b-instruct", "Llama 3 8B"),
        # Mistral text-only models (verified: text-only modality)
        # Mistral Large Series
        ("mistralai/mistral-large-2512", "Mistral Large 3 2512"),
        ("mistralai/mistral-large-2411", "Mistral Large 2411"),
        ("mistralai/mistral-large-2407", "Mistral Large 2407"),
        # Mistral Medium Series
        ("mistralai/mistral-medium-3.1", "Mistral Medium 3.1"),
        ("mistralai/mistral-medium-3", "Mistral Medium 3"),
        # Codestral (Code-focused)
        ("mistralai/codestral-2508", "Codestral 2508"),
        ("mistralai/codestral-mamba", "Codestral Mamba"),
        # Devstral (Agentic coding)
        ("mistralai/devstral-2512", "Devstral 2 2512"),
        ("mistralai/devstral-2512:free", "Devstral 2 2512 (Free)"),
        # Mixtral (MoE)
        ("mistralai/mixtral-8x22b-instruct", "Mixtral 8x22B"),
        ("mistralai/mixtral-8x7b-instruct", "Mixtral 8x7B"),
        # Mistral Small/7B
        ("mistralai/mistral-7b-instruct:free", "Mistral 7B (Free)"),
        ("mistralai/mistral-7b-instruct", "Mistral 7B"),
        # DeepSeek text-only models (verified: text-only modality)
        # DeepSeek V3.x Series
        ("deepseek/deepseek-v3.2", "DeepSeek V3.2"),
        ("deepseek/deepseek-v3.2-speciale", "DeepSeek V3.2 Speciale"),
        ("deepseek/deepseek-chat-v3.1", "DeepSeek V3.1"),
        ("deepseek/deepseek-v3.1-terminus", "DeepSeek V3.1 Terminus"),
        ("deepseek/deepseek-chat", "DeepSeek V3"),
        ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3 0324"),
        # DeepSeek R1 (Reasoning)
        ("deepseek/deepseek-r1-0528", "DeepSeek R1 0528"),
        ("deepseek/deepseek-r1-0528:free", "DeepSeek R1 0528 (Free)"),
        ("deepseek/deepseek-r1", "DeepSeek R1"),
        # DeepSeek R1 Distill (Smaller/Faster)
        ("deepseek/deepseek-r1-distill-llama-70b", "R1 Distill Llama 70B"),
        ("deepseek/deepseek-r1-distill-qwen-32b", "R1 Distill Qwen 32B"),
        ("deepseek/deepseek-r1-distill-qwen-14b", "R1 Distill Qwen 14B"),
        ("deepseek/deepseek-r1-distill-llama-8b", "R1 Distill Llama 8B"),
        ("deepseek/deepseek-r1-distill-qwen-7b", "R1 Distill Qwen 7B"),
        # DeepSeek Prover (Math)
        ("deepseek/deepseek-prover-v2", "DeepSeek Prover V2"),
        # Qwen text-only models (verified: text-only modality)
        # Qwen 3
        ("qwen/qwen3-235b-a22b", "Qwen 3 235B"),
        ("qwen/qwen3-32b", "Qwen 3 32B"),
        ("qwen/qwen3-14b", "Qwen 3 14B"),
        ("qwen/qwen3-8b", "Qwen 3 8B"),
        ("qwen/qwen3-4b:free", "Qwen 3 4B (Free)"),
        # Qwen 3 Coder
        ("qwen/qwen3-coder:free", "Qwen 3 Coder (Free)"),
        ("qwen/qwen3-coder", "Qwen 3 Coder"),
        ("qwen/qwen3-coder-plus", "Qwen 3 Coder Plus"),
        # Qwen 2.5
        ("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
        ("qwen/qwen-2.5-7b-instruct", "Qwen 2.5 7B"),
        ("qwen/qwen-2.5-coder-32b-instruct", "Qwen 2.5 Coder 32B"),
        # QwQ (Reasoning)
        ("qwen/qwq-32b", "QwQ 32B"),
        # Qwen API Models
        ("qwen/qwen-max", "Qwen Max"),
        ("qwen/qwen-plus", "Qwen Plus"),
        ("qwen/qwen-turbo", "Qwen Turbo"),
        # Google Gemma text-only models (verified: text-only modality)
        # Gemma 3n (Text-only, smaller efficient models)
        ("google/gemma-3n-e4b-it:free", "Gemma 3n 4B (Free)"),
        ("google/gemma-3n-e4b-it", "Gemma 3n 4B"),
        ("google/gemma-3n-e2b-it:free", "Gemma 3n 2B (Free)"),
        # Gemma 2 (Text-only)
        ("google/gemma-2-27b-it", "Gemma 2 27B"),
        ("google/gemma-2-9b-it", "Gemma 2 9B"),
        ("google/gemma-2-9b-it:free", "Gemma 2 9B (Free)"),
    ],
    "vllm": [],  # vLLM servers are dynamically scanned
    "openai": [
        ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ],
    "anthropic": [],  # All Claude 3+ models support vision
    "google": [
        ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ],
}


def _get_llm_workers() -> List[WorkerDefinition]:
    """Get LLM workers from catalog (supports_training=False)."""
    return [w for w in get_worker_catalog() if not w.supports_training]


def _get_rl_workers() -> List[WorkerDefinition]:
    """Get RL workers from catalog (supports_training=True)."""
    return [w for w in get_worker_catalog() if w.supports_training]


def _get_rl_evaluation_workers() -> List[WorkerDefinition]:
    """Get RL workers that support policy loading for evaluation."""
    return [w for w in get_worker_catalog() if w.supports_training and w.supports_policy_load]


def _get_registered_envs(prefix: str) -> List[str]:
    """Get registered gymnasium environments matching a prefix.

    Args:
        prefix: Environment name prefix (e.g., "MiniGrid-", "BabyAI-", "MiniHack-", "NetHack").

    Returns:
        Sorted list of environment IDs matching the prefix.
    """
    try:
        import gymnasium

        # Import the package that registers environments with gymnasium
        # MiniGrid and BabyAI environments are registered by the minigrid package
        if prefix in ("MiniGrid-", "BabyAI-"):
            try:
                import minigrid  # noqa: F401 - registers envs on import
            except ImportError:
                _LOGGER.debug("minigrid package not installed")
                return []

        # MiniHack environments are registered by the minihack package
        if prefix == "MiniHack-":
            try:
                import minihack  # noqa: F401 - registers envs on import
            except ImportError:
                _LOGGER.debug("minihack package not installed")
                return []

        # NetHack environments are registered by the nle package
        if prefix == "NetHack":
            try:
                import nle  # noqa: F401 - registers envs on import
            except ImportError:
                _LOGGER.debug("nle package not installed")
                return []

        envs = [
            env_id for env_id in gymnasium.registry.keys()
            if env_id.startswith(prefix)
        ]
        return sorted(envs)
    except Exception as e:
        _LOGGER.warning(f"Could not load {prefix} environments: {e}")
        return []


class PlayerAssignmentRow(QtWidgets.QWidget):
    """Single player assignment row within a multi-agent operator.

    Each row allows configuring a worker for one player in a multi-agent game.
    Displays: Player label, Worker dropdown, Provider dropdown, Model dropdown, API Key.
    """

    assignment_changed = pyqtSignal()  # Emitted when any field changes

    def __init__(
        self,
        player_id: str,
        player_label: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize a player assignment row.

        Args:
            player_id: The internal player ID (e.g., "player_0", "black_0").
            player_label: Human-readable label (e.g., "White", "Black").
            parent: Parent widget.
        """
        super().__init__(parent)
        self._player_id = player_id
        self._player_label = player_label
        self._updating = False
        self._vllm_servers: List[VLLMServerInfo] = []

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        # Main vertical layout with two rows
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        # Row 1: Player label, Type selector, Worker dropdown, Provider (LLM only)
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)

        # Player label with ID
        label_text = f"{self._player_label} ({self._player_id})"
        player_label = QtWidgets.QLabel(label_text, self)
        player_label.setFixedWidth(120)
        player_label.setStyleSheet("font-weight: bold;")
        row1.addWidget(player_label)

        # Type selector (LLM / RL)
        row1.addWidget(QtWidgets.QLabel("Type:", self))
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.setFixedWidth(60)
        self._type_combo.addItems(["LLM", "RL"])
        row1.addWidget(self._type_combo)

        # Worker dropdown
        row1.addWidget(QtWidgets.QLabel("Worker:", self))
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setMinimumWidth(140)
        # Populated dynamically by _update_worker_dropdown()
        row1.addWidget(self._worker_combo)

        # Provider dropdown (LLM only)
        self._provider_label = QtWidgets.QLabel("Provider:", self)
        row1.addWidget(self._provider_label)
        self._client_combo = QtWidgets.QComboBox(self)
        self._client_combo.setMinimumWidth(100)
        for client_name, (display_name, _, _) in LLM_CLIENTS.items():
            self._client_combo.addItem(display_name, client_name)
        row1.addWidget(self._client_combo)

        row1.addStretch()
        main_layout.addLayout(row1)

        # Row 2: LLM settings container (Model/Server, API Key)
        self._llm_row = QtWidgets.QWidget(self)
        llm_layout = QtWidgets.QHBoxLayout(self._llm_row)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(6)

        # Spacer to align with row1 (same width as player label)
        spacer = QtWidgets.QWidget(self._llm_row)
        spacer.setFixedWidth(120)
        llm_layout.addWidget(spacer)

        # Model dropdown
        self._model_label = QtWidgets.QLabel("Model:", self._llm_row)
        llm_layout.addWidget(self._model_label)
        self._model_combo = QtWidgets.QComboBox(self._llm_row)
        self._model_combo.setMinimumWidth(160)
        self._model_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        llm_layout.addWidget(self._model_combo)

        # API Key field
        self._api_key_label = QtWidgets.QLabel("API Key:", self._llm_row)
        llm_layout.addWidget(self._api_key_label)
        self._api_key_edit = QtWidgets.QLineEdit(self._llm_row)
        self._api_key_edit.setPlaceholderText("API key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.setMinimumWidth(120)
        llm_layout.addWidget(self._api_key_edit)

        llm_layout.addStretch()
        main_layout.addWidget(self._llm_row)

        # Row 3: RL settings container (Policy path)
        self._rl_row = QtWidgets.QWidget(self)
        rl_layout = QtWidgets.QHBoxLayout(self._rl_row)
        rl_layout.setContentsMargins(0, 0, 0, 0)
        rl_layout.setSpacing(6)

        # Spacer to align with row1 (same width as player label)
        rl_spacer = QtWidgets.QWidget(self._rl_row)
        rl_spacer.setFixedWidth(120)
        rl_layout.addWidget(rl_spacer)

        # Policy path field
        rl_layout.addWidget(QtWidgets.QLabel("Policy:", self._rl_row))
        self._policy_path_edit = QtWidgets.QLineEdit(self._rl_row)
        self._policy_path_edit.setPlaceholderText("Path to trained policy/checkpoint")
        self._policy_path_edit.setMinimumWidth(200)
        rl_layout.addWidget(self._policy_path_edit)

        # Browse button
        self._browse_btn = QtWidgets.QPushButton("Browse...", self._rl_row)
        self._browse_btn.setFixedWidth(70)
        self._browse_btn.clicked.connect(self._on_browse_policy)
        rl_layout.addWidget(self._browse_btn)

        rl_layout.addStretch()
        main_layout.addWidget(self._rl_row)
        self._rl_row.hide()  # Hidden by default (LLM is default type)

        # Initialize dropdowns and visibility
        self._update_worker_dropdown()
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self._update_type_visibility()

    def _connect_signals(self) -> None:
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_changed)
        self._client_combo.currentIndexChanged.connect(self._on_client_changed)
        self._model_combo.currentIndexChanged.connect(self._on_changed)
        self._api_key_edit.textChanged.connect(self._on_changed)
        self._policy_path_edit.textChanged.connect(self._on_changed)

    def _on_changed(self) -> None:
        if not self._updating:
            self.assignment_changed.emit()

    def _on_client_changed(self) -> None:
        if self._updating:
            return
        self._update_model_dropdown()
        self._update_api_key_visibility()
        self.assignment_changed.emit()

    def _on_type_changed(self) -> None:
        """Handle worker type change (LLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_type_visibility()
        self.assignment_changed.emit()

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type (LLM or RL)."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        worker_type = self._type_combo.currentText().lower()
        if worker_type == "llm":
            workers = _get_llm_workers()
        else:
            # RL: only workers that support policy loading for evaluation
            workers = _get_rl_evaluation_workers()

        for worker in workers:
            self._worker_combo.addItem(worker.display_name, worker.worker_id)

        # Restore selection if possible
        if current_worker:
            idx = self._worker_combo.findData(current_worker)
            if idx >= 0:
                self._worker_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_type_visibility(self) -> None:
        """Show/hide LLM or RL settings based on selected type."""
        is_llm = self._type_combo.currentText().lower() == "llm"
        # LLM row visibility
        self._llm_row.setVisible(is_llm)
        self._provider_label.setVisible(is_llm)
        self._client_combo.setVisible(is_llm)
        # RL row visibility
        self._rl_row.setVisible(not is_llm)

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

    def _update_model_dropdown(self) -> None:
        """Update model dropdown based on selected provider."""
        self._updating = True
        current_data = self._model_combo.currentData()
        self._model_combo.clear()

        client_name = self._client_combo.currentData()
        if client_name:
            if client_name == "vllm":
                self._model_label.setText("Server:")
                running_servers = [s for s in self._vllm_servers if s.status == "running"]
                if running_servers:
                    for server in running_servers:
                        self._model_combo.addItem(server.display_name, server)
                else:
                    self._model_combo.addItem("(No running servers)", None)
            else:
                self._model_label.setText("Model:")
                models = LLM_CLIENT_MODELS.get(client_name, [])
                for model_id, display_name in models:
                    self._model_combo.addItem(display_name, model_id)

        # Restore selection if possible
        if current_data:
            if client_name == "vllm" and isinstance(current_data, VLLMServerInfo):
                for i in range(self._model_combo.count()):
                    item_data = self._model_combo.itemData(i)
                    if isinstance(item_data, VLLMServerInfo) and item_data.server_id == current_data.server_id:
                        self._model_combo.setCurrentIndex(i)
                        break
            else:
                idx = self._model_combo.findData(current_data)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_api_key_visibility(self) -> None:
        """Show/hide API key field based on selected provider."""
        client_name = self._client_combo.currentData()
        if client_name and client_name in LLM_CLIENTS:
            _, requires_api_key, _ = LLM_CLIENTS[client_name]
            self._api_key_label.setVisible(requires_api_key)
            self._api_key_edit.setVisible(requires_api_key)

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update available vLLM servers."""
        self._vllm_servers = servers
        if self._client_combo.currentData() == "vllm":
            self._update_model_dropdown()

    def get_assignment(self) -> WorkerAssignment:
        """Get the WorkerAssignment for this player.

        Returns:
            WorkerAssignment with worker_id, worker_type, and settings.
        """
        worker_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""

        settings: Dict[str, Any] = {}

        if worker_type == "rl":
            # RL worker settings: policy path
            policy_path = self._policy_path_edit.text().strip()
            if policy_path:
                settings["policy_path"] = policy_path
            # Default worker if none selected
            if not worker_id:
                worker_id = "ray_worker"
        else:
            # LLM worker settings: client, model, API key
            client_name = self._client_combo.currentData() or "openrouter"
            api_key = self._api_key_edit.text().strip()

            settings["client_name"] = client_name

            if api_key:
                settings["api_key"] = api_key

            # Handle vLLM vs other providers
            if client_name == "vllm":
                server_info = self._model_combo.currentData()
                if isinstance(server_info, VLLMServerInfo):
                    settings["model_id"] = server_info.model_id
                    settings["base_url"] = server_info.base_url
                else:
                    settings["model_id"] = ""
                    settings["base_url"] = "http://localhost:8000/v1"
            else:
                settings["model_id"] = self._model_combo.currentData() or ""
                if client_name in LLM_CLIENTS:
                    _, _, default_base_url = LLM_CLIENTS[client_name]
                    if default_base_url:
                        settings["base_url"] = default_base_url

            # Default worker if none selected
            if not worker_id:
                worker_id = "balrog_worker"

        return WorkerAssignment(
            worker_id=worker_id,
            worker_type=worker_type,
            settings=settings,
        )

    @property
    def player_id(self) -> str:
        return self._player_id


class PlayerAssignmentPanel(QtWidgets.QWidget):
    """Panel showing all player assignments for a multi-agent environment.

    Contains a PlayerAssignmentRow for each player in the game.
    """

    assignments_changed = pyqtSignal()  # Emitted when any player assignment changes

    def __init__(
        self,
        game_name: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the player assignment panel.

        Args:
            game_name: Name of the PettingZoo game (e.g., "chess_v6").
            parent: Parent widget.
        """
        super().__init__(parent)
        self._game_name = game_name
        self._rows: Dict[str, PlayerAssignmentRow] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        # Header
        header = QtWidgets.QLabel("Player Assignments:", self)
        header.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(header)

        # Get player info from PETTINGZOO_GAMES
        game_info = PETTINGZOO_GAMES.get(self._game_name, {})
        players = game_info.get("players", ["player_0", "player_1"])
        labels = game_info.get("player_labels", {})

        # Create row for each player
        for player_id in players:
            player_label = labels.get(player_id, player_id)
            row = PlayerAssignmentRow(player_id, player_label, self)
            row.assignment_changed.connect(self.assignments_changed)
            layout.addWidget(row)
            self._rows[player_id] = row

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update all rows with available vLLM servers."""
        for row in self._rows.values():
            row.set_vllm_servers(servers)

    def get_worker_assignments(self) -> Dict[str, WorkerAssignment]:
        """Get all player -> worker assignments.

        Returns:
            Dict mapping player_id to WorkerAssignment.
        """
        return {
            player_id: row.get_assignment()
            for player_id, row in self._rows.items()
        }

    @property
    def game_name(self) -> str:
        return self._game_name


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
    vllm_refresh_requested = pyqtSignal()  # request parent to refresh vLLM servers

    def __init__(
        self,
        operator_id: str,
        initial_config: Optional[OperatorConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._operator_id = operator_id
        self._updating = False  # Prevent signal loops
        self._vllm_servers: List[VLLMServerInfo] = []  # Available vLLM servers
        self._player_panel: Optional[PlayerAssignmentPanel] = None  # Multi-agent panel

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

        # Operator index label (1-indexed for UI, updated by _update_ui_state)
        try:
            idx = int(self._operator_id.split("_")[-1]) + 1
        except (ValueError, IndexError):
            idx = 1
        self._index_label = QtWidgets.QLabel(f"#{idx}", self)
        self._index_label.setFixedWidth(20)
        self._index_label.setStyleSheet("font-weight: bold; color: #666;")
        row1.addWidget(self._index_label)

        # Display name - shows default "Operator N" when empty
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setPlaceholderText(f"Operator {idx}")
        self._name_edit.setFixedWidth(120)
        row1.addWidget(self._name_edit)

        # Type selector (LLM / VLM / RL)
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.addItems(["LLM", "VLM", "RL"])
        self._type_combo.setFixedWidth(70)
        row1.addWidget(self._type_combo)

        # Worker dropdown
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setMinimumWidth(160)
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
        # Row 2: Two-column layout for environment and display settings
        # ============================================================
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(16)

        # --- Left Column: Environment Selection + Load Button ---
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)

        left_form = QtWidgets.QFormLayout()
        left_form.setSpacing(6)
        left_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self._env_combo = QtWidgets.QComboBox(self)
        self._env_combo.setMinimumWidth(200)
        left_form.addRow("Env Family:", self._env_combo)

        self._task_combo = QtWidgets.QComboBox(self)
        self._task_combo.setMinimumWidth(200)
        self._task_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        task_view = self._task_combo.view()
        if task_view is not None:
            task_view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
            )
        left_form.addRow("Environment:", self._task_combo)

        left_col.addLayout(left_form)

        # Load button and size label under environment selection
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self._init_btn = QtWidgets.QPushButton("Load Environment", self)
        self._init_btn.setToolTip("Load and initialize this environment for the operator")
        btn_row.addWidget(self._init_btn)

        self._size_label = QtWidgets.QLabel("", self)
        self._size_label.setStyleSheet("color: #666; font-size: 10px;")
        self._size_label.setToolTip("Actual displayed dimensions after scaling (width × height)")
        self._size_label.hide()
        btn_row.addWidget(self._size_label)

        btn_row.addStretch()
        left_col.addLayout(btn_row)

        row2.addLayout(left_col)

        # --- Right Column: Display Settings ---
        right_col = QtWidgets.QFormLayout()
        right_col.setSpacing(6)
        right_col.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Container size dropdown (operator card boundary)
        self._container_size_combo = QtWidgets.QComboBox(self)
        self._container_size_combo.setToolTip(
            "Operator container size.\n"
            "Controls the size of the operator card boundary."
        )
        self._container_size_combo.addItem("Auto", 0)
        self._container_size_combo.addItem("400px", 400)
        self._container_size_combo.addItem("512px", 512)
        self._container_size_combo.addItem("600px", 600)
        self._container_size_combo.addItem("768px", 768)
        self._container_size_combo.addItem("800px", 800)
        self._container_size_combo.addItem("1024px", 1024)
        self._container_size_combo.addItem("1280px", 1280)
        self._container_size_combo.addItem("1440px", 1440)
        self._container_size_combo.addItem("1600px", 1600)
        self._container_size_combo.addItem("1920px", 1920)
        self._container_size_combo.setCurrentIndex(5)  # Default to 800px
        self._container_size_combo.setFixedWidth(100)
        right_col.addRow("Container:", self._container_size_combo)

        # Image scale dropdown (RGB image scaling)
        self._image_scale_combo = QtWidgets.QComboBox(self)
        self._image_scale_combo.setToolTip(
            "Image scaling resolution.\n"
            "The RGB frame will be scaled to this size before display."
        )
        self._image_scale_combo.addItem("Native", 0)
        self._image_scale_combo.addItem("256px", 256)
        self._image_scale_combo.addItem("384px", 384)
        self._image_scale_combo.addItem("512px", 512)
        self._image_scale_combo.addItem("768px", 768)
        self._image_scale_combo.addItem("1024px", 1024)
        self._image_scale_combo.addItem("1280px", 1280)
        self._image_scale_combo.addItem("1440px", 1440)
        self._image_scale_combo.setCurrentIndex(0)  # Default to Native
        self._image_scale_combo.setFixedWidth(100)
        right_col.addRow("Image:", self._image_scale_combo)

        row2.addLayout(right_col)

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

        # LLM Model/Server selector (label changes based on provider)
        self._model_label = QtWidgets.QLabel("Model:", self._llm_container)
        llm_layout.addWidget(self._model_label)
        self._model_combo = QtWidgets.QComboBox(self._llm_container)
        self._model_combo.setMinimumWidth(200)  # Wider for server names
        self._model_combo.setMaxVisibleItems(20)  # Limit dropdown height with scrollbar
        llm_layout.addWidget(self._model_combo)

        llm_layout.addStretch()

        # API Key field
        self._api_key_label = QtWidgets.QLabel("API Key:", self._llm_container)
        llm_layout.addWidget(self._api_key_label)
        self._api_key_edit = QtWidgets.QLineEdit(self._llm_container)
        self._api_key_edit.setPlaceholderText("Set env var or enter key")
        self._api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._api_key_edit.setMinimumWidth(180)
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

        # === Multi-agent player assignment container ===
        # Created dynamically when pettingzoo environment is selected
        self._player_panel_container = QtWidgets.QWidget(self)
        self._player_panel_layout = QtWidgets.QVBoxLayout(self._player_panel_container)
        self._player_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._player_panel_container.hide()  # Hidden until pettingzoo selected
        main_layout.addWidget(self._player_panel_container)

        # Store main_layout reference for dynamic updates
        self._main_layout = main_layout

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
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._operator_id))
        self._init_btn.clicked.connect(lambda: self.initialize_requested.emit(self._operator_id))

        # LLM-specific signals
        self._client_combo.currentIndexChanged.connect(self._on_client_changed)
        self._model_combo.currentIndexChanged.connect(self._on_config_changed)
        self._api_key_edit.textChanged.connect(self._on_config_changed)

        # RL-specific signals
        self._policy_path_edit.textChanged.connect(self._on_config_changed)

    def _on_type_changed(self) -> None:
        """Handle operator type change (LLM <-> VLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._update_model_dropdown()  # Update models based on VLM vs LLM
        self._update_type_specific_visibility()
        self._on_config_changed()

    def _on_env_changed(self) -> None:
        """Handle environment family change."""
        if self._updating:
            return
        self._update_task_dropdown()
        self._update_multiagent_panel()
        self._update_type_specific_visibility()
        self._on_config_changed()

    def _on_task_changed(self) -> None:
        """Handle task/game change within an environment family."""
        if self._updating:
            return
        # For pettingzoo, different games have different players
        if self._is_pettingzoo_selected():
            self._update_multiagent_panel()
        self._on_config_changed()

    def _is_pettingzoo_selected(self) -> bool:
        """Check if pettingzoo environment family is selected."""
        return self._env_combo.currentText() == "pettingzoo"

    def _update_multiagent_panel(self) -> None:
        """Update the multi-agent player assignment panel based on selected game."""
        # Remove existing player panel
        if self._player_panel is not None:
            self._player_panel_layout.removeWidget(self._player_panel)
            self._player_panel.deleteLater()
            self._player_panel = None

        # Only show for pettingzoo environments
        if not self._is_pettingzoo_selected():
            self._player_panel_container.hide()
            return

        # Get the selected game
        game_name = self._task_combo.currentText()
        if not game_name or game_name not in PETTINGZOO_GAMES:
            self._player_panel_container.hide()
            return

        # Create new player panel for this game
        self._player_panel = PlayerAssignmentPanel(game_name, self._player_panel_container)
        self._player_panel.assignments_changed.connect(self._on_config_changed)
        self._player_panel.set_vllm_servers(self._vllm_servers)
        self._player_panel_layout.addWidget(self._player_panel)
        self._player_panel_container.show()

    def _on_client_changed(self) -> None:
        """Handle LLM client change."""
        if self._updating:
            return
        # If switching to vLLM, request parent to refresh server list
        if self._client_combo.currentData() == "vllm":
            self.vllm_refresh_requested.emit()
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
        """Show/hide LLM/VLM or RL specific widgets based on operator type and env.

        For single-agent environments: Show single worker row (LLM or RL).
        For multi-agent (pettingzoo): Hide single worker row, show player panel.
        """
        operator_type = self._type_combo.currentText().lower()
        is_llm_or_vlm = operator_type in ("llm", "vlm")
        is_pettingzoo = self._is_pettingzoo_selected()

        # For pettingzoo: hide single-worker UI, workers are per-player
        if is_pettingzoo:
            self._llm_container.hide()
            self._rl_container.hide()
            self._worker_combo.hide()
            # Type dropdown should also be hidden - all players use LLM workers
            self._type_combo.hide()
        else:
            # Single-agent mode: show appropriate container
            self._worker_combo.show()
            self._type_combo.show()
            self._llm_container.setVisible(is_llm_or_vlm)
            self._rl_container.setVisible(not is_llm_or_vlm)

            if is_llm_or_vlm:
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
        """Update model/server dropdown based on selected LLM client and operator type.

        For vLLM: Shows running servers from vLLM Server widget
        For VLM type: Shows VLM_CLIENT_MODELS (vision-capable models)
        For LLM type: Shows LLM_CLIENT_MODELS (text-only models)
        """
        self._updating = True
        current_data = self._model_combo.currentData()
        self._model_combo.clear()

        client_name = self._client_combo.currentData()
        operator_type = self._type_combo.currentText().lower()

        if client_name:
            if client_name == "vllm":
                # Show running vLLM servers instead of models
                self._model_label.setText("Server:")
                running_servers = [s for s in self._vllm_servers if s.status == "running"]
                if running_servers:
                    for server in running_servers:
                        # Store full server info for base_url lookup
                        self._model_combo.addItem(server.display_name, server)
                else:
                    # No running servers - show placeholder
                    self._model_combo.addItem("(No running servers)", None)
            else:
                # Other providers: show models based on operator type (VLM vs LLM)
                self._model_label.setText("Model:")
                # Use VLM models for VLM type, LLM models for LLM type
                if operator_type == "vlm":
                    model_dict = VLM_CLIENT_MODELS
                else:
                    model_dict = LLM_CLIENT_MODELS

                if client_name in model_dict:
                    models = model_dict[client_name]
                else:
                    models = []
                for model_id, display_name in models:
                    self._model_combo.addItem(display_name, model_id)

        # Restore selection if possible
        if current_data:
            # For vLLM, match by server_id; for others, match by model_id
            if client_name == "vllm" and isinstance(current_data, VLLMServerInfo):
                for i in range(self._model_combo.count()):
                    item_data = self._model_combo.itemData(i)
                    if isinstance(item_data, VLLMServerInfo) and item_data.server_id == current_data.server_id:
                        self._model_combo.setCurrentIndex(i)
                        break
            else:
                idx = self._model_combo.findData(current_data)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)

        self._updating = False

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update the list of available vLLM servers.

        Called by parent widget when vLLM server status changes.

        Args:
            servers: List of VLLMServerInfo from the vLLM server widget.
        """
        self._vllm_servers = servers
        # Refresh dropdown if vLLM is currently selected
        if self._client_combo.currentData() == "vllm":
            self._update_model_dropdown()
        # Also update player panel if it exists (for multi-agent mode)
        if self._player_panel is not None:
            self._player_panel.set_vllm_servers(servers)

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        # LLM and VLM both use LLM workers (same BALROG worker, different image settings)
        if operator_type in ("llm", "vlm"):
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
        """Update environment family dropdown based on selected type."""
        self._updating = True
        current_env = self._env_combo.currentText()
        self._env_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        # LLM and VLM use same env families (text-based reasoning environments)
        if operator_type in ("llm", "vlm"):
            envs = LLM_ENV_FAMILIES
        else:
            # RL: all environment families
            envs = RL_ENV_FAMILIES

        self._env_combo.addItems(envs)

        # Restore selection if possible
        if current_env:
            idx = self._env_combo.findText(current_env)
            if idx >= 0:
                self._env_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_task_dropdown(self) -> None:
        """Update environment dropdown based on selected environment family."""
        self._updating = True
        current_task = self._task_combo.currentText()
        self._task_combo.clear()

        env_family = self._env_combo.currentText()

        # Get environments for this family
        # Some families load dynamically from gymnasium, others use static lists
        envs: List[str] = []

        if env_family == "babyai":
            # Dynamically load all BabyAI environments from gymnasium registry
            envs = _get_registered_envs("BabyAI-")
            if not envs:
                envs = ["BabyAI-GoToRedBall-v0", "BabyAI-GoToObj-v0", "BabyAI-GoToLocal-v0"]
        elif env_family == "minigrid":
            # Dynamically load all MiniGrid environments from gymnasium registry
            envs = _get_registered_envs("MiniGrid-")
            if not envs:
                envs = ["MiniGrid-Empty-5x5-v0", "MiniGrid-DoorKey-5x5-v0"]
        elif env_family == "minihack":
            # Dynamically load all MiniHack environments from gymnasium registry
            envs = _get_registered_envs("MiniHack-")
            if not envs:
                envs = ["MiniHack-Room-5x5-v0", "MiniHack-Corridor-R5-v0"]
        elif env_family == "nle":
            # Dynamically load all NetHack environments from gymnasium registry
            envs = _get_registered_envs("NetHack")
            if not envs:
                envs = ["NetHackScore-v0", "NetHackStaircase-v0", "NetHackEat-v0"]
        elif env_family in ENV_FAMILIES:
            # Use static list from ENV_FAMILIES
            envs = list(ENV_FAMILIES[env_family])
        else:
            envs = [BALROG_DEFAULT_TASK]

        self._task_combo.addItems(envs if envs else ["default"])

        # Restore selection if possible
        if current_task:
            idx = self._task_combo.findText(current_task)
            if idx >= 0:
                self._task_combo.setCurrentIndex(idx)

        # Always show environment dropdown
        self._task_combo.setVisible(True)

        self._updating = False

    def _load_config(self, config: OperatorConfig) -> None:
        """Load configuration into UI elements."""
        self._updating = True

        self._name_edit.setText(config.display_name)

        # Determine type: check if it's VLM based on max_image_history setting
        operator_type = config.operator_type
        if operator_type in ("llm", "vlm") and config.settings:
            # If max_image_history > 0, it's VLM mode
            max_image_history = config.settings.get("max_image_history", 0)
            if max_image_history > 0:
                operator_type = "vlm"
            else:
                operator_type = "llm"

        # Set type: LLM=0, VLM=1, RL=2
        type_map = {"llm": 0, "vlm": 1, "rl": 2}
        type_idx = type_map.get(operator_type, 0)
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

        # Load LLM/VLM-specific settings
        if operator_type in ("llm", "vlm") and config.settings:
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
        elif operator_type == "rl" and config.settings:
            policy_path = config.settings.get("policy_path", "")
            self._policy_path_edit.setText(policy_path)

        # Update visibility
        self._update_type_specific_visibility()

        self._updating = False

    def get_config(self) -> OperatorConfig:
        """Get current configuration from UI elements.

        Returns:
            OperatorConfig with single_agent or multi_agent configuration
            depending on whether pettingzoo environment is selected.
        """
        # Extract index from operator_id (e.g., "operator_0" -> 0) for display name
        try:
            idx = int(self._operator_id.split("_")[-1]) + 1
        except (ValueError, IndexError):
            idx = 1
        display_name = self._name_edit.text() or f"Operator {idx}"
        env_name = self._env_combo.currentText() or "babyai"
        task = self._task_combo.currentText() or BALROG_DEFAULT_TASK

        # Multi-agent mode: pettingzoo with player assignments
        if self._is_pettingzoo_selected() and self._player_panel is not None:
            player_workers = self._player_panel.get_worker_assignments()
            # Add container_size and image_scale to first worker's settings
            # (accessed via config.settings property)
            if player_workers:
                first_player = next(iter(player_workers.keys()))
                container_size = self._container_size_combo.currentData()
                if container_size and container_size > 0:
                    player_workers[first_player].settings["container_size"] = container_size
                image_scale = self._image_scale_combo.currentData()
                if image_scale and image_scale > 0:
                    player_workers[first_player].settings["image_scale"] = image_scale
            return OperatorConfig.multi_agent(
                operator_id=self._operator_id,
                display_name=display_name,
                env_name=env_name,
                task=task,
                player_workers=player_workers,
            )

        # Single-agent mode: standard LLM/VLM/RL configuration
        operator_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""

        # Build settings based on operator type
        settings: Dict[str, Any] = {}

        if operator_type in ("llm", "vlm"):
            # LLM/VLM-specific settings
            client_name = self._client_combo.currentData() or "openai"
            api_key = self._api_key_edit.text().strip()

            settings["client_name"] = client_name

            # VLM mode: 0 = text-only (LLM), 1 = vision mode (VLM)
            # The type dropdown now directly controls this
            settings["max_image_history"] = 1 if operator_type == "vlm" else 0

            # Only include API key if provided
            if api_key:
                settings["api_key"] = api_key

            # Handle vLLM server selection vs model selection
            if client_name == "vllm":
                # vLLM: get model_id and base_url from selected server
                server_info = self._model_combo.currentData()
                if isinstance(server_info, VLLMServerInfo):
                    settings["model_id"] = server_info.model_id
                    settings["base_url"] = server_info.base_url
                else:
                    # No server selected - use empty values
                    settings["model_id"] = ""
                    settings["base_url"] = "http://localhost:8000/v1"
            else:
                # Other providers: get model_id from dropdown
                settings["model_id"] = self._model_combo.currentData() or ""
                # Include default base_url if available
                if client_name in LLM_CLIENTS:
                    _, _, default_base_url = LLM_CLIENTS[client_name]
                    if default_base_url:
                        settings["base_url"] = default_base_url

        else:
            # RL-specific settings
            policy_path = self._policy_path_edit.text().strip()
            if policy_path:
                settings["policy_path"] = policy_path

        # Common settings - container size and image scale
        container_size = self._container_size_combo.currentData()
        if container_size and container_size > 0:
            settings["container_size"] = container_size
        image_scale = self._image_scale_combo.currentData()
        if image_scale and image_scale > 0:
            settings["image_scale"] = image_scale

        return OperatorConfig.single_agent(
            operator_id=self._operator_id,
            display_name=display_name,
            worker_id=worker_id,
            worker_type=operator_type,
            env_name=env_name,
            task=task,
            settings=settings,
        )

    def set_environment_size(
        self, width: int, height: int, container_size: Optional[int] = None
    ) -> None:
        """Set the environment size label after loading.

        Args:
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        if container_size:
            self._size_label.setText(f"Image: {width}×{height} | Container: {container_size}px")
        else:
            self._size_label.setText(f"Image: {width}×{height}")
        self._size_label.show()

    def clear_environment_size(self) -> None:
        """Clear the environment size label."""
        self._size_label.setText("")
        self._size_label.hide()

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
    vllm_refresh_requested = pyqtSignal()  # request to refresh vLLM server list

    def __init__(
        self,
        max_operators: int = MAX_OPERATORS,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._max_operators = max_operators
        self._rows: Dict[str, OperatorConfigRow] = {}
        self._next_index = 0
        self._vllm_servers: List[VLLMServerInfo] = []  # Cache for new rows

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QLabel("Configure Operators", self)
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Scroll area for operator rows - expands to fill available space
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(300)
        # No maximum height - let it fill available space
        scroll.setMinimumWidth(450)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

        self._rows_container = QtWidgets.QWidget(scroll)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()

        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll, 1)  # Stretch factor 1 to fill available space

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
            config = OperatorConfig.single_agent(
                operator_id=operator_id,
                display_name=f"Operator {len(self._rows) + 1}",
                worker_id="balrog_worker",
                worker_type="llm",
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
        row.vllm_refresh_requested.connect(self.vllm_refresh_requested)  # Propagate to parent

        # Pass cached vLLM server list to new row
        if self._vllm_servers:
            row.set_vllm_servers(self._vllm_servers)

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

    def set_vllm_servers(self, servers: List[VLLMServerInfo]) -> None:
        """Update all operator rows with available vLLM servers.

        Called when vLLM server status changes. Propagates the server list
        to all OperatorConfigRow widgets so they can update their dropdowns.

        Args:
            servers: List of VLLMServerInfo from the vLLM server widget.
        """
        # Cache for new rows that will be added later
        self._vllm_servers = servers
        _LOGGER.debug(f"Updating operator rows with {len(servers)} vLLM servers")
        for row in self._rows.values():
            row.set_vllm_servers(servers)

    def set_operator_environment_size(
        self, operator_id: str, width: int, height: int, container_size: Optional[int] = None
    ) -> None:
        """Set the environment size for a specific operator.

        Called after an environment is successfully loaded to display
        the rendered dimensions in the operator config row.

        Args:
            operator_id: The operator's unique ID
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        if operator_id in self._rows:
            self._rows[operator_id].set_environment_size(width, height, container_size)
            _LOGGER.debug(f"Set environment size for {operator_id}: {width}×{height}, container: {container_size}")


__all__ = [
    "OperatorConfigRow",
    "OperatorConfigWidget",
    "PlayerAssignmentRow",
    "PlayerAssignmentPanel",
    "VLLMServerInfo",
    "MAX_OPERATORS",
    "LLM_CLIENTS",
    "LLM_CLIENT_MODELS",
    "VLM_CLIENT_MODELS",
    "PETTINGZOO_GAMES",
]
