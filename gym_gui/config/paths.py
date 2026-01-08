from __future__ import annotations

"""Centralized filesystem paths used across the Gym GUI application."""

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

# Writable runtime artifacts (shared with worker stack). Prefer repo-level var/.
VAR_ROOT = (_REPO_ROOT / "var").resolve()
VAR_REPLAY_DIR = VAR_ROOT / "replay"  # HDF5 replay storage (frames, observations)
VAR_TELEMETRY_DIR = VAR_ROOT / "telemetry"
VAR_CACHE_DIR = VAR_ROOT / "cache"
VAR_TMP_DIR = VAR_ROOT / "tmp"
VAR_LOGS_DIR = VAR_ROOT / "logs"
VAR_TRAINER_DIR = VAR_ROOT / "trainer"
VAR_TENSORBOARD_DIR = VAR_TRAINER_DIR / "runs"
VAR_WANDB_DIR = VAR_TRAINER_DIR / "runs"  # WANDB manifests stored per-run like TensorBoard
VAR_TRAINER_DB = VAR_TRAINER_DIR / "trainer.sqlite"
VAR_DATA_DIR = VAR_ROOT / "data"
VAR_MODELS_DIR = VAR_ROOT / "models"  # LLM models for vLLM serving
VAR_MODELS_HF_CACHE = VAR_MODELS_DIR / "huggingface"  # HuggingFace cache
VAR_MODELS_KATAGO_DIR = VAR_MODELS_DIR / "katago"  # KataGo neural network models
VAR_MODELS_GO_AI_DIR = VAR_MODELS_DIR / "go_ai"  # Go AI engines config/models
VAR_OPERATORS_DIR = VAR_ROOT / "operators"  # Operator subprocess data
VAR_OPERATORS_LOGS_DIR = VAR_OPERATORS_DIR / "logs"  # Operator subprocess logs
VAR_OPERATORS_TELEMETRY_DIR = VAR_OPERATORS_DIR / "telemetry"  # Operator telemetry (steps, episodes)
VAR_VLLM_DIR = VAR_ROOT / "vllm"  # vLLM server logs and state
VAR_BIN_DIR = VAR_ROOT / "bin"  # Project-local binaries (KataGo, etc.)


def ensure_var_directories() -> None:
    """Create the writable directory structure if it does not exist."""

    for path in (
        VAR_ROOT,
        VAR_REPLAY_DIR,
        VAR_TELEMETRY_DIR,
        VAR_CACHE_DIR,
        VAR_TMP_DIR,
        VAR_LOGS_DIR,
        VAR_TRAINER_DIR,
        VAR_TENSORBOARD_DIR,
        VAR_WANDB_DIR,
        VAR_DATA_DIR,
        VAR_MODELS_DIR,
        VAR_MODELS_HF_CACHE,
        VAR_MODELS_KATAGO_DIR,
        VAR_MODELS_GO_AI_DIR,
        VAR_OPERATORS_DIR,
        VAR_OPERATORS_LOGS_DIR,
        VAR_OPERATORS_TELEMETRY_DIR,
        VAR_VLLM_DIR,
        VAR_BIN_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "VAR_ROOT",
    "VAR_REPLAY_DIR",
    "VAR_TELEMETRY_DIR",
    "VAR_CACHE_DIR",
    "VAR_TMP_DIR",
    "VAR_LOGS_DIR",
    "VAR_TRAINER_DIR",
    "VAR_TRAINER_DB",
    "VAR_TENSORBOARD_DIR",
    "VAR_WANDB_DIR",
    "VAR_DATA_DIR",
    "VAR_MODELS_DIR",
    "VAR_MODELS_HF_CACHE",
    "VAR_MODELS_KATAGO_DIR",
    "VAR_MODELS_GO_AI_DIR",
    "VAR_OPERATORS_DIR",
    "VAR_OPERATORS_LOGS_DIR",
    "VAR_OPERATORS_TELEMETRY_DIR",
    "VAR_VLLM_DIR",
    "VAR_BIN_DIR",
    "ensure_var_directories",
]
