"""BALROG Worker - LLM-based agent for BALROG environments.

This worker provides a subprocess interface for running LLM agents
on BALROG benchmark environments (BabyAI, MiniHack, Crafter, NetHack, TextWorld).

Key Features:
- LLM-based agents with multiple strategies (naive, CoT, robust, few-shot)
- Support for text-only and vision-language models
- Multiple environment families (BabyAI, MiniGrid, MiniHack, Crafter, NetHack, TextWorld)
- Multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter, vLLM)
- Interactive step-by-step execution for synchronized comparison

Agent Types:
- Naive: Direct action selection from LLM
- CoT: Chain-of-thought reasoning before action
- Robust Naive: Error-aware naive agent
- Robust CoT: Error-aware CoT agent
- Few-shot: In-context learning with examples
- Dummy: Random baseline for testing

Supported Environments:
- BabyAI/MiniGrid: Simple gridworld tasks
- MiniHack: NetHack-based procedural environments
- Crafter: Minecraft-inspired survival tasks
- NetHack (NLE): Full NetHack game
- TextWorld: Text-based adventure games
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "MOSAIC Team"

from balrog_worker.config import BarlogWorkerConfig, load_worker_config
from balrog_worker.runtime import BarlogWorkerRuntime, InteractiveRuntime


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="BALROG Worker",
        version=__version__,
        description="LLM-based agents for BALROG benchmark environments",
        author="MOSAIC Team",
        homepage="https://github.com/BALROG-ai/BALROG",
        upstream_library="balrog",
        upstream_version="1.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="balrog",
        supported_paradigms=("llm_agent",),  # Single LLM agent per run
        env_families=("babyai", "minigrid", "minihack", "crafter", "nle", "textworld"),
        action_spaces=("discrete", "text"),
        observation_spaces=("text", "image", "mixed"),
        max_agents=1,  # Single agent per environment
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=False,  # BALROG doesn't use traditional RL checkpoints
        supports_pause_resume=True,  # Via InteractiveRuntime
        requires_gpu=False,  # LLM inference can use CPU or GPU
        gpu_memory_mb=None,  # Depends on LLM provider (local vs API)
        cpu_cores=1,
        estimated_memory_mb=1024,  # Depends on environment and LLM
    )

    return metadata, capabilities


def main(*args, **kwargs):
    """CLI entry point (lazy loaded)."""
    from balrog_worker.cli import main as _main
    return _main(*args, **kwargs)


__all__ = [
    "__version__",
    "BarlogWorkerConfig",
    "load_worker_config",
    "BarlogWorkerRuntime",
    "InteractiveRuntime",
    "get_worker_metadata",
    "main",
]
