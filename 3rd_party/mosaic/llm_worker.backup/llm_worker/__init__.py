"""MOSAIC LLM Worker - Multi-agent LLM coordination with Theory of Mind.

This is MOSAIC's native LLM worker for multi-agent environments.
It provides:
- Multi-agent LLM coordination strategies (3 levels)
- Theory of Mind observation modes
- Support for multiple LLM providers (OpenRouter, OpenAI, Anthropic, Google, vLLM)
- Interactive and autonomous execution modes
- BALROG benchmark environment support (NLE, MiniHack, BabyAI, Crafter, TextWorld, BabaIsAI)
- Agent types: Naive, Chain-of-Thought, Robust variants, Few-Shot, Custom
"""

__version__ = "0.1.0"

from .config import LLMWorkerConfig
from .runtime import LLMWorkerRuntime, InteractiveLLMRuntime
from .analytics import write_analytics_manifest

# Agent imports
from .agents import (
    BaseAgent,
    NaiveAgent,
    ChainOfThoughtAgent,
    RobustNaiveAgent,
    RobustCoTAgent,
    FewShotAgent,
    DummyAgent,
    CustomAgent,
    AgentFactory,
    create_agent,
)

# Prompt builder imports
from .prompt_builder import (
    HistoryPromptBuilder,
    Message,
    create_prompt_builder,
    create_simple_prompt_builder,
)

# Client imports
from .clients import BaseLLMClient, LLMResponse, create_client


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="MOSAIC LLM Worker",
        version=__version__,
        description="Multi-agent LLM worker with coordination strategies and Theory of Mind",
        author="MOSAIC Team",
        homepage="https://github.com/MOSAIC-RL/GUI_BDI_RL",
        upstream_library=None,  # Native MOSAIC - no upstream
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="llm",
        supported_paradigms=("multi_agent", "human_vs_ai", "ai_vs_ai"),
        env_families=("pettingzoo", "gymnasium", "multigrid"),
        action_spaces=("discrete",),
        observation_spaces=("structured", "text"),
        max_agents=8,
        supports_self_play=True,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=True,
        requires_gpu=False,  # API-based by default
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    # Core
    "LLMWorkerConfig",
    "LLMWorkerRuntime",
    "InteractiveLLMRuntime",
    "write_analytics_manifest",
    "get_worker_metadata",
    # Agents
    "BaseAgent",
    "NaiveAgent",
    "ChainOfThoughtAgent",
    "RobustNaiveAgent",
    "RobustCoTAgent",
    "FewShotAgent",
    "DummyAgent",
    "CustomAgent",
    "AgentFactory",
    "create_agent",
    # Prompt builder
    "HistoryPromptBuilder",
    "Message",
    "create_prompt_builder",
    "create_simple_prompt_builder",
    # Clients
    "BaseLLMClient",
    "LLMResponse",
    "create_client",
]
