"""Protocol definitions for MOSAIC workers.

All workers must implement these protocols to ensure consistent integration
with the trainer daemon and GUI.
"""

from __future__ import annotations

from typing import Protocol, Dict, Any, Optional, runtime_checkable
from dataclasses import dataclass


@runtime_checkable
class WorkerConfig(Protocol):
    """Configuration protocol that all worker configs must implement.

    This protocol ensures that all worker configurations provide the minimum
    required fields and serialization methods.

    Required Attributes:
        run_id: Unique identifier for this training run (ULID format)
        seed: Random seed for reproducibility (None = random)

    Required Methods:
        to_dict(): Serialize configuration to dictionary
        from_dict(): Deserialize configuration from dictionary

    Example:
        @dataclass(frozen=True)
        class MyWorkerConfig:
            run_id: str
            seed: Optional[int] = None
            algo: str = "ppo"
            env_id: str = "CartPole-v1"

            def to_dict(self) -> Dict[str, Any]:
                return asdict(self)

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> MyWorkerConfig:
                return cls(**data)
    """

    run_id: str
    seed: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to JSON-compatible dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerConfig:
        """Deserialize configuration from dictionary.

        Args:
            data: Dictionary containing configuration values

        Returns:
            Configuration instance
        """
        ...


@runtime_checkable
class WorkerRuntime(Protocol):
    """Execution protocol that all worker runtimes must implement.

    This protocol defines the standard lifecycle methods that the trainer
    daemon expects from all workers.

    Required Methods:
        run(): Execute the worker's main training/evaluation loop

    Example:
        class MyWorkerRuntime:
            def __init__(self, config: MyWorkerConfig):
                self.config = config

            def run(self) -> Dict[str, Any]:
                # Setup
                # Execute training
                # Cleanup
                return {
                    "status": "completed",
                    "episodes": 100,
                    "total_reward": 5000.0
                }
    """

    def run(self) -> Dict[str, Any]:
        """Execute the worker's main training/evaluation loop.

        This method should:
        1. Setup the environment and algorithm
        2. Execute the training/evaluation loop
        3. Emit telemetry events (run_started, run_completed, run_failed)
        4. Generate analytics manifest
        5. Clean up resources

        Returns:
            Dictionary containing execution results:
            - status: "completed" | "failed" | "cancelled"
            - episodes: Number of episodes completed
            - total_reward: Cumulative reward (optional)
            - metrics: Additional worker-specific metrics

        Raises:
            RuntimeError: If execution fails
        """
        ...


@dataclass(frozen=True)
class WorkerCapabilities:
    """Declares worker capabilities and resource requirements.

    Used by the trainer daemon to validate run submissions and allocate
    resources appropriately.

    Attributes:
        worker_type: Unique identifier (e.g., "cleanrl", "ray", "xuance")
        supported_paradigms: List of stepping paradigms supported
        env_families: Environment families supported (e.g., ["gymnasium", "pettingzoo"])
        action_spaces: Action space types supported (e.g., ["discrete", "continuous"])
        observation_spaces: Observation space types supported
        max_agents: Maximum concurrent agents (1 for single-agent)
        supports_self_play: Whether worker supports self-play
        supports_population: Whether worker supports population-based training
        supports_checkpointing: Whether worker can save/load checkpoints
        supports_pause_resume: Whether worker can be paused and resumed
        requires_gpu: Whether worker requires GPU
        gpu_memory_mb: Estimated GPU memory required (None = unknown)
        cpu_cores: Recommended CPU cores
        estimated_memory_mb: Estimated RAM required

    Example:
        capabilities = WorkerCapabilities(
            worker_type="cleanrl",
            supported_paradigms=("sequential",),
            env_families=("gymnasium",),
            action_spaces=("discrete", "continuous"),
            observation_spaces=("vector", "image"),
            max_agents=1,
            supports_self_play=False,
            supports_population=False,
            supports_checkpointing=True,
            supports_pause_resume=False,
            requires_gpu=False,
            gpu_memory_mb=None,
            cpu_cores=1,
            estimated_memory_mb=512
        )
    """

    worker_type: str
    supported_paradigms: tuple[str, ...]
    env_families: tuple[str, ...]
    action_spaces: tuple[str, ...]
    observation_spaces: tuple[str, ...]
    max_agents: int
    supports_self_play: bool
    supports_population: bool
    supports_checkpointing: bool
    supports_pause_resume: bool
    requires_gpu: bool
    gpu_memory_mb: Optional[int]
    cpu_cores: int
    estimated_memory_mb: int

    def supports_paradigm(self, paradigm: str) -> bool:
        """Check if paradigm is supported."""
        return paradigm in self.supported_paradigms

    def supports_env_family(self, family: str) -> bool:
        """Check if environment family is supported."""
        return family in self.env_families

    def is_multi_agent(self) -> bool:
        """Check if worker supports multiple agents."""
        return self.max_agents > 1


@dataclass(frozen=True)
class WorkerMetadata:
    """Metadata about a worker implementation.

    Used for documentation, discovery, and validation.

    Attributes:
        name: Human-readable worker name
        version: Worker version (semver)
        description: Brief description of worker's purpose
        author: Worker author/maintainer
        homepage: URL to worker documentation
        upstream_library: Name of wrapped RL library (if any)
        upstream_version: Version of upstream library
        license: License identifier (SPDX format)

    Example:
        metadata = WorkerMetadata(
            name="CleanRL Worker",
            version="1.0.0",
            description="Single-file RL implementations",
            author="MOSAIC Team",
            homepage="https://github.com/vwxyzjn/cleanrl",
            upstream_library="cleanrl",
            upstream_version="2.0.0",
            license="MIT"
        )
    """

    name: str
    version: str
    description: str
    author: str
    homepage: str
    upstream_library: Optional[str]
    upstream_version: Optional[str]
    license: str
