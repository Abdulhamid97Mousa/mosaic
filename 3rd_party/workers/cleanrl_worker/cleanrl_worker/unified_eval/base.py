"""Base protocol for algorithm-specific action selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ActionSelector(Protocol):
    """Protocol for algorithm-specific model loading and action selection.

    Each CleanRL algorithm family implements this protocol to handle:
    1. Loading model weights from checkpoint files
    2. Selecting actions given observations

    This allows the unified evaluator to work with any algorithm
    without needing algorithm-specific code in the evaluation loop.
    """

    @abstractmethod
    def load(
        self,
        model_path: str,
        envs: Any,
        device: str,
        **kwargs: Any,
    ) -> None:
        """Load model weights from a checkpoint file.

        Args:
            model_path: Path to the checkpoint file (.cleanrl_model)
            envs: Gymnasium vector environment (for model initialization)
            device: Device to load model to ("cpu" or "cuda")
            **kwargs: Additional algorithm-specific parameters
        """
        ...

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action(s) given observation(s).

        Args:
            obs: Observation array from the environment

        Returns:
            Action array to send to environment.step()
        """
        ...

    def close(self) -> None:
        """Clean up resources (optional)."""
        pass


class BaseSelector(ABC):
    """Base class for ActionSelector implementations with common utilities."""

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.envs: Any = None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def _to_tensor(self, obs: np.ndarray) -> Any:
        """Convert numpy observation to torch tensor."""
        import torch
        return torch.tensor(obs, dtype=torch.float32).to(self.device)
