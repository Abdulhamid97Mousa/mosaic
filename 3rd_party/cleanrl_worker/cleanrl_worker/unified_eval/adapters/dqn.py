"""DQN family ActionSelector adapter."""

from __future__ import annotations

import random
from typing import Any, Optional, Type

import numpy as np
import torch

from ..base import BaseSelector


class DQNSelector(BaseSelector):
    """ActionSelector for DQN-family algorithms.

    Works with: dqn, dqn_atari, pqn, rainbow_atari, qdagger
    """

    def __init__(
        self,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        epsilon: float = 0.05,
    ) -> None:
        """Initialize DQN selector.

        Args:
            model_cls: Optional QNetwork class. If not provided, will be loaded dynamically.
            epsilon: Exploration rate for epsilon-greedy (default 0.05)
        """
        super().__init__()
        self._model_cls = model_cls
        self._model: Optional[torch.nn.Module] = None
        self.epsilon = epsilon

    def load(
        self,
        model_path: str,
        envs: Any,
        device: str,
        *,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Load DQN model from checkpoint.

        Args:
            model_path: Path to .cleanrl_model checkpoint
            envs: Gymnasium vector environment
            device: Device to load model to
            model_cls: Override QNetwork class (optional)
            epsilon: Override exploration rate (optional)
        """
        self.device = device
        self.envs = envs

        if epsilon is not None:
            self.epsilon = epsilon

        # Use provided model_cls or the one from init
        cls = model_cls or self._model_cls
        if cls is None:
            raise ValueError("model_cls must be provided either at init or load time")

        self._model = cls(envs).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self._model.load_state_dict(checkpoint)
        self._model.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy with Q-values."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return np.array([
                self.envs.single_action_space.sample()
                for _ in range(self.envs.num_envs)
            ])

        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)
            q_values = self._model(obs_tensor)
            actions = torch.argmax(q_values, dim=1)

        return actions.cpu().numpy()

    def close(self) -> None:
        """Clean up resources."""
        self._model = None
