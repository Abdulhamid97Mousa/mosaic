"""C51 (Distributional RL) ActionSelector adapter."""

from __future__ import annotations

import random
from typing import Any, Optional, Type

import numpy as np
import torch

from ..base import BaseSelector


class C51Selector(BaseSelector):
    """ActionSelector for C51 distributional RL algorithms.

    Works with: c51, c51_atari
    """

    def __init__(
        self,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        epsilon: float = 0.05,
    ) -> None:
        """Initialize C51 selector.

        Args:
            model_cls: Optional QNetwork class.
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
        """Load C51 model from checkpoint.

        C51 checkpoints have a special format: {"args": {...}, "model_weights": state_dict}

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

        cls = model_cls or self._model_cls
        if cls is None:
            raise ValueError("model_cls must be provided either at init or load time")

        # C51 checkpoint has special format with args
        checkpoint = torch.load(model_path, map_location="cpu")

        # Handle both formats: dict with args or simple state_dict
        if isinstance(checkpoint, dict) and "args" in checkpoint:
            args = checkpoint["args"]
            self._model = cls(
                envs,
                n_atoms=args.get("n_atoms", 51),
                v_min=args.get("v_min", -10),
                v_max=args.get("v_max", 10),
            ).to(device)
            self._model.load_state_dict(checkpoint["model_weights"])
        else:
            # Fallback for simple state_dict format
            self._model = cls(envs).to(device)
            self._model.load_state_dict(checkpoint)

        self._model.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using C51's get_action method with epsilon-greedy."""
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
            actions, _ = self._model.get_action(obs_tensor)

        return actions.cpu().numpy()

    def close(self) -> None:
        """Clean up resources."""
        self._model = None
