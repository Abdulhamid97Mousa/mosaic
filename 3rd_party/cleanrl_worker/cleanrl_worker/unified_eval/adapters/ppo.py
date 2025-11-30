"""PPO family ActionSelector adapter."""

from __future__ import annotations

from typing import Any, Optional, Type

import numpy as np
import torch

from ..base import BaseSelector


class PPOSelector(BaseSelector):
    """ActionSelector for PPO-family algorithms.

    Works with: ppo, ppo_continuous_action, ppo_atari, ppg_procgen, rpo_continuous_action
    """

    def __init__(self, agent_cls: Optional[Type[torch.nn.Module]] = None) -> None:
        """Initialize PPO selector.

        Args:
            agent_cls: Optional Agent class. If not provided, will be loaded dynamically.
        """
        super().__init__()
        self._agent_cls = agent_cls
        self._agent: Optional[torch.nn.Module] = None

    def load(
        self,
        model_path: str,
        envs: Any,
        device: str,
        *,
        agent_cls: Optional[Type[torch.nn.Module]] = None,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """Load PPO model from checkpoint.

        Args:
            model_path: Path to .cleanrl_model checkpoint
            envs: Gymnasium vector environment
            device: Device to load model to
            agent_cls: Override Agent class (optional)
            model_cls: Alias for agent_cls (for consistency with other adapters)
        """
        self.device = device
        self.envs = envs

        # Use provided agent_cls, model_cls, or the one from init
        cls = agent_cls or model_cls or self._agent_cls
        if cls is None:
            raise ValueError("agent_cls/model_cls must be provided either at init or load time")

        self._agent = cls(envs).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self._agent.load_state_dict(checkpoint)
        self._agent.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using PPO's get_action_and_value method."""
        if self._agent is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)
            actions, _, _, _ = self._agent.get_action_and_value(obs_tensor)

        return actions.cpu().numpy()

    def close(self) -> None:
        """Clean up resources."""
        self._agent = None
