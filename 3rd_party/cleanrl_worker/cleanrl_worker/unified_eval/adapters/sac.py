"""SAC ActionSelector adapter."""

from __future__ import annotations

from typing import Any, Optional, Type

import numpy as np
import torch

from ..base import BaseSelector


class SACSelector(BaseSelector):
    """ActionSelector for SAC algorithm.

    Works with: sac_continuous_action, sac_atari
    """

    def __init__(
        self,
        actor_cls: Optional[Type[torch.nn.Module]] = None,
        qf_cls: Optional[Type[torch.nn.Module]] = None,
        deterministic: bool = True,
    ) -> None:
        """Initialize SAC selector.

        Args:
            actor_cls: Optional Actor class.
            qf_cls: Optional SoftQNetwork class (not used for eval).
            deterministic: Whether to use deterministic action selection (default True)
        """
        super().__init__()
        self._actor_cls = actor_cls
        self._qf_cls = qf_cls
        self._actor: Optional[torch.nn.Module] = None
        self.deterministic = deterministic

    def load(
        self,
        model_path: str,
        envs: Any,
        device: str,
        *,
        actor_cls: Optional[Type[torch.nn.Module]] = None,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        qf_cls: Optional[Type[torch.nn.Module]] = None,
        deterministic: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Load SAC model from checkpoint.

        SAC checkpoint format: tuple (actor_state_dict, qf1_state_dict, qf2_state_dict)

        Args:
            model_path: Path to .cleanrl_model checkpoint
            envs: Gymnasium vector environment
            device: Device to load model to
            actor_cls: Override Actor class (optional)
            model_cls: Alias for actor_cls (for consistency with other adapters)
            qf_cls: Override SoftQNetwork class (optional, not used for eval)
            deterministic: Override deterministic flag (optional)
        """
        self.device = device
        self.envs = envs

        if deterministic is not None:
            self.deterministic = deterministic

        cls = actor_cls or model_cls or self._actor_cls
        if cls is None:
            raise ValueError("actor_cls/model_cls must be provided either at init or load time")

        # SAC checkpoint is triple: (actor_params, qf1_params, qf2_params)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, tuple):
            actor_params = checkpoint[0]
        else:
            actor_params = checkpoint

        self._actor = cls(envs).to(device)
        self._actor.load_state_dict(actor_params)
        self._actor.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using SAC's actor.get_action method."""
        if self._actor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)

            # SAC actor has get_action method that returns (action, log_prob, mean)
            if hasattr(self._actor, "get_action"):
                actions, _, mean = self._actor.get_action(obs_tensor)
                if self.deterministic:
                    actions = mean
            else:
                # Fallback for simpler actor implementations
                actions = self._actor(obs_tensor)

        return actions.cpu().numpy()

    def close(self) -> None:
        """Clean up resources."""
        self._actor = None
