"""TD3 ActionSelector adapter."""

from __future__ import annotations

from typing import Any, Optional, Type

import numpy as np
import torch

from ..base import BaseSelector


class TD3Selector(BaseSelector):
    """ActionSelector for TD3 algorithm.

    Works with: td3_continuous_action
    """

    def __init__(
        self,
        actor_cls: Optional[Type[torch.nn.Module]] = None,
        qf_cls: Optional[Type[torch.nn.Module]] = None,
        exploration_noise: float = 0.1,
    ) -> None:
        """Initialize TD3 selector.

        Args:
            actor_cls: Optional Actor class.
            qf_cls: Optional QNetwork class (not used for eval).
            exploration_noise: Standard deviation of exploration noise (default 0.1)
        """
        super().__init__()
        self._actor_cls = actor_cls
        self._qf_cls = qf_cls
        self._actor: Optional[torch.nn.Module] = None
        self.exploration_noise = exploration_noise

    def load(
        self,
        model_path: str,
        envs: Any,
        device: str,
        *,
        actor_cls: Optional[Type[torch.nn.Module]] = None,
        model_cls: Optional[Type[torch.nn.Module]] = None,
        qf_cls: Optional[Type[torch.nn.Module]] = None,
        exploration_noise: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Load TD3 model from checkpoint.

        TD3 checkpoint format: tuple (actor_state_dict, qf1_state_dict, qf2_state_dict)

        Args:
            model_path: Path to .cleanrl_model checkpoint
            envs: Gymnasium vector environment
            device: Device to load model to
            actor_cls: Override Actor class (optional)
            model_cls: Alias for actor_cls (for consistency with other adapters)
            qf_cls: Override QNetwork class (optional, not used for eval)
            exploration_noise: Override exploration noise (optional)
        """
        self.device = device
        self.envs = envs

        if exploration_noise is not None:
            self.exploration_noise = exploration_noise

        cls = actor_cls or model_cls or self._actor_cls
        if cls is None:
            raise ValueError("actor_cls/model_cls must be provided either at init or load time")

        # TD3 checkpoint is triple: (actor_params, qf1_params, qf2_params)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, tuple):
            actor_params = checkpoint[0]
        else:
            actor_params = checkpoint

        self._actor = cls(envs).to(device)
        self._actor.load_state_dict(actor_params)
        self._actor.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action using actor network with exploration noise."""
        if self._actor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            obs_tensor = self._to_tensor(obs)
            actions = self._actor(obs_tensor)

            # Add exploration noise
            if self.exploration_noise > 0:
                action_scale = getattr(self._actor, "action_scale", 1.0)
                noise = torch.normal(
                    0,
                    action_scale * self.exploration_noise,
                    size=actions.shape,
                    device=self.device,
                )
                actions = actions + noise

        # Clip to action space bounds
        actions_np = actions.cpu().numpy()
        low = self.envs.single_action_space.low
        high = self.envs.single_action_space.high
        return np.clip(actions_np, low, high)

    def close(self) -> None:
        """Clean up resources."""
        self._actor = None
