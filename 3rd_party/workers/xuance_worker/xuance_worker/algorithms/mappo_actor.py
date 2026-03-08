"""Standalone MLP actor extractor for MAPPO/IPPO trained policies.

Loads actor weights from a XuanCe checkpoint and runs single-agent inference
without XuanCe's multi-agent batching infrastructure.

Works for both MAPPO and IPPO checkpoints (they share the same policy class).

Checkpoint structure (XuanCe state_dict keys):
    actor_representation.{agent_key}.model.{layer}.weight/bias
    actor.{agent_key}.model.{layer}.weight/bias

With use_parameter_sharing=True:
    - Single set of weights under agent_key="agent_0"
    - Agent identity encoded via one-hot vector concatenated to representation output
    - n_agents must be provided to construct the one-hot

With use_parameter_sharing=False:
    - Separate weights per agent: agent_key = "agent_0", "agent_1", ...
    - No one-hot needed (n_agents=None)

Usage:
    # Parameter sharing (1v1 soccer, shared policy):
    actor = MAPPOActor.from_checkpoint("model.pth", agent_key="agent_0",
                                        agent_index=0, n_agents=2)
    action = actor.act(obs_147_floats)

    # No parameter sharing (2v2 soccer, per-agent policies):
    actor = MAPPOActor.from_checkpoint("model.pth", agent_key="agent_0")
    action = actor.act(obs_147_floats)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class MAPPOActor:
    """Deploy a MAPPO/IPPO-trained MLP policy for a single agent."""

    def __init__(
        self,
        representation: nn.Sequential,
        actor_head: nn.Sequential,
        agent_index: int | None,
        n_agents: int | None,
        device: torch.device,
    ) -> None:
        self.representation = representation.to(device).eval()
        self.actor_head = actor_head.to(device).eval()
        self.agent_index = agent_index
        self.n_agents = n_agents
        self.device = device

        # Pre-compute one-hot if using parameter sharing
        self._one_hot: torch.Tensor | None = None
        if n_agents is not None and agent_index is not None:
            oh = torch.zeros(1, n_agents, device=device)
            oh[0, agent_index] = 1.0
            self._one_hot = oh

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        agent_key: str = "agent_0",
        agent_index: int | None = None,
        n_agents: int | None = None,
        device: str = "cpu",
    ) -> MAPPOActor:
        """Load a trained MLP actor from a XuanCe checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            agent_key: Key in the state dict (e.g., "agent_0").
            agent_index: Agent index for one-hot ID. Required if n_agents is set.
            n_agents: Total number of agents. Set for parameter-sharing checkpoints.
            device: PyTorch device string.

        Returns:
            MAPPOActor ready for inference.
        """
        dev = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)

        # XuanCe saves the policy state_dict under the top level
        state_dict = checkpoint if isinstance(checkpoint, dict) and "actor_representation" not in str(list(checkpoint.keys())[:1]) else checkpoint
        # Handle nested dict: sometimes it's checkpoint['policy']
        if "policy" in checkpoint:
            state_dict = checkpoint["policy"]

        repr_sd = _extract_subdict(state_dict, f"actor_representation.{agent_key}.")
        actor_sd = _extract_subdict(state_dict, f"actor.{agent_key}.")

        representation = _build_sequential_from_state_dict(repr_sd)
        actor_head = _build_sequential_from_state_dict(actor_sd)

        representation.load_state_dict(repr_sd)
        actor_head.load_state_dict(actor_sd)

        return cls(
            representation=representation,
            actor_head=actor_head,
            agent_index=agent_index,
            n_agents=n_agents,
            device=dev,
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        """Select a deterministic (greedy) action.

        Args:
            observation: Flat observation array (e.g., 147 floats for view_size=7).

        Returns:
            Integer action index.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, obs_dim)

        # Representation forward
        state = self.representation(obs)  # (1, repr_dim)

        # Concatenate one-hot agent ID if parameter sharing
        if self._one_hot is not None:
            state = torch.cat([state, self._one_hot.expand(state.shape[0], -1)], dim=-1)

        # Actor head → logits
        logits = self.actor_head(state)  # (1, n_actions)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def act_stochastic(self, observation: np.ndarray) -> int:
        """Select a stochastic action (sample from policy distribution).

        Useful for evaluation with diversity or when generating training data.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        state = self.representation(obs)
        if self._one_hot is not None:
            state = torch.cat([state, self._one_hot.expand(state.shape[0], -1)], dim=-1)

        logits = self.actor_head(state)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def __repr__(self) -> str:
        sharing = f"agent_index={self.agent_index}, n_agents={self.n_agents}" if self.n_agents else "no sharing"
        return f"MAPPOActor({sharing}, device={self.device})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_subdict(state_dict: dict, prefix: str) -> OrderedDict:
    """Extract keys with a given prefix and strip the prefix."""
    result = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            result[k[len(prefix):]] = v
    return result


def _build_sequential_from_state_dict(sd: OrderedDict) -> nn.Sequential:
    """Reconstruct an nn.Sequential from weight/bias tensors.

    XuanCe's Basic_MLP stores layers as model.{idx}.weight / model.{idx}.bias.
    The actor head uses the same pattern. We infer layer sizes from shapes.

    LayerNorm layers appear as model.{idx}.weight with shape (hidden_dim,)
    and matching bias — but without a second dimension, distinguishing them
    from Linear layers whose weight shape is (out_features, in_features).
    """
    # Collect layer indices
    layer_indices: set[int] = set()
    for key in sd:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_indices.add(int(parts[1]))

    layers: list[nn.Module] = []
    for idx in sorted(layer_indices):
        weight_key = f"model.{idx}.weight"
        bias_key = f"model.{idx}.bias"

        if weight_key not in sd:
            continue

        weight = sd[weight_key]

        if weight.dim() == 2:
            # Linear layer: weight shape (out_features, in_features)
            out_f, in_f = weight.shape
            layer = nn.Linear(in_f, out_f, bias=(bias_key in sd))
            layers.append(layer)
        elif weight.dim() == 1:
            # LayerNorm: weight shape (features,)
            features = weight.shape[0]
            layer = nn.LayerNorm(features)
            layers.append(layer)
        # Skip unknown shapes

    return nn.Sequential(*layers)
