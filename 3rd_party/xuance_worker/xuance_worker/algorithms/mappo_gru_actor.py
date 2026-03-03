"""Standalone GRU actor extractor for MAPPO/IPPO trained policies.

Same as MAPPOActor but maintains per-agent GRU hidden state across timesteps.
Call reset() at episode boundaries to zero hidden state.

Checkpoint structure (GRU variant):
    actor_representation.{agent_key}.fc_net.{layer}.weight/bias    (MLP before GRU)
    actor_representation.{agent_key}.rnn.weight_ih_l0 / ...        (GRU weights)
    actor.{agent_key}.model.{layer}.weight/bias                    (actor head)

Architecture: MLP(obs) -> GRU(hidden) -> actor_head(action)

Usage:
    actor = MAPPOGRUActor.from_checkpoint("model.pth", agent_key="agent_0")
    actor.reset()  # call at episode start
    for obs in episode:
        action = actor.act(obs)  # hidden state updates internally
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class MAPPOGRUActor:
    """Deploy a MAPPO/IPPO-trained GRU policy for a single agent."""

    def __init__(
        self,
        fc_net: nn.Sequential | None,
        gru: nn.GRU,
        actor_head: nn.Sequential,
        agent_index: int | None,
        n_agents: int | None,
        device: torch.device,
    ) -> None:
        self.fc_net = fc_net.to(device).eval() if fc_net is not None else None
        self.gru = gru.to(device).eval()
        self.actor_head = actor_head.to(device).eval()
        self.agent_index = agent_index
        self.n_agents = n_agents
        self.device = device

        # GRU dimensions
        self.hidden_size = gru.hidden_size
        self.num_layers = gru.num_layers

        # Hidden state (initialized on reset)
        self._h: torch.Tensor | None = None

        # Pre-compute one-hot if using parameter sharing
        self._one_hot: torch.Tensor | None = None
        if n_agents is not None and agent_index is not None:
            oh = torch.zeros(1, n_agents, device=device)
            oh[0, agent_index] = 1.0
            self._one_hot = oh

        self.reset()

    def reset(self) -> None:
        """Reset GRU hidden state. Call at episode start."""
        self._h = torch.zeros(
            self.num_layers, 1, self.hidden_size, device=self.device
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        agent_key: str = "agent_0",
        agent_index: int | None = None,
        n_agents: int | None = None,
        device: str = "cpu",
    ) -> MAPPOGRUActor:
        """Load a trained GRU actor from a XuanCe checkpoint.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            agent_key: Key in the state dict (e.g., "agent_0").
            agent_index: Agent index for one-hot ID. Required if n_agents is set.
            n_agents: Total number of agents. Set for parameter-sharing checkpoints.
            device: PyTorch device string.

        Returns:
            MAPPOGRUActor ready for inference.
        """
        dev = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)

        state_dict = checkpoint
        if "policy" in checkpoint:
            state_dict = checkpoint["policy"]

        repr_prefix = f"actor_representation.{agent_key}."
        actor_prefix = f"actor.{agent_key}."

        repr_sd = _extract_subdict(state_dict, repr_prefix)
        actor_sd = _extract_subdict(state_dict, actor_prefix)

        # Split representation into fc_net and rnn parts
        fc_sd, gru_sd = _split_rnn_repr(repr_sd)

        # Build modules
        fc_net = _build_sequential_from_state_dict(fc_sd) if fc_sd else None
        gru = _build_gru_from_state_dict(gru_sd)
        actor_head = _build_sequential_from_state_dict(actor_sd)

        # Load weights
        if fc_net is not None:
            fc_net.load_state_dict(fc_sd)
        gru.load_state_dict(gru_sd)
        actor_head.load_state_dict(actor_sd)

        return cls(
            fc_net=fc_net,
            gru=gru,
            actor_head=actor_head,
            agent_index=agent_index,
            n_agents=n_agents,
            device=dev,
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        """Select a deterministic (greedy) action, updating hidden state.

        Args:
            observation: Flat observation array (e.g., 147 floats).

        Returns:
            Integer action index.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, obs_dim)

        # MLP before GRU
        if self.fc_net is not None:
            obs = self.fc_net(obs)  # (1, fc_out_dim)

        # GRU expects (batch, seq_len, input_size)
        gru_input = obs.unsqueeze(1)  # (1, 1, input_dim)
        gru_output, self._h = self.gru(gru_input, self._h)
        state = gru_output.squeeze(1)  # (1, hidden_size)

        # Concatenate one-hot agent ID if parameter sharing
        if self._one_hot is not None:
            state = torch.cat([state, self._one_hot], dim=-1)

        # Actor head -> logits
        logits = self.actor_head(state)  # (1, n_actions)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def act_stochastic(self, observation: np.ndarray) -> int:
        """Select a stochastic action, updating hidden state."""
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if self.fc_net is not None:
            obs = self.fc_net(obs)

        gru_input = obs.unsqueeze(1)
        gru_output, self._h = self.gru(gru_input, self._h)
        state = gru_output.squeeze(1)

        if self._one_hot is not None:
            state = torch.cat([state, self._one_hot], dim=-1)

        logits = self.actor_head(state)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    def __repr__(self) -> str:
        sharing = f"agent_index={self.agent_index}, n_agents={self.n_agents}" if self.n_agents else "no sharing"
        return (
            f"MAPPOGRUActor(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, {sharing}, device={self.device})"
        )


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


def _split_rnn_repr(repr_sd: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
    """Split Basic_RNN representation into fc_net and rnn sub-dicts.

    XuanCe's Basic_RNN has:
        fc_net.model.{idx}.weight/bias    (MLP layers before GRU)
        rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0
        (optional: normalize_before.weight/bias, normalize_after.weight/bias)
    """
    fc_sd = OrderedDict()
    gru_sd = OrderedDict()

    for key, val in repr_sd.items():
        if key.startswith("fc_net."):
            # Strip "fc_net." prefix so it looks like "model.{idx}.weight"
            fc_sd[key[len("fc_net."):]] = val
        elif key.startswith("rnn."):
            # Strip "rnn." prefix so it looks like "weight_ih_l0" etc.
            gru_sd[key[len("rnn."):]] = val
        # Skip normalize_before/after — not needed for inference

    return fc_sd, gru_sd


def _build_gru_from_state_dict(gru_sd: OrderedDict) -> nn.GRU:
    """Reconstruct nn.GRU from its state_dict weights.

    Infers input_size, hidden_size, and num_layers from weight shapes.
    GRU weight_ih_l{layer} has shape (3 * hidden_size, input_size_for_layer).
    """
    # Find the number of layers
    layer_indices = set()
    for key in gru_sd:
        if key.startswith("weight_ih_l"):
            layer_idx = int(key.split("weight_ih_l")[1])
            layer_indices.add(layer_idx)

    num_layers = max(layer_indices) + 1 if layer_indices else 1

    # Get dimensions from first layer
    w_ih_0 = gru_sd["weight_ih_l0"]
    three_h, input_size = w_ih_0.shape
    hidden_size = three_h // 3

    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )
    return gru


def _build_sequential_from_state_dict(sd: OrderedDict) -> nn.Sequential:
    """Reconstruct an nn.Sequential from weight/bias tensors.

    Handles both Linear layers (weight.dim()==2) and LayerNorm (weight.dim()==1).
    """
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
            out_f, in_f = weight.shape
            layer = nn.Linear(in_f, out_f, bias=(bias_key in sd))
            layers.append(layer)
        elif weight.dim() == 1:
            features = weight.shape[0]
            layer = nn.LayerNorm(features)
            layers.append(layer)

    return nn.Sequential(*layers)
