"""
GRU-based recurrent agent architectures.

This module provides GRU (Gated Recurrent Unit) agent architectures for
environments requiring memory and sequential decision-making, such as
partially observable environments.

These agents are algorithm-agnostic and can be used with PPO, A2C, and other
policy gradient methods.

Usage:
    from cleanrl_worker.agents.gru import GRUAgent

    # Create agent for vectorized environments
    agent = GRUAgent(envs, gru_hidden_size=128).to(device)

    # Get action and value with hidden state
    action, log_prob, entropy, value, new_hidden = agent.get_action_and_value(
        obs, hidden_state, done
    )
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights and constant bias.

    This initialization scheme from PPO helps with stable training.

    Args:
        layer: PyTorch layer to initialize
        std: Standard deviation for orthogonal initialization
        bias_const: Constant value for bias initialization

    Returns:
        The initialized layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GRUAgent(nn.Module):
    """GRU-based recurrent agent for partially observable environments.

    Uses an MLP encoder followed by GRU for recurrent processing,
    with separate actor and critic heads.

    Architecture:
        - MLP encoder (256 -> 256)
        - GRU (256 -> hidden_size)
        - Actor head (hidden_size -> n_actions)
        - Critic head (hidden_size -> 1)

    Attributes:
        network: MLP encoder
        gru: GRU recurrent layer
        actor: Policy network (outputs action logits)
        critic: Value function network
    """

    def __init__(self, envs, gru_hidden_size=128, gru_num_layers=1):
        """Initialize the agent.

        Args:
            envs: Vectorized environment (for observation/action space info)
            gru_hidden_size: Hidden size of GRU (default: 128)
            gru_num_layers: Number of GRU layers (default: 1)
        """
        super().__init__()

        # Get observation space size
        obs_shape = envs.single_observation_space.shape
        if len(obs_shape) == 1:
            # Flattened observation
            input_size = obs_shape[0]
        else:
            # Multi-dimensional observation
            input_size = int(np.prod(obs_shape))

        n_actions = envs.single_action_space.n

        # MLP encoder for observations
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        # GRU for recurrent processing
        self.gru = nn.GRU(256, gru_hidden_size, num_layers=gru_num_layers)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Actor and critic heads
        self.actor = layer_init(nn.Linear(gru_hidden_size, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(gru_hidden_size, 1), std=1)

    def get_states(self, x, gru_state, done):
        """Process observations through encoder and GRU.

        Args:
            x: Observation tensor of shape (batch, obs_dim)
            gru_state: GRU hidden state of shape (num_layers, batch, hidden_size)
            done: Done flags of shape (batch,)

        Returns:
            Tuple of (hidden_features, new_gru_state)
        """
        # Flatten observation if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        hidden = self.network(x)

        # GRU logic with episode reset handling
        batch_size = gru_state.shape[1]
        hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, gru_state = self.gru(
                h.unsqueeze(0),
                (1.0 - d).view(1, -1, 1) * gru_state,
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, gru_state

    def get_value(self, x, gru_state, done):
        """Get state value estimate.

        Args:
            x: Observation tensor of shape (batch, obs_dim)
            gru_state: GRU hidden state
            done: Done flags

        Returns:
            Value tensor of shape (batch, 1)
        """
        hidden, _ = self.get_states(x, gru_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, gru_state, done, action=None, action_mask=None):
        """Get action, log probability, entropy, value, and new hidden state.

        This is the main method used during training rollouts.

        Args:
            x: Observation tensor of shape (batch, obs_dim)
            gru_state: GRU hidden state
            done: Done flags
            action: Optional action tensor. If None, samples from policy.
            action_mask: Optional action mask tensor. Should be 0 for valid actions,
                        -inf for invalid actions. Shape: (n_actions,) or (batch, n_actions)

        Returns:
            Tuple of (action, log_prob, entropy, value, new_gru_state)
        """
        hidden, gru_state = self.get_states(x, gru_state, done)
        logits = self.actor(hidden)

        # Apply action mask if provided (sets invalid action logits to -inf)
        if action_mask is not None:
            logits = logits + action_mask

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), gru_state
