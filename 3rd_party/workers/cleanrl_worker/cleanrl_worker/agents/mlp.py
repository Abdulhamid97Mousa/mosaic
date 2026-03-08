"""
MLP-based agent architectures.

This module provides simple MLP (Multi-Layer Perceptron) agent architectures
for environments with flat observation spaces like CartPole, MountainCar, etc.

These agents are algorithm-agnostic and can be used with PPO, A2C, and other
policy gradient methods.

Usage:
    from cleanrl_worker.agents.mlp import MLPAgent

    # Create agent for vectorized environments
    agent = MLPAgent(envs).to(device)

    # Get action and value
    action, log_prob, entropy, value = agent.get_action_and_value(obs)
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


class MLPAgent(nn.Module):
    """MLP-based agent for simple environments with flat observation spaces.

    Uses separate actor and critic networks with tanh activations,
    following the original PPO paper architecture.

    Architecture:
        - 2-layer MLP with 64 hidden units
        - Tanh activation (standard for PPO)
        - Separate actor and critic networks

    Attributes:
        critic: Value function network
        actor: Policy network (outputs action logits)
    """

    def __init__(self, envs, hidden_dim=64):
        """Initialize the agent.

        Args:
            envs: Vectorized environment (for observation/action space info)
            hidden_dim: Dimension of hidden layers (default: 64)
        """
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        n_actions = envs.single_action_space.n

        # Critic network (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01),
        )

    def get_value(self, x):
        """Get state value estimate.

        Args:
            x: Observation tensor of shape (batch, obs_dim)

        Returns:
            Value tensor of shape (batch, 1)
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value.

        This is the main method used during training rollouts.

        Args:
            x: Observation tensor of shape (batch, obs_dim)
            action: Optional action tensor. If None, samples from policy.

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
