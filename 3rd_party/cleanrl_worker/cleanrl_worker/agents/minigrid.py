"""
MiniGrid/BabyAI agent architectures.

This module provides neural network architectures for MiniGrid and BabyAI environments,
designed to be algorithm-agnostic and reusable across PPO, A2C, DQN, etc.

The MinigridCNN architecture is inspired by the original BabyAI paper (ICLR 2019).
For environments requiring instruction processing (language-grounded tasks),
consider using the full ACModel from babyai.model.

Usage:
    from cleanrl_worker.agents.minigrid import MinigridAgent, MinigridCNN

    # Create agent for vectorized environments
    agent = MinigridAgent(envs).to(device)

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


class MinigridCNN(nn.Module):
    """CNN feature extractor for MiniGrid/BabyAI 7x7x3 image observations.

    Architecture:
        - 3 convolutional layers with ReLU activation
        - Designed for small 7x7 grids with 3 channels (object type, color, state)
        - Outputs a flat feature vector

    This is a simplified architecture compared to BabyAI's FiLM-based model,
    suitable for tasks that don't require instruction processing.
    """

    def __init__(self, input_shape, hidden_dim=64):
        """Initialize the CNN.

        Args:
            input_shape: Tuple of (H, W, C) for input observations
            hidden_dim: Number of filters in hidden conv layers
        """
        super().__init__()

        # CNN for 7x7x3 MiniGrid observations
        # Input: (batch, 3, 7, 7) after permutation
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, hidden_dim, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output dimension
        with torch.no_grad():
            sample = torch.zeros(1, 3, input_shape[0], input_shape[1])
            self.output_dim = self.network(sample).shape[1]

    def forward(self, x):
        """Forward pass through the CNN.

        Args:
            x: Tensor of shape (batch, C, H, W), already preprocessed

        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        return self.network(x)


class MinigridAgent(nn.Module):
    """CNN-based agent for MiniGrid/BabyAI environments.

    Designed for (7, 7, 3) image observations from MiniGrid environments.
    Architecture inspired by the original BabyAI paper (ICLR 2019).

    This agent uses a shared CNN backbone for both actor and critic,
    which is more parameter-efficient than separate networks.

    Attributes:
        network: CNN feature extractor
        actor: Policy head (outputs action logits)
        critic: Value head (outputs state value)
    """

    def __init__(self, envs, hidden_dim=128):
        """Initialize the agent.

        Args:
            envs: Vectorized environment (for observation/action space info)
            hidden_dim: Dimension of hidden layers in actor/critic heads
        """
        super().__init__()
        obs_shape = envs.single_observation_space.shape  # (7, 7, 3)
        n_actions = envs.single_action_space.n

        # Shared CNN backbone
        self.cnn = MinigridCNN(obs_shape)

        # Actor head (policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.cnn.output_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cnn.output_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def _preprocess(self, x):
        """Preprocess observations: (batch, H, W, C) -> (batch, C, H, W), normalize.

        MiniGrid observations are uint8 in range [0, 255] or encoded integers.
        We normalize to [0, 1] for stable training.
        """
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return x

    def get_value(self, x):
        """Get state value estimate.

        Args:
            x: Observation tensor of shape (batch, H, W, C)

        Returns:
            Value tensor of shape (batch, 1)
        """
        x = self._preprocess(x)
        hidden = self.cnn(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value.

        This is the main method used during training rollouts.

        Args:
            x: Observation tensor of shape (batch, H, W, C)
            action: Optional action tensor. If None, samples from policy.

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        x = self._preprocess(x)
        hidden = self.cnn(x)

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
