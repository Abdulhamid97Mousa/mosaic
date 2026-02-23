"""
Agent architectures for different environment types.

This module provides neural network architectures for various environment types,
designed to be algorithm-agnostic and reusable across PPO, A2C, DQN, etc.

Available agents:
    - MinigridAgent: CNN-based agent for MiniGrid/BabyAI 7x7x3 observations
    - MinigridCNN: Feature extractor for MiniGrid observations
    - MLPAgent: MLP-based agent for flat observation spaces

Usage:
    from cleanrl_worker.agents import MinigridAgent, MLPAgent

    # For MiniGrid/BabyAI environments
    agent = MinigridAgent(envs).to(device)

    # For simple environments (CartPole, etc.)
    agent = MLPAgent(envs).to(device)
"""

from .minigrid import MinigridAgent, MinigridCNN
from .mlp import MLPAgent

__all__ = ["MinigridAgent", "MinigridCNN", "MLPAgent"]
