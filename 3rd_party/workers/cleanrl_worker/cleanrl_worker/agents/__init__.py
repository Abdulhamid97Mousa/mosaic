"""
Agent architectures for different environment types.

This module provides neural network architectures for various environment types,
designed to be algorithm-agnostic and reusable across PPO, A2C, DQN, etc.

Available agents:
    - MinigridAgent: CNN-based agent for MiniGrid/BabyAI 7x7x3 observations
    - MinigridCNN: Feature extractor for MiniGrid observations
    - MLPAgent: MLP-based agent for flat observation spaces
    - GRUAgent: GRU-based recurrent agent for partially observable environments

Usage:
    from cleanrl_worker.agents import MinigridAgent, MLPAgent, GRUAgent

    # For MiniGrid/BabyAI environments
    agent = MinigridAgent(envs).to(device)

    # For simple environments (CartPole, etc.)
    agent = MLPAgent(envs).to(device)

    # For partially observable environments (MOSAIC MultiGrid, etc.)
    agent = GRUAgent(envs, gru_hidden_size=128).to(device)
"""

from .minigrid import MinigridAgent, MinigridCNN
from .mlp import MLPAgent
from .gru import GRUAgent

__all__ = ["MinigridAgent", "MinigridCNN", "MLPAgent", "GRUAgent"]
