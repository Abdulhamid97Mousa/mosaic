"""Standalone Q-Learning implementation (no matplotlib, no legacy deps)."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

import numpy as np

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import LOG_WORKER_BDI_EVENT

LOGGER = logging.getLogger(__name__)
_log = partial(log_constant, LOGGER)


class Environment(Protocol):
    """Minimal environment interface (matches Gymnasium-like adapters)."""

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]: ...
    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]: ...


@dataclass
class QLearningAgent:
    """Tabular Q-learning agent (no plotting)."""

    observation_space_n: int
    action_space_n: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    q_table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.observation_space_n, self.action_space_n))

    def select_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space_n)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Q-learning update."""
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


@dataclass
class QLearningRuntime:
    """Runtime for Q-learning training (no telemetry, no plotting)."""

    env: Environment
    agent: QLearningAgent
    max_episodes: int = 1000
    max_steps: int = 100

    def get_action(self, state: int, training: bool = True) -> int:
        """Select action (for compatibility with HeadlessTrainer)."""
        return self.agent.select_action(state, training=training)

    def update_q_online(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Update Q-table (for compatibility with HeadlessTrainer)."""
        self.agent.update(state, action, reward, next_state, done)

    def train(self) -> dict[str, Any]:
        """Train the agent and return summary stats."""
        episode_rewards: list[float] = []
        episode_steps: list[int] = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            steps = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.agent.update(state, action, reward, next_state, done)
                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            self.agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_steps.append(steps)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                _log(LOG_WORKER_BDI_EVENT, message=f"Episode {episode + 1}: training progress", extra={"avg_reward": float(avg_reward), "epsilon": self.agent.epsilon})

        return {
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "final_epsilon": self.agent.epsilon,
            "mean_reward": float(np.mean(episode_rewards[-100:])),
        }


def create_agent(adapter: Environment, **kwargs: Any) -> QLearningAgent:
    """Factory for Q-learning agent (extracts space sizes from adapter)."""
    # FrozenLakeAdapter has observation_space_n and action_space_n properties
    obs_n = getattr(adapter, "observation_space_n", getattr(adapter, "num_states", 16))
    act_n = getattr(adapter, "action_space_n", getattr(adapter, "num_actions", 4))
    return QLearningAgent(observation_space_n=obs_n, action_space_n=act_n, **kwargs)


def create_runtime(env: Environment, agent: QLearningAgent, **kwargs: Any) -> QLearningRuntime:
    """Factory for Q-learning runtime."""
    return QLearningRuntime(env=env, agent=agent, **kwargs)


__all__ = ["QLearningAgent", "QLearningRuntime", "create_agent", "create_runtime"]
