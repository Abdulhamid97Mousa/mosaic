# HeMAC: Heterogeneous Multi-Agent Challenge

## Overview

**HeMAC** (Heterogeneous Multi-Agent Challenge) is a standardized, PettingZoo-based benchmark environment for Heterogeneous Multi-Agent Reinforcement Learning (HeMARL). Published at ECAI 2025 by ThalesGroup, it proposes multiple scenarios where agents with diverse sensors, resources, or capabilities must cooperate to solve complex tasks under partial observability.

## Key Features

- **Rich Heterogeneity**: Multiple distinct agent types (Quadcopters, Observers, Provisioners) with unique observation and action spaces, capabilities, and roles
- **Multi-Stage Benchmarking**: Three challenges (Simple Fleet, Fleet, Complex Fleet) with increasing difficulty and heterogeneity
- **Scenario Variety**: Each challenge contains several scenarios for fine control over agent compositions and environmental complexity
- **Partial Observability**: Agents perceive the world through unique, limited sensors and information, increasing coordination complexity
- **Flexible Spaces**: Both discrete and continuous action spaces are supported for all agent types
- **Extensibility**: Easily add new agent types, capabilities, and scenarios

## Agent Types

### Quadcopter
- **Role**: Low-altitude, agile agents that can reach targets
- **Capabilities**: Fast movement, target acquisition
- **Limitations**: Limited energy and capacity
- **Action Space**: Discrete(5) or Continuous(3)

### Observer
- **Role**: High-altitude, fast agents with broad forward-facing views
- **Capabilities**: Wide sensor range, guide Quadcopters
- **Limitations**: Cannot directly reach targets
- **Action Space**: Discrete(5) or Continuous(3)

### Provisioner
- **Role**: Ground vehicles navigating a road network
- **Capabilities**: Recharge/support aerial agents, assist with target retrieval
- **Limitations**: Restricted to road network
- **Action Space**: Discrete(5)

## Challenge Levels

### Simple Fleet
- **Agents**: Quadcopters + Observers
- **Objective**: Reach as many moving targets as possible
- **Key Mechanic**: Observers must guide Quadcopters
- **Scenarios**: 1q1o, 3q1o, 5q2o

### Fleet
- **Agents**: Quadcopters + Observers
- **Objective**: Multi-target acquisition with constraints
- **Key Mechanics**: Energy constraints, obstacles, limited communication range
- **Scenarios**: 3q1o, 10q3o, 20q5o

### Complex Fleet
- **Agents**: Quadcopters + Observers + Provisioners
- **Objective**: High heterogeneity cooperation
- **Key Mechanics**: Energy/capacity limits, provisioners restricted to roads, complex cooperation
- **Scenarios**: 3q1o1p, 5q2o1p

## Environment Details

- **Map**: Randomly generated with obstacles and special structures
- **Targets**: Moving targets that agents must find and reach
- **Observation**: Agent-specific local observations based on sensors and roles
- **Rewards**: Cooperative team reward based on targets reached
- **Episode Length**: Configurable (default: 300-900 steps)

## Usage Example

```python
from hemac import HeMAC_v0

env = HeMAC_v0.env(render_mode="human")
env.reset(seed=0)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # Insert your policy here
        action = env.action_space(agent).sample()
    env.step(action)
env.close()
```

## Research Insights

The HeMAC paper shows that while state-of-the-art methods (like MAPPO) excel at simpler tasks, their performance degrades with increased heterogeneity. Simpler algorithms (like IPPO) sometimes outperform them under these conditions, highlighting the unique challenges of heterogeneous multi-agent coordination.

## References

- **Paper**: Dansereau et al. (2025). "The Heterogeneous Multi-Agent Challenge". ECAI 2025.
- **Repository**: https://github.com/ThalesGroup/hemac
- **ArXiv**: https://arxiv.org/abs/2509.19512
