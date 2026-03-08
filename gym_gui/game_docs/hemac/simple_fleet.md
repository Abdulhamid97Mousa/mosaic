# HeMAC: Simple Fleet Challenge

## Overview

The **Simple Fleet** challenge is the entry-level scenario in HeMAC, designed to introduce the basic mechanics of heterogeneous multi-agent cooperation. It features Quadcopters and Observers working together to reach moving targets.

## Agent Composition

- **Quadcopters**: Agile, low-altitude agents that can reach targets
- **Observers**: High-altitude scouts with broad forward-facing views

## Objective

Reach as many moving targets as possible within the episode time limit. Observers must guide Quadcopters to targets using their superior sensor range.

## Key Mechanics

- **Partial Observability**: Each agent has limited sensor range
- **Coordination**: Observers see targets from afar and must guide Quadcopters
- **Target Movement**: Targets move randomly, requiring dynamic coordination
- **Communication**: Implicit through shared observations (within communication range)

## Scenarios

### 1q1o (1 Quadcopter, 1 Observer)
- **Difficulty**: Easy
- **Focus**: Basic coordination between two heterogeneous agents
- **Learning Goal**: Understand observer-quadcopter communication

### 3q1o (3 Quadcopters, 1 Observer)
- **Difficulty**: Medium
- **Focus**: One observer coordinating multiple quadcopters
- **Learning Goal**: Resource allocation and prioritization

### 5q2o (5 Quadcopters, 2 Observers)
- **Difficulty**: Hard
- **Focus**: Multiple observers coordinating multiple quadcopters
- **Learning Goal**: Scalable coordination strategies

## Action Spaces

### Quadcopter
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands

### Observer
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands

## Observation Space

Each agent receives:
- **Position**: Own position in the environment
- **Velocity**: Own velocity vector
- **Sensor Data**: Detected targets and other agents within sensor range
- **Communication**: Information from nearby agents

## Rewards

- **Team Reward**: +1 for each target reached by any quadcopter
- **Cooperative**: All agents receive the same reward
- **Episode**: Cumulative reward over the episode

## Training Tips

1. **Start Simple**: Begin with 1q1o to understand basic coordination
2. **Observer Strategy**: Train observers to scout and communicate target locations
3. **Quadcopter Strategy**: Train quadcopters to follow observer guidance
4. **Curriculum Learning**: Progress from 1q1o → 3q1o → 5q2o
5. **Communication**: Leverage implicit communication through shared observations

## Benchmark Results

From the ECAI 2025 paper:
- **MAPPO**: Strong performance on 1q1o, degrades on 5q2o
- **IPPO**: Consistent performance across scenarios
- **Random**: Baseline performance ~10-20% of optimal

## Usage Example

```python
from hemac import HeMAC_v0

# Simple Fleet 3q1o scenario
env = HeMAC_v0.env(
    n_drones=3,
    n_observers=1,
    n_provisioners=0,
    max_cycles=300,
    render_mode="human"
)

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
env.close()
```
