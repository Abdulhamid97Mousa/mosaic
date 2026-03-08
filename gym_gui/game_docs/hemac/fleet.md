# HeMAC: Fleet Challenge

## Overview

The **Fleet** challenge is the intermediate-level scenario in HeMAC, introducing energy constraints, obstacles, and limited communication range. It features Quadcopters and Observers working together under more realistic constraints.

## Agent Composition

- **Quadcopters**: Agile, low-altitude agents with limited energy
- **Observers**: High-altitude scouts with broad views

## Objective

Reach as many moving targets as possible while managing energy constraints, navigating obstacles, and coordinating within limited communication range.

## Key Mechanics

- **Energy Constraints**: Quadcopters have limited energy that depletes with movement
- **Obstacles**: Environment contains obstacles that block movement and sensors
- **Limited Communication**: Agents can only communicate within a limited range
- **Target Movement**: Targets move dynamically, requiring adaptive strategies
- **Recharging**: Quadcopters must return to base or find energy sources

## Scenarios

### 3q1o (3 Quadcopters, 1 Observer)
- **Difficulty**: Medium
- **Focus**: Energy management with basic coordination
- **Learning Goal**: Balance exploration and energy conservation

### 10q3o (10 Quadcopters, 3 Observers)
- **Difficulty**: Hard
- **Focus**: Large-scale coordination with multiple observers
- **Learning Goal**: Scalable multi-agent strategies

### 20q5o (20 Quadcopters, 5 Observers)
- **Difficulty**: Very Hard
- **Focus**: Massive-scale heterogeneous coordination
- **Learning Goal**: Emergent coordination patterns

## Action Spaces

### Quadcopter
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands
- **Energy Cost**: Each action consumes energy

### Observer
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands
- **No Energy Constraint**: Observers have unlimited energy

## Observation Space

Each agent receives:
- **Position**: Own position in the environment
- **Velocity**: Own velocity vector
- **Energy**: Current energy level (quadcopters only)
- **Sensor Data**: Detected targets, agents, and obstacles within sensor range
- **Communication**: Information from agents within communication range
- **Map Features**: Nearby obstacles and terrain

## Rewards

- **Team Reward**: +1 for each target reached by any quadcopter
- **Energy Penalty**: Small penalty for energy depletion
- **Cooperative**: All agents receive the same reward
- **Episode**: Cumulative reward over the episode

## Challenges

1. **Energy Management**: Quadcopters must balance exploration and energy conservation
2. **Obstacle Navigation**: Agents must navigate around obstacles
3. **Communication Limits**: Coordination is limited by communication range
4. **Scalability**: Large numbers of agents require efficient coordination
5. **Dynamic Targets**: Targets move, requiring adaptive strategies

## Training Tips

1. **Energy Awareness**: Train quadcopters to monitor and manage energy levels
2. **Observer Efficiency**: Observers should maximize coverage with minimal movement
3. **Communication Protocol**: Develop efficient communication strategies
4. **Obstacle Avoidance**: Incorporate obstacle detection and avoidance
5. **Hierarchical Learning**: Use hierarchical RL for large-scale scenarios
6. **Curriculum**: Progress from 3q1o → 10q3o → 20q5o

## Benchmark Results

From the ECAI 2025 paper:
- **MAPPO**: Performance degrades significantly with scale
- **IPPO**: More robust to scale, but lower peak performance
- **Hierarchical Methods**: Show promise for 20q5o scenario

## Usage Example

```python
from hemac import HeMAC_v0

# Fleet 10q3o scenario
env = HeMAC_v0.env(
    n_drones=10,
    n_observers=3,
    n_provisioners=0,
    max_cycles=600,
    min_obstacles=5,
    max_obstacles=10,
    observer_comm_range=150,
    render_mode="human"
)

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # Check energy level for quadcopters
        if "energy" in observation and observation["energy"] < 0.2:
            action = 0  # Return to base or conserve energy
        else:
            action = env.action_space(agent).sample()
    env.step(action)
env.close()
```
