# HeMAC: Complex Fleet Challenge

## Overview

The **Complex Fleet** challenge is the most advanced scenario in HeMAC, introducing all three agent types (Quadcopters, Observers, Provisioners) with maximum heterogeneity. This challenge tests the limits of heterogeneous multi-agent coordination.

## Agent Composition

- **Quadcopters**: Agile, low-altitude agents with limited energy and capacity
- **Observers**: High-altitude scouts with broad views
- **Provisioners**: Ground vehicles restricted to road networks

## Objective

Reach as many moving targets as possible while managing energy constraints, capacity limits, provisioner logistics, and complex three-way coordination between heterogeneous agent types.

## Key Mechanics

- **Energy Constraints**: Quadcopters have limited energy that depletes with movement
- **Capacity Limits**: Quadcopters can only carry limited targets
- **Road Network**: Provisioners are restricted to predefined road networks
- **Recharging**: Provisioners can recharge quadcopters at designated locations
- **Target Retrieval**: Provisioners assist with target retrieval and transport
- **Obstacles**: Complex environment with obstacles blocking movement and sensors
- **Limited Communication**: Agents can only communicate within limited range
- **Three-Way Coordination**: Observers guide, Quadcopters execute, Provisioners support

## Scenarios

### 3q1o1p (3 Quadcopters, 1 Observer, 1 Provisioner)
- **Difficulty**: Hard
- **Focus**: Basic three-agent-type coordination
- **Learning Goal**: Understand provisioner role and logistics

### 5q2o1p (5 Quadcopters, 2 Observers, 1 Provisioner)
- **Difficulty**: Very Hard
- **Focus**: Scaled coordination with single provisioner bottleneck
- **Learning Goal**: Resource allocation and provisioner scheduling

## Action Spaces

### Quadcopter
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands
- **Energy Cost**: Each action consumes energy
- **Capacity**: Limited target carrying capacity

### Observer
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Continuous(3)**: [vx, vy, vz] velocity commands
- **No Constraints**: Unlimited energy, no capacity limits

### Provisioner
- **Discrete(5)**: NOOP, FORWARD, BACKWARD, LEFT, RIGHT
- **Road Constraint**: Can only move on predefined road network
- **Support Actions**: Recharge quadcopters, assist with target retrieval

## Observation Space

Each agent receives:
- **Position**: Own position in the environment
- **Velocity**: Own velocity vector
- **Energy**: Current energy level (quadcopters only)
- **Capacity**: Current capacity usage (quadcopters only)
- **Sensor Data**: Detected targets, agents, and obstacles within sensor range
- **Communication**: Information from agents within communication range
- **Road Network**: Nearby road segments (provisioners only)
- **Map Features**: Nearby obstacles and terrain

## Rewards

- **Team Reward**: +1 for each target reached and retrieved
- **Energy Penalty**: Small penalty for energy depletion
- **Capacity Penalty**: Penalty for exceeding capacity
- **Cooperative**: All agents receive the same reward
- **Episode**: Cumulative reward over the episode

## Challenges

1. **Three-Way Coordination**: Coordinating three heterogeneous agent types
2. **Energy Management**: Quadcopters must balance exploration and energy conservation
3. **Capacity Management**: Quadcopters must manage target carrying capacity
4. **Provisioner Logistics**: Provisioners must efficiently support multiple quadcopters
5. **Road Network Constraint**: Provisioners are limited to road network
6. **Communication Limits**: Coordination is limited by communication range
7. **Scalability**: Large numbers of agents require efficient coordination
8. **Dynamic Targets**: Targets move, requiring adaptive strategies

## Training Tips

1. **Hierarchical Learning**: Use hierarchical RL with role-specific policies
2. **Energy Awareness**: Train quadcopters to monitor and manage energy levels
3. **Capacity Planning**: Train quadcopters to plan target retrieval based on capacity
4. **Provisioner Scheduling**: Develop efficient provisioner routing and scheduling
5. **Observer Efficiency**: Observers should maximize coverage and guide both quadcopters and provisioners
6. **Communication Protocol**: Develop efficient three-way communication strategies
7. **Curriculum**: Progress from 3q1o1p → 5q2o1p
8. **Multi-Agent Credit Assignment**: Address credit assignment across heterogeneous agents

## Benchmark Results

From the ECAI 2025 paper:
- **MAPPO**: Struggles with high heterogeneity, performance degrades significantly
- **IPPO**: More robust but lower peak performance
- **Hierarchical Methods**: Show promise but require careful design
- **Centralized Training**: Benefits from centralized critic with decentralized execution

## Usage Example

```python
from hemac import HeMAC_v0

# Complex Fleet 3q1o1p scenario
env = HeMAC_v0.env(
    n_drones=3,
    n_observers=1,
    n_provisioners=1,
    max_cycles=900,
    min_obstacles=5,
    max_obstacles=10,
    observer_comm_range=150,
    rescuing_targets=True,  # Enable target retrieval
    render_mode="human"
)

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # Agent-specific logic
        if "provisioner" in agent:
            # Provisioner: move to support quadcopters
            action = env.action_space(agent).sample()
        elif "observer" in agent:
            # Observer: scout and guide
            action = env.action_space(agent).sample()
        else:  # quadcopter
            # Quadcopter: check energy and capacity
            if "energy" in observation and observation["energy"] < 0.2:
                action = 0  # Return to provisioner
            elif "capacity" in observation and observation["capacity"] > 0.8:
                action = 0  # Return to base
            else:
                action = env.action_space(agent).sample()
    env.step(action)
env.close()
```

## Research Insights

The Complex Fleet challenge highlights the unique difficulties of heterogeneous multi-agent coordination:
- **Heterogeneity Penalty**: Performance degrades significantly with increased agent type diversity
- **Credit Assignment**: Difficult to assign credit across agents with different capabilities
- **Communication Overhead**: Three-way communication is more complex than two-way
- **Logistics Complexity**: Provisioner scheduling adds significant complexity
- **Emergent Behavior**: Successful strategies often involve emergent coordination patterns

This challenge is ideal for research on:
- Heterogeneous multi-agent RL algorithms
- Hierarchical multi-agent learning
- Multi-agent credit assignment
- Communication protocols for heterogeneous teams
- Logistics and resource allocation in multi-agent systems
