# Rock Paper Scissors (rps_v2)

## Overview
Rock Paper Scissors is a simultaneous-move game where 2 players choose rock, paper, or scissors. Rock beats scissors, scissors beats paper, and paper beats rock. This is a zero-sum game often used for game theory research.

## Environment Details
- **Type**: Simultaneous-move, competitive
- **Players**: 2 agents
- **Action Space**: Discrete(3)
  - 0: Rock
  - 1: Paper
  - 2: Scissors
- **Observation Space**: Dict with:
  - `observation`: Box containing encoded game state (previous moves history)
  - `action_mask`: Binary mask (all actions always legal)

## Game Rules
1. Both players simultaneously choose rock, paper, or scissors
2. Winner determined by:
   - Rock beats Scissors
   - Scissors beats Paper
   - Paper beats Rock
   - Same choice results in a tie
3. Game continues for multiple rounds

## Rewards
- Win: +1
- Loss: -1
- Tie: 0
- Zero-sum: sum of all rewards equals 0

## Usage Example
```python
from pettingzoo.classic import rps_v2

env = rps_v2.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
```

## Key Features
- Simple simultaneous-move game
- Perfect for testing multi-agent algorithms
- No dominant strategy (Nash equilibrium: uniform random)
- Fast execution for rapid training
- Observation includes move history for learning patterns

## Strategy Notes
- Optimal strategy: play each action with 1/3 probability
- Any deterministic or biased strategy can be exploited
- Useful for testing opponent modeling and adaptation

## References
- PettingZoo Documentation: https://pettingzoo.farama.org/environments/classic/rps/
- Game Theory: https://en.wikipedia.org/wiki/Rock_paper_scissors
