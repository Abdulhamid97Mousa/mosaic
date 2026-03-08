# Leduc Hold'em (leduc_holdem_v4)

## Overview
Leduc Hold'em is a simplified poker variant designed for AI research. It uses a 6-card deck (2 suits × 3 ranks) with 2 players, making it more tractable for game-theoretic analysis while retaining key poker elements like betting and bluffing.

## Environment Details
- **Type**: Turn-based, competitive
- **Players**: 2 agents
- **Action Space**: Discrete(4)
  - 0: Fold (forfeit current hand)
  - 1: Call (match current bet)
  - 2: Raise (increase bet by fixed amount)
  - 3: Check (pass action if no bet to call)
- **Observation Space**: Dict with:
  - `observation`: Box containing encoded game state (private card, public card, betting history)
  - `action_mask`: Binary mask indicating legal actions

## Game Rules
1. 6-card deck: Jack, Queen, King in 2 suits (6 cards total)
2. Each player receives 1 private card
3. First betting round (2 bet limit)
4. 1 public community card revealed
5. Second betting round (4 bet limit)
6. Showdown: player with matching card (pair) or higher card wins

## Rewards
- Winner receives all chips from the pot
- Loser receives negative reward equal to chips lost
- Zero-sum: sum of all rewards equals 0

## Usage Example
```python
from pettingzoo.classic import leduc_holdem_v4

env = leduc_holdem_v4.env()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(observation["action_mask"])
    env.step(action)
```

## Key Features
- Simplified 6-card deck for tractability
- Two betting rounds with different limits
- Pairs beat high cards
- Ideal for game theory research
- Fast gameplay for training

## References
- PettingZoo Documentation: https://pettingzoo.farama.org/environments/classic/leduc_holdem/
- Original Paper: Southey et al. (2005) "Bayes' Bluff: Opponent Modelling in Poker"
