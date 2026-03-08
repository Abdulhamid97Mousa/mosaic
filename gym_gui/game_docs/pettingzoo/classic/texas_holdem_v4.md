# Texas Hold'em (texas_holdem_v4)

## Overview
Texas Hold'em is a popular poker variant where 2-6 players compete to win chips by forming the best five-card hand using their two private cards and five community cards. This is a turn-based, zero-sum game with betting rounds.

## Environment Details
- **Type**: Turn-based, competitive
- **Players**: 2-6 agents
- **Action Space**: Discrete(4)
  - 0: Fold (forfeit current hand)
  - 1: Call (match current bet)
  - 2: Raise (increase bet by fixed amount)
  - 3: Check (pass action if no bet to call)
- **Observation Space**: Dict with:
  - `observation`: Box containing encoded game state (hand cards, community cards, betting info)
  - `action_mask`: Binary mask indicating legal actions

## Game Rules
1. Each player receives 2 private hole cards
2. Betting round 1 (pre-flop)
3. 3 community cards revealed (flop)
4. Betting round 2
5. 1 community card revealed (turn)
6. Betting round 3
7. 1 community card revealed (river)
8. Final betting round
9. Showdown: best 5-card hand wins

## Rewards
- Winner receives all chips from the pot
- Losers receive negative rewards equal to chips lost
- Zero-sum: sum of all rewards equals 0

## Usage Example
```python
from pettingzoo.classic import texas_holdem_v4

env = texas_holdem_v4.env(num_players=4)
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
- Standard 52-card deck
- Fixed raise amounts
- Automatic pot management
- Action masking for legal moves only
- Supports 2-6 players

## References
- PettingZoo Documentation: https://pettingzoo.farama.org/environments/classic/texas_holdem/
- Texas Hold'em Rules: https://en.wikipedia.org/wiki/Texas_hold_%27em
