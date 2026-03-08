# Hanabi (hanabi_v5)

## Overview
Hanabi is a cooperative card game where 2-5 players work together to play cards in the correct order. Players can see everyone's cards except their own, and must give hints to help teammates play cards correctly. The goal is to maximize the team score.

## Environment Details
- **Type**: Turn-based, cooperative
- **Players**: 2-5 agents
- **Action Space**: Discrete(variable)
  - Play card actions: 0-4 (play card from hand position)
  - Discard card actions: 5-9 (discard card from hand position)
  - Hint actions: 10+ (give color or rank hint to another player)
  - Total actions depend on number of players and hint types
- **Observation Space**: Dict with:
  - `observation`: Box containing encoded game state (visible cards, hints, fireworks, info tokens)
  - `action_mask`: Binary mask indicating legal actions

## Game Rules
1. Deck contains 5 colors × 5 ranks (1,1,1,2,2,3,3,4,4,5)
2. Players cannot see their own cards
3. On each turn, a player must:
   - Play a card (if correct, adds to fireworks; if wrong, loses a life)
   - Discard a card (gains an info token)
   - Give a hint (costs an info token, tells another player about their cards)
4. Goal: Build five fireworks (one per color) from 1 to 5
5. Game ends when: all cards played, 3 lives lost, or perfect score (25) achieved

## Rewards
- Cooperative: all players receive the same reward
- +1 for each card successfully played
- Maximum score: 25 (perfect game)
- Game ends with 0 lives: score is final fireworks total

## Usage Example
```python
from pettingzoo.classic import hanabi_v5

env = hanabi_v5.env(num_players=3, colors=5, ranks=5, hand_size=5)
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
- Fully cooperative gameplay
- Partial observability (cannot see own cards)
- Communication through hints only
- Configurable: number of players, colors, ranks, hand size
- Challenging coordination problem

## Configuration Options
- `num_players`: 2-5 (default: 2)
- `colors`: 1-5 (default: 5)
- `ranks`: 1-5 (default: 5)
- `hand_size`: 2-5 (default: 5 for 2-3 players, 4 for 4-5 players)
- `max_information_tokens`: 3-8 (default: 8)
- `max_life_tokens`: 1-3 (default: 3)

## References
- PettingZoo Documentation: https://pettingzoo.farama.org/environments/classic/hanabi/
- Hanabi Rules: https://en.wikipedia.org/wiki/Hanabi_(card_game)
- Hanabi Learning Environment: https://github.com/deepmind/hanabi-learning-environment
