# Frozen Lake - Gymnasium Documentation

Source: https://gymnasium.farama.org/environments/toy_text/frozen_lake/

## Description
Frozen lake involves crossing a frozen lake from start to goal without falling into any holes by walking over the frozen lake. The player may not always move in the intended direction due to the slippery nature of the frozen lake.

The game starts with the player at location `[0,0]` of the frozen lake grid world with the goal located at far extent of the world e.g. `[3,3]` for the 4x4 environment.

## Default Maps

### 4x4 Map
```
"SFFF",
"FHFH",
"FFFH",
"HFFG"
```
**Hole count: 4 holes** (H tiles)

### 8x8 Map  
```
"SFFFFFFF",
"FFFFFFFF",
"FFFHFFFF",
"FFFFFHFF",
"FFFHFFFF",
"FHHFFFHF",
"FHFFHFHF",
"FFFHFFFG",
```
**Hole count: 10 holes** (H tiles)
- Row 2, col 3: H
- Row 3, col 5: H
- Row 4, col 3: H
- Row 5, col 1-2: HH
- Row 5, col 7: H
- Row 6, col 1: H
- Row 6, col 4: H
- Row 6, col 6: H
- Row 7, col 3: H

## Arguments

- `desc=None`: Used to specify maps non-preloaded maps. If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are `None` a random 8x8 map with 80% of locations frozen will be generated.

- `map_name="4x4"` or `"8x8"`: Helps load two predefined map names

- `is_slippery=True`: If true the player will move in intended direction with probability specified by the `success_rate`

- `success_rate=1.0/3.0`: Used to specify the probability of moving in the intended direction when is_slippery=True

- `reward_schedule=(1, 0, 0)`: Used to specify reward amounts for reaching certain tiles (Goal, Hole, Frozen)

## Tile Letters
- "S" for Start tile
- "G" for Goal tile  
- "F" for frozen tile
- "H" for a tile with a hole

## Assets Attribution
Elf and stool from https://franuka.itch.io/rpg-snow-tileset
All other assets by Mel Tillery http://www.cyaneus.com/

## Episode Truncation
- 100 steps for FrozenLake4x4
- 200 steps for FrozenLake8x8
