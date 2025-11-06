# ALE (Atari) — Introduction and Integration

This brief introduces the Atari Learning Environment (ALE) game family in our GUI, adds initial ALE game pages, and documents the code files involved in bringing ALE into the app. We support these variants to start: `Adventure-v4`, `ALE/Adventure-v5`, `AirRaid-v4`, `ALE/AirRaid-v5`, `Assault-v4`, and `ALE/Assault-v5`.

## What is ALE?

ALE is the canonical interface for Atari 2600 environments used in RL research. It exposes multiple observation types (RGB, RAM, grayscale) and a discrete action space (often `Discrete(18)` for full action sets). See the upstream docs: <https://ale.farama.org/environments/>

## New game family: ALE

- Family name: `ALE` (separate from Gym’s `Atari` and similar to how `MiniGrid` is modelled)
- Initial games: Adventure, AirRaid, Assault
  - Variants: `Adventure-v4`, `ALE/Adventure-v5`, `AirRaid-v4`, `ALE/AirRaid-v5`, `Assault-v4`, `ALE/Assault-v5`

## Files involved (code)

- Adapters
  - `gym_gui/core/adapters/ale.py`
    - `ALEAdapter`: base RGB-array adapter for ALE environments
    - `AdventureV4Adapter`: id=`Adventure-v4`
    - `AdventureV5Adapter`: id=`ALE/Adventure-v5`
    - `AirRaidV4Adapter`: id=`AirRaid-v4`
    - `AirRaidV5Adapter`: id=`ALE/AirRaid-v5`
    - `AssaultV4Adapter`: id=`Assault-v4`
    - `AssaultV5Adapter`: id=`ALE/Assault-v5`
    - `ALE_ADAPTERS`: mapping from `GameId` to adapter class
  - `gym_gui/core/adapters/__init__.py`
    - Now exports `ALEAdapter`, `AdventureV4Adapter`, `AdventureV5Adapter`, `AirRaidV4Adapter`, `AirRaidV5Adapter`, `AssaultV4Adapter`, `AssaultV5Adapter`, and `ALE_ADAPTERS`

- Enums and defaults
  - `gym_gui/core/enums.py`
    - New family: `EnvironmentFamily.ALE`
    - New game ids: `GameId.ADVENTURE_V4`, `GameId.ALE_ADVENTURE_V5`, `GameId.AIR_RAID_V4`, `GameId.ALE_AIR_RAID_V5`, `GameId.ASSAULT_V4`, `GameId.ALE_ASSAULT_V5`
    - Display naming: `get_game_display_name` preserves the `ALE/` namespace (returns values as-is for ALE entries)
    - Default maps updated:
      - `ENVIRONMENT_FAMILY_BY_GAME[ADVENTURE_V4|ALE_ADVENTURE_V5] = EnvironmentFamily.ALE`
      - Similar ALE family mappings for AirRaid and Assault
      - `DEFAULT_RENDER_MODES[...] = RenderMode.RGB_ARRAY` for ALE games
      - `DEFAULT_CONTROL_MODES[...]` for ALE games include human play: `(HUMAN_ONLY, AGENT_ONLY, HYBRID_TURN_BASED, HYBRID_HUMAN_AGENT)`

- Human input (optional mappings)
  - `gym_gui/controllers/human_input.py`
    - Added `_ALE_MAPPINGS` for Adventure (Discrete(18)), AirRaid (Discrete(6)), and Assault (Discrete(7))
      - Basic moves: arrows/WASD; fire: Space; diagonals: Q/E/Z/C; fire-combos on I/J/K/L/U/O/N/M
      - Fallback still applies for other ALE games if not explicitly mapped

## Files involved (docs)

- Game docs
  - `gym_gui/game_docs/ALE/__init__.py`
    - Overview, action meanings, observation types, and supported variants for Adventure, AirRaid, and Assault

- Day-21 knowledge base (this file)
  - `docs/1.0_DAY_21/TASK_4/README.md`
    - Introduction, integration notes, and file inventory

## Supported variants and making environments

- `Adventure-v4` (classic v4)
  - Example: `gymnasium.make("Adventure-v4")`
- `ALE/Adventure-v5` (ALE namespace v5)
  - Example: `gymnasium.make("ALE/Adventure-v5")`
- `AirRaid-v4` and `ALE/AirRaid-v5`
  - Example: `gymnasium.make("AirRaid-v4")`, `gymnasium.make("ALE/AirRaid-v5")`
- `Assault-v4` and `ALE/Assault-v5`
  - Example: `gymnasium.make("Assault-v4")`, `gymnasium.make("ALE/Assault-v5")`

Both use RGB rendering in the adapter. Additional observation types (RAM, grayscale) can be enabled in future by extending the adapter’s `gym_kwargs()` and processing hooks.

## Setup notes

- Install dependencies: ensure `gymnasium`, `ale-py`, and `AutoROM[accept-rom-license]` are installed. ROMs must be installed via AutoROM.
- Namespace registration: our ALE adapter module imports `ale_py.env` at import-time to auto-register the `ALE/` namespace in Gymnasium. If you construct environments directly (without importing our adapter), import `ale_py.env` yourself first or use the adapter path.

## Next steps

- Add more ALE games (e.g., Breakout/Pong under ALE namespace)
- Optional: expose `difficulty` and `mode` as adapter config
- Optional: human-input presets per ALE game for better playability
