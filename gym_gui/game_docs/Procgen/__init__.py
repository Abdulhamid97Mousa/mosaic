"""Procgen game documentation module.

Procgen provides 16 procedurally-generated game-like environments designed
to measure sample efficiency and generalization in reinforcement learning.

Environments (16 total):
    - BigFish, BossFight, CaveFlyer, Chaser
    - Climber, CoinRun, Dodgeball, FruitBot
    - Heist, Jumper, Leaper, Maze
    - Miner, Ninja, Plunder, StarPilot

Reference:
    Cobbe et al. (2019). Leveraging Procedural Generation to Benchmark RL.
    https://github.com/openai/procgen
"""

from __future__ import annotations

from .ProcgenEnv import (
    get_procgen_html,
    PROCGEN_ENV_DESCRIPTIONS,
    PROCGEN_HTML_MAP,
    PROCGEN_BIGFISH_HTML,
    PROCGEN_BOSSFIGHT_HTML,
    PROCGEN_CAVEFLYER_HTML,
    PROCGEN_CHASER_HTML,
    PROCGEN_CLIMBER_HTML,
    PROCGEN_COINRUN_HTML,
    PROCGEN_DODGEBALL_HTML,
    PROCGEN_FRUITBOT_HTML,
    PROCGEN_HEIST_HTML,
    PROCGEN_JUMPER_HTML,
    PROCGEN_LEAPER_HTML,
    PROCGEN_MAZE_HTML,
    PROCGEN_MINER_HTML,
    PROCGEN_NINJA_HTML,
    PROCGEN_PLUNDER_HTML,
    PROCGEN_STARPILOT_HTML,
)

__all__ = [
    "get_procgen_html",
    "PROCGEN_ENV_DESCRIPTIONS",
    "PROCGEN_HTML_MAP",
    "PROCGEN_BIGFISH_HTML",
    "PROCGEN_BOSSFIGHT_HTML",
    "PROCGEN_CAVEFLYER_HTML",
    "PROCGEN_CHASER_HTML",
    "PROCGEN_CLIMBER_HTML",
    "PROCGEN_COINRUN_HTML",
    "PROCGEN_DODGEBALL_HTML",
    "PROCGEN_FRUITBOT_HTML",
    "PROCGEN_HEIST_HTML",
    "PROCGEN_JUMPER_HTML",
    "PROCGEN_LEAPER_HTML",
    "PROCGEN_MAZE_HTML",
    "PROCGEN_MINER_HTML",
    "PROCGEN_NINJA_HTML",
    "PROCGEN_PLUNDER_HTML",
    "PROCGEN_STARPILOT_HTML",
]
