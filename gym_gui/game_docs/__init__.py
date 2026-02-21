"""Aggregated environment documentation helpers."""

from __future__ import annotations

from typing import Dict

from gym_gui.core.enums import GameId
from gym_gui.game_docs.Gymnasium.ToyText import (
    TAXI_HTML,
    FROZEN_HTML,
    FROZEN_V2_HTML,
    CLIFF_HTML,
    BLACKJACK_HTML,
)
from gym_gui.game_docs.Gymnasium.Box2D import (
    LUNAR_LANDER_HTML,
    CAR_RACING_HTML,
    BIPEDAL_WALKER_HTML,
)
from gym_gui.game_docs.Gymnasium.MuJuCo import (
    ANT_HTML,
    HALF_CHEETAH_HTML,
    HOPPER_HTML,
    HUMANOID_HTML,
    HUMANOID_STANDUP_HTML,
    INVERTED_DOUBLE_PENDULUM_HTML,
    INVERTED_PENDULUM_HTML,
    PUSHER_HTML,
    REACHER_HTML,
    SWIMMER_HTML,
    WALKER2D_HTML,
)
from gym_gui.game_docs.MiniGrid import (
    MINIGRID_EMPTY_HTML,
    get_empty_html,
    MINIGRID_DOORKEY_HTML,
    get_doorkey_html,
    MINIGRID_LAVAGAP_HTML,
    get_lavagap_html,
    MINIGRID_DYNAMIC_OBSTACLES_HTML,
    get_dynamic_obstacles_html,
    MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML,
    MINIGRID_MULTIROOM_HTML,
    get_multiroom_html,
    MINIGRID_OBSTRUCTED_MAZE_HTML,
    get_obstructed_maze_html,
    MINIGRID_CROSSING_HTML,
    get_crossing_html,
    MINIGRID_REDBLUEDOORS_HTML,
    get_redbluedoors_html,
)
from gym_gui.game_docs.ALE import (
    ADVENTURE_HTML,
    AIR_RAID_HTML,
    ASSAULT_HTML,
)

from gym_gui.game_docs.crafter import (
    CRAFTER_REWARD_HTML,
    CRAFTER_NO_REWARD_HTML,
)

from gym_gui.game_docs.OpenSpiel import (
    CHECKERS_HTML,
)

from gym_gui.game_docs.Draughts import (
    AMERICAN_CHECKERS_HTML,
    RUSSIAN_CHECKERS_HTML,
    INTERNATIONAL_DRAUGHTS_HTML,
)

from gym_gui.game_docs.PettingZoo import (
    CHESS_HTML,
    CONNECT_FOUR_HTML,
    GO_HTML,
)

from gym_gui.game_docs.Mosaic_MultiGrid import (
    MOSAIC_SOCCER_HTML,
    MOSAIC_SOCCER_BASE_HTML,
    MOSAIC_SOCCER_1VS1_HTML,
    MOSAIC_COLLECT_HTML,
    MOSAIC_COLLECT_BASE_HTML,
    MOSAIC_COLLECT2VS2_HTML,
    MOSAIC_COLLECT2VS2_BASE_HTML,
    MOSAIC_BASKETBALL_HTML,
    MOSAIC_OVERVIEW_HTML,
    get_mosaic_multigrid_html,
)
from gym_gui.game_docs.MultiGrid_INI import (
    get_ini_multigrid_html,
)

from gym_gui.game_docs.SMAC import (
    SMAC_3M_HTML,
    SMAC_8M_HTML,
    SMAC_2S3Z_HTML,
    SMAC_3S5Z_HTML,
    SMAC_5M_VS_6M_HTML,
    SMAC_MMM2_HTML,
)

from gym_gui.game_docs.SMACv2 import (
    SMACV2_TERRAN_HTML,
    SMACV2_PROTOSS_HTML,
    SMACV2_ZERG_HTML,
)

from gym_gui.game_docs.Procgen import (
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

from gym_gui.game_docs import BabyAI
from gym_gui.game_docs.BabyAI import (
    # GoTo family
    BABYAI_GOTO_REDBALL_GREY_HTML, get_goto_redball_grey_html,
    BABYAI_GOTO_REDBALL_HTML, get_goto_redball_html,
    BABYAI_GOTO_REDBALL_NODISTS_HTML, get_goto_redball_nodists_html,
    BABYAI_GOTO_OBJ_HTML, get_goto_obj_html,
    BABYAI_GOTO_LOCAL_HTML, get_goto_local_html,
    BABYAI_GOTO_HTML, get_goto_html,
    BABYAI_GOTO_IMPUNLOCK_HTML, get_goto_impunlock_html,
    BABYAI_GOTO_SEQ_HTML, get_goto_seq_html,
    BABYAI_GOTO_REDBLUEBALL_HTML, get_goto_redblueball_html,
    BABYAI_GOTO_DOOR_HTML, get_goto_door_html,
    BABYAI_GOTO_OBJDOOR_HTML, get_goto_objdoor_html,
    # Open family
    BABYAI_OPEN_HTML, get_open_html,
    BABYAI_OPEN_REDDOOR_HTML, get_open_reddoor_html,
    BABYAI_OPEN_DOOR_HTML, get_open_door_html,
    BABYAI_OPEN_TWODOORS_HTML, get_open_twodoors_html,
    BABYAI_OPEN_DOORSORDER_HTML, get_open_doorsorder_html,
    # Pickup family
    BABYAI_PICKUP_HTML, get_pickup_html,
    BABYAI_UNBLOCK_PICKUP_HTML, get_unblock_pickup_html,
    BABYAI_PICKUP_LOC_HTML, get_pickup_loc_html,
    BABYAI_PICKUP_DIST_HTML, get_pickup_dist_html,
    BABYAI_PICKUP_ABOVE_HTML, get_pickup_above_html,
    # Unlock family
    BABYAI_UNLOCK_HTML, get_unlock_html,
    BABYAI_UNLOCK_LOCAL_HTML, get_unlock_local_html,
    BABYAI_KEY_INBOX_HTML, get_key_inbox_html,
    BABYAI_UNLOCK_PICKUP_HTML, get_unlock_pickup_html,
    BABYAI_BLOCKED_UNLOCK_PICKUP_HTML, get_blocked_unlock_pickup_html,
    BABYAI_UNLOCK_TO_UNLOCK_HTML, get_unlock_to_unlock_html,
    # PutNext family
    BABYAI_PUTNEXT_LOCAL_HTML, get_putnext_local_html,
    BABYAI_PUTNEXT_HTML, get_putnext_html,
    # Complex environments
    BABYAI_ACTION_OBJDOOR_HTML, get_action_objdoor_html,
    BABYAI_FINDOBJ_HTML, get_findobj_html,
    BABYAI_KEYCORRIDOR_HTML, get_keycorridor_html,
    BABYAI_ONEROOM_HTML, get_oneroom_html,
    BABYAI_MOVETWOACROSS_HTML, get_movetwoacross_html,
    BABYAI_SYNTH_HTML, get_synth_html,
    BABYAI_SYNTHLOC_HTML, get_synthloc_html,
    BABYAI_SYNTHSEQ_HTML, get_synthseq_html,
    BABYAI_MINIBOSSLEVEL_HTML, get_minibosslevel_html,
    BABYAI_BOSSLEVEL_HTML, get_bosslevel_html,
    BABYAI_BOSSLEVEL_NOUNLOCK_HTML, get_bosslevel_nounlock_html,
)

try:  # Optional dependency for ViZDoom docs (kept lightweight)
    from gym_gui.game_docs.ViZDoom import (  # pragma: no cover - documentation only
        VIZDOOM_BASIC_HTML,
        VIZDOOM_DEADLY_CORRIDOR_HTML,
        VIZDOOM_DEFEND_THE_CENTER_HTML,
        VIZDOOM_DEFEND_THE_LINE_HTML,
        VIZDOOM_DEATHMATCH_HTML,
        VIZDOOM_HEALTH_GATHERING_HTML,
        VIZDOOM_HEALTH_GATHERING_SUPREME_HTML,
        VIZDOOM_MY_WAY_HOME_HTML,
        VIZDOOM_PREDICT_POSITION_HTML,
        VIZDOOM_TAKE_COVER_HTML,
    )
    _VIZDOOM_DOCS_AVAILABLE = True
except Exception:  # pragma: no cover - ViZDoom optional
    _VIZDOOM_DOCS_AVAILABLE = False

try:  # Optional dependency for NetHack/MiniHack docs
    from gym_gui.game_docs.NetHack import (  # pragma: no cover - documentation only
        NETHACK_CONTROLS_HTML,
        # Navigation
        MINIHACK_ROOM_HTML,
        MINIHACK_CORRIDOR_HTML,
        MINIHACK_MAZEWALK_HTML,
        MINIHACK_RIVER_HTML,
        # Skills
        MINIHACK_SKILLS_SIMPLE_HTML,
        MINIHACK_SKILLS_LAVA_HTML,
        MINIHACK_SKILLS_WOD_HTML,
        MINIHACK_SKILLS_QUEST_HTML,
        # Exploration
        MINIHACK_EXPLORE_MAZE_HTML,
        MINIHACK_HIDENSEEK_HTML,
        MINIHACK_MEMENTO_HTML,
    )
    _NETHACK_DOCS_AVAILABLE = True
except Exception:  # pragma: no cover - NetHack optional
    _NETHACK_DOCS_AVAILABLE = False

_DEFAULT_DOC = (
    "<h3>Documentation unavailable</h3>"
    "<p>This environment does not yet have a descriptive blurb."
    " Check the upstream project for details.</p>"
)

GAME_INFO: Dict[GameId, str] = {
    GameId.TAXI: TAXI_HTML,
    GameId.FROZEN_LAKE: FROZEN_HTML,
    GameId.FROZEN_LAKE_V2: FROZEN_V2_HTML,
    GameId.CLIFF_WALKING: CLIFF_HTML,
    GameId.BLACKJACK: BLACKJACK_HTML,
    GameId.LUNAR_LANDER: LUNAR_LANDER_HTML,
    GameId.CAR_RACING: CAR_RACING_HTML,
    GameId.BIPEDAL_WALKER: BIPEDAL_WALKER_HTML,
    GameId.ANT: ANT_HTML,
    GameId.HALF_CHEETAH: HALF_CHEETAH_HTML,
    GameId.HOPPER: HOPPER_HTML,
    GameId.HUMANOID: HUMANOID_HTML,
    GameId.HUMANOID_STANDUP: HUMANOID_STANDUP_HTML,
    GameId.INVERTED_DOUBLE_PENDULUM: INVERTED_DOUBLE_PENDULUM_HTML,
    GameId.INVERTED_PENDULUM: INVERTED_PENDULUM_HTML,
    GameId.PUSHER: PUSHER_HTML,
    GameId.REACHER: REACHER_HTML,
    GameId.SWIMMER: SWIMMER_HTML,
    GameId.WALKER2D: WALKER2D_HTML,
    GameId.CRAFTER_REWARD: CRAFTER_REWARD_HTML,
    GameId.CRAFTER_NO_REWARD: CRAFTER_NO_REWARD_HTML,
}

_MINIGRID_EMPTY_VARIANTS = (
    GameId.MINIGRID_EMPTY_5x5,
    GameId.MINIGRID_EMPTY_RANDOM_5x5,
    GameId.MINIGRID_EMPTY_6x6,
    GameId.MINIGRID_EMPTY_RANDOM_6x6,
    GameId.MINIGRID_EMPTY_8x8,
    GameId.MINIGRID_EMPTY_16x16,
)
GAME_INFO.update({gid: MINIGRID_EMPTY_HTML for gid in _MINIGRID_EMPTY_VARIANTS})

_MINIGRID_DOORKEY_VARIANTS = (
    GameId.MINIGRID_DOORKEY_5x5,
    GameId.MINIGRID_DOORKEY_6x6,
    GameId.MINIGRID_DOORKEY_8x8,
    GameId.MINIGRID_DOORKEY_16x16,
)
GAME_INFO.update({gid: MINIGRID_DOORKEY_HTML for gid in _MINIGRID_DOORKEY_VARIANTS})

_MINIGRID_LAVAGAP_VARIANTS = (
    GameId.MINIGRID_LAVAGAP_S5,
    GameId.MINIGRID_LAVAGAP_S6,
    GameId.MINIGRID_LAVAGAP_S7,
)
GAME_INFO.update({gid: MINIGRID_LAVAGAP_HTML for gid in _MINIGRID_LAVAGAP_VARIANTS})

_MINIGRID_DYNAMIC_OBSTACLES_VARIANTS = (
    GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16,
)
GAME_INFO.update({gid: MINIGRID_DYNAMIC_OBSTACLES_HTML for gid in _MINIGRID_DYNAMIC_OBSTACLES_VARIANTS})

GAME_INFO[GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP] = MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML

_MINIGRID_MULTIROOM_VARIANTS = (
    GameId.MINIGRID_MULTIROOM_N2_S4,
    GameId.MINIGRID_MULTIROOM_N4_S5,
    GameId.MINIGRID_MULTIROOM_N6,
)
GAME_INFO.update({gid: MINIGRID_MULTIROOM_HTML for gid in _MINIGRID_MULTIROOM_VARIANTS})

GAME_INFO[GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB] = MINIGRID_OBSTRUCTED_MAZE_HTML
GAME_INFO[GameId.MINIGRID_OBSTRUCTED_MAZE_FULL] = MINIGRID_OBSTRUCTED_MAZE_HTML

# RedBlueDoors variants
_MINIGRID_REDBLUEDOORS_VARIANTS = (
    GameId.MINIGRID_REDBLUE_DOORS_6x6,
    GameId.MINIGRID_REDBLUE_DOORS_8x8,
)
GAME_INFO.update({gid: MINIGRID_REDBLUEDOORS_HTML for gid in _MINIGRID_REDBLUEDOORS_VARIANTS})

# ALE mappings
GAME_INFO[GameId.ADVENTURE_V4] = ADVENTURE_HTML
GAME_INFO[GameId.ALE_ADVENTURE_V5] = ADVENTURE_HTML
GAME_INFO[GameId.AIR_RAID_V4] = AIR_RAID_HTML
GAME_INFO[GameId.ALE_AIR_RAID_V5] = AIR_RAID_HTML
GAME_INFO[GameId.ASSAULT_V4] = ASSAULT_HTML
GAME_INFO[GameId.ALE_ASSAULT_V5] = ASSAULT_HTML

if _VIZDOOM_DOCS_AVAILABLE:
    GAME_INFO.update(
        {
            GameId.VIZDOOM_BASIC: VIZDOOM_BASIC_HTML,
            GameId.VIZDOOM_PREDICT_POSITION: VIZDOOM_PREDICT_POSITION_HTML,
            GameId.VIZDOOM_TAKE_COVER: VIZDOOM_TAKE_COVER_HTML,
            GameId.VIZDOOM_DEFEND_THE_CENTER: VIZDOOM_DEFEND_THE_CENTER_HTML,
            GameId.VIZDOOM_DEFEND_THE_LINE: VIZDOOM_DEFEND_THE_LINE_HTML,
            GameId.VIZDOOM_DEADLY_CORRIDOR: VIZDOOM_DEADLY_CORRIDOR_HTML,
            GameId.VIZDOOM_HEALTH_GATHERING: VIZDOOM_HEALTH_GATHERING_HTML,
            GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: VIZDOOM_HEALTH_GATHERING_SUPREME_HTML,
            GameId.VIZDOOM_MY_WAY_HOME: VIZDOOM_MY_WAY_HOME_HTML,
            GameId.VIZDOOM_DEATHMATCH: VIZDOOM_DEATHMATCH_HTML,
        }
    )

# PettingZoo Classic mappings
GAME_INFO[GameId.CHESS] = CHESS_HTML
GAME_INFO[GameId.CONNECT_FOUR] = CONNECT_FOUR_HTML
GAME_INFO[GameId.GO] = GO_HTML

# OpenSpiel mappings
GAME_INFO[GameId.OPEN_SPIEL_CHECKERS] = CHECKERS_HTML

# Draughts/Checkers variants (custom MOSAIC implementations)
GAME_INFO[GameId.AMERICAN_CHECKERS] = AMERICAN_CHECKERS_HTML
GAME_INFO[GameId.RUSSIAN_CHECKERS] = RUSSIAN_CHECKERS_HTML
GAME_INFO[GameId.INTERNATIONAL_DRAUGHTS] = INTERNATIONAL_DRAUGHTS_HTML

# MOSAIC MultiGrid mappings (competitive team-based environments)
# Deprecated originals
GAME_INFO[GameId.MOSAIC_MULTIGRID_SOCCER] = MOSAIC_SOCCER_BASE_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT] = MOSAIC_COLLECT_BASE_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT2VS2] = MOSAIC_COLLECT2VS2_BASE_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT_1VS1] = MOSAIC_OVERVIEW_HTML
# IndAgObs variants
GAME_INFO[GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS] = MOSAIC_SOCCER_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_SOCCER_1VS1_INDAGOBS] = MOSAIC_SOCCER_1VS1_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS] = MOSAIC_COLLECT_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS] = MOSAIC_COLLECT2VS2_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT_1VS1_INDAGOBS] = MOSAIC_COLLECT2VS2_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_BASKETBALL_INDAGOBS] = MOSAIC_BASKETBALL_HTML
# TeamObs variants
GAME_INFO[GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS] = MOSAIC_SOCCER_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS] = MOSAIC_COLLECT2VS2_HTML
GAME_INFO[GameId.MOSAIC_MULTIGRID_BASKETBALL_TEAMOBS] = MOSAIC_BASKETBALL_HTML

# SMAC v1 mappings (hand-designed cooperative micromanagement maps)
GAME_INFO.update({
    GameId.SMAC_3M: SMAC_3M_HTML,
    GameId.SMAC_8M: SMAC_8M_HTML,
    GameId.SMAC_2S3Z: SMAC_2S3Z_HTML,
    GameId.SMAC_3S5Z: SMAC_3S5Z_HTML,
    GameId.SMAC_5M_VS_6M: SMAC_5M_VS_6M_HTML,
    GameId.SMAC_MMM2: SMAC_MMM2_HTML,
})

# SMACv2 mappings (procedural unit generation)
GAME_INFO.update({
    GameId.SMACV2_TERRAN: SMACV2_TERRAN_HTML,
    GameId.SMACV2_PROTOSS: SMACV2_PROTOSS_HTML,
    GameId.SMACV2_ZERG: SMACV2_ZERG_HTML,
})

# RWARE (Robotic Warehouse) mappings
try:
    from gym_gui.game_docs.RWARE import (
        RWARE_TINY_2AG_HTML,
        RWARE_TINY_4AG_HTML,
        RWARE_SMALL_2AG_HTML,
        RWARE_SMALL_4AG_HTML,
        RWARE_MEDIUM_2AG_HTML,
        RWARE_MEDIUM_4AG_HTML,
        RWARE_MEDIUM_4AG_EASY_HTML,
        RWARE_MEDIUM_4AG_HARD_HTML,
        RWARE_LARGE_4AG_HTML,
        RWARE_LARGE_4AG_HARD_HTML,
        RWARE_LARGE_8AG_HTML,
        RWARE_LARGE_8AG_HARD_HTML,
    )

    GAME_INFO.update({
        GameId.RWARE_TINY_2AG: RWARE_TINY_2AG_HTML,
        GameId.RWARE_TINY_4AG: RWARE_TINY_4AG_HTML,
        GameId.RWARE_SMALL_2AG: RWARE_SMALL_2AG_HTML,
        GameId.RWARE_SMALL_4AG: RWARE_SMALL_4AG_HTML,
        GameId.RWARE_MEDIUM_2AG: RWARE_MEDIUM_2AG_HTML,
        GameId.RWARE_MEDIUM_4AG: RWARE_MEDIUM_4AG_HTML,
        GameId.RWARE_MEDIUM_4AG_EASY: RWARE_MEDIUM_4AG_EASY_HTML,
        GameId.RWARE_MEDIUM_4AG_HARD: RWARE_MEDIUM_4AG_HARD_HTML,
        GameId.RWARE_LARGE_4AG: RWARE_LARGE_4AG_HTML,
        GameId.RWARE_LARGE_4AG_HARD: RWARE_LARGE_4AG_HARD_HTML,
        GameId.RWARE_LARGE_8AG: RWARE_LARGE_8AG_HTML,
        GameId.RWARE_LARGE_8AG_HARD: RWARE_LARGE_8AG_HARD_HTML,
    })
except ImportError:
    pass  # rware docs not available

# Procgen mappings
GAME_INFO.update({
    GameId.PROCGEN_BIGFISH: PROCGEN_BIGFISH_HTML,
    GameId.PROCGEN_BOSSFIGHT: PROCGEN_BOSSFIGHT_HTML,
    GameId.PROCGEN_CAVEFLYER: PROCGEN_CAVEFLYER_HTML,
    GameId.PROCGEN_CHASER: PROCGEN_CHASER_HTML,
    GameId.PROCGEN_CLIMBER: PROCGEN_CLIMBER_HTML,
    GameId.PROCGEN_COINRUN: PROCGEN_COINRUN_HTML,
    GameId.PROCGEN_DODGEBALL: PROCGEN_DODGEBALL_HTML,
    GameId.PROCGEN_FRUITBOT: PROCGEN_FRUITBOT_HTML,
    GameId.PROCGEN_HEIST: PROCGEN_HEIST_HTML,
    GameId.PROCGEN_JUMPER: PROCGEN_JUMPER_HTML,
    GameId.PROCGEN_LEAPER: PROCGEN_LEAPER_HTML,
    GameId.PROCGEN_MAZE: PROCGEN_MAZE_HTML,
    GameId.PROCGEN_MINER: PROCGEN_MINER_HTML,
    GameId.PROCGEN_NINJA: PROCGEN_NINJA_HTML,
    GameId.PROCGEN_PLUNDER: PROCGEN_PLUNDER_HTML,
    GameId.PROCGEN_STARPILOT: PROCGEN_STARPILOT_HTML,
})

# BabyAI mappings (language-grounded instruction following)
GAME_INFO.update({
    # GoTo family
    GameId.BABYAI_GOTO_REDBALL_GREY: BABYAI_GOTO_REDBALL_GREY_HTML,
    GameId.BABYAI_GOTO_REDBALL: BABYAI_GOTO_REDBALL_HTML,
    GameId.BABYAI_GOTO_REDBALL_NODISTS: BABYAI_GOTO_REDBALL_NODISTS_HTML,
    GameId.BABYAI_GOTO_OBJ: BABYAI_GOTO_OBJ_HTML,
    GameId.BABYAI_GOTO_LOCAL: BABYAI_GOTO_LOCAL_HTML,
    GameId.BABYAI_GOTO: BABYAI_GOTO_HTML,
    GameId.BABYAI_GOTO_IMPUNLOCK: BABYAI_GOTO_IMPUNLOCK_HTML,
    GameId.BABYAI_GOTO_SEQ: BABYAI_GOTO_SEQ_HTML,
    GameId.BABYAI_GOTO_REDBLUEBALL: BABYAI_GOTO_REDBLUEBALL_HTML,
    GameId.BABYAI_GOTO_DOOR: BABYAI_GOTO_DOOR_HTML,
    GameId.BABYAI_GOTO_OBJDOOR: BABYAI_GOTO_OBJDOOR_HTML,
    # Open family
    GameId.BABYAI_OPEN: BABYAI_OPEN_HTML,
    GameId.BABYAI_OPEN_REDDOOR: BABYAI_OPEN_REDDOOR_HTML,
    GameId.BABYAI_OPEN_DOOR: BABYAI_OPEN_DOOR_HTML,
    GameId.BABYAI_OPEN_TWODOORS: BABYAI_OPEN_TWODOORS_HTML,
    GameId.BABYAI_OPEN_DOORSORDER_N2: BABYAI_OPEN_DOORSORDER_HTML,
    GameId.BABYAI_OPEN_DOORSORDER_N4: BABYAI_OPEN_DOORSORDER_HTML,
    # Pickup family
    GameId.BABYAI_PICKUP: BABYAI_PICKUP_HTML,
    GameId.BABYAI_UNBLOCK_PICKUP: BABYAI_UNBLOCK_PICKUP_HTML,
    GameId.BABYAI_PICKUP_LOC: BABYAI_PICKUP_LOC_HTML,
    GameId.BABYAI_PICKUP_DIST: BABYAI_PICKUP_DIST_HTML,
    GameId.BABYAI_PICKUP_ABOVE: BABYAI_PICKUP_ABOVE_HTML,
    # Unlock family
    GameId.BABYAI_UNLOCK: BABYAI_UNLOCK_HTML,
    GameId.BABYAI_UNLOCK_LOCAL: BABYAI_UNLOCK_LOCAL_HTML,
    GameId.BABYAI_KEY_INBOX: BABYAI_KEY_INBOX_HTML,
    GameId.BABYAI_UNLOCK_PICKUP: BABYAI_UNLOCK_PICKUP_HTML,
    GameId.BABYAI_BLOCKED_UNLOCK_PICKUP: BABYAI_BLOCKED_UNLOCK_PICKUP_HTML,
    GameId.BABYAI_UNLOCK_TO_UNLOCK: BABYAI_UNLOCK_TO_UNLOCK_HTML,
    # PutNext family
    GameId.BABYAI_PUTNEXT_LOCAL: BABYAI_PUTNEXT_LOCAL_HTML,
    GameId.BABYAI_PUTNEXT: BABYAI_PUTNEXT_HTML,
    # Complex environments
    GameId.BABYAI_ACTION_OBJDOOR: BABYAI_ACTION_OBJDOOR_HTML,
    GameId.BABYAI_FINDOBJ_S5: BABYAI_FINDOBJ_HTML,
    GameId.BABYAI_KEYCORRIDOR_S3R1: BABYAI_KEYCORRIDOR_HTML,
    GameId.BABYAI_KEYCORRIDOR_S3R2: BABYAI_KEYCORRIDOR_HTML,
    GameId.BABYAI_KEYCORRIDOR_S3R3: BABYAI_KEYCORRIDOR_HTML,
    GameId.BABYAI_ONEROOM_S8: BABYAI_ONEROOM_HTML,
    GameId.BABYAI_MOVETWOACROSS_S8N9: BABYAI_MOVETWOACROSS_HTML,
    GameId.BABYAI_SYNTH: BABYAI_SYNTH_HTML,
    GameId.BABYAI_SYNTHLOC: BABYAI_SYNTHLOC_HTML,
    GameId.BABYAI_SYNTHSEQ: BABYAI_SYNTHSEQ_HTML,
    GameId.BABYAI_MINIBOSSLEVEL: BABYAI_MINIBOSSLEVEL_HTML,
    GameId.BABYAI_BOSSLEVEL: BABYAI_BOSSLEVEL_HTML,
    GameId.BABYAI_BOSSLEVEL_NOUNLOCK: BABYAI_BOSSLEVEL_NOUNLOCK_HTML,
})

# Jumanji mappings (JAX-based environments)
try:
    from gym_gui.game_docs.Jumanji import (
        # Phase 1: Logic
        GAME2048_HTML,
        MINESWEEPER_HTML,
        RUBIKS_CUBE_HTML,
        SLIDING_PUZZLE_HTML,
        SUDOKU_HTML,
        GRAPH_COLORING_HTML,
        # Phase 2: Packing
        BINPACK_HTML,
        FLATPACK_HTML,
        JOBSHOP_HTML,
        KNAPSACK_HTML,
        TETRIS_HTML,
        # Phase 3: Routing
        CLEANER_HTML,
        CONNECTOR_HTML,
        CVRP_HTML,
        MAZE_HTML,
        MMST_HTML,
        MULTI_CVRP_HTML,
        PACMAN_HTML,
        ROBOT_WAREHOUSE_HTML,
        SNAKE_HTML,
        SOKOBAN_HTML,
        TSP_HTML,
    )
    GAME_INFO.update({
        # Phase 1: Logic
        GameId.JUMANJI_GAME2048: GAME2048_HTML,
        GameId.JUMANJI_MINESWEEPER: MINESWEEPER_HTML,
        GameId.JUMANJI_RUBIKS_CUBE: RUBIKS_CUBE_HTML,
        GameId.JUMANJI_SLIDING_PUZZLE: SLIDING_PUZZLE_HTML,
        GameId.JUMANJI_SUDOKU: SUDOKU_HTML,
        GameId.JUMANJI_GRAPH_COLORING: GRAPH_COLORING_HTML,
        # Phase 2: Packing
        GameId.JUMANJI_BINPACK: BINPACK_HTML,
        GameId.JUMANJI_FLATPACK: FLATPACK_HTML,
        GameId.JUMANJI_JOBSHOP: JOBSHOP_HTML,
        GameId.JUMANJI_KNAPSACK: KNAPSACK_HTML,
        GameId.JUMANJI_TETRIS: TETRIS_HTML,
        # Phase 3: Routing
        GameId.JUMANJI_CLEANER: CLEANER_HTML,
        GameId.JUMANJI_CONNECTOR: CONNECTOR_HTML,
        GameId.JUMANJI_CVRP: CVRP_HTML,
        GameId.JUMANJI_MAZE: MAZE_HTML,
        GameId.JUMANJI_MMST: MMST_HTML,
        GameId.JUMANJI_MULTI_CVRP: MULTI_CVRP_HTML,
        GameId.JUMANJI_PACMAN: PACMAN_HTML,
        GameId.JUMANJI_ROBOT_WAREHOUSE: ROBOT_WAREHOUSE_HTML,
        GameId.JUMANJI_SNAKE: SNAKE_HTML,
        GameId.JUMANJI_SOKOBAN: SOKOBAN_HTML,
        GameId.JUMANJI_TSP: TSP_HTML,
    })
except ImportError:
    pass  # Jumanji docs optional


from gym_gui.game_docs.mosaic_welcome import MULTI_KEYBOARD_HTML


def _get_meltingpot_doc(env_id: str) -> str:
    """Return HTML documentation for a MeltingPot substrate.

    Extracts the base substrate name from env_id (e.g.
    ``"meltingpot/collaborative_cooking__ring"`` -> ``"collaborative_cooking"``)
    and maps it to the corresponding documentation module.
    """
    from gym_gui.game_docs.MeltingPot import (
        MELTINGPOT_COLLABORATIVE_COOKING_HTML,
        MELTINGPOT_CLEAN_UP_HTML,
        MELTINGPOT_COMMONS_HARVEST_HTML,
        MELTINGPOT_TERRITORY_HTML,
        MELTINGPOT_KING_OF_THE_HILL_HTML,
        MELTINGPOT_PRISONERS_DILEMMA_HTML,
        MELTINGPOT_STAG_HUNT_HTML,
        MELTINGPOT_ALLELOPATHIC_HARVEST_HTML,
    )

    _DOC_MAP = {
        "allelopathic_harvest": MELTINGPOT_ALLELOPATHIC_HARVEST_HTML,
        "clean_up": MELTINGPOT_CLEAN_UP_HTML,
        "collaborative_cooking": MELTINGPOT_COLLABORATIVE_COOKING_HTML,
        "commons_harvest": MELTINGPOT_COMMONS_HARVEST_HTML,
        "paintball": MELTINGPOT_KING_OF_THE_HILL_HTML,
        "prisoners_dilemma_in_the_matrix": MELTINGPOT_PRISONERS_DILEMMA_HTML,
        "stag_hunt_in_the_matrix": MELTINGPOT_STAG_HUNT_HTML,
        "territory": MELTINGPOT_TERRITORY_HTML,
    }

    # "meltingpot/collaborative_cooking__ring" -> "collaborative_cooking"
    substrate = env_id.split("/", 1)[-1]
    base = substrate.split("__")[0]

    return _DOC_MAP.get(base, _DEFAULT_DOC)


def _get_overcooked_doc(env_id: str) -> str:
    """Return HTML documentation for an Overcooked layout."""
    from gym_gui.game_docs.Overcooked import (
        OVERCOOKED_CRAMPED_ROOM_HTML,
        OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML,
        OVERCOOKED_COORDINATION_RING_HTML,
        OVERCOOKED_FORCED_COORDINATION_HTML,
        OVERCOOKED_COUNTER_CIRCUIT_HTML,
    )

    _DOC_MAP = {
        "cramped_room": OVERCOOKED_CRAMPED_ROOM_HTML,
        "asymmetric_advantages": OVERCOOKED_ASYMMETRIC_ADVANTAGES_HTML,
        "coordination_ring": OVERCOOKED_COORDINATION_RING_HTML,
        "forced_coordination": OVERCOOKED_FORCED_COORDINATION_HTML,
        "counter_circuit": OVERCOOKED_COUNTER_CIRCUIT_HTML,
    }

    # "overcooked/cramped_room" -> "cramped_room"
    layout = env_id.split("/", 1)[-1]

    return _DOC_MAP.get(layout, _DEFAULT_DOC)


def get_game_info(game_id: GameId) -> str:
    """Return HTML documentation for the specified environment."""

    # Handle Empty variants dynamically
    if game_id in (GameId.MINIGRID_EMPTY_5x5, GameId.MINIGRID_EMPTY_RANDOM_5x5,
                   GameId.MINIGRID_EMPTY_6x6, GameId.MINIGRID_EMPTY_RANDOM_6x6,
                   GameId.MINIGRID_EMPTY_8x8, GameId.MINIGRID_EMPTY_16x16):
        return get_empty_html(game_id.value)
    
    # Handle DoorKey variants dynamically
    if game_id in (GameId.MINIGRID_DOORKEY_5x5, GameId.MINIGRID_DOORKEY_6x6, 
                   GameId.MINIGRID_DOORKEY_8x8, GameId.MINIGRID_DOORKEY_16x16):
        return get_doorkey_html(game_id.value)
    
    # Handle LavaGap variants dynamically
    if game_id in (GameId.MINIGRID_LAVAGAP_S5, GameId.MINIGRID_LAVAGAP_S6, 
                   GameId.MINIGRID_LAVAGAP_S7):
        return get_lavagap_html(game_id.value)
    
    # Handle Dynamic Obstacles variants dynamically
    if game_id in (GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5, GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5,
                   GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6, GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6,
                   GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8, GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16):
        return get_dynamic_obstacles_html(game_id.value)
    
    # Handle MultiRoom variants dynamically
    if game_id in (GameId.MINIGRID_MULTIROOM_N2_S4, GameId.MINIGRID_MULTIROOM_N4_S5,
                   GameId.MINIGRID_MULTIROOM_N6):
        return get_multiroom_html(game_id.value)
    
    # Handle Obstructed Maze variants dynamically
    if game_id in (GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB, GameId.MINIGRID_OBSTRUCTED_MAZE_FULL):
        return get_obstructed_maze_html(game_id.value)
    
    # Handle Crossing variants dynamically
    if game_id in (GameId.MINIGRID_LAVA_CROSSING_S9N1, GameId.MINIGRID_LAVA_CROSSING_S9N2,
                   GameId.MINIGRID_LAVA_CROSSING_S9N3, GameId.MINIGRID_LAVA_CROSSING_S11N5,
                   GameId.MINIGRID_SIMPLE_CROSSING_S9N1, GameId.MINIGRID_SIMPLE_CROSSING_S9N2,
                   GameId.MINIGRID_SIMPLE_CROSSING_S9N3, GameId.MINIGRID_SIMPLE_CROSSING_S11N5):
        return get_crossing_html(game_id.value)
    
    # Handle RedBlueDoors variants dynamically
    if game_id in (GameId.MINIGRID_REDBLUE_DOORS_6x6, GameId.MINIGRID_REDBLUE_DOORS_8x8):
        return get_redbluedoors_html(game_id.value)

    # Handle INI MultiGrid variants dynamically
    _ini_multigrid_variants = (
        GameId.INI_MULTIGRID_EMPTY_5X5,
        GameId.INI_MULTIGRID_EMPTY_RANDOM_5X5,
        GameId.INI_MULTIGRID_EMPTY_6X6,
        GameId.INI_MULTIGRID_EMPTY_RANDOM_6X6,
        GameId.INI_MULTIGRID_EMPTY_8X8,
        GameId.INI_MULTIGRID_EMPTY_16X16,
        GameId.INI_MULTIGRID_RED_BLUE_DOORS_6X6,
        GameId.INI_MULTIGRID_RED_BLUE_DOORS_8X8,
        GameId.INI_MULTIGRID_LOCKED_HALLWAY_2ROOMS,
        GameId.INI_MULTIGRID_LOCKED_HALLWAY_4ROOMS,
        GameId.INI_MULTIGRID_LOCKED_HALLWAY_6ROOMS,
        GameId.INI_MULTIGRID_BLOCKED_UNLOCK_PICKUP,
        GameId.INI_MULTIGRID_PLAYGROUND,
    )
    if game_id in _ini_multigrid_variants:
        ini_doc = get_ini_multigrid_html(game_id.value)
        return ini_doc + "\n\n" + MULTI_KEYBOARD_HTML

    # Handle MOSAIC MultiGrid variants dynamically
    _mosaic_multigrid_variants = (
        # Deprecated originals
        GameId.MOSAIC_MULTIGRID_SOCCER,
        GameId.MOSAIC_MULTIGRID_COLLECT,
        GameId.MOSAIC_MULTIGRID_COLLECT2VS2,
        GameId.MOSAIC_MULTIGRID_COLLECT_1VS1,
        # IndAgObs variants
        GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS,
        GameId.MOSAIC_MULTIGRID_SOCCER_1VS1_INDAGOBS,
        GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS,
        GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS,
        GameId.MOSAIC_MULTIGRID_COLLECT_1VS1_INDAGOBS,
        GameId.MOSAIC_MULTIGRID_BASKETBALL_INDAGOBS,
        # TeamObs variants
        GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS,
        GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS,
        GameId.MOSAIC_MULTIGRID_BASKETBALL_TEAMOBS,
    )
    if game_id in _mosaic_multigrid_variants:
        mosaic_doc = get_mosaic_multigrid_html(game_id.value)
        return mosaic_doc + "\n\n" + MULTI_KEYBOARD_HTML

    # Handle MeltingPot variants by substrate name
    from gym_gui.core.enums import ENVIRONMENT_FAMILY_BY_GAME, EnvironmentFamily
    family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)
    if family == EnvironmentFamily.MELTINGPOT:
        doc = _get_meltingpot_doc(game_id.value)
        return doc + "\n\n" + MULTI_KEYBOARD_HTML

    # Handle Overcooked variants by layout name
    if family == EnvironmentFamily.OVERCOOKED:
        doc = _get_overcooked_doc(game_id.value)
        return doc + "\n\n" + MULTI_KEYBOARD_HTML

    # Get base documentation from flat dict
    base_doc = GAME_INFO.get(game_id, _DEFAULT_DOC)
    return base_doc


__all__ = ["GAME_INFO", "get_game_info"]
