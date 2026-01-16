from __future__ import annotations

"""Enumerations that describe the Gym GUI domain model."""

from enum import Enum, auto
from typing import Iterable


class StrEnum(str, Enum):
    """Minimal stand-in for :class:`enum.StrEnum` (Python 3.11+)."""

    def __new__(cls, value: str) -> "StrEnum":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj


class EnvironmentFamily(StrEnum):
    """High-level groupings that organise Gymnasium environments."""

    TOY_TEXT = "toy_text"
    BOX2D = "box2d"
    CLASSIC_CONTROL = "classic_control"
    ATARI = "atari"
    ALE = "ale"
    MUJOCO = "mujoco"
    MINIGRID = "minigrid"
    BABYAI = "babyai"  # BabyAI language-grounded instruction following (built on MiniGrid)
    VIZDOOM = "vizdoom"
    MINIHACK = "minihack"  # MiniHack sandbox environments (built on NLE)
    NETHACK = "nethack"  # Full NetHack game (via NLE)
    CRAFTER = "crafter"  # Crafter open world survival benchmark
    PROCGEN = "procgen"  # Procgen procedural benchmark (16 environments)
    JUMANJI = "jumanji"  # Jumanji JAX-based logic puzzle environments
    TEXTWORLD = "textworld"  # TextWorld text-based game environment (Microsoft Research)
    BABAISAI = "babaisai"  # BabaIsAI rule manipulation puzzle benchmark (ICML 2024)
    PYBULLET_DRONES = "pybullet_drones"  # PyBullet Drones quadcopter environments (University of Toronto)
    PETTINGZOO = "pettingzoo"
    PETTINGZOO_CLASSIC = "pettingzoo_classic"  # PettingZoo Classic: turn-based games (Chess, Go, Connect Four, etc.)
    OPEN_SPIEL = "open_spiel"  # OpenSpiel + custom draughts variants (American, Russian, International)
    MULTIGRID = "multigrid"  # gym-multigrid multi-agent grid environments (Soccer, Collect)
    MELTINGPOT = "meltingpot"  # Melting Pot multi-agent social scenarios (Google DeepMind) via Shimmy
    OVERCOOKED = "overcooked"  # Overcooked-AI cooperative cooking (2 agents, human-AI coordination)
    OTHER = "other"  # Fallback for unknown environments (not displayed in UI)


class GameId(StrEnum):
    """Canonical Gymnasium environment identifiers supported by the GUI."""

    FROZEN_LAKE = "FrozenLake-v1"
    FROZEN_LAKE_V2 = "FrozenLake-v2"
    CLIFF_WALKING = "CliffWalking-v0"  # Gymnasium 1.0.0 only has v0
    TAXI = "Taxi-v3"
    BLACKJACK = "Blackjack-v1"
    LUNAR_LANDER = "LunarLander-v3"
    CAR_RACING = "CarRacing-v3"
    BIPEDAL_WALKER = "BipedalWalker-v3"
    CART_POLE = "CartPole-v1"
    ACROBOT = "Acrobot-v1"
    MOUNTAIN_CAR = "MountainCar-v0"
    PONG_NO_FRAMESKIP = "PongNoFrameskip-v4"
    BREAKOUT_NO_FRAMESKIP = "BreakoutNoFrameskip-v4"
    ADVENTURE_V4 = "Adventure-v4"
    ALE_ADVENTURE_V5 = "ALE/Adventure-v5"
    AIR_RAID_V4 = "AirRaid-v4"
    ALE_AIR_RAID_V5 = "ALE/AirRaid-v5"
    ASSAULT_V4 = "Assault-v4"
    ALE_ASSAULT_V5 = "ALE/Assault-v5"
    # ─────────────────────────────────────────────────────────────────────────
    # ALE (Arcade Learning Environment) - Complete 128 Game Library
    # Using ALE/{GameName}-v5 format (recommended for new projects)
    # ─────────────────────────────────────────────────────────────────────────
    ALE_ALIEN_V5 = "ALE/Alien-v5"
    ALE_AMIDAR_V5 = "ALE/Amidar-v5"
    ALE_ASTERIX_V5 = "ALE/Asterix-v5"
    ALE_ASTEROIDS_V5 = "ALE/Asteroids-v5"
    ALE_ATLANTIS_V5 = "ALE/Atlantis-v5"
    ALE_ATLANTIS2_V5 = "ALE/Atlantis2-v5"
    ALE_BACKGAMMON_V5 = "ALE/Backgammon-v5"
    ALE_BANK_HEIST_V5 = "ALE/BankHeist-v5"
    ALE_BASIC_MATH_V5 = "ALE/BasicMath-v5"
    ALE_BATTLE_ZONE_V5 = "ALE/BattleZone-v5"
    ALE_BEAM_RIDER_V5 = "ALE/BeamRider-v5"
    ALE_BERZERK_V5 = "ALE/Berzerk-v5"
    ALE_BLACKJACK_V5 = "ALE/Blackjack-v5"
    ALE_BOWLING_V5 = "ALE/Bowling-v5"
    ALE_BOXING_V5 = "ALE/Boxing-v5"
    ALE_BREAKOUT_V5 = "ALE/Breakout-v5"
    ALE_CARNIVAL_V5 = "ALE/Carnival-v5"
    ALE_CASINO_V5 = "ALE/Casino-v5"
    ALE_CENTIPEDE_V5 = "ALE/Centipede-v5"
    ALE_CHOPPER_COMMAND_V5 = "ALE/ChopperCommand-v5"
    ALE_CRAZY_CLIMBER_V5 = "ALE/CrazyClimber-v5"
    ALE_CROSSBOW_V5 = "ALE/Crossbow-v5"
    ALE_DARKCHAMBERS_V5 = "ALE/Darkchambers-v5"
    ALE_DEFENDER_V5 = "ALE/Defender-v5"
    ALE_DEMON_ATTACK_V5 = "ALE/DemonAttack-v5"
    ALE_DONKEY_KONG_V5 = "ALE/DonkeyKong-v5"
    ALE_DOUBLE_DUNK_V5 = "ALE/DoubleDunk-v5"
    ALE_EARTHWORLD_V5 = "ALE/Earthworld-v5"
    ALE_ELEVATOR_ACTION_V5 = "ALE/ElevatorAction-v5"
    ALE_ENDURO_V5 = "ALE/Enduro-v5"
    ALE_ENTOMBED_V5 = "ALE/Entombed-v5"
    ALE_ET_V5 = "ALE/Et-v5"
    ALE_FISHING_DERBY_V5 = "ALE/FishingDerby-v5"
    ALE_FLAG_CAPTURE_V5 = "ALE/FlagCapture-v5"
    ALE_FREEWAY_V5 = "ALE/Freeway-v5"
    ALE_FROGGER_V5 = "ALE/Frogger-v5"
    ALE_FROSTBITE_V5 = "ALE/Frostbite-v5"
    ALE_GALAXIAN_V5 = "ALE/Galaxian-v5"
    ALE_GOPHER_V5 = "ALE/Gopher-v5"
    ALE_GRAVITAR_V5 = "ALE/Gravitar-v5"
    ALE_HANGMAN_V5 = "ALE/Hangman-v5"
    ALE_HAUNTED_HOUSE_V5 = "ALE/HauntedHouse-v5"
    ALE_HERO_V5 = "ALE/Hero-v5"
    ALE_HUMAN_CANNONBALL_V5 = "ALE/HumanCannonball-v5"
    ALE_ICE_HOCKEY_V5 = "ALE/IceHockey-v5"
    ALE_JAMESBOND_V5 = "ALE/Jamesbond-v5"
    ALE_JOURNEY_ESCAPE_V5 = "ALE/JourneyEscape-v5"
    ALE_KABOOM_V5 = "ALE/Kaboom-v5"
    ALE_KANGAROO_V5 = "ALE/Kangaroo-v5"
    ALE_KEYSTONE_KAPERS_V5 = "ALE/KeystoneKapers-v5"
    ALE_KING_KONG_V5 = "ALE/KingKong-v5"
    ALE_KLAX_V5 = "ALE/Klax-v5"
    ALE_KOOLAID_V5 = "ALE/Koolaid-v5"
    ALE_KRULL_V5 = "ALE/Krull-v5"
    ALE_KUNG_FU_MASTER_V5 = "ALE/KungFuMaster-v5"
    ALE_LASER_GATES_V5 = "ALE/LaserGates-v5"
    ALE_LOST_LUGGAGE_V5 = "ALE/LostLuggage-v5"
    ALE_MARIO_BROS_V5 = "ALE/MarioBros-v5"
    ALE_MINIATURE_GOLF_V5 = "ALE/MiniatureGolf-v5"
    ALE_MONTEZUMA_REVENGE_V5 = "ALE/MontezumaRevenge-v5"
    ALE_MR_DO_V5 = "ALE/MrDo-v5"
    ALE_MS_PACMAN_V5 = "ALE/MsPacman-v5"
    ALE_NAME_THIS_GAME_V5 = "ALE/NameThisGame-v5"
    ALE_OTHELLO_V5 = "ALE/Othello-v5"
    ALE_PACMAN_V5 = "ALE/Pacman-v5"
    ALE_PHOENIX_V5 = "ALE/Phoenix-v5"
    ALE_PITFALL_V5 = "ALE/Pitfall-v5"
    ALE_PITFALL2_V5 = "ALE/Pitfall2-v5"
    ALE_PONG_V5 = "ALE/Pong-v5"
    ALE_POOYAN_V5 = "ALE/Pooyan-v5"
    ALE_PRIVATE_EYE_V5 = "ALE/PrivateEye-v5"
    ALE_QBERT_V5 = "ALE/Qbert-v5"
    ALE_RIVERRAID_V5 = "ALE/Riverraid-v5"
    ALE_ROAD_RUNNER_V5 = "ALE/RoadRunner-v5"
    ALE_ROBOTANK_V5 = "ALE/Robotank-v5"
    ALE_SEAQUEST_V5 = "ALE/Seaquest-v5"
    ALE_SIR_LANCELOT_V5 = "ALE/SirLancelot-v5"
    ALE_SKIING_V5 = "ALE/Skiing-v5"
    ALE_SOLARIS_V5 = "ALE/Solaris-v5"
    ALE_SPACE_INVADERS_V5 = "ALE/SpaceInvaders-v5"
    ALE_SPACE_WAR_V5 = "ALE/SpaceWar-v5"
    ALE_STAR_GUNNER_V5 = "ALE/StarGunner-v5"
    ALE_SUPERMAN_V5 = "ALE/Superman-v5"
    ALE_SURROUND_V5 = "ALE/Surround-v5"
    ALE_TENNIS_V5 = "ALE/Tennis-v5"
    ALE_TETRIS_V5 = "ALE/Tetris-v5"
    ALE_TIC_TAC_TOE_3D_V5 = "ALE/TicTacToe3D-v5"
    ALE_TIME_PILOT_V5 = "ALE/TimePilot-v5"
    ALE_TRONDEAD_V5 = "ALE/Trondead-v5"
    ALE_TURMOIL_V5 = "ALE/Turmoil-v5"
    ALE_TUTANKHAM_V5 = "ALE/Tutankham-v5"
    ALE_UP_N_DOWN_V5 = "ALE/UpNDown-v5"
    ALE_VENTURE_V5 = "ALE/Venture-v5"
    ALE_VIDEO_CHECKERS_V5 = "ALE/VideoCheckers-v5"
    ALE_VIDEO_CHESS_V5 = "ALE/VideoChess-v5"
    ALE_VIDEO_CUBE_V5 = "ALE/VideoCube-v5"
    ALE_VIDEO_PINBALL_V5 = "ALE/VideoPinball-v5"
    ALE_WIZARD_OF_WOR_V5 = "ALE/WizardOfWor-v5"
    ALE_WORD_ZAPPER_V5 = "ALE/WordZapper-v5"
    ALE_YARS_REVENGE_V5 = "ALE/YarsRevenge-v5"
    ALE_ZAXXON_V5 = "ALE/Zaxxon-v5"
    ANT = "Ant-v5"
    HALF_CHEETAH = "HalfCheetah-v5"
    HOPPER = "Hopper-v5"
    HUMANOID = "Humanoid-v5"
    HUMANOID_STANDUP = "HumanoidStandup-v5"
    INVERTED_DOUBLE_PENDULUM = "InvertedDoublePendulum-v5"
    INVERTED_PENDULUM = "InvertedPendulum-v5"
    PUSHER = "Pusher-v5"
    REACHER = "Reacher-v5"
    SWIMMER = "Swimmer-v5"
    WALKER2D = "Walker2d-v5"
    MINIGRID_EMPTY_5x5 = "MiniGrid-Empty-5x5-v0"
    MINIGRID_EMPTY_RANDOM_5x5 = "MiniGrid-Empty-Random-5x5-v0"
    MINIGRID_EMPTY_6x6 = "MiniGrid-Empty-6x6-v0"
    MINIGRID_EMPTY_RANDOM_6x6 = "MiniGrid-Empty-Random-6x6-v0"
    MINIGRID_EMPTY_8x8 = "MiniGrid-Empty-8x8-v0"
    MINIGRID_EMPTY_16x16 = "MiniGrid-Empty-16x16-v0"
    MINIGRID_DOORKEY_5x5 = "MiniGrid-DoorKey-5x5-v0"
    MINIGRID_DOORKEY_6x6 = "MiniGrid-DoorKey-6x6-v0"
    MINIGRID_DOORKEY_8x8 = "MiniGrid-DoorKey-8x8-v0"
    MINIGRID_DOORKEY_16x16 = "MiniGrid-DoorKey-16x16-v0"
    MINIGRID_LAVAGAP_S5 = "MiniGrid-LavaGapS5-v0"
    MINIGRID_LAVAGAP_S6 = "MiniGrid-LavaGapS6-v0"
    MINIGRID_LAVAGAP_S7 = "MiniGrid-LavaGapS7-v0"
    MINIGRID_DYNAMIC_OBSTACLES_5X5 = "MiniGrid-Dynamic-Obstacles-5x5-v0"
    MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5 = "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"
    MINIGRID_DYNAMIC_OBSTACLES_6X6 = "MiniGrid-Dynamic-Obstacles-6x6-v0"
    MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6 = "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
    MINIGRID_DYNAMIC_OBSTACLES_8X8 = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    MINIGRID_DYNAMIC_OBSTACLES_16X16 = "MiniGrid-Dynamic-Obstacles-16x16-v0"
    MINIGRID_BLOCKED_UNLOCK_PICKUP = "MiniGrid-BlockedUnlockPickup-v0"
    MINIGRID_MULTIROOM_N2_S4 = "MiniGrid-MultiRoom-N2-S4-v0"
    MINIGRID_MULTIROOM_N4_S5 = "MiniGrid-MultiRoom-N4-S5-v0"
    MINIGRID_MULTIROOM_N6 = "MiniGrid-MultiRoom-N6-v0"
    MINIGRID_OBSTRUCTED_MAZE_1DLHB = "MiniGrid-ObstructedMaze-1Dlhb-v1"
    MINIGRID_OBSTRUCTED_MAZE_FULL = "MiniGrid-ObstructedMaze-Full-v1"
    MINIGRID_LAVA_CROSSING_S9N1 = "MiniGrid-LavaCrossingS9N1-v0"
    MINIGRID_LAVA_CROSSING_S9N2 = "MiniGrid-LavaCrossingS9N2-v0"
    MINIGRID_LAVA_CROSSING_S9N3 = "MiniGrid-LavaCrossingS9N3-v0"
    MINIGRID_LAVA_CROSSING_S11N5 = "MiniGrid-LavaCrossingS11N5-v0"
    MINIGRID_SIMPLE_CROSSING_S9N1 = "MiniGrid-SimpleCrossingS9N1-v0"
    MINIGRID_SIMPLE_CROSSING_S9N2 = "MiniGrid-SimpleCrossingS9N2-v0"
    MINIGRID_SIMPLE_CROSSING_S9N3 = "MiniGrid-SimpleCrossingS9N3-v0"
    MINIGRID_SIMPLE_CROSSING_S11N5 = "MiniGrid-SimpleCrossingS11N5-v0"
    MINIGRID_REDBLUE_DOORS_6x6 = "MiniGrid-RedBlueDoors-6x6-v0"
    MINIGRID_REDBLUE_DOORS_8x8 = "MiniGrid-RedBlueDoors-8x8-v0"
    # ─────────────────────────────────────────────────────────────────────────
    # BabyAI Environments (language-grounded instruction following on MiniGrid)
    # ─────────────────────────────────────────────────────────────────────────
    # GoTo family
    BABYAI_GOTO_REDBALL_GREY = "BabyAI-GoToRedBallGrey-v0"
    BABYAI_GOTO_REDBALL = "BabyAI-GoToRedBall-v0"
    BABYAI_GOTO_REDBALL_NODISTS = "BabyAI-GoToRedBallNoDists-v0"
    BABYAI_GOTO_OBJ = "BabyAI-GoToObj-v0"
    BABYAI_GOTO_LOCAL = "BabyAI-GoToLocal-v0"
    BABYAI_GOTO = "BabyAI-GoTo-v0"
    BABYAI_GOTO_IMPUNLOCK = "BabyAI-GoToImpUnlock-v0"
    BABYAI_GOTO_SEQ = "BabyAI-GoToSeq-v0"
    BABYAI_GOTO_REDBLUEBALL = "BabyAI-GoToRedBlueBall-v0"
    BABYAI_GOTO_DOOR = "BabyAI-GoToDoor-v0"
    BABYAI_GOTO_OBJDOOR = "BabyAI-GoToObjDoor-v0"
    # Open family
    BABYAI_OPEN = "BabyAI-Open-v0"
    BABYAI_OPEN_REDDOOR = "BabyAI-OpenRedDoor-v0"
    BABYAI_OPEN_DOOR = "BabyAI-OpenDoor-v0"
    BABYAI_OPEN_TWODOORS = "BabyAI-OpenTwoDoors-v0"
    BABYAI_OPEN_DOORSORDER_N2 = "BabyAI-OpenDoorsOrderN2-v0"
    BABYAI_OPEN_DOORSORDER_N4 = "BabyAI-OpenDoorsOrderN4-v0"
    # Pickup family
    BABYAI_PICKUP = "BabyAI-Pickup-v0"
    BABYAI_UNBLOCK_PICKUP = "BabyAI-UnblockPickup-v0"
    BABYAI_PICKUP_LOC = "BabyAI-PickupLoc-v0"
    BABYAI_PICKUP_DIST = "BabyAI-PickupDist-v0"
    BABYAI_PICKUP_ABOVE = "BabyAI-PickupAbove-v0"
    # Unlock family
    BABYAI_UNLOCK = "BabyAI-Unlock-v0"
    BABYAI_UNLOCK_LOCAL = "BabyAI-UnlockLocal-v0"
    BABYAI_KEY_INBOX = "BabyAI-KeyInBox-v0"
    BABYAI_UNLOCK_PICKUP = "BabyAI-UnlockPickup-v0"
    BABYAI_BLOCKED_UNLOCK_PICKUP = "BabyAI-BlockedUnlockPickup-v0"
    BABYAI_UNLOCK_TO_UNLOCK = "BabyAI-UnlockToUnlock-v0"
    # PutNext family
    BABYAI_PUTNEXT_LOCAL = "BabyAI-PutNextLocal-v0"
    BABYAI_PUTNEXT = "BabyAI-PutNext-v0"
    # Complex environments
    BABYAI_ACTION_OBJDOOR = "BabyAI-ActionObjDoor-v0"
    BABYAI_FINDOBJ_S5 = "BabyAI-FindObjS5-v0"
    BABYAI_KEYCORRIDOR_S3R1 = "BabyAI-KeyCorridorS3R1-v0"
    BABYAI_KEYCORRIDOR_S3R2 = "BabyAI-KeyCorridorS3R2-v0"
    BABYAI_KEYCORRIDOR_S3R3 = "BabyAI-KeyCorridorS3R3-v0"
    BABYAI_ONEROOM_S8 = "BabyAI-OneRoomS8-v0"
    BABYAI_MOVETWOACROSS_S8N9 = "BabyAI-MoveTwoAcrossS8N9-v0"
    BABYAI_SYNTH = "BabyAI-Synth-v0"
    BABYAI_SYNTHLOC = "BabyAI-SynthLoc-v0"
    BABYAI_SYNTHSEQ = "BabyAI-SynthSeq-v0"
    BABYAI_MINIBOSSLEVEL = "BabyAI-MiniBossLevel-v0"
    BABYAI_BOSSLEVEL = "BabyAI-BossLevel-v0"
    BABYAI_BOSSLEVEL_NOUNLOCK = "BabyAI-BossLevelNoUnlock-v0"
    VIZDOOM_BASIC = "ViZDoom-Basic-v0"
    VIZDOOM_DEADLY_CORRIDOR = "ViZDoom-DeadlyCorridor-v0"
    VIZDOOM_DEFEND_THE_CENTER = "ViZDoom-DefendTheCenter-v0"
    VIZDOOM_DEFEND_THE_LINE = "ViZDoom-DefendTheLine-v0"
    VIZDOOM_HEALTH_GATHERING = "ViZDoom-HealthGathering-v0"
    VIZDOOM_HEALTH_GATHERING_SUPREME = "ViZDoom-HealthGatheringSupreme-v0"
    VIZDOOM_MY_WAY_HOME = "ViZDoom-MyWayHome-v0"
    VIZDOOM_PREDICT_POSITION = "ViZDoom-PredictPosition-v0"
    VIZDOOM_TAKE_COVER = "ViZDoom-TakeCover-v0"
    VIZDOOM_DEATHMATCH = "ViZDoom-Deathmatch-v0"
    # PettingZoo Classic Board Games
    CHESS = "chess_v6"
    CONNECT_FOUR = "connect_four_v3"
    GO = "go_v5"
    TIC_TAC_TOE = "tictactoe_v3"

    # ─────────────────────────────────────────────────────────────────────────
    # MiniHack Environments (sandbox RL environments built on NLE)
    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    MINIHACK_ROOM_5X5 = "MiniHack-Room-5x5-v0"
    MINIHACK_ROOM_15X15 = "MiniHack-Room-15x15-v0"
    MINIHACK_CORRIDOR_R2 = "MiniHack-Corridor-R2-v0"
    MINIHACK_CORRIDOR_R3 = "MiniHack-Corridor-R3-v0"
    MINIHACK_CORRIDOR_R5 = "MiniHack-Corridor-R5-v0"
    MINIHACK_MAZEWALK_9X9 = "MiniHack-MazeWalk-9x9-v0"
    MINIHACK_MAZEWALK_15X15 = "MiniHack-MazeWalk-15x15-v0"
    MINIHACK_MAZEWALK_45X19 = "MiniHack-MazeWalk-45x19-v0"
    MINIHACK_RIVER = "MiniHack-River-v0"
    MINIHACK_RIVER_NARROW = "MiniHack-River-Narrow-v0"
    # Skills
    MINIHACK_EAT = "MiniHack-Eat-v0"
    MINIHACK_WEAR = "MiniHack-Wear-v0"
    MINIHACK_WIELD = "MiniHack-Wield-v0"
    MINIHACK_ZAP = "MiniHack-Zap-v0"
    MINIHACK_READ = "MiniHack-Read-v0"
    MINIHACK_QUAFF = "MiniHack-Quaff-v0"
    MINIHACK_PUTON = "MiniHack-PutOn-v0"
    MINIHACK_LAVACROSS = "MiniHack-LavaCross-v0"
    MINIHACK_WOD_EASY = "MiniHack-WoD-Easy-v0"
    MINIHACK_WOD_MEDIUM = "MiniHack-WoD-Medium-v0"
    MINIHACK_WOD_HARD = "MiniHack-WoD-Hard-v0"
    # Exploration
    MINIHACK_EXPLOREMAZE_EASY = "MiniHack-ExploreMaze-Easy-v0"
    MINIHACK_EXPLOREMAZE_HARD = "MiniHack-ExploreMaze-Hard-v0"
    MINIHACK_HIDENSEEK = "MiniHack-HideNSeek-v0"
    MINIHACK_MEMENTO_F2 = "MiniHack-Memento-F2-v0"
    MINIHACK_MEMENTO_F4 = "MiniHack-Memento-F4-v0"

    # ─────────────────────────────────────────────────────────────────────────
    # NetHack (Full Game via NLE)
    # ─────────────────────────────────────────────────────────────────────────
    NETHACK_FULL = "NetHackChallenge-v0"
    NETHACK_SCORE = "NetHackScore-v0"
    NETHACK_STAIRCASE = "NetHackStaircase-v0"
    NETHACK_STAIRCASE_PET = "NetHackStaircasePet-v0"
    NETHACK_ORACLE = "NetHackOracle-v0"
    NETHACK_GOLD = "NetHackGold-v0"
    NETHACK_EAT = "NetHackEat-v0"
    NETHACK_SCOUT = "NetHackScout-v0"

    # ─────────────────────────────────────────────────────────────────────────
    # Crafter (Open World Survival Benchmark)
    # ─────────────────────────────────────────────────────────────────────────
    CRAFTER_REWARD = "CrafterReward-v1"
    CRAFTER_NO_REWARD = "CrafterNoReward-v1"

    # ─────────────────────────────────────────────────────────────────────────
    # TextWorld (Text-Based Game Environment - Microsoft Research)
    # ─────────────────────────────────────────────────────────────────────────
    TEXTWORLD_SIMPLE = "TextWorld-Simple-v0"
    TEXTWORLD_COIN_COLLECTOR = "TextWorld-CoinCollector-v0"
    TEXTWORLD_TREASURE_HUNTER = "TextWorld-TreasureHunter-v0"
    TEXTWORLD_COOKING = "TextWorld-Cooking-v0"
    TEXTWORLD_CUSTOM = "TextWorld-Custom-v0"

    # ─────────────────────────────────────────────────────────────────────────
    # BabaIsAI (Rule Manipulation Puzzle Benchmark - ICML 2024)
    # ─────────────────────────────────────────────────────────────────────────
    BABAISAI_DEFAULT = "BabaIsAI-Default-v0"

    # ─────────────────────────────────────────────────────────────────────────
    # Procgen (Procedurally Generated Benchmark - 16 environments)
    # ─────────────────────────────────────────────────────────────────────────
    PROCGEN_BIGFISH = "procgen:procgen-bigfish-v0"
    PROCGEN_BOSSFIGHT = "procgen:procgen-bossfight-v0"
    PROCGEN_CAVEFLYER = "procgen:procgen-caveflyer-v0"
    PROCGEN_CHASER = "procgen:procgen-chaser-v0"
    PROCGEN_CLIMBER = "procgen:procgen-climber-v0"
    PROCGEN_COINRUN = "procgen:procgen-coinrun-v0"
    PROCGEN_DODGEBALL = "procgen:procgen-dodgeball-v0"
    PROCGEN_FRUITBOT = "procgen:procgen-fruitbot-v0"
    PROCGEN_HEIST = "procgen:procgen-heist-v0"
    PROCGEN_JUMPER = "procgen:procgen-jumper-v0"
    PROCGEN_LEAPER = "procgen:procgen-leaper-v0"
    PROCGEN_MAZE = "procgen:procgen-maze-v0"
    PROCGEN_MINER = "procgen:procgen-miner-v0"
    PROCGEN_NINJA = "procgen:procgen-ninja-v0"
    PROCGEN_PLUNDER = "procgen:procgen-plunder-v0"
    PROCGEN_STARPILOT = "procgen:procgen-starpilot-v0"
    # Jumanji (JAX-based Logic Puzzle Environments)
    JUMANJI_GAME2048 = "jumanji/Game2048-v1"
    JUMANJI_MINESWEEPER = "jumanji/Minesweeper-v0"
    JUMANJI_RUBIKS_CUBE = "jumanji/RubiksCube-v0"
    JUMANJI_SLIDING_PUZZLE = "jumanji/SlidingTilePuzzle-v0"
    JUMANJI_SUDOKU = "jumanji/Sudoku-v0"
    JUMANJI_GRAPH_COLORING = "jumanji/GraphColoring-v1"
    # Jumanji Phase 2: Packing Environments
    JUMANJI_BINPACK = "jumanji/BinPack-v2"
    JUMANJI_FLATPACK = "jumanji/FlatPack-v0"
    JUMANJI_JOBSHOP = "jumanji/JobShop-v0"
    JUMANJI_KNAPSACK = "jumanji/Knapsack-v1"
    JUMANJI_TETRIS = "jumanji/Tetris-v0"
    # Jumanji Phase 3: Routing Environments
    JUMANJI_CLEANER = "jumanji/Cleaner-v0"
    JUMANJI_CONNECTOR = "jumanji/Connector-v2"
    JUMANJI_CVRP = "jumanji/CVRP-v1"
    JUMANJI_MAZE = "jumanji/Maze-v0"
    JUMANJI_MMST = "jumanji/MMST-v0"
    JUMANJI_MULTI_CVRP = "jumanji/MultiCVRP-v0"
    JUMANJI_PACMAN = "jumanji/PacMan-v1"
    JUMANJI_ROBOT_WAREHOUSE = "jumanji/RobotWarehouse-v0"
    JUMANJI_SNAKE = "jumanji/Snake-v1"
    JUMANJI_SOKOBAN = "jumanji/Sokoban-v0"
    JUMANJI_TSP = "jumanji/TSP-v1"
    # ─────────────────────────────────────────────────────────────────────────
    # PyBullet Drones - Quadcopter Control Environments (University of Toronto)
    # Paper: Panerati et al. (2021) "Learning to Fly"
    # Repository: https://github.com/utiasDSL/gym-pybullet-drones
    # ─────────────────────────────────────────────────────────────────────────
    PYBULLET_HOVER_AVIARY = "hover-aviary-v0"
    PYBULLET_MULTIHOVER_AVIARY = "multihover-aviary-v0"
    PYBULLET_CTRL_AVIARY = "ctrl-aviary-v0"
    PYBULLET_VELOCITY_AVIARY = "velocity-aviary-v0"
    # ─────────────────────────────────────────────────────────────────────────
    # OpenSpiel - Board Games via Shimmy PettingZoo Wrapper
    # Repository: https://github.com/google-deepmind/open_spiel
    # Shimmy: https://shimmy.farama.org/environments/open_spiel/
    # ─────────────────────────────────────────────────────────────────────────
    OPEN_SPIEL_CHECKERS = "open_spiel/checkers"

    # ─────────────────────────────────────────────────────────────────────────
    # Draughts/Checkers Variants (Custom implementations with proper rules)
    # ─────────────────────────────────────────────────────────────────────────
    # American Checkers (8x8) - No backward captures, no flying kings
    AMERICAN_CHECKERS = "draughts/american_checkers"
    # Russian Checkers (8x8) - Men can capture backward, flying kings
    RUSSIAN_CHECKERS = "draughts/russian_checkers"
    # International Draughts (10x10) - Men can capture backward, flying kings, 20 pieces
    INTERNATIONAL_DRAUGHTS = "draughts/international_draughts"

    # ─────────────────────────────────────────────────────────────────────────
    # gym-multigrid (Multi-Agent Grid Environments)
    # Repository: https://github.com/ArnaudFickinger/gym-multigrid
    # Location: 3rd_party/gym-multigrid/
    # ─────────────────────────────────────────────────────────────────────────
    MULTIGRID_SOCCER = "MultiGrid-Soccer-v0"  # 4 agents, 2v2 soccer
    MULTIGRID_COLLECT = "MultiGrid-Collect-v0"  # 3 agents, collect balls

    # ─────────────────────────────────────────────────────────────────────────
    # Melting Pot (Multi-Agent Social Scenarios - Google DeepMind)
    # Repository: https://github.com/google-deepmind/meltingpot
    # Shimmy: https://shimmy.farama.org/environments/meltingpot/
    # NOTE: Linux/macOS only (Windows NOT supported)
    # ─────────────────────────────────────────────────────────────────────────
    # Cooperative Scenarios
    MELTINGPOT_COLLABORATIVE_COOKING = "meltingpot/collaborative_cooking__circuit"  # Up to 9 agents, cooking cooperation
    MELTINGPOT_CLEAN_UP = "meltingpot/clean_up__repeated"  # Up to 7 agents, public goods game
    MELTINGPOT_COMMONS_HARVEST = "meltingpot/commons_harvest__open"  # Up to 16 agents, tragedy of the commons

    # Competitive Scenarios
    MELTINGPOT_TERRITORY = "meltingpot/territory__rooms"  # Up to 8 agents, territory control
    MELTINGPOT_KING_OF_THE_HILL = "meltingpot/king_of_the_hill__repeated"  # Up to 16 agents, area control

    # Mixed-Motive (Cooperation + Competition)
    MELTINGPOT_PRISONERS_DILEMMA = "meltingpot/prisoners_dilemma_in_the_matrix__repeated"  # 2 agents, game theory classic
    MELTINGPOT_STAG_HUNT = "meltingpot/stag_hunt_in_the_matrix__repeated"  # 2 agents, coordination dilemma
    MELTINGPOT_ALLELOPATHIC_HARVEST = "meltingpot/allelopathic_harvest__open"  # Up to 16 agents, resource competition

    # ─────────────────────────────────────────────────────────────────────────
    # Overcooked-AI (Cooperative Cooking - Human-AI Coordination)
    # Repository: https://github.com/HumanCompatibleAI/overcooked_ai
    # Paper: https://arxiv.org/abs/1910.05789 (NeurIPS 2019)
    # Location: 3rd_party/overcooked_ai/
    # ─────────────────────────────────────────────────────────────────────────
    OVERCOOKED_CRAMPED_ROOM = "overcooked/cramped_room"  # 2 agents, tight kitchen coordination
    OVERCOOKED_ASYMMETRIC_ADVANTAGES = "overcooked/asymmetric_advantages"  # 2 agents, asymmetric resource access
    OVERCOOKED_COORDINATION_RING = "overcooked/coordination_ring"  # 2 agents, circular kitchen layout
    OVERCOOKED_FORCED_COORDINATION = "overcooked/forced_coordination"  # 2 agents, explicit coordination required
    OVERCOOKED_COUNTER_CIRCUIT = "overcooked/counter_circuit"  # 2 agents, circuit-style counter layout


def get_game_display_name(game_id: GameId) -> str:
    """Get the display name for a GameId with family prefix.
    
    Converts actual environment IDs to user-friendly display names:
    - 'FrozenLake-v1' → 'Gym-ToyText-FrozenLake-v1'
    - 'LunarLander-v3' → 'Gym-Box2D-LunarLander-v3'
    - 'MiniGrid-Empty-5x5-v0' → 'MiniGrid-Empty-5x5-v0' (no prefix, separate library)
    """
    value = game_id.value
    
    # MiniGrid is a separate library, not part of Gym - keep as-is
    if value.startswith("MiniGrid-"):
        return value
    # ALE is a separate namespace (Atari via ALE) - keep as-is
    if value.startswith("ALE/"):
        return value
    # ViZDoom games already include descriptive prefix
    if value.startswith("ViZDoom-"):
        return value
    # MiniHack environments already include descriptive prefix
    if value.startswith("MiniHack-"):
        return value
    # NetHack environments already include descriptive prefix
    if value.startswith("NetHack"):
        return value
    # Crafter environments
    if value.startswith("Crafter"):
        return value
    # Procgen environments (procgen:procgen-name-v0 → Procgen-Name)
    if value.startswith("procgen:"):
        # Extract game name: "procgen:procgen-coinrun-v0" → "coinrun"
        env_part = value.split(":")[1]  # "procgen-coinrun-v0"
        name_part = env_part.replace("procgen-", "").replace("-v0", "")  # "coinrun"
        return f"Procgen-{name_part.title()}"
    # PettingZoo board games
    if game_id == GameId.CHESS:
        return "PettingZoo-Chess"
    if game_id == GameId.CONNECT_FOUR:
        return "PettingZoo-ConnectFour"
    if game_id == GameId.GO:
        return "PettingZoo-Go"
    if game_id == GameId.TIC_TAC_TOE:
        return "PettingZoo-TicTacToe"
    # OpenSpiel board games (via Shimmy)
    if value.startswith("open_spiel/"):
        game_name = value.split("/")[1].replace("_", " ").title()
        return f"OpenSpiel-{game_name}"
    # MultiGrid environments (multi-agent grid worlds)
    if value.startswith("MultiGrid-"):
        return value

    # Determine Gym family based on enum
    if game_id in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2, GameId.CLIFF_WALKING, 
                   GameId.TAXI, GameId.BLACKJACK):
        return f"Gym-ToyText-{value}"
    elif game_id in (GameId.LUNAR_LANDER, GameId.CAR_RACING, GameId.BIPEDAL_WALKER):
        return f"Gym-Box2D-{value}"
    elif game_id in (
        GameId.CART_POLE,
        GameId.ACROBOT,
        GameId.MOUNTAIN_CAR,
    ):
        return f"Gym-ClassicControl-{value}"
    elif game_id in (
        GameId.PONG_NO_FRAMESKIP,
        GameId.BREAKOUT_NO_FRAMESKIP,
    ):
        return f"Gym-Atari-{value}"
    elif game_id in (
        GameId.ADVENTURE_V4,
        GameId.AIR_RAID_V4,
        GameId.ASSAULT_V4,
    ):
        # Legacy non-namespaced Atari environments
        return f"Atari-{value}"
    elif game_id in (
        GameId.ANT,
        GameId.HALF_CHEETAH,
        GameId.HOPPER,
        GameId.HUMANOID,
        GameId.HUMANOID_STANDUP,
        GameId.INVERTED_DOUBLE_PENDULUM,
        GameId.INVERTED_PENDULUM,
        GameId.PUSHER,
        GameId.REACHER,
        GameId.SWIMMER,
        GameId.WALKER2D,
    ):
        return f"Gym-MuJoCo-{value}"
    else:
        return f"Gym-{value}"


class ControlMode(StrEnum):
    """Who is currently in control of the environment."""

    HUMAN_ONLY = "human_only"
    AGENT_ONLY = "agent_only"
    HYBRID_TURN_BASED = "hybrid_turn_based"
    HYBRID_HUMAN_AGENT = "hybrid_human_agent"
    MULTI_AGENT_COOP = "multi_agent_coop"
    MULTI_AGENT_COMPETITIVE = "multi_agent_competitive"


class InputMode(StrEnum):
    """Input mode for keyboard controls.

    Controls how keyboard input is processed for human gameplay.
    Users can choose between these modes in the Game Configuration panel.
    """

    STATE_BASED = "state_based"
    """Tracks all currently pressed keys and computes combined actions.

    - Enables simultaneous key presses (e.g., Up+Right for diagonal movement)
    - Best for real-time arcade games (Procgen, Atari, ViZDoom, etc.)
    - Keys are sampled on each game tick
    - Supports WASD and arrow keys interchangeably
    """

    SHORTCUT_BASED = "shortcut_based"
    """Uses Qt's QShortcut mechanism for single-key actions.

    - Each key press immediately triggers an action
    - Best for turn-based or step-by-step games
    - Traditional input mode with immediate response
    - No simultaneous key combination support
    """


# Human-readable labels and descriptions for InputMode
INPUT_MODE_INFO: dict[InputMode, tuple[str, str]] = {
    InputMode.STATE_BASED: (
        "State-Based (Real-time)",
        "Enables simultaneous key combinations (e.g., Up+Right for diagonal). "
        "Best for arcade games where you hold multiple keys at once.",
    ),
    InputMode.SHORTCUT_BASED: (
        "Shortcut-Based (Immediate)",
        "Each key press triggers an immediate action. "
        "Best for turn-based games or step-by-step control.",
    ),
}


class RenderMode(StrEnum):
    """Rendering strategies supported by the UI."""

    ANSI = "ansi"
    ASCII = "ascii"
    GRID = "grid"
    RGB_ARRAY = "rgb_array"
    SURFACE = "surface"


class ActionSpaceType(StrEnum):
    """Simplified view over Gymnasium action space types."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_BINARY = "multi_binary"
    MULTI_DISCRETE = "multi_discrete"


class AgentRole(StrEnum):
    """Roles an agent can play when multiple controllers are present."""

    PRIMARY = "primary"
    ASSIST = "assist"
    SPECTATOR = "spectator"


class AdapterCapability(Enum):
    """Flags that describe optional behaviours for adapters."""

    RECORD_SUPPORT = auto()
    FAST_RESET = auto()
    MULTI_AGENT = auto()


class SteppingParadigm(StrEnum):
    """Defines how RL agents interact with the environment.

    This enum describes the stepping model - how actions are collected and applied.
    It is orthogonal to ControlMode (who controls) and EnvironmentFamily (library).

    NOTE: This enum is for RL training paradigms ONLY.
    Non-RL systems like MuJoCo MPC (optimal control) are managed separately
    by gym_gui/services/mujoco_mpc_controller/.

    See Also:
        - docs/1.0_DAY_41/TASK_1/01_paradigm_comparison.md for POSG vs AEC details
        - docs/1.0_DAY_41/TASK_1/00_multi_paradigm_orchestrator_plan.md for architecture
    """

    SINGLE_AGENT = "single_agent"
    """Gymnasium single-agent: one agent, one action per step.

    Used by: CleanRL, stable-baselines3, MuJoCo envs (HalfCheetah, etc.)
    API: obs, reward, done, info = env.step(action)
    """

    SEQUENTIAL = "sequential"
    """PettingZoo AEC / OpenSpiel: agents take turns, one at a time.

    Used by: PettingZoo env(), Chess, Go, turn-based games
    API: for agent in env.agent_iter(): env.step(action)
    """

    SIMULTANEOUS = "simultaneous"
    """RLlib / PettingZoo Parallel: all agents act at once.

    Used by: RLlib MultiAgentEnv, PettingZoo parallel_env(), MPE
    API: obs_dict, rewards, dones, infos = env.step(action_dict)
    """

    HIERARCHICAL = "hierarchical"
    """BDI + RL: high-level goals decomposed into low-level RL actions.

    Used by: Jason/BDI workers, goal-driven agents
    API: Custom goal → action mapping with RL sub-policies
    """


ENVIRONMENT_FAMILY_BY_GAME: dict[GameId, EnvironmentFamily] = {
    GameId.FROZEN_LAKE: EnvironmentFamily.TOY_TEXT,
    GameId.FROZEN_LAKE_V2: EnvironmentFamily.TOY_TEXT,
    GameId.CLIFF_WALKING: EnvironmentFamily.TOY_TEXT,
    GameId.TAXI: EnvironmentFamily.TOY_TEXT,
    GameId.BLACKJACK: EnvironmentFamily.TOY_TEXT,
    GameId.LUNAR_LANDER: EnvironmentFamily.BOX2D,
    GameId.CAR_RACING: EnvironmentFamily.BOX2D,
    GameId.BIPEDAL_WALKER: EnvironmentFamily.BOX2D,
    GameId.CART_POLE: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.ACROBOT: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.MOUNTAIN_CAR: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.PONG_NO_FRAMESKIP: EnvironmentFamily.ATARI,
    GameId.BREAKOUT_NO_FRAMESKIP: EnvironmentFamily.ATARI,
    GameId.ADVENTURE_V4: EnvironmentFamily.ALE,
    GameId.ALE_ADVENTURE_V5: EnvironmentFamily.ALE,
    GameId.AIR_RAID_V4: EnvironmentFamily.ALE,
    GameId.ALE_AIR_RAID_V5: EnvironmentFamily.ALE,
    GameId.ASSAULT_V4: EnvironmentFamily.ALE,
    GameId.ALE_ASSAULT_V5: EnvironmentFamily.ALE,
    # ALE Complete Game Library
    GameId.ALE_ALIEN_V5: EnvironmentFamily.ALE,
    GameId.ALE_AMIDAR_V5: EnvironmentFamily.ALE,
    GameId.ALE_ASTERIX_V5: EnvironmentFamily.ALE,
    GameId.ALE_ASTEROIDS_V5: EnvironmentFamily.ALE,
    GameId.ALE_ATLANTIS_V5: EnvironmentFamily.ALE,
    GameId.ALE_ATLANTIS2_V5: EnvironmentFamily.ALE,
    GameId.ALE_BACKGAMMON_V5: EnvironmentFamily.ALE,
    GameId.ALE_BANK_HEIST_V5: EnvironmentFamily.ALE,
    GameId.ALE_BASIC_MATH_V5: EnvironmentFamily.ALE,
    GameId.ALE_BATTLE_ZONE_V5: EnvironmentFamily.ALE,
    GameId.ALE_BEAM_RIDER_V5: EnvironmentFamily.ALE,
    GameId.ALE_BERZERK_V5: EnvironmentFamily.ALE,
    GameId.ALE_BLACKJACK_V5: EnvironmentFamily.ALE,
    GameId.ALE_BOWLING_V5: EnvironmentFamily.ALE,
    GameId.ALE_BOXING_V5: EnvironmentFamily.ALE,
    GameId.ALE_BREAKOUT_V5: EnvironmentFamily.ALE,
    GameId.ALE_CARNIVAL_V5: EnvironmentFamily.ALE,
    GameId.ALE_CASINO_V5: EnvironmentFamily.ALE,
    GameId.ALE_CENTIPEDE_V5: EnvironmentFamily.ALE,
    GameId.ALE_CHOPPER_COMMAND_V5: EnvironmentFamily.ALE,
    GameId.ALE_CRAZY_CLIMBER_V5: EnvironmentFamily.ALE,
    GameId.ALE_CROSSBOW_V5: EnvironmentFamily.ALE,
    GameId.ALE_DARKCHAMBERS_V5: EnvironmentFamily.ALE,
    GameId.ALE_DEFENDER_V5: EnvironmentFamily.ALE,
    GameId.ALE_DEMON_ATTACK_V5: EnvironmentFamily.ALE,
    GameId.ALE_DONKEY_KONG_V5: EnvironmentFamily.ALE,
    GameId.ALE_DOUBLE_DUNK_V5: EnvironmentFamily.ALE,
    GameId.ALE_EARTHWORLD_V5: EnvironmentFamily.ALE,
    GameId.ALE_ELEVATOR_ACTION_V5: EnvironmentFamily.ALE,
    GameId.ALE_ENDURO_V5: EnvironmentFamily.ALE,
    GameId.ALE_ENTOMBED_V5: EnvironmentFamily.ALE,
    GameId.ALE_ET_V5: EnvironmentFamily.ALE,
    GameId.ALE_FISHING_DERBY_V5: EnvironmentFamily.ALE,
    GameId.ALE_FLAG_CAPTURE_V5: EnvironmentFamily.ALE,
    GameId.ALE_FREEWAY_V5: EnvironmentFamily.ALE,
    GameId.ALE_FROGGER_V5: EnvironmentFamily.ALE,
    GameId.ALE_FROSTBITE_V5: EnvironmentFamily.ALE,
    GameId.ALE_GALAXIAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_GOPHER_V5: EnvironmentFamily.ALE,
    GameId.ALE_GRAVITAR_V5: EnvironmentFamily.ALE,
    GameId.ALE_HANGMAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_HAUNTED_HOUSE_V5: EnvironmentFamily.ALE,
    GameId.ALE_HERO_V5: EnvironmentFamily.ALE,
    GameId.ALE_HUMAN_CANNONBALL_V5: EnvironmentFamily.ALE,
    GameId.ALE_ICE_HOCKEY_V5: EnvironmentFamily.ALE,
    GameId.ALE_JAMESBOND_V5: EnvironmentFamily.ALE,
    GameId.ALE_JOURNEY_ESCAPE_V5: EnvironmentFamily.ALE,
    GameId.ALE_KABOOM_V5: EnvironmentFamily.ALE,
    GameId.ALE_KANGAROO_V5: EnvironmentFamily.ALE,
    GameId.ALE_KEYSTONE_KAPERS_V5: EnvironmentFamily.ALE,
    GameId.ALE_KING_KONG_V5: EnvironmentFamily.ALE,
    GameId.ALE_KLAX_V5: EnvironmentFamily.ALE,
    GameId.ALE_KOOLAID_V5: EnvironmentFamily.ALE,
    GameId.ALE_KRULL_V5: EnvironmentFamily.ALE,
    GameId.ALE_KUNG_FU_MASTER_V5: EnvironmentFamily.ALE,
    GameId.ALE_LASER_GATES_V5: EnvironmentFamily.ALE,
    GameId.ALE_LOST_LUGGAGE_V5: EnvironmentFamily.ALE,
    GameId.ALE_MARIO_BROS_V5: EnvironmentFamily.ALE,
    GameId.ALE_MINIATURE_GOLF_V5: EnvironmentFamily.ALE,
    GameId.ALE_MONTEZUMA_REVENGE_V5: EnvironmentFamily.ALE,
    GameId.ALE_MR_DO_V5: EnvironmentFamily.ALE,
    GameId.ALE_MS_PACMAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_NAME_THIS_GAME_V5: EnvironmentFamily.ALE,
    GameId.ALE_OTHELLO_V5: EnvironmentFamily.ALE,
    GameId.ALE_PACMAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_PHOENIX_V5: EnvironmentFamily.ALE,
    GameId.ALE_PITFALL_V5: EnvironmentFamily.ALE,
    GameId.ALE_PITFALL2_V5: EnvironmentFamily.ALE,
    GameId.ALE_PONG_V5: EnvironmentFamily.ALE,
    GameId.ALE_POOYAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_PRIVATE_EYE_V5: EnvironmentFamily.ALE,
    GameId.ALE_QBERT_V5: EnvironmentFamily.ALE,
    GameId.ALE_RIVERRAID_V5: EnvironmentFamily.ALE,
    GameId.ALE_ROAD_RUNNER_V5: EnvironmentFamily.ALE,
    GameId.ALE_ROBOTANK_V5: EnvironmentFamily.ALE,
    GameId.ALE_SEAQUEST_V5: EnvironmentFamily.ALE,
    GameId.ALE_SIR_LANCELOT_V5: EnvironmentFamily.ALE,
    GameId.ALE_SKIING_V5: EnvironmentFamily.ALE,
    GameId.ALE_SOLARIS_V5: EnvironmentFamily.ALE,
    GameId.ALE_SPACE_INVADERS_V5: EnvironmentFamily.ALE,
    GameId.ALE_SPACE_WAR_V5: EnvironmentFamily.ALE,
    GameId.ALE_STAR_GUNNER_V5: EnvironmentFamily.ALE,
    GameId.ALE_SUPERMAN_V5: EnvironmentFamily.ALE,
    GameId.ALE_SURROUND_V5: EnvironmentFamily.ALE,
    GameId.ALE_TENNIS_V5: EnvironmentFamily.ALE,
    GameId.ALE_TETRIS_V5: EnvironmentFamily.ALE,
    GameId.ALE_TIC_TAC_TOE_3D_V5: EnvironmentFamily.ALE,
    GameId.ALE_TIME_PILOT_V5: EnvironmentFamily.ALE,
    GameId.ALE_TRONDEAD_V5: EnvironmentFamily.ALE,
    GameId.ALE_TURMOIL_V5: EnvironmentFamily.ALE,
    GameId.ALE_TUTANKHAM_V5: EnvironmentFamily.ALE,
    GameId.ALE_UP_N_DOWN_V5: EnvironmentFamily.ALE,
    GameId.ALE_VENTURE_V5: EnvironmentFamily.ALE,
    GameId.ALE_VIDEO_CHECKERS_V5: EnvironmentFamily.ALE,
    GameId.ALE_VIDEO_CHESS_V5: EnvironmentFamily.ALE,
    GameId.ALE_VIDEO_CUBE_V5: EnvironmentFamily.ALE,
    GameId.ALE_VIDEO_PINBALL_V5: EnvironmentFamily.ALE,
    GameId.ALE_WIZARD_OF_WOR_V5: EnvironmentFamily.ALE,
    GameId.ALE_WORD_ZAPPER_V5: EnvironmentFamily.ALE,
    GameId.ALE_YARS_REVENGE_V5: EnvironmentFamily.ALE,
    GameId.ALE_ZAXXON_V5: EnvironmentFamily.ALE,
    GameId.ANT: EnvironmentFamily.MUJOCO,
    GameId.HALF_CHEETAH: EnvironmentFamily.MUJOCO,
    GameId.HOPPER: EnvironmentFamily.MUJOCO,
    GameId.HUMANOID: EnvironmentFamily.MUJOCO,
    GameId.HUMANOID_STANDUP: EnvironmentFamily.MUJOCO,
    GameId.INVERTED_DOUBLE_PENDULUM: EnvironmentFamily.MUJOCO,
    GameId.INVERTED_PENDULUM: EnvironmentFamily.MUJOCO,
    GameId.PUSHER: EnvironmentFamily.MUJOCO,
    GameId.REACHER: EnvironmentFamily.MUJOCO,
    GameId.SWIMMER: EnvironmentFamily.MUJOCO,
    GameId.WALKER2D: EnvironmentFamily.MUJOCO,
    GameId.MINIGRID_EMPTY_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_8x8: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_16x16: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_8x8: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_16x16: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S7: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S9N1: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S9N2: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S9N3: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S11N5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_REDBLUE_DOORS_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_REDBLUE_DOORS_8x8: EnvironmentFamily.MINIGRID,
    # BabyAI environments
    GameId.BABYAI_GOTO_REDBALL_GREY: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_REDBALL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_REDBALL_NODISTS: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_OBJ: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_LOCAL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_IMPUNLOCK: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_SEQ: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_REDBLUEBALL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_DOOR: EnvironmentFamily.BABYAI,
    GameId.BABYAI_GOTO_OBJDOOR: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN_REDDOOR: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN_DOOR: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN_TWODOORS: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN_DOORSORDER_N2: EnvironmentFamily.BABYAI,
    GameId.BABYAI_OPEN_DOORSORDER_N4: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PICKUP: EnvironmentFamily.BABYAI,
    GameId.BABYAI_UNBLOCK_PICKUP: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PICKUP_LOC: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PICKUP_DIST: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PICKUP_ABOVE: EnvironmentFamily.BABYAI,
    GameId.BABYAI_UNLOCK: EnvironmentFamily.BABYAI,
    GameId.BABYAI_UNLOCK_LOCAL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_KEY_INBOX: EnvironmentFamily.BABYAI,
    GameId.BABYAI_UNLOCK_PICKUP: EnvironmentFamily.BABYAI,
    GameId.BABYAI_BLOCKED_UNLOCK_PICKUP: EnvironmentFamily.BABYAI,
    GameId.BABYAI_UNLOCK_TO_UNLOCK: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PUTNEXT_LOCAL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_PUTNEXT: EnvironmentFamily.BABYAI,
    GameId.BABYAI_ACTION_OBJDOOR: EnvironmentFamily.BABYAI,
    GameId.BABYAI_FINDOBJ_S5: EnvironmentFamily.BABYAI,
    GameId.BABYAI_KEYCORRIDOR_S3R1: EnvironmentFamily.BABYAI,
    GameId.BABYAI_KEYCORRIDOR_S3R2: EnvironmentFamily.BABYAI,
    GameId.BABYAI_KEYCORRIDOR_S3R3: EnvironmentFamily.BABYAI,
    GameId.BABYAI_ONEROOM_S8: EnvironmentFamily.BABYAI,
    GameId.BABYAI_MOVETWOACROSS_S8N9: EnvironmentFamily.BABYAI,
    GameId.BABYAI_SYNTH: EnvironmentFamily.BABYAI,
    GameId.BABYAI_SYNTHLOC: EnvironmentFamily.BABYAI,
    GameId.BABYAI_SYNTHSEQ: EnvironmentFamily.BABYAI,
    GameId.BABYAI_MINIBOSSLEVEL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_BOSSLEVEL: EnvironmentFamily.BABYAI,
    GameId.BABYAI_BOSSLEVEL_NOUNLOCK: EnvironmentFamily.BABYAI,
    GameId.VIZDOOM_BASIC: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_DEADLY_CORRIDOR: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_DEFEND_THE_CENTER: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_DEFEND_THE_LINE: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_HEALTH_GATHERING: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_MY_WAY_HOME: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_PREDICT_POSITION: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_TAKE_COVER: EnvironmentFamily.VIZDOOM,
    GameId.VIZDOOM_DEATHMATCH: EnvironmentFamily.VIZDOOM,
    GameId.CHESS: EnvironmentFamily.PETTINGZOO_CLASSIC,
    GameId.CONNECT_FOUR: EnvironmentFamily.PETTINGZOO_CLASSIC,
    GameId.GO: EnvironmentFamily.PETTINGZOO_CLASSIC,
    GameId.TIC_TAC_TOE: EnvironmentFamily.PETTINGZOO_CLASSIC,
    # MiniHack environments
    GameId.MINIHACK_ROOM_5X5: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_ROOM_15X15: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_CORRIDOR_R2: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_CORRIDOR_R3: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_CORRIDOR_R5: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_MAZEWALK_9X9: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_MAZEWALK_15X15: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_MAZEWALK_45X19: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_RIVER: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_RIVER_NARROW: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_EAT: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_WEAR: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_WIELD: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_ZAP: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_READ: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_QUAFF: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_PUTON: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_LAVACROSS: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_WOD_EASY: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_WOD_MEDIUM: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_WOD_HARD: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_EXPLOREMAZE_EASY: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_EXPLOREMAZE_HARD: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_HIDENSEEK: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_MEMENTO_F2: EnvironmentFamily.MINIHACK,
    GameId.MINIHACK_MEMENTO_F4: EnvironmentFamily.MINIHACK,
    # NetHack (full game) environments
    GameId.NETHACK_FULL: EnvironmentFamily.NETHACK,
    GameId.NETHACK_SCORE: EnvironmentFamily.NETHACK,
    GameId.NETHACK_STAIRCASE: EnvironmentFamily.NETHACK,
    GameId.NETHACK_STAIRCASE_PET: EnvironmentFamily.NETHACK,
    GameId.NETHACK_ORACLE: EnvironmentFamily.NETHACK,
    GameId.NETHACK_GOLD: EnvironmentFamily.NETHACK,
    GameId.NETHACK_EAT: EnvironmentFamily.NETHACK,
    GameId.NETHACK_SCOUT: EnvironmentFamily.NETHACK,
    # Crafter environments
    GameId.CRAFTER_REWARD: EnvironmentFamily.CRAFTER,
    GameId.CRAFTER_NO_REWARD: EnvironmentFamily.CRAFTER,
    # TextWorld environments (text-based games)
    GameId.TEXTWORLD_SIMPLE: EnvironmentFamily.TEXTWORLD,
    GameId.TEXTWORLD_COIN_COLLECTOR: EnvironmentFamily.TEXTWORLD,
    GameId.TEXTWORLD_TREASURE_HUNTER: EnvironmentFamily.TEXTWORLD,
    GameId.TEXTWORLD_COOKING: EnvironmentFamily.TEXTWORLD,
    GameId.TEXTWORLD_CUSTOM: EnvironmentFamily.TEXTWORLD,
    # BabaIsAI environments (rule manipulation puzzles)
    GameId.BABAISAI_DEFAULT: EnvironmentFamily.BABAISAI,
    # Procgen environments (16 procedurally generated games)
    GameId.PROCGEN_BIGFISH: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_BOSSFIGHT: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_CAVEFLYER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_CHASER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_CLIMBER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_COINRUN: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_DODGEBALL: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_FRUITBOT: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_HEIST: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_JUMPER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_LEAPER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_MAZE: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_MINER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_NINJA: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_PLUNDER: EnvironmentFamily.PROCGEN,
    GameId.PROCGEN_STARPILOT: EnvironmentFamily.PROCGEN,
    # Jumanji Logic Puzzle Environments
    GameId.JUMANJI_GAME2048: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_MINESWEEPER: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_RUBIKS_CUBE: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_SLIDING_PUZZLE: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_SUDOKU: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_GRAPH_COLORING: EnvironmentFamily.JUMANJI,
    # Jumanji Phase 2: Packing
    GameId.JUMANJI_BINPACK: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_FLATPACK: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_JOBSHOP: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_KNAPSACK: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_TETRIS: EnvironmentFamily.JUMANJI,
    # Jumanji Phase 3: Routing
    GameId.JUMANJI_CLEANER: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_CONNECTOR: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_CVRP: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_MAZE: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_MMST: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_MULTI_CVRP: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_PACMAN: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_ROBOT_WAREHOUSE: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_SNAKE: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_SOKOBAN: EnvironmentFamily.JUMANJI,
    GameId.JUMANJI_TSP: EnvironmentFamily.JUMANJI,
    # PyBullet Drones
    GameId.PYBULLET_HOVER_AVIARY: EnvironmentFamily.PYBULLET_DRONES,
    GameId.PYBULLET_MULTIHOVER_AVIARY: EnvironmentFamily.PYBULLET_DRONES,
    GameId.PYBULLET_CTRL_AVIARY: EnvironmentFamily.PYBULLET_DRONES,
    GameId.PYBULLET_VELOCITY_AVIARY: EnvironmentFamily.PYBULLET_DRONES,
    # OpenSpiel + custom draughts variants (all under open_spiel family)
    GameId.OPEN_SPIEL_CHECKERS: EnvironmentFamily.OPEN_SPIEL,
    GameId.AMERICAN_CHECKERS: EnvironmentFamily.OPEN_SPIEL,  # Custom implementation
    GameId.RUSSIAN_CHECKERS: EnvironmentFamily.OPEN_SPIEL,  # Custom implementation
    GameId.INTERNATIONAL_DRAUGHTS: EnvironmentFamily.OPEN_SPIEL,  # Custom implementation
    # gym-multigrid (multi-agent grid environments)
    GameId.MULTIGRID_SOCCER: EnvironmentFamily.MULTIGRID,
    GameId.MULTIGRID_COLLECT: EnvironmentFamily.MULTIGRID,
    # Melting Pot (multi-agent social scenarios)
    GameId.MELTINGPOT_COLLABORATIVE_COOKING: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_CLEAN_UP: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_COMMONS_HARVEST: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_TERRITORY: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_KING_OF_THE_HILL: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_PRISONERS_DILEMMA: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_STAG_HUNT: EnvironmentFamily.MELTINGPOT,
    GameId.MELTINGPOT_ALLELOPATHIC_HARVEST: EnvironmentFamily.MELTINGPOT,
    # Overcooked-AI (cooperative cooking)
    GameId.OVERCOOKED_CRAMPED_ROOM: EnvironmentFamily.OVERCOOKED,
    GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES: EnvironmentFamily.OVERCOOKED,
    GameId.OVERCOOKED_COORDINATION_RING: EnvironmentFamily.OVERCOOKED,
    GameId.OVERCOOKED_FORCED_COORDINATION: EnvironmentFamily.OVERCOOKED,
    GameId.OVERCOOKED_COUNTER_CIRCUIT: EnvironmentFamily.OVERCOOKED,
}


DEFAULT_RENDER_MODES: dict[GameId, RenderMode] = {
    GameId.FROZEN_LAKE: RenderMode.GRID,
    GameId.FROZEN_LAKE_V2: RenderMode.GRID,
    GameId.CLIFF_WALKING: RenderMode.GRID,
    GameId.TAXI: RenderMode.GRID,
    GameId.BLACKJACK: RenderMode.RGB_ARRAY,
    GameId.LUNAR_LANDER: RenderMode.RGB_ARRAY,
    GameId.CAR_RACING: RenderMode.RGB_ARRAY,
    GameId.BIPEDAL_WALKER: RenderMode.RGB_ARRAY,
    GameId.CART_POLE: RenderMode.RGB_ARRAY,
    GameId.ACROBOT: RenderMode.RGB_ARRAY,
    GameId.MOUNTAIN_CAR: RenderMode.RGB_ARRAY,
    GameId.PONG_NO_FRAMESKIP: RenderMode.RGB_ARRAY,
    GameId.BREAKOUT_NO_FRAMESKIP: RenderMode.RGB_ARRAY,
    GameId.ADVENTURE_V4: RenderMode.RGB_ARRAY,
    GameId.ALE_ADVENTURE_V5: RenderMode.RGB_ARRAY,
    GameId.AIR_RAID_V4: RenderMode.RGB_ARRAY,
    GameId.ALE_AIR_RAID_V5: RenderMode.RGB_ARRAY,
    GameId.ASSAULT_V4: RenderMode.RGB_ARRAY,
    GameId.ALE_ASSAULT_V5: RenderMode.RGB_ARRAY,
    GameId.ANT: RenderMode.RGB_ARRAY,
    GameId.HALF_CHEETAH: RenderMode.RGB_ARRAY,
    GameId.HOPPER: RenderMode.RGB_ARRAY,
    GameId.HUMANOID: RenderMode.RGB_ARRAY,
    GameId.HUMANOID_STANDUP: RenderMode.RGB_ARRAY,
    GameId.INVERTED_DOUBLE_PENDULUM: RenderMode.RGB_ARRAY,
    GameId.INVERTED_PENDULUM: RenderMode.RGB_ARRAY,
    GameId.PUSHER: RenderMode.RGB_ARRAY,
    GameId.REACHER: RenderMode.RGB_ARRAY,
    GameId.SWIMMER: RenderMode.RGB_ARRAY,
    GameId.WALKER2D: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_8x8: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_16x16: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_8x8: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_16x16: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S7: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S9N1: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S9N2: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S9N3: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S11N5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_REDBLUE_DOORS_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_REDBLUE_DOORS_8x8: RenderMode.RGB_ARRAY,
    # BabyAI environments
    GameId.BABYAI_GOTO_REDBALL_GREY: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_REDBALL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_REDBALL_NODISTS: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_OBJ: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_LOCAL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_IMPUNLOCK: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_SEQ: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_REDBLUEBALL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_DOOR: RenderMode.RGB_ARRAY,
    GameId.BABYAI_GOTO_OBJDOOR: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN_REDDOOR: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN_DOOR: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN_TWODOORS: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN_DOORSORDER_N2: RenderMode.RGB_ARRAY,
    GameId.BABYAI_OPEN_DOORSORDER_N4: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PICKUP: RenderMode.RGB_ARRAY,
    GameId.BABYAI_UNBLOCK_PICKUP: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PICKUP_LOC: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PICKUP_DIST: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PICKUP_ABOVE: RenderMode.RGB_ARRAY,
    GameId.BABYAI_UNLOCK: RenderMode.RGB_ARRAY,
    GameId.BABYAI_UNLOCK_LOCAL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_KEY_INBOX: RenderMode.RGB_ARRAY,
    GameId.BABYAI_UNLOCK_PICKUP: RenderMode.RGB_ARRAY,
    GameId.BABYAI_BLOCKED_UNLOCK_PICKUP: RenderMode.RGB_ARRAY,
    GameId.BABYAI_UNLOCK_TO_UNLOCK: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PUTNEXT_LOCAL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_PUTNEXT: RenderMode.RGB_ARRAY,
    GameId.BABYAI_ACTION_OBJDOOR: RenderMode.RGB_ARRAY,
    GameId.BABYAI_FINDOBJ_S5: RenderMode.RGB_ARRAY,
    GameId.BABYAI_KEYCORRIDOR_S3R1: RenderMode.RGB_ARRAY,
    GameId.BABYAI_KEYCORRIDOR_S3R2: RenderMode.RGB_ARRAY,
    GameId.BABYAI_KEYCORRIDOR_S3R3: RenderMode.RGB_ARRAY,
    GameId.BABYAI_ONEROOM_S8: RenderMode.RGB_ARRAY,
    GameId.BABYAI_MOVETWOACROSS_S8N9: RenderMode.RGB_ARRAY,
    GameId.BABYAI_SYNTH: RenderMode.RGB_ARRAY,
    GameId.BABYAI_SYNTHLOC: RenderMode.RGB_ARRAY,
    GameId.BABYAI_SYNTHSEQ: RenderMode.RGB_ARRAY,
    GameId.BABYAI_MINIBOSSLEVEL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_BOSSLEVEL: RenderMode.RGB_ARRAY,
    GameId.BABYAI_BOSSLEVEL_NOUNLOCK: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_BASIC: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_DEADLY_CORRIDOR: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_DEFEND_THE_CENTER: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_DEFEND_THE_LINE: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_HEALTH_GATHERING: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_MY_WAY_HOME: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_PREDICT_POSITION: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_TAKE_COVER: RenderMode.RGB_ARRAY,
    GameId.VIZDOOM_DEATHMATCH: RenderMode.RGB_ARRAY,
    # Board games use custom Qt widget rendering, not RGB_ARRAY
    GameId.CHESS: RenderMode.RGB_ARRAY,  # Fallback, but we use InteractiveChessBoard
    GameId.CONNECT_FOUR: RenderMode.RGB_ARRAY,  # Fallback, but we use InteractiveConnectFourBoard
    GameId.GO: RenderMode.RGB_ARRAY,  # Fallback, but we use InteractiveGoBoard
    GameId.TIC_TAC_TOE: RenderMode.RGB_ARRAY,  # Fallback, but we use InteractiveTicTacToeBoard
    # MiniHack - pixel rendering (16x16 tiles)
    GameId.MINIHACK_ROOM_5X5: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_ROOM_15X15: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_CORRIDOR_R2: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_CORRIDOR_R3: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_CORRIDOR_R5: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_MAZEWALK_9X9: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_MAZEWALK_15X15: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_MAZEWALK_45X19: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_RIVER: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_RIVER_NARROW: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_EAT: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_WEAR: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_WIELD: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_ZAP: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_READ: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_QUAFF: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_PUTON: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_LAVACROSS: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_WOD_EASY: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_WOD_MEDIUM: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_WOD_HARD: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_EXPLOREMAZE_EASY: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_EXPLOREMAZE_HARD: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_HIDENSEEK: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_MEMENTO_F2: RenderMode.RGB_ARRAY,
    GameId.MINIHACK_MEMENTO_F4: RenderMode.RGB_ARRAY,
    # NetHack - pixel rendering
    GameId.NETHACK_FULL: RenderMode.RGB_ARRAY,
    GameId.NETHACK_SCORE: RenderMode.RGB_ARRAY,
    GameId.NETHACK_STAIRCASE: RenderMode.RGB_ARRAY,
    GameId.NETHACK_STAIRCASE_PET: RenderMode.RGB_ARRAY,
    GameId.NETHACK_ORACLE: RenderMode.RGB_ARRAY,
    GameId.NETHACK_GOLD: RenderMode.RGB_ARRAY,
    GameId.NETHACK_EAT: RenderMode.RGB_ARRAY,
    GameId.NETHACK_SCOUT: RenderMode.RGB_ARRAY,
    # Crafter - RGB observation (64x64x3)
    GameId.CRAFTER_REWARD: RenderMode.RGB_ARRAY,
    GameId.CRAFTER_NO_REWARD: RenderMode.RGB_ARRAY,
    # TextWorld - ANSI text output
    GameId.TEXTWORLD_SIMPLE: RenderMode.ANSI,
    GameId.TEXTWORLD_COIN_COLLECTOR: RenderMode.ANSI,
    GameId.TEXTWORLD_TREASURE_HUNTER: RenderMode.ANSI,
    GameId.TEXTWORLD_COOKING: RenderMode.ANSI,
    GameId.TEXTWORLD_CUSTOM: RenderMode.ANSI,
    # BabaIsAI - RGB observation
    GameId.BABAISAI_DEFAULT: RenderMode.RGB_ARRAY,
    # Procgen - RGB observation (64x64x3)
    GameId.PROCGEN_BIGFISH: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_BOSSFIGHT: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_CAVEFLYER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_CHASER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_CLIMBER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_COINRUN: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_DODGEBALL: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_FRUITBOT: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_HEIST: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_JUMPER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_LEAPER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_MAZE: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_MINER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_NINJA: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_PLUNDER: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_STARPILOT: RenderMode.RGB_ARRAY,
    # Jumanji Logic Puzzle Environments
    GameId.JUMANJI_GAME2048: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_MINESWEEPER: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_RUBIKS_CUBE: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_SLIDING_PUZZLE: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_SUDOKU: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_GRAPH_COLORING: RenderMode.RGB_ARRAY,
    # Jumanji Phase 2: Packing
    GameId.JUMANJI_BINPACK: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_FLATPACK: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_JOBSHOP: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_KNAPSACK: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_TETRIS: RenderMode.RGB_ARRAY,
    # Jumanji Phase 3: Routing
    GameId.JUMANJI_CLEANER: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_CONNECTOR: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_CVRP: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_MAZE: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_MMST: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_MULTI_CVRP: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_PACMAN: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_ROBOT_WAREHOUSE: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_SNAKE: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_SOKOBAN: RenderMode.RGB_ARRAY,
    GameId.JUMANJI_TSP: RenderMode.RGB_ARRAY,
    # PyBullet Drones (3D physics simulation with PyBullet rendering)
    GameId.PYBULLET_HOVER_AVIARY: RenderMode.RGB_ARRAY,
    GameId.PYBULLET_MULTIHOVER_AVIARY: RenderMode.RGB_ARRAY,
    GameId.PYBULLET_CTRL_AVIARY: RenderMode.RGB_ARRAY,
    GameId.PYBULLET_VELOCITY_AVIARY: RenderMode.RGB_ARRAY,
    # OpenSpiel (board games rendered via Shimmy)
    GameId.OPEN_SPIEL_CHECKERS: RenderMode.RGB_ARRAY,
    # Draughts/Checkers variants
    GameId.AMERICAN_CHECKERS: RenderMode.RGB_ARRAY,
    GameId.RUSSIAN_CHECKERS: RenderMode.RGB_ARRAY,
    GameId.INTERNATIONAL_DRAUGHTS: RenderMode.RGB_ARRAY,
    # gym-multigrid (multi-agent grid environments)
    GameId.MULTIGRID_SOCCER: RenderMode.RGB_ARRAY,
    GameId.MULTIGRID_COLLECT: RenderMode.RGB_ARRAY,
    # Melting Pot (multi-agent social scenarios via Shimmy)
    GameId.MELTINGPOT_COLLABORATIVE_COOKING: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_CLEAN_UP: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_COMMONS_HARVEST: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_TERRITORY: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_KING_OF_THE_HILL: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_PRISONERS_DILEMMA: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_STAG_HUNT: RenderMode.RGB_ARRAY,
    GameId.MELTINGPOT_ALLELOPATHIC_HARVEST: RenderMode.RGB_ARRAY,
    # Overcooked-AI (cooperative cooking)
    GameId.OVERCOOKED_CRAMPED_ROOM: RenderMode.RGB_ARRAY,
    GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES: RenderMode.RGB_ARRAY,
    GameId.OVERCOOKED_COORDINATION_RING: RenderMode.RGB_ARRAY,
    GameId.OVERCOOKED_FORCED_COORDINATION: RenderMode.RGB_ARRAY,
    GameId.OVERCOOKED_COUNTER_CIRCUIT: RenderMode.RGB_ARRAY,
}


DEFAULT_CONTROL_MODES: dict[GameId, Iterable[ControlMode]] = {
    GameId.FROZEN_LAKE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.FROZEN_LAKE_V2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.CLIFF_WALKING: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.TAXI: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.BLACKJACK: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.LUNAR_LANDER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.CAR_RACING: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.BIPEDAL_WALKER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.CART_POLE: (ControlMode.AGENT_ONLY,),
    GameId.ACROBOT: (ControlMode.AGENT_ONLY,),
    GameId.MOUNTAIN_CAR: (ControlMode.AGENT_ONLY,),
    GameId.PONG_NO_FRAMESKIP: (ControlMode.AGENT_ONLY,),
    GameId.BREAKOUT_NO_FRAMESKIP: (ControlMode.AGENT_ONLY,),
    GameId.ADVENTURE_V4: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.ALE_ADVENTURE_V5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.AIR_RAID_V4: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.ALE_AIR_RAID_V5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.ASSAULT_V4: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.ALE_ASSAULT_V5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.ANT: (ControlMode.AGENT_ONLY,),
    GameId.HALF_CHEETAH: (ControlMode.AGENT_ONLY,),
    GameId.HOPPER: (ControlMode.AGENT_ONLY,),
    GameId.HUMANOID: (ControlMode.AGENT_ONLY,),
    GameId.HUMANOID_STANDUP: (ControlMode.AGENT_ONLY,),
    GameId.INVERTED_DOUBLE_PENDULUM: (ControlMode.AGENT_ONLY,),
    GameId.INVERTED_PENDULUM: (ControlMode.AGENT_ONLY,),
    GameId.PUSHER: (ControlMode.AGENT_ONLY,),
    GameId.REACHER: (ControlMode.AGENT_ONLY,),
    GameId.SWIMMER: (ControlMode.AGENT_ONLY,),
    GameId.WALKER2D: (ControlMode.AGENT_ONLY,),
    GameId.MINIGRID_EMPTY_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_RANDOM_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_RANDOM_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_8x8: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_16x16: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_8x8: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_16x16: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S7: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N1: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N3: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S11N5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_REDBLUE_DOORS_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_REDBLUE_DOORS_8x8: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.VIZDOOM_BASIC: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_DEADLY_CORRIDOR: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_DEFEND_THE_CENTER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_DEFEND_THE_LINE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_HEALTH_GATHERING: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_MY_WAY_HOME: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_PREDICT_POSITION: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_TAKE_COVER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    GameId.VIZDOOM_DEATHMATCH: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    ),
    # Board games - Human Only for now (both players are human)
    GameId.CHESS: (
        ControlMode.HUMAN_ONLY,
    ),
    GameId.CONNECT_FOUR: (
        ControlMode.HUMAN_ONLY,
    ),
    GameId.GO: (
        ControlMode.HUMAN_ONLY,
    ),
    GameId.TIC_TAC_TOE: (
        ControlMode.HUMAN_ONLY,
    ),
    # MiniHack - turn-based roguelike (supports Human Control)
    GameId.MINIHACK_ROOM_5X5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_ROOM_15X15: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_CORRIDOR_R2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_CORRIDOR_R3: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_CORRIDOR_R5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_MAZEWALK_9X9: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_MAZEWALK_15X15: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_MAZEWALK_45X19: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_RIVER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_RIVER_NARROW: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_EAT: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_WEAR: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_WIELD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_ZAP: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_READ: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_QUAFF: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_PUTON: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_LAVACROSS: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_WOD_EASY: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_WOD_MEDIUM: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_WOD_HARD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_EXPLOREMAZE_EASY: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_EXPLOREMAZE_HARD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_HIDENSEEK: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_MEMENTO_F2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.MINIHACK_MEMENTO_F4: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    # NetHack (full game) - turn-based roguelike
    GameId.NETHACK_FULL: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_SCORE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_STAIRCASE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_STAIRCASE_PET: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_ORACLE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_GOLD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_EAT: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.NETHACK_SCOUT: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    # Crafter - turn-based open world survival (supports Human Control)
    GameId.CRAFTER_REWARD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.CRAFTER_NO_REWARD: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    # TextWorld - text-based adventure games (text command input)
    GameId.TEXTWORLD_SIMPLE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.TEXTWORLD_COIN_COLLECTOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.TEXTWORLD_TREASURE_HUNTER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.TEXTWORLD_COOKING: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.TEXTWORLD_CUSTOM: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # BabaIsAI - turn-based puzzle game (supports Human Control)
    GameId.BABAISAI_DEFAULT: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # Procgen - procedurally generated games (supports Human Control)
    GameId.PROCGEN_BIGFISH: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_BOSSFIGHT: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_CAVEFLYER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_CHASER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_CLIMBER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_COINRUN: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_DODGEBALL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_FRUITBOT: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_HEIST: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_JUMPER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_LEAPER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_MAZE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_MINER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_NINJA: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_PLUNDER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.PROCGEN_STARPILOT: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # Jumanji - JAX-based logic puzzle environments (turn-based discrete actions)
    GameId.JUMANJI_GAME2048: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_MINESWEEPER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_RUBIKS_CUBE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_SLIDING_PUZZLE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_SUDOKU: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_GRAPH_COLORING: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # Jumanji Phase 2: Packing (turn-based discrete actions)
    GameId.JUMANJI_BINPACK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_FLATPACK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_JOBSHOP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_KNAPSACK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_TETRIS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # Jumanji Phase 3: Routing (turn-based discrete actions)
    GameId.JUMANJI_CLEANER: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_CONNECTOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_CVRP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_MAZE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_MMST: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_MULTI_CVRP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_PACMAN: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_ROBOT_WAREHOUSE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_SNAKE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_SOKOBAN: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.JUMANJI_TSP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # BabyAI - language-grounded instruction following (same controls as MiniGrid)
    GameId.BABYAI_GOTO_REDBALL_GREY: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_REDBALL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_REDBALL_NODISTS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_OBJ: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_LOCAL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_IMPUNLOCK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_SEQ: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_REDBLUEBALL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_DOOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_GOTO_OBJDOOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN_REDDOOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN_DOOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN_TWODOORS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN_DOORSORDER_N2: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_OPEN_DOORSORDER_N4: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PICKUP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_UNBLOCK_PICKUP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PICKUP_LOC: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PICKUP_DIST: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PICKUP_ABOVE: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_UNLOCK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_UNLOCK_LOCAL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_KEY_INBOX: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_UNLOCK_PICKUP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_BLOCKED_UNLOCK_PICKUP: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_UNLOCK_TO_UNLOCK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PUTNEXT_LOCAL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_PUTNEXT: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_ACTION_OBJDOOR: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_FINDOBJ_S5: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_KEYCORRIDOR_S3R1: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_KEYCORRIDOR_S3R2: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_KEYCORRIDOR_S3R3: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_ONEROOM_S8: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_MOVETWOACROSS_S8N9: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_SYNTH: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_SYNTHLOC: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_SYNTHSEQ: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_MINIBOSSLEVEL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_BOSSLEVEL: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.BABYAI_BOSSLEVEL_NOUNLOCK: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # PyBullet Drones - Continuous control quadcopter environments
    # Note: These require continuous actions (RPM/velocity), primarily agent-controlled
    GameId.PYBULLET_HOVER_AVIARY: (ControlMode.AGENT_ONLY,),
    GameId.PYBULLET_MULTIHOVER_AVIARY: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.PYBULLET_CTRL_AVIARY: (ControlMode.AGENT_ONLY,),
    GameId.PYBULLET_VELOCITY_AVIARY: (ControlMode.AGENT_ONLY,),
    # OpenSpiel - Turn-based board games via Shimmy
    GameId.OPEN_SPIEL_CHECKERS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # Draughts/Checkers variants - Turn-based board games with proper rule implementations
    GameId.AMERICAN_CHECKERS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.RUSSIAN_CHECKERS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    GameId.INTERNATIONAL_DRAUGHTS: (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    # gym-multigrid - Multi-agent environments (simultaneous stepping)
    GameId.MULTIGRID_SOCCER: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP, ControlMode.MULTI_AGENT_COMPETITIVE),
    GameId.MULTIGRID_COLLECT: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP, ControlMode.MULTI_AGENT_COMPETITIVE),
    # Melting Pot - Multi-agent social scenarios (parallel stepping)
    GameId.MELTINGPOT_COLLABORATIVE_COOKING: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.MELTINGPOT_CLEAN_UP: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.MELTINGPOT_COMMONS_HARVEST: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP, ControlMode.MULTI_AGENT_COMPETITIVE),
    GameId.MELTINGPOT_TERRITORY: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COMPETITIVE),
    GameId.MELTINGPOT_KING_OF_THE_HILL: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COMPETITIVE),
    GameId.MELTINGPOT_PRISONERS_DILEMMA: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COMPETITIVE),
    GameId.MELTINGPOT_STAG_HUNT: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.MELTINGPOT_ALLELOPATHIC_HARVEST: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COMPETITIVE),
    # Overcooked-AI - 2-agent cooperative cooking
    GameId.OVERCOOKED_CRAMPED_ROOM: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.OVERCOOKED_COORDINATION_RING: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.OVERCOOKED_FORCED_COORDINATION: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    GameId.OVERCOOKED_COUNTER_CIRCUIT: (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
}


# ─────────────────────────────────────────────────────────────────────────────
# Stepping Paradigm Mappings
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARADIGM_BY_FAMILY: dict[EnvironmentFamily, SteppingParadigm] = {
    # Maps environment families to their default stepping paradigm.
    # Adapters can override this by setting their own `stepping_paradigm` attribute.
    EnvironmentFamily.TOY_TEXT: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.BOX2D: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.CLASSIC_CONTROL: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.ATARI: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.ALE: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.MUJOCO: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.MINIGRID: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.BABYAI: SteppingParadigm.SINGLE_AGENT,  # Language-grounded MiniGrid
    EnvironmentFamily.VIZDOOM: SteppingParadigm.SINGLE_AGENT,
    EnvironmentFamily.MINIHACK: SteppingParadigm.SINGLE_AGENT,  # Turn-based roguelike
    EnvironmentFamily.NETHACK: SteppingParadigm.SINGLE_AGENT,  # Turn-based roguelike
    EnvironmentFamily.CRAFTER: SteppingParadigm.SINGLE_AGENT,  # Turn-based survival game
    EnvironmentFamily.PROCGEN: SteppingParadigm.SINGLE_AGENT,  # Procedurally generated games
    EnvironmentFamily.JUMANJI: SteppingParadigm.SINGLE_AGENT,  # JAX-based logic puzzles (turn-based)
    EnvironmentFamily.TEXTWORLD: SteppingParadigm.SINGLE_AGENT,  # Text-based adventure games
    EnvironmentFamily.BABAISAI: SteppingParadigm.SINGLE_AGENT,  # Turn-based puzzle game
    EnvironmentFamily.PETTINGZOO: SteppingParadigm.SEQUENTIAL,  # AEC by default
    EnvironmentFamily.PETTINGZOO_CLASSIC: SteppingParadigm.SEQUENTIAL,  # Chess, Go, etc.
    EnvironmentFamily.PYBULLET_DRONES: SteppingParadigm.SINGLE_AGENT,  # Single/multi-agent quadcopter control
    EnvironmentFamily.OPEN_SPIEL: SteppingParadigm.SEQUENTIAL,  # Turn-based board games (Checkers, etc.)
    EnvironmentFamily.MULTIGRID: SteppingParadigm.SIMULTANEOUS,  # Multi-agent grid envs (all agents act at once)
    EnvironmentFamily.MELTINGPOT: SteppingParadigm.SIMULTANEOUS,  # Multi-agent social scenarios (parallel stepping)
    EnvironmentFamily.OVERCOOKED: SteppingParadigm.SIMULTANEOUS,  # 2-agent cooperative cooking (parallel stepping)
    EnvironmentFamily.OTHER: SteppingParadigm.SINGLE_AGENT,  # Fallback
}


def get_stepping_paradigm(game_id: GameId) -> SteppingParadigm:
    """Infer the stepping paradigm for a game based on its environment family.

    Args:
        game_id: The game identifier.

    Returns:
        The stepping paradigm for the game. Defaults to SINGLE_AGENT if unknown.
    """
    family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id, EnvironmentFamily.OTHER)
    return DEFAULT_PARADIGM_BY_FAMILY.get(family, SteppingParadigm.SINGLE_AGENT)


__all__ = [
    "EnvironmentFamily",
    "GameId",
    "ControlMode",
    "InputMode",
    "INPUT_MODE_INFO",
    "RenderMode",
    "ActionSpaceType",
    "AgentRole",
    "AdapterCapability",
    "SteppingParadigm",
    "ENVIRONMENT_FAMILY_BY_GAME",
    "DEFAULT_RENDER_MODES",
    "DEFAULT_CONTROL_MODES",
    "DEFAULT_PARADIGM_BY_FAMILY",
    "get_game_display_name",
    "get_stepping_paradigm",
]
