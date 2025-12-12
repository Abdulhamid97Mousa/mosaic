"""PettingZoo-specific enumerations for multi-agent environments."""

from __future__ import annotations

from enum import auto
from typing import Dict, List, Tuple

from gym_gui.core.enums import StrEnum, ControlMode, RenderMode


class PettingZooFamily(StrEnum):
    """PettingZoo environment families."""

    ATARI = "atari"
    BUTTERFLY = "butterfly"
    CLASSIC = "classic"
    MPE = "mpe"
    SISL = "sisl"


class PettingZooAPIType(StrEnum):
    """PettingZoo API types for environment interaction."""

    AEC = "aec"  # Agent Environment Cycle (turn-based)
    PARALLEL = "parallel"  # Simultaneous actions


class PettingZooEnvId(StrEnum):
    """PettingZoo environment identifiers.

    These map to the actual PettingZoo environment import paths.
    """

    # ═══════════════════════════════════════════════════════════════
    # Butterfly Family (Parallel - Cooperative Visual)
    # ═══════════════════════════════════════════════════════════════
    KNIGHTS_ARCHERS_ZOMBIES = "knights_archers_zombies_v10"
    PISTONBALL = "pistonball_v6"
    COOPERATIVE_PONG = "cooperative_pong_v5"

    # ═══════════════════════════════════════════════════════════════
    # MPE Family (Parallel - Multi-Particle Environments)
    # ═══════════════════════════════════════════════════════════════
    SIMPLE_SPREAD = "simple_spread_v3"
    SIMPLE_ADVERSARY = "simple_adversary_v3"
    SIMPLE_TAG = "simple_tag_v3"
    SIMPLE_PUSH = "simple_push_v3"
    SIMPLE_CRYPTO = "simple_crypto_v3"
    SIMPLE_REFERENCE = "simple_reference_v3"
    SIMPLE_SPEAKER_LISTENER = "simple_speaker_listener_v4"
    SIMPLE_WORLD_COMM = "simple_world_comm_v3"

    # ═══════════════════════════════════════════════════════════════
    # SISL Family (Parallel - Cooperative Continuous Control)
    # ═══════════════════════════════════════════════════════════════
    MULTIWALKER = "multiwalker_v9"
    WATERWORLD = "waterworld_v4"
    PURSUIT = "pursuit_v4"

    # ═══════════════════════════════════════════════════════════════
    # Classic Family (AEC - Turn-based Strategy Games)
    # ═══════════════════════════════════════════════════════════════
    CHESS = "chess_v6"
    GO = "go_v5"
    HANABI = "hanabi_v5"
    TEXAS_HOLDEM = "texas_holdem_v4"
    TEXAS_HOLDEM_NO_LIMIT = "texas_holdem_no_limit_v6"
    CONNECT_FOUR = "connect_four_v3"
    TIC_TAC_TOE = "tictactoe_v3"
    GIN_RUMMY = "gin_rummy_v4"
    LEDUC_HOLDEM = "leduc_holdem_v4"
    RPS = "rps_v2"  # Rock Paper Scissors
    BACKGAMMON = "backgammon_v3"

    # ═══════════════════════════════════════════════════════════════
    # Atari Family (Parallel - Competitive/Cooperative 2-Player)
    # Requires ROM license acceptance
    # Note: Some games are cooperative (space_invaders), others competitive
    # ═══════════════════════════════════════════════════════════════
    # Classic Competitive
    PONG = "pong_v3"
    BOXING = "boxing_v2"
    TENNIS = "tennis_v3"
    DOUBLE_DUNK = "double_dunk_v3"
    ICE_HOCKEY = "ice_hockey_v2"
    SURROUND = "surround_v2"
    VIDEO_CHECKERS = "video_checkers_v4"
    OTHELLO = "othello_v3"
    JOUST = "joust_v3"
    COMBAT_PLANE = "combat_plane_v2"
    COMBAT_TANK = "combat_tank_v2"
    WARLORDS = "warlords_v3"
    WIZARD_OF_WOR = "wizard_of_wor_v3"
    MARIO_BROS = "mario_bros_v3"
    ENTOMBED_COMPETITIVE = "entombed_competitive_v3"
    FLAG_CAPTURE = "flag_capture_v2"
    # Cooperative
    SPACE_INVADERS = "space_invaders_v2"
    ENTOMBED_COOPERATIVE = "entombed_cooperative_v3"
    SPACE_WAR = "space_war_v2"
    MAZE_CRAZE = "maze_craze_v3"
    QUADRAPONG = "quadrapong_v4"


# Environment metadata: maps env_id -> (family, api_type, display_name, description)
PETTINGZOO_ENV_METADATA: Dict[PettingZooEnvId, Tuple[PettingZooFamily, PettingZooAPIType, str, str]] = {
    # Butterfly
    PettingZooEnvId.KNIGHTS_ARCHERS_ZOMBIES: (
        PettingZooFamily.BUTTERFLY,
        PettingZooAPIType.PARALLEL,
        "Knights Archers Zombies",
        "Cooperative game where knights and archers defend against zombies",
    ),
    PettingZooEnvId.PISTONBALL: (
        PettingZooFamily.BUTTERFLY,
        PettingZooAPIType.PARALLEL,
        "Pistonball",
        "Cooperative game where pistons push a ball to the left",
    ),
    PettingZooEnvId.COOPERATIVE_PONG: (
        PettingZooFamily.BUTTERFLY,
        PettingZooAPIType.PARALLEL,
        "Cooperative Pong",
        "Cooperative variant of Pong where agents work together",
    ),
    # MPE
    PettingZooEnvId.SIMPLE_SPREAD: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Spread",
        "Agents must spread out and cover all landmarks",
    ),
    PettingZooEnvId.SIMPLE_ADVERSARY: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Adversary",
        "Adversary tries to reach target while agents try to block",
    ),
    PettingZooEnvId.SIMPLE_TAG: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Tag",
        "Predator-prey pursuit game",
    ),
    PettingZooEnvId.SIMPLE_PUSH: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Push",
        "Agent pushes adversary away from landmark",
    ),
    PettingZooEnvId.SIMPLE_CRYPTO: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Crypto",
        "Communication game with encryption",
    ),
    PettingZooEnvId.SIMPLE_REFERENCE: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Simple Reference",
        "Referential game with communication",
    ),
    PettingZooEnvId.SIMPLE_SPEAKER_LISTENER: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "Speaker Listener",
        "One agent speaks, another listens and acts",
    ),
    PettingZooEnvId.SIMPLE_WORLD_COMM: (
        PettingZooFamily.MPE,
        PettingZooAPIType.PARALLEL,
        "World Comm",
        "Multiple speakers and listeners",
    ),
    # SISL
    PettingZooEnvId.MULTIWALKER: (
        PettingZooFamily.SISL,
        PettingZooAPIType.PARALLEL,
        "Multiwalker",
        "Multiple bipedal walkers carry a package together",
    ),
    PettingZooEnvId.WATERWORLD: (
        PettingZooFamily.SISL,
        PettingZooAPIType.PARALLEL,
        "Waterworld",
        "Agents navigate and capture food while avoiding poison",
    ),
    PettingZooEnvId.PURSUIT: (
        PettingZooFamily.SISL,
        PettingZooAPIType.PARALLEL,
        "Pursuit",
        "Predators pursue prey in a grid world",
    ),
    # Classic (AEC)
    PettingZooEnvId.CHESS: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Chess",
        "Standard chess game",
    ),
    PettingZooEnvId.GO: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Go",
        "Classic board game Go",
    ),
    PettingZooEnvId.HANABI: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Hanabi",
        "Cooperative card game with limited information",
    ),
    PettingZooEnvId.TEXAS_HOLDEM: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Texas Hold'em",
        "Limit Texas Hold'em poker",
    ),
    PettingZooEnvId.TEXAS_HOLDEM_NO_LIMIT: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Texas Hold'em No Limit",
        "No-limit Texas Hold'em poker",
    ),
    PettingZooEnvId.CONNECT_FOUR: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Connect Four",
        "Classic Connect Four game",
    ),
    PettingZooEnvId.TIC_TAC_TOE: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Tic-Tac-Toe",
        "Classic Tic-Tac-Toe game",
    ),
    PettingZooEnvId.GIN_RUMMY: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Gin Rummy",
        "Two-player card game",
    ),
    PettingZooEnvId.LEDUC_HOLDEM: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Leduc Hold'em",
        "Simplified poker variant",
    ),
    PettingZooEnvId.RPS: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Rock Paper Scissors",
        "Classic Rock Paper Scissors game",
    ),
    PettingZooEnvId.BACKGAMMON: (
        PettingZooFamily.CLASSIC,
        PettingZooAPIType.AEC,
        "Backgammon",
        "Classic backgammon board game",
    ),
    # Atari - Competitive Games
    PettingZooEnvId.PONG: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Pong",
        "Classic two-player Pong - first to 21 points wins",
    ),
    PettingZooEnvId.BOXING: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Boxing",
        "Two-player boxing match - score points by landing punches",
    ),
    PettingZooEnvId.TENNIS: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Tennis",
        "Two-player tennis with full court gameplay",
    ),
    PettingZooEnvId.DOUBLE_DUNK: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Double Dunk",
        "Two-on-two basketball game with dunking mechanics",
    ),
    PettingZooEnvId.ICE_HOCKEY: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Ice Hockey",
        "Fast-paced two-player ice hockey",
    ),
    PettingZooEnvId.SURROUND: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Surround",
        "Tron-like light cycle game - trap your opponent",
    ),
    PettingZooEnvId.VIDEO_CHECKERS: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Video Checkers",
        "Classic checkers/draughts board game",
    ),
    PettingZooEnvId.OTHELLO: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Othello",
        "Classic Reversi/Othello board game",
    ),
    PettingZooEnvId.JOUST: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Joust",
        "Aerial combat on flying ostriches - competitive or co-op",
    ),
    PettingZooEnvId.COMBAT_PLANE: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Combat: Plane",
        "Two-player aerial dogfight with biplanes",
    ),
    PettingZooEnvId.COMBAT_TANK: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Combat: Tank",
        "Two-player tank battle with obstacles",
    ),
    PettingZooEnvId.WARLORDS: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Warlords",
        "Medieval castle defense - protect your king",
    ),
    PettingZooEnvId.WIZARD_OF_WOR: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Wizard of Wor",
        "Dungeon maze shooter - competitive or co-op",
    ),
    PettingZooEnvId.MARIO_BROS: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Mario Bros",
        "Original Mario Bros - clear enemies from sewers",
    ),
    PettingZooEnvId.ENTOMBED_COMPETITIVE: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Entombed (Competitive)",
        "Race through an endless maze - outlast your opponent",
    ),
    PettingZooEnvId.FLAG_CAPTURE: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Flag Capture",
        "Capture the flag strategy game",
    ),
    # Atari - Cooperative Games
    PettingZooEnvId.SPACE_INVADERS: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Space Invaders",
        "Cooperative alien defense - work together to survive",
    ),
    PettingZooEnvId.ENTOMBED_COOPERATIVE: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Entombed (Cooperative)",
        "Cooperative maze survival - escape together",
    ),
    PettingZooEnvId.SPACE_WAR: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Space War",
        "Space combat around a gravitational star",
    ),
    PettingZooEnvId.MAZE_CRAZE: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Maze Craze",
        "Race through randomly generated mazes",
    ),
    PettingZooEnvId.QUADRAPONG: (
        PettingZooFamily.ATARI,
        PettingZooAPIType.PARALLEL,
        "Quadrapong",
        "Four-player Pong variant with four paddles",
    ),
}


def get_envs_by_family(family: PettingZooFamily) -> List[PettingZooEnvId]:
    """Get all environment IDs for a given family."""
    return [
        env_id
        for env_id, (env_family, _, _, _) in PETTINGZOO_ENV_METADATA.items()
        if env_family == family
    ]


def get_api_type(env_id: PettingZooEnvId) -> PettingZooAPIType:
    """Get the API type (AEC or Parallel) for an environment."""
    return PETTINGZOO_ENV_METADATA[env_id][1]


def get_display_name(env_id: PettingZooEnvId) -> str:
    """Get the display name for an environment."""
    return PETTINGZOO_ENV_METADATA[env_id][2]


def get_description(env_id: PettingZooEnvId) -> str:
    """Get the description for an environment."""
    return PETTINGZOO_ENV_METADATA[env_id][3]


def is_aec_env(env_id: PettingZooEnvId) -> bool:
    """Check if an environment uses the AEC (turn-based) API."""
    return get_api_type(env_id) == PettingZooAPIType.AEC


def is_parallel_env(env_id: PettingZooEnvId) -> bool:
    """Check if an environment uses the Parallel (simultaneous) API."""
    return get_api_type(env_id) == PettingZooAPIType.PARALLEL


# Environments that support human control (turn-based games with discrete actions)
HUMAN_CONTROLLABLE_ENVS: frozenset[PettingZooEnvId] = frozenset({
    # All AEC environments support human control (turn-based)
    # Currently limited to board games with interactive UI support
    PettingZooEnvId.CHESS,
    PettingZooEnvId.CONNECT_FOUR,
    PettingZooEnvId.GO,
    PettingZooEnvId.TIC_TAC_TOE,
})


# Control modes supported by PettingZoo environments
PETTINGZOO_CONTROL_MODES: Dict[PettingZooEnvId, Tuple[ControlMode, ...]] = {
    # AEC (turn-based) - support human control
    PettingZooEnvId.CHESS: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.CONNECT_FOUR: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.TIC_TAC_TOE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.RPS: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.GO: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.HANABI: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.TEXAS_HOLDEM: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.TEXAS_HOLDEM_NO_LIMIT: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.GIN_RUMMY: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.LEDUC_HOLDEM: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.BACKGAMMON: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    # Parallel (simultaneous) - mostly agent-only or multi-agent
    PettingZooEnvId.KNIGHTS_ARCHERS_ZOMBIES: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.PISTONBALL: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.COOPERATIVE_PONG: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SIMPLE_SPREAD: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SIMPLE_ADVERSARY: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.SIMPLE_TAG: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.SIMPLE_PUSH: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.SIMPLE_CRYPTO: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SIMPLE_REFERENCE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SIMPLE_SPEAKER_LISTENER: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SIMPLE_WORLD_COMM: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.MULTIWALKER: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.WATERWORLD: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.PURSUIT: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.PONG: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.BOXING: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.TENNIS: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.SPACE_INVADERS: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.DOUBLE_DUNK: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.ICE_HOCKEY: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    # New Atari - Competitive
    PettingZooEnvId.SURROUND: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.VIDEO_CHECKERS: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.OTHELLO: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.JOUST: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
        ControlMode.MULTI_AGENT_COOP,  # Can be played cooperatively
    ),
    PettingZooEnvId.COMBAT_PLANE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.COMBAT_TANK: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.WARLORDS: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.WIZARD_OF_WOR: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
        ControlMode.MULTI_AGENT_COOP,  # Can be played cooperatively
    ),
    PettingZooEnvId.MARIO_BROS: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.ENTOMBED_COMPETITIVE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.FLAG_CAPTURE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    # New Atari - Cooperative
    PettingZooEnvId.ENTOMBED_COOPERATIVE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    ),
    PettingZooEnvId.SPACE_WAR: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.MAZE_CRAZE: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    PettingZooEnvId.QUADRAPONG: (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
}


# Default render modes for PettingZoo environments
PETTINGZOO_RENDER_MODES: Dict[PettingZooEnvId, RenderMode] = {
    env_id: RenderMode.RGB_ARRAY for env_id in PettingZooEnvId
}


# Game Type Classification
class PettingZooGameType(StrEnum):
    """Classification of multi-agent game types."""

    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    MIXED = "mixed"  # Can be played both ways


# Map environments to their game type
PETTINGZOO_GAME_TYPES: Dict[PettingZooEnvId, PettingZooGameType] = {
    # Cooperative environments
    PettingZooEnvId.HANABI: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.KNIGHTS_ARCHERS_ZOMBIES: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.PISTONBALL: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.COOPERATIVE_PONG: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SIMPLE_SPREAD: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SIMPLE_CRYPTO: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SIMPLE_REFERENCE: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SIMPLE_SPEAKER_LISTENER: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SIMPLE_WORLD_COMM: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.MULTIWALKER: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.WATERWORLD: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.PURSUIT: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.SPACE_INVADERS: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.ENTOMBED_COOPERATIVE: PettingZooGameType.COOPERATIVE,
    PettingZooEnvId.MARIO_BROS: PettingZooGameType.COOPERATIVE,

    # Competitive environments
    PettingZooEnvId.CHESS: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.GO: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.CONNECT_FOUR: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.TIC_TAC_TOE: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.RPS: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.BACKGAMMON: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.TEXAS_HOLDEM: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.TEXAS_HOLDEM_NO_LIMIT: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.GIN_RUMMY: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.LEDUC_HOLDEM: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.SIMPLE_ADVERSARY: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.SIMPLE_TAG: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.SIMPLE_PUSH: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.PONG: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.BOXING: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.TENNIS: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.DOUBLE_DUNK: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.ICE_HOCKEY: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.SURROUND: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.VIDEO_CHECKERS: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.OTHELLO: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.COMBAT_PLANE: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.COMBAT_TANK: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.WARLORDS: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.ENTOMBED_COMPETITIVE: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.FLAG_CAPTURE: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.SPACE_WAR: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.MAZE_CRAZE: PettingZooGameType.COMPETITIVE,
    PettingZooEnvId.QUADRAPONG: PettingZooGameType.COMPETITIVE,

    # Mixed (can be played competitively or cooperatively)
    PettingZooEnvId.JOUST: PettingZooGameType.MIXED,
    PettingZooEnvId.WIZARD_OF_WOR: PettingZooGameType.MIXED,
}


def get_game_type(env_id: PettingZooEnvId) -> PettingZooGameType:
    """Get the game type (cooperative, competitive, or mixed) for an environment."""
    return PETTINGZOO_GAME_TYPES.get(env_id, PettingZooGameType.COMPETITIVE)


def get_cooperative_envs() -> List[PettingZooEnvId]:
    """Get all cooperative environment IDs."""
    return [
        env_id for env_id, game_type in PETTINGZOO_GAME_TYPES.items()
        if game_type == PettingZooGameType.COOPERATIVE
    ]


def get_competitive_envs() -> List[PettingZooEnvId]:
    """Get all competitive environment IDs."""
    return [
        env_id for env_id, game_type in PETTINGZOO_GAME_TYPES.items()
        if game_type == PettingZooGameType.COMPETITIVE
    ]


def get_human_vs_agent_envs() -> List[PettingZooEnvId]:
    """Get environments that support human vs agent play (AEC turn-based games)."""
    return list(HUMAN_CONTROLLABLE_ENVS)


__all__ = [
    "PettingZooFamily",
    "PettingZooAPIType",
    "PettingZooEnvId",
    "PettingZooGameType",
    "PETTINGZOO_ENV_METADATA",
    "PETTINGZOO_CONTROL_MODES",
    "PETTINGZOO_RENDER_MODES",
    "PETTINGZOO_GAME_TYPES",
    "HUMAN_CONTROLLABLE_ENVS",
    "get_envs_by_family",
    "get_api_type",
    "get_display_name",
    "get_description",
    "get_game_type",
    "is_aec_env",
    "is_parallel_env",
    "get_cooperative_envs",
    "get_competitive_envs",
    "get_human_vs_agent_envs",
]
