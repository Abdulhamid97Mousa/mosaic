"""Environment adapter base classes and concrete implementations."""

from .base import (
    AdapterContext,
    AdapterNotReadyError,
    AdapterStep,
    EnvironmentAdapter,
    UnsupportedModeError,
    WorkerCapabilities,
)
from .paradigm import (
    ParadigmAdapter,
    ParadigmStepResult,
    SingleAgentParadigmAdapter,
    SequentialParadigmAdapter,
    SimultaneousParadigmAdapter,
    HierarchicalParadigmAdapter,
    create_paradigm_adapter,
)
from .toy_text import (
    FrozenLakeAdapter,
    FrozenLakeV2Adapter,
    CliffWalkingAdapter,
    TaxiAdapter,
    TOY_TEXT_ADAPTERS,
)
from .box2d import (
    Box2DAdapter,
    LunarLanderAdapter,
    CarRacingAdapter,
    BipedalWalkerAdapter,
    BOX2D_ADAPTERS,
)
from .minigrid import (
    MiniGridAdapter,
    MiniGridEmpty5x5Adapter,
    MiniGridEmptyRandom5x5Adapter,
    MiniGridEmpty6x6Adapter,
    MiniGridEmptyRandom6x6Adapter,
    MiniGridEmpty8x8Adapter,
    MiniGridEmpty16x16Adapter,
    MiniGridDoorKey5x5Adapter,
    MiniGridDoorKey6x6Adapter,
    MiniGridDoorKey8x8Adapter,
    MiniGridDoorKey16x16Adapter,
    MiniGridLavaGapS5Adapter,
    MiniGridLavaGapS6Adapter,
    MiniGridLavaGapS7Adapter,
    MiniGridLavaCrossingS9N1Adapter,
    MiniGridLavaCrossingS9N2Adapter,
    MiniGridLavaCrossingS9N3Adapter,
    MiniGridLavaCrossingS11N5Adapter,
    MiniGridSimpleCrossingS9N1Adapter,
    MiniGridSimpleCrossingS9N2Adapter,
    MiniGridSimpleCrossingS9N3Adapter,
    MiniGridSimpleCrossingS11N5Adapter,
    MINIGRID_ADAPTERS,
)
from .ale import (
    ALEAdapter,
    AdventureV4Adapter,
    AdventureV5Adapter,
    AirRaidV4Adapter,
    AirRaidV5Adapter,
    AssaultV4Adapter,
    AssaultV5Adapter,
    ALE_ADAPTERS,
)

try:  # Optional dependency
    from .vizdoom import (  # pragma: no cover - exercised in integration
        ViZDoomAdapter,
        ViZDoomBasicAdapter,
        ViZDoomConfig,
        ViZDoomDeadlyCorridorAdapter,
        ViZDoomDefendTheCenterAdapter,
        ViZDoomDefendTheLineAdapter,
        ViZDoomDeathmatchAdapter,
        ViZDoomHealthGatheringAdapter,
        ViZDoomHealthGatheringSupremeAdapter,
        ViZDoomMyWayHomeAdapter,
        ViZDoomPredictPositionAdapter,
        ViZDoomTakeCoverAdapter,
        VIZDOOM_ADAPTERS,
    )
    _VIZDOOM_AVAILABLE = True
except Exception:  # pragma: no cover - vizdoom optional
    ViZDoomAdapter = None  # type: ignore
    ViZDoomBasicAdapter = None  # type: ignore
    ViZDoomConfig = None  # type: ignore
    ViZDoomDeadlyCorridorAdapter = None  # type: ignore
    ViZDoomDefendTheCenterAdapter = None  # type: ignore
    ViZDoomDefendTheLineAdapter = None  # type: ignore
    ViZDoomDeathmatchAdapter = None  # type: ignore
    ViZDoomHealthGatheringAdapter = None  # type: ignore
    ViZDoomHealthGatheringSupremeAdapter = None  # type: ignore
    ViZDoomMyWayHomeAdapter = None  # type: ignore
    ViZDoomPredictPositionAdapter = None  # type: ignore
    ViZDoomTakeCoverAdapter = None  # type: ignore
    VIZDOOM_ADAPTERS = {}
    _VIZDOOM_AVAILABLE = False

try:  # Optional dependency - PettingZoo multi-agent environments
    from .pettingzoo import (  # pragma: no cover - pettingzoo optional
        PettingZooAdapter,
        PettingZooConfig,
        ChessAdapter,
        ConnectFourAdapter,
        TicTacToeAdapter,
        GoAdapter,
        SimpleSpreadAdapter,
        SimpleTagAdapter,
        PistonballAdapter,
        MultiwalkerAdapter,
        PETTINGZOO_ADAPTERS,
        create_pettingzoo_adapter,
    )
    _PETTINGZOO_AVAILABLE = True
except Exception:  # pragma: no cover - pettingzoo optional
    PettingZooAdapter = None  # type: ignore
    PettingZooConfig = None  # type: ignore
    ChessAdapter = None  # type: ignore
    ConnectFourAdapter = None  # type: ignore
    TicTacToeAdapter = None  # type: ignore
    GoAdapter = None  # type: ignore
    SimpleSpreadAdapter = None  # type: ignore
    SimpleTagAdapter = None  # type: ignore
    PistonballAdapter = None  # type: ignore
    MultiwalkerAdapter = None  # type: ignore
    PETTINGZOO_ADAPTERS = {}
    create_pettingzoo_adapter = None  # type: ignore
    _PETTINGZOO_AVAILABLE = False

try:  # Optional dependency - MiniHack (sandbox RL on NLE)
    from .minihack import (  # pragma: no cover - minihack optional
        MiniHackAdapter,
        MiniHackConfig,
        MiniHackRoom5x5Adapter,
        MiniHackRoom15x15Adapter,
        MiniHackCorridorR2Adapter,
        MiniHackCorridorR3Adapter,
        MiniHackCorridorR5Adapter,
        MiniHackMazeWalk9x9Adapter,
        MiniHackMazeWalk15x15Adapter,
        MiniHackMazeWalk45x19Adapter,
        MiniHackRiverAdapter,
        MiniHackRiverNarrowAdapter,
        MiniHackEatAdapter,
        MiniHackWearAdapter,
        MiniHackWieldAdapter,
        MiniHackZapAdapter,
        MiniHackReadAdapter,
        MiniHackQuaffAdapter,
        MiniHackPutOnAdapter,
        MiniHackLavaCrossAdapter,
        MiniHackWoDEasyAdapter,
        MiniHackWoDMediumAdapter,
        MiniHackWoDHardAdapter,
        MiniHackExploreMazeEasyAdapter,
        MiniHackExploreMazeHardAdapter,
        MiniHackHideNSeekAdapter,
        MiniHackMementoF2Adapter,
        MiniHackMementoF4Adapter,
        MINIHACK_ADAPTERS,
    )
    _MINIHACK_AVAILABLE = True
except Exception:  # pragma: no cover - minihack optional
    MiniHackAdapter = None  # type: ignore
    MiniHackConfig = None  # type: ignore
    MiniHackRoom5x5Adapter = None  # type: ignore
    MiniHackRoom15x15Adapter = None  # type: ignore
    MiniHackCorridorR2Adapter = None  # type: ignore
    MiniHackCorridorR3Adapter = None  # type: ignore
    MiniHackCorridorR5Adapter = None  # type: ignore
    MiniHackMazeWalk9x9Adapter = None  # type: ignore
    MiniHackMazeWalk15x15Adapter = None  # type: ignore
    MiniHackMazeWalk45x19Adapter = None  # type: ignore
    MiniHackRiverAdapter = None  # type: ignore
    MiniHackRiverNarrowAdapter = None  # type: ignore
    MiniHackEatAdapter = None  # type: ignore
    MiniHackWearAdapter = None  # type: ignore
    MiniHackWieldAdapter = None  # type: ignore
    MiniHackZapAdapter = None  # type: ignore
    MiniHackReadAdapter = None  # type: ignore
    MiniHackQuaffAdapter = None  # type: ignore
    MiniHackPutOnAdapter = None  # type: ignore
    MiniHackLavaCrossAdapter = None  # type: ignore
    MiniHackWoDEasyAdapter = None  # type: ignore
    MiniHackWoDMediumAdapter = None  # type: ignore
    MiniHackWoDHardAdapter = None  # type: ignore
    MiniHackExploreMazeEasyAdapter = None  # type: ignore
    MiniHackExploreMazeHardAdapter = None  # type: ignore
    MiniHackHideNSeekAdapter = None  # type: ignore
    MiniHackMementoF2Adapter = None  # type: ignore
    MiniHackMementoF4Adapter = None  # type: ignore
    MINIHACK_ADAPTERS = {}
    _MINIHACK_AVAILABLE = False

try:  # Optional dependency - NetHack (full game via NLE)
    from .nethack import (  # pragma: no cover - nethack optional
        NetHackAdapter,
        NetHackConfig,
        NetHackChallengeAdapter,
        NetHackScoreAdapter,
        NetHackStaircaseAdapter,
        NetHackStaircasePetAdapter,
        NetHackOracleAdapter,
        NetHackGoldAdapter,
        NetHackEatAdapter as NetHackEatTaskAdapter,  # Avoid conflict with MiniHack
        NetHackScoutAdapter,
        NETHACK_ADAPTERS,
    )
    _NETHACK_AVAILABLE = True
except Exception:  # pragma: no cover - nethack optional
    NetHackAdapter = None  # type: ignore
    NetHackConfig = None  # type: ignore
    NetHackChallengeAdapter = None  # type: ignore
    NetHackScoreAdapter = None  # type: ignore
    NetHackStaircaseAdapter = None  # type: ignore
    NetHackStaircasePetAdapter = None  # type: ignore
    NetHackOracleAdapter = None  # type: ignore
    NetHackGoldAdapter = None  # type: ignore
    NetHackEatTaskAdapter = None  # type: ignore
    NetHackScoutAdapter = None  # type: ignore
    NETHACK_ADAPTERS = {}
    _NETHACK_AVAILABLE = False

try:  # Optional dependency - Crafter (open world survival benchmark)
    from .crafter import (  # pragma: no cover - crafter optional
        CrafterAdapter,
        CrafterRewardAdapter,
        CrafterNoRewardAdapter,
        CRAFTER_ADAPTERS,
        CRAFTER_ACHIEVEMENTS,
        CRAFTER_ACTIONS,
    )
    _CRAFTER_AVAILABLE = True
except Exception:  # pragma: no cover - crafter optional
    CrafterAdapter = None  # type: ignore
    CrafterRewardAdapter = None  # type: ignore
    CrafterNoRewardAdapter = None  # type: ignore
    CRAFTER_ADAPTERS = {}
    CRAFTER_ACHIEVEMENTS = []  # type: ignore
    CRAFTER_ACTIONS = []  # type: ignore
    _CRAFTER_AVAILABLE = False

# Standalone adapters for Human vs Agent mode (simplified state-based interfaces)
from .chess_adapter import ChessAdapter as ChessHvAAdapter, ChessState
from .connect_four_adapter import ConnectFourAdapter as ConnectFourHvAAdapter, ConnectFourState
from .go_adapter import GoAdapter as GoHvAAdapter, GoState
from .tictactoe_adapter import TicTacToeAdapter as TicTacToeHvAAdapter, TicTacToeState

__all__ = [
    # Base classes
    "AdapterContext",
    "AdapterNotReadyError",
    "AdapterStep",
    "EnvironmentAdapter",
    "UnsupportedModeError",
    "WorkerCapabilities",
    # Paradigm adapters
    "ParadigmAdapter",
    "ParadigmStepResult",
    "SingleAgentParadigmAdapter",
    "SequentialParadigmAdapter",
    "SimultaneousParadigmAdapter",
    "HierarchicalParadigmAdapter",
    "create_paradigm_adapter",
    # Toy text adapters
    "FrozenLakeAdapter",
    "FrozenLakeV2Adapter",
    "CliffWalkingAdapter",
    "TaxiAdapter",
    "TOY_TEXT_ADAPTERS",
    "Box2DAdapter",
    "LunarLanderAdapter",
    "CarRacingAdapter",
    "BipedalWalkerAdapter",
    "BOX2D_ADAPTERS",
    "MiniGridAdapter",
    "MiniGridEmpty5x5Adapter",
    "MiniGridEmptyRandom5x5Adapter",
    "MiniGridEmpty6x6Adapter",
    "MiniGridEmptyRandom6x6Adapter",
    "MiniGridEmpty8x8Adapter",
    "MiniGridEmpty16x16Adapter",
    "MiniGridDoorKey5x5Adapter",
    "MiniGridDoorKey6x6Adapter",
    "MiniGridDoorKey8x8Adapter",
    "MiniGridDoorKey16x16Adapter",
    "MiniGridLavaGapS5Adapter",
    "MiniGridLavaGapS6Adapter",
    "MiniGridLavaGapS7Adapter",
    "MiniGridLavaCrossingS9N1Adapter",
    "MiniGridLavaCrossingS9N2Adapter",
    "MiniGridLavaCrossingS9N3Adapter",
    "MiniGridLavaCrossingS11N5Adapter",
    "MiniGridSimpleCrossingS9N1Adapter",
    "MiniGridSimpleCrossingS9N2Adapter",
    "MiniGridSimpleCrossingS9N3Adapter",
    "MiniGridSimpleCrossingS11N5Adapter",
    "MINIGRID_ADAPTERS",
    "ALEAdapter",
    "AdventureV4Adapter",
    "AdventureV5Adapter",
    "AirRaidV4Adapter",
    "AirRaidV5Adapter",
    "AssaultV4Adapter",
    "AssaultV5Adapter",
    "ALE_ADAPTERS",
]

if _VIZDOOM_AVAILABLE:
    __all__ += [
        "ViZDoomAdapter",
        "ViZDoomConfig",
        "ViZDoomBasicAdapter",
        "ViZDoomDeadlyCorridorAdapter",
        "ViZDoomDefendTheCenterAdapter",
        "ViZDoomDefendTheLineAdapter",
        "ViZDoomDeathmatchAdapter",
        "ViZDoomHealthGatheringAdapter",
        "ViZDoomHealthGatheringSupremeAdapter",
        "ViZDoomMyWayHomeAdapter",
        "ViZDoomPredictPositionAdapter",
        "ViZDoomTakeCoverAdapter",
        "VIZDOOM_ADAPTERS",
    ]

if _PETTINGZOO_AVAILABLE:
    __all__ += [
        "PettingZooAdapter",
        "PettingZooConfig",
        "ChessAdapter",
        "ConnectFourAdapter",
        "TicTacToeAdapter",
        "GoAdapter",
        "SimpleSpreadAdapter",
        "SimpleTagAdapter",
        "PistonballAdapter",
        "MultiwalkerAdapter",
        "PETTINGZOO_ADAPTERS",
        "create_pettingzoo_adapter",
    ]

if _MINIHACK_AVAILABLE:
    __all__ += [
        "MiniHackAdapter",
        "MiniHackConfig",
        "MiniHackRoom5x5Adapter",
        "MiniHackRoom15x15Adapter",
        "MiniHackCorridorR2Adapter",
        "MiniHackCorridorR3Adapter",
        "MiniHackCorridorR5Adapter",
        "MiniHackMazeWalk9x9Adapter",
        "MiniHackMazeWalk15x15Adapter",
        "MiniHackMazeWalk45x19Adapter",
        "MiniHackRiverAdapter",
        "MiniHackRiverNarrowAdapter",
        "MiniHackEatAdapter",
        "MiniHackWearAdapter",
        "MiniHackWieldAdapter",
        "MiniHackZapAdapter",
        "MiniHackReadAdapter",
        "MiniHackQuaffAdapter",
        "MiniHackPutOnAdapter",
        "MiniHackLavaCrossAdapter",
        "MiniHackWoDEasyAdapter",
        "MiniHackWoDMediumAdapter",
        "MiniHackWoDHardAdapter",
        "MiniHackExploreMazeEasyAdapter",
        "MiniHackExploreMazeHardAdapter",
        "MiniHackHideNSeekAdapter",
        "MiniHackMementoF2Adapter",
        "MiniHackMementoF4Adapter",
        "MINIHACK_ADAPTERS",
    ]

if _NETHACK_AVAILABLE:
    __all__ += [
        "NetHackAdapter",
        "NetHackConfig",
        "NetHackChallengeAdapter",
        "NetHackScoreAdapter",
        "NetHackStaircaseAdapter",
        "NetHackStaircasePetAdapter",
        "NetHackOracleAdapter",
        "NetHackGoldAdapter",
        "NetHackEatTaskAdapter",
        "NetHackScoutAdapter",
        "NETHACK_ADAPTERS",
    ]

if _CRAFTER_AVAILABLE:
    __all__ += [
        "CrafterAdapter",
        "CrafterRewardAdapter",
        "CrafterNoRewardAdapter",
        "CRAFTER_ADAPTERS",
        "CRAFTER_ACHIEVEMENTS",
        "CRAFTER_ACTIONS",
    ]

# Human vs Agent mode adapters (always available)
__all__ += [
    "ChessHvAAdapter",
    "ChessState",
    "ConnectFourHvAAdapter",
    "ConnectFourState",
    "GoHvAAdapter",
    "GoState",
    "TicTacToeHvAAdapter",
    "TicTacToeState",
]
