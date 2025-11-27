"""Environment adapter base classes and concrete implementations."""

from .base import (
    AdapterContext,
    AdapterNotReadyError,
    AdapterStep,
    EnvironmentAdapter,
    UnsupportedModeError,
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

__all__ = [
    "AdapterContext",
    "AdapterNotReadyError",
    "AdapterStep",
    "EnvironmentAdapter",
    "UnsupportedModeError",
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
