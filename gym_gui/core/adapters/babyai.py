"""BabyAI environment adapters providing language-grounded instruction following.

BabyAI environments extend MiniGrid with natural language mission instructions.
The agent must interpret language commands like "go to the red ball" or
"pick up the key after you open the door" and execute multi-step plans.

Since BabyAI shares the same observation/action space as MiniGrid, these adapters
inherit from MiniGridAdapter and simply specialize the default environment ID.
"""

from __future__ import annotations

from gym_gui.core.adapters.minigrid import MiniGridAdapter, MINIGRID_ACTIONS
from gym_gui.core.enums import GameId

# BabyAI uses the same action space as MiniGrid
BABYAI_ACTIONS = MINIGRID_ACTIONS


# -----------------------------------------------------------------------------
# GoTo family - Navigate to objects
# -----------------------------------------------------------------------------


class BabyAIGoToRedBallGreyAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToRedBallGrey-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_REDBALL_GREY.value


class BabyAIGoToRedBallAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToRedBall-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_REDBALL.value


class BabyAIGoToRedBallNoDistsAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToRedBallNoDists-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_REDBALL_NODISTS.value


class BabyAIGoToObjAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToObj-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_OBJ.value


class BabyAIGoToLocalAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToLocal-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_LOCAL.value


class BabyAIGoToAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoTo-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO.value


class BabyAIGoToImpUnlockAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToImpUnlock-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_IMPUNLOCK.value


class BabyAIGoToSeqAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToSeq-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_SEQ.value


class BabyAIGoToRedBlueBallAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToRedBlueBall-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_REDBLUEBALL.value


class BabyAIGoToDoorAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToDoor-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_DOOR.value


class BabyAIGoToObjDoorAdapter(MiniGridAdapter):
    """Adapter for BabyAI-GoToObjDoor-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_GOTO_OBJDOOR.value


# -----------------------------------------------------------------------------
# Open family - Open doors
# -----------------------------------------------------------------------------


class BabyAIOpenAdapter(MiniGridAdapter):
    """Adapter for BabyAI-Open-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN.value


class BabyAIOpenRedDoorAdapter(MiniGridAdapter):
    """Adapter for BabyAI-OpenRedDoor-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN_REDDOOR.value


class BabyAIOpenDoorAdapter(MiniGridAdapter):
    """Adapter for BabyAI-OpenDoor-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN_DOOR.value


class BabyAIOpenTwoDoorsAdapter(MiniGridAdapter):
    """Adapter for BabyAI-OpenTwoDoors-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN_TWODOORS.value


class BabyAIOpenDoorsOrderN2Adapter(MiniGridAdapter):
    """Adapter for BabyAI-OpenDoorsOrderN2-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN_DOORSORDER_N2.value


class BabyAIOpenDoorsOrderN4Adapter(MiniGridAdapter):
    """Adapter for BabyAI-OpenDoorsOrderN4-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_OPEN_DOORSORDER_N4.value


# -----------------------------------------------------------------------------
# Pickup family - Pick up objects
# -----------------------------------------------------------------------------


class BabyAIPickupAdapter(MiniGridAdapter):
    """Adapter for BabyAI-Pickup-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PICKUP.value


class BabyAIUnblockPickupAdapter(MiniGridAdapter):
    """Adapter for BabyAI-UnblockPickup-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_UNBLOCK_PICKUP.value


class BabyAIPickupLocAdapter(MiniGridAdapter):
    """Adapter for BabyAI-PickupLoc-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PICKUP_LOC.value


class BabyAIPickupDistAdapter(MiniGridAdapter):
    """Adapter for BabyAI-PickupDist-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PICKUP_DIST.value


class BabyAIPickupAboveAdapter(MiniGridAdapter):
    """Adapter for BabyAI-PickupAbove-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PICKUP_ABOVE.value


# -----------------------------------------------------------------------------
# Unlock family - Unlock doors with keys
# -----------------------------------------------------------------------------


class BabyAIUnlockAdapter(MiniGridAdapter):
    """Adapter for BabyAI-Unlock-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_UNLOCK.value


class BabyAIUnlockLocalAdapter(MiniGridAdapter):
    """Adapter for BabyAI-UnlockLocal-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_UNLOCK_LOCAL.value


class BabyAIKeyInBoxAdapter(MiniGridAdapter):
    """Adapter for BabyAI-KeyInBox-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_KEY_INBOX.value


class BabyAIUnlockPickupAdapter(MiniGridAdapter):
    """Adapter for BabyAI-UnlockPickup-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_UNLOCK_PICKUP.value


class BabyAIBlockedUnlockPickupAdapter(MiniGridAdapter):
    """Adapter for BabyAI-BlockedUnlockPickup-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_BLOCKED_UNLOCK_PICKUP.value


class BabyAIUnlockToUnlockAdapter(MiniGridAdapter):
    """Adapter for BabyAI-UnlockToUnlock-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_UNLOCK_TO_UNLOCK.value


# -----------------------------------------------------------------------------
# PutNext family - Put objects next to other objects
# -----------------------------------------------------------------------------


class BabyAIPutNextLocalAdapter(MiniGridAdapter):
    """Adapter for BabyAI-PutNextLocal-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PUTNEXT_LOCAL.value


class BabyAIPutNextAdapter(MiniGridAdapter):
    """Adapter for BabyAI-PutNext-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_PUTNEXT.value


# -----------------------------------------------------------------------------
# Complex environments - Multi-step reasoning
# -----------------------------------------------------------------------------


class BabyAIActionObjDoorAdapter(MiniGridAdapter):
    """Adapter for BabyAI-ActionObjDoor-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_ACTION_OBJDOOR.value


class BabyAIFindObjS5Adapter(MiniGridAdapter):
    """Adapter for BabyAI-FindObjS5-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_FINDOBJ_S5.value


class BabyAIKeyCorridorS3R1Adapter(MiniGridAdapter):
    """Adapter for BabyAI-KeyCorridorS3R1-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_KEYCORRIDOR_S3R1.value


class BabyAIKeyCorridorS3R2Adapter(MiniGridAdapter):
    """Adapter for BabyAI-KeyCorridorS3R2-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_KEYCORRIDOR_S3R2.value


class BabyAIKeyCorridorS3R3Adapter(MiniGridAdapter):
    """Adapter for BabyAI-KeyCorridorS3R3-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_KEYCORRIDOR_S3R3.value


class BabyAIOneRoomS8Adapter(MiniGridAdapter):
    """Adapter for BabyAI-OneRoomS8-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_ONEROOM_S8.value


class BabyAIMoveTwoAcrossS8N9Adapter(MiniGridAdapter):
    """Adapter for BabyAI-MoveTwoAcrossS8N9-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_MOVETWOACROSS_S8N9.value


class BabyAISynthAdapter(MiniGridAdapter):
    """Adapter for BabyAI-Synth-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_SYNTH.value


class BabyAISynthLocAdapter(MiniGridAdapter):
    """Adapter for BabyAI-SynthLoc-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_SYNTHLOC.value


class BabyAISynthSeqAdapter(MiniGridAdapter):
    """Adapter for BabyAI-SynthSeq-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_SYNTHSEQ.value


class BabyAIMiniBossLevelAdapter(MiniGridAdapter):
    """Adapter for BabyAI-MiniBossLevel-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_MINIBOSSLEVEL.value


class BabyAIBossLevelAdapter(MiniGridAdapter):
    """Adapter for BabyAI-BossLevel-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_BOSSLEVEL.value


class BabyAIBossLevelNoUnlockAdapter(MiniGridAdapter):
    """Adapter for BabyAI-BossLevelNoUnlock-v0."""

    DEFAULT_ENV_ID = GameId.BABYAI_BOSSLEVEL_NOUNLOCK.value


# -----------------------------------------------------------------------------
# Adapter registry
# -----------------------------------------------------------------------------


BABYAI_ADAPTERS: dict[GameId, type[MiniGridAdapter]] = {
    # GoTo family
    GameId.BABYAI_GOTO_REDBALL_GREY: BabyAIGoToRedBallGreyAdapter,
    GameId.BABYAI_GOTO_REDBALL: BabyAIGoToRedBallAdapter,
    GameId.BABYAI_GOTO_REDBALL_NODISTS: BabyAIGoToRedBallNoDistsAdapter,
    GameId.BABYAI_GOTO_OBJ: BabyAIGoToObjAdapter,
    GameId.BABYAI_GOTO_LOCAL: BabyAIGoToLocalAdapter,
    GameId.BABYAI_GOTO: BabyAIGoToAdapter,
    GameId.BABYAI_GOTO_IMPUNLOCK: BabyAIGoToImpUnlockAdapter,
    GameId.BABYAI_GOTO_SEQ: BabyAIGoToSeqAdapter,
    GameId.BABYAI_GOTO_REDBLUEBALL: BabyAIGoToRedBlueBallAdapter,
    GameId.BABYAI_GOTO_DOOR: BabyAIGoToDoorAdapter,
    GameId.BABYAI_GOTO_OBJDOOR: BabyAIGoToObjDoorAdapter,
    # Open family
    GameId.BABYAI_OPEN: BabyAIOpenAdapter,
    GameId.BABYAI_OPEN_REDDOOR: BabyAIOpenRedDoorAdapter,
    GameId.BABYAI_OPEN_DOOR: BabyAIOpenDoorAdapter,
    GameId.BABYAI_OPEN_TWODOORS: BabyAIOpenTwoDoorsAdapter,
    GameId.BABYAI_OPEN_DOORSORDER_N2: BabyAIOpenDoorsOrderN2Adapter,
    GameId.BABYAI_OPEN_DOORSORDER_N4: BabyAIOpenDoorsOrderN4Adapter,
    # Pickup family
    GameId.BABYAI_PICKUP: BabyAIPickupAdapter,
    GameId.BABYAI_UNBLOCK_PICKUP: BabyAIUnblockPickupAdapter,
    GameId.BABYAI_PICKUP_LOC: BabyAIPickupLocAdapter,
    GameId.BABYAI_PICKUP_DIST: BabyAIPickupDistAdapter,
    GameId.BABYAI_PICKUP_ABOVE: BabyAIPickupAboveAdapter,
    # Unlock family
    GameId.BABYAI_UNLOCK: BabyAIUnlockAdapter,
    GameId.BABYAI_UNLOCK_LOCAL: BabyAIUnlockLocalAdapter,
    GameId.BABYAI_KEY_INBOX: BabyAIKeyInBoxAdapter,
    GameId.BABYAI_UNLOCK_PICKUP: BabyAIUnlockPickupAdapter,
    GameId.BABYAI_BLOCKED_UNLOCK_PICKUP: BabyAIBlockedUnlockPickupAdapter,
    GameId.BABYAI_UNLOCK_TO_UNLOCK: BabyAIUnlockToUnlockAdapter,
    # PutNext family
    GameId.BABYAI_PUTNEXT_LOCAL: BabyAIPutNextLocalAdapter,
    GameId.BABYAI_PUTNEXT: BabyAIPutNextAdapter,
    # Complex environments
    GameId.BABYAI_ACTION_OBJDOOR: BabyAIActionObjDoorAdapter,
    GameId.BABYAI_FINDOBJ_S5: BabyAIFindObjS5Adapter,
    GameId.BABYAI_KEYCORRIDOR_S3R1: BabyAIKeyCorridorS3R1Adapter,
    GameId.BABYAI_KEYCORRIDOR_S3R2: BabyAIKeyCorridorS3R2Adapter,
    GameId.BABYAI_KEYCORRIDOR_S3R3: BabyAIKeyCorridorS3R3Adapter,
    GameId.BABYAI_ONEROOM_S8: BabyAIOneRoomS8Adapter,
    GameId.BABYAI_MOVETWOACROSS_S8N9: BabyAIMoveTwoAcrossS8N9Adapter,
    GameId.BABYAI_SYNTH: BabyAISynthAdapter,
    GameId.BABYAI_SYNTHLOC: BabyAISynthLocAdapter,
    GameId.BABYAI_SYNTHSEQ: BabyAISynthSeqAdapter,
    GameId.BABYAI_MINIBOSSLEVEL: BabyAIMiniBossLevelAdapter,
    GameId.BABYAI_BOSSLEVEL: BabyAIBossLevelAdapter,
    GameId.BABYAI_BOSSLEVEL_NOUNLOCK: BabyAIBossLevelNoUnlockAdapter,
}


__all__ = [
    "BABYAI_ACTIONS",
    # GoTo family
    "BabyAIGoToRedBallGreyAdapter",
    "BabyAIGoToRedBallAdapter",
    "BabyAIGoToRedBallNoDistsAdapter",
    "BabyAIGoToObjAdapter",
    "BabyAIGoToLocalAdapter",
    "BabyAIGoToAdapter",
    "BabyAIGoToImpUnlockAdapter",
    "BabyAIGoToSeqAdapter",
    "BabyAIGoToRedBlueBallAdapter",
    "BabyAIGoToDoorAdapter",
    "BabyAIGoToObjDoorAdapter",
    # Open family
    "BabyAIOpenAdapter",
    "BabyAIOpenRedDoorAdapter",
    "BabyAIOpenDoorAdapter",
    "BabyAIOpenTwoDoorsAdapter",
    "BabyAIOpenDoorsOrderN2Adapter",
    "BabyAIOpenDoorsOrderN4Adapter",
    # Pickup family
    "BabyAIPickupAdapter",
    "BabyAIUnblockPickupAdapter",
    "BabyAIPickupLocAdapter",
    "BabyAIPickupDistAdapter",
    "BabyAIPickupAboveAdapter",
    # Unlock family
    "BabyAIUnlockAdapter",
    "BabyAIUnlockLocalAdapter",
    "BabyAIKeyInBoxAdapter",
    "BabyAIUnlockPickupAdapter",
    "BabyAIBlockedUnlockPickupAdapter",
    "BabyAIUnlockToUnlockAdapter",
    # PutNext family
    "BabyAIPutNextLocalAdapter",
    "BabyAIPutNextAdapter",
    # Complex environments
    "BabyAIActionObjDoorAdapter",
    "BabyAIFindObjS5Adapter",
    "BabyAIKeyCorridorS3R1Adapter",
    "BabyAIKeyCorridorS3R2Adapter",
    "BabyAIKeyCorridorS3R3Adapter",
    "BabyAIOneRoomS8Adapter",
    "BabyAIMoveTwoAcrossS8N9Adapter",
    "BabyAISynthAdapter",
    "BabyAISynthLocAdapter",
    "BabyAISynthSeqAdapter",
    "BabyAIMiniBossLevelAdapter",
    "BabyAIBossLevelAdapter",
    "BabyAIBossLevelNoUnlockAdapter",
    # Registry
    "BABYAI_ADAPTERS",
]
