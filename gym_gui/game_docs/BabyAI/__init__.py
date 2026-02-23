"""BabyAI game documentation module.

BabyAI environments are language-grounded instruction following tasks
built on top of MiniGrid. They test agents on navigation, object manipulation,
and multi-step reasoning from natural language instructions.
"""
from __future__ import annotations

# GoTo family
from .GoToRedBallGrey import BABYAI_GOTO_REDBALL_GREY_HTML, get_goto_redball_grey_html
from .GoToRedBall import BABYAI_GOTO_REDBALL_HTML, get_goto_redball_html
from .GoToRedBallNoDists import BABYAI_GOTO_REDBALL_NODISTS_HTML, get_goto_redball_nodists_html
from .GoToObj import BABYAI_GOTO_OBJ_HTML, get_goto_obj_html
from .GoToLocal import BABYAI_GOTO_LOCAL_HTML, get_goto_local_html
from .GoTo import BABYAI_GOTO_HTML, get_goto_html
from .GoToImpUnlock import BABYAI_GOTO_IMPUNLOCK_HTML, get_goto_impunlock_html
from .GoToSeq import BABYAI_GOTO_SEQ_HTML, get_goto_seq_html
from .GoToRedBlueBall import BABYAI_GOTO_REDBLUEBALL_HTML, get_goto_redblueball_html
from .GoToDoor import BABYAI_GOTO_DOOR_HTML, get_goto_door_html
from .GoToObjDoor import BABYAI_GOTO_OBJDOOR_HTML, get_goto_objdoor_html

# Open family
from .Open import BABYAI_OPEN_HTML, get_open_html
from .OpenRedDoor import BABYAI_OPEN_REDDOOR_HTML, get_open_reddoor_html
from .OpenDoor import BABYAI_OPEN_DOOR_HTML, get_open_door_html
from .OpenTwoDoors import BABYAI_OPEN_TWODOORS_HTML, get_open_twodoors_html
from .OpenDoorsOrder import BABYAI_OPEN_DOORSORDER_HTML, get_open_doorsorder_html

# Pickup family
from .Pickup import BABYAI_PICKUP_HTML, get_pickup_html
from .UnblockPickup import BABYAI_UNBLOCK_PICKUP_HTML, get_unblock_pickup_html
from .PickupLoc import BABYAI_PICKUP_LOC_HTML, get_pickup_loc_html
from .PickupDist import BABYAI_PICKUP_DIST_HTML, get_pickup_dist_html
from .PickupAbove import BABYAI_PICKUP_ABOVE_HTML, get_pickup_above_html

# Unlock family
from .Unlock import BABYAI_UNLOCK_HTML, get_unlock_html
from .UnlockLocal import BABYAI_UNLOCK_LOCAL_HTML, get_unlock_local_html
from .KeyInBox import BABYAI_KEY_INBOX_HTML, get_key_inbox_html
from .UnlockPickup import BABYAI_UNLOCK_PICKUP_HTML, get_unlock_pickup_html
from .BlockedUnlockPickup import BABYAI_BLOCKED_UNLOCK_PICKUP_HTML, get_blocked_unlock_pickup_html
from .UnlockToUnlock import BABYAI_UNLOCK_TO_UNLOCK_HTML, get_unlock_to_unlock_html

# PutNext family
from .PutNextLocal import BABYAI_PUTNEXT_LOCAL_HTML, get_putnext_local_html
from .PutNext import BABYAI_PUTNEXT_HTML, get_putnext_html

# Complex environments
from .ActionObjDoor import BABYAI_ACTION_OBJDOOR_HTML, get_action_objdoor_html
from .FindObjS5 import BABYAI_FINDOBJ_HTML, get_findobj_html
from .KeyCorridor import BABYAI_KEYCORRIDOR_HTML, get_keycorridor_html
from .OneRoomS8 import BABYAI_ONEROOM_HTML, get_oneroom_html
from .MoveTwoAcrossS8N9 import BABYAI_MOVETWOACROSS_HTML, get_movetwoacross_html
from .Synth import BABYAI_SYNTH_HTML, get_synth_html
from .SynthLoc import BABYAI_SYNTHLOC_HTML, get_synthloc_html
from .SynthSeq import BABYAI_SYNTHSEQ_HTML, get_synthseq_html
from .MiniBossLevel import BABYAI_MINIBOSSLEVEL_HTML, get_minibosslevel_html
from .BossLevel import BABYAI_BOSSLEVEL_HTML, get_bosslevel_html
from .BossLevelNoUnlock import BABYAI_BOSSLEVEL_NOUNLOCK_HTML, get_bosslevel_nounlock_html

__all__ = [
    # GoTo family
    "BABYAI_GOTO_REDBALL_GREY_HTML",
    "get_goto_redball_grey_html",
    "BABYAI_GOTO_REDBALL_HTML",
    "get_goto_redball_html",
    "BABYAI_GOTO_REDBALL_NODISTS_HTML",
    "get_goto_redball_nodists_html",
    "BABYAI_GOTO_OBJ_HTML",
    "get_goto_obj_html",
    "BABYAI_GOTO_LOCAL_HTML",
    "get_goto_local_html",
    "BABYAI_GOTO_HTML",
    "get_goto_html",
    "BABYAI_GOTO_IMPUNLOCK_HTML",
    "get_goto_impunlock_html",
    "BABYAI_GOTO_SEQ_HTML",
    "get_goto_seq_html",
    "BABYAI_GOTO_REDBLUEBALL_HTML",
    "get_goto_redblueball_html",
    "BABYAI_GOTO_DOOR_HTML",
    "get_goto_door_html",
    "BABYAI_GOTO_OBJDOOR_HTML",
    "get_goto_objdoor_html",
    # Open family
    "BABYAI_OPEN_HTML",
    "get_open_html",
    "BABYAI_OPEN_REDDOOR_HTML",
    "get_open_reddoor_html",
    "BABYAI_OPEN_DOOR_HTML",
    "get_open_door_html",
    "BABYAI_OPEN_TWODOORS_HTML",
    "get_open_twodoors_html",
    "BABYAI_OPEN_DOORSORDER_HTML",
    "get_open_doorsorder_html",
    # Pickup family
    "BABYAI_PICKUP_HTML",
    "get_pickup_html",
    "BABYAI_UNBLOCK_PICKUP_HTML",
    "get_unblock_pickup_html",
    "BABYAI_PICKUP_LOC_HTML",
    "get_pickup_loc_html",
    "BABYAI_PICKUP_DIST_HTML",
    "get_pickup_dist_html",
    "BABYAI_PICKUP_ABOVE_HTML",
    "get_pickup_above_html",
    # Unlock family
    "BABYAI_UNLOCK_HTML",
    "get_unlock_html",
    "BABYAI_UNLOCK_LOCAL_HTML",
    "get_unlock_local_html",
    "BABYAI_KEY_INBOX_HTML",
    "get_key_inbox_html",
    "BABYAI_UNLOCK_PICKUP_HTML",
    "get_unlock_pickup_html",
    "BABYAI_BLOCKED_UNLOCK_PICKUP_HTML",
    "get_blocked_unlock_pickup_html",
    "BABYAI_UNLOCK_TO_UNLOCK_HTML",
    "get_unlock_to_unlock_html",
    # PutNext family
    "BABYAI_PUTNEXT_LOCAL_HTML",
    "get_putnext_local_html",
    "BABYAI_PUTNEXT_HTML",
    "get_putnext_html",
    # Complex environments
    "BABYAI_ACTION_OBJDOOR_HTML",
    "get_action_objdoor_html",
    "BABYAI_FINDOBJ_HTML",
    "get_findobj_html",
    "BABYAI_KEYCORRIDOR_HTML",
    "get_keycorridor_html",
    "BABYAI_ONEROOM_HTML",
    "get_oneroom_html",
    "BABYAI_MOVETWOACROSS_HTML",
    "get_movetwoacross_html",
    "BABYAI_SYNTH_HTML",
    "get_synth_html",
    "BABYAI_SYNTHLOC_HTML",
    "get_synthloc_html",
    "BABYAI_SYNTHSEQ_HTML",
    "get_synthseq_html",
    "BABYAI_MINIBOSSLEVEL_HTML",
    "get_minibosslevel_html",
    "BABYAI_BOSSLEVEL_HTML",
    "get_bosslevel_html",
    "BABYAI_BOSSLEVEL_NOUNLOCK_HTML",
    "get_bosslevel_nounlock_html",
]
