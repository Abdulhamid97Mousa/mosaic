"""SMACv2 / SMACv2 game documentation module.

SMACv2 (StarCraft Multi-Agent Challenge v2) is the procedurally-generated
version of SMAC, featuring variable team compositions each episode.

SMACv2 (procedural generation):
    - 10gen_terran, 10gen_protoss, 10gen_zerg

Repository: https://github.com/oxwhirl/smacv2
"""
from __future__ import annotations

from .SMACv2_Terran import SMACV2_TERRAN_HTML, get_smacv2_terran_html
from .SMACv2_Protoss import SMACV2_PROTOSS_HTML, get_smacv2_protoss_html
from .SMACv2_Zerg import SMACV2_ZERG_HTML, get_smacv2_zerg_html

__all__ = [
    # SMACv2
    "SMACV2_TERRAN_HTML",
    "SMACV2_PROTOSS_HTML",
    "SMACV2_ZERG_HTML",
    "get_smacv2_terran_html",
    "get_smacv2_protoss_html",
    "get_smacv2_zerg_html",
]
