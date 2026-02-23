"""SMAC game documentation module.

SMAC (StarCraft Multi-Agent Challenge) is the standard cooperative multi-agent
RL benchmark using StarCraft II.  Agents control individual units in symmetric
micromanagement scenarios.

SMAC v1 (hand-designed maps):
    - 3m, 8m, 2s3z, 3s5z, 5m_vs_6m, MMM2

Repository: https://github.com/oxwhirl/smac
"""
from __future__ import annotations

from .SMAC_3m import SMAC_3M_HTML, get_smac_3m_html
from .SMAC_8m import SMAC_8M_HTML, get_smac_8m_html
from .SMAC_2s3z import SMAC_2S3Z_HTML, get_smac_2s3z_html
from .SMAC_3s5z import SMAC_3S5Z_HTML, get_smac_3s5z_html
from .SMAC_5m_vs_6m import SMAC_5M_VS_6M_HTML, get_smac_5m_vs_6m_html
from .SMAC_MMM2 import SMAC_MMM2_HTML, get_smac_mmm2_html

__all__ = [
    "SMAC_3M_HTML",
    "SMAC_8M_HTML",
    "SMAC_2S3Z_HTML",
    "SMAC_3S5Z_HTML",
    "SMAC_5M_VS_6M_HTML",
    "SMAC_MMM2_HTML",
    "get_smac_3m_html",
    "get_smac_8m_html",
    "get_smac_2s3z_html",
    "get_smac_3s5z_html",
    "get_smac_5m_vs_6m_html",
    "get_smac_mmm2_html",
]
