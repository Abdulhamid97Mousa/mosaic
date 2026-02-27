# 3rd_party/xuance_worker/tests/conftest.py
"""Shared fixtures for xuance_worker tests."""

from __future__ import annotations

import pytest


# Path to the trained IPPO checkpoint (collect_1vs1, phase1)
CHECKPOINT_PATH = (
    "/home/hamid/Desktop/software/mosaic/var/trainer/custom_scripts/"
    "01KHZWG493V7QFPQ0NNXQW1Y1A/checkpoints/torch/collect_1vs1/"
    "seed_1_2026_0221_183918/final_train_model.pth"
)


@pytest.fixture
def checkpoint_path():
    """Return the path to the trained IPPO collect_1vs1 checkpoint."""
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return CHECKPOINT_PATH
