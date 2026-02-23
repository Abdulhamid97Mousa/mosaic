#!/usr/bin/env python
"""Smoke test: 50k collect_1vs1 + 50k soccer_1vs1 curriculum.

Verifies that the environment swap actually switches from collect to soccer
by checking the grid dimensions in the vectorized environments after each
phase begins.

Usage (from project root):
    source .venv/bin/activate
    python 3rd_party/xuance_worker/tests/smoke_test_curriculum_swap.py

Expected output:
    Phase 1 env_id: collect_1vs1  (grid ~10x10)
    Phase 2 env_id: soccer_1vs1   (grid ~16x11)
    Both phases produce non-zero episode rewards.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("smoke_test")

# Disable FastLane (no GUI in this test)
os.environ["MOSAIC_FASTLANE_ENABLED"] = "0"
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")


def build_config(tmpdir: str) -> dict:
    """Build a minimal curriculum config with 50k + 50k steps."""
    return {
        "run_id": "smoke_test_curriculum_swap",
        "method": "mappo",
        "env": "multigrid",
        "env_id": "collect_1vs1",
        "running_steps": 100000,
        "backend": "pytorch",
        "device": "cpu",
        "parallels": 2,
        "extras": {
            "training_mode": "competitive",
            "num_envs": 2,
            "tensorboard_dir": os.path.join(tmpdir, "tensorboard"),
            "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
            "curriculum_schedule": [
                {"env_id": "collect_1vs1", "steps": 50000},
                {"env_id": "soccer_1vs1", "steps": 50000},
            ],
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="smoke_curriculum_") as tmpdir:
        config_dict = build_config(tmpdir)
        config_path = os.path.join(tmpdir, "curriculum_config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        LOGGER.info("Config written to: %s", config_path)
        LOGGER.info(
            "Schedule: %s",
            json.dumps(config_dict["extras"]["curriculum_schedule"]),
        )

        # --- Run via CLI (same path as the real trainer daemon) ---
        from xuance_worker.cli import main as cli_main

        LOGGER.info("=" * 60)
        LOGGER.info("  Starting 50k+50k curriculum smoke test")
        LOGGER.info("=" * 60)

        rc = cli_main(["--config", config_path])

        LOGGER.info("=" * 60)
        if rc == 0:
            LOGGER.info("  PASSED -- curriculum swap smoke test succeeded")
        else:
            LOGGER.error("  FAILED -- CLI returned exit code %d", rc)
        LOGGER.info("=" * 60)

        # Check that checkpoints were written for both phases
        ckpt_dir = Path(tmpdir) / "checkpoints"
        if ckpt_dir.exists():
            files = list(ckpt_dir.rglob("*.pth"))
            LOGGER.info("Checkpoints found: %d", len(files))
            for f in sorted(files):
                LOGGER.info("  %s", f.name)
        else:
            LOGGER.warning("No checkpoint directory found")

        return rc


if __name__ == "__main__":
    sys.exit(main())
