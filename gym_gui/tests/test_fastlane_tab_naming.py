"""Tests for worker-aware FastLane tab naming logic.

Ensures:
- Ray worker runs create a single grid tab named "Ray-Live-{env}-{N}W-{run_id[:8]}"
  with all workers displayed in a grid layout
- CleanRL worker runs create tabs named "CleanRL-Live-{agent_id}"
- Metadata extraction helpers work correctly
- Grid dimension calculations are correct
"""

import pytest
from typing import Dict, Any, List, Tuple


class TestMetadataExtraction:
    """Test metadata extraction helper methods."""

    @pytest.fixture
    def ray_metadata(self) -> Dict[str, Any]:
        """Sample Ray worker metadata."""
        return {
            "ui": {
                "worker_id": "ray_worker",
                "env_id": "multiwalker_v9",
                "family": "sisl",
                "algorithm": "PPO",
                "paradigm": "parameter_sharing",
                "mode": "training",
            },
            "worker": {
                "worker_id": "ray_worker",
                "module": "ray_worker.cli",
                "config": {
                    "run_id": "01KCC1DJ2BJM9PWZ727ACYZMJX",
                    "environment": {
                        "family": "sisl",
                        "env_id": "multiwalker_v9",
                        "api_type": "parallel",
                    },
                    "paradigm": "parameter_sharing",
                },
            },
        }

    @pytest.fixture
    def cleanrl_metadata(self) -> Dict[str, Any]:
        """Sample CleanRL worker metadata."""
        return {
            "ui": {
                "worker_id": "cleanrl_worker",
                "env_id": "CartPole-v1",
                "algorithm": "PPO",
                "mode": "training",
            },
            "worker": {
                "worker_id": "cleanrl_worker",
                "module": "cleanrl_worker.cli",
                "agent_id": "agent_cartpole",
                "config": {
                    "run_id": "cleanrl-run-123",
                    "env_id": "CartPole-v1",
                },
            },
        }

    def test_get_worker_id_from_ray_metadata(self, ray_metadata):
        """Extract worker_id from Ray metadata."""
        worker_id = _get_worker_id_from_metadata(ray_metadata)
        assert worker_id == "ray_worker"

    def test_get_worker_id_from_cleanrl_metadata(self, cleanrl_metadata):
        """Extract worker_id from CleanRL metadata."""
        worker_id = _get_worker_id_from_metadata(cleanrl_metadata)
        assert worker_id == "cleanrl_worker"

    def test_get_worker_id_from_ui_fallback(self):
        """Falls back to ui.worker_id if worker.worker_id missing."""
        metadata = {
            "ui": {"worker_id": "test_worker"},
            "worker": {"module": "some.module"},
        }
        worker_id = _get_worker_id_from_metadata(metadata)
        assert worker_id == "test_worker"

    def test_get_worker_id_empty_for_missing(self):
        """Returns empty string if worker_id not found."""
        metadata = {"ui": {}, "worker": {}}
        worker_id = _get_worker_id_from_metadata(metadata)
        assert worker_id == ""

    def test_get_env_id_from_ray_metadata(self, ray_metadata):
        """Extract env_id from Ray metadata."""
        env_id = _get_env_id_from_metadata(ray_metadata)
        assert env_id == "multiwalker_v9"

    def test_get_env_id_from_cleanrl_metadata(self, cleanrl_metadata):
        """Extract env_id from CleanRL metadata."""
        env_id = _get_env_id_from_metadata(cleanrl_metadata)
        assert env_id == "CartPole-v1"

    def test_get_env_id_from_worker_config_fallback(self):
        """Falls back to worker.config.environment.env_id."""
        metadata = {
            "ui": {},
            "worker": {
                "config": {
                    "environment": {"env_id": "waterworld_v4"}
                }
            },
        }
        env_id = _get_env_id_from_metadata(metadata)
        assert env_id == "waterworld_v4"

    def test_get_env_id_empty_for_missing(self):
        """Returns empty string if env_id not found."""
        metadata = {"ui": {}, "worker": {}}
        env_id = _get_env_id_from_metadata(metadata)
        assert env_id == ""

    def test_get_num_workers_from_metadata(self):
        """Extract num_workers from metadata."""
        metadata = {
            "worker": {
                "config": {
                    "resources": {"num_workers": 2},
                }
            }
        }
        num_workers = _get_num_workers_from_metadata(metadata)
        assert num_workers == 2

    def test_get_num_workers_default_to_zero(self):
        """Returns 0 if resources.num_workers not found."""
        metadata = {"worker": {"config": {}}}
        num_workers = _get_num_workers_from_metadata(metadata)
        assert num_workers == 0


class TestGridDimensions:
    """Test grid dimension calculations for multi-worker display."""

    def test_grid_1_worker(self):
        """1 worker: 1x1 grid."""
        assert _compute_grid_dimensions(1) == (1, 1)

    def test_grid_2_workers(self):
        """2 workers: 1x2 grid."""
        assert _compute_grid_dimensions(2) == (1, 2)

    def test_grid_3_workers(self):
        """3 workers: 2x2 grid (one cell empty)."""
        assert _compute_grid_dimensions(3) == (2, 2)

    def test_grid_4_workers(self):
        """4 workers: 2x2 grid."""
        assert _compute_grid_dimensions(4) == (2, 2)

    def test_grid_5_workers(self):
        """5 workers: 2x3 grid (one cell empty)."""
        assert _compute_grid_dimensions(5) == (2, 3)

    def test_grid_6_workers(self):
        """6 workers: 2x3 grid."""
        assert _compute_grid_dimensions(6) == (2, 3)

    def test_grid_9_workers(self):
        """9 workers: 3x3 grid."""
        assert _compute_grid_dimensions(9) == (3, 3)

    def test_grid_0_workers(self):
        """0 workers: defaults to 1x1."""
        assert _compute_grid_dimensions(0) == (1, 1)


class TestRayGridTabNaming:
    """Test Ray grid tab naming (single tab for ACTIVE workers only).

    RLlib architecture:
    - Worker 0 (local) = coordinator, doesn't sample when num_workers > 0
    - Workers 1..N (remote) = active samplers

    We only show ACTIVE workers:
    - num_workers=0: W0 samples (1 cell)
    - num_workers=2: W1, W2 sample (2 cells, W0 is coordinator)
    """

    def test_ray_grid_tab_title_multi_worker(self):
        """With num_workers=2, shows 2 active workers (W1, W2)."""
        env_id = "multiwalker_v9"
        run_id = "01KCC1DJ2BJM9PWZ727ACYZMJX"
        run_mode = "training"
        num_workers = 2  # Active: W1, W2 = 2 workers

        title = _build_ray_grid_tab_title(env_id, run_id, run_mode, num_workers)

        assert title == "Ray-Live-multiwalker_v9-2W-01KCC1DJ"

    def test_ray_grid_tab_title_eval_mode(self):
        """Ray eval mode uses Ray-Eval prefix."""
        env_id = "waterworld_v4"
        run_id = "01ABCDEF12345678"
        run_mode = "policy_eval"
        num_workers = 1  # Active: W1 = 1 worker

        title = _build_ray_grid_tab_title(env_id, run_id, run_mode, num_workers)

        assert title == "Ray-Eval-waterworld_v4-1W-01ABCDEF"

    def test_ray_grid_tab_title_single_worker(self):
        """With num_workers=0, W0 does sampling."""
        env_id = ""
        run_id = "01KCC1DJ2BJM9PWZ"
        run_mode = "training"
        num_workers = 0  # Active: W0 = 1 worker

        title = _build_ray_grid_tab_title(env_id, run_id, run_mode, num_workers)

        assert title == "Ray-Live-MultiAgent-1W-01KCC1DJ"

    def test_ray_active_worker_indices_multi_worker(self):
        """With num_workers=2, active workers are [1, 2] (W0 is coordinator)."""
        num_workers = 2
        worker_indices = _get_ray_active_worker_indices(num_workers)
        assert worker_indices == [1, 2]

    def test_ray_active_worker_indices_single_worker(self):
        """With num_workers=0, W0 is active."""
        num_workers = 0
        worker_indices = _get_ray_active_worker_indices(num_workers)
        assert worker_indices == [0]

    def test_ray_worker_stream_ids_multi_worker(self):
        """Stream IDs for active workers (W1, W2)."""
        run_id = "01KCC1DJ2BJM9PWZ727ACYZMJX"
        num_workers = 2

        stream_ids = _build_ray_stream_ids(run_id, num_workers)

        assert stream_ids == [
            "01KCC1DJ2BJM9PWZ727ACYZMJX-w1",
            "01KCC1DJ2BJM9PWZ727ACYZMJX-w2",
        ]

    def test_ray_worker_stream_ids_single_worker(self):
        """Single worker (W0) stream ID."""
        run_id = "01KCC1DJ2BJM9PWZ727ACYZMJX"
        num_workers = 0

        stream_ids = _build_ray_stream_ids(run_id, num_workers)

        assert stream_ids == ["01KCC1DJ2BJM9PWZ727ACYZMJX-w0"]


class TestCleanRLTabNaming:
    """Test CleanRL tab naming (unchanged - single tab per run)."""

    def test_cleanrl_worker_live_tab_name(self):
        """CleanRL worker creates CleanRL-Live-{agent_id} tab."""
        worker_id = "cleanrl_worker"
        env_id = "CartPole-v1"
        run_id = "cleanrl-run-123"
        run_mode = "training"
        agent_id = "agent_cartpole"

        title = _build_tab_title(worker_id, env_id, run_id, run_mode, agent_id)

        assert title == "CleanRL-Live-agent_cartpole"

    def test_cleanrl_worker_eval_tab_name(self):
        """CleanRL worker eval creates CleanRL-Eval-{agent_id} tab."""
        worker_id = "cleanrl_worker"
        env_id = "CartPole-v1"
        run_id = "cleanrl-run-123"
        run_mode = "policy_eval"
        agent_id = "agent_eval"

        title = _build_tab_title(worker_id, env_id, run_id, run_mode, agent_id)

        assert title == "CleanRL-Eval-agent_eval"

    def test_cleanrl_worker_default_agent(self):
        """CleanRL worker uses 'agent' if agent_id missing."""
        worker_id = "cleanrl_worker"
        env_id = "CartPole-v1"
        run_id = "cleanrl-run-123"
        run_mode = "training"
        agent_id = ""

        title = _build_tab_title(worker_id, env_id, run_id, run_mode, agent_id)

        assert title == "CleanRL-Live-agent"

    def test_unknown_worker_defaults_to_cleanrl_style(self):
        """Unknown workers use CleanRL-style naming."""
        worker_id = "custom_worker"
        env_id = "MyEnv-v1"
        run_id = "custom-run-123"
        run_mode = "training"
        agent_id = "my_agent"

        title = _build_tab_title(worker_id, env_id, run_id, run_mode, agent_id)

        assert title == "CleanRL-Live-my_agent"

    def test_empty_worker_id_defaults_to_cleanrl_style(self):
        """Empty worker_id uses CleanRL-style naming."""
        worker_id = ""
        env_id = "Env-v1"
        run_id = "run-123"
        run_mode = "training"
        agent_id = "default"

        title = _build_tab_title(worker_id, env_id, run_id, run_mode, agent_id)

        assert title == "CleanRL-Live-default"


# Helper functions that mirror the main_window.py implementation
# These are used by tests to verify the logic without Qt dependencies

import math


def _compute_grid_dimensions(num_workers: int) -> Tuple[int, int]:
    """Compute optimal grid dimensions (rows, cols) for given number of workers.

    Returns dimensions that create a roughly square grid:
    - 1 worker: 1x1
    - 2 workers: 1x2
    - 3-4 workers: 2x2
    - 5-6 workers: 2x3
    - 7-9 workers: 3x3
    - etc.
    """
    if num_workers <= 0:
        return (1, 1)
    if num_workers == 1:
        return (1, 1)
    if num_workers == 2:
        return (1, 2)

    cols = math.ceil(math.sqrt(num_workers))
    rows = math.ceil(num_workers / cols)
    return (rows, cols)


def _get_worker_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract worker_id from metadata (e.g., 'ray_worker', 'cleanrl_worker')."""
    # Try worker.worker_id first
    worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
    if isinstance(worker_meta, dict):
        worker_id = worker_meta.get("worker_id")
        if isinstance(worker_id, str) and worker_id.strip():
            return worker_id.strip()
    # Try ui.worker_id
    ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
    if isinstance(ui_meta, dict):
        worker_id = ui_meta.get("worker_id")
        if isinstance(worker_id, str) and worker_id.strip():
            return worker_id.strip()
    return ""


def _get_env_id_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract environment ID from metadata for tab naming."""
    # Try ui.env_id first (set by forms)
    ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
    if isinstance(ui_meta, dict):
        env_id = ui_meta.get("env_id")
        if isinstance(env_id, str) and env_id.strip():
            return env_id.strip()
    # Try worker.config.environment.env_id
    worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
    if isinstance(worker_meta, dict):
        worker_config = worker_meta.get("config")
        if isinstance(worker_config, dict):
            env_config = worker_config.get("environment")
            if isinstance(env_config, dict):
                env_id = env_config.get("env_id")
                if isinstance(env_id, str) and env_id.strip():
                    return env_id.strip()
    return ""


def _get_num_workers_from_metadata(metadata: Dict[str, Any]) -> int:
    """Extract num_workers from metadata (default 0 for single worker).

    Mirrors the logic in FastLaneTabHandler.get_num_workers
    """
    worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
    if isinstance(worker_meta, dict):
        worker_config = worker_meta.get("config")
        if isinstance(worker_config, dict):
            resources = worker_config.get("resources")
            if isinstance(resources, dict):
                num_workers = resources.get("num_workers")
                if isinstance(num_workers, int):
                    return num_workers
    return 0


def _get_ray_active_worker_indices(num_workers: int) -> List[int]:
    """Get list of ACTIVE worker indices only.

    RLlib architecture:
    - num_workers=0: W0 samples → [0]
    - num_workers>0: W1..WN sample (W0 is coordinator) → [1, 2, ..., N]
    """
    if num_workers == 0:
        return [0]
    return list(range(1, num_workers + 1))


def _build_ray_grid_tab_title(
    env_id: str,
    run_id: str,
    run_mode: str,
    num_workers: int,
) -> str:
    """Build single grid tab title for active Ray workers.

    Mirrors the logic in main_window.py:_open_ray_fastlane_tabs
    Creates one tab with format: Ray-Live-{env}-{N}W-{run_id[:8]}
    where N is the number of ACTIVE workers.
    """
    env_label = env_id or "MultiAgent"
    mode_prefix = "Ray-Eval" if run_mode == "policy_eval" else "Ray-Live"
    # Active workers: W0 if num_workers=0, else W1..WN
    active_workers = 1 if num_workers == 0 else num_workers
    return f"{mode_prefix}-{env_label}-{active_workers}W-{run_id[:8]}"


def _build_ray_stream_ids(run_id: str, num_workers: int) -> List[str]:
    """Build stream IDs for ACTIVE Ray workers only.

    - num_workers=0: [{run_id}-w0]
    - num_workers=2: [{run_id}-w1, {run_id}-w2]
    """
    worker_indices = _get_ray_active_worker_indices(num_workers)
    return [f"{run_id}-w{idx}" for idx in worker_indices]


def _build_tab_title(
    worker_id: str,
    env_id: str,
    run_id: str,
    run_mode: str,
    canonical_agent_id: str,
) -> str:
    """Build tab title for non-Ray workers (CleanRL, etc).

    Mirrors the logic in main_window.py:_open_single_fastlane_tab
    """
    # CleanRL or other workers: CleanRL-Live-{agent_id}
    if run_mode == "policy_eval":
        return f"CleanRL-Eval-{canonical_agent_id or 'agent'}"
    else:
        return f"CleanRL-Live-{canonical_agent_id or 'agent'}"
