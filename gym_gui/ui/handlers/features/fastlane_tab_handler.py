"""FastLane tab management handler for MainWindow.

Extracts FastLane tab creation and metadata extraction logic from MainWindow.
Handles both Ray multi-worker tabs and single CleanRL tabs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from gym_gui.ui.widgets.render_tabs import RenderTabs
    from gym_gui.logging_config.helpers import LogConstantMixin
    from gym_gui.core.enums import GameId

_LOGGER = logging.getLogger(__name__)


class FastLaneTabHandler:
    """Handler for FastLane tab creation and management.

    Manages:
    - Opening Ray multi-worker FastLane grid tabs
    - Opening single CleanRL FastLane tabs
    - Metadata extraction for tab configuration
    - Tab deduplication tracking
    """

    def __init__(
        self,
        render_tabs: "RenderTabs",
        log_callback: Optional[Callable[..., None]] = None,
    ):
        """Initialize the handler.

        Args:
            render_tabs: RenderTabs widget for adding tabs.
            log_callback: Optional callback for structured logging.
        """
        self._render_tabs = render_tabs
        self._log = log_callback or (lambda *args, **kwargs: None)
        self._fastlane_tabs_open: Set[Tuple[str, str]] = set()

    @property
    def tabs_open(self) -> Set[Tuple[str, str]]:
        """Return the set of open FastLane tab keys."""
        return self._fastlane_tabs_open

    def clear_tabs_for_run(self, run_id: str) -> None:
        """Remove tracking for all tabs associated with a run."""
        self._fastlane_tabs_open = {
            key for key in self._fastlane_tabs_open if key[0] != run_id
        }

    def maybe_open_fastlane_tab(
        self,
        run_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Open a FastLane tab if metadata supports it.

        Args:
            run_id: Training run ID.
            agent_id: Agent ID.
            metadata: Run metadata dictionary.
        """
        _LOGGER.info(
            "maybe_open_fastlane_tab called: run_id=%s, agent_id=%s, metadata_keys=%s",
            run_id, agent_id, list(metadata.keys()) if metadata else None
        )

        if not metadata:
            self._log(
                message="FastLane tab skipped; metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            _LOGGER.warning("FastLane tab skipped; metadata missing for run_id=%s", run_id)
            return

        supports = self.metadata_supports_fastlane(metadata)
        _LOGGER.info(
            "metadata_supports_fastlane=%s for run_id=%s (ui.fastlane_only=%s, worker.module=%s)",
            supports, run_id,
            metadata.get("ui", {}).get("fastlane_only"),
            metadata.get("worker", {}).get("module")
        )

        if not supports:
            self._log(
                message="FastLane tab skipped; metadata does not advertise fastlane",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return

        canonical_agent_id = self.get_canonical_agent_id(metadata, agent_id)
        run_mode = self.get_run_mode(metadata)
        worker_id = self.get_worker_id(metadata)
        env_id = self.get_env_id(metadata)

        # For Ray workers with num_workers > 0, create multi-worker tab
        if worker_id == "ray_worker":
            num_workers = self.get_num_workers(metadata)
            self.open_ray_fastlane_tabs(
                run_id, canonical_agent_id, run_mode, env_id, num_workers
            )
        else:
            # CleanRL or other workers: single tab
            self.open_single_fastlane_tab(
                run_id, canonical_agent_id, run_mode, env_id, worker_id
            )

    # Board game environments that should use BoardGameFastLaneTab
    _BOARD_GAME_ENVS: Set[str] = {
        "chess_v6", "go_v5", "connect_four_v3", "tictactoe_v3",
        "chess", "go", "connect_four", "tictactoe",
    }

    def _is_board_game_env(self, env_id: str) -> bool:
        """Check if env_id is a board game that needs interactive rendering."""
        if not env_id:
            return False
        env_lower = env_id.lower()
        # Check exact match
        if env_lower in self._BOARD_GAME_ENVS:
            return True
        # Check prefix match (e.g., "chess_v6" matches "chess")
        for board_game in self._BOARD_GAME_ENVS:
            if env_lower.startswith(board_game):
                return True
        return False

    def open_ray_fastlane_tabs(
        self,
        run_id: str,
        agent_id: str,
        run_mode: str,
        env_id: str,
        num_workers: int,
    ) -> None:
        """Open a single grid-based FastLane tab for active Ray workers.

        RLlib architecture:
        - num_workers=0: W0 samples (1 cell)
        - num_workers=2: W1, W2 sample (2 cells, W0 is coordinator)

        For board games (chess, go, etc.), uses BoardGameFastLaneTab for
        interactive rendering with FEN/metadata.
        """
        # Use run_id as key - one grid tab per run
        key = (run_id, "ray_grid")
        if key in self._fastlane_tabs_open:
            self._log(
                message="Ray FastLane tab already open",
                extra={"run_id": run_id},
            )
            return

        env_label = env_id or "MultiAgent"
        mode_prefix = "Ray-Eval" if run_mode == "policy_eval" else "Ray-Live"
        # Active workers: W0 if num_workers=0, else W1..WN (num_workers count)
        active_workers = 1 if num_workers == 0 else num_workers
        # First active worker index
        first_worker_idx = 0 if num_workers == 0 else 1

        # For board games, use BoardGameFastLaneTab for interactive rendering
        if self._is_board_game_env(env_id):
            self._open_board_game_fastlane_tab(
                run_id, agent_id, run_mode, env_id, key, first_worker_idx, mode_prefix
            )
            return

        # Standard multi-worker tab for non-board-game environments
        from gym_gui.ui.widgets.ray_multi_worker_fastlane_tab import RayMultiWorkerFastLaneTab

        try:
            tab = RayMultiWorkerFastLaneTab(
                run_id,
                num_workers,
                env_id=env_id,
                run_mode=run_mode,
                parent=self._render_tabs,
            )
        except Exception as exc:
            self._log(
                message="Failed to create Ray multi-worker FastLane tab",
                extra={"run_id": run_id, "num_workers": num_workers},
                exc_info=exc,
            )
            return

        # Tab title: Ray-Live-{env}-{N}W-{run_id[:8]}
        title = f"{mode_prefix}-{env_label}-{active_workers}W-{run_id[:8]}"
        self._render_tabs.add_dynamic_tab(run_id, title, tab)
        self._fastlane_tabs_open.add(key)
        self._log(
            message="Opened Ray multi-worker FastLane grid tab",
            extra={"run_id": run_id, "active_workers": active_workers, "title": title},
        )

    def _open_board_game_fastlane_tab(
        self,
        run_id: str,
        agent_id: str,
        run_mode: str,
        env_id: str,
        key: Tuple[str, str],
        worker_idx: int,
        mode_prefix: str,
    ) -> None:
        """Open a board game FastLane tab with interactive rendering."""
        from gym_gui.ui.widgets.board_game_fastlane_tab import BoardGameFastLaneTab
        from gym_gui.core.enums import GameId

        # Map env_id to GameId
        game_id = self._env_id_to_game_id(env_id)

        # Agent ID for FastLane: {run_id}_W{worker_idx}
        fastlane_agent_id = f"{run_id}_W{worker_idx}"

        try:
            tab = BoardGameFastLaneTab(
                run_id,
                fastlane_agent_id,
                game_id=game_id,
                mode_label=f"{mode_prefix} Board",
                run_mode=run_mode,
                parent=self._render_tabs,
            )
        except Exception as exc:
            self._log(
                message="Failed to create BoardGame FastLane tab",
                extra={"run_id": run_id, "env_id": env_id},
                exc_info=exc,
            )
            return

        # Tab title: Ray-Live-Chess-{run_id[:8]}
        title = f"{mode_prefix}-{env_id}-{run_id[:8]}"
        self._render_tabs.add_dynamic_tab(run_id, title, tab)
        self._fastlane_tabs_open.add(key)
        self._log(
            message="Opened BoardGame FastLane tab",
            extra={"run_id": run_id, "env_id": env_id, "game_id": str(game_id), "title": title},
        )

    def _env_id_to_game_id(self, env_id: str) -> Optional["GameId"]:
        """Map environment ID to GameId enum."""
        from gym_gui.core.enums import GameId

        env_lower = (env_id or "").lower()
        if "chess" in env_lower:
            return GameId.CHESS
        if "go" in env_lower:
            return GameId.GO
        if "connect_four" in env_lower:
            return GameId.CONNECT_FOUR
        if "tictactoe" in env_lower:
            return GameId.TIC_TAC_TOE
        return None

    def open_single_fastlane_tab(
        self,
        run_id: str,
        agent_id: str,
        run_mode: str,
        env_id: str,
        worker_id: str,
    ) -> None:
        """Open a single FastLane tab (for CleanRL and other non-Ray workers)."""
        from gym_gui.ui.widgets.fastlane_tab import FastLaneTab

        key = (run_id, agent_id)
        if key in self._fastlane_tabs_open:
            self._log(
                message="FastLane tab already tracked",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return

        try:
            mode_label = "Fast lane (evaluation)" if run_mode == "policy_eval" else "Fast lane"
            tab = FastLaneTab(
                run_id,
                agent_id,
                mode_label=mode_label,
                run_mode=run_mode,
                parent=self._render_tabs,
            )
        except Exception as exc:
            self._log(
                message="Failed to create FastLane tab",
                extra={"run_id": run_id, "agent_id": agent_id},
                exc_info=exc,
            )
            return

        # Worker-specific tab naming: {Prefix}-{Mode}-{env_id or agent_id}
        prefix = self._get_worker_prefix(worker_id)
        mode_suffix = "Eval" if run_mode == "policy_eval" else "Live"
        # Prefer env_id for XuanCe, fall back to agent_id for CleanRL compatibility
        label = env_id if env_id else (agent_id or "default")
        title = f"{prefix}-{mode_suffix}-{label}"

        self._render_tabs.add_dynamic_tab(run_id, title, tab)
        self._fastlane_tabs_open.add(key)
        self._log(
            message="Opened FastLane tab",
            extra={"run_id": run_id, "agent_id": agent_id, "title": title},
        )

    # -------------------------------------------------------------------------
    # Worker prefix mapping
    # -------------------------------------------------------------------------

    _WORKER_PREFIXES: Dict[str, str] = {
        "cleanrl_worker": "CleanRL",
        "xuance_worker": "XuanCe",
        "ray_worker": "Ray",
    }

    def _get_worker_prefix(self, worker_id: str) -> str:
        """Get display prefix for a worker type."""
        return self._WORKER_PREFIXES.get(worker_id, "Worker")

    # -------------------------------------------------------------------------
    # Metadata extraction methods
    # -------------------------------------------------------------------------

    def get_num_workers(self, metadata: Dict[str, Any]) -> int:
        """Extract num_workers from metadata (default 0 for single worker)."""
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

    def get_canonical_agent_id(self, metadata: Dict[str, Any], fallback: str) -> str:
        """Extract canonical agent ID from metadata."""
        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if isinstance(worker_meta, dict):
            meta_agent = worker_meta.get("agent_id")
            if isinstance(meta_agent, str) and meta_agent.strip():
                return meta_agent.strip()
            worker_config = worker_meta.get("config")
            if isinstance(worker_config, dict):
                config_agent = worker_config.get("agent_id")
                if isinstance(config_agent, str) and config_agent.strip():
                    return config_agent.strip()
        return fallback

    def get_worker_id(self, metadata: Dict[str, Any]) -> str:
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

    def get_env_id(self, metadata: Dict[str, Any]) -> str:
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

    def get_run_mode(self, metadata: Dict[str, Any]) -> str:
        """Extract run mode from metadata."""
        ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
        if isinstance(ui_meta, dict):
            mode = ui_meta.get("run_mode")
            if isinstance(mode, str) and mode.strip():
                return mode.strip().lower()
        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if isinstance(worker_meta, dict):
            config = worker_meta.get("config")
            if isinstance(config, dict):
                extras = config.get("extras")
                if isinstance(extras, dict):
                    mode = extras.get("mode")
                    if isinstance(mode, str) and mode.strip():
                        return mode.strip().lower()
        return "train"

    def metadata_supports_fastlane(self, metadata: Dict[str, Any]) -> bool:
        """Return True if the run metadata indicates FastLane visuals are available."""

        def _is_truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return False

        ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
        if isinstance(ui_meta, dict):
            fl_only = ui_meta.get("fastlane_only")
            if fl_only is not None:
                # Explicitly set: respect True/False without falling through
                return _is_truthy(fl_only)

        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if not isinstance(worker_meta, dict):
            return False

        module_name = str(worker_meta.get("module") or "").lower()
        worker_kind = str(worker_meta.get("worker_kind") or "").lower()
        worker_identifier = str(worker_meta.get("worker_id") or "").lower()

        # CleanRL worker detection
        if "cleanrl_worker" in module_name or worker_kind == "cleanrl" or worker_identifier == "cleanrl_worker":
            return True

        # Ray worker detection
        if "ray_worker" in module_name or worker_kind == "ray" or worker_identifier == "ray_worker":
            return True

        # XuanCe worker detection
        if "xuance_worker" in module_name or worker_kind == "xuance" or worker_identifier == "xuance_worker":
            return True

        worker_config = worker_meta.get("config")
        if isinstance(worker_config, dict):
            extras = worker_config.get("extras")
            if isinstance(extras, dict):
                if _is_truthy(extras.get("fastlane_only")) or _is_truthy(extras.get("fastlane_enabled")):
                    return True

        return False
