"""Presenter for XuanCe worker analytics lane."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

_LOGGER = logging.getLogger(__name__)


class XuanCeWorkerPresenter:
    """Placeholder presenter for the XuanCe analytics worker.

    The XuanCe worker provides 46+ RL algorithms for single-agent and multi-agent
    training. This presenter is a placeholder until full analytics integration lands.

    Training runs originate from the XuanCe training form, and analytics tabs will be
    implemented once manifest ingestion is complete.

    Supported algorithms:
    - Single-agent: DQN, PPO, SAC, TD3, DDPG, DreamerV3
    - Multi-agent: MAPPO, QMIX, MADDPG, VDN, COMA, IAC

    Supported environments:
    - Gymnasium (classic_control, atari, mujoco, box2d)
    - PettingZoo (MPE, SISL)
    - SMAC (StarCraft Multi-Agent Challenge)
    - Google Football
    """

    @property
    def id(self) -> str:
        return "xuance_worker"

    def build_train_request(self, policy_path: Any, current_game: Optional[Any]) -> dict:
        """Policy evaluation is not yet supported for XuanCe workers.

        Policy loading and evaluation will be implemented in Phase 6.
        """
        raise NotImplementedError("XuanCe worker does not support policy evaluation yet.")

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent: Any) -> List[Any]:
        """Create worker-specific UI tabs for a running XuanCe training.

        For XuanCe, this creates FastLane tabs with paradigm-aware naming:
        - Single-agent: XuanCe-SA-Live-{env_id}
        - Multi-agent: XuanCe-MA-Live-{env_id}

        Args:
            run_id: Unique run identifier.
            agent_id: Agent identifier.
            first_payload: First telemetry payload containing metadata.
            parent: Parent Qt widget.

        Returns:
            List of QWidget tab instances.
        """
        tabs = []

        # Extract metadata (following Ray worker pattern - see ray_worker_presenter.py:149-150)
        metadata = first_payload.get("metadata", {})
        ui_meta = metadata.get("ui", {})

        # Check if FastLane is enabled
        fastlane_enabled = ui_meta.get("fastlane_enabled", False)
        if not fastlane_enabled:
            _LOGGER.info(
                "FastLane not enabled for XuanCe run: run_id=%s (fastlane_enabled=%s)",
                run_id, fastlane_enabled
            )
            return tabs

        # Get env_id and paradigm from metadata
        env_id = ui_meta.get("env_id", "env")
        paradigm = ui_meta.get("paradigm", "single_agent")

        _LOGGER.info(
            "XuanCe create_tabs: run_id=%s, env_id=%s, paradigm=%s, fastlane_enabled=%s",
            run_id, env_id, paradigm, fastlane_enabled
        )

        try:
            from gym_gui.ui.widgets.fastlane_tab import FastLaneTab

            # Determine paradigm prefix
            # SA = Single-Agent, MA = Multi-Agent
            paradigm_prefix = "MA" if paradigm == "multi_agent" else "SA"

            # Tab naming: XuanCe-{SA|MA}-Live-{env_id}
            # Examples:
            #   - XuanCe-SA-Live-CartPole-v1 (single-agent)
            #   - XuanCe-MA-Live-simple_spread_v3 (multi-agent)
            tab_name = f"XuanCe-{paradigm_prefix}-Live-{env_id}-{run_id[:8]}"

            fastlane_tab = FastLaneTab(
                run_id=run_id,
                agent_id=agent_id,
                mode_label="Fast lane",
                run_mode="train",
                parent=parent,
            )

            tabs.append((tab_name, fastlane_tab))

            _LOGGER.info(
                "Created XuanCe FastLane tab: %s (paradigm=%s)",
                tab_name, paradigm
            )

        except ImportError as e:
            _LOGGER.warning("FastLane tab not available: %s", e)
        except Exception as exc:
            _LOGGER.exception("Failed to create XuanCe FastLane tab: %s", exc)

        return tabs


__all__ = ["XuanCeWorkerPresenter"]
