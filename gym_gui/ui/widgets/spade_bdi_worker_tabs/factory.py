"""Factory for creating SPADE-BDI worker UI tabs.

This module provides TabFactory, which encapsulates the logic for
instantiating the standard set of SPADE-BDI tabs with proper configuration
and environment awareness.

Tabs created:
- AgentOnlineTab: Primary statistics view
- AgentReplayTab: Historical episode browser
- AgentOnlineGridTab: Live grid visualization
- AgentOnlineRawTab: Debug stream
- AgentOnlineVideoTab: RGB video stream (visual envs only)
"""

from typing import Optional
import logging

from gym_gui.services.service_locator import get_service_locator
from gym_gui.core.enums import GameId
from gym_gui.rendering import RendererRegistry
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_WORKER_TABS_TRACE,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_WORKER_TABS_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
)

from .agent_online_tab import AgentOnlineTab
from .agent_online_grid_tab import AgentOnlineGridTab
from .agent_online_raw_tab import AgentOnlineRawTab
from .agent_online_video_tab import AgentOnlineVideoTab
from .agent_replay_tab import AgentReplayTab


logger = logging.getLogger(__name__)


class TabFactory:
    """Factory for creating SPADE-BDI worker tabs.

    Creates a standard set of tabs for monitoring and analyzing a running
    agent. Handles environment detection (ToyText vs. visual) and conditional
    tab creation based on renderer capabilities.
    """

    def create_tabs(self, run_id: str, agent_id: str, first_payload: dict, parent) -> list:
        """Create all tabs for a running agent.

        Instantiates tabs based on environment type detected from payload.
        ToyText environments (FrozenLake, CliffWalking, Taxi, MiniGrid, GridWorld)
        get grid visualization; visual environments also get video tab.

        Args:
            run_id: Unique run identifier
            agent_id: Agent identifier within the run
            first_payload: First telemetry step containing metadata
            parent: Parent Qt widget

        Returns:
            list: List of created QWidget tab instances
        """
        locator = get_service_locator()
        renderer_registry = locator.resolve(RendererRegistry)

        # Extract game_id and determine environment family
        game_id_str = first_payload.get("game_id", "").lower()
        is_toytext = any(
            name in game_id_str
            for name in ["frozenlake", "cliffwalking", "taxi", "gridworld", "minigrid"]
        )

        # Convert game_id string to GameId enum if possible
        game_id_enum = None
        if game_id_str:
            try:
                game_id_enum = GameId(game_id_str)
            except (ValueError, KeyError):
                log_constant(logger, LOG_UI_WORKER_TABS_WARNING, message=
                    "Invalid game_id in first_payload",
                    extra={"run_id": run_id, "agent_id": agent_id, "game_id": game_id_str},
                )

        # Create standard tabs
        online = AgentOnlineTab(run_id, agent_id, parent=parent)
        replay = AgentReplayTab(run_id, agent_id, parent=parent)
        grid = AgentOnlineGridTab(
            run_id,
            agent_id,
            game_id=game_id_enum,
            renderer_registry=renderer_registry,
            parent=parent,
        )
        raw = AgentOnlineRawTab(run_id, agent_id, parent=parent)

        tabs = [online, replay, grid, raw]

        # Create video tab only for visual environments
        if not is_toytext:
            video = AgentOnlineVideoTab(run_id, agent_id, parent=parent)
            tabs.append(video)

        log_constant(logger, LOG_UI_WORKER_TABS_TRACE, message=
            "Created tabs for agent",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "game_id": game_id_str,
                "is_toytext": is_toytext,
                "tab_count": len(tabs),
            },
        )

        return tabs
