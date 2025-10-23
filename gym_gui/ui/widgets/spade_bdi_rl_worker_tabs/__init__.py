"""SPADE-BDI worker UI tabs package.

This package contains all worker-specific UI tabs for the SPADE-BDI RL integration:
- AgentOnlineTab: Default real-time view combining grid + stats
- AgentOnlineGridTab: Live grid visualization + episode stats
- AgentOnlineRawTab: Raw JSON step data for debugging
- AgentOnlineVideoTab: Live RGB frame visualization
- AgentReplayTab: Historical replay of completed training runs

Re-exports are provided for backward compatibility and to simplify imports
from other modules.

Usage:
    from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs import (
        AgentOnlineTab,
        AgentOnlineGridTab,
        AgentOnlineRawTab,
        AgentOnlineVideoTab,
        AgentReplayTab,
    )
"""

from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.agent_online_grid_tab import (
    AgentOnlineGridTab,
)
from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.agent_online_raw_tab import (
    AgentOnlineRawTab,
)
from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.agent_online_tab import (
    AgentOnlineTab,
)
from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.agent_online_video_tab import (
    AgentOnlineVideoTab,
)
from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.agent_replay_tab import (
    AgentReplayTab,
)
from gym_gui.ui.widgets.spade_bdi_rl_worker_tabs.factory import (
    TabFactory,
)

from gym_gui.ui.widgets.agent_train_dialog import AgentTrainDialog

__all__ = [
    "AgentOnlineTab",
    "AgentOnlineGridTab",
    "AgentOnlineRawTab",
    "AgentOnlineVideoTab",
    "AgentReplayTab",
    "TabFactory",
]
