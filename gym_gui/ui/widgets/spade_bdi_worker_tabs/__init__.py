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
    from gym_gui.ui.widgets.spade_bdi_worker_tabs import (
        AgentOnlineTab,
        AgentOnlineGridTab,
        AgentOnlineRawTab,
        AgentOnlineVideoTab,
        AgentReplayTab,
    )
"""

from gym_gui.ui.widgets.spade_bdi_worker_tabs.agent_online_grid_tab import (
    AgentOnlineGridTab,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.agent_online_raw_tab import (
    AgentOnlineRawTab,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.agent_online_tab import (
    AgentOnlineTab,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.agent_online_video_tab import (
    AgentOnlineVideoTab,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.agent_replay_tab import (
    AgentReplayTab,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.factory import (
    TabFactory,
)

from gym_gui.ui.forms import get_worker_form_factory
from gym_gui.ui.widgets.spade_bdi_train_form import SpadeBdiTrainForm
from gym_gui.ui.widgets.spade_bdi_policy_selection_form import SpadeBdiPolicySelectionForm

__all__ = [
    "AgentOnlineTab",
    "AgentOnlineGridTab",
    "AgentOnlineRawTab",
    "AgentOnlineVideoTab",
    "AgentReplayTab",
    "TabFactory",
]

_form_factory = get_worker_form_factory()

# Register SPADE-BDI specific forms if not already registered
if not _form_factory.has_train_form("spade_bdi_worker"):
    _form_factory.register_train_form(
        "spade_bdi_worker",
        lambda parent=None, **kwargs: SpadeBdiTrainForm(parent=parent, **kwargs),
    )

if not _form_factory.has_policy_form("spade_bdi_worker_worker"):
    _form_factory.register_policy_form(
        "spade_bdi_worker",
        lambda parent=None, **kwargs: SpadeBdiPolicySelectionForm(parent=parent, **kwargs),
    )
