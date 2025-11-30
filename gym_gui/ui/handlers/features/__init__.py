"""Feature-specific handlers for cross-cutting concerns.

These handlers manage features that span multiple environments or modes:
- GameConfigHandler: Environment configuration changes
- MPCHandler: MuJoCo MPC launcher
- LogHandler: Log filtering and display
- HumanVsAgentHandler: Human vs Agent AI setup and game state

Available handlers:
- GameConfigHandler: Environment config (slippery toggle, reload)
- MPCHandler: MuJoCo MPC process launch/stop
- LogHandler: Log filtering
- HumanVsAgentHandler: AI opponent setup and game state management
"""

from gym_gui.ui.handlers.features.game_config import GameConfigHandler
from gym_gui.ui.handlers.features.mpc import MPCHandler
from gym_gui.ui.handlers.features.log import LogHandler
from gym_gui.ui.handlers.features.human_vs_agent import HumanVsAgentHandler

__all__ = [
    "GameConfigHandler",
    "MPCHandler",
    "LogHandler",
    "HumanVsAgentHandler",
]
