"""Worker-level defaults and internal configuration constants.

These values are implementation details for the SPADE-BDI worker and may
change without notice. Keep domain-specific defaults close to the worker
code so the GUI/UI can import them when necessary (e.g., to seed forms).
"""

# ---------------------------------------------------------------------------
# Agent credentials / networking
# ---------------------------------------------------------------------------

DEFAULT_AGENT_JID = "agent@localhost"
DEFAULT_AGENT_PASSWORD = "secret"

# SPADE agent start timeout (seconds)
DEFAULT_AGENT_START_TIMEOUT_S = 10.0

# ejabberd docker-compose defaults (for local dev stack)
DEFAULT_EJABBERD_HOST = "localhost"
DEFAULT_EJABBERD_PORT = 5222

# ---------------------------------------------------------------------------
# Worker runtime configuration
# ---------------------------------------------------------------------------

# Default delay between environment steps for observation (seconds)
DEFAULT_STEP_DELAY_S = 0.14

# Telemetry buffer defaults (per run/agent)
DEFAULT_WORKER_TELEMETRY_BUFFER_SIZE = 2048
DEFAULT_WORKER_EPISODE_BUFFER_SIZE = 100

# Default epsilon schedule when switching to cached policies
DEFAULT_CACHED_POLICY_EPSILON = 0.0
DEFAULT_ONLINE_POLICY_EPSILON = 0.1

# ---------------------------------------------------------------------------
# Q-Learning algorithm defaults (centralized; used by BDI agent/runtime)
# ---------------------------------------------------------------------------

# Learning rate (alpha)
DEFAULT_Q_ALPHA = 0.1

# Discount factor (gamma)
DEFAULT_Q_GAMMA = 0.99

# Initial exploration rate for a fresh agent (epsilon)
DEFAULT_Q_EPSILON_INIT = 1.0

# Maximum number of steps per episode before truncation
DEFAULT_MAX_EPISODE_STEPS = 100

__all__ = [
    "DEFAULT_AGENT_JID",
    "DEFAULT_AGENT_PASSWORD",
    "DEFAULT_AGENT_START_TIMEOUT_S",
    "DEFAULT_EJABBERD_HOST",
    "DEFAULT_EJABBERD_PORT",
    "DEFAULT_STEP_DELAY_S",
    "DEFAULT_WORKER_TELEMETRY_BUFFER_SIZE",
    "DEFAULT_WORKER_EPISODE_BUFFER_SIZE",
    "DEFAULT_CACHED_POLICY_EPSILON",
    "DEFAULT_ONLINE_POLICY_EPSILON",
    # Q-Learning defaults (centralized)
    "DEFAULT_Q_ALPHA",
    "DEFAULT_Q_GAMMA",
    "DEFAULT_Q_EPSILON_INIT",
    "DEFAULT_MAX_EPISODE_STEPS",
]
