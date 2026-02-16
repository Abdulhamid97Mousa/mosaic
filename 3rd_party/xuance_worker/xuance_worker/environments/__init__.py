"""Custom environment wrappers for XuanCe training in MOSAIC.

This module provides XuanCe-compatible environment wrappers that extend
RawMultiAgentEnv for environments not natively supported by XuanCe.

These wrappers enable MOSAIC to train multi-agent RL policies on additional
environment families while maintaining compatibility with XuanCe's training
infrastructure (runners, agents, etc.).

Usage:
    # Register environments with XuanCe at startup
    from xuance_worker.environments import register_mosaic_environments
    register_mosaic_environments()

    # Or import specific environments
    from xuance_worker.environments import MultiGrid_Env
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

_REGISTERED = False


def register_mosaic_environments() -> None:
    """Register MOSAIC custom environments with XuanCe's registry.

    This function adds MOSAIC's environment wrappers to XuanCe's
    REGISTRY_MULTI_AGENT_ENV, enabling them to be used with XuanCe's
    training infrastructure.

    This should be called once at startup, typically in xuance_worker's
    runtime initialization.

    Example:
        >>> from xuance_worker.environments import register_mosaic_environments
        >>> register_mosaic_environments()
        >>> # Now 'multigrid' is available as an env_name in XuanCe configs
    """
    global _REGISTERED

    if _REGISTERED:
        _logger.debug("MOSAIC environments already registered with XuanCe")
        return

    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
        from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env

        # Register MultiGrid
        if "multigrid" not in REGISTRY_MULTI_AGENT_ENV:
            REGISTRY_MULTI_AGENT_ENV["multigrid"] = MultiGrid_Env
            REGISTRY_MULTI_AGENT_ENV["MultiGrid"] = MultiGrid_Env
            _logger.info("Registered MultiGrid_Env with XuanCe registry")

        _REGISTERED = True
        _logger.info("MOSAIC environments registered with XuanCe successfully")

    except ImportError as e:
        _logger.warning(f"Could not register MOSAIC environments with XuanCe: {e}")
    except Exception as e:
        _logger.error(f"Error registering MOSAIC environments: {e}")


def get_registered_environments() -> list[str]:
    """Return list of MOSAIC environments registered with XuanCe.

    Returns:
        List of environment names that have been registered.
    """
    registered = []
    try:
        from xuance.environment.multi_agent_env import REGISTRY_MULTI_AGENT_ENV
        for name in ["multigrid", "MultiGrid"]:
            if name in REGISTRY_MULTI_AGENT_ENV:
                registered.append(name)
    except ImportError:
        pass
    return registered


# Export for direct import
def _get_multigrid_env():
    """Lazy import of MultiGrid_Env."""
    from xuance_worker.environments.mosaic_multigrid import MultiGrid_Env
    return MultiGrid_Env


__all__ = [
    "register_mosaic_environments",
    "get_registered_environments",
]
