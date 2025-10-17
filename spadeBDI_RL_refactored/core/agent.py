"""BDI agent lifecycle helpers bridging the refactored stack to SPADE.

⚠️ DEPRECATED: This module has broken imports from legacy spadeBDI_RL code.
   The legacy codebase uses relative imports that don't work when imported
   from the refactored package. See 1.0_DAY_8_BDI_AGENT_TESTING_FINDINGS.md
   for details.

   USE: spadeBDI_RL_refactored.worker with HeadlessTrainer instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# DISABLED: Legacy imports are broken
# from spadeBDI_RL.src import spade_bdi_rl_agent as legacy_agent

from ..assets import asl_path

LOGGER = logging.getLogger(__name__)

DEFAULT_JID = "agent@localhost"
DEFAULT_PASSWORD = "secret"
_DEFAULT_START_TIMEOUT = 10.0


class LegacyImportError(ImportError):
    """Raised when trying to use legacy BDI agent features."""

    def __init__(self, msg: str = "Legacy BDI agent code has broken imports. Use pure RL worker instead."):
        super().__init__(msg)


def docker_compose_path() -> Path:
    """Return the bundled docker-compose manifest for ejabberd."""

    return Path(__file__).resolve().parents[1] / "infrastructure" / "docker-compose.yaml"


def resolve_asl(path: Optional[str | Path] = None) -> Path:
    """Resolve the AgentSpeak source used by the SPADE-BDI agent."""

    candidate = Path(path).expanduser().resolve() if path else asl_path()
    if not candidate.exists():
        raise FileNotFoundError(f"ASL file not found at {candidate}")
    return candidate


@dataclass(slots=True)
class AgentHandle:
    """Simple lifecycle wrapper over the legacy BDI+RL agent.
    
    ⚠️ DEPRECATED: Cannot be instantiated due to legacy import issues.
    """

    agent: None  # Was: legacy_agent.BDIRLAgentOnlinePolicy
    jid: str
    password: str
    started: bool = False

    async def start(self, auto_register: bool = True, timeout: float = _DEFAULT_START_TIMEOUT) -> None:
        raise LegacyImportError("BDI agent cannot be started - legacy code has broken imports")
        if self.started:
            return
        try:
            await asyncio.wait_for(self.agent.start(auto_register=auto_register), timeout=timeout)
            await self.agent.setup()
        except Exception:  # noqa: BLE001 - propagate after cleanup
            with contextlib.suppress(Exception):
                await self.agent.stop()
            raise
        else:
            self.started = True
            LOGGER.info("BDI agent '%s' started", self.jid)

    async def stop(self) -> None:
        if not self.started:
            return
        try:
            await self.agent.stop()
        finally:
            self.started = False
            LOGGER.info("BDI agent '%s' stopped", self.jid)


def create_agent(
    jid: str = DEFAULT_JID,
    password: str = DEFAULT_PASSWORD,
    *,
    asl_file: Optional[str | Path] = None,
    ensure_account: bool = True,
) -> AgentHandle:
    """Instantiate the legacy BDI agent with refactored defaults.
    
    ⚠️ DEPRECATED: Raises LegacyImportError due to broken imports.
    """
    raise LegacyImportError("Cannot create BDI agent - use HeadlessTrainer with pure RL instead")


async def create_and_start_agent(
    jid: str = DEFAULT_JID,
    password: str = DEFAULT_PASSWORD,
    *,
    asl_file: Optional[str | Path] = None,
    ensure_account: bool = True,
    verify_connection: bool = True,
    start_timeout: float = _DEFAULT_START_TIMEOUT,
) -> AgentHandle:
    """Instantiate and start the BDI agent, ensuring prerequisites when requested.
    
    ⚠️ DEPRECATED: Raises LegacyImportError due to broken imports.
    """
    raise LegacyImportError("Cannot create and start BDI agent - use HeadlessTrainer with pure RL instead")


__all__ = [
    "AgentHandle",
    "create_agent",
    "create_and_start_agent",
    "docker_compose_path",
    "resolve_asl",
    "DEFAULT_JID",
    "DEFAULT_PASSWORD",
]
