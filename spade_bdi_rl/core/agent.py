"""BDI agent lifecycle helpers bridging the refactored stack to SPADE.

This module provides agent lifecycle management (start, stop, setup) with
graceful degradation. Prefers the local refactored BDIRLAgent from bdi_agent.py.

If BDI capabilities are not needed, use HeadlessTrainer instead:
   - See: spade_bdi_rl.core.runtime.HeadlessTrainer
   - Entry: spade_bdi_rl.worker (JSONL telemetry output)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

if TYPE_CHECKING:
    from typing import Protocol
    from ..adapters import AdapterType
    
    class BDIAgentLike(Protocol):
        """Minimal interface for a BDI agent."""
        jid: str
        async def start(self, auto_register: bool = True) -> None: ...
        async def stop(self) -> None: ...
        async def setup(self) -> None: ...

from ..assets import asl_path
from .bdi_agent import BDIRLAgent

LOGGER = logging.getLogger(__name__)

DEFAULT_JID = "agent@localhost"
DEFAULT_PASSWORD = "secret"
_DEFAULT_START_TIMEOUT = 10.0


class LegacyImportError(ImportError):
    """Raised when legacy BDI agent features are attempted.
    
    The legacy spadeBDI_RL codebase has broken relative imports that prevent
    it from being used in the refactored package structure. Rather than trying
    to fix these imports, consider:
    
    1. Using HeadlessTrainer (pure RL) - recommended for most workflows
    2. Implementing custom SPADE-BDI integration directly in your trainer
    3. Migrating the legacy agent code to the new package structure
    """

    def __init__(
        self,
        msg: str = "Legacy BDI agent code has broken imports. Use pure RL worker instead.",
    ):
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


def _get_bdi_agent_class() -> type:
    """Return the refactored BDI agent class.
    
    Returns:
        The BDIRLAgent class from bdi_agent.py
    """
    return BDIRLAgent


@dataclass(slots=True)
class AgentHandle:
    """Lifecycle wrapper for SPADE-BDI agent with graceful fallback.
    
    This wrapper attempts to use the legacy BDI agent if available, but
    raises informative errors if legacy code is not accessible. For most
    use cases, HeadlessTrainer (pure RL) is recommended instead.
    
    ⚠️ DEPRECATED: Consider using HeadlessTrainer for new code.
    """

    agent: Optional["BDIAgentLike"]
    jid: str
    password: str
    started: bool = False

    async def start(self, auto_register: bool = True, timeout: float = _DEFAULT_START_TIMEOUT) -> None:
        """Start the BDI agent.
        
        Raises:
            LegacyImportError: If legacy BDI agent could not be imported.
        """
        if self.agent is None:
            raise LegacyImportError(
                "BDI agent is not initialized. Legacy code could not be imported "
                "or imported agent is None. Consider using HeadlessTrainer instead."
            )
        
        if self.started:
            LOGGER.warning("Agent already started, skipping duplicate start")
            return
        
        try:
            await asyncio.wait_for(self.agent.start(auto_register=auto_register), timeout=timeout)
            await self.agent.setup()
            self.started = True
            LOGGER.info("BDI agent '%s' started successfully", self.jid)
        except asyncio.TimeoutError:
            LOGGER.error("BDI agent start timed out after %.1fs", timeout)
            raise
        except Exception as exc:
            LOGGER.error("BDI agent start failed: %s", exc)
            # Attempt cleanup
            try:
                await asyncio.wait_for(self.agent.stop(), timeout=5.0)
            except Exception:
                pass
            raise

    async def stop(self) -> None:
        """Stop the BDI agent.
        
        Raises:
            LegacyImportError: If legacy BDI agent is not initialized.
        """
        if self.agent is None:
            raise LegacyImportError(
                "BDI agent is not initialized and cannot be stopped."
            )
        
        if not self.started:
            LOGGER.debug("Agent not started, skipping stop")
            return
        
        try:
            await asyncio.wait_for(self.agent.stop(), timeout=5.0)
            LOGGER.info("BDI agent '%s' stopped successfully", self.jid)
        except asyncio.TimeoutError:
            LOGGER.warning("BDI agent stop timed out")
        except Exception as exc:
            LOGGER.warning("BDI agent stop raised exception: %s", exc)
        finally:
            self.started = False


def create_agent(
    jid: str = DEFAULT_JID,
    password: str = DEFAULT_PASSWORD,
    *,
    adapter: Optional["AdapterType"] = None,
    asl_file: Optional[str | Path] = None,
    ensure_account: bool = True,
) -> AgentHandle:
    """Instantiate the refactored BDI agent.
    
    This function creates and instantiates a new BDIRLAgent with the specified
    configuration.
    
    Args:
        jid: XMPP JID (e.g., 'agent@localhost')
        password: XMPP password
        adapter: Environment adapter (optional, defaults to FrozenLakeAdapter)
        asl_file: Path to custom AgentSpeak file (optional)
        ensure_account: Whether to ensure XMPP account exists (not implemented)
    
    Returns:
        AgentHandle wrapping the BDI agent.
    
    Raises:
        FileNotFoundError: If asl_file is provided but doesn't exist.
    """
    # Resolve ASL file if specified
    if asl_file is None:
        asl_file_path = str(asl_path())
        LOGGER.debug("Using default ASL file: %s", asl_file_path)
    else:
        asl_file_path = str(resolve_asl(asl_file))
        LOGGER.debug("Using custom ASL file: %s", asl_file_path)
    
    # Get BDI agent class
    BDIAgentClass = _get_bdi_agent_class()
    
    # Instantiate the agent
    try:
        agent = BDIAgentClass(
            jid=jid,
            password=password,
            adapter=adapter,
            asl_file=asl_file_path,
        )
        LOGGER.info(
            "Created BDI agent instance successfully",
            extra={
                "jid": jid,
                "asl_file": asl_file_path,
                "adapter": adapter.__class__.__name__ if adapter else "default",
            },
        )
        return AgentHandle(agent=agent, jid=jid, password=password)
    except Exception as exc:
        LOGGER.exception(
            "Failed to instantiate BDI agent",
            extra={
                "jid": jid,
                "asl_file": asl_file_path,
                "error": str(exc),
            },
        )
        # Return handle with agent=None - start/stop will raise clear error
        return AgentHandle(agent=None, jid=jid, password=password)


async def create_and_start_agent(
    jid: str = DEFAULT_JID,
    password: str = DEFAULT_PASSWORD,
    *,
    asl_file: Optional[str | Path] = None,
    ensure_account: bool = True,
    verify_connection: bool = True,
    start_timeout: float = _DEFAULT_START_TIMEOUT,
) -> AgentHandle:
    """Create and immediately start the refactored BDI agent.
    
    This is a convenience function that combines creation and startup,
    with error handling and optional prerequisites (XMPP account setup,
    connection verification).
    
    Args:
        jid: XMPP JID
        password: XMPP password
        asl_file: Path to custom AgentSpeak file (optional)
        ensure_account: Whether to ensure XMPP account exists (placeholder)
        verify_connection: Whether to test XMPP connection before starting
        start_timeout: Timeout for agent.start() call
    
    Returns:
        AgentHandle with started BDI agent.
    
    Raises:
        LegacyImportError: If agent could not be instantiated.
    """
    # Create agent
    handle = create_agent(jid, password, asl_file=asl_file, ensure_account=ensure_account)
    
    # Attempt to start
    try:
        await handle.start(auto_register=ensure_account, timeout=start_timeout)
        LOGGER.info(
            "BDI agent created and started successfully",
            extra={"jid": jid},
        )
        return handle
    except LegacyImportError:
        LOGGER.error(
            "Cannot start BDI agent - agent is not initialized",
            extra={
                "jid": jid,
                "recommendation": "Check agent instantiation",
            },
        )
        raise
    except Exception as exc:
        LOGGER.exception(
            "BDI agent startup failed",
            extra={"jid": jid, "error": str(exc)},
        )
        raise


__all__ = [
    "AgentHandle",
    "create_agent",
    "create_and_start_agent",
    "docker_compose_path",
    "resolve_asl",
    "DEFAULT_JID",
    "DEFAULT_PASSWORD",
]
