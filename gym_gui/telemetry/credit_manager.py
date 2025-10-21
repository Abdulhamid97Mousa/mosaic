"""Credit-based backpressure system for telemetry flow control.

This module implements a lightweight credit system that prevents telemetry queue
overflow by controlling the producer rate. The UI advertises available capacity
("credits"), and the producer respects these credits before publishing events.

Key concepts:
- Credits represent "how many more events I can handle right now"
- UI sends CREDIT_GRANT messages when its queue drops below threshold
- Producer pauses UI path when credits=0 (but continues DB writes)
- This ensures no data loss while preventing UI overflow
"""

import logging
from typing import Dict, Tuple

_LOGGER = logging.getLogger(__name__)


class CreditManager:
    """Manages credit allocation per (run_id, agent_id) pair.
    
    Tracks available credits for each stream and provides methods to
    check, consume, and replenish credits.
    """

    def __init__(self, initial_credits: int = 200) -> None:
        """Initialize the credit manager.
        
        Args:
            initial_credits: Initial credit allocation per stream
        """
        self._initial_credits = initial_credits
        self._credits: Dict[Tuple[str, str], int] = {}  # (run_id, agent_id) -> credits
        self._total_dropped: Dict[Tuple[str, str], int] = {}  # Track total dropped per stream
        _LOGGER.info(f"CreditManager initialized with initial_credits={initial_credits}")

    def get_credits(self, run_id: str, agent_id: str) -> int:
        """Get current credits for a stream.
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
            
        Returns:
            Current credit count (0 if stream not initialized)
        """
        key = (run_id, agent_id)
        return self._credits.get(key, 0)

    def initialize_stream(self, run_id: str, agent_id: str) -> None:
        """Initialize credits for a new stream.
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
        """
        key = (run_id, agent_id)
        if key not in self._credits:
            self._credits[key] = self._initial_credits
            self._total_dropped[key] = 0
            _LOGGER.debug(
                f"Initialized credits for stream",
                extra={"run_id": run_id, "agent_id": agent_id, "credits": self._initial_credits},
            )

    def consume_credit(self, run_id: str, agent_id: str) -> bool:
        """Attempt to consume one credit.
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
            
        Returns:
            True if credit was available and consumed, False if no credits
        """
        key = (run_id, agent_id)
        self.initialize_stream(run_id, agent_id)
        
        if self._credits[key] > 0:
            self._credits[key] -= 1
            return True
        else:
            self._total_dropped[key] = self._total_dropped.get(key, 0) + 1
            return False

    def grant_credits(self, run_id: str, agent_id: str, amount: int) -> None:
        """Grant credits to a stream (called when UI queue drops below threshold).
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
            amount: Number of credits to grant
        """
        key = (run_id, agent_id)
        self.initialize_stream(run_id, agent_id)
        
        old_credits = self._credits[key]
        self._credits[key] = min(self._credits[key] + amount, self._initial_credits * 2)
        
        _LOGGER.debug(
            f"Granted credits to stream",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "amount": amount,
                "old_credits": old_credits,
                "new_credits": self._credits[key],
            },
        )

    def get_dropped_count(self, run_id: str, agent_id: str) -> int:
        """Get total events dropped due to no credits.
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
            
        Returns:
            Total dropped count
        """
        key = (run_id, agent_id)
        return self._total_dropped.get(key, 0)

    def reset_stream(self, run_id: str, agent_id: str) -> None:
        """Reset credits for a stream (e.g., when run completes).
        
        Args:
            run_id: Training run ID
            agent_id: Agent ID
        """
        key = (run_id, agent_id)
        self._credits.pop(key, None)
        self._total_dropped.pop(key, None)
        _LOGGER.debug(
            f"Reset credits for stream",
            extra={"run_id": run_id, "agent_id": agent_id},
        )

    def get_stats(self) -> Dict[str, int]:
        """Get credit statistics for all streams.
        
        Returns:
            Dict mapping "run_id:agent_id" to current credits
        """
        return {
            f"{run_id}:{agent_id}": credits
            for (run_id, agent_id), credits in self._credits.items()
        }


# Global singleton instance
_credit_manager: CreditManager | None = None


def get_credit_manager() -> CreditManager:
    """Get or create the global CreditManager singleton.
    
    Returns:
        The global CreditManager instance
    """
    global _credit_manager
    if _credit_manager is None:
        _credit_manager = CreditManager()
    return _credit_manager


def reset_credit_manager() -> None:
    """Reset the global CreditManager (for testing)."""
    global _credit_manager
    _credit_manager = None

