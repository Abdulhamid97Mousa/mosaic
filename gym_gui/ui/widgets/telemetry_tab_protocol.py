"""Protocol defining contract for telemetry tabs.

This module defines the TelemetryTab protocol that all telemetry display tabs
must implement. This ensures consistent behavior and makes it easier to add
new telemetry tab types.
"""

from typing import Any, Dict, Optional, Protocol


class TelemetryTab(Protocol):
    """Contract for telemetry display tabs.
    
    All telemetry tabs must implement this protocol to ensure consistent
    lifecycle management and data handling. This protocol defines the
    interface that the application expects from any telemetry tab.
    
    Implementing Classes:
    - AgentOnlineGridTab
    - AgentOnlineVideoTab
    - AgentOnlineRawTab
    - LiveTelemetryTab
    
    Example:
        def display_telemetry(tab: TelemetryTab, step_data: Dict[str, Any]) -> None:
            '''Display telemetry in any tab implementing the protocol.'''
            tab.on_step(step_data)
    """
    
    # Required attributes
    run_id: str
    """Unique identifier for the training run."""
    
    agent_id: str
    """Unique identifier for the agent."""
    
    def on_step(
        self,
        step: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when new step data arrives.
        
        This method is called for each step during training. Implementations
        should update their display based on the step data.
        
        Args:
            step: Step data dict with keys:
                - reward: float - The reward for this step
                - observation: Any - The observation from the environment
                - terminated: bool - Whether the episode terminated
                - truncated: bool - Whether the episode was truncated
                - info: dict - Additional info from the environment
                - seed: Optional[int] - Random seed used
                
            metadata: Optional metadata dict with run context:
                - run_id: str - The run identifier
                - agent_id: str - The agent identifier
                - timestamp: float - When the step occurred
                - episode: int - Current episode number
                - step_in_episode: int - Step number within episode
        
        Raises:
            Should not raise exceptions. Implementations should handle
            invalid data gracefully and log errors.
        """
        ...
    
    def on_episode_end(self, summary: Dict[str, Any]) -> None:
        """Called when episode finishes.
        
        This method is called when an episode completes. Implementations
        should update their display with episode summary information.
        
        Args:
            summary: Episode summary dict with keys:
                - total_reward: float - Total reward for the episode
                - steps: int - Number of steps in the episode
                - terminated: bool - Whether episode terminated normally
                - truncated: bool - Whether episode was truncated
                - seed: Optional[int] - Random seed used
                - control_mode: Optional[str] - Control mode (e.g., "HUMAN_ONLY")
        
        Raises:
            Should not raise exceptions. Implementations should handle
            invalid data gracefully and log errors.
        """
        ...
    
    def refresh(self) -> None:
        """Refresh display from current state.
        
        This method is called to refresh the display without new data.
        Useful for updating UI after configuration changes or when
        resuming from pause.
        
        Raises:
            Should not raise exceptions.
        """
        ...
    
    def clear(self) -> None:
        """Clear all displayed data.
        
        This method is called to clear the tab's display. Useful when
        starting a new run or resetting the UI.
        
        Raises:
            Should not raise exceptions.
        """
        ...

