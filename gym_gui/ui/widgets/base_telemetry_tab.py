"""Base class for telemetry display tabs.

This module provides a common base class for all telemetry tabs to eliminate
code duplication and ensure consistent behavior across different tab types.
"""

from typing import Any, Dict, Optional

from qtpy import QtWidgets


class BaseTelemetryTab(QtWidgets.QWidget):
    """Abstract base for all telemetry tabs (grid, video, raw, live).

    This base class provides:
    - Common initialization for run_id and agent_id
    - Factory methods for standard UI components (header, stats group)
    - Lifecycle methods for events (on_step, on_episode_end)
    - Common interface for all telemetry tabs

    Subclasses must implement:
    - _build_ui(): Build the specific tab UI

    Subclasses may override:
    - on_step(): Handle new step data
    - on_episode_end(): Handle episode completion
    - refresh(): Refresh display
    - clear(): Clear all data
    """

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize base telemetry tab.

        Args:
            run_id: Unique identifier for the training run
            agent_id: Unique identifier for the agent
            parent: Parent widget
        """
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the tab's UI. Subclasses must implement this.

        This method is called during initialization and should construct
        all UI elements specific to the tab type.

        Subclasses should override this method to build their specific UI.
        """
        raise NotImplementedError("Subclasses must implement _build_ui()")
    
    def _build_header(self) -> QtWidgets.QHBoxLayout:
        """Factory method for standard header layout.
        
        Creates a consistent header showing run_id and agent_id across all tabs.
        The header includes:
        - Run ID (truncated to 12 chars)
        - Agent ID
        - Stretch space
        
        Returns:
            QHBoxLayout containing the header widgets
            
        Example:
            layout = QtWidgets.QVBoxLayout(self)
            layout.addLayout(self._build_header())
        """
        header = QtWidgets.QHBoxLayout()
        self._run_label = QtWidgets.QLabel(
            f"<b>Run:</b> {self.run_id[:12]}..."
        )
        self._agent_label = QtWidgets.QLabel(f"<b>Agent:</b> {self.agent_id}")
        header.addWidget(self._run_label)
        header.addWidget(self._agent_label)
        header.addStretch()
        return header
    
    def _build_stats_group(self) -> tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
        """Factory for standard stats group.
        
        Creates a consistent stats group box with grid layout for displaying
        training statistics across all tabs.
        
        Returns:
            Tuple of (QGroupBox, QGridLayout) for adding stats widgets
            
        Example:
            stats_group, stats_layout = self._build_stats_group()
            stats_layout.addWidget(QtWidgets.QLabel("Episodes:"), 0, 0)
            stats_layout.addWidget(self._episodes_label, 0, 1)
            layout.addWidget(stats_group)
        """
        group = QtWidgets.QGroupBox("Training Statistics", self)
        layout = QtWidgets.QGridLayout(group)
        return group, layout
    
    def on_step(
        self,
        step: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when new step data arrives.
        
        Subclasses may override to handle step updates. Default implementation
        does nothing.
        
        Args:
            step: Step data dict with keys like:
                - reward: float
                - observation: Any
                - terminated: bool
                - truncated: bool
                - info: dict
                - seed: Optional[int]
            metadata: Optional metadata dict with run context
        """
        pass
    
    def on_episode_end(self, summary: Dict[str, Any]) -> None:
        """Called when episode finishes.
        
        Subclasses may override to handle episode completion. Default
        implementation does nothing.
        
        Args:
            summary: Episode summary dict with keys like:
                - total_reward: float
                - steps: int
                - terminated: bool
                - truncated: bool
                - seed: Optional[int]
                - control_mode: Optional[str]
        """
        pass
    
    def refresh(self) -> None:
        """Refresh display from current state.
        
        Subclasses may override to refresh their display. Default
        implementation does nothing.
        """
        pass
    
    def clear(self) -> None:
        """Clear all displayed data.
        
        Subclasses may override to clear their data. Default implementation
        does nothing.
        """
        pass

