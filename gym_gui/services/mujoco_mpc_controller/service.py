"""MuJoCo MPC Controller Service implementation.

This service manages the lifecycle of MuJoCo MPC sessions, including:
- Starting/stopping the MJPC agent server
- Managing task configuration
- Streaming visualization frames to the UI
- Exposing cost and trajectory telemetry
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gym_gui.core.mujoco_mpc_enums import (
    MuJoCoMPCPlannerType,
    MuJoCoMPCSessionState,
    MuJoCoMPCTaskId,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class MuJoCoMPCSession:
    """Represents an active MuJoCo MPC session.

    Attributes:
        session_id: Unique identifier for this session
        task_id: The MJPC task being run
        planner_type: The planner algorithm in use
        state: Current session state
        server_port: gRPC port the agent server is listening on
        total_cost: Current total cost from the planner
        step_count: Number of simulation steps executed
    """
    session_id: str
    task_id: str
    planner_type: MuJoCoMPCPlannerType
    state: MuJoCoMPCSessionState = MuJoCoMPCSessionState.IDLE
    server_port: Optional[int] = None
    total_cost: float = 0.0
    step_count: int = 0
    error_message: Optional[str] = None


@dataclass
class MuJoCoMPCControllerConfig:
    """Configuration for the MuJoCo MPC Controller Service.

    Attributes:
        default_task: Default task to load
        default_planner: Default planner algorithm
        real_time_speed: Default simulation speed ratio
        server_binary_path: Path to agent_server binary (None = auto-detect)
    """
    default_task: str = "Cartpole"
    default_planner: MuJoCoMPCPlannerType = MuJoCoMPCPlannerType.PREDICTIVE_SAMPLING
    real_time_speed: float = 1.0
    server_binary_path: Optional[str] = None


class MuJoCoMPCControllerService:
    """Service for managing MuJoCo MPC sessions.

    This service is responsible for:
    1. Launching and managing MJPC agent server processes
    2. Providing task selection and configuration
    3. Streaming visualization frames to the UI render area
    4. Exposing real-time telemetry (costs, trajectories)

    Usage:
        service = MuJoCoMPCControllerService()
        await service.start()

        session = await service.create_session(
            task_id="Cartpole",
            planner_type=MuJoCoMPCPlannerType.PREDICTIVE_SAMPLING,
        )

        # Stream frames to UI
        async for frame in service.stream_frames(session.session_id):
            render_widget.update_frame(frame)

        await service.stop_session(session.session_id)
        await service.shutdown()
    """

    def __init__(self, config: Optional[MuJoCoMPCControllerConfig] = None):
        """Initialize the MuJoCo MPC Controller Service.

        Args:
            config: Service configuration. If None, uses defaults.
        """
        self._config = config or MuJoCoMPCControllerConfig()
        self._sessions: dict[str, MuJoCoMPCSession] = {}
        self._agents: dict[str, Any] = {}  # session_id -> mujoco_mpc.Agent
        self._is_running = False
        self._frame_callbacks: dict[str, list[Callable]] = {}
        self._telemetry_callbacks: dict[str, list[Callable]] = {}

        _LOGGER.info("MuJoCoMPCControllerService initialized")

    async def start(self) -> None:
        """Start the service."""
        if self._is_running:
            _LOGGER.warning("Service already running")
            return

        self._is_running = True
        _LOGGER.info("MuJoCoMPCControllerService started")

    async def shutdown(self) -> None:
        """Shutdown the service and cleanup all sessions."""
        _LOGGER.info("Shutting down MuJoCoMPCControllerService...")

        # Stop all active sessions
        for session_id in list(self._sessions.keys()):
            await self.stop_session(session_id)

        self._is_running = False
        _LOGGER.info("MuJoCoMPCControllerService shutdown complete")

    async def create_session(
        self,
        task_id: str,
        planner_type: Optional[MuJoCoMPCPlannerType] = None,
        real_time_speed: float = 1.0,
    ) -> MuJoCoMPCSession:
        """Create a new MuJoCo MPC session.

        Args:
            task_id: MJPC task identifier (e.g., "Cartpole", "Humanoid Track")
            planner_type: Planner algorithm to use. If None, uses default.
            real_time_speed: Simulation speed ratio (0.0 to 1.0)

        Returns:
            The created MuJoCoMPCSession

        Raises:
            RuntimeError: If service is not running
            ValueError: If task_id is invalid
        """
        if not self._is_running:
            raise RuntimeError("Service is not running")

        import secrets
        session_id = f"mjpc_{secrets.token_hex(8)}"

        planner = planner_type or self._config.default_planner

        session = MuJoCoMPCSession(
            session_id=session_id,
            task_id=task_id,
            planner_type=planner,
            state=MuJoCoMPCSessionState.INITIALIZING,
        )
        self._sessions[session_id] = session

        _LOGGER.info(
            f"Creating MuJoCo MPC session: {session_id}, "
            f"task={task_id}, planner={planner.value}"
        )

        try:
            # TODO: Import and initialize mujoco_mpc.Agent
            # This will be implemented when we complete the agent_wrapper
            #
            # from mujoco_mpc import Agent
            # agent = Agent(
            #     task_id=task_id,
            #     real_time_speed=real_time_speed,
            #     server_binary_path=self._config.server_binary_path,
            # )
            # self._agents[session_id] = agent
            # session.server_port = agent.port
            # session.state = MuJoCoMPCSessionState.RUNNING

            _LOGGER.warning(
                "Agent initialization not yet implemented - session created in IDLE state"
            )
            session.state = MuJoCoMPCSessionState.IDLE

        except Exception as e:
            _LOGGER.error(f"Failed to create session: {e}")
            session.state = MuJoCoMPCSessionState.ERROR
            session.error_message = str(e)

        return session

    async def stop_session(self, session_id: str) -> None:
        """Stop and cleanup a MuJoCo MPC session.

        Args:
            session_id: The session to stop
        """
        if session_id not in self._sessions:
            _LOGGER.warning(f"Session not found: {session_id}")
            return

        _LOGGER.info(f"Stopping session: {session_id}")

        session = self._sessions[session_id]
        session.state = MuJoCoMPCSessionState.TERMINATED

        # Cleanup agent
        if session_id in self._agents:
            agent = self._agents.pop(session_id)
            try:
                agent.close()
            except Exception as e:
                _LOGGER.error(f"Error closing agent: {e}")

        # Remove callbacks
        self._frame_callbacks.pop(session_id, None)
        self._telemetry_callbacks.pop(session_id, None)

        # Remove session
        del self._sessions[session_id]

        _LOGGER.info(f"Session stopped: {session_id}")

    def get_session(self, session_id: str) -> Optional[MuJoCoMPCSession]:
        """Get a session by ID.

        Args:
            session_id: The session ID to look up

        Returns:
            The session if found, None otherwise
        """
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[MuJoCoMPCSession]:
        """List all active sessions.

        Returns:
            List of all MuJoCoMPCSession objects
        """
        return list(self._sessions.values())

    def get_available_tasks(self) -> list[MuJoCoMPCTaskId]:
        """Get list of available MJPC tasks.

        Returns:
            List of available task IDs
        """
        return list(MuJoCoMPCTaskId)

    def get_available_planners(self) -> list[MuJoCoMPCPlannerType]:
        """Get list of available planner types.

        Returns:
            List of available planner types
        """
        return list(MuJoCoMPCPlannerType)

    def register_frame_callback(
        self,
        session_id: str,
        callback: Callable[[bytes], None],
    ) -> None:
        """Register a callback for receiving visualization frames.

        Args:
            session_id: The session to receive frames from
            callback: Function to call with each frame (RGB bytes)
        """
        if session_id not in self._frame_callbacks:
            self._frame_callbacks[session_id] = []
        self._frame_callbacks[session_id].append(callback)

    def register_telemetry_callback(
        self,
        session_id: str,
        callback: Callable[[dict], None],
    ) -> None:
        """Register a callback for receiving telemetry updates.

        Args:
            session_id: The session to receive telemetry from
            callback: Function to call with telemetry dict
        """
        if session_id not in self._telemetry_callbacks:
            self._telemetry_callbacks[session_id] = []
        self._telemetry_callbacks[session_id].append(callback)

    async def step(self, session_id: str) -> None:
        """Execute a single simulation step.

        Args:
            session_id: The session to step
        """
        if session_id not in self._agents:
            _LOGGER.warning(f"No agent for session: {session_id}")
            return

        agent = self._agents[session_id]
        session = self._sessions[session_id]

        try:
            # Run planner and step physics
            agent.planner_step()
            agent.step()
            session.step_count += 1

            # Get telemetry
            session.total_cost = agent.get_total_cost()

            # Notify telemetry callbacks
            telemetry = {
                "step": session.step_count,
                "total_cost": session.total_cost,
                "cost_terms": agent.get_cost_term_values(),
            }
            for callback in self._telemetry_callbacks.get(session_id, []):
                callback(telemetry)

        except Exception as e:
            _LOGGER.error(f"Error during step: {e}")
            session.state = MuJoCoMPCSessionState.ERROR
            session.error_message = str(e)
