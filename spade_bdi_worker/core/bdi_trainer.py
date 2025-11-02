"""BDI-RL training loop integrating SPADE-BDI agents with RL algorithms.

This module provides the BDITrainer class which extends HeadlessTrainer with
BDI reasoning capabilities using SPADE agents and AgentSpeak plans.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from ..adapters import AdapterType

from .config import RunConfig
from .runtime import HeadlessTrainer, EpisodeMetrics
from .telemetry_worker import TelemetryEmitter
from gym_gui.logging_config.log_constants import (
    LOG_WORKER_BDI_EVENT,
    LOG_WORKER_BDI_WARNING,
    LOG_WORKER_BDI_ERROR,
    LOG_WORKER_BDI_DEBUG,
)
from gym_gui.logging_config.helpers import log_constant
import logging

_LOGGER = logging.getLogger(__name__)


class BDITrainer(HeadlessTrainer):
    """Training loop with BDI agent integration via SPADE.

    This trainer extends HeadlessTrainer with SPADE-BDI agent capabilities,
    allowing AgentSpeak plans to guide the RL training process.

    Args:
        adapter: Environment adapter (FrozenLake, CliffWalking, Taxi, etc.)
        config: Run configuration
        emitter: Telemetry emitter for JSONL output
        jid: XMPP JID for SPADE agent (e.g., 'agent@localhost')
        password: XMPP password
        asl_file: Optional path to custom AgentSpeak (.asl) file
    """

    def __init__(
        self,
        adapter: AdapterType,
        config: RunConfig,
        emitter: TelemetryEmitter,
        *,
        jid: str = "agent@localhost",
        password: str = "secret",
        asl_file: Optional[str] = None,
    ) -> None:
        # Initialize base trainer
        super().__init__(adapter, config, emitter)

        # BDI-specific configuration
        self.jid = jid
        self.password = password
        self.asl_file = asl_file
        self.bdi_agent: Optional[Any] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        self.log_constant(
            LOG_WORKER_BDI_EVENT,
            message="BDITrainer initialized",
            extra={
                "jid": jid,
                "game_id": config.game_id,
                "asl_file": asl_file or "default",
            },
        )

    def run(self) -> int:
        """Execute BDI-guided training/evaluation with JSONL telemetry.

        This method:
        1. Starts the SPADE-BDI agent
        2. Runs episodes with BDI reasoning integration
        3. Stops the agent and cleans up

        Returns:
            0 on success, non-zero on failure
        """
        config_payload = self._build_config_payload()
        self.emitter.run_started(
            self.config.run_id,
            config_payload,
            worker_id=self.config.worker_id,
        )

        try:
            # Start BDI agent
            self._start_bdi_agent()

            # Run episodes with BDI integration
            summaries: list[EpisodeMetrics] = []
            for episode_index in range(self.config.max_episodes):
                # CRITICAL: Separation of concerns for reproducibility and environment variation
                #
                # seed (config.seed): Base seed for reproducible experiment (e.g., seed=1)
                #   - Constant throughout the run
                #   - Used for reproducibility across runs
                #   - Does NOT change per episode
                #
                # episode_number: Display value for user (seed + episode_index)
                #   - For seed=1: episodes 1, 2, 3, 4, ...
                #   - For seed=39: episodes 39, 40, 41, 42, ...
                #
                # episode_seed: Unique seed for environment variation per episode
                #   - Derived from episode_index (0, 1, 2, 3, ...)
                #   - Each episode gets different environment state
                #   - Allows agent to learn from diverse experiences
                #
                episode_number = self.config.seed + episode_index
                episode_seed = episode_index  # Unique seed per episode (0, 1, 2, 3, ...)
                summary = self._run_bdi_episode(episode_index, episode_number, episode_seed)
                summaries.append(summary)

                episode_metadata = self._build_episode_metadata(episode_index, summary, episode_seed)
                self.emitter.episode(
                    self.config.run_id,
                    episode_number,  # Pass episode_number (display value = seed + episode_index)
                    reward=summary.total_reward,
                    steps=summary.steps,
                    success=summary.success,
                    metadata=episode_metadata,
                    worker_id=self.config.worker_id,
                )

            # Save policy if configured
            if self._should_save:
                metadata = self._build_policy_metadata()
                path = self.policy_store.save(self.agent.q_table, metadata)
                self.emitter.artifact(
                    self.config.run_id,
                    "policy",
                    str(path),
                    worker_id=self.config.worker_id,
                )

            self.emitter.run_completed(
                self.config.run_id,
                status="completed",
                worker_id=self.config.worker_id,
            )
            return 0

        except Exception as exc:
            self.log_constant(
                LOG_WORKER_BDI_ERROR,
                message="BDI training failed",
                exc_info=exc,
            )
            self.emitter.run_completed(
                self.config.run_id,
                status="failed",
                error=str(exc),
                worker_id=self.config.worker_id,
            )
            return 1

        finally:
            self._stop_bdi_agent()

    def _start_bdi_agent(self) -> None:
        """Initialize and start the SPADE-BDI agent."""
        try:
            # Create BDI-RL agent with environment adapter
            # Type ignore: adapter is dynamically typed but create_agent accepts AdapterType
            from .bdi_agent import create_agent  # Import here to avoid errors if SPADE is not available

            agent_handle = create_agent(
                jid=self.jid,
                password=self.password,
                adapter=self.adapter,  # type: ignore
                asl_file=self.asl_file,
            )
            
            self.bdi_agent = agent_handle.agent
            
            # Set event sink for telemetry (optional)
            if self.bdi_agent is not None:
                self.bdi_agent.set_event_sink(self._on_agent_event)
            
            # Start SPADE agent (async)
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_until_complete(agent_handle.start(auto_register=True))
            
            self.log_constant(
                LOG_WORKER_BDI_EVENT,
                message="BDI agent started successfully",
                extra={
                    "jid": self.jid,
                    "game_id": self.config.game_id,
                    "asl_file": self.asl_file or "default",
                },
            )
        except Exception as exc:
            self.log_constant(
                LOG_WORKER_BDI_WARNING,
                message="BDI agent unavailable, falling back to RL-only mode",
                extra={"exception": str(exc)},
                exc_info=exc,
            )
            self.bdi_agent = None

    def _on_agent_event(self, event: Dict[str, Any]) -> None:
        """Handle events emitted by the BDI agent."""
        # This callback bridges BDI agent events to telemetry
        # Can be extended for custom telemetry integration
        self.log_constant(
            LOG_WORKER_BDI_DEBUG,
            message="BDI agent event",
            extra=dict(event),
        )

    def _stop_bdi_agent(self) -> None:
        """Stop and cleanup the SPADE-BDI agent."""
        if self.bdi_agent is not None:
            self.log_constant(
                LOG_WORKER_BDI_EVENT,
                message="Stopping BDI agent",
                extra={"jid": self.jid},
            )
            try:
                if self._event_loop is not None and not self._event_loop.is_closed():
                    self._event_loop.run_until_complete(self.bdi_agent.stop())
                    self._event_loop.close()
            except Exception as exc:
                self.log_constant(
                    LOG_WORKER_BDI_WARNING,
                    message="Error stopping BDI agent",
                    extra={"exception": str(exc)},
                    exc_info=exc,
                )
            finally:
                self.bdi_agent = None
                self._event_loop = None

    def _run_bdi_episode(self, episode_index: int, episode_number: int, episode_seed: int) -> EpisodeMetrics:
        """Run single episode with BDI reasoning integration.

        This extends the base episode runner with BDI agent execution.

        Args:
            episode_index: 0-based loop counter (0, 1, 2, 3, ...)
            episode_number: Display value for telemetry (seed + episode_index)
            episode_seed: Unique seed for environment variation (derived from episode_index)

        Returns:
            Episode metrics (reward, steps, success)
        """
        # For now, we delegate to the base RL implementation which properly emits step telemetry
        # The BDI agent integration with proper step-by-step telemetry will be added
        # once the SPADE agent interface is fully integrated

        if self.bdi_agent is None:
            log_constant(_LOGGER, LOG_WORKER_BDI_WARNING, message="BDI agent not initialized; falling back to RL-only mode")
            return self._run_episode(episode_index, episode_number, episode_seed)

        # TODO: When BDI agent is fully integrated, this should:
        # 1. Consult BDI agent beliefs/desires/intentions before each step
        # 2. Let AgentSpeak plans influence action selection
        # 3. Emit step telemetry via self.emitter.step() for each step
        # 4. Update BDI beliefs after environment transitions

        # For now, use base RL implementation to ensure telemetry works
        log_constant(_LOGGER, LOG_WORKER_BDI_DEBUG, message="Running BDI episode (RL mode with telemetry)", extra={"episode_number": episode_number, "note": "BDI reasoning pending full SPADE integration"})

        return self._run_episode(episode_index, episode_number, episode_seed)

    def _build_config_payload(self) -> Dict[str, Any]:
        """Build configuration payload with BDI-specific fields."""
        return {
            "run_id": self.config.run_id,
            "game_id": self.config.game_id,
            "seed": self.config.seed,
            "max_episodes": self.config.max_episodes,
            "max_steps_per_episode": self.config.max_steps_per_episode,
            "policy_strategy": self.config.policy_strategy.value,
            "policy_path": str(self.config.ensure_policy_path()),
            "agent_id": self.config.agent_id,
            "worker_id": self.config.worker_id,
            "capture_video": self.config.capture_video,
            "headless": self.config.headless,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
            "asl_file": self.asl_file or "default",
            "extra": self.config.extra,
        }

    def _build_episode_metadata(self, episode_index: int, summary: EpisodeMetrics, episode_seed: int) -> Dict[str, Any]:
        """Build episode metadata with BDI-specific fields."""
        # episode_index = 0-based loop counter (0, 1, 2, 3, ...)
        # episode_seed = unique seed per episode (0, 1, 2, 3, ...) for environment variation
        # seed = base seed (self.config.seed) for reproducibility
        # episode (display) = seed + episode_index
        episode = self.config.seed + episode_index
        metadata = {
            "control_mode": "bdi_agent",
            "run_id": self.config.run_id,
            "agent_id": self.config.agent_id,
            "worker_id": self.config.worker_id,
            "game_id": self.config.game_id,
            "seed": self.config.seed,  # Base seed for reproducibility
            "episode_index": episode_index,  # 0-based counter
            "episode": episode,  # Display value (seed + episode_index)
            "episode_seed": episode_seed,  # Unique seed per episode for environment variation
            "policy_strategy": self.config.policy_strategy.value,
            "success": summary.success,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
        }
        game_config_snapshot = self._game_config_snapshot()
        if game_config_snapshot is not None:
            metadata["game_config"] = dict(game_config_snapshot)
        return metadata

    def _game_config_snapshot(self) -> Optional[Dict[str, Any]]:
        """Capture immutable snapshot of the active game configuration.

        Returns a plain dictionary suitable for telemetry metadata or ``None``
        when the adapter does not expose any configuration payload."""

        config_obj = getattr(self.adapter, "_game_config", None)
        if config_obj is None:
            config_obj = getattr(self.adapter, "defaults", None)
        if config_obj is None:
            return None

        try:
            if is_dataclass(config_obj) and not isinstance(config_obj, type):
                snapshot: Dict[str, Any] = asdict(config_obj)
            elif is_dataclass(config_obj) and isinstance(config_obj, type):
                snapshot = {
                    field_name: getattr(config_obj, field_name)
                    for field_name in getattr(config_obj, "__dataclass_fields__", {})
                }
            elif isinstance(config_obj, Mapping):
                snapshot = dict(config_obj)
            else:
                snapshot = {}
                for attribute in dir(config_obj):
                    if attribute.startswith("_"):
                        continue
                    try:
                        value = getattr(config_obj, attribute)
                    except AttributeError:
                        continue
                    if callable(value):
                        continue
                    snapshot[attribute] = value
        except Exception:  # noqa: BLE001 - defensive snapshotting
            return None

        def _normalise(value: Any) -> Any:
            if isinstance(value, tuple):
                return [_normalise(item) for item in value]
            if isinstance(value, list):
                return [_normalise(item) for item in value]
            if isinstance(value, Mapping):
                return {k: _normalise(v) for k, v in value.items()}
            return value

        normalised = {key: _normalise(val) for key, val in snapshot.items()}
        return normalised if normalised else None

    def _build_policy_metadata(self) -> Dict[str, Any]:
        """Build policy save metadata with BDI-specific fields."""
        return {
            "run_id": self.config.run_id,
            "game_id": self.config.game_id,
            "agent_id": self.config.agent_id,
            "worker_id": self.config.worker_id,
            "episodes": self.config.max_episodes,
            "strategy": self.config.policy_strategy.value,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
            "asl_file": self.asl_file or "default",
        }
