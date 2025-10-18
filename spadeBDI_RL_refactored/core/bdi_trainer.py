"""BDI-RL training loop integrating SPADE-BDI agents with RL algorithms.

This module provides the BDITrainer class which extends HeadlessTrainer with
BDI reasoning capabilities using SPADE agents and AgentSpeak plans.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from ..adapters import AdapterType
else:
    # At runtime, we need the actual types for isinstance checks and proper type hints
    from ..adapters import FrozenLakeAdapter, FrozenLakeV2Adapter, CliffWalkingAdapter, TaxiAdapter
    AdapterType = Union[FrozenLakeAdapter, FrozenLakeV2Adapter, CliffWalkingAdapter, TaxiAdapter]

from .config import RunConfig
from .runtime import HeadlessTrainer, EpisodeMetrics
from .telemetry import TelemetryEmitter

LOGGER = logging.getLogger(__name__)


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

        LOGGER.info(
            "BDITrainer initialized",
            extra={
                "jid": jid,
                "env_id": config.env_id,
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
        self.emitter.run_started(self.config.run_id, config_payload)

        try:
            # Start BDI agent
            self._start_bdi_agent()

            # Run episodes with BDI integration
            summaries: list[EpisodeMetrics] = []
            for episode in range(self.config.max_episodes):
                summary = self._run_bdi_episode(episode)
                summaries.append(summary)

                episode_metadata = self._build_episode_metadata(episode, summary)
                self.emitter.episode(
                    self.config.run_id,
                    episode,
                    reward=summary.total_reward,
                    steps=summary.steps,
                    success=summary.success,
                    metadata=episode_metadata,
                )

            # Save policy if configured
            if self._should_save:
                metadata = self._build_policy_metadata()
                path = self.policy_store.save(self.agent.q_table, metadata)
                self.emitter.artifact(self.config.run_id, "policy", str(path))

            self.emitter.run_completed(self.config.run_id, status="completed")
            return 0

        except Exception as exc:
            LOGGER.exception("BDI training failed", exc_info=exc)
            self.emitter.run_completed(
                self.config.run_id,
                status="failed",
                error=str(exc),
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
            
            LOGGER.info(
                "BDI agent started successfully",
                extra={
                    "jid": self.jid,
                    "env_id": self.config.env_id,
                    "asl_file": self.asl_file or "default",
                },
            )
        except Exception as exc:
            LOGGER.error("Failed to start BDI agent: %s", exc, exc_info=True)
            raise

    def _on_agent_event(self, event: Dict[str, Any]) -> None:
        """Handle events emitted by the BDI agent."""
        # This callback bridges BDI agent events to telemetry
        # Can be extended for custom telemetry integration
        LOGGER.debug("BDI agent event", extra=event)

    def _stop_bdi_agent(self) -> None:
        """Stop and cleanup the SPADE-BDI agent."""
        if self.bdi_agent is not None:
            LOGGER.info("Stopping BDI agent", extra={"jid": self.jid})
            try:
                if self._event_loop is not None and not self._event_loop.is_closed():
                    self._event_loop.run_until_complete(self.bdi_agent.stop())
                    self._event_loop.close()
            except Exception as exc:
                LOGGER.warning("Error stopping BDI agent: %s", exc)
            finally:
                self.bdi_agent = None
                self._event_loop = None

    def _run_bdi_episode(self, episode_index: int) -> EpisodeMetrics:
        """Run single episode with BDI reasoning integration.

        This extends the base episode runner with BDI agent execution.

        Args:
            episode_index: Zero-based episode counter

        Returns:
            Episode metrics (reward, steps, success)
        """
        # For now, we delegate to the base RL implementation which properly emits step telemetry
        # The BDI agent integration with proper step-by-step telemetry will be added
        # once the SPADE agent interface is fully integrated

        if self.bdi_agent is None:
            LOGGER.warning("BDI agent not initialized; falling back to RL-only mode")
            return self._run_episode(episode_index)

        # TODO: When BDI agent is fully integrated, this should:
        # 1. Consult BDI agent beliefs/desires/intentions before each step
        # 2. Let AgentSpeak plans influence action selection
        # 3. Emit step telemetry via self.emitter.step() for each step
        # 4. Update BDI beliefs after environment transitions

        # For now, use base RL implementation to ensure telemetry works
        LOGGER.debug(
            "Running BDI episode (RL mode with telemetry)",
            extra={"episode": episode_index, "note": "BDI reasoning pending full SPADE integration"},
        )

        return self._run_episode(episode_index)

    def _build_config_payload(self) -> Dict[str, Any]:
        """Build configuration payload with BDI-specific fields."""
        return {
            "run_id": self.config.run_id,
            "env_id": self.config.env_id,
            "seed": self.config.seed,
            "max_episodes": self.config.max_episodes,
            "max_steps_per_episode": self.config.max_steps_per_episode,
            "policy_strategy": self.config.policy_strategy.value,
            "policy_path": str(self.config.ensure_policy_path()),
            "agent_id": self.config.agent_id,
            "capture_video": self.config.capture_video,
            "headless": self.config.headless,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
            "asl_file": self.asl_file or "default",
            "extra": self.config.extra,
        }

    def _build_episode_metadata(self, episode: int, summary: EpisodeMetrics) -> Dict[str, Any]:
        """Build episode metadata with BDI-specific fields."""
        episode_seed = self.config.seed + episode
        return {
            "control_mode": "bdi_agent",
            "run_id": self.config.run_id,
            "agent_id": self.config.agent_id,
            "env_id": self.config.env_id,
            "seed": episode_seed,
            "episode_index": episode,
            "policy_strategy": self.config.policy_strategy.value,
            "success": summary.success,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
        }

    def _build_policy_metadata(self) -> Dict[str, Any]:
        """Build policy save metadata with BDI-specific fields."""
        return {
            "run_id": self.config.run_id,
            "env_id": self.config.env_id,
            "agent_id": self.config.agent_id,
            "episodes": self.config.max_episodes,
            "strategy": self.config.policy_strategy.value,
            "bdi_enabled": True,
            "bdi_jid": self.jid,
            "asl_file": self.asl_file or "default",
        }
