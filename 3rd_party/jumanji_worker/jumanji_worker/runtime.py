"""Runtime orchestration for Jumanji training.

This module provides the JumanjiWorkerRuntime class that wraps Jumanji's
native JAX-based A2C training with MOSAIC's telemetry and analytics
infrastructure.

The runtime supports:
- A2C (Advantage Actor-Critic) training using Jumanji's implementation
- Random agent baseline for testing
- GPU/TPU acceleration via JAX
- TelemetryEmitter integration for lifecycle events
- Analytics manifest generation
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import JumanjiWorkerConfig

# MOSAIC imports
try:
    from gym_gui.core.worker import TelemetryEmitter
    from gym_gui.config.paths import VAR_TRAINER_DIR, ensure_var_directories
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    TelemetryEmitter = None
    VAR_TRAINER_DIR = Path("var/trainer")

    def ensure_var_directories():
        pass

# JAX imports (lazy to handle missing JAX gracefully)
try:
    import jax
    import jax.numpy as jnp
    import optax
    import jumanji
    from jumanji.wrappers import VmapAutoResetWrapper
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None
    jnp = None
    optax = None
    jumanji = None

LOGGER = logging.getLogger(__name__)


class JumanjiWorkerRuntime:
    """Orchestrate Jumanji algorithm execution.

    This runtime wraps Jumanji's A2C training infrastructure and integrates
    with MOSAIC's telemetry and analytics systems.

    Example:
        >>> config = JumanjiWorkerConfig(
        ...     run_id="test_run",
        ...     env_id="Game2048-v1",
        ...     agent="a2c",
        ...     num_epochs=100,
        ... )
        >>> runtime = JumanjiWorkerRuntime(config)
        >>> summary = runtime.run()
    """

    def __init__(
        self,
        config: JumanjiWorkerConfig,
        *,
        dry_run: bool = False,
    ) -> None:
        """Initialize runtime.

        Args:
            config: Worker configuration
            dry_run: If True, validate config without executing
        """
        self._config = config
        self._dry_run = dry_run

        # Initialize telemetry emitter
        if _HAS_GYM_GUI:
            self._emitter = TelemetryEmitter(run_id=config.run_id, logger=LOGGER)
        else:
            self._emitter = None

    @property
    def config(self) -> JumanjiWorkerConfig:
        """Get the configuration."""
        return self._config

    @property
    def dry_run(self) -> bool:
        """Check if this is a dry run."""
        return self._dry_run

    def _setup_jax_device(self) -> None:
        """Configure JAX device based on config."""
        if not _HAS_JAX:
            raise ImportError(
                "JAX is required for Jumanji training. "
                "Install with: pip install jax jaxlib"
            )

        device = self._config.device.lower()

        if device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
            LOGGER.info("JAX configured for CPU")
        elif device == "gpu":
            # JAX will automatically use GPU if available
            devices = jax.devices("gpu")
            if not devices:
                LOGGER.warning("No GPU found, falling back to CPU")
                jax.config.update("jax_platform_name", "cpu")
            else:
                LOGGER.info(f"JAX using {len(devices)} GPU(s)")
        elif device == "tpu":
            devices = jax.devices("tpu")
            if not devices:
                LOGGER.warning("No TPU found, falling back to CPU")
                jax.config.update("jax_platform_name", "cpu")
            else:
                LOGGER.info(f"JAX using {len(devices)} TPU(s)")

    def _env_name_from_id(self, env_id: str) -> str:
        """Extract environment name from ID (e.g., 'Game2048-v1' -> 'game_2048')."""
        import re
        name = env_id.split("-v")[0]
        # Convert CamelCase to snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def run(self) -> Dict[str, Any]:
        """Execute the configured Jumanji training.

        Implements the WorkerRuntime protocol.

        Returns:
            Dictionary containing execution results:
            - status: "completed" | "failed" | "dry-run"
            - env_id: Environment ID
            - agent: Agent type
            - epochs_completed: Number of epochs completed
            - config: Configuration dict
        """
        # Emit run_started
        if self._emitter:
            self._emitter.run_started({
                "worker_type": "jumanji",
                "env_id": self._config.env_id,
                "agent": self._config.agent,
                "num_epochs": self._config.num_epochs,
                "device": self._config.device,
            })

        if self._dry_run:
            LOGGER.info("Dry-run mode | env_id=%s", self._config.env_id)
            summary = {
                "status": "dry-run",
                "env_id": self._config.env_id,
                "agent": self._config.agent,
                "config": self._config.to_dict(),
            }
            if self._emitter:
                self._emitter.run_completed(summary)
            return summary

        try:
            self._setup_jax_device()

            # Setup directories
            if _HAS_GYM_GUI:
                ensure_var_directories()
            run_dir = (VAR_TRAINER_DIR / "runs" / self._config.run_id).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            LOGGER.info(
                "Starting Jumanji training | env_id=%s agent=%s epochs=%d device=%s",
                self._config.env_id,
                self._config.agent,
                self._config.num_epochs,
                self._config.device,
            )

            result = self._run_training(run_dir)

            # Generate analytics manifest
            self._write_analytics_manifest(run_dir)

            # Emit run_completed
            if self._emitter:
                self._emitter.run_completed({
                    "status": "completed",
                    "env_id": self._config.env_id,
                    "agent": self._config.agent,
                    "epochs_completed": self._config.num_epochs,
                })

            LOGGER.info("Training completed | env_id=%s", self._config.env_id)

            return result

        except Exception as e:
            LOGGER.error("Jumanji training failed: %s", e, exc_info=True)

            if self._emitter:
                self._emitter.run_failed({
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

            raise

    def _run_training(self, run_dir: Path) -> Dict[str, Any]:
        """Execute the actual training loop.

        Args:
            run_dir: Directory for run outputs

        Returns:
            Training result dictionary
        """
        # Create environment
        env = jumanji.make(self._config.env_id)
        wrapped_env = VmapAutoResetWrapper(env)

        LOGGER.info("Created environment: %s", self._config.env_id)

        # Initialize PRNG
        key = jax.random.PRNGKey(self._config.seed or 0)

        if self._config.agent == "a2c":
            return self._run_a2c_training(wrapped_env, key, run_dir)
        else:
            return self._run_random_baseline(wrapped_env, key, run_dir)

    def _run_a2c_training(
        self,
        env,
        key: jax.Array,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Run A2C training using Jumanji's implementation.

        Args:
            env: Wrapped Jumanji environment
            key: JAX PRNG key
            run_dir: Run output directory

        Returns:
            Training result dictionary
        """
        from jumanji.training.agents.a2c import A2CAgent

        env_name = self._env_name_from_id(self._config.env_id)

        # Get actor-critic networks for this environment
        actor_critic_networks = self._get_actor_critic_networks(env_name, env.unwrapped)

        # Setup optimizer
        optimizer = optax.adam(self._config.learning_rate)

        # Create A2C agent
        agent = A2CAgent(
            env=env,
            n_steps=self._config.n_steps,
            total_batch_size=self._config.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=self._config.normalize_advantage,
            discount_factor=self._config.discount_factor,
            bootstrapping_factor=self._config.bootstrapping_factor,
            l_pg=self._config.l_pg,
            l_td=self._config.l_td,
            l_en=self._config.l_en,
        )

        LOGGER.info("Created A2C agent with networks for %s", env_name)

        # Initialize training state
        key, init_key = jax.random.split(key)
        training_state = self._init_training_state(env, agent, init_key)

        # Calculate steps per epoch
        steps_per_epoch = (
            self._config.n_steps
            * self._config.total_batch_size
            * self._config.num_learner_steps_per_epoch
        )

        # Training loop
        all_metrics = []
        start_time = time.monotonic()

        for epoch in range(self._config.num_epochs):
            epoch_start = time.monotonic()

            # Run one epoch
            training_state, metrics = self._run_epoch(agent, training_state)

            # Block until results ready
            jax.block_until_ready((training_state, metrics))

            # Collect metrics
            metrics_dict = jax.device_get(metrics)
            all_metrics.append(metrics_dict)

            epoch_duration = time.monotonic() - epoch_start

            # Log progress
            total_loss = float(metrics_dict.get("total_loss", 0))
            LOGGER.info(
                "Epoch %d/%d | loss=%.4f | time=%.2fs",
                epoch + 1,
                self._config.num_epochs,
                total_loss,
                epoch_duration,
            )

            # Emit heartbeat every 10 epochs
            if self._emitter and (epoch + 1) % 10 == 0:
                self._emitter.heartbeat({
                    "status": "training",
                    "epoch": epoch + 1,
                    "env_steps": (epoch + 1) * steps_per_epoch,
                    "loss": total_loss,
                })

        total_duration = time.monotonic() - start_time

        # Save checkpoint if requested
        if self._config.save_checkpoint:
            self._save_checkpoint(training_state, run_dir)

        return {
            "status": "completed",
            "env_id": self._config.env_id,
            "agent": self._config.agent,
            "epochs_completed": self._config.num_epochs,
            "total_env_steps": self._config.num_epochs * steps_per_epoch,
            "total_duration_s": total_duration,
            "final_loss": float(all_metrics[-1].get("total_loss", 0)) if all_metrics else None,
            "config": self._config.to_dict(),
        }

    def _run_random_baseline(
        self,
        env,
        key: jax.Array,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Run random agent baseline.

        Args:
            env: Wrapped Jumanji environment
            key: JAX PRNG key
            run_dir: Run output directory

        Returns:
            Baseline result dictionary
        """
        from jumanji.training.agents.random import RandomAgent

        env_name = self._env_name_from_id(self._config.env_id)

        # Get random policy for this environment
        random_policy = self._get_random_policy(env_name, env.unwrapped)

        # Create random agent
        agent = RandomAgent(
            env=env,
            n_steps=self._config.n_steps,
            total_batch_size=self._config.total_batch_size,
            random_policy=random_policy,
        )

        LOGGER.info("Created random agent for %s", env_name)

        # Initialize
        key, init_key = jax.random.split(key)
        training_state = self._init_training_state(env, agent, init_key)

        # Run episodes to collect baseline metrics
        steps_per_epoch = self._config.n_steps * self._config.total_batch_size
        total_steps = 0
        total_reward = 0.0

        start_time = time.monotonic()

        for epoch in range(self._config.num_epochs):
            training_state, metrics = self._run_epoch(agent, training_state)
            jax.block_until_ready((training_state, metrics))

            metrics_dict = jax.device_get(metrics)
            total_steps += steps_per_epoch

            if epoch % 10 == 0:
                LOGGER.info("Random baseline | epoch %d/%d", epoch + 1, self._config.num_epochs)

        total_duration = time.monotonic() - start_time

        return {
            "status": "completed",
            "env_id": self._config.env_id,
            "agent": "random",
            "epochs_completed": self._config.num_epochs,
            "total_env_steps": total_steps,
            "total_duration_s": total_duration,
            "config": self._config.to_dict(),
        }

    def _get_actor_critic_networks(self, env_name: str, env):
        """Get actor-critic networks for the environment.

        Args:
            env_name: Environment name (snake_case)
            env: Unwrapped Jumanji environment

        Returns:
            Actor-critic networks tuple
        """
        from jumanji.training import networks

        # Map environment names to their network factories
        network_factories = {
            "game_2048": lambda: networks.make_actor_critic_networks_game_2048(env),
            "graph_coloring": lambda: networks.make_actor_critic_networks_graph_coloring(env),
            "minesweeper": lambda: networks.make_actor_critic_networks_minesweeper(env),
            "rubiks_cube": lambda: networks.make_actor_critic_networks_rubiks_cube(env),
            "sliding_tile_puzzle": lambda: networks.make_actor_critic_networks_sliding_tile_puzzle(env),
            "sudoku": lambda: networks.make_actor_critic_networks_sudoku(env),
        }

        factory = network_factories.get(env_name)
        if factory is None:
            raise ValueError(f"No network factory for environment: {env_name}")

        return factory()

    def _get_random_policy(self, env_name: str, env):
        """Get random policy for the environment.

        Args:
            env_name: Environment name (snake_case)
            env: Unwrapped Jumanji environment

        Returns:
            Random policy function
        """
        from jumanji.training import networks

        policy_factories = {
            "game_2048": networks.make_random_policy_game_2048,
            "graph_coloring": networks.make_random_policy_graph_coloring,
            "minesweeper": lambda: networks.make_random_policy_minesweeper(env),
            "rubiks_cube": lambda: networks.make_random_policy_rubiks_cube(env),
            "sliding_tile_puzzle": networks.make_random_policy_sliding_tile_puzzle,
            "sudoku": lambda: networks.make_random_policy_sudoku(env),
        }

        factory = policy_factories.get(env_name)
        if factory is None:
            raise ValueError(f"No random policy for environment: {env_name}")

        result = factory()
        return result if callable(result) else result

    def _init_training_state(self, env, agent, key: jax.Array):
        """Initialize training state.

        Args:
            env: Wrapped environment
            agent: A2C or Random agent
            key: JAX PRNG key

        Returns:
            Initial training state
        """
        num_devices = jax.local_device_count()
        batch_size_per_device = self._config.total_batch_size // num_devices

        # Initialize parameters
        key, params_key = jax.random.split(key)
        params_state = agent.init_params(params_key)

        # Initialize acting state
        key, *env_keys = jax.random.split(key, 1 + num_devices * batch_size_per_device)
        env_keys = jnp.stack(env_keys).reshape((num_devices, batch_size_per_device, -1))

        # Replicate across devices
        params_state = jax.device_put_replicated(params_state, jax.local_devices())

        from jumanji.training.types import TrainingState

        # The actual initialization depends on Jumanji's API
        # This is a simplified version - the real implementation would
        # properly initialize the acting state using the agent's init method
        return agent.init(params_key)

    def _run_epoch(self, agent, training_state) -> Tuple[Any, Dict]:
        """Run one training epoch.

        Args:
            agent: Training agent
            training_state: Current training state

        Returns:
            Tuple of (new_training_state, metrics)
        """
        # Run learner steps
        @functools.partial(jax.pmap, axis_name="devices")
        def epoch_fn(ts):
            ts, metrics = jax.lax.scan(
                lambda s, _: agent.run_epoch(s),
                ts,
                None,
                self._config.num_learner_steps_per_epoch,
            )
            # Average metrics across steps
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return ts, metrics

        return epoch_fn(training_state)

    def _save_checkpoint(self, training_state, run_dir: Path) -> None:
        """Save model checkpoint.

        Args:
            training_state: Training state to save
            run_dir: Run output directory
        """
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / "final_checkpoint.pkl"

        import pickle
        with open(ckpt_path, "wb") as f:
            pickle.dump(jax.device_get(training_state), f)

        LOGGER.info("Saved checkpoint to %s", ckpt_path)

    def _write_analytics_manifest(self, run_dir: Path) -> None:
        """Write analytics manifest to run directory.

        Args:
            run_dir: Run output directory
        """
        try:
            from .analytics import write_analytics_manifest
            manifest_path = write_analytics_manifest(
                run_dir,
                config=self._config.to_dict(),
                run_id=self._config.run_id,
            )
            LOGGER.info("Analytics manifest written to %s", manifest_path)
        except Exception as e:
            LOGGER.warning("Failed to write analytics manifest: %s", e)


__all__ = ["JumanjiWorkerRuntime"]
