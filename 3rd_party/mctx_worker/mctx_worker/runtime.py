"""MCTX GPU-accelerated training runtime.

This module provides the core training logic for turn-based games using
mctx (DeepMind's MCTS library) and Pgx (GPU-accelerated game environments).

Key Features:
- GPU-accelerated self-play using JAX
- AlphaZero / MuZero / Gumbel MuZero MCTS
- Vectorized game simulation with Pgx
- Neural network training with Optax

Training Loop:
1. Self-play: Run N parallel games on GPU using MCTS
2. Collect: Store (state, policy, value) tuples in replay buffer
3. Train: Sample batch and update neural network
4. Repeat until convergence

Analytics Integration:
- TensorBoard: Logs to var/trainer/runs/{run_id}/tensorboard/
- WandB: Optional integration
- analytics.json: Manifest file for GUI integration
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

# Import Pgx for GPU-accelerated game environments
import pgx

# Import mctx for MCTS algorithms
import mctx

from gym_gui.core.worker import TelemetryEmitter
from gym_gui.logging_config.helpers import log_constant

from .config import (
    MCTXWorkerConfig,
    MCTXAlgorithm,
    PGXEnvironment,
    load_worker_config,
)
from .analytics import write_analytics_manifest

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# Neural Network Architecture
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block for the policy/value network."""

    channels: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        return nn.relu(x + residual)


class AlphaZeroNetwork(nn.Module):
    """AlphaZero-style dual-head network for policy and value prediction.

    Architecture:
    - Input: Board state observation
    - Backbone: Convolutional residual network
    - Policy head: Conv -> FC -> softmax logits
    - Value head: Conv -> FC -> tanh
    """

    num_actions: int
    channels: int = 128
    num_blocks: int = 8

    @nn.compact
    def __call__(self, x):
        # Initial convolution
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        # Residual blocks
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.channels)(x)

        # Policy head
        policy = nn.Conv(32, kernel_size=(1, 1))(x)
        policy = nn.BatchNorm(use_running_average=True)(policy)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))  # Flatten
        policy = nn.Dense(self.num_actions)(policy)

        # Value head
        value = nn.Conv(1, kernel_size=(1, 1))(x)
        value = nn.BatchNorm(use_running_average=True)(value)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))  # Flatten
        value = nn.Dense(256)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value.squeeze(-1)


class MLPNetwork(nn.Module):
    """MLP network for non-image observations (e.g., poker games)."""

    num_actions: int
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape((x.shape[0], -1))

        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Policy head
        policy = nn.Dense(self.num_actions)(x)

        # Value head
        value = nn.Dense(64)(x)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value.squeeze(-1)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Simple replay buffer for self-play data."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], num_actions: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.size = 0
        self.idx = 0

        # Pre-allocate arrays
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.policies = np.zeros((capacity, num_actions), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, policy: np.ndarray, value: float) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.idx] = obs
        self.policies[self.idx] = policy
        self.values[self.idx] = value
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self, obs: np.ndarray, policies: np.ndarray, values: np.ndarray
    ) -> None:
        """Add a batch of transitions to the buffer."""
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.add(obs[i], policies[i], values[i])

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple:
        """Sample a batch of transitions."""
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.policies[indices],
            self.values[indices],
        )


# =============================================================================
# MCTS Root Functions
# =============================================================================

def make_recurrent_fn(
    network_apply: Callable,
    params: Any,
    env: Any,
) -> Callable:
    """Create recurrent function for mctx.

    For AlphaZero, this is a simple network evaluation.
    For MuZero, this would include dynamics model.
    """

    def recurrent_fn(params, rng_key, action, embedding):
        """Recurrent function for MCTS simulation.

        Args:
            params: Network parameters
            rng_key: JAX random key
            action: Action to take
            embedding: Current state embedding (we use raw state for AlphaZero)

        Returns:
            RecurrentFnOutput with next embedding, reward, discount, prior_logits, value
        """
        # For AlphaZero, embedding is the raw state
        # We step the environment and evaluate the new state
        del params, rng_key  # Unused for AlphaZero

        # Note: In a full implementation, we'd step the environment here
        # For simplicity, we return dummy values (real implementation would
        # use the dynamics model or actual env stepping)
        batch_size = embedding.shape[0]
        num_actions = env.num_actions

        # Placeholder: real implementation would compute these
        next_embedding = embedding  # Would be next state
        reward = jnp.zeros(batch_size)
        discount = jnp.ones(batch_size) * 0.99
        prior_logits = jnp.zeros((batch_size, num_actions))
        value = jnp.zeros(batch_size)

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=value,
        ), next_embedding

    return recurrent_fn


# =============================================================================
# Training State
# =============================================================================

class TrainState(train_state.TrainState):
    """Extended train state with batch stats for BatchNorm."""

    batch_stats: Any = None


def create_train_state(
    rng: jax.random.PRNGKey,
    network: nn.Module,
    learning_rate: float,
    obs_shape: Tuple[int, ...],
) -> TrainState:
    """Create initial training state."""
    # Initialize network
    dummy_input = jnp.ones((1, *obs_shape))
    variables = network.init(rng, dummy_input)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


# =============================================================================
# Training Functions
# =============================================================================

@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey,
    num_actions: int,
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step.

    Args:
        state: Training state
        batch: (observations, target_policies, target_values)
        rng: Random key
        num_actions: Number of actions (static)

    Returns:
        Updated state and metrics dict
    """
    obs, target_policy, target_value = batch

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (policy_logits, value), new_model_state = state.apply_fn(
            variables, obs, mutable=["batch_stats"]
        )

        # Policy loss (cross-entropy)
        policy_loss = optax.softmax_cross_entropy(policy_logits, target_policy)
        policy_loss = jnp.mean(policy_loss)

        # Value loss (MSE)
        value_loss = jnp.mean((value - target_value) ** 2)

        # Total loss
        total_loss = policy_loss + value_loss

        return total_loss, (policy_loss, value_loss, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (policy_loss, value_loss, new_model_state)), grads = grad_fn(
        state.params
    )

    # Update parameters
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_model_state["batch_stats"])

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    return state, metrics


# =============================================================================
# Runtime Class
# =============================================================================

class MCTXWorkerRuntime:
    """Runtime for GPU-accelerated MCTS training.

    This class handles:
    - Pgx environment initialization
    - Neural network setup
    - Self-play data generation
    - Training loop
    - Checkpointing and logging
    """

    def __init__(self, config: MCTXWorkerConfig) -> None:
        """Initialize the runtime.

        Args:
            config: Worker configuration
        """
        self.config = config

        # Initialize JAX
        self._setup_jax()

        # Create environment
        self._env = pgx.make(config.env_id)
        self._num_actions = self._env.num_actions

        # Get observation shape from environment
        self._obs_shape = self._get_obs_shape()

        # Create network
        self._network = self._create_network()

        # Create replay buffer
        self._buffer = ReplayBuffer(
            capacity=config.training.replay_buffer_size,
            obs_shape=self._obs_shape,
            num_actions=self._num_actions,
        )

        # Analytics
        self._writer: Optional["SummaryWriter"] = None
        self._wandb_run: Optional[Any] = None

        # Telemetry
        self._emitter = TelemetryEmitter(run_id=config.run_id, logger=_LOGGER)

        # RNG
        self._rng = jax.random.PRNGKey(config.seed)
        self._np_rng = np.random.default_rng(config.seed)

    def _setup_jax(self) -> None:
        """Configure JAX for the specified device."""
        device = self.config.device.lower()

        if device == "gpu":
            # Check for GPU availability
            devices = jax.devices("gpu")
            if not devices:
                _LOGGER.warning("No GPU found, falling back to CPU")
                device = "cpu"
            else:
                _LOGGER.info(f"Using GPU: {devices[0]}")
        elif device == "tpu":
            devices = jax.devices("tpu")
            if not devices:
                _LOGGER.warning("No TPU found, falling back to CPU")
                device = "cpu"
            else:
                _LOGGER.info(f"Using TPU: {devices[0]}")

        # Set platform
        os.environ["JAX_PLATFORM_NAME"] = device

    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Get observation shape from environment."""
        # Reset environment to get initial state
        key = jax.random.PRNGKey(0)
        state = self._env.init(key)
        obs = state.observation
        return obs.shape

    def _create_network(self) -> nn.Module:
        """Create the neural network based on observation shape."""
        cfg = self.config.network

        # Use CNN for image-like observations, MLP otherwise
        if len(self._obs_shape) >= 3:
            return AlphaZeroNetwork(
                num_actions=self._num_actions,
                channels=cfg.channels,
                num_blocks=cfg.num_res_blocks,
            )
        else:
            return MLPNetwork(
                num_actions=self._num_actions,
                hidden_dims=cfg.hidden_dims,
            )

    def _setup_analytics(self) -> None:
        """Set up TensorBoard and WandB logging."""
        # Ensure directories exist
        run_dir = Path(self.config.checkpoint_path or f"var/trainer/runs/{self.config.run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        tb_dir = run_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(tb_dir))
            _LOGGER.info(f"TensorBoard logging to: {tb_dir}")
        except ImportError:
            _LOGGER.warning("tensorboard not installed")

        # Write analytics manifest
        manifest_path = write_analytics_manifest(self.config)
        _LOGGER.info(f"Analytics manifest written: {manifest_path}")

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard and WandB."""
        if self._writer:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, step)
            self._writer.flush()

        if self._wandb_run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                _LOGGER.debug(f"WandB logging error: {e}")

    def _self_play_game(
        self,
        state: TrainState,
        num_games: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run self-play games to generate training data.

        Args:
            state: Current training state
            num_games: Number of parallel games to run

        Returns:
            Tuple of (observations, policies, values) from games
        """
        # Initialize games
        self._rng, key = jax.random.split(self._rng)
        keys = jax.random.split(key, num_games)
        env_states = jax.vmap(self._env.init)(keys)

        # Storage for game data
        all_obs = []
        all_policies = []
        all_outcomes = []

        # Run games until all terminated
        max_steps = 500  # Safety limit
        step = 0

        while not jnp.all(env_states.terminated) and step < max_steps:
            # Get current observations
            obs = env_states.observation

            # Get policy and value from network
            variables = {"params": state.params, "batch_stats": state.batch_stats}
            policy_logits, values = state.apply_fn(variables, obs)

            # Apply legal action mask
            legal_mask = env_states.legal_action_mask
            masked_logits = jnp.where(
                legal_mask, policy_logits, jnp.finfo(jnp.float32).min
            )

            # Sample actions (or use MCTS policy)
            self._rng, key = jax.random.split(self._rng)

            if self.config.algorithm == MCTXAlgorithm.ALPHAZERO:
                # Simple softmax sampling for now
                # Full implementation would use mctx.alphazero_policy
                probs = jax.nn.softmax(masked_logits, axis=-1)
                actions = jax.random.categorical(key, jnp.log(probs + 1e-8))
            else:
                # Simple sampling fallback
                probs = jax.nn.softmax(masked_logits, axis=-1)
                actions = jax.random.categorical(key, jnp.log(probs + 1e-8))

            # Store data for non-terminated games
            active = ~env_states.terminated
            if jnp.any(active):
                active_obs = obs[active]
                active_probs = jax.nn.softmax(masked_logits[active], axis=-1)
                all_obs.append(np.array(active_obs))
                all_policies.append(np.array(active_probs))

            # Step environment
            env_states = jax.vmap(self._env.step)(env_states, actions)
            step += 1

        # Get game outcomes (rewards)
        rewards = np.array(env_states.rewards)

        # Assign values based on game outcome
        # For two-player zero-sum games, winner gets +1, loser gets -1
        if len(all_obs) > 0:
            all_obs = np.concatenate(all_obs, axis=0)
            all_policies = np.concatenate(all_policies, axis=0)

            # Simple value assignment (real implementation would propagate outcomes)
            mean_reward = np.mean(rewards)
            all_values = np.full(len(all_obs), mean_reward, dtype=np.float32)

            return all_obs, all_policies, all_values
        else:
            return np.array([]), np.array([]), np.array([])

    def run(self) -> Dict[str, Any]:
        """Run the training loop.

        Returns:
            Final training metrics
        """
        _LOGGER.info(f"Starting MCTX training: {self.config.run_id}")
        _LOGGER.info(f"Environment: {self.config.env_id}")
        _LOGGER.info(f"Algorithm: {self.config.algorithm.value}")
        _LOGGER.info(f"Device: {self.config.device}")

        # Set up analytics
        self._setup_analytics()

        # Create training state
        self._rng, key = jax.random.split(self._rng)
        train_state_obj = create_train_state(
            key,
            self._network,
            self.config.training.learning_rate,
            self._obs_shape,
        )

        # Emit run started
        self._emitter.run_started({
            "worker_type": "mctx",
            "env_id": self.config.env_id,
            "algorithm": self.config.algorithm.value,
            "device": self.config.device,
        })

        total_steps = self.config.max_steps or 100000
        games_per_iter = self.config.training.games_per_iteration
        batch_size = self.config.training.batch_size
        min_buffer = self.config.training.min_buffer_size

        start_time = time.time()
        global_step = 0
        iteration = 0

        try:
            while global_step < total_steps:
                iteration += 1

                # Self-play phase
                obs, policies, values = self._self_play_game(
                    train_state_obj, games_per_iter
                )

                if len(obs) > 0:
                    self._buffer.add_batch(obs, policies, values)
                    global_step += len(obs)

                # Training phase (if buffer has enough samples)
                if self._buffer.size >= min_buffer:
                    # Sample batch
                    batch_obs, batch_pol, batch_val = self._buffer.sample(
                        batch_size, self._np_rng
                    )

                    # Convert to JAX arrays
                    batch = (
                        jnp.array(batch_obs),
                        jnp.array(batch_pol),
                        jnp.array(batch_val),
                    )

                    # Train
                    self._rng, key = jax.random.split(self._rng)
                    train_state_obj, metrics = train_step(
                        train_state_obj, batch, key, self._num_actions
                    )

                    # Log metrics
                    elapsed = time.time() - start_time
                    metrics["steps_per_second"] = global_step / elapsed
                    metrics["buffer_size"] = self._buffer.size

                    self._log_metrics(
                        {f"train/{k}": float(v) for k, v in metrics.items()},
                        global_step,
                    )

                # Progress report and heartbeat
                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0

                    # Get latest metrics for heartbeat
                    heartbeat_metrics = {
                        "iteration": iteration,
                        "global_step": global_step,
                        "buffer_size": self._buffer.size,
                        "steps_per_second": steps_per_sec,
                        "progress_pct": (global_step / total_steps) * 100 if total_steps > 0 else 0,
                    }

                    # Emit heartbeat for monitoring
                    self._emitter.heartbeat(heartbeat_metrics)

                    print(
                        f"[PROGRESS] iteration={iteration} | "
                        f"steps={global_step}/{total_steps} ({heartbeat_metrics['progress_pct']:.1f}%) | "
                        f"buffer={self._buffer.size} | "
                        f"steps/sec={steps_per_sec:.0f}"
                    )
                    sys.stdout.flush()

                # Checkpoint
                if (
                    self.config.training.checkpoint_interval > 0
                    and iteration % self.config.training.checkpoint_interval == 0
                ):
                    self._save_checkpoint(train_state_obj, iteration)

            # Save final policy
            final_policy_path = self._save_policy(train_state_obj, iteration, final=True)

            _LOGGER.info("Training completed")
            print(f"[COMPLETE] run_id={self.config.run_id} status=success")
            sys.stdout.flush()

            self._emitter.run_completed({
                "final_step": global_step,
                "iterations": iteration,
                "policy_path": str(final_policy_path),
            })

            return {
                "final_step": global_step,
                "iterations": iteration,
                "policy_path": str(final_policy_path),
            }

        except Exception as e:
            _LOGGER.error(f"Training failed: {e}", exc_info=True)
            print(f"[ERROR] run_id={self.config.run_id} error={e}")
            sys.stdout.flush()
            self._emitter.run_failed({"error": str(e)})
            raise

        finally:
            self._cleanup()

    def _save_checkpoint(self, state: TrainState, iteration: int) -> None:
        """Save a checkpoint."""
        ckpt_dir = Path(
            self.config.checkpoint_path or f"var/trainer/runs/{self.config.run_id}"
        ) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"checkpoint_{iteration}.pkl"

        import pickle
        with open(ckpt_path, "wb") as f:
            pickle.dump(
                {
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                    "opt_state": state.opt_state,
                    "step": state.step,
                    "iteration": iteration,
                },
                f,
            )

        _LOGGER.info(f"Checkpoint saved: {ckpt_path}")
        print(f"[CHECKPOINT] path={ckpt_path}")
        sys.stdout.flush()

    def _save_policy(self, state: TrainState, iteration: int, final: bool = False) -> Path:
        """Save a clean policy file for inference/evaluation.

        This saves only the network parameters needed for inference,
        separate from training state (optimizer, batch stats, etc.).

        Args:
            state: Current training state
            iteration: Current iteration number
            final: Whether this is the final policy

        Returns:
            Path to the saved policy file
        """
        run_dir = Path(
            self.config.checkpoint_path or f"var/trainer/runs/{self.config.run_id}"
        )
        policy_dir = run_dir / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)

        # Save policy with metadata
        policy_name = "policy_final.pkl" if final else f"policy_iter_{iteration}.pkl"
        policy_path = policy_dir / policy_name

        import pickle
        policy_data = {
            "params": state.params,
            "batch_stats": state.batch_stats,
            # Metadata for loading
            "env_id": self.config.env_id,
            "algorithm": self.config.algorithm.value if hasattr(self.config.algorithm, "value") else str(self.config.algorithm),
            "num_actions": self._num_actions,
            "obs_shape": self._obs_shape,
            "network_config": {
                "num_res_blocks": self.config.network.num_res_blocks,
                "channels": self.config.network.channels,
                "hidden_dims": list(self.config.network.hidden_dims),
                "use_resnet": self.config.network.use_resnet,
            },
            "mcts_config": {
                "num_simulations": self.config.mcts.num_simulations,
                "dirichlet_alpha": self.config.mcts.dirichlet_alpha,
                "temperature": self.config.mcts.temperature,
            },
            "iteration": iteration,
            "run_id": self.config.run_id,
        }

        with open(policy_path, "wb") as f:
            pickle.dump(policy_data, f)

        _LOGGER.info(f"Policy saved: {policy_path}")
        print(f"[POLICY] path={policy_path} final={final}")
        sys.stdout.flush()

        return policy_path

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._writer:
            self._writer.close()
            self._writer = None

        if self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop training and cleanup."""
        self._cleanup()


__all__ = [
    "MCTXWorkerRuntime",
    "AlphaZeroNetwork",
    "MLPNetwork",
    "ReplayBuffer",
]
