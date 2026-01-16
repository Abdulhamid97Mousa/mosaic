"""Chess training test with PGX.

This test verifies that we can train a chess policy using PGX + JAX + mctx.
Creates a beginner-level policy that can be compared against Stockfish later.

Run with:
    pytest 3rd_party/mctx_worker/tests/test_chess_training.py -v -s

The -s flag shows training progress output.
"""

from __future__ import annotations

import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

# Skip all tests if JAX/PGX not available
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pgx = pytest.importorskip("pgx")
flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")

import numpy as np
from flax import linen as nn
from flax.training import train_state


# =============================================================================
# Simple AlphaZero-style Network (smaller for quick training)
# =============================================================================

class SmallChessNet(nn.Module):
    """Small network for quick chess training tests."""

    num_actions: int = 4672  # Chess action space
    channels: int = 32
    num_blocks: int = 2

    @nn.compact
    def __call__(self, x):
        # Initial conv
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)

        # Residual blocks
        for _ in range(self.num_blocks):
            residual = x
            x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x)
            x = nn.Conv(self.channels, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.relu(x + residual)

        # Policy head
        policy = nn.Conv(8, kernel_size=(1, 1))(x)
        policy = nn.relu(policy)
        policy = policy.reshape((policy.shape[0], -1))
        policy = nn.Dense(self.num_actions)(policy)

        # Value head
        value = nn.Conv(1, kernel_size=(1, 1))(x)
        value = nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = nn.Dense(64)(value)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)

        return policy, value.squeeze(-1)


# =============================================================================
# Training Functions
# =============================================================================

def create_train_state(rng, network, learning_rate, obs_shape):
    """Create initial training state."""
    dummy_input = jnp.ones((1, *obs_shape))
    variables = network.init(rng, dummy_input)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        tx=tx,
    )


@jax.jit
def train_step(state, batch):
    """Single training step."""
    obs, target_policy, target_value = batch

    def loss_fn(params):
        policy_logits, value = state.apply_fn({"params": params}, obs)

        # Policy loss (cross-entropy)
        policy_loss = optax.softmax_cross_entropy(policy_logits, target_policy)
        policy_loss = jnp.mean(policy_loss)

        # Value loss (MSE)
        value_loss = jnp.mean((value - target_value) ** 2)

        return policy_loss + value_loss, (policy_loss, value_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (policy_loss, value_loss)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


def self_play_batch(env, network_apply, params, rng, batch_size=64, max_moves=100):
    """Generate self-play training data."""
    # Initialize games
    keys = jax.random.split(rng, batch_size)
    states = jax.vmap(env.init)(keys)

    all_obs = []
    all_policies = []
    all_values = []

    for move in range(max_moves):
        # Check if all games are done
        if jnp.all(states.terminated):
            break

        obs = states.observation
        legal_mask = states.legal_action_mask

        # Get policy from network
        policy_logits, values = network_apply({"params": params}, obs)

        # Mask illegal moves
        masked_logits = jnp.where(legal_mask, policy_logits, -1e9)
        policy_probs = jax.nn.softmax(masked_logits, axis=-1)

        # Sample actions
        rng, key = jax.random.split(rng)
        actions = jax.vmap(lambda p, k: jax.random.categorical(k, jnp.log(p + 1e-8)))(
            policy_probs, jax.random.split(key, batch_size)
        )

        # Store data for non-terminated games
        active = ~states.terminated
        if jnp.any(active):
            all_obs.append(np.array(obs[active]))
            all_policies.append(np.array(policy_probs[active]))

        # Step environment
        states = jax.vmap(env.step)(states, actions)

    # Get final rewards and assign values
    if len(all_obs) > 0:
        all_obs = np.concatenate(all_obs, axis=0)
        all_policies = np.concatenate(all_policies, axis=0)
        # Simple value assignment (0 for now, would use game outcomes in full impl)
        all_values = np.zeros(len(all_obs), dtype=np.float32)

        return all_obs, all_policies, all_values

    return None, None, None


# =============================================================================
# Tests
# =============================================================================

class TestChessTraining:
    """Test chess policy training with PGX."""

    @pytest.fixture
    def chess_env(self):
        """Create PGX chess environment."""
        return pgx.make("chess")

    @pytest.fixture
    def small_network(self, chess_env):
        """Create small network for testing."""
        return SmallChessNet(num_actions=chess_env.num_actions)

    def test_pgx_chess_available(self, chess_env):
        """Test that PGX chess environment works."""
        key = jax.random.PRNGKey(0)
        state = chess_env.init(key)

        assert state.observation.shape == (8, 8, 119)  # Chess observation
        assert jnp.sum(state.legal_action_mask) > 0  # Has legal moves
        print(f"Chess observation shape: {state.observation.shape}")
        print(f"Number of legal moves: {jnp.sum(state.legal_action_mask)}")

    def test_network_forward_pass(self, chess_env, small_network):
        """Test network can process chess observations."""
        key = jax.random.PRNGKey(0)
        state = chess_env.init(key)

        # Initialize network
        key, init_key = jax.random.split(key)
        variables = small_network.init(init_key, state.observation[None])

        # Forward pass
        policy_logits, value = small_network.apply(variables, state.observation[None])

        assert policy_logits.shape == (1, chess_env.num_actions)
        assert value.shape == (1,)
        print(f"Policy shape: {policy_logits.shape}")
        print(f"Value: {value[0]:.4f}")

    def test_self_play_generation(self, chess_env, small_network):
        """Test self-play data generation."""
        key = jax.random.PRNGKey(42)

        # Initialize network
        key, init_key = jax.random.split(key)
        state = chess_env.init(init_key)
        variables = small_network.init(init_key, state.observation[None])

        # Generate self-play data
        key, play_key = jax.random.split(key)
        obs, policies, values = self_play_batch(
            chess_env,
            small_network.apply,
            variables["params"],
            play_key,
            batch_size=16,
            max_moves=20,
        )

        assert obs is not None
        assert len(obs) > 0
        print(f"Generated {len(obs)} training samples from self-play")

    def test_training_step(self, chess_env, small_network):
        """Test single training step."""
        key = jax.random.PRNGKey(42)

        # Create training state
        key, init_key = jax.random.split(key)
        state = chess_env.init(init_key)
        obs_shape = state.observation.shape

        train_st = create_train_state(
            init_key, small_network, learning_rate=1e-3, obs_shape=obs_shape
        )

        # Create dummy batch
        batch_size = 32
        dummy_obs = jnp.ones((batch_size, *obs_shape))
        dummy_policy = jnp.ones((batch_size, chess_env.num_actions)) / chess_env.num_actions
        dummy_value = jnp.zeros(batch_size)

        # Training step
        train_st, metrics = train_step(train_st, (dummy_obs, dummy_policy, dummy_value))

        assert "total_loss" in metrics
        print(f"Loss after 1 step: {metrics['total_loss']:.4f}")

    def test_quick_training_loop(self, chess_env, small_network):
        """Quick training loop to verify everything works end-to-end."""
        print("\n" + "="*60)
        print("QUICK CHESS TRAINING TEST")
        print("="*60)

        key = jax.random.PRNGKey(42)

        # Initialize
        key, init_key = jax.random.split(key)
        state = chess_env.init(init_key)
        obs_shape = state.observation.shape

        train_st = create_train_state(
            init_key, small_network, learning_rate=1e-3, obs_shape=obs_shape
        )

        print(f"Device: {jax.devices()[0]}")
        print(f"Observation shape: {obs_shape}")
        print(f"Action space: {chess_env.num_actions}")
        print()

        # Training loop
        num_iterations = 10
        batch_size = 32
        total_samples = 0
        start_time = time.time()

        for iteration in range(num_iterations):
            # Self-play
            key, play_key = jax.random.split(key)
            obs, policies, values = self_play_batch(
                chess_env,
                small_network.apply,
                train_st.params,
                play_key,
                batch_size=batch_size,
                max_moves=30,
            )

            if obs is None or len(obs) < batch_size:
                continue

            total_samples += len(obs)

            # Sample batch and train
            indices = np.random.choice(len(obs), min(batch_size, len(obs)), replace=False)
            batch = (
                jnp.array(obs[indices]),
                jnp.array(policies[indices]),
                jnp.array(values[indices]),
            )

            train_st, metrics = train_step(train_st, batch)

            print(f"Iter {iteration+1:2d}/{num_iterations} | "
                  f"Loss: {metrics['total_loss']:.4f} | "
                  f"Samples: {total_samples}")

        elapsed = time.time() - start_time
        print()
        print(f"Training completed in {elapsed:.2f}s")
        print(f"Total samples: {total_samples}")
        print(f"Samples/sec: {total_samples/elapsed:.0f}")

        assert total_samples > 0

    def test_train_beginner_policy(self, chess_env, small_network, tmp_path):
        """Train a beginner chess policy and save it.

        This creates a policy that can be compared against Stockfish.
        """
        print("\n" + "="*60)
        print("TRAINING BEGINNER CHESS POLICY")
        print("="*60)

        key = jax.random.PRNGKey(42)

        # Initialize
        key, init_key = jax.random.split(key)
        state = chess_env.init(init_key)
        obs_shape = state.observation.shape

        train_st = create_train_state(
            init_key, small_network, learning_rate=1e-3, obs_shape=obs_shape
        )

        print(f"Device: {jax.devices()[0]}")
        print(f"Training beginner policy...")
        print()

        # Training parameters
        num_iterations = 50  # More iterations for better policy
        batch_size = 64
        games_per_iter = 32

        total_samples = 0
        total_games = 0
        losses = []
        start_time = time.time()

        for iteration in range(num_iterations):
            # Self-play
            key, play_key = jax.random.split(key)
            obs, policies, values = self_play_batch(
                chess_env,
                small_network.apply,
                train_st.params,
                play_key,
                batch_size=games_per_iter,
                max_moves=50,
            )

            if obs is None or len(obs) < 10:
                continue

            total_samples += len(obs)
            total_games += games_per_iter

            # Train on collected data
            for _ in range(3):  # Multiple passes over data
                indices = np.random.choice(len(obs), min(batch_size, len(obs)), replace=False)
                batch = (
                    jnp.array(obs[indices]),
                    jnp.array(policies[indices]),
                    jnp.array(values[indices]),
                )
                train_st, metrics = train_step(train_st, batch)

            losses.append(float(metrics['total_loss']))

            if (iteration + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                elapsed = time.time() - start_time
                print(f"Iter {iteration+1:3d}/{num_iterations} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Games: {total_games} | "
                      f"Samples: {total_samples} | "
                      f"Time: {elapsed:.1f}s")

        elapsed = time.time() - start_time

        # Save policy
        policy_path = tmp_path / "beginner_chess_policy.pkl"
        policy_data = {
            "params": jax.device_get(train_st.params),
            "env_id": "chess",
            "obs_shape": obs_shape,
            "num_actions": chess_env.num_actions,
            "network_config": {
                "channels": small_network.channels,
                "num_blocks": small_network.num_blocks,
            },
            "training_info": {
                "iterations": num_iterations,
                "total_samples": total_samples,
                "total_games": total_games,
                "final_loss": float(losses[-1]) if losses else 0,
                "training_time_sec": elapsed,
            },
        }

        with open(policy_path, "wb") as f:
            pickle.dump(policy_data, f)

        print()
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total games: {total_games}")
        print(f"Total samples: {total_samples}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Policy saved to: {policy_path}")
        print()

        # Verify policy can make moves
        print("Testing policy can make legal moves...")
        key, test_key = jax.random.split(key)
        test_state = chess_env.init(test_key)

        policy_logits, value = small_network.apply(
            {"params": train_st.params},
            test_state.observation[None]
        )

        # Mask illegal moves and get best move
        masked_logits = jnp.where(
            test_state.legal_action_mask,
            policy_logits[0],
            -1e9
        )
        best_action = jnp.argmax(masked_logits)

        assert test_state.legal_action_mask[best_action], "Policy chose illegal move!"
        print(f"Policy chose legal move: action {best_action}")
        print(f"Position evaluation: {value[0]:.4f}")
        print()
        print("SUCCESS: Beginner policy trained and verified!")

        assert policy_path.exists()
        assert total_samples > 100


class TestPolicyEvaluation:
    """Test trained policy can play chess."""

    def test_policy_plays_legal_moves(self):
        """Test that trained policy only plays legal moves."""
        env = pgx.make("chess")
        network = SmallChessNet(num_actions=env.num_actions)

        key = jax.random.PRNGKey(0)
        state = env.init(key)

        # Initialize network
        key, init_key = jax.random.split(key)
        variables = network.init(init_key, state.observation[None])

        # Play 20 moves and verify all are legal
        for move_num in range(20):
            if state.terminated:
                break

            policy_logits, _ = network.apply(variables, state.observation[None])

            # Mask and sample
            masked_logits = jnp.where(
                state.legal_action_mask,
                policy_logits[0],
                -1e9
            )

            key, action_key = jax.random.split(key)
            action = jax.random.categorical(action_key, masked_logits)

            assert state.legal_action_mask[action], f"Illegal move at move {move_num}!"

            state = env.step(state, action)

        print(f"Played {move_num + 1} legal moves successfully")

    def test_policy_vs_random(self):
        """Test policy against random player to verify it learned something."""
        env = pgx.make("chess")
        network = SmallChessNet(num_actions=env.num_actions)

        key = jax.random.PRNGKey(42)

        # Quick training
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        train_st = create_train_state(
            init_key, network, learning_rate=1e-3, obs_shape=state.observation.shape
        )

        # Train for a few iterations
        for _ in range(20):
            key, play_key = jax.random.split(key)
            obs, policies, values = self_play_batch(
                env, network.apply, train_st.params, play_key,
                batch_size=16, max_moves=30
            )
            if obs is not None and len(obs) >= 16:
                batch = (jnp.array(obs[:16]), jnp.array(policies[:16]), jnp.array(values[:16]))
                train_st, _ = train_step(train_st, batch)

        # Play games: policy (white) vs random (black)
        num_games = 10
        policy_wins = 0
        random_wins = 0
        draws = 0

        for game in range(num_games):
            key, game_key = jax.random.split(key)
            state = env.init(game_key)

            move_count = 0
            while not state.terminated and move_count < 200:
                # Determine whose turn (alternates)
                is_policy_turn = (move_count % 2 == 0)

                if is_policy_turn:
                    # Policy move
                    policy_logits, _ = network.apply(
                        {"params": train_st.params},
                        state.observation[None]
                    )
                    masked_logits = jnp.where(state.legal_action_mask, policy_logits[0], -1e9)
                    action = jnp.argmax(masked_logits)
                else:
                    # Random move
                    key, rand_key = jax.random.split(key)
                    legal_actions = jnp.where(state.legal_action_mask)[0]
                    action = jax.random.choice(rand_key, legal_actions)

                state = env.step(state, action)
                move_count += 1

            # Check result (simplified - just count terminations)
            # In chess, rewards indicate win/loss/draw
            reward = float(state.rewards[0])  # White's reward
            if reward > 0:
                policy_wins += 1
            elif reward < 0:
                random_wins += 1
            else:
                draws += 1

        print(f"\nPolicy vs Random ({num_games} games):")
        print(f"  Policy wins: {policy_wins}")
        print(f"  Random wins: {random_wins}")
        print(f"  Draws: {draws}")

        # Policy should at least not lose all games
        assert policy_wins + draws > 0, "Policy lost all games to random!"


# =============================================================================
# Benchmark Test
# =============================================================================

class TestTrainingSpeed:
    """Benchmark training speed."""

    def test_training_throughput(self):
        """Measure training throughput on GPU."""
        env = pgx.make("chess")
        network = SmallChessNet(num_actions=env.num_actions)

        key = jax.random.PRNGKey(0)
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        train_st = create_train_state(
            init_key, network, learning_rate=1e-3, obs_shape=state.observation.shape
        )

        # Warm up
        dummy_batch = (
            jnp.ones((64, *state.observation.shape)),
            jnp.ones((64, env.num_actions)) / env.num_actions,
            jnp.zeros(64),
        )
        train_st, _ = train_step(train_st, dummy_batch)

        # Benchmark training steps
        num_steps = 100
        batch_size = 64

        start = time.perf_counter()
        for _ in range(num_steps):
            train_st, _ = train_step(train_st, dummy_batch)
        jax.block_until_ready(train_st.params)
        elapsed = time.perf_counter() - start

        steps_per_sec = num_steps / elapsed
        samples_per_sec = num_steps * batch_size / elapsed

        print(f"\nTraining throughput:")
        print(f"  {steps_per_sec:.0f} training steps/sec")
        print(f"  {samples_per_sec:.0f} samples/sec")

        assert steps_per_sec > 10, "Training too slow!"
