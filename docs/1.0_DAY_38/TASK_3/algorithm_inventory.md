# CleanRL Algorithm Inventory

## Quick Reference

| Category | Count | Adapter | Key Characteristic |
|----------|-------|---------|-------------------|
| PPO Family | 14 | PPOSelector | `get_action_and_value()` |
| DQN Family | 7 | DQNSelector | `argmax(Q(s))` |
| C51 (Distributional) | 2 | C51Selector | `n_atoms`, `v_min`, `v_max` |
| DDPG | 1 | DDPGSelector | Actor-Critic tuple |
| TD3 | 1 | TD3Selector | Twin Q-networks |
| SAC | 2 | SACSelector | Soft Actor-Critic |
| **PyTorch Total** | **27** | **6 adapters** | |
| JAX Algorithms | 10 | (future) | Flax models |

---

## Full Algorithm List

### PPO Family (14 algorithms)

```
ppo                           # Discrete, Classic Control
ppo_continuous_action         # Continuous, MuJoCo
ppo_atari                     # Discrete, Atari
ppo_atari_lstm                # Discrete, Atari + LSTM
ppo_atari_envpool             # Discrete, Atari + envpool
ppo_atari_multigpu            # Discrete, Atari + Multi-GPU
ppo_procgen                   # Discrete, Procgen
ppo_pettingzoo_ma_atari       # Discrete, Multi-Agent
ppo_rnd_envpool               # Discrete, Atari + RND
ppo_continuous_action_isaacgym # Continuous, IsaacGym
ppo_trxl                      # Discrete, Transformer XL
ppg_procgen                   # Discrete, Phasic Policy Gradient
rpo_continuous_action         # Continuous, Robust PPO
```

### DQN Family (7 algorithms)

```
dqn                           # Discrete, Classic Control
dqn_atari                     # Discrete, Atari
pqn                           # Discrete, Periodic Q-Network
pqn_atari_envpool             # Discrete, Atari + envpool
pqn_atari_envpool_lstm        # Discrete, Atari + LSTM
qdagger_dqn_atari_impalacnn   # Discrete, QDagger
rainbow_atari                 # Discrete, Rainbow DQN
```

### C51 / Distributional (2 algorithms)

```
c51                           # Discrete, Classic Control
c51_atari                     # Discrete, Atari
```

### DDPG (1 algorithm)

```
ddpg_continuous_action        # Continuous, MuJoCo
```

### TD3 (1 algorithm)

```
td3_continuous_action         # Continuous, MuJoCo
```

### SAC (2 algorithms)

```
sac_continuous_action         # Continuous, MuJoCo
sac_atari                     # Discrete, Atari
```

### JAX Algorithms (10 algorithms)

```
dqn_jax                       # JAX/Flax DQN
dqn_atari_jax                 # JAX/Flax DQN Atari
ddpg_continuous_action_jax    # JAX/Flax DDPG
td3_continuous_action_jax     # JAX/Flax TD3
c51_jax                       # JAX/Flax C51
c51_atari_jax                 # JAX/Flax C51 Atari
ppo_atari_envpool_xla_jax     # JAX/Flax PPO
ppo_atari_envpool_xla_jax_scan # JAX/Flax PPO with scan
qdagger_dqn_atari_jax_impalacnn # JAX/Flax QDagger
```

---

## Checkpoint Formats

| Format | Algorithms | Loading Pattern |
|--------|------------|-----------------|
| Simple `state_dict` | PPO, DQN, PQN, Rainbow | `torch.load()` → `model.load_state_dict()` |
| Dict `{args, weights}` | C51 | `checkpoint["model_weights"]` |
| Tuple `(actor, qf)` | DDPG | `actor_params, qf_params = torch.load()` |
| Tuple `(actor, qf1, qf2)` | TD3, SAC | `actor_params, qf1, qf2 = torch.load()` |

---

## make_env Signatures

| Signature | Algorithms |
|-----------|------------|
| `(env_id, idx, capture_video, run_name, gamma)` | PPO family |
| `(env_id, idx, seed, capture_video, run_name)` | DQN, DDPG, TD3, SAC, C51 |

---

## Action Selection Methods

| Method | Algorithms |
|--------|------------|
| `agent.get_action_and_value(obs)` → `(actions, _, _, _)` | PPO family |
| `argmax(model(obs), dim=1)` | DQN family |
| `model.get_action(obs)` → `(actions, _)` | C51 family |
| `actor(obs)` + noise | DDPG, TD3 |
| `actor.get_action(obs)` → `(actions, _, _)` | SAC |
