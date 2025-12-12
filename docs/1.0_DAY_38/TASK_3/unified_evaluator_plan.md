# Unified CleanRL Evaluator Plan

## Problem Statement

The current CleanRL evaluation system has separate `*_eval.py` files for each algorithm, leading to:

- Code duplication (each file has nearly identical episode loops)
- Difficult maintenance when adding new algorithms
- Inconsistent interfaces across algorithms
- Over-engineered monkey-patching in `sitecustomize.py`

## Proposed Solution: Unified Evaluator with Algorithm Adapters

Following XuanCe's pattern of explicit, simple code, we propose a **unified evaluation system** that:

1. Has a single evaluation loop for all algorithms
2. Uses algorithm-specific "ActionSelector" adapters for model loading and action selection
3. Explicit checkpoint loading (no monkey-patching)
4. Clean separation of concerns

---

## Algorithm Inventory

### PyTorch Algorithms (37 total)

#### Category 1: Single Agent Model (PPO Family) - 14 algorithms

These use `Agent` class with `get_action_and_value()` method.

| Algorithm | File | Model Class | Action Space | Environment |
|-----------|------|-------------|--------------|-------------|
| ppo | `ppo.py` | Agent | Discrete | Classic Control |
| ppo_continuous_action | `ppo_continuous_action.py` | Agent | Continuous | MuJoCo |
| ppo_atari | `ppo_atari.py` | Agent | Discrete | Atari |
| ppo_atari_lstm | `ppo_atari_lstm.py` | Agent | Discrete | Atari |
| ppo_atari_envpool | `ppo_atari_envpool.py` | Agent | Discrete | Atari (envpool) |
| ppo_atari_multigpu | `ppo_atari_multigpu.py` | Agent | Discrete | Atari |
| ppo_procgen | `ppo_procgen.py` | Agent | Discrete | Procgen |
| ppo_pettingzoo_ma_atari | `ppo_pettingzoo_ma_atari.py` | Agent | Discrete | Multi-Agent Atari |
| ppo_rnd_envpool | `ppo_rnd_envpool.py` | Agent | Discrete | Atari (RND) |
| ppo_continuous_action_isaacgym | `ppo_continuous_action_isaacgym/` | Agent | Continuous | IsaacGym |
| ppo_trxl | `ppo_trxl/ppo_trxl.py` | Agent | Discrete | PoM Env |
| ppg_procgen | `ppg_procgen.py` | Agent | Discrete | Procgen |
| rpo_continuous_action | `rpo_continuous_action.py` | Agent | Continuous | MuJoCo |

**Checkpoint format:** Simple `state_dict`
**Action selection:** `agent.get_action_and_value(obs)` → returns `(actions, _, _, _)`

#### Category 2: Single QNetwork (DQN Family) - 7 algorithms

These use `QNetwork` class called directly with argmax.

| Algorithm | File | Model Class | Action Space | Environment |
|-----------|------|-------------|--------------|-------------|
| dqn | `dqn.py` | QNetwork | Discrete | Classic Control |
| dqn_atari | `dqn_atari.py` | QNetwork | Discrete | Atari |
| pqn | `pqn.py` | QNetwork | Discrete | Classic Control |
| pqn_atari_envpool | `pqn_atari_envpool.py` | QNetwork | Discrete | Atari (envpool) |
| pqn_atari_envpool_lstm | `pqn_atari_envpool_lstm.py` | QNetwork | Discrete | Atari (LSTM) |
| qdagger_dqn_atari_impalacnn | `qdagger_dqn_atari_impalacnn.py` | QNetwork | Discrete | Atari |
| rainbow_atari | `rainbow_atari.py` | QNetwork | Discrete | Atari |

**Checkpoint format:** Simple `state_dict`
**Action selection:** `argmax(model(obs), dim=1)` with epsilon-greedy

#### Category 3: Distributional RL (C51 Family) - 2 algorithms

These use `QNetwork` with distributional parameters.

| Algorithm | File | Model Class | Action Space | Environment |
|-----------|------|-------------|--------------|-------------|
| c51 | `c51.py` | QNetwork | Discrete | Classic Control |
| c51_atari | `c51_atari.py` | QNetwork | Discrete | Atari |

**Checkpoint format:** Dict with `{"args": {...}, "model_weights": state_dict}`
**Action selection:** `model.get_action(obs)` with epsilon-greedy
**Extra constructor args:** `n_atoms`, `v_min`, `v_max`

#### Category 4: Actor-Critic Continuous (DDPG) - 1 algorithm

Uses separate Actor and QNetwork.

| Algorithm | File | Model Classes | Action Space | Environment |
|-----------|------|---------------|--------------|-------------|
| ddpg_continuous_action | `ddpg_continuous_action.py` | (Actor, QNetwork) | Continuous | MuJoCo |

**Checkpoint format:** Tuple `(actor_state_dict, qf_state_dict)`
**Action selection:** `actor(obs)` + exploration noise

#### Category 5: Twin Q-Network (TD3) - 1 algorithm

Uses Actor and twin Q-networks.

| Algorithm | File | Model Classes | Action Space | Environment |
|-----------|------|---------------|--------------|-------------|
| td3_continuous_action | `td3_continuous_action.py` | (Actor, QNetwork) | Continuous | MuJoCo |

**Checkpoint format:** Tuple `(actor_state_dict, qf1_state_dict, qf2_state_dict)`
**Action selection:** `actor(obs)` + exploration noise

#### Category 6: Soft Actor-Critic (SAC) - 2 algorithms

Uses Actor and SoftQNetwork.

| Algorithm | File | Model Classes | Action Space | Environment |
|-----------|------|---------------|--------------|-------------|
| sac_continuous_action | `sac_continuous_action.py` | (Actor, SoftQNetwork) | Continuous | MuJoCo |
| sac_atari | `sac_atari.py` | (Actor, SoftQNetwork) | Discrete | Atari |

**Checkpoint format:** Tuple `(actor_state_dict, qf1_state_dict, qf2_state_dict)`
**Action selection:** `actor.get_action(obs)` (stochastic) or deterministic mean

### JAX Algorithms (10 total)

| Algorithm | File | Framework |
|-----------|------|-----------|
| dqn_jax | `dqn_jax.py` | JAX/Flax |
| dqn_atari_jax | `dqn_atari_jax.py` | JAX/Flax |
| ddpg_continuous_action_jax | `ddpg_continuous_action_jax.py` | JAX/Flax |
| td3_continuous_action_jax | `td3_continuous_action_jax.py` | JAX/Flax |
| c51_jax | `c51_jax.py` | JAX/Flax |
| c51_atari_jax | `c51_atari_jax.py` | JAX/Flax |
| ppo_atari_envpool_xla_jax | `ppo_atari_envpool_xla_jax.py` | JAX/Flax |
| ppo_atari_envpool_xla_jax_scan | `ppo_atari_envpool_xla_jax_scan.py` | JAX/Flax |
| qdagger_dqn_atari_jax_impalacnn | `qdagger_dqn_atari_jax_impalacnn.py` | JAX/Flax |

**Note:** JAX algorithms will need separate adapters due to different model loading/inference patterns.

---

## Architecture Design

### File Structure

```
cleanrl_worker/
├── unified_eval/
│   ├── __init__.py
│   ├── evaluator.py           # Main unified evaluation loop
│   ├── registry.py            # Algorithm → Adapter mapping
│   ├── base.py                # ActionSelector protocol
│   └── adapters/
│       ├── __init__.py
│       ├── ppo.py             # PPO family adapter
│       ├── dqn.py             # DQN family adapter
│       ├── c51.py             # C51 family adapter
│       ├── ddpg.py            # DDPG adapter
│       ├── td3.py             # TD3 adapter
│       ├── sac.py             # SAC adapter
│       └── jax/               # JAX adapters (future)
│           ├── __init__.py
│           └── ...
```

### Core Components

#### 1. ActionSelector Protocol (`base.py`)

```python
from typing import Protocol, runtime_checkable
import numpy as np
import torch

@runtime_checkable
class ActionSelector(Protocol):
    """Protocol for algorithm-specific model loading and action selection."""

    def load(
        self,
        model_path: str,
        envs,
        device: torch.device,
        **kwargs
    ) -> None:
        """Load model weights from checkpoint."""
        ...

    def select_action(self, obs: torch.Tensor) -> np.ndarray:
        """Select action given observation."""
        ...

    def close(self) -> None:
        """Cleanup resources."""
        ...
```

#### 2. Unified Evaluator (`evaluator.py`)

```python
def evaluate(
    selector: ActionSelector,
    envs,
    eval_episodes: int,
    writer: SummaryWriter | None = None,
) -> list[float]:
    """Algorithm-agnostic evaluation loop.

    This single function handles evaluation for ALL CleanRL algorithms.
    The algorithm-specific logic is encapsulated in the ActionSelector.
    """
    obs, _ = envs.reset()
    episodic_returns = []
    episode_lengths = []

    while len(episodic_returns) < eval_episodes:
        actions = selector.select_action(obs)
        next_obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_return = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    episodic_returns.append(ep_return)
                    episode_lengths.append(ep_length)

                    # Log to TensorBoard if writer provided
                    if writer:
                        step = len(episodic_returns)
                        writer.add_scalar("eval/episodic_return", ep_return, step)
                        writer.add_scalar("eval/episode_length", ep_length, step)

        obs = next_obs

    selector.close()
    return episodic_returns
```

#### 3. Adapter Registry (`registry.py`)

```python
ADAPTER_REGISTRY: dict[str, type[ActionSelector]] = {
    # PPO Family
    "ppo": PPOSelector,
    "ppo_continuous_action": PPOSelector,
    "ppo_atari": PPOSelector,
    "ppo_atari_lstm": PPOLSTMSelector,
    "ppo_atari_envpool": PPOSelector,
    "ppo_atari_multigpu": PPOSelector,
    "ppo_procgen": PPOSelector,
    "ppo_pettingzoo_ma_atari": PPOMultiAgentSelector,
    "ppo_rnd_envpool": PPOSelector,
    "ppg_procgen": PPOSelector,  # Same interface as PPO
    "rpo_continuous_action": PPOSelector,  # Same interface as PPO

    # DQN Family
    "dqn": DQNSelector,
    "dqn_atari": DQNSelector,
    "pqn": DQNSelector,
    "pqn_atari_envpool": DQNSelector,
    "pqn_atari_envpool_lstm": DQNLSTMSelector,
    "qdagger_dqn_atari_impalacnn": DQNSelector,
    "rainbow_atari": DQNSelector,

    # C51 Family
    "c51": C51Selector,
    "c51_atari": C51Selector,

    # Actor-Critic Continuous
    "ddpg_continuous_action": DDPGSelector,
    "td3_continuous_action": TD3Selector,
    "sac_continuous_action": SACSelector,
    "sac_atari": SACDiscreteSelector,
}
```

---

## Adapter Implementations

### PPO Family Adapter (`adapters/ppo.py`)

```python
class PPOSelector:
    """ActionSelector for PPO-family algorithms."""

    def __init__(self, agent_cls: type):
        self.agent_cls = agent_cls
        self.agent = None
        self.device = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device
        self.agent = self.agent_cls(envs).to(device)
        self.agent.load_state_dict(torch.load(model_path, map_location=device))
        self.agent.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            actions, _, _, _ = self.agent.get_action_and_value(
                torch.tensor(obs, dtype=torch.float32).to(self.device)
            )
        return actions.cpu().numpy()

    def close(self):
        pass
```

### DQN Family Adapter (`adapters/dqn.py`)

```python
class DQNSelector:
    """ActionSelector for DQN-family algorithms."""

    def __init__(self, model_cls: type, epsilon: float = 0.05):
        self.model_cls = model_cls
        self.epsilon = epsilon
        self.model = None
        self.device = None
        self.envs = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device
        self.envs = envs
        self.model = self.model_cls(envs).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self.epsilon:
            return np.array([self.envs.single_action_space.sample()
                           for _ in range(self.envs.num_envs)])

        with torch.no_grad():
            q_values = self.model(torch.tensor(obs, dtype=torch.float32).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def close(self):
        pass
```

### C51 Adapter (`adapters/c51.py`)

```python
class C51Selector:
    """ActionSelector for C51 distributional RL."""

    def __init__(self, model_cls: type, epsilon: float = 0.05):
        self.model_cls = model_cls
        self.epsilon = epsilon
        self.model = None
        self.device = None
        self.envs = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device
        self.envs = envs

        # C51 checkpoint has special format with args
        checkpoint = torch.load(model_path, map_location="cpu")
        args = checkpoint["args"]

        self.model = self.model_cls(
            envs,
            n_atoms=args["n_atoms"],
            v_min=args["v_min"],
            v_max=args["v_max"],
        ).to(device)
        self.model.load_state_dict(checkpoint["model_weights"])
        self.model.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self.epsilon:
            return np.array([self.envs.single_action_space.sample()
                           for _ in range(self.envs.num_envs)])

        with torch.no_grad():
            actions, _ = self.model.get_action(
                torch.tensor(obs, dtype=torch.float32).to(self.device)
            )
        return actions.cpu().numpy()

    def close(self):
        pass
```

### DDPG Adapter (`adapters/ddpg.py`)

```python
class DDPGSelector:
    """ActionSelector for DDPG."""

    def __init__(self, actor_cls: type, qf_cls: type, exploration_noise: float = 0.1):
        self.actor_cls = actor_cls
        self.qf_cls = qf_cls
        self.exploration_noise = exploration_noise
        self.actor = None
        self.device = None
        self.envs = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device
        self.envs = envs

        # DDPG checkpoint is tuple: (actor_params, qf_params)
        actor_params, qf_params = torch.load(model_path, map_location=device)

        self.actor = self.actor_cls(envs).to(device)
        self.actor.load_state_dict(actor_params)
        self.actor.eval()
        # Note: QNetwork not needed for evaluation

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            actions = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        return actions.cpu().numpy().clip(
            self.envs.single_action_space.low,
            self.envs.single_action_space.high,
        )

    def close(self):
        pass
```

### TD3 Adapter (`adapters/td3.py`)

```python
class TD3Selector:
    """ActionSelector for TD3."""

    def __init__(self, actor_cls: type, qf_cls: type, exploration_noise: float = 0.1):
        self.actor_cls = actor_cls
        self.qf_cls = qf_cls
        self.exploration_noise = exploration_noise
        self.actor = None
        self.device = None
        self.envs = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device
        self.envs = envs

        # TD3 checkpoint is triple: (actor_params, qf1_params, qf2_params)
        actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)

        self.actor = self.actor_cls(envs).to(device)
        self.actor.load_state_dict(actor_params)
        self.actor.eval()
        # Note: QNetworks not needed for evaluation

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            actions = self.actor(torch.tensor(obs, dtype=torch.float32).to(self.device))
            actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
        return actions.cpu().numpy().clip(
            self.envs.single_action_space.low,
            self.envs.single_action_space.high,
        )

    def close(self):
        pass
```

### SAC Adapter (`adapters/sac.py`)

```python
class SACSelector:
    """ActionSelector for SAC (continuous action)."""

    def __init__(self, actor_cls: type, qf_cls: type, deterministic: bool = True):
        self.actor_cls = actor_cls
        self.qf_cls = qf_cls
        self.deterministic = deterministic
        self.actor = None
        self.device = None

    def load(self, model_path: str, envs, device: torch.device, **kwargs):
        self.device = device

        # SAC checkpoint is triple: (actor_params, qf1_params, qf2_params)
        actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)

        self.actor = self.actor_cls(envs).to(device)
        self.actor.load_state_dict(actor_params)
        self.actor.eval()

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.deterministic:
                # Use mean of the action distribution
                actions, _, _ = self.actor.get_action(
                    torch.tensor(obs, dtype=torch.float32).to(self.device)
                )
            else:
                actions, _, _ = self.actor.get_action(
                    torch.tensor(obs, dtype=torch.float32).to(self.device)
                )
        return actions.cpu().numpy()

    def close(self):
        pass
```

---

## Integration with MOSAIC Runtime

### Changes to `runtime.py`

Replace the current evaluation code with:

```python
from cleanrl_worker.unified_eval import evaluate, get_adapter

def run_eval(self, config: EvalConfig):
    """Run evaluation using unified evaluator."""

    # Get the appropriate adapter for this algorithm
    adapter = get_adapter(config.algorithm, config.model_cls)

    # Create environment
    envs = gym.vector.SyncVectorEnv([
        config.make_env(config.env_id, 0, config.capture_video, config.run_name, config.gamma)
    ])

    # Load model
    adapter.load(config.model_path, envs, config.device)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config.tensorboard_dir)

    # Run evaluation
    returns = evaluate(adapter, envs, config.eval_episodes, writer=writer)

    writer.close()
    envs.close()

    return returns
```

### Remove from `sitecustomize.py`

The following sections can be **removed** since we're using explicit loading:

- Lines 154-216: `nn.Module.to()` patch for checkpoint auto-loading
- Lines 218-245: `ppo_eval.evaluate` patch

These were workarounds; the unified evaluator handles this properly.

---

## Implementation Phases

### Phase 1: Core Infrastructure (Priority: High)

1. Create `unified_eval/` package structure
2. Implement `ActionSelector` protocol in `base.py`
3. Implement unified `evaluate()` function in `evaluator.py`
4. Create adapter registry in `registry.py`

### Phase 2: PyTorch Adapters (Priority: High)

1. Implement `PPOSelector` - covers 14 algorithms
2. Implement `DQNSelector` - covers 7 algorithms
3. Implement `C51Selector` - covers 2 algorithms
4. Implement `DDPGSelector` - covers 1 algorithm
5. Implement `TD3Selector` - covers 1 algorithm
6. Implement `SACSelector` - covers 2 algorithms

**Total PyTorch coverage: 27 algorithms with 6 adapters**

### Phase 3: Special Cases (Priority: Medium)

1. `PPOLSTMSelector` for LSTM-based algorithms
2. `PPOMultiAgentSelector` for PettingZoo
3. `DQNLSTMSelector` for LSTM DQN variants

### Phase 4: JAX Support (Priority: Low)

1. Create `adapters/jax/` subpackage
2. Implement JAX equivalents of each adapter
3. Handle Flax model loading

### Phase 5: Integration & Cleanup (Priority: High)

1. Update `runtime.py` to use unified evaluator
2. Remove monkey-patches from `sitecustomize.py`
3. Update `eval_registry.py` to use new adapters
4. Add tests for each adapter

---

## Benefits

1. **Single evaluation loop** - No code duplication
2. **Easy to extend** - Add new algorithm = add new adapter
3. **Explicit loading** - No monkey-patching magic
4. **Type-safe** - Protocol-based design with runtime checking
5. **Testable** - Each adapter can be unit tested independently
6. **Maintainable** - Clear separation of concerns

---

## Summary Table

| Adapter Class | Algorithms Covered | Checkpoint Format | Action Method |
|--------------|-------------------|-------------------|---------------|
| PPOSelector | 14 | state_dict | get_action_and_value() |
| DQNSelector | 7 | state_dict | argmax(model()) |
| C51Selector | 2 | {args, weights} | get_action() |
| DDPGSelector | 1 | (actor, qf) tuple | actor() + noise |
| TD3Selector | 1 | (actor, qf1, qf2) tuple | actor() + noise |
| SACSelector | 2 | (actor, qf1, qf2) tuple | get_action() |

**Total: 6 adapters cover 27 PyTorch algorithms**
