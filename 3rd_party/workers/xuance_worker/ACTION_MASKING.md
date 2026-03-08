# Action Masking for XuanCe Worker

## Overview

Action masking improves training efficiency by preventing agents from selecting irrelevant actions during training. For Soccer/Basketball environments, we mask actions [0, 6, 7] (noop, toggle, done), reducing the effective action space from 8 to 5 actions.

## Approach 2: Invalid Action Masking (Recommended)

**Why Approach 2?**
- Policy outputs 8 actions (full action space)
- During training, invalid actions have their logits set to -inf
- Agent learns to never select masked actions
- During deployment, policy works directly with environment (no wrapper needed)
- Clean deployment: same 8-action input/output during training and inference

## Implementation Status

**CleanRL Worker**: ✅ Implemented
- Added `--mask-invalid-actions` flag to PPO-GRU algorithm
- Masks actions [0, 6, 7] by default
- See: `cleanrl_worker/algorithms/ppo_gru.py`

**XuanCe Worker**: ⚠️ Requires XuanCe Library Modification
- XuanCe is a third-party library with its own policy implementation
- Action masking needs to be added to XuanCe's policy classes
- Alternative: Use environment wrapper (but this changes action space)

## Masked Actions for Soccer/Basketball

```
Action Space (8 actions):
  0: noop    ❌ MASKED (only for AEC mode, not needed in training)
  1: left    ✅ KEEP
  2: right   ✅ KEEP
  3: forward ✅ KEEP
  4: pickup  ✅ KEEP
  5: drop    ✅ KEEP
  6: toggle  ❌ MASKED (no toggleable objects in Soccer/Basketball)
  7: done    ❌ MASKED (episode terminates automatically)
```

**Effective action space**: 5 actions [1, 2, 3, 4, 5]
**Reduction**: 37.5% (8 → 5 actions)

## Benefits

1. **Faster Learning**: Eliminates exploration of irrelevant actions
2. **Better Sample Efficiency**: Agent focuses on task-relevant behaviors
3. **Clean Deployment**: No wrapper needed, policy dimension matches environment
4. **Consistent Interface**: Same 8-action space during training and deployment

## TODO: XuanCe Implementation

To implement action masking in XuanCe worker, modify:

1. **Policy Classes** (`xuance/xuance/torch/policies/categorical_marl.py`):
   ```python
   def forward(self, observation, agent_ids, avail_actions=None, **kwargs):
       # Add action_mask parameter
       logits = self.actor_net(...)

       # Apply action mask (set invalid actions to -inf)
       if action_mask is not None:
           logits = logits + action_mask

       dist = Categorical(logits=logits)
       return dist
   ```

2. **Agent Classes** (`xuance/xuance/torch/agents/multi_agent_rl/mappo_agents.py`):
   - Pass action_mask through to policy.forward()
   - Create mask tensor from configuration

3. **Configuration** (add to config JSON):
   ```json
   {
     "mask_invalid_actions": true,
     "invalid_actions": [0, 6, 7]
   }
   ```

## Workaround: Environment Wrapper

Until XuanCe library is modified, you can use an environment wrapper:

```python
class InvalidActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, invalid_actions=[0, 6, 7]):
        super().__init__(env)
        self.invalid_actions = invalid_actions
        # Note: This keeps action_space at 8 dimensions

    def step(self, action):
        # Remap invalid actions to valid ones (fallback)
        if action in self.invalid_actions:
            action = 3  # Default to forward
        return self.env.step(action)
```

**Limitation**: This wrapper doesn't prevent the policy from learning invalid actions, it just remaps them at execution time. True action masking requires modifying the policy's logits before sampling.

## Notes for Training Scripts

All XuanCe training scripts should include this note:

```bash
# ACTION MASKING NOTE:
# This script trains with the full 8-action space. To enable action masking
# (mask actions [0, 6, 7]: noop, toggle, done), XuanCe library modifications
# are required. See: 3rd_party/workers/xuance_worker/ACTION_MASKING.md
#
# Benefits of action masking:
#   - 37.5% reduction in action space (8 → 5 effective actions)
#   - Faster learning by eliminating irrelevant exploration
#   - Policy still outputs 8 dimensions (clean deployment)
```
