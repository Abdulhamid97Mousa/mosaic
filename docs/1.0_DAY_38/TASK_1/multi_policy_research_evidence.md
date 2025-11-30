# Multi-Policy / Multi-Brain Research Evidence

## Overview

This document provides evidence from academic research and industry practice for scenarios where multiple "brains" (policies/actors) control agents in the same environment.

---

## 1. Self-Play Training

### 1.1 AlphaGo / AlphaZero (DeepMind, 2016-2017)

**Paper:** "Mastering the game of Go with deep neural networks and tree search" (Nature, 2016)

**Architecture:**
```
Current Policy (training)  vs  Past Self (frozen snapshot)
         ↓                              ↓
      agent_0                        agent_1
         ↓                              ↓
   Learn from game              Provide diverse opponents
```

**Key insight:** The same environment (Go board) has two agents controlled by:
- **Current policy** - Being trained
- **Historical policy** - Frozen snapshot for diverse opponents

### 1.2 OpenAI Five (2018-2019)

**Paper:** "Dota 2 with Large Scale Deep Reinforcement Learning" (arXiv, 2019)

**Architecture:**
```
5 Allied Agents (team_0)     vs     5 Enemy Agents (team_1)
        ↓                                    ↓
   Current Policy                    Past Policy Pool
   (being trained)                   (sampled from history)
```

**Evidence for multiple brains:**
- 5 agents share ONE policy (parameter sharing)
- But opponent team uses DIFFERENT policy (from past checkpoints)
- Policy pool contains 100+ historical snapshots

---

## 2. Human-AI Collaboration

### 2.1 Overcooked-AI (UC Berkeley, 2019)

**Paper:** "On the Utility of Learning about Humans for Human-AI Coordination"

**GitHub:** https://github.com/HumanCompatibleAI/overcooked_ai

**Architecture:**
```
Human Player (agent_0)     +     AI Teammate (agent_1)
        ↓                              ↓
   Keyboard input              Trained cooperative policy
        ↓                              ↓
   PolicyClient              Population-based policy
```

**Key finding:** AI trained with **diverse human models** performs better than AI trained only with itself.

### 2.2 PAIRED (DeepMind, 2020)

**Paper:** "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design"

**Architecture:**
```
Protagonist Agent     Antagonist Agent     Environment Generator
       ↓                    ↓                      ↓
  Policy A (RL)        Policy B (RL)         Policy C (RL)
       ↓                    ↓                      ↓
  Solve tasks          Make tasks hard       Design curriculum
```

**Evidence:** Three separate policies, each with different objectives, in the same training loop.

---

## 3. Mixed Human-AI Gameplay

### 3.1 Hanabi Learning Environment (DeepMind, 2019)

**Paper:** "The Hanabi Challenge: A New Frontier for AI Research"

**Scenario:** Cooperative card game requiring communication

```
Human Player          AI Player 1          AI Player 2
     ↓                     ↓                    ↓
  External            Policy A             Policy B
  (via API)         (trained)            (different arch)
```

**Research question:** Can AI learn to cooperate with humans who use different conventions?

### 3.2 StarCraft II Human vs AI (DeepMind, 2019)

**Paper:** "Grandmaster level in StarCraft II using multi-agent reinforcement learning"

**AlphaStar League:**
```
Main Agent        Exploiter 1        Exploiter 2        Human Replays
    ↓                 ↓                  ↓                   ↓
 Policy A         Policy B           Policy C           Imitation
    ↓                 ↓                  ↓                   ↓
General play    Counter main      Counter exploiter   Human strategies
```

**Evidence:** Multiple distinct policies trained to counter each other.

---

## 4. RLlib Multi-Agent Examples

### 4.1 Rock-Paper-Scissors

**Source:** `ray/rllib/examples/multi_agent/rock_paper_scissors_*.py`

```python
# Different policies can use different algorithms
policies = {
    "policy_1": PolicySpec(algorithm_class=PPO),
    "policy_2": PolicySpec(algorithm_class=DQN),  # Different algorithm!
    "random": PolicySpec(policy_class=RandomPolicy),
}

policy_mapping_fn = lambda agent_id, episode, **kwargs: {
    "player_0": "policy_1",   # PPO agent
    "player_1": "policy_2",   # DQN agent
    "player_2": "random",     # Random baseline
}[agent_id]
```

### 4.2 Two-Trainer Example

**Source:** `ray/rllib/examples/multi_agent/two_trainer_workflow.py`

```python
# Two completely separate training loops
ppo_trainer = PPO(config=ppo_config)
dqn_trainer = DQN(config=dqn_config)

while True:
    # PPO trains on its experiences
    ppo_results = ppo_trainer.train()
    # DQN trains on its experiences
    dqn_results = dqn_trainer.train()
    # Exchange weights periodically
    ppo_trainer.set_weights({"opponent": dqn_trainer.get_weights(["main"])["main"]})
```

---

## 5. Population-Based Training (PBT)

### 5.1 Concept

**Paper:** "Population Based Training of Neural Networks" (DeepMind, 2017)

```
Population of Policies
┌─────────────────────────────────────────────────────┐
│  Policy 1    Policy 2    Policy 3    Policy 4      │
│     ↓           ↓           ↓           ↓          │
│  Explore    Exploit     Explore     Exploit        │
│     ↓           ↓           ↓           ↓          │
│  lr=0.01    lr=0.001    lr=0.005    lr=0.002       │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
            Natural Selection
            (copy best, mutate)
```

**Evidence:** Multiple policies with different hyperparameters compete and evolve.

### 5.2 OpenAI's Hide-and-Seek (2019)

**Paper:** "Emergent Tool Use From Multi-Agent Autocurricula"

```
Hiders (team)              vs              Seekers (team)
     ↓                                         ↓
 Policy Pool                              Policy Pool
 (PBT evolution)                         (PBT evolution)
     ↓                                         ↓
Emergent cooperation                   Emergent strategies
```

---

## 6. Implications for Your Architecture

### Evidence Summary

| Scenario | Agent 1 | Agent 2 | Research Example |
|----------|---------|---------|------------------|
| Self-play | Current policy | Past policy | AlphaGo, OpenAI Five |
| Human-AI | Human input | Trained policy | Overcooked-AI |
| Mixed | Algorithm A | Algorithm B | RLlib multi-trainer |
| Population | Policy v1 | Policy v2 | PBT, Hide-and-Seek |

### Your ActorService Maps To:

```python
# Scenario: Human vs Stockfish Chess
actor_mapping = {
    "player_0": HumanKeyboardActor(),  # Human plays white
    "player_1": StockfishActor(),      # AI plays black
}

# Scenario: Self-play with frozen opponent
actor_mapping = {
    "player_0": CurrentPolicyActor(),  # Training
    "player_1": FrozenPolicyActor(),   # Checkpoint from 1000 steps ago
}

# Scenario: Population-based opponents
actor_mapping = {
    "player_0": CurrentPolicyActor(),
    "player_1": SampledFromPoolActor(),  # Random selection from policy pool
}
```

### Recommended Renaming

Based on industry terminology:

| Current | Recommended | Reason |
|---------|-------------|--------|
| `Actor` | `PolicyController` | Aligns with RLlib's policy concept |
| `ActorService` | `PolicyMappingService` | Matches RLlib's `policy_mapping_fn` |
| `HumanKeyboardActor` | `HumanPolicyController` | Consistent naming |
| `CleanRLWorkerActor` | `ExternalPolicyController` | Like RLlib's ExternalEnv |

---

## 7. References

1. Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature.
2. OpenAI. (2019). "Dota 2 with Large Scale Deep Reinforcement Learning." arXiv.
3. Carroll, M., et al. (2019). "On the Utility of Learning about Humans for Human-AI Coordination." NeurIPS.
4. Dennis, M., et al. (2020). "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design." NeurIPS.
5. Bard, N., et al. (2019). "The Hanabi Challenge: A New Frontier for AI Research." arXiv.
6. Vinyals, O., et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature.
7. Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks." arXiv.
8. Baker, B., et al. (2019). "Emergent Tool Use From Multi-Agent Autocurricula." arXiv.
