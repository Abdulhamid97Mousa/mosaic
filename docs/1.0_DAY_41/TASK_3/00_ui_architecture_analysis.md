# UI Architecture Analysis: Current State vs Composable Design

## Executive Summary

The current MOSAIC UI has **two fundamental problems**:

1. **Hardcoded Scenarios**: "Human vs Agent", "Cooperation", "Competition" are fixed categories that don't map to the flexible backend abstractions
2. **Widget Explosion**: Each environment family, game type, and worker adds more widgets to the sidebar, making it unmanageable

This document analyzes the current state and proposes a composable architecture.

> **Related Research (DAY 38):**
> - [common_ground_and_differences.md](../../1.0_DAY_38/TASK_1/common_ground_and_differences.md) - gym_gui vs Ray architecture
> - [multi_policy_research_evidence.md](../../1.0_DAY_38/TASK_1/multi_policy_research_evidence.md) - AlphaGo, OpenAI Five patterns
> - [ray_architecture_analysis.md](../../1.0_DAY_38/TASK_1/ray_architecture_analysis.md) - RLlib policy mapping
> - [gym_gui_vs_ray_comparison.md](../../1.0_DAY_38/TASK_1/gym_gui_vs_ray_comparison.md) - Side-by-side comparison

---

## 0. Why Unified Flow is Better: The Core Argument

### 0.1 The Research Foundation (DAY 38)

From **multi_policy_research_evidence.md**, we identified these real-world patterns:

| Pattern | Example | What's Happening |
|---------|---------|------------------|
| **Self-Play** | AlphaGo, OpenAI Five | Current policy vs frozen past policy |
| **Human-AI** | Overcooked-AI | Human input + trained policy |
| **Mixed Algorithms** | RLlib multi-trainer | PPO agent vs DQN agent vs Random |
| **Population-Based** | Hide-and-Seek | Policy pool with evolution |

**Key Insight:** All these patterns are just **different configurations of the same primitives**:
- Which **environment**?
- Which **policy** controls which **agent**?
- Which **worker** provides each policy?
- What **mode** (train/eval/play)?

### 0.2 What RLlib Does Right

From **ray_architecture_analysis.md**:

```python
# RLlib: ONE configuration mechanism handles ALL scenarios
config.multi_agent(
    policies={
        "human": PolicySpec(policy_class=ExternalPolicy),
        "ai_aggressive": PolicySpec(config={"lr": 0.01}),
        "ai_defensive": PolicySpec(config={"lr": 0.001}),
        "random": PolicySpec(policy_class=RandomPolicy),
    },
    policy_mapping_fn=lambda agent_id, episode, **kw: {
        "player_0": "human",
        "player_1": "ai_aggressive",
    }[agent_id],
)
```

**ONE interface** handles:
- Human vs AI (player_0=human, player_1=ai)
- AI vs AI training (player_0=ai_v1, player_1=ai_v2)
- Self-play (player_0=current, player_1=frozen_past)
- Cooperation (all agents → same policy)
- Competition (different policies competing)

### 0.3 What MOSAIC UI Does Wrong

Current MOSAIC has **separate UI paths** for each scenario:

```
Multi-Agent Mode
├── Human vs Agent    ← Hardcoded: assumes exactly 1 human, 1 AI
├── Cooperation       ← Hardcoded: assumes all agents cooperate
└── Competition       ← Hardcoded: assumes all agents compete
```

**Problems:**

1. **Can't Express Mixed Scenarios:**
   - What if I want 2 humans vs 2 AIs? (no tab for that)
   - What if I want human + 2 different AIs? (no tab for that)
   - What if player_0 is CleanRL and player_1 is LLM? (no tab for that)

2. **Can't Express Self-Play:**
   - Where do I configure "current policy vs frozen checkpoint"?
   - Where do I configure "policy A vs policy B from population"?

3. **Duplicated Configuration:**
   - Environment selection appears in 4 places
   - Worker selection appears in 3 places
   - Each with slightly different UI

### 0.4 Why Unified Flow Solves This

**The Unified Flow** is better because it's **isomorphic to the backend abstractions**:

```
Backend (PolicyMappingService)         UI (Unified Flow)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Environment                      →     Step 1: Environment Selection
AgentPolicyBinding[]             →     Step 2: Agent Configuration Table
  - agent_id                     →       Column: Agent
  - policy_id                    →       Column: Actor/Policy
  - worker_id                    →       Column: Worker
WorkerConfig                     →     Step 3: Worker Configuration
RunMode (train/eval/play)        →     Step 4: Run Mode
```

**Every scenario becomes a different configuration, not a different UI:**

| Scenario | Step 2 Configuration |
|----------|----------------------|
| Human vs Stockfish | player_0: Human/—/Play, player_1: Stockfish/—/Play |
| CleanRL Self-Play | player_0: CleanRL/cleanrl/Train, player_1: CleanRL/cleanrl/Frozen |
| Human vs CleanRL | player_0: Human/—/Play, player_1: CleanRL/cleanrl/Eval |
| Mixed Training | player_0: CleanRL/cleanrl/Train, player_1: RLlib/rllib/Train |
| 3 Humans | all agents: Human/—/Play |
| LLM vs CleanRL | player_0: LLM/llm/Eval, player_1: CleanRL/cleanrl/Eval |

**No new UI code needed for new scenarios!**

### 0.5 The Critical Gap: Current UI Can't Express Backend Capabilities

From **common_ground_and_differences.md**, we identified this gap:

```
Backend (PolicyMappingService - Phase 2.1 ✅ Done)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PolicyMappingService:
  - _bindings: Dict[agent_id, AgentPolicyBinding]
  - select_action(agent_id, snapshot)     ← Per-agent!
  - select_actions(observations)          ← All agents at once!

UI (Current - ❌ Can't use this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No widget to configure per-agent bindings
- No widget to select worker per agent
- Hardcoded tabs instead of composable config
```

**The Unified Flow is the UI that CAN express what PolicyMappingService provides.**

---

## 1. Current UI Structure Analysis

### 1.1 Top-Level Tabs

```
ControlPanelWidget (1600 lines!)
├── Tab: "Human Control"
│   ├── Environment Group (Family, Game, Seed, Load)
│   ├── Game Configuration (dynamic, per-game)
│   ├── Control Mode (Human Only, Human + Agent, Agent Only)
│   ├── Game Control Flow (Start, Pause, Continue, Terminate, Step, Reset)
│   ├── Telemetry Mode (Fast Lane, Dual Path)
│   └── Status (Step, Reward, Turn, etc.)
│
├── Tab: "Single-Agent Mode"
│   ├── Active Actor (dropdown)
│   ├── Worker Integration (dropdown)
│   └── Headless Training (Train Agent, Evaluate Policy, Resume Training)
│
├── Tab: "Multi-Agent Mode" (MultiAgentTab)
│   ├── Subtab: "Human vs Agent" (HumanVsAgentTab)
│   │   ├── Environment (Family, Game, Seed)
│   │   ├── Configure AI Opponent (dialog)
│   │   ├── Player Assignment
│   │   ├── Game Controls
│   │   └── Game Status
│   │
│   ├── Subtab: "Cooperation" (MultiAgentCooperationTab)
│   │   ├── Worker Integration
│   │   ├── Cooperative Environment
│   │   └── Actions (Train, Load Policy)
│   │
│   └── Subtab: "Competition" (MultiAgentCompetitionTab)
│       ├── Worker Integration
│       ├── Competition Environment
│       ├── Training Mode (Self-Play, Population, League)
│       └── Actions (Train, Load Policy, Tournament)
│
└── Tab: "MuJoCo MPC"
    └── Launcher for MPC visualization
```

### 1.2 Problems Identified

#### Problem 1: Redundant Configuration Across Tabs

| Configuration | Human Control | Single-Agent | Human vs Agent | Cooperation | Competition |
|---------------|---------------|--------------|----------------|-------------|-------------|
| Environment   | ✓             | ✗            | ✓              | ✓           | ✓           |
| Seed          | ✓             | ✗            | ✓              | ✗           | ✗           |
| Worker        | ✗             | ✓            | ✗ (hidden)     | ✓           | ✓           |
| Actor         | ✗             | ✓            | ✗              | ✗           | ✗           |
| Mode          | ✓             | ✗            | ✗              | ✗           | ✗           |

**Result**: Same concepts scattered across different widgets.

#### Problem 2: Missing Critical Configuration

The UI **cannot express** these backend capabilities:

| Capability | Backend Support | UI Support |
|------------|-----------------|------------|
| Per-agent policy binding | PolicyMappingService | ❌ No widget |
| Worker selection per agent | AgentPolicyBinding.worker_id | ❌ No widget |
| Paradigm awareness | SteppingParadigm | ❌ Hidden |
| Multi-worker training | PolicyMappingService | ❌ Not possible |
| Headless vs Rendered toggle | TrainerService | ⚠️ Scattered |

#### Problem 3: Hardcoded Game Categories

```python
# In MultiAgentTab
class HumanVsAgentTab:  # Only for AEC games
class MultiAgentCooperationTab:  # Only for "cooperative" games
class MultiAgentCompetitionTab:  # Only for "competitive" games
```

**But**: Many games are MIXED (can be cooperative OR competitive depending on training setup). The categories are artificial.

#### Problem 4: Widget Explosion Pattern

Every new feature adds more widgets:

```python
# ControlPanelConfig has 20+ fields:
frozen_lake_config: FrozenLakeConfig
taxi_config: TaxiConfig
cliff_walking_config: CliffWalkingConfig
lunar_lander_config: LunarLanderConfig
car_racing_config: CarRacingConfig
bipedal_walker_config: BipedalWalkerConfig
minigrid_empty_config: MiniGridConfig
minigrid_doorkey_5x5_config: MiniGridConfig
minigrid_doorkey_6x6_config: MiniGridConfig
# ... and growing
```

---

## 2. The Composable Alternative

### 2.1 Core Principle

**Replace hardcoded scenarios with composable primitives:**

```
Current (Scenario-Based)          Proposed (Composable)
━━━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━
"Human vs Agent" tab       →     Environment + Agent Config
"Cooperation" tab          →     Environment + Agent Config
"Competition" tab          →     Environment + Agent Config
"Single-Agent Mode" tab    →     Environment + Agent Config

All 4 scenarios become ONE unified flow
```

### 2.2 Proposed Widget Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED CONFIGURATION PANEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 1: ENVIRONMENT SELECTION                                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Family:      [Gymnasium ▼] [PettingZoo ▼] [ViZDoom ▼] [AirSim]  │  │ │
│  │  │  Environment: [CartPole-v1                                    ▼] │  │ │
│  │  │  Seed:        [42        ] ☐ Reuse                               │  │ │
│  │  │                                                                   │  │ │
│  │  │  ╭─────────────────────────────────────────────────────────────╮ │  │ │
│  │  │  │ Paradigm: SINGLE_AGENT                                      │ │  │ │
│  │  │  │ Agents: 1 (agent_0)                                         │ │  │ │
│  │  │  │ Action Space: Discrete(2)                                   │ │  │ │
│  │  │  ╰─────────────────────────────────────────────────────────────╯ │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │  [ Load Environment ]                                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 2: AGENT CONFIGURATION (Per-Agent Policy Binding)               │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Agent        Actor             Worker           Mode            │  │ │
│  │  ├──────────────────────────────────────────────────────────────────┤  │ │
│  │  │  agent_0      [Human        ▼]  [—           ▼]  [Play       ▼] │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  For single-agent: Shows 1 row                                         │ │
│  │  For multi-agent (Chess): Shows 2 rows                                 │ │
│  │  For MPE (10 agents): Shows 10 rows or "Apply to All" option           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 3: WORKER CONFIGURATION (Dynamic based on selection)            │ │
│  │  ╭─────────────────────────────────────────────────────────────────╮   │ │
│  │  │  No workers selected. Select a worker above to configure.       │   │ │
│  │  ╰─────────────────────────────────────────────────────────────────╯   │ │
│  │                                                                         │ │
│  │  OR (if CleanRL selected):                                             │ │
│  │  ╭─────────────────────────────────────────────────────────────────╮   │ │
│  │  │  CleanRL Worker (agent_0)                                       │   │ │
│  │  │  Algorithm:       [PPO ▼]                                       │   │ │
│  │  │  Learning Rate:   [0.0003  ]                                    │   │ │
│  │  │  Total Timesteps: [100000  ]                                    │   │ │
│  │  │  [ Advanced Settings... ]                                       │   │ │
│  │  ╰─────────────────────────────────────────────────────────────────╯   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  STEP 4: RUN MODE                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │  ○ Interactive (with rendering)                                  │  │ │
│  │  │  ○ Headless Training (no rendering)                              │  │ │
│  │  │  ○ Evaluation (load trained policy)                              │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  │  [ Launch ]                                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 How Scenarios Map to Composable Config

| Old Scenario | Environment | Agent 0 | Agent 1 | Mode |
|--------------|-------------|---------|---------|------|
| Human Control (Gym) | CartPole-v1 | Human/Local/Play | — | Interactive |
| Single-Agent Training | CartPole-v1 | CleanRL/cleanrl/Train | — | Headless |
| Human vs Agent (Chess) | Chess-v6 | Human/Local/Play | CleanRL/cleanrl/Eval | Interactive |
| Cooperation (MPE) | simple_spread_v3 | CleanRL/cleanrl/Train | CleanRL/cleanrl/Train | Headless |
| Competition (Chess) | Chess-v6 | CleanRL/cleanrl/Train | CleanRL/cleanrl/Train | Headless |

**All scenarios expressible through the same interface!**

---

## 3. Key Design Questions

### Question 1: Where Does Worker-Specific Config Go?

**Option A: Inline Expansion**
```
Agent Config Row expands to show worker config when clicked
↓
[ agent_0 | CleanRL ▼ | cleanrl ▼ | Train ▼ ]
  ├── Algorithm: PPO
  ├── Learning Rate: 0.0003
  └── Total Timesteps: 100000
```

**Option B: Modal Dialog**
```
[ agent_0 | CleanRL ▼ | cleanrl ▼ | Train ▼ ] [⚙️ Configure]
                                                    ↓
                                            Opens dialog with
                                            full worker config
```

**Option C: Separate Section**
```
STEP 3: WORKER CONFIGURATION
Shows config for all selected workers (grouped)
```

**Recommendation**: Option C (separate section) for clarity, with "Quick Settings" inline.

### Question 2: How to Handle 10+ Agents?

For environments like MPE with many agents:

**Option A: Individual Rows**
```
agent_0  [Human ▼] [—  ▼] [Play ▼]
agent_1  [CleanRL ▼] [cleanrl ▼] [Train ▼]
agent_2  [CleanRL ▼] [cleanrl ▼] [Train ▼]
... (scrollable)
```

**Option B: Bulk Assignment**
```
☑ All Agents  [CleanRL ▼] [cleanrl ▼] [Train ▼]
──────────────────────────────────────────────
Exceptions:
  agent_0  [Human ▼] [— ▼] [Play ▼]
```

**Option C: Policy Groups**
```
Group "Team A": agents 0-4  → CleanRL, Train
Group "Team B": agents 5-9  → RLlib, Train
Group "Human":  agent_0     → Human, Play
```

**Recommendation**: Option B (bulk with exceptions) for simplicity.

### Question 3: How to Handle Env-Specific Config?

Currently: `build_frozenlake_controls()`, `build_minigrid_controls()`, etc.

**Proposed**: Dynamic form generation from config schema:

```python
class EnvConfigSchema(Protocol):
    """Schema for environment-specific configuration."""

    @classmethod
    def fields(cls) -> List[ConfigField]:
        """Return list of configurable fields."""
        ...

# FrozenLake would provide:
fields = [
    ConfigField(name="is_slippery", type=bool, default=False),
    ConfigField(name="grid_size", type=int, default=8, range=(4, 16)),
]
```

UI generates widgets from schema, no hardcoded per-environment code.

---

## 4. Proposed Widget Classes

### 4.1 New Core Widgets

```python
# gym_gui/ui/widgets/unified_config/

class EnvironmentSelector(QWidget):
    """Unified environment selection with paradigm detection."""
    environment_selected = Signal(EnvConfig)  # Emits full config

class AgentConfigTable(QWidget):
    """Per-agent policy/worker/mode configuration."""
    bindings_changed = Signal(Dict[str, AgentPolicyBinding])

class WorkerConfigPanel(QWidget):
    """Dynamic worker configuration based on selection."""
    # Generates form from WorkerCapabilities.config_schema

class RunModeSelector(QWidget):
    """Interactive/Headless/Evaluation selection."""
    mode_changed = Signal(RunMode)

class UnifiedLauncher(QWidget):
    """Combines all above into single Launch workflow."""
    launch_requested = Signal(LaunchConfig)
```

### 4.2 Migration Strategy

**Phase 1**: Create new widgets alongside existing
**Phase 2**: Add "Unified Config (Beta)" tab
**Phase 3**: User testing and refinement
**Phase 4**: Replace old tabs, keep as "Legacy Mode" option
**Phase 5**: Remove legacy code

---

## 5. Backend ↔ UI Mapping

| Backend Abstraction | UI Widget |
|---------------------|-----------|
| `SteppingParadigm` | Auto-detected label in EnvironmentSelector |
| `PolicyMappingService` | AgentConfigTable |
| `AgentPolicyBinding` | One row in AgentConfigTable |
| `WorkerCapabilities` | Filter for Worker dropdown |
| `WorkerCapabilities.config_schema` | WorkerConfigPanel form generation |
| `ParadigmAdapter` | Hidden (backend only) |

---

## 6. Success Criteria

1. **Single Entry Point**: One flow covers all current scenarios
2. **No Hardcoded Scenarios**: "Human vs Agent" is just a configuration
3. **Composable**: Any agent can use any policy from any worker
4. **Scalable**: Adding new worker doesn't require new widgets
5. **Discoverable**: User can see all options without navigating tabs
6. **Backward Compatible**: Presets replicate old workflows

---

## 7. Open Questions for Discussion

1. **Keep Legacy Tabs?** During transition, or remove entirely?
2. **Environment Config Location?** Inline with env selector, or separate section?
3. **Presets System?** Save/load configurations for common scenarios?
4. **Validation Feedback?** How to show incompatible configurations?

---

## Related Documents

- [TASK_1: Multi-Paradigm Orchestrator](../TASK_1/README.md) - Backend abstractions
- [TASK_2: Cognitive Orchestration Layer](../TASK_2/README.md) - LLM/VLM integration
- [PolicyMappingService Plan](../TASK_1/03_policy_mapping_service_plan.md) - Service design
