# DAY 41 - TASK 3: UI Architecture for Multi-Framework Support

## Problem Statement

The current MOSAIC UI is **hardcoded for specific use cases** rather than being **composable from abstractions**. This makes it impossible to properly support multiple RL frameworks (CleanRL, RLlib, Jason, LLM, VLM) without creating a maze of widgets.

> **Related Tasks:**
> - [TASK_1: Multi-Paradigm Orchestrator](../TASK_1/README.md) - Backend abstractions
> - [TASK_2: Cognitive Orchestration Layer](../TASK_2/README.md) - LLM/VLM integration
>
> **Related Research (DAY 38):**
> - [common_ground_and_differences.md](../../1.0_DAY_38/TASK_1/common_ground_and_differences.md)
> - [multi_policy_research_evidence.md](../../1.0_DAY_38/TASK_1/multi_policy_research_evidence.md)
> - [ray_architecture_analysis.md](../../1.0_DAY_38/TASK_1/ray_architecture_analysis.md)

## Decision: Thin Wrappers + Advanced Tab

**Approach chosen:** Keep existing tabs as thin wrappers, add "Advanced" tab for full flexibility.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tabs                                                                        │
├──────────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│ Human        │ Single-Agent │ Multi-Agent  │ MuJoCo MPC   │ Advanced       │
│ Control      │ Mode         │ Mode         │ (keep as-is) │ (NEW)          │
├──────────────┴──────────────┴──────────────┴──────────────┴────────────────┤
│                                                                              │
│  Existing tabs: Thin wrappers that generate PolicyMappingService configs    │
│  Advanced tab: Full Unified Flow for scenarios other tabs can't express    │
│  MuJoCo MPC: Keep as-is (not RL, doesn't use PolicyMappingService)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Documents

| # | Document | Description | Status |
|---|----------|-------------|--------|
| 00 | [UI Architecture Analysis](./00_ui_architecture_analysis.md) | Deep analysis of current vs proposed UI | ✅ Complete |
| 01 | [UI Migration Plan](./01_ui_migration_plan.md) | Step-by-step migration plan | ✅ Phase 1-2 Done |
| 02 | [Advanced Tab Components](./02_advanced_tab_components.md) | Technical component documentation | ✅ Complete |
| 03 | [Advanced Tab UI Reference](./03_advanced_tab_ui_reference.md) | Complete UI reference with examples | ✅ Complete |

## Progress Tracking

### Phase 1: Advanced Tab Widgets ✅

- [x] Create `gym_gui/ui/widgets/advanced_config/` module
- [x] Implement `EnvironmentSelector` widget (Step 1)
- [x] Implement `AgentConfigTable` widget (Step 2)
- [x] Implement `WorkerConfigPanel` widget (Step 3)
- [x] Implement `RunModeSelector` widget (Step 4)
- [x] Implement `AdvancedConfigTab` container

### Phase 2: Integration ✅

- [x] Add Advanced tab to `ControlPanelWidget`
- [x] Connect signals (`advanced_launch_requested`, `advanced_env_load_requested`)
- [x] Pyright verification: 0 errors
- [x] Pytest verification: 396 tests passed

### Phase 2.5: UI Enhancements ✅

- [x] **Agent Count Display**: EnvironmentSelector now shows prominent agent count
  - Single-Agent environments: "Single-Agent"
  - Multi-Agent environments: "Multi-Agent (N agents)" in blue/bold
  - Agent list truncated for environments with many agents (e.g., pistonball_v6 with 20 pistons)
- [x] **Worker Configuration Panel**: All workers now have configuration schemas
  - `local`: Render Mode, Record Video
  - `cleanrl`: Algorithm, Learning Rate, Total Timesteps, Parallel Envs, Capture Video
  - `rllib`: Algorithm, Num Workers, Envs per Worker, Framework
  - `xuance`: Algorithm (MAPPO, MADDPG, QMIX, etc.), Learning Rate, Batch Size, Backend
  - `llm`: Model, Temperature, Max Tokens, System Prompt
  - `jason`: Agent File, MAS File, Debug Mode
  - `spade_bdi`: XMPP Server, Agent JID, Debug Mode
- [x] **Extended Actor/Policy Options**: Added more policy choices
  - CleanRL: PPO, DQN, SAC
  - RLlib: PPO, DQN
  - XuanCe MARL: MAPPO, MADDPG, QMIX
  - Others: Stockfish, LLM, BDI

### Phase 3: Thin Wrappers 📋

- [ ] Refactor HumanVsAgentTab to use PolicyMappingService
- [ ] Refactor SingleAgentTab to use PolicyMappingService
- [ ] Refactor CooperationTab to use PolicyMappingService
- [ ] Refactor CompetitionTab to use PolicyMappingService

### Phase 4: SessionController Integration 📋

- [ ] Update SessionController to use PolicyMappingService for action selection

## The Core Problem

### Current UI Structure (Hardcoded Scenarios)

```
Multi-Agent Mode
├── Human Vs Agent     ← Hardcoded scenario
│   ├── Environment (Family, Game, Seed)
│   ├── Configure AI Opponent    ← WHAT opponent? CleanRL? LLM? RLlib?
│   ├── Player Assignment
│   └── Game Controls
├── Cooperation        ← Another hardcoded scenario
└── Competition        ← Another hardcoded scenario
```

**Missing Configuration Points:**
- Which WORKER trains/runs each agent?
- Which POLICY each agent uses?
- CleanRL vs RLlib vs Jason vs LLM?
- Headless vs rendered training?
- Worker-specific parameters?

### The Fundamental Questions

1. **Where does the user select the WORKER?** (CleanRL, RLlib, Jason, LLM, VLM)
2. **Where does the user configure WORKER-SPECIFIC parameters?** (Each worker has different configs)
3. **How do we avoid sidebar explosion?** (Each worker adds more widgets)
4. **How do we handle per-agent configuration in multi-agent?**

## Design Options Under Consideration

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Wizard-style flow | Clean, guided | Multiple clicks |
| B | Tabbed configuration | All visible | Overwhelming |
| C | Collapsed sections | Current approach | Sidebar explosion |
| D | Modal/Dialog | Sidebar clean | Hidden complexity |

## Progress Tracking

- [ ] Analyze current UI widget structure
- [ ] Map current widgets to abstractions
- [ ] Identify redundant/conflicting widgets
- [ ] Design composable widget architecture
- [ ] Create wireframes for new design
- [ ] Plan migration from current to new UI

## Key Insight

The UI should reflect the **backend abstractions** we built in TASK_1:

```
Backend (TASK_1)              UI (TASK_3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PolicyMappingService    →    Agent Configuration Panel
├── agent_id            →    Agent dropdown/list
├── policy_id           →    Policy/Actor dropdown
└── worker_id           →    Worker dropdown

WorkerCapabilities      →    Dynamic Worker Config Panel
├── paradigm            →    (auto-detected from env)
├── action_spaces       →    (validated against env)
└── config_schema       →    Generated form from schema

SteppingParadigm        →    (hidden, auto-detected)
```

The UI should NOT have hardcoded "Human vs Agent" / "Cooperation" / "Competition" categories. Instead, it should let users **compose** any configuration from the primitives.
