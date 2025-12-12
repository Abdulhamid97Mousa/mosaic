# Cognitive Orchestration Layer: LLM/VLM-Guided Multi-Paradigm RL

## Related Documents

| Document | Description |
|----------|-------------|
| [TASK_1: Multi-Paradigm Orchestrator](../TASK_1/README.md) | Stepping paradigms (POSG, AEC, etc.) |
| [Paradigm Comparison](../TASK_1/01_paradigm_comparison.md) | POSG vs AEC vs EFG comparison |

> **Terminology Note:** "AEC" in TASK_1 refers to **Agent Environment Cycle** (PettingZoo stepping paradigm).
> "Agentic Episodic Control" is a separate paper about LLM integration - critiqued below.

---

## 1. Motivation

### 1.1 The Problem with Existing Approaches

The **Agentic Episodic Control (AEC)** paper [arXiv:2506.01442] proposes integrating LLMs with RL through:
- LLM-based semantic state encoding
- World-Graph working memory
- Critical state detection for exploration/exploitation

**However, this approach is fundamentally limited:**

| Limitation | Description |
|------------|-------------|
| **Single Role** | LLM only serves as state encoder, not as policy, planner, or reward |
| **Fixed Architecture** | Tightly coupled to BabyAI-Text benchmark |
| **No Paradigm Awareness** | Assumes single-agent sequential stepping |
| **No Multi-Agent Support** | Cannot coordinate multiple agents |
| **Task-Specific** | Cannot transfer to robotics, games, or embodied AI |

### 1.2 The Survey Taxonomy

The survey "The Evolving Landscape of LLM- and VLM-Integrated RL" [arXiv:2502.15214] identifies **three primary roles** for foundation models in RL:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM/VLM Roles in RL (Survey)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │ LLM/VLM as Agent │───┬── Parametric (fine-tuned)                 │
│  └──────────────────┘   └── Non-Parametric (frozen + RAG)           │
│                                                                      │
│  ┌────────────────────┐                                             │
│  │ LLM/VLM as Planner │───┬── Comprehensive (all sub-goals upfront) │
│  └────────────────────┘   └── Incremental (step-by-step)            │
│                                                                      │
│  ┌───────────────────┐                                              │
│  │ LLM/VLM as Reward │───┬── Reward Function (code generation)      │
│  └───────────────────┘   └── Reward Model (learned preferences)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The survey identifies these as *independent* roles. A truly general framework should support **all roles simultaneously** and add a **fourth role** that the survey doesn't address:

> **LLM/VLM as Meta-Controller**: Dynamic selection of stepping paradigm and agent coordination strategy.

---

## 2. MOSAIC's Cognitive Orchestration Layer

### 2.1 Conceptual Architecture

MOSAIC extends the survey taxonomy with a **Cognitive Orchestration Layer** that sits above the paradigm execution layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  COGNITIVE ORCHESTRATION LAYER                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Foundation Model Core                        │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐  │  │
│  │  │  Agent  │ │ Planner │ │ Reward  │ │   Meta-Controller   │  │  │
│  │  │  Role   │ │  Role   │ │  Role   │ │       (NEW)         │  │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────────┬──────────┘  │  │
│  │       │           │           │                  │             │  │
│  │       └───────────┴───────────┴──────────────────┘             │  │
│  │                           │                                     │  │
│  │                    Role Compositor                              │  │
│  │              (combines multiple FM roles)                       │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PARADIGM EXECUTION LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │ SINGLE_AGENT │  │  SEQUENTIAL  │  │ SIMULTANEOUS │  │HIERARCHIC││
│  │  (Gymnasium) │  │    (AEC)     │  │   (POSG)     │  │  (BDI)   ││
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘│
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      WORKER EXECUTION LAYER                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐ │
│  │ CleanRL │  │  RLlib  │  │PettingZoo│ │  Jason  │  │  Future   │ │
│  │ Worker  │  │ Worker  │  │ Worker  │  │ Worker  │  │  Workers  │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 The Four FM Roles in MOSAIC

| Role | Survey Status | MOSAIC Extension |
|------|---------------|------------------|
| **Agent** | Covered | + Multi-agent coordination |
| **Planner** | Covered | + Paradigm-aware planning |
| **Reward** | Covered | + Multi-agent reward shaping |
| **Meta-Controller** | **NOT in survey** | Paradigm selection, agent coordination |

### 2.3 Why Meta-Controller is Novel

The survey explicitly states:
> "LLM/VLM as Planner... generates sub-goals for complex tasks"

But it does NOT address:
- **When** should agents act simultaneously vs sequentially?
- **How** should multi-agent coordination be structured?
- **Which** paradigm is optimal for a given task/environment?

**MOSAIC's Meta-Controller fills this gap.**

---

## 3. Comparison: AEC vs MOSAIC

### 3.1 Architecture Comparison

| Aspect | AEC (2025) | MOSAIC Cognitive Layer |
|--------|------------|------------------------|
| **FM Role** | State encoder only | Agent + Planner + Reward + Meta-Controller |
| **Paradigm** | Implicit single-agent | Explicit multi-paradigm selection |
| **Multi-Agent** | Not supported | First-class citizen |
| **Memory** | Episodic + World-Graph | Pluggable memory modules |
| **Benchmarks** | BabyAI-Text only | Gymnasium, PettingZoo, RLlib, BDI |
| **Extensibility** | Fixed pipeline | Modular, configurable |

### 3.2 Why AEC's Approach is Limited

```python
# AEC's Approach (Simplified)
class AgenticEpisodicControl:
    """AEC uses LLM ONLY for state encoding."""

    def __init__(self):
        self.llm_encoder = LLMSemanticEncoder()      # LLM role: encoder
        self.world_graph = WorldGraphMemory()         # Fixed memory
        self.episodic_memory = EpisodicMemory()       # Fixed memory
        self.critical_detector = CriticalStateDetector()  # LLM role: classifier

    def step(self, state):
        # LLM generates embedding
        embedding = self.llm_encoder(state)

        # LLM detects if critical state
        if self.critical_detector(embedding):
            return self.episodic_memory.lookup(embedding)  # Exploit
        else:
            return self.world_graph.guided_action(state)   # Explore
```

**Problems:**
1. LLM cannot directly select actions (no Agent role)
2. LLM cannot generate sub-goals (no Planner role)
3. LLM cannot shape rewards (no Reward role)
4. No paradigm awareness - assumes single-agent sequential

### 3.3 MOSAIC's General Approach

```python
# MOSAIC's Cognitive Orchestration Layer
class CognitiveOrchestrator:
    """Supports ALL FM roles with paradigm awareness."""

    def __init__(self):
        self.foundation_model = FoundationModel()  # LLM or VLM
        self.role_config = RoleConfiguration()     # Which roles are active
        self.paradigm_selector = ParadigmSelector()
        self.memory_modules = MemoryRegistry()     # Pluggable

    def orchestrate(self, observation, context):
        # 1. Meta-Controller: Select paradigm (NEW - not in survey)
        paradigm = self.select_paradigm(observation, context)

        # 2. Planner role (if enabled)
        if self.role_config.planner_enabled:
            subgoals = self.foundation_model.generate_plan(observation)

        # 3. Agent role (if enabled)
        if self.role_config.agent_enabled:
            actions = self.foundation_model.select_actions(
                observation, paradigm, agents=context.active_agents
            )

        # 4. Reward role (if enabled)
        if self.role_config.reward_enabled:
            shaped_reward = self.foundation_model.shape_reward(
                observation, actions, context
            )

        return OrchestratedResult(paradigm, actions, subgoals, shaped_reward)

    def select_paradigm(self, observation, context) -> SteppingParadigm:
        """Use FM reasoning to choose coordination strategy."""
        # This is the META-CONTROLLER role - novel contribution
        prompt = f"""
        Task: {context.task_description}
        Environment: {context.env_type}
        Agents: {context.agent_ids}

        Should agents act:
        1. SIMULTANEOUSLY (all at once, like team sports)
        2. SEQUENTIALLY (turn-based, like chess)

        Consider: coordination needs, information sharing, timing constraints.
        """
        return self.foundation_model.reason(prompt)
```

---

## 4. Technical Design

### 4.1 Foundation Model Role Enum

```python
class FoundationModelRole(StrEnum):
    """How the LLM/VLM integrates with RL.

    Based on survey taxonomy [arXiv:2502.15214] plus MOSAIC extension.
    """
    # Survey roles
    AGENT = "agent"              # FM selects actions directly
    PLANNER = "planner"          # FM generates sub-goals/plans
    REWARD = "reward"            # FM shapes reward signal

    # MOSAIC extension (NOT in survey)
    META_CONTROLLER = "meta"     # FM selects paradigm and coordination

    # Sub-roles (from survey)
    AGENT_PARAMETRIC = "agent_parametric"       # Fine-tuned FM
    AGENT_NON_PARAMETRIC = "agent_frozen"       # Frozen FM + RAG
    PLANNER_COMPREHENSIVE = "planner_full"      # All sub-goals upfront
    PLANNER_INCREMENTAL = "planner_step"        # Step-by-step
    REWARD_FUNCTION = "reward_function"         # Code generation
    REWARD_MODEL = "reward_model"               # Learned preferences
```

### 4.2 Cognitive Orchestrator Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from gym_gui.core.enums import SteppingParadigm


@dataclass
class CognitiveContext:
    """Context for cognitive orchestration decisions."""

    task_description: str
    env_type: str
    agent_ids: List[str]
    observations: Dict[str, Any]
    history: List[Dict[str, Any]]
    episode_step: int
    total_reward: float


@dataclass
class OrchestratedDecision:
    """Result of cognitive orchestration."""

    paradigm: SteppingParadigm
    actions: Dict[str, Any]
    subgoals: Optional[List[str]] = None
    reward_shaping: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None  # Interpretability
    confidence: float = 1.0


class CognitiveOrchestrator(ABC):
    """Abstract base for cognitive orchestration.

    This layer sits above the ParadigmAdapter and provides:
    1. Dynamic paradigm selection (Meta-Controller role)
    2. Optional planning (Planner role)
    3. Optional direct action selection (Agent role)
    4. Optional reward shaping (Reward role)

    Unlike AEC which uses LLM only for encoding, MOSAIC supports
    all FM roles from the survey taxonomy plus paradigm selection.
    """

    @property
    @abstractmethod
    def active_roles(self) -> tuple[FoundationModelRole, ...]:
        """Which FM roles are currently active."""
        ...

    @abstractmethod
    def select_paradigm(
        self,
        context: CognitiveContext,
    ) -> SteppingParadigm:
        """Meta-Controller: Choose stepping paradigm based on task semantics.

        This is MOSAIC's novel contribution - not covered in the survey.

        Args:
            context: Current task and environment context.

        Returns:
            The optimal SteppingParadigm for this situation.
        """
        ...

    @abstractmethod
    def generate_plan(
        self,
        context: CognitiveContext,
        mode: str = "incremental",  # "comprehensive" or "incremental"
    ) -> List[str]:
        """Planner role: Generate sub-goals for the task.

        Args:
            context: Current context.
            mode: Planning mode (from survey taxonomy).

        Returns:
            List of sub-goal descriptions.
        """
        ...

    @abstractmethod
    def select_actions(
        self,
        context: CognitiveContext,
        paradigm: SteppingParadigm,
    ) -> Dict[str, Any]:
        """Agent role: Select actions for active agents.

        Args:
            context: Current context with observations.
            paradigm: How agents should coordinate.

        Returns:
            Dict mapping agent_id to action.
        """
        ...

    @abstractmethod
    def shape_reward(
        self,
        context: CognitiveContext,
        actions: Dict[str, Any],
        env_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """Reward role: Shape rewards using FM knowledge.

        Args:
            context: Current context.
            actions: Actions that were taken.
            env_rewards: Raw environment rewards.

        Returns:
            Shaped rewards (may differ from env_rewards).
        """
        ...

    def orchestrate(
        self,
        context: CognitiveContext,
    ) -> OrchestratedDecision:
        """Full orchestration pipeline using all active roles."""

        # 1. Meta-Controller (always active in MOSAIC)
        paradigm = self.select_paradigm(context)

        # 2. Planner (optional)
        subgoals = None
        if FoundationModelRole.PLANNER in self.active_roles:
            subgoals = self.generate_plan(context)

        # 3. Agent (optional - may defer to RL policy)
        actions = {}
        if FoundationModelRole.AGENT in self.active_roles:
            actions = self.select_actions(context, paradigm)

        return OrchestratedDecision(
            paradigm=paradigm,
            actions=actions,
            subgoals=subgoals,
        )
```

### 4.3 Integration with Phase 1

The Cognitive Orchestration Layer builds on Phase 1 abstractions:

```
Phase 1 (Completed)              Cognitive Layer (Phase 2)
═══════════════════              ═════════════════════════

SteppingParadigm ────────────────► CognitiveOrchestrator.select_paradigm()
     │                                      │
     ▼                                      ▼
ParadigmAdapter ◄────────────────── OrchestratedDecision.paradigm
     │                                      │
     ▼                                      ▼
PolicyController ◄─────────────────── FoundationModelRole.AGENT
     │                                      │
     ▼                                      ▼
WorkerCapabilities ◄───────────────── Runtime compatibility check
```

---

## 5. Novel Contributions

### 5.1 What MOSAIC Adds Beyond the Survey

| Contribution | Survey Status | MOSAIC Implementation |
|--------------|---------------|----------------------|
| Multi-paradigm support | Not addressed | `SteppingParadigm` enum + adapters |
| FM-driven paradigm selection | Not addressed | `CognitiveOrchestrator.select_paradigm()` |
| Unified role compositor | Not addressed | Single FM serving multiple roles |
| Multi-agent FM coordination | Briefly mentioned | First-class multi-agent support |
| Paradigm-aware planning | Not addressed | Plans include coordination strategy |

### 5.2 What MOSAIC Adds Beyond AEC

| Contribution | AEC Status | MOSAIC Implementation |
|--------------|------------|----------------------|
| FM as policy | Not supported | `FoundationModelRole.AGENT` |
| FM as planner | Not supported | `FoundationModelRole.PLANNER` |
| FM as reward | Not supported | `FoundationModelRole.REWARD` |
| Multi-paradigm | Fixed single-agent | Dynamic paradigm selection |
| Multi-environment | BabyAI-Text only | Gymnasium, PettingZoo, RLlib, BDI |
| Modular memory | Fixed episodic + graph | Pluggable memory modules |

---

## 6. Paper Positioning

### 6.1 Suggested Title Evolution

```
Original:
  "MOSAIC: Multi-Agent ..."

Options:
  1. "MOSAIC: Cognitively Orchestrated Multi-Agent Reinforcement Learning"
  2. "MOSAIC: Foundation Model-Guided Multi-Paradigm Multi-Agent Systems"
  3. "MOSAIC: Adaptive Multi-Agent Coordination via LLM/VLM Reasoning"
  4. "Beyond Episodic Control: Cognitive Orchestration for Multi-Paradigm MARL"
```

### 6.2 Key Claims

1. **First framework** to support dynamic paradigm selection via FM reasoning
2. **Unified architecture** that supports all FM roles (Agent, Planner, Reward, Meta-Controller)
3. **Generalizes AEC** by not limiting FM to state encoding
4. **Extends survey taxonomy** with Meta-Controller role for paradigm selection

### 6.3 Comparison Table for Paper

| Feature | Survey Taxonomy | AEC (2025) | MOSAIC |
|---------|-----------------|------------|--------|
| FM as Agent | ✓ | ✗ | ✓ |
| FM as Planner | ✓ | ✗ | ✓ |
| FM as Reward | ✓ | ✗ | ✓ |
| FM as Meta-Controller | ✗ | ✗ | ✓ |
| Multi-paradigm | ✗ | ✗ | ✓ |
| Multi-agent | Partial | ✗ | ✓ |
| Paradigm-aware | ✗ | ✗ | ✓ |

---

## 7. Implementation Roadmap

### 7.1 Phase 2: Cognitive Orchestration Layer

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | Define `FoundationModelRole` enum | 📋 Planned |
| 2.2 | Define `CognitiveContext` dataclass | 📋 Planned |
| 2.3 | Define `CognitiveOrchestrator` ABC | 📋 Planned |
| 2.4 | Implement `LLMOrchestrator` (OpenAI/Anthropic) | 📋 Planned |
| 2.5 | Implement `VLMOrchestrator` (CLIP/LLaVA) | 📋 Planned |
| 2.6 | Integrate with `ParadigmAdapter` | 📋 Planned |
| 2.7 | Add memory module interface | 📋 Planned |

### 7.2 Dependencies on Phase 1 (Completed)

- [x] `SteppingParadigm` enum
- [x] `WorkerCapabilities` dataclass
- [x] `PolicyController` protocol
- [x] `ParadigmAdapter` ABC
- [x] Adapter paradigm fields

---

## 8. References

### 8.1 Primary Sources

1. **Survey**: Schoepp et al. "The Evolving Landscape of LLM- and VLM-Integrated Reinforcement Learning" arXiv:2502.15214, Feb 2025
2. **AEC**: "Agentic Episodic Control" arXiv:2506.01442, Jun 2025

### 8.2 Key Papers from Survey

| Role | Paper | Key Contribution |
|------|-------|------------------|
| Agent (Parametric) | TWOSOME [Tan 2024] | Fine-tuned LLM policy |
| Agent (Non-Parametric) | Reflexion [Shinn 2023] | Self-reflective LLM agent |
| Planner (Comprehensive) | PSL [Dalal 2024] | LLM task decomposition |
| Planner (Incremental) | SayCan [Ichter 2022] | Grounded LLM planning |
| Reward (Function) | Eureka [Ma 2024] | LLM reward code generation |
| Reward (Model) | MineCLIP [Fan 2022] | VLM reward model |

### 8.3 MOSAIC's Position

MOSAIC's Cognitive Orchestration Layer:
- **Unifies** all roles from the survey taxonomy
- **Extends** with Meta-Controller for paradigm selection
- **Generalizes** beyond AEC's narrow encoding-only approach
- **Enables** true multi-paradigm, multi-agent foundation model integration
