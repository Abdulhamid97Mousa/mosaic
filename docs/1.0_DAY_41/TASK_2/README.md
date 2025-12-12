# DAY 41 - TASK 2: Cognitive Orchestration Layer (LLM/VLM Integration)

## Problem Statement

Reinforcement learning agents lack the semantic reasoning and world knowledge that LLMs/VLMs provide. Current approaches (like Agentic Episodic Control) are **too narrow**, limiting the foundation model to a single role (state encoding).

MOSAIC's Cognitive Orchestration Layer provides a **general framework** that supports:
- **All FM roles** from the literature (Agent, Planner, Reward)
- **Plus a new role**: Meta-Controller for paradigm selection

> **Note:** This is separate from TASK_1 (Multi-Paradigm Orchestrator) which focuses on stepping paradigms (POSG, AEC, etc.). The Cognitive Layer will integrate with TASK_1's abstractions.

## Terminology Clarification

| Abbreviation | Full Name | Context |
|--------------|-----------|---------|
| **AEC** (Paradigm) | Agent Environment Cycle | PettingZoo's sequential stepping API |
| **AEC** (Paper) | Agentic Episodic Control | LLM integration paper [arXiv:2506.01442] |

To avoid confusion, we use:
- "Sequential" or "AEC paradigm" for PettingZoo
- "Agentic Episodic Control paper" for the LLM paper

## Documents

| # | Document | Description | Status |
|---|----------|-------------|--------|
| 00 | [Cognitive Orchestration Layer](./00_cognitive_orchestration_layer.md) | LLM/VLM-guided multi-paradigm RL | 📋 Planning |

## Key Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│                  COGNITIVE ORCHESTRATION LAYER                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Foundation Model Core                        │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────────┐  │  │
│  │  │  Agent  │ │ Planner │ │ Reward  │ │   Meta-Controller   │  │  │
│  │  │  Role   │ │  Role   │ │  Role   │ │       (NEW)         │  │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│              TASK_1: PARADIGM EXECUTION LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ SINGLE_AGENT │  │  SEQUENTIAL  │  │ SIMULTANEOUS │  │HIERARCHIC│ │
│  │  (Gymnasium) │  │    (AEC)     │  │   (POSG)     │  │  (BDI)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

## FM Roles (Survey Taxonomy + MOSAIC Extension)

| Role | Survey | MOSAIC | Description |
|------|--------|--------|-------------|
| Agent | ✓ | ✓ | FM selects actions directly |
| Planner | ✓ | ✓ | FM generates sub-goals/plans |
| Reward | ✓ | ✓ | FM shapes reward signal |
| **Meta-Controller** | ✗ | ✓ | FM selects paradigm and coordination (NEW) |

## Dependencies

### From TASK_1 (Completed)

- `SteppingParadigm` enum
- `ParadigmAdapter` ABC
- `PolicyController` protocol
- `WorkerCapabilities` dataclass

## Progress Tracking

- [ ] Define `FoundationModelRole` enum
- [ ] Define `CognitiveContext` dataclass
- [ ] Define `CognitiveOrchestrator` ABC
- [ ] Implement `LLMOrchestrator` (OpenAI/Anthropic)
- [ ] Implement `VLMOrchestrator` (CLIP/LLaVA)
- [ ] Integrate with TASK_1's `ParadigmAdapter`

## References

1. **Survey**: "The Evolving Landscape of LLM- and VLM-Integrated RL" [arXiv:2502.15214]
2. **Agentic Episodic Control Paper**: [arXiv:2506.01442] - Critique in doc 00

## Related Documents

- [TASK_1: Multi-Paradigm Orchestrator](../TASK_1/README.md)
- [DAY 40: Publication Roadmap](../../1.0_DAY_40/TASK_1/)
