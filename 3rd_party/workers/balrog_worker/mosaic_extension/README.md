# MOSAIC Extensions for BALROG

This directory contains **MOSAIC's novel contributions** to extend BALROG for multi-agent environments.

## Purpose

BALROG (Benchmark for Large Language Models on Robotic Decision-making) supports single-agent text-based RL environments. MOSAIC extends this to multi-agent scenarios with:

1. **Multi-Agent Coordination Strategies** - Three levels of LLM coordination
2. **Theory of Mind Observations** - Two observation modes (egocentric vs visible teammates)
3. **Role-Based Multi-Agent Prompting** - Explicit role assignment for team-based games

## Structure

```
mosaic_extension/
├── multigrid/              # MultiGrid multi-agent extension
│   ├── prompts.py         # 3-level coordination prompt generation
│   ├── observations.py    # Text observation conversion (2 modes)
│   └── wrapper.py         # Environment wrapper integrating prompts + observations
└── README.md              # This file
```

## MultiGrid Extension

### Coordination Levels

**Level 1: Emergent (Minimal)**
- Let LLMs figure out coordination naturally
- No explicit cooperation strategies in prompt
- Tests emergent multi-agent behavior

**Level 2: Basic Hints**
- Add cooperation tips in system prompt
- Guide LLMs toward teamwork without being prescriptive
- Balance between emergence and guidance

**Level 3: Role-Based**
- Assign explicit roles (e.g., Forward/Defender in Soccer)
- Detailed role-specific strategies
- Maximum coordination guidance

### Observation Modes

**Egocentric Only**
- Agent sees only its own partial view
- Decentralized, realistic for embodied agents
- Based on: [arXiv:2402.01680v2](https://arxiv.org/html/2402.01680v2)

**Visible Teammates**
- Include visible teammates in observation text
- Enables Theory of Mind reasoning
- Better for cooperative game scenarios
- Based on: [OpenReview cfL8zApofK](https://openreview.net/forum?id=cfL8zApofK)

## Integration

These extensions integrate with BALROG through `balrog_worker/environments.py`, which wraps environments before passing them to BALROG agents.

## Citation

If you use MOSAIC's MultiGrid extensions, please cite:

```
@software{mosaic_multigrid_2025,
  title={MOSAIC MultiGrid Extension for BALROG},
  author={[Your Team]},
  year={2025},
  howpublished={\url{https://github.com/[your-repo]}}
}
```

## Research Questions Addressed

1. **RQ1**: How do different coordination levels affect LLM multi-agent performance?
2. **RQ2**: Does Theory of Mind observation (visible teammates) improve cooperation?
3. **RQ3**: Can role-based prompting enable effective team strategies in LLMs?

## Environments Supported

- **MultiGrid-Soccer-v0**: 4 agents (2v2), zero-sum team game
- **MultiGrid-Collect-v0**: 3 agents, competitive ball collection

## License

Same license as parent MOSAIC project.
