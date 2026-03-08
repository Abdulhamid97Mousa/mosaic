# MOSAIC LLM Worker

**MOSAIC's native multi-agent LLM worker with coordination strategies and Theory of Mind.**

This is MOSAIC's contribution to LLM-based multi-agent coordination research. It is inspired by [BALROG](https://github.com/balrog-ai/BALROG) but designed from the ground up for multi-agent environments.

## Key Features

- **Multi-Agent Native**: Designed for PettingZoo and MultiGrid multi-agent environments
- **3 Coordination Levels**: Research on how explicit guidance affects LLM coordination
- **Theory of Mind**: Observation modes to study social reasoning in LLMs
- **Multiple LLM Providers**: OpenRouter, OpenAI, Anthropic, vLLM support
- **GUI Integration**: Interactive mode for MOSAIC GUI

## Installation

```bash
# Install from MOSAIC root
pip install -e 3rd_party/mosaic/llm_worker

# Or with dependencies
pip install -r requirements/mosaic_llm_worker.txt
pip install -e 3rd_party/mosaic/llm_worker
```

## Quick Start

```bash
# Set API key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run on MultiGrid Soccer (2v2)
mosaic-llm-worker --task MultiGrid-Soccer-v0 \
  --model anthropic/claude-3.5-sonnet \
  --coordination-level 2 \
  --num-episodes 1
```

## Coordination Levels

MOSAIC implements three coordination levels for research:

### Level 1: Emergent
Minimal guidance - let LLMs discover coordination naturally.

```bash
mosaic-llm-worker --task MultiGrid-Soccer-v0 --coordination-level 1
```

### Level 2: Basic Hints
Add cooperation tips without being prescriptive.

```bash
mosaic-llm-worker --task MultiGrid-Soccer-v0 --coordination-level 2
```

### Level 3: Role-Based
Explicit roles (Forward/Defender) with detailed strategies.

```bash
mosaic-llm-worker --task MultiGrid-Soccer-v0 --coordination-level 3
```

## Observation Modes

### Egocentric (Default)
Agent sees only its own partial view - decentralized control.

```bash
mosaic-llm-worker --observation-mode egocentric
```

### Visible Teammates (Theory of Mind)
Include visible teammate information to enable reasoning about others.

```bash
mosaic-llm-worker --observation-mode visible_teammates
```

## LLM Providers

### OpenRouter (Recommended)
Access to multiple models through unified API.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
mosaic-llm-worker --client openrouter --model anthropic/claude-3.5-sonnet
```

### Direct OpenAI
```bash
export OPENAI_API_KEY=sk-...
mosaic-llm-worker --client openai --model gpt-4o
```

### Direct Anthropic
```bash
export ANTHROPIC_API_KEY=sk-ant-...
mosaic-llm-worker --client anthropic --model claude-3-5-sonnet-20241022
```

### vLLM (Local GPU)
```bash
# Start vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Run worker
mosaic-llm-worker --client vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --api-base-url http://localhost:8000/v1
```

## Interactive Mode (GUI)

For MOSAIC GUI integration:

```bash
mosaic-llm-worker --interactive --task MultiGrid-Soccer-v0
```

Commands (JSON via stdin):
- `{"type": "reset"}` - Reset environment
- `{"type": "get_action", "agent_id": 0}` - Get LLM action for agent
- `{"type": "step", "actions": {"agent_0": 3, ...}}` - Step environment
- `{"type": "quit"}` - Exit

## Research Questions

This worker is designed to study:

1. **RQ1**: How do coordination levels affect multi-agent LLM performance?
2. **RQ2**: Can LLMs coordinate effectively without explicit guidance?
3. **RQ3**: Does Theory of Mind observation improve cooperation?
4. **RQ4**: Do role-based strategies improve team-based game outcomes?

## Configuration

Full configuration via JSON file:

```json
{
  "run_id": "experiment-001",
  "task": "MultiGrid-Soccer-v0",
  "num_agents": 4,
  "client_name": "openrouter",
  "model_id": "anthropic/claude-3.5-sonnet",
  "coordination_level": 2,
  "observation_mode": "visible_teammates",
  "num_episodes": 10,
  "max_steps_per_episode": 100
}
```

```bash
mosaic-llm-worker --config experiment.json
```

## Package Structure

```
mosaic_llm_worker/
├── __init__.py          # Worker metadata for MOSAIC discovery
├── cli.py               # Command-line interface
├── config.py            # Configuration dataclass
├── runtime.py           # Autonomous + Interactive runtimes
├── prompts/             # Coordination prompt strategies
│   ├── base.py          # Abstract prompt generator
│   └── multigrid.py     # MultiGrid-specific prompts
├── observations/        # Observation text generation
│   └── theory_of_mind.py
└── clients/             # LLM API clients
    ├── base.py          # Abstract client + factory
    ├── openrouter.py
    ├── openai_client.py
    └── anthropic_client.py
```

## License

MIT License - MOSAIC Team
