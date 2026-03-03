# MOSAIC VLM Worker

**MOSAIC's Vision-Language Model worker with image observation support for multi-agent environments.**

This is a VLM-focused fork of the [LLM Worker](../llm_worker/), designed for multimodal agents that process both text and image observations. It is inspired by [BALROG](https://github.com/balrog-ai/BALROG) and extends it with vision capabilities.

## Key Features

- **Vision-Enabled by Default**: `max_image_history=1` — agents receive image observations alongside text
- **Multi-Agent Native**: Designed for PettingZoo and MultiGrid multi-agent environments
- **3 Coordination Levels**: Research on how explicit guidance affects VLM coordination
- **Theory of Mind**: Observation modes to study social reasoning in VLMs
- **Multiple VLM Providers**: OpenRouter, OpenAI (GPT-4V), Anthropic (Claude 3), vLLM support
- **GUI Integration**: Interactive mode for MOSAIC GUI

## Installation

```bash
# Install from MOSAIC root
pip install -e 3rd_party/mosaic/vlm_worker

# Or with dependencies
pip install -e "3rd_party/mosaic/vlm_worker[all]"
```

## Quick Start

```bash
# Set API key
export OPENROUTER_API_KEY=sk-or-v1-...

# Run on MultiGrid Soccer (2v2) with vision
vlm-worker --run-id test123 --env multigrid --task MultiGrid-Soccer-v0 \
  --model anthropic/claude-3.5-sonnet \
  --num-episodes 1
```

## Text-Only Fallback

Set `--max-image-history 0` to disable image observations:

```bash
vlm-worker --run-id test123 --env babyai --task BabyAI-GoToRedBall-v0 \
  --max-image-history 0
```

## License

MIT License - MOSAIC Team
