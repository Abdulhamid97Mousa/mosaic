# MOSAIC Native Workers Migration Plan

## Overview

Migrate workers to MOSAIC namespace with full rename:
- `human_worker` → `mosaic_human_worker`
- Create new `mosaic_llm_worker` (inspired by balrog_worker, but independent)

---

## Part 1: Migrate human_worker → mosaic_human_worker

### Current State
```
3rd_party/human_worker/           # Old location (to be deprecated)
3rd_party/mosaic/human_worker/    # New location (has copy, needs rename)
```

### Target State
```
3rd_party/mosaic/human_worker/
├── mosaic_human_worker/          # Renamed from human_worker/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   └── runtime.py
├── tests/
├── pyproject.toml                # Updated with new name
└── README.md
```

### Migration Steps

#### Step 1.1: Rename Package Directory
```bash
mv 3rd_party/mosaic/human_worker/human_worker/ 3rd_party/mosaic/human_worker/mosaic_human_worker/
```

#### Step 1.2: Update pyproject.toml (mosaic/human_worker/)
Change:
- `name = "human-worker"` → `name = "mosaic-human-worker"`
- `human-worker = "human_worker.cli:main"` → `mosaic-human-worker = "mosaic_human_worker.cli:main"`
- `human = "human_worker:get_worker_metadata"` → `human = "mosaic_human_worker:get_worker_metadata"`
- `include = ["human_worker*"]` → `include = ["mosaic_human_worker*"]`

#### Step 1.3: Update Internal Imports
In `mosaic_human_worker/*.py` files, change:
- `from human_worker.config import ...` → `from mosaic_human_worker.config import ...`
- `from human_worker.runtime import ...` → `from mosaic_human_worker.runtime import ...`

#### Step 1.4: Update Main pyproject.toml
In `/home/hamid/Desktop/Projects/GUI_BDI_RL/pyproject.toml`:
- `"3rd_party/human_worker"` → `"3rd_party/mosaic/human_worker"`
- `"human_worker*"` → `"mosaic_human_worker*"`
- `human = "human_worker:get_worker_metadata"` → `human = "mosaic_human_worker:get_worker_metadata"`

#### Step 1.5: Update egg-info Directory
```bash
rm -rf 3rd_party/mosaic/human_worker/human_worker.egg-info/
# Will be regenerated with new name on pip install -e .
```

---

## Part 2: Create mosaic_llm_worker

### Design Philosophy
- **Independent**: No BALROG dependency
- **Multi-agent native**: Designed for PettingZoo multi-agent environments
- **Inspired by**: balrog_worker concepts + mosaic_extension coordination strategies

### Package Structure
```
3rd_party/mosaic/llm_worker/
├── mosaic_llm_worker/
│   ├── __init__.py               # get_worker_metadata()
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # MosaicLLMWorkerConfig
│   ├── runtime.py                # Autonomous + Interactive runtimes
│   ├── prompts/                  # Coordination strategies
│   │   ├── __init__.py
│   │   ├── base.py               # BasePromptGenerator
│   │   └── multigrid.py          # MultiGrid-specific (from mosaic_extension)
│   ├── observations/             # Multi-agent observation handling
│   │   ├── __init__.py
│   │   └── theory_of_mind.py     # ToM observation modes
│   └── clients/                  # LLM API clients
│       ├── __init__.py
│       ├── base.py               # AbstractLLMClient
│       ├── openrouter.py
│       ├── openai_client.py
│       ├── anthropic_client.py
│       └── vllm_client.py
├── tests/
│   ├── __init__.py
│   └── test_mosaic_llm_worker.py
├── pyproject.toml
└── README.md
```

### Implementation Details

#### 2.1: `mosaic_llm_worker/__init__.py`
```python
def get_worker_metadata() -> tuple:
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="MOSAIC LLM Worker",
        version=__version__,
        description="Multi-agent LLM worker with coordination strategies",
        author="MOSAIC Team",
        homepage="https://github.com/MOSAIC-RL/GUI_BDI_RL",
        upstream_library=None,  # No upstream - MOSAIC native
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="llm",
        supported_paradigms=("multi_agent", "human_vs_ai", "ai_vs_ai"),
        env_families=("pettingzoo", "gymnasium", "multigrid"),
        action_spaces=("discrete",),
        observation_spaces=("structured", "text"),
        max_agents=8,
        supports_self_play=True,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=True,
        requires_gpu=False,  # API-based by default
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities
```

#### 2.2: `mosaic_llm_worker/config.py`
```python
@dataclass
class MosaicLLMWorkerConfig:
    # Protocol required
    run_id: str
    seed: int | None = None

    # Environment
    env_name: str = "MultiGrid-Soccer-v0"
    task: str = ""
    num_agents: int = 2

    # LLM settings
    client_name: str = "openrouter"  # openrouter, openai, anthropic, vllm
    model_id: str = "anthropic/claude-3.5-sonnet"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 30.0

    # Multi-agent coordination
    coordination_level: int = 1  # 1=emergent, 2=hints, 3=role-based
    observation_mode: str = "egocentric"  # egocentric, visible_teammates
    agent_roles: list[str] | None = None  # e.g., ["leader", "follower"]

    # Execution
    num_episodes: int = 1
    max_steps_per_episode: int = 100

    # Telemetry
    telemetry_dir: str = "./telemetry"
    emit_jsonl: bool = True
    emit_stdout: bool = True
```

#### 2.3: `mosaic_llm_worker/prompts/` (from mosaic_extension)
Port coordination strategies from `balrog_worker/mosaic_extension/multigrid/prompts.py`:
- **Level 1 (Emergent)**: Minimal guidance
- **Level 2 (Basic Hints)**: Cooperation hints
- **Level 3 (Role-Based)**: Explicit roles with strategies

#### 2.4: `mosaic_llm_worker/observations/` (from mosaic_extension)
Port observation generation from `balrog_worker/mosaic_extension/multigrid/observations.py`:
- Egocentric observation mode
- Visible teammates mode (Theory of Mind)

#### 2.5: `mosaic_llm_worker/clients/`
Implement LLM clients independently (no BALROG import):
- Simple API wrapper classes
- Retry logic with exponential backoff
- Response parsing

#### 2.6: `mosaic_llm_worker/runtime.py`
Two runtime modes:
- **AutonomousRuntime**: Runs N episodes autonomously
- **InteractiveRuntime**: JSON stdin/stdout for GUI integration

---

## Part 3: Update Main pyproject.toml

### Changes Required

```toml
# [tool.setuptools.packages.find] where:
# Remove:
"3rd_party/human_worker",
# Add:
"3rd_party/mosaic/human_worker",
"3rd_party/mosaic/llm_worker",

# [tool.setuptools.packages.find] include:
# Remove:
"human_worker*",
# Add:
"mosaic_human_worker*",
"mosaic_llm_worker*",

# [project.entry-points."mosaic.workers"]:
# Change:
human = "human_worker:get_worker_metadata"
# To:
human = "mosaic_human_worker:get_worker_metadata"
# Add:
llm = "mosaic_llm_worker:get_worker_metadata"
```

---

## Part 4: Requirements

### Create requirements/mosaic_llm_worker.txt
```
-r base.txt

# LLM API clients
openai>=1.0.0
anthropic>=0.18.0
httpx>=0.27.0

# Optional: Local inference
# vllm>=0.6.0

# Environment support
pettingzoo>=1.24.0
gymnasium>=0.29.0
```

---

## Implementation Order

1. **Phase 1: Migrate human_worker** (Steps 1.1-1.5)
   - Rename package directory
   - Update pyproject.toml
   - Update imports
   - Update main pyproject.toml

2. **Phase 2: Create mosaic_llm_worker structure**
   - Create directory structure
   - Create pyproject.toml
   - Create __init__.py with metadata

3. **Phase 3: Implement core modules**
   - config.py
   - Port prompts/ from mosaic_extension
   - Port observations/ from mosaic_extension

4. **Phase 4: Implement LLM clients**
   - base.py (abstract)
   - openrouter.py
   - openai_client.py
   - anthropic_client.py

5. **Phase 5: Implement runtime**
   - AutonomousRuntime
   - InteractiveRuntime
   - Telemetry emission

6. **Phase 6: CLI and integration**
   - cli.py
   - Update main pyproject.toml
   - Create requirements file

7. **Phase 7: Testing**
   - Unit tests
   - Integration tests

---

## Files Summary

### Modified Files
- `3rd_party/mosaic/human_worker/pyproject.toml`
- `3rd_party/mosaic/human_worker/mosaic_human_worker/__init__.py`
- `3rd_party/mosaic/human_worker/mosaic_human_worker/cli.py`
- `3rd_party/mosaic/human_worker/mosaic_human_worker/config.py`
- `3rd_party/mosaic/human_worker/mosaic_human_worker/runtime.py`
- `/home/hamid/Desktop/Projects/GUI_BDI_RL/pyproject.toml`

### New Files
- `3rd_party/mosaic/llm_worker/pyproject.toml`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/__init__.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/cli.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/config.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/runtime.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/prompts/__init__.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/prompts/base.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/prompts/multigrid.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/observations/__init__.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/observations/theory_of_mind.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/clients/__init__.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/clients/base.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/clients/openrouter.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/clients/openai_client.py`
- `3rd_party/mosaic/llm_worker/mosaic_llm_worker/clients/anthropic_client.py`
- `3rd_party/mosaic/llm_worker/tests/__init__.py`
- `3rd_party/mosaic/llm_worker/tests/test_mosaic_llm_worker.py`
- `3rd_party/mosaic/llm_worker/README.md`
- `requirements/mosaic_llm_worker.txt`

### Deleted/Deprecated
- `3rd_party/human_worker/` - Can be removed after migration verified
- `3rd_party/mosaic/human_worker/human_worker/` - Renamed to mosaic_human_worker/
