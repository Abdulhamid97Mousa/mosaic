# LLM Worker Integration Plan (Ollama + LangChain)

**Date:** 2025-11-27
**Status:** Planning
**Goal:** Integrate local LLM capabilities via Ollama for intelligent agent decision-making in both single-agent (Gymnasium) and multi-agent (PettingZoo) environments

---

## 1. Overview

### What is Ollama?

[Ollama](https://github.com/ollama/ollama) is an open-source platform for running Large Language Models locally:
- **Zero API keys** - No cloud dependencies
- **Complete privacy** - All inference runs locally
- **Multiple models** - Llama, Mistral, Gemma, DeepSeek, Phi, etc.
- **REST API** - Easy integration via HTTP or Python SDK

### What is LangChain?

[LangChain](https://python.langchain.com/) is a framework for developing LLM-powered applications:
- **🔗 Chains** - Sequences of LLM calls
- **🤖 Agents** - LLM decides which actions to take
- **🧠 Memory** - Persisting state between calls
- **📚 Data Augmented Generation** - RAG capabilities
- **Ollama Integration** - Native support via `langchain-ollama`

### Use Cases in Mosaic/gym_gui

| Use Case | Description |
|----------|-------------|
| **Single-Agent LLM** | Use LLM to select actions in Gymnasium environments |
| **Multi-Agent LLM** | LLM agents in PettingZoo (Chess, Poker, Tic-Tac-Toe) |
| **Action Masking** | Respect valid action constraints from environments |
| **Game Strategy** | Natural language reasoning for turn-based games |
| **Hybrid Control** | LLM provides high-level strategy, RL handles low-level control |
| **Explanation** | Generate human-readable explanations of agent decisions |

### System Requirements

| RAM | Model Size |
|-----|------------|
| 8 GB | 7B parameter models (Mistral 7B, Llama 3.2 3B) |
| 16 GB | 13B parameter models |
| 32 GB | 33B+ parameter models |

---

## 2. Architecture

### 2.1 Overall Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Mosaic GUI                              │
├─────────────────────────────────────────────────────────────────┤
│  Human Control │ Single-Agent │ Multi-Agent │ LLM Agent (NEW)  │
│       Tab      │     Tab      │     Tab     │       Tab        │
└────────────────┴──────────────┴─────────────┴──────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Worker                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ OllamaClient│  │ LLMAgent    │  │ PromptTemplates         │ │
│  │ (API calls) │  │ (decision)  │  │ (game-specific prompts) │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Ollama Server                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Local Models: Llama 3.2, Mistral, Gemma, DeepSeek, etc │   │
│  └─────────────────────────────────────────────────────────┘   │
│  REST API: http://localhost:11434                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Worker Package Structure

Following the established pattern from `cleanrl_worker`:

```
3rd_party/llm_worker/
├── llm_worker/
│   ├── __init__.py           # Package exports
│   ├── client.py             # Ollama API client wrapper
│   ├── config.py             # Configuration dataclasses
│   ├── agent.py              # LLM-based agent for Gymnasium
│   ├── prompts/              # Prompt templates
│   │   ├── __init__.py
│   │   ├── base.py           # Base prompt class
│   │   ├── frozenlake.py     # FrozenLake-specific prompts
│   │   ├── cartpole.py       # CartPole-specific prompts
│   │   ├── minigrid.py       # MiniGrid-specific prompts
│   │   └── pettingzoo.py     # Multi-agent game prompts
│   ├── reasoning.py          # Chain-of-thought reasoning
│   ├── memory.py             # Conversation/episode memory
│   └── cli.py                # CLI entry point
├── pyproject.toml
└── README.md
```

---

## 3. Component Design

### 3.1 Ollama Client (`llm_worker/client.py`)

```python
"""Ollama API client wrapper."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator
import httpx
from ollama import Client, AsyncClient, ChatResponse


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout: float = 30.0


class OllamaClient:
    """Wrapper for Ollama API with sync and async support."""

    def __init__(self, config: OllamaConfig):
        self._config = config
        self._client = Client(host=config.host)
        self._async_client: Optional[AsyncClient] = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        response = self._client.list()
        return [m["name"] for m in response.get("models", [])]

    def pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry."""
        self._client.pull(model)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate text completion."""
        response = self._client.generate(
            model=self._config.model,
            prompt=prompt,
            system=system,
            options={
                "temperature": kwargs.get("temperature", self._config.temperature),
                "num_predict": kwargs.get("max_tokens", self._config.max_tokens),
            },
        )
        return response["response"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> ChatResponse:
        """Chat completion with message history."""
        return self._client.chat(
            model=self._config.model,
            messages=messages,
            options={
                "temperature": kwargs.get("temperature", self._config.temperature),
                "num_predict": kwargs.get("max_tokens", self._config.max_tokens),
            },
        )

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> AsyncIterator[str] | ChatResponse:
        """Async chat with optional streaming."""
        if self._async_client is None:
            self._async_client = AsyncClient(host=self._config.host)

        if stream:
            async for chunk in await self._async_client.chat(
                model=self._config.model,
                messages=messages,
                stream=True,
            ):
                yield chunk["message"]["content"]
        else:
            return await self._async_client.chat(
                model=self._config.model,
                messages=messages,
            )
```

### 3.2 LLM Agent (`llm_worker/agent.py`)

```python
"""LLM-based agent for Gymnasium environments."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import numpy as np

from .client import OllamaClient, OllamaConfig
from .prompts.base import PromptTemplate
from .memory import EpisodeMemory


@dataclass
class LLMAgentConfig:
    """Configuration for LLM agent."""
    ollama: OllamaConfig
    prompt_template: str = "default"
    use_chain_of_thought: bool = True
    memory_length: int = 10  # Number of past steps to include
    action_format: str = "json"  # "json" or "natural"


class LLMAgent:
    """Agent that uses LLM for action selection in Gymnasium environments."""

    def __init__(
        self,
        config: LLMAgentConfig,
        action_space: Any,
        observation_space: Any,
        env_name: str,
    ):
        self._config = config
        self._client = OllamaClient(config.ollama)
        self._action_space = action_space
        self._observation_space = observation_space
        self._env_name = env_name
        self._memory = EpisodeMemory(max_length=config.memory_length)
        self._prompt_template = self._load_prompt_template(env_name)

    def _load_prompt_template(self, env_name: str) -> PromptTemplate:
        """Load environment-specific prompt template."""
        # Dynamic import based on environment
        ...

    def select_action(
        self,
        observation: Any,
        info: Dict[str, Any],
    ) -> int:
        """Select action using LLM reasoning."""
        # Build prompt with observation, action space, and memory
        prompt = self._prompt_template.build(
            observation=observation,
            action_space=self._action_space,
            memory=self._memory.get_recent(),
            info=info,
        )

        # Get LLM response
        if self._config.use_chain_of_thought:
            response = self._client.generate(
                prompt=prompt,
                system=self._prompt_template.system_prompt,
            )
            action = self._parse_action_with_cot(response)
        else:
            response = self._client.generate(prompt=prompt)
            action = self._parse_action(response)

        return action

    def _parse_action(self, response: str) -> int:
        """Parse action from LLM response."""
        # Try JSON parsing first
        try:
            data = json.loads(response)
            return int(data.get("action", 0))
        except json.JSONDecodeError:
            pass

        # Fall back to regex/heuristic parsing
        ...

    def _parse_action_with_cot(self, response: str) -> int:
        """Parse action from chain-of-thought response."""
        # Extract action from reasoning chain
        ...

    def update_memory(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Update episode memory."""
        self._memory.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

    def reset(self) -> None:
        """Reset agent for new episode."""
        self._memory.clear()
```

### 3.3 Prompt Templates (`llm_worker/prompts/`)

#### Base Template (`llm_worker/prompts/base.py`)

```python
"""Base prompt template for LLM agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PromptTemplate(ABC):
    """Base class for environment-specific prompt templates."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role."""
        ...

    @abstractmethod
    def build(
        self,
        observation: Any,
        action_space: Any,
        memory: List[Dict],
        info: Dict[str, Any],
    ) -> str:
        """Build the full prompt for action selection."""
        ...

    @abstractmethod
    def format_observation(self, observation: Any) -> str:
        """Format observation for the prompt."""
        ...

    @abstractmethod
    def format_action_space(self, action_space: Any) -> str:
        """Format available actions for the prompt."""
        ...
```

#### FrozenLake Template (`llm_worker/prompts/frozenlake.py`)

```python
"""FrozenLake-specific prompt template."""

from .base import PromptTemplate


class FrozenLakePrompt(PromptTemplate):
    """Prompt template for FrozenLake environment."""

    @property
    def system_prompt(self) -> str:
        return """You are an AI agent playing FrozenLake, a grid-world navigation game.

GOAL: Navigate from Start (S) to Goal (G) without falling into Holes (H).
TILES: S=Start, F=Frozen(safe), H=Hole(death), G=Goal(win)

You must choose the best action to reach the goal safely.
Think step by step about which direction moves you closer to G while avoiding H.

Respond with a JSON object: {"reasoning": "your thought process", "action": <0-3>}
Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP"""

    def build(
        self,
        observation: Any,
        action_space: Any,
        memory: list,
        info: dict,
    ) -> str:
        grid = self._render_grid(observation, info)
        history = self._format_history(memory)

        return f"""Current grid state:
{grid}

Your position: marked as 'A'
Goal position: marked as 'G'

Recent history:
{history}

What action should you take? Think carefully about avoiding holes."""

    def _render_grid(self, observation: int, info: dict) -> str:
        """Render the grid with agent position."""
        # Convert observation to grid representation
        ...

    def _format_history(self, memory: list) -> str:
        """Format recent action history."""
        if not memory:
            return "No previous moves."
        lines = []
        for step in memory[-5:]:
            action_name = ["LEFT", "DOWN", "RIGHT", "UP"][step["action"]]
            lines.append(f"- Moved {action_name}, reward: {step['reward']}")
        return "\n".join(lines)
```

### 3.4 Configuration (`llm_worker/config.py`)

```python
"""Configuration for LLM worker."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 30.0


@dataclass
class AgentConfig:
    """LLM agent configuration."""
    use_chain_of_thought: bool = True
    memory_length: int = 10
    action_format: str = "json"
    retry_on_parse_error: bool = True
    max_retries: int = 3


@dataclass
class LLMWorkerConfig:
    """Complete LLM worker configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Environment overrides
    env_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_config(path: Optional[Path] = None) -> LLMWorkerConfig:
    """Load configuration from YAML file."""
    if path is None:
        return LLMWorkerConfig()

    with open(path) as f:
        data = yaml.safe_load(f)

    return LLMWorkerConfig(
        ollama=OllamaConfig(**data.get("ollama", {})),
        agent=AgentConfig(**data.get("agent", {})),
        env_configs=data.get("environments", {}),
    )
```

---

## 4. GUI Integration

### 4.1 LLM Agent Tab

Add a new tab to the control panel for LLM agent configuration:

```python
"""LLM Agent tab for control panel."""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal


class LLMAgentTab(QtWidgets.QWidget):
    """Tab widget for LLM agent configuration."""

    # Signals
    model_changed = pyqtSignal(str)
    start_agent_requested = pyqtSignal(dict)
    stop_agent_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Ollama Connection Group
        conn_group = QtWidgets.QGroupBox("Ollama Connection", self)
        conn_layout = QtWidgets.QFormLayout(conn_group)

        self._host_edit = QtWidgets.QLineEdit("http://localhost:11434")
        conn_layout.addRow("Host:", self._host_edit)

        self._status_label = QtWidgets.QLabel("Not connected")
        conn_layout.addRow("Status:", self._status_label)

        self._connect_button = QtWidgets.QPushButton("Connect")
        conn_layout.addRow(self._connect_button)

        layout.addWidget(conn_group)

        # Model Selection Group
        model_group = QtWidgets.QGroupBox("Model Selection", self)
        model_layout = QtWidgets.QFormLayout(model_group)

        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems([
            "llama3.2:3b",
            "llama3.2:1b",
            "mistral:7b",
            "gemma2:2b",
            "phi3:mini",
            "deepseek-r1:7b",
        ])
        model_layout.addRow("Model:", self._model_combo)

        self._pull_button = QtWidgets.QPushButton("Pull Model")
        model_layout.addRow(self._pull_button)

        layout.addWidget(model_group)

        # Agent Parameters Group
        params_group = QtWidgets.QGroupBox("Agent Parameters", self)
        params_layout = QtWidgets.QFormLayout(params_group)

        self._temperature_spin = QtWidgets.QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 2.0)
        self._temperature_spin.setValue(0.7)
        self._temperature_spin.setSingleStep(0.1)
        params_layout.addRow("Temperature:", self._temperature_spin)

        self._cot_checkbox = QtWidgets.QCheckBox("Chain-of-Thought")
        self._cot_checkbox.setChecked(True)
        params_layout.addRow(self._cot_checkbox)

        self._memory_spin = QtWidgets.QSpinBox()
        self._memory_spin.setRange(0, 50)
        self._memory_spin.setValue(10)
        params_layout.addRow("Memory Length:", self._memory_spin)

        layout.addWidget(params_group)

        # Control Buttons
        buttons = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Start LLM Agent")
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._stop_button.setEnabled(False)
        buttons.addWidget(self._start_button)
        buttons.addWidget(self._stop_button)
        layout.addLayout(buttons)

        # Reasoning Display
        reasoning_group = QtWidgets.QGroupBox("Agent Reasoning", self)
        reasoning_layout = QtWidgets.QVBoxLayout(reasoning_group)
        self._reasoning_text = QtWidgets.QTextEdit()
        self._reasoning_text.setReadOnly(True)
        self._reasoning_text.setMaximumHeight(150)
        reasoning_layout.addWidget(self._reasoning_text)
        layout.addWidget(reasoning_group)

        layout.addStretch()

    def update_reasoning(self, reasoning: str):
        """Display LLM reasoning in the text area."""
        self._reasoning_text.setText(reasoning)

    def set_connected(self, connected: bool, models: list = None):
        """Update connection status."""
        if connected:
            self._status_label.setText("Connected")
            self._status_label.setStyleSheet("color: green;")
            if models:
                self._model_combo.clear()
                self._model_combo.addItems(models)
        else:
            self._status_label.setText("Not connected")
            self._status_label.setStyleSheet("color: red;")
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure
1. Create `3rd_party/llm_worker/` directory structure
2. Implement `OllamaClient` with sync/async support
3. Create `requirements/llm_worker.txt`
4. Test basic connection and generation

### Phase 2: Agent Implementation
1. Implement `LLMAgent` class
2. Create base `PromptTemplate` class
3. Implement FrozenLake prompt template
4. Add episode memory management

### Phase 3: GUI Integration
1. Create `LLMAgentTab` widget
2. Add tab to control panel
3. Connect signals to main window
4. Display reasoning in real-time

### Phase 4: Environment Templates
1. Add CartPole prompt template
2. Add MiniGrid prompt template
3. Add PettingZoo (multi-agent) templates
4. Test across environments

### Phase 5: Advanced Features
1. Implement chain-of-thought reasoning
2. Add action explanation generation
3. Support for vision models (screenshots)
4. Hybrid LLM + RL control modes

---

## 6. Files to Create

### New Files

| File | Purpose |
|------|---------|
| `3rd_party/llm_worker/llm_worker/__init__.py` | Package exports |
| `3rd_party/llm_worker/llm_worker/client.py` | Ollama API client |
| `3rd_party/llm_worker/llm_worker/config.py` | Configuration dataclasses |
| `3rd_party/llm_worker/llm_worker/agent.py` | LLM agent implementation |
| `3rd_party/llm_worker/llm_worker/memory.py` | Episode memory |
| `3rd_party/llm_worker/llm_worker/prompts/__init__.py` | Prompt templates package |
| `3rd_party/llm_worker/llm_worker/prompts/base.py` | Base prompt template |
| `3rd_party/llm_worker/llm_worker/prompts/frozenlake.py` | FrozenLake prompts |
| `3rd_party/llm_worker/llm_worker/cli.py` | CLI entry point |
| `3rd_party/llm_worker/pyproject.toml` | Package definition |
| `requirements/llm_worker.txt` | Dependencies |
| `gym_gui/ui/widgets/llm_agent_tab.py` | GUI tab widget |

### Modified Files

| File | Change |
|------|--------|
| `gym_gui/ui/widgets/control_panel.py` | Add LLM Agent tab |
| `gym_gui/ui/workers.py` | Register LLM worker |

---

## 7. Dependencies

### Location: `requirements/llm_worker.txt`

```txt
# LLM Worker Dependencies (Ollama)
# Install with: pip install -r requirements/llm_worker.txt

# Include base requirements
-r base.txt

# Ollama Python SDK
ollama>=0.3.0

# Async HTTP client (used by ollama internally)
httpx>=0.27.0

# Configuration
pyyaml>=6.0

# Optional: LangChain integration
# langchain>=0.2.0
# langchain-community>=0.2.0
```

---

## 8. Ollama Installation

### Linux (Ubuntu 22.04)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.2:3b
ollama pull mistral:7b
```

### Docker

```bash
# Run Ollama in Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Pull model inside container
docker exec -it ollama ollama pull llama3.2:3b
```

### Verify Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello, world!",
  "stream": false
}'
```

---

## 9. Usage Examples

### Basic LLM Agent

```python
from llm_worker import OllamaClient, OllamaConfig, LLMAgent, LLMAgentConfig
import gymnasium as gym

# Configure Ollama
ollama_config = OllamaConfig(
    host="http://localhost:11434",
    model="llama3.2:3b",
    temperature=0.7,
)

# Create environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array")

# Create LLM agent
agent_config = LLMAgentConfig(
    ollama=ollama_config,
    use_chain_of_thought=True,
    memory_length=10,
)
agent = LLMAgent(
    config=agent_config,
    action_space=env.action_space,
    observation_space=env.observation_space,
    env_name="FrozenLake-v1",
)

# Run episode
observation, info = env.reset()
done = False

while not done:
    action = agent.select_action(observation, info)
    next_observation, reward, terminated, truncated, info = env.step(action)
    agent.update_memory(observation, action, reward, next_observation, terminated or truncated)
    observation = next_observation
    done = terminated or truncated

env.close()
```

### Async Streaming

```python
import asyncio
from llm_worker import OllamaClient, OllamaConfig

async def stream_reasoning():
    client = OllamaClient(OllamaConfig(model="llama3.2:3b"))

    messages = [
        {"role": "system", "content": "You are a game-playing AI."},
        {"role": "user", "content": "I'm at position (2,3) in a 4x4 grid. Goal is at (3,3). What should I do?"},
    ]

    async for chunk in client.chat_async(messages, stream=True):
        print(chunk, end="", flush=True)

asyncio.run(stream_reasoning())
```

---

## 10. Model Recommendations

| Model | Size | RAM | Use Case |
|-------|------|-----|----------|
| `llama3.2:1b` | 1.3GB | 4GB | Fast responses, simple games |
| `llama3.2:3b` | 2.0GB | 6GB | Good balance, recommended default |
| `mistral:7b` | 4.1GB | 8GB | Better reasoning, slower |
| `gemma2:2b` | 1.6GB | 4GB | Fast, good for simple tasks |
| `phi3:mini` | 2.3GB | 6GB | Microsoft's efficient model |
| `deepseek-r1:7b` | 4.7GB | 10GB | Strong reasoning capabilities |
| `llama3.2-vision:11b` | 7.9GB | 12GB | Can process screenshots |

---

## 11. Summary

The LLM Worker provides:

1. **Local LLM Integration** - No API keys, complete privacy via Ollama
2. **Intelligent Agents** - LLM-based decision making for Gymnasium environments
3. **Chain-of-Thought** - Visible reasoning process for transparency
4. **Environment Templates** - Game-specific prompts for optimal performance
5. **GUI Integration** - Tab in control panel for easy configuration

This enables a new paradigm of agent control where natural language reasoning drives action selection, complementing traditional RL approaches.

---

## 12. LangChain + PettingZoo Integration

The LangChain framework provides a powerful pattern for creating LLM agents that can play turn-based games in PettingZoo environments. This section documents the integration approach based on the official LangChain tutorial.

### 12.1 Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangChain Agent Hierarchy                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐                                       │
│  │   GymnasiumAgent     │  Base agent for single-agent envs     │
│  │   ────────────────── │                                       │
│  │   • Message history  │                                       │
│  │   • Retry logic      │                                       │
│  │   • Action parsing   │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│  ┌──────────▼───────────┐                                       │
│  │  PettingZooAgent     │  Multi-agent with env name tracking   │
│  │  ──────────────────  │                                       │
│  │  • Agent name prefix │                                       │
│  │  • Env description   │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│  ┌──────────▼───────────┐                                       │
│  │  ActionMaskAgent     │  Respects valid action constraints    │
│  │  ──────────────────  │                                       │
│  │  • Action masking    │                                       │
│  │  • Valid action list │                                       │
│  └──────────────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 Core Agent Classes

#### GymnasiumAgent (Base Class)

```python
"""Base LLM agent for Gymnasium environments using LangChain."""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from gymnasium import Env


class GymnasiumAgent:
    """LLM agent for single-agent Gymnasium environments."""

    def __init__(
        self,
        model: ChatOllama,
        env: Env,
    ):
        self.model = model
        self.env = env
        self.message_history: list = []
        self.max_retries = 5  # Retry on invalid action

    def get_system_message(self) -> str:
        """Build system message describing the environment."""
        return f"""You are playing a game. Here are the rules:

Environment: {self.env.unwrapped.__class__.__name__}

Your goal is to choose optimal actions to win the game.

IMPORTANT:
- Respond with ONLY a single integer representing your action
- Do not include any explanation or text, just the number
- Valid actions are integers from 0 to {self.env.action_space.n - 1}
"""

    def get_observation_message(self, observation) -> str:
        """Format observation as human message."""
        return f"Current observation:\n{observation}\n\nChoose your action:"

    def get_action(self, observation, reward: float, terminated: bool) -> int:
        """Select action using LLM reasoning."""
        # Build messages
        if not self.message_history:
            self.message_history.append(
                SystemMessage(content=self.get_system_message())
            )

        # Add observation
        self.message_history.append(
            HumanMessage(content=self.get_observation_message(observation))
        )

        # Get LLM response with retry logic
        for attempt in range(self.max_retries):
            response = self.model.invoke(self.message_history)
            self.message_history.append(response)

            try:
                action = int(response.content.strip())
                if 0 <= action < self.env.action_space.n:
                    return action
            except ValueError:
                pass

            # Invalid action - ask LLM to try again
            retry_msg = f"Invalid action '{response.content}'. Please respond with only an integer from 0 to {self.env.action_space.n - 1}."
            self.message_history.append(HumanMessage(content=retry_msg))

        # Default to random action after max retries
        return self.env.action_space.sample()

    def reset(self):
        """Reset agent state for new episode."""
        self.message_history = []
```

#### PettingZooAgent (Multi-Agent Extension)

```python
"""LLM agent for PettingZoo multi-agent environments."""

from pettingzoo import AECEnv


class PettingZooAgent(GymnasiumAgent):
    """LLM agent for PettingZoo AEC (turn-based) environments."""

    def __init__(
        self,
        model: ChatOllama,
        env: AECEnv,
        name: str,  # Agent name in the environment
    ):
        super().__init__(model, env)
        self.name = name

    def get_system_message(self) -> str:
        """Build system message with agent name and env description."""
        env_name = self.env.unwrapped.__class__.__name__

        return f"""You are playing {env_name} as player '{self.name}'.

Game Description:
{self.env.unwrapped.__doc__ or 'A multi-agent game.'}

Rules:
- You are player '{self.name}'
- Respond with ONLY a single integer representing your action
- Do not include any explanation, just the number
- Valid actions are integers from 0 to {self.env.action_space(self.name).n - 1}

Play strategically to win!
"""

    def get_observation_message(self, observation) -> str:
        """Format observation with agent context."""
        return f"""It is your turn ({self.name}).

Observation:
{observation}

Choose your action (integer only):"""
```

#### ActionMaskAgent (Action Constraint Support)

```python
"""LLM agent that respects action masks from environments."""

import numpy as np


class ActionMaskAgent(PettingZooAgent):
    """LLM agent that handles action masking for valid moves only."""

    def get_action(
        self,
        observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> int:
        """Select action respecting action mask."""
        # Extract action mask from info
        action_mask = info.get("action_mask")
        if action_mask is None:
            # No mask - use parent implementation
            return super().get_action(observation, reward, terminated)

        # Get valid actions
        valid_actions = np.where(action_mask == 1)[0].tolist()

        if not valid_actions:
            # No valid actions (should not happen)
            return 0

        # Build messages with action constraint
        if not self.message_history:
            self.message_history.append(
                SystemMessage(content=self.get_system_message())
            )

        # Add observation with valid actions
        obs_msg = f"""{self.get_observation_message(observation)}

Valid actions for this turn: {valid_actions}
You MUST choose from these valid actions only."""

        self.message_history.append(HumanMessage(content=obs_msg))

        # Get LLM response with retry logic
        for attempt in range(self.max_retries):
            response = self.model.invoke(self.message_history)
            self.message_history.append(response)

            try:
                action = int(response.content.strip())
                if action in valid_actions:
                    return action
            except ValueError:
                pass

            # Invalid action - remind of valid options
            retry_msg = f"Invalid action '{response.content}'. You must choose from: {valid_actions}"
            self.message_history.append(HumanMessage(content=retry_msg))

        # Default to first valid action after max retries
        return valid_actions[0]
```

### 12.3 Game Loop Implementation

```python
"""Game loop for LLM agents in PettingZoo environments."""

from langchain_ollama import ChatOllama
from pettingzoo.classic import rps_v2, tictactoe_v3, texas_holdem_v4


def run_game(env_fn, num_games: int = 1, render: bool = False):
    """Run games with LLM agents."""

    # Create Ollama-backed LLM (local, no API key needed)
    llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    for game_num in range(num_games):
        env = env_fn()

        # Create agent for each player
        agents = {
            name: ActionMaskAgent(model=llm, env=env, name=name)
            for name in env.possible_agents
        }

        env.reset()

        # Game loop (AEC API - agents take turns)
        for agent_name in env.agent_iter():
            observation, reward, terminated, truncated, info = env.last()

            if terminated or truncated:
                action = None  # Dead agent passes
            else:
                agent = agents[agent_name]
                action = agent.get_action(
                    observation, reward, terminated, truncated, info
                )

            env.step(action)

            if render:
                env.render()

        # Get final rewards
        final_rewards = {
            name: env.rewards.get(name, 0)
            for name in env.possible_agents
        }

        print(f"Game {game_num + 1} - Final rewards: {final_rewards}")

        # Reset all agents
        for agent in agents.values():
            agent.reset()

        env.close()


# Example usage
if __name__ == "__main__":
    # Rock-Paper-Scissors
    run_game(rps_v2.env, num_games=3)

    # Tic-Tac-Toe
    run_game(tictactoe_v3.env, num_games=2)

    # Texas Hold'em
    run_game(texas_holdem_v4.env, num_games=1)
```

### 12.4 Environment-Specific Prompts

#### Rock-Paper-Scissors

```python
class RPSAgent(ActionMaskAgent):
    """Specialized agent for Rock-Paper-Scissors."""

    def get_system_message(self) -> str:
        return """You are playing Rock-Paper-Scissors.

Actions:
- 0 = Rock (beats Scissors, loses to Paper)
- 1 = Paper (beats Rock, loses to Scissors)
- 2 = Scissors (beats Paper, loses to Rock)

Strategy tips:
- Try to predict your opponent's pattern
- Mix your choices to be unpredictable
- Remember: Rock beats Scissors, Scissors beats Paper, Paper beats Rock

Respond with ONLY the action number (0, 1, or 2)."""
```

#### Tic-Tac-Toe

```python
class TicTacToeAgent(ActionMaskAgent):
    """Specialized agent for Tic-Tac-Toe."""

    def get_system_message(self) -> str:
        return """You are playing Tic-Tac-Toe.

Board positions (0-8):
 0 | 1 | 2
-----------
 3 | 4 | 5
-----------
 6 | 7 | 8

Strategy:
- Take the center (4) if available
- Take corners (0, 2, 6, 8) for flexibility
- Block opponent's winning moves
- Look for winning opportunities (3 in a row)

Respond with ONLY the position number (0-8)."""
```

#### Texas Hold'em

```python
class TexasHoldemAgent(ActionMaskAgent):
    """Specialized agent for Texas Hold'em Poker."""

    def get_system_message(self) -> str:
        return """You are playing Texas Hold'em Poker.

Actions:
- 0 = Fold (give up the hand)
- 1 = Check/Call (match the current bet)
- 2 = Raise (increase the bet)
- 3 = All-in (bet everything)

Hand Rankings (high to low):
Royal Flush > Straight Flush > Four of a Kind > Full House >
Flush > Straight > Three of a Kind > Two Pair > One Pair > High Card

Strategy:
- Fold weak hands early
- Raise with strong hands
- Consider pot odds and opponent behavior
- Bluff occasionally to stay unpredictable

Respond with ONLY the action number from valid actions."""
```

### 12.5 Updated Package Structure

```
3rd_party/llm_worker/
├── llm_worker/
│   ├── __init__.py
│   ├── client.py              # Ollama API client
│   ├── config.py              # Configuration
│   ├── agents/                # LangChain agents (NEW)
│   │   ├── __init__.py
│   │   ├── gymnasium.py       # GymnasiumAgent
│   │   ├── pettingzoo.py      # PettingZooAgent
│   │   └── action_mask.py     # ActionMaskAgent
│   ├── games/                 # Game-specific agents (NEW)
│   │   ├── __init__.py
│   │   ├── rps.py            # Rock-Paper-Scissors
│   │   ├── tictactoe.py      # Tic-Tac-Toe
│   │   ├── chess.py          # Chess
│   │   └── poker.py          # Texas Hold'em
│   ├── prompts/               # Prompt templates
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ...
│   ├── memory.py              # Episode memory
│   └── cli.py                 # CLI entry point
├── pyproject.toml
└── README.md
```

### 12.6 Updated Dependencies

Update `requirements/llm_worker.txt`:

```txt
# LLM Worker Dependencies (Ollama + LangChain)
# Install with: pip install -r requirements/llm_worker.txt

# Include base requirements
-r base.txt

# Ollama Python SDK
ollama>=0.3.0

# Async HTTP client
httpx>=0.27.0

# Configuration
pyyaml>=6.0

# JSON schema validation
jsonschema>=4.20.0

# ═══════════════════════════════════════════════════════════════
# LangChain Integration (for advanced agent workflows)
# ═══════════════════════════════════════════════════════════════
langchain>=0.3.0
langchain-core>=0.3.0
langchain-ollama>=0.2.0

# PettingZoo integration
pettingzoo>=1.24.0
```

### 12.7 Using with Ollama (vs OpenAI)

The key advantage of this approach is using **Ollama locally** instead of OpenAI's API:

| Feature | OpenAI | Ollama |
|---------|--------|--------|
| **API Key** | Required | Not needed |
| **Privacy** | Data sent to cloud | 100% local |
| **Cost** | Pay per token | Free |
| **Latency** | Network dependent | Local, fast |
| **Models** | GPT-4, GPT-3.5 | Llama, Mistral, Gemma, etc. |

```python
# OpenAI (original tutorial)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# Ollama (our approach - no API key!)
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

### 12.8 Integration with Mosaic GUI

The LangChain agents integrate into the existing Mosaic architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Mosaic GUI                                 │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Agent Tab (PettingZoo)                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Environment: [Tic-Tac-Toe ▼]                            │   │
│  │ Player 1: [LLM Agent ▼]  Model: [llama3.2:3b ▼]        │   │
│  │ Player 2: [LLM Agent ▼]  Model: [mistral:7b ▼]         │   │
│  │ [Start Game] [Reset]                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  Agent Reasoning Panel                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Player 1 (llama3.2): "Taking center position 4 for     │   │
│  │ optimal control. Center gives most winning paths."     │   │
│  │                                                         │   │
│  │ Player 2 (mistral): "Blocking corner 0 to prevent      │   │
│  │ opponent's diagonal win. Will aim for position 2."     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Ollama Model Library](https://ollama.com/library)
- [LangChain + Ollama](https://python.langchain.com/docs/integrations/llms/ollama)
- [LangChain + PettingZoo Tutorial](https://python.langchain.com/docs/integrations/providers/pettingzoo/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [AgentGym-RL](https://agentgym-rl.github.io/) - Training LLM agents for decision making
- [LMRL-Gym](https://lmrl-gym.github.io/) - Benchmarks for multi-turn RL with LLMs
