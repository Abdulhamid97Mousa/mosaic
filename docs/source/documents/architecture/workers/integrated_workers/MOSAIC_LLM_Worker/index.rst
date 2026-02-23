MOSAIC LLM Worker
=================

The MOSAIC LLM Worker is MOSAIC's **native multi-agent LLM coordination and
adversarial evaluation** worker. It drives pre-trained language models through
MOSAIC environments with built-in support for **LLM coordination** (cooperative
teammates), **LLM adversarial** (competitive opponents), **Theory of Mind
observations**, and **three coordination-level prompt strategies**.

The MOSAIC LLM Worker grew out of the :doc:`BALROG Worker <../BALROG_Worker/index>`
integration.  After wrapping BALROG with a
:doc:`shim <../../concept>` for single-agent LLM evaluation, the potential of
multi-agent setups became clear --- both
:doc:`homogeneous </documents/architecture/operators/homogenous_decision_makers/index>`
(all-LLM teams) and
:doc:`hybrid </documents/architecture/operators/hybrid_decision_maker/index>`
(LLM + RL + Human) configurations.  This motivated extending the BALROG
foundation into a purpose-built worker for multi-agent LLM research, supporting
cooperative coordination, adversarial matchups, and cross-paradigm evaluation
within the MOSAIC :doc:`Operator <../../concept>` framework.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Multi-agent LLM coordination and adversarial (also single-agent)
   * - **Task Type**
     - LLM coordination (cooperative teams), LLM adversarial (competitive
       opponents), heterogeneous hybrid teams (LLM + RL + Human)
   * - **Model Support**
     - OpenRouter (unified), OpenAI, Anthropic, Google Gemini, vLLM (local)
   * - **Environments**
     - MultiGrid (Soccer 1v1/2v2, Collect), BabyAI, MiniGrid, MiniHack,
       Crafter, TextWorld, BabaIsAI, MeltingPot, PettingZoo
   * - **Execution**
     - Subprocess (autonomous or interactive step-by-step)
   * - **GPU required**
     - No (API-based) / Optional (vLLM local inference)
   * - **Source**
     - ``3rd_party/mosaic/llm_worker/llm_worker/``
   * - **Entry point**
     - ``llm-worker`` (CLI)

Overview
--------

The MOSAIC LLM Worker bridges pre-trained language models and MOSAIC's
multi-agent environments. It converts raw grid observations into natural
language, feeds them to an LLM, and parses the LLM's text response back into
discrete actions.

This enables two complementary research directions:

- **LLM Coordination:** Can LLMs cooperate as teammates? Do they develop
  emergent strategies? Does Theory of Mind information improve team play?
- **LLM Adversarial:** How do different LLM models perform head-to-head?
  Can an LLM team compete against RL-trained policies?

Combined with MOSAIC's :doc:`hybrid decision-maker
</documents/architecture/operators/hybrid_decision_maker/index>`, the worker
enables heterogeneous experiments (e.g., RL + LLM teammates vs RL + Random
opponents) that were previously impossible in any single framework.

Key features:

- **LLM coordination:** cooperative multi-agent teams with shared objectives
- **LLM adversarial:** head-to-head matchups between different LLM models or paradigms
- **Homogeneous and hybrid setups:** all-LLM teams or mixed LLM + RL + Human teams
- **Theory of Mind observations:** egocentric or teammate-aware text
- **3 coordination levels:** emergent, basic hints, role-based (Forward/Defender)
- **Pluggable API backends:** OpenRouter, OpenAI, Anthropic, Google Gemini, vLLM
- **6 agent strategies:** naive, chain-of-thought, robust variants, few-shot, dummy
- **Dual runtime modes:** autonomous (batch episodes) or interactive (GUI step-by-step)
- **Action-selector mode:** for PettingZoo games where GUI owns the environment
- **JSONL telemetry:** streamed to GUI and written to disk
Architecture
------------

The worker follows the standard MOSAIC :doc:`shim pattern <../../concept>` with
two runtime modes:

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Operator Config<br/>(per-player model)"]
           DAEMON["Operator Launcher"]
       end

       subgraph "LLM Worker Subprocess"
           CLI["cli.py<br/>(llm-worker)"]
           CFG["config.py<br/>(LLMWorkerConfig)"]
           RT["runtime.py<br/>(LLMWorkerRuntime /<br/>InteractiveLLMRuntime)"]
           OBS["observations.py<br/>(grid → text)"]
           PROMPT["prompts.py<br/>(3 coordination levels)"]
           CLIENT["client.py<br/>(OpenAI / Claude / Gemini)"]
       end

       subgraph "LLM API"
           API["OpenRouter / OpenAI<br/>Anthropic / Gemini / vLLM"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> OBS
       RT --> PROMPT
       RT --> CLIENT
       CLIENT -->|"chat.completions"| API

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style OBS fill:#dda0dd,stroke:#993399,color:#333
       style PROMPT fill:#dda0dd,stroke:#993399,color:#333
       style CLIENT fill:#ff7f50,stroke:#cc5500,color:#fff
       style API fill:#e8e8e8,stroke:#999

Observation Pipeline
--------------------

Raw grid observations are converted to natural language before being sent to
the LLM. The pipeline handles both single-agent and multi-agent environments:

.. code-block:: text

   3x3x3 numpy array  ──►  observations.py  ──►  Natural language  ──►  LLM
                            (type/color/state      "You see:
                             decoding)              - red ball 1 step ahead
                                                    - green goal 2 steps east
                                                    You are facing: EAST
                                                    You are carrying: nothing"

**Two observation modes** (for Theory of Mind research):

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Mode
     - Description
     - Research Purpose
   * - **Egocentric**
     - Agent sees only its own 3x3 local view
     - Decentralized control baseline
   * - **Visible Teammates**
     - Egocentric + teammate positions, directions, carrying status
     - Theory of Mind: can LLMs reason about teammate intentions?

Coordination Levels
-------------------

Three prompt strategies study how explicit guidance affects multi-agent
coordination:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Level
     - Strategy
     - Description
   * - **Level 1**
     - Emergent
     - Minimal guidance. Tests whether LLMs discover coordination naturally
       without hints.
   * - **Level 2**
     - Basic Hints
     - Adds cooperation tips ("spread out", "don't all chase the ball").
       Balances emergence with guidance.
   * - **Level 3**
     - Role-Based
     - Explicit Forward/Defender roles with detailed strategies.
       Tests whether role division improves team performance.


Supported Environments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Environment
     - Agents
     - Actions
     - Notes
   * - Soccer 1v1
     - 2
     - 8 (Legacy)
     - Team Green vs Team Red, first to 2 goals
   * - Soccer 2v2
     - 4
     - 8 (Legacy)
     - 16x11 FIFA grid, passing and stealing mechanics
   * - Collect 1v1 / 2v2
     - 2--4
     - 8 (Legacy)
     - Ball collection race
   * - BabyAI / MiniGrid
     - 1
     - 7
     - GoTo, Pickup, Open tasks with text descriptions
   * - MeltingPot
     - 2--16
     - varies
     - Social dilemmas, cooperation and competition substrates
   * - Crafter
     - 1
     - varies
     - Open-world survival via BALROG wrapper
   * - PettingZoo
     - 2+
     - varies
     - Chess, Connect Four, Go, Tic-Tac-Toe (action-selector mode)

**MultiGrid action space (Legacy --- Soccer, Collect):**

.. code-block:: text

   0: still     - do nothing (wait in place)
   1: left      - turn left 90 degrees
   2: right     - turn right 90 degrees
   3: forward   - move one step in facing direction
   4: pickup    - pick up object or steal from opponent
   5: drop      - drop held object (scores at goal, or pass to teammate)
   6: toggle    - interact with object in front
   7: done      - signal completion

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Backend
     - Models
     - Notes
   * - **OpenRouter** (default)
     - All major providers via unified API:
       GPT-4o, Claude 3.5, Gemini, Llama, Mistral, etc.
     - Requires ``OPENROUTER_API_KEY``.
       Free-tier models available (Nemotron 3 Nano, Arcee Trinity, etc.)
   * - **OpenAI**
     - GPT-4o, GPT-4-turbo, GPT-3.5-turbo
     - Requires ``OPENAI_API_KEY``
   * - **Anthropic**
     - Claude 3 Opus/Sonnet/Haiku
     - Requires ``ANTHROPIC_API_KEY``
   * - **Google Gemini**
     - Gemini 2.0 Flash, Gemini 1.5 Pro
     - Requires ``GOOGLE_API_KEY``
   * - **vLLM (local)**
     - Any HuggingFace-compatible model
     - Self-hosted, ``--base-url http://localhost:8000/v1``

Agent Strategies
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``naive``
     - Direct observation-to-action mapping. Minimal prompt.
   * - ``cot``
     - Chain-of-thought reasoning before action selection.
   * - ``robust_naive``
     - Naive with retry and fallback on parse failure.
   * - ``robust_cot``
     - Chain-of-thought with retry and fallback.
   * - ``few_shot``
     - In-context learning with example trajectories.
   * - ``dummy``
     - Random actions for baseline comparison.

Runtime Modes
-------------

**Autonomous mode** (batch episodes):

.. code-block:: bash

   llm-worker --run-id test123 \
       --env multigrid \
       --task MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0 \
       --client openrouter \
       --model nvidia/nemotron-3-nano-30b-a3b:free \
       --num-episodes 10 --max-steps 200

**Interactive mode** (GUI step-by-step):

.. code-block:: bash

   llm-worker --run-id test123 --interactive \
       --env multigrid \
       --task MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0

Interactive mode reads JSON commands from stdin and emits telemetry to stdout:

.. code-block:: json

   {"cmd": "reset", "seed": 42}
   {"cmd": "step"}
   {"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}
   {"cmd": "select_action", "observation": "...", "player_id": "player_0"}
   {"cmd": "stop"}

Multi-Agent Configurations
--------------------------

The MOSAIC LLM Worker supports the full range of
:doc:`operator configurations </documents/architecture/operators/concept>`:

**LLM Adversarial** --- different models compete head-to-head:

.. code-block:: text

   Soccer 1v1:
     Agent 0 (Green) ──► Nemotron 3 Nano (OpenRouter, free)
     Agent 1 (Red)   ──► Arcee Trinity Large (OpenRouter, free)

   Soccer 2v2:
     Agent 0 (Green) ──► Model A     Agent 2 (Red) ──► Model B
     Agent 1 (Green) ──► Model A     Agent 3 (Red) ──► Model B

**LLM Coordination** --- same or different models cooperate as teammates:

.. code-block:: text

   Soccer 2v2 (homogeneous):
     Agent 0 (Green) ──► GPT-4o      Agent 2 (Red) ──► GPT-4o
     Agent 1 (Green) ──► GPT-4o      Agent 3 (Red) ──► GPT-4o

**Hybrid (LLM + RL)** --- cross-paradigm teams (see
:doc:`hybrid decision-maker </documents/architecture/operators/hybrid_decision_maker/index>`):

.. code-block:: text

   Soccer 2v2:
     Agent 0 (Green) ──► RL (MAPPO)   Agent 2 (Red) ──► RL (MAPPO)
     Agent 1 (Green) ──► LLM (GPT-4o) Agent 3 (Red) ──► Random Baseline

Each agent runs in its own worker subprocess. The GUI collects actions
from all workers simultaneously and steps the environment in parallel mode.

Configuration
-------------

**JSON config** (launched by GUI or CLI):

.. code-block:: json

   {
     "run_id": "soccer_llm_vs_llm_001",
     "env_name": "multigrid",
     "task": "MosaicMultiGrid-Soccer-1vs1-IndAgObs-v0",
     "client_name": "openrouter",
     "model_id": "nvidia/nemotron-3-nano-30b-a3b:free",
     "agent_type": "cot",
     "num_episodes": 10,
     "max_steps": 200,
     "temperature": 0.7,
     "observation_mode": "visible_teammates",
     "coordination_level": 2,
     "role": "forward"
   }

**MultiGrid-specific config fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``agent_id``
     - ``0``
     - Agent index for multi-agent environments (0--3)
   * - ``observation_mode``
     - ``visible_teammates``
     - ``"egocentric"`` or ``"visible_teammates"`` (Theory of Mind)
   * - ``coordination_level``
     - ``1``
     - ``1`` = Emergent, ``2`` = Basic Hints, ``3`` = Role-Based
   * - ``role``
     - ``None``
     - Agent role for Level 3: ``"forward"`` or ``"defender"``


.. toctree::
   :maxdepth: 1

   installation
   common_errors
