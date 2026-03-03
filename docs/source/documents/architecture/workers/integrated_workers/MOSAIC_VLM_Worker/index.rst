MOSAIC VLM Worker
=================

The MOSAIC VLM Worker is MOSAIC's **native Vision-Language Model worker** for
evaluating multimodal agents in RL environments.  It extends the
:doc:`LLM Worker <../MOSAIC_LLM_Worker/index>` with **image observation
support**, enabling VLM models to perceive raw RGB frames from environments
like Crafter, BabyAI, and MultiGrid alongside structured text descriptions.

The VLM Worker shares the same architecture, agent strategies, and coordination
levels as the LLM Worker.  The key difference is that it can include
**image history** in prompts, allowing vision-capable models (GPT-4o, Claude 3,
Gemini) to reason directly over visual observations rather than relying
solely on text descriptions generated from grid arrays.

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Multi-agent VLM coordination and adversarial (also single-agent)
   * - **Task Type**
     - VLM evaluation with image observations, multimodal reasoning,
       cooperative teams, adversarial opponents
   * - **Model Support**
     - OpenRouter (unified), OpenAI, Anthropic, Google Gemini, vLLM (local)
   * - **Environments**
     - MultiGrid (Soccer 1v1/2v2, Collect), BabyAI, MiniGrid, MiniHack,
       Crafter, TextWorld, BabaIsAI, PettingZoo
   * - **Execution**
     - Subprocess (autonomous or interactive step-by-step)
   * - **GPU required**
     - No (API-based) / Optional (vLLM local inference)
   * - **Source**
     - ``3rd_party/mosaic/vlm_worker/vlm_worker/``
   * - **Entry point**
     - ``vlm-worker`` (CLI)

Overview
--------

The VLM Worker converts environment frames into multimodal prompts that
combine text descriptions with RGB images.  This enables two research
directions beyond what the text-only LLM Worker provides:

- **Visual grounding:** Can VLMs identify objects, navigate mazes, or
  coordinate with teammates using pixel observations instead of symbolic
  text?
- **Multimodal vs text-only:** Does adding image context improve LLM
  performance in grid-world environments, or is text sufficient?

Key features:

- **Image observation support:** configurable image history depth
  (``max_image_history ≥ 1`` for VLM mode, ``0`` for text-only fallback)
- **Same agent strategies as LLM Worker:** naive, chain-of-thought, robust
  variants, few-shot, dummy
- **Same 3 coordination levels:** emergent, basic hints, role-based
- **Pluggable API backends:** OpenRouter, OpenAI, Anthropic, Google Gemini,
  vLLM
- **Dual runtime modes:** autonomous (batch episodes) or interactive (GUI
  step-by-step)
- **JSONL telemetry:** streamed to GUI and written to disk

Architecture
------------

The VLM Worker follows the same :doc:`shim pattern <../../concept>` as the
LLM Worker, with an additional image encoding step in the observation
pipeline:

.. mermaid::

   graph TB
       subgraph "MOSAIC GUI"
           FORM["Operator Config<br/>(per-player model)"]
           DAEMON["Operator Launcher"]
       end

       subgraph "VLM Worker Subprocess"
           CLI["cli.py<br/>(vlm-worker)"]
           CFG["config.py<br/>(VLMWorkerConfig)"]
           RT["runtime.py<br/>(VLMWorkerRuntime /<br/>InteractiveVLMRuntime)"]
           OBS["observations.py<br/>(grid → text + image)"]
           PROMPT["prompts.py<br/>(3 coordination levels)"]
           CLIENT["client.py<br/>(OpenAI / Claude / Gemini)"]
       end

       subgraph "VLM API"
           API["OpenRouter / OpenAI<br/>Anthropic / Gemini / vLLM"]
       end

       FORM -->|"config JSON"| DAEMON
       DAEMON -->|"spawn"| CLI
       CLI --> CFG --> RT
       RT --> OBS
       RT --> PROMPT
       RT --> CLIENT
       CLIENT -->|"chat.completions<br/>(text + images)"| API

       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style CLI fill:#9370db,stroke:#6a0dad,color:#fff
       style CFG fill:#9370db,stroke:#6a0dad,color:#fff
       style RT fill:#9370db,stroke:#6a0dad,color:#fff
       style OBS fill:#dda0dd,stroke:#993399,color:#333
       style PROMPT fill:#dda0dd,stroke:#993399,color:#333
       style CLIENT fill:#9370db,stroke:#6a0dad,color:#fff
       style API fill:#e8e8e8,stroke:#999

VLM vs LLM Worker
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 38 37

   * - Aspect
     - VLM Worker
     - LLM Worker
   * - **Observations**
     - Text + RGB images (multimodal)
     - Text only
   * - **Image history**
     - ``max_image_history ≥ 1``
     - N/A
   * - **Use case**
     - Visual grounding, multimodal reasoning
     - Text-based reasoning, Theory of Mind
   * - **CLI command**
     - ``vlm-worker``
     - ``llm-worker``
   * - **Config class**
     - ``VLMWorkerConfig``
     - ``LLMWorkerConfig``
   * - **Runtime classes**
     - ``VLMWorkerRuntime``, ``InteractiveVLMRuntime``
     - ``LLMWorkerRuntime``, ``InteractiveLLMRuntime``

Both workers share the same directory structure (``agents/``,
``environments/``, ``config/``, ``prompt_builder/``), agent strategies, and
coordination levels.

Agent Strategies
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``naive``
     - Direct observation-to-action mapping with image context.
   * - ``cot``
     - Chain-of-thought reasoning over text and image observations.
   * - ``robust_naive``
     - Naive with retry and fallback on parse failure.
   * - ``robust_cot``
     - Chain-of-thought with retry and fallback.
   * - ``few_shot``
     - In-context learning with example trajectories.
   * - ``dummy``
     - Random actions for baseline comparison (ignores images).

Supported Environments
----------------------

All environments supported by the :doc:`LLM Worker <../MOSAIC_LLM_Worker/index>`
are also supported by the VLM Worker.  Environments that provide RGB render
frames benefit most from VLM mode:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Environment
     - RGB Support
     - Notes
   * - MultiGrid (Soccer, Collect)
     - ✅
     - Full grid rendering with agent colors and ball positions
   * - BabyAI / MiniGrid
     - ✅
     - Partial observability grid renders
   * - Crafter
     - ✅
     - Rich survival environment with diverse visual elements
   * - MiniHack / NLE
     - ✅
     - Roguelike tile-based rendering
   * - TextWorld
     - ❌
     - Text-only (falls back to LLM-style prompts)
   * - PettingZoo
     - ✅
     - Board game renders (Chess, Connect Four, Go)

Runtime Modes
-------------

**Autonomous mode** (batch episodes with image observations):

.. code-block:: bash

   vlm-worker --run-id test123 \
       --env crafter \
       --client openrouter \
       --model openai/gpt-4o-mini \
       --max-image-history 1 \
       --num-episodes 10 --max-steps 200

**Text-only fallback** (equivalent to LLM Worker):

.. code-block:: bash

   vlm-worker --run-id test123 \
       --env minihack \
       --max-image-history 0 \
       --num-episodes 5

**Interactive mode** (GUI step-by-step):

.. code-block:: bash

   vlm-worker --run-id test123 --interactive \
       --env multigrid \
       --task MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0

Interactive mode reads JSON commands from stdin and emits telemetry to stdout,
identical to the LLM Worker protocol.

Configuration
-------------

**JSON config** (launched by GUI or CLI):

.. code-block:: json

   {
     "run_id": "vlm_crafter_001",
     "env_name": "crafter",
     "task": "CrafterReward-v1",
     "client_name": "openrouter",
     "model_id": "openai/gpt-4o-mini",
     "agent_type": "cot",
     "max_image_history": 1,
     "num_episodes": 5,
     "max_steps": 200,
     "temperature": 0.7
   }

**VLM-specific config fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``max_image_history``
     - ``1``
     - Number of past image frames to include in prompts.
       ``0`` = text-only fallback, ``≥ 1`` = VLM multimodal mode.
   * - ``max_text_history``
     - varies
     - Maximum text history entries alongside images
   * - ``render_mode``
     - ``None``
     - ``"rgb_array"`` to capture frames for VLM input

All other fields (``client_name``, ``model_id``, ``agent_type``,
``coordination_level``, ``observation_mode``, etc.) are identical to the
:doc:`LLM Worker <../MOSAIC_LLM_Worker/index>`.

CLI Reference
-------------

.. code-block:: text

   vlm-worker --run-id <id> [options]

   Environment:
     --env {babyai,minihack,crafter,...}    Environment family (default: babyai)
     --task <name>                          Gymnasium environment ID
     --max-steps <int>                      Max steps per episode (default: 100)
     --num-episodes <int>                   Episodes to run (default: 5)
     --seed <int>                           Random seed
     --render-mode {rgb_array,human}        Render mode for image capture

   VLM Client:
     --client {openrouter,openai,...}       API backend (default: openrouter)
     --model <model_id>                     Model identifier
     --api-key <key>                        API key (or use env vars)
     --base-url <url>                       Custom endpoint (for vLLM)
     --temperature <float>                  Sampling temperature (default: 0.7)
     --timeout <float>                      Request timeout (default: 60)

   Agent:
     --agent-type {naive,cot,...}           Agent strategy (default: naive)
     --max-image-history <int>              Image frames in prompt (default: 1)

   Output:
     --telemetry-dir <path>                 Telemetry output directory
     --no-jsonl                             Disable JSONL output
     --verbose                              Enable DEBUG logging
     --interactive                          GUI step-by-step mode
     --config <path.json>                   Load config from JSON file

.. toctree::
   :maxdepth: 1

   installation
   common_errors
