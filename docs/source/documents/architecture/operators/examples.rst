Operator Examples
=================

MOSAIC ships with several operator types, each demonstrating a
different decision-making strategy.

Human Operator
--------------

**Category:** ``human``

The simplest operator, it returns ``None`` for every action,
signalling the GUI to inject keyboard input.

.. code-block:: python

   class HumanOperator:
       @property
       def id(self) -> str:
           return "human_keyboard"

       @property
       def name(self) -> str:
           return "Human (Keyboard)"

       def select_action(self, observation, legal_actions=None):
           return None  # GUI injects keyboard action

       def reset(self, seed=None):
           pass

       def on_step_result(self, observation, action, reward,
                          terminated, truncated):
           pass

The Human Operator is always available and registered at startup.
No subprocess is spawned, actions come directly from Qt keyboard
events.

**Best for:** Manual exploration, debugging, understanding environments.

BALROG LLM Operator
--------------------

**Category:** ``llm``
**Worker:** ``barlog_worker``
**Environments:** BabyAI, MiniGrid, MiniHack, Crafter, TextWorld

The BALROG operator uses the
`BALROG benchmark <https://github.com/balrog-ai/BALROG>`_ prompting
style to evaluate LLMs as agents in grid-world environments.

.. mermaid::

   graph LR
       GUI["GUI"] -->|"stdin"| BW["barlog_worker<br/>InteractiveRuntime"]
       BW --> AGENT["BALROG Agent<br/>(naive/cot/robust)"]
       AGENT --> LLM["LLM API<br/>(vLLM / OpenRouter)"]
       LLM --> AGENT
       AGENT --> ENV["BabyAI / MiniGrid"]
       BW -->|"stdout"| GUI

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style BW fill:#ff7f50,stroke:#cc5500,color:#fff
       style LLM fill:#ffd700,stroke:#b8860b

**Prompting style:** Single-turn, BALROG-style:

.. code-block:: text

   System: You are an agent in a grid world.
   Available actions: turn_left, turn_right, forward, ...

   User: [observation text/image]
   What action do you take?

   Assistant: forward

**Configuration:**

.. code-block:: python

   config = OperatorConfig.single_agent(
       operator_id="llm_balrog",
       operator_type="llm",
       worker_id="barlog_worker",
       display_name="GPT-4o on BabyAI",
       env_name="babyai",
       task="BabyAI-GoToRedBall-v0",
       settings={
           "client_name": "vllm",
           "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
           "base_url": "http://127.0.0.1:8000/v1",
           "agent_type": "naive",
       },
   )

**Supported LLM providers:**

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Provider
     - ``client_name``
     - Notes
   * - vLLM (local)
     - ``vllm``
     - Local GPU inference, fastest
   * - OpenRouter
     - ``openrouter``
     - Multi-model gateway (GPT-4, Claude, Llama)
   * - OpenAI
     - ``openai``
     - GPT-4o, GPT-4-turbo
   * - Anthropic
     - ``anthropic``
     - Claude 3.5 Sonnet, Claude 3 Opus
   * - Google
     - ``google``
     - Gemini Pro, Gemini Ultra

Chess LLM Operator
-------------------

**Category:** ``llm``
**Worker:** ``chess_worker``
**Environments:** PettingZoo ``chess_v6``

A specialized operator that uses the
`llm_chess <https://github.com/facebookresearch/llm-chess>`_
prompting style for multi-turn chess play.

.. mermaid::

   graph TB
       subgraph "Chess Worker"
           RT["ChessWorkerRuntime"]
           CONV["Multi-turn<br/>Conversation"]
           VALID["Move Validator<br/>(regex + legal check)"]
       end

       LLM["LLM API"] <--> CONV
       RT --> CONV --> VALID
       VALID -->|"valid UCI move"| OUT["stdout response"]

       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style LLM fill:#ffd700,stroke:#b8860b

**Key differences from BALROG:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - BALROG Operator
     - Chess Operator
   * - Prompting
     - Single-turn
     - Multi-turn dialog
   * - Actions
     - Raw text to action
     - ``get_board``, ``get_legal_moves``, ``make_move``
   * - Validation
     - None
     - Regex + legal move check
   * - Retry
     - No (falls back to random)
     - Yes (up to 3 attempts)

**Multi-turn conversation example:**

.. code-block:: text

   System: You are a professional chess player playing as white.
   Actions: get_current_board, get_legal_moves, make_move <UCI>

   User: Current position: [board]. Legal moves: e2e4, d2d4, g1f3...
   LLM:  get_legal_moves
   User: Legal moves: e2e4, d2d4, g1f3, b1c3, ...
   LLM:  make_move e2e4

**Retry on invalid move:**

.. code-block:: text

   LLM:  make_move e5e6    (illegal)
   User: Invalid move 'e5e6'. Legal moves are: e2e4, d2d4, ...
   LLM:  make_move e2e4    (valid)

**Configuration:**

.. code-block:: python

   config = OperatorConfig.single_agent(
       operator_id="chess_llm",
       operator_type="llm",
       worker_id="chess_worker",
       display_name="Claude on Chess",
       env_name="pettingzoo",
       task="chess_v6",
       settings={
           "client_name": "openrouter",
           "model_id": "anthropic/claude-3.5-sonnet",
           "base_url": "https://openrouter.ai/api/v1",
           "api_key": "sk-or-...",
           "temperature": 0.3,
           "max_retries": 3,
       },
   )

RL Operator (CleanRL Interactive)
---------------------------------

**Category:** ``rl``
**Worker:** ``cleanrl_worker``
**Environments:** Any Gymnasium environment

The RL operator loads a trained CleanRL checkpoint and runs inference
step-by-step under GUI control.  This enables side-by-side comparison
of trained RL policies against LLM agents on the same environment
with shared seeds.

.. mermaid::

   graph LR
       GUI["GUI"] -->|"stdin"| CW["cleanrl_worker<br/>InteractiveRuntime"]
       CW --> POLICY["Trained Policy<br/>(PPO / DQN)"]
       CW --> ENV["Gymnasium Env"]
       POLICY -->|"action"| ENV
       CW -->|"stdout"| GUI

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style CW fill:#ff7f50,stroke:#cc5500,color:#fff
       style POLICY fill:#f0e68c,stroke:#bdb76b

**Configuration:**

.. code-block:: python

   config = OperatorConfig.single_agent(
       operator_id="rl_ppo",
       operator_type="rl",
       worker_id="cleanrl_worker",
       display_name="PPO on MiniGrid",
       env_name="minigrid",
       task="MiniGrid-Empty-8x8-v0",
       settings={
           "policy_path": "var/runs/ppo_minigrid/model.cleanrl_model",
           "algorithm": "ppo",
       },
   )

**Wrapper considerations:** The ``InteractiveRuntime`` must apply the
same observation wrappers used during training (e.g., ``ImgObsWrapper``
and ``FlattenObservation`` for MiniGrid).

MOSAIC MultiGrid Operator (Multi-Agent)
----------------------------------------

**Category:** ``llm`` or ``rl``
**Environments:** ``mosaic_multigrid`` family (Soccer, Collect, Basketball)

The mosaic_multigrid family provides competitive team-based multi-agent
games. Environments are registered via the ``mosaic-multigrid`` PyPI
package and created using ``gym.make()``.

**Configuration (2v2 Soccer with IndAgObs):**

.. code-block:: python

   config = OperatorConfig.multi_agent(
       operator_id="soccer_2v2",
       operator_type="rl",
       worker_id="cleanrl_worker",
       display_name="Soccer 2v2 IndAgObs",
       env_name="mosaic_multigrid",
       task="MosaicMultiGrid-Soccer-2vs2-IndAgObs-v0",
       workers={
           "agent_0": WorkerConfig(worker_id="cleanrl_worker", settings={}),
           "agent_1": WorkerConfig(worker_id="cleanrl_worker", settings={}),
           "agent_2": WorkerConfig(worker_id="cleanrl_worker", settings={}),
           "agent_3": WorkerConfig(worker_id="cleanrl_worker", settings={}),
       },
   )

**Available environment tiers:**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Tier
     - Envs
     - Notes
   * - Original (v1.0.2)
     - 4
     - Deprecated, backward compatible
   * - IndAgObs (v4.0.0)
     - 6
     - Individual agent observations, recommended for RL training
   * - TeamObs (v4.0.0)
     - 3
     - SMAC-style teammate awareness (2v2+ only)

**Best for:** Multi-agent RL research, team coordination, competitive
zero-sum games.

.. note::

   The ``mosaic_multigrid`` package (v4.4.0+) uses the modern
   **Gymnasium** API. All environments are registered via
   ``gymnasium.register()`` in ``mosaic_multigrid.envs``. The preview
   and runtime use ``gymnasium.make(task)`` -- never hardcode class
   imports.

Random Baseline Operator
-------------------------

**Category:** ``baseline``
**Worker:** ``operators_worker``
**Environments:** Any

The simplest non-human operator -- selects random actions from the
environment's action space.  Used as a performance baseline in
scripted experiments.

.. code-block:: python

   config = OperatorConfig.single_agent(
       operator_id="random_1",
       operator_type="baseline",
       worker_id="operators_worker",
       display_name="Random Agent",
       env_name="minigrid",
       task="MiniGrid-Empty-8x8-v0",
   )

Multi-Agent Comparison
----------------------

The most powerful use of operators is **side-by-side comparison** of
different decision-making strategies on the same environment:

.. mermaid::

   graph TB
       subgraph "Shared Environment Configuration"
           SEED["Shared Seed: 42"]
           ENV["MiniGrid-Empty-8x8-v0"]
       end

       subgraph "Operator 1"
           O1["LLM Operator<br/>GPT-4o"]
           R1["Render Container 1"]
       end

       subgraph "Operator 2"
           O2["RL Operator<br/>PPO (trained)"]
           R2["Render Container 2"]
       end

       subgraph "Operator 3"
           O3["Random Baseline"]
           R3["Render Container 3"]
       end

       SEED --> O1
       SEED --> O2
       SEED --> O3
       ENV --> O1
       ENV --> O2
       ENV --> O3

       style SEED fill:#f0e68c,stroke:#bdb76b
       style O1 fill:#9370db,stroke:#6a0dad,color:#fff
       style O2 fill:#9370db,stroke:#6a0dad,color:#fff
       style O3 fill:#9370db,stroke:#6a0dad,color:#fff
       style R1 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style R2 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style R3 fill:#4a90d9,stroke:#2e5a87,color:#fff

Each operator gets its own:

- Environment instance (same seed for reproducibility)
- Render container in the GUI
- Subprocess with independent state
- Telemetry output file

**Script for 3-way comparison:**

.. code-block:: python

   # compare_llm_rl_random.py

   operators = [
       {
           "id": "llm_gpt4o",
           "name": "GPT-4o",
           "type": "llm",
           "worker_id": "barlog_worker",
           "env_name": "babyai",
           "task": "BabyAI-GoToRedBall-v0",
           "settings": {
               "client_name": "openrouter",
               "model_id": "openai/gpt-4o",
           },
       },
       {
           "id": "rl_ppo",
           "name": "Trained PPO",
           "type": "rl",
           "worker_id": "cleanrl_worker",
           "env_name": "minigrid",
           "task": "BabyAI-GoToRedBall-v0",
           "settings": {
               "policy_path": "var/runs/ppo_babyai/model.cleanrl_model",
               "algorithm": "ppo",
           },
       },
       {
           "id": "random_baseline",
           "name": "Random",
           "type": "baseline",
           "worker_id": "operators_worker",
           "env_name": "minigrid",
           "task": "BabyAI-GoToRedBall-v0",
       },
   ]

   execution = {
       "num_episodes": 50,
       "seeds": list(range(1000, 1050)),
       "step_delay_ms": 0,           # Fastest (no visual pacing)
       "env_mode": "procedural",     # Different layout per episode
   }

Comparison Table
----------------

.. list-table::
   :header-rows: 1
   :widths: 16 14 14 14 14 14 14

   * - Feature
     - Human
     - BALROG LLM
     - Chess LLM
     - CleanRL RL
     - Random
     - MOSAIC LLM
   * - **Category**
     - human
     - llm
     - llm
     - rl
     - baseline
     - llm
   * - **Subprocess**
     - No
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - **Training**
     - No
     - No
     - No
     - Yes (offline)
     - No
     - No
   * - **Multi-agent**
     - Yes
     - No
     - Yes
     - No
     - No
     - Yes
   * - **Retry Logic**
     - N/A
     - No
     - Yes (3x)
     - N/A
     - N/A
     - No
   * - **GPU Required**
     - No
     - Optional
     - Optional
     - No
     - No
     - Optional
