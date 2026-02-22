Homogeneous Decision-Makers
===========================

A **homogeneous** setup is one where every agent in an experiment uses
the same decision-making paradigm -- all RL, all LLM, all human, or all
baseline.  This is the simplest and most common configuration in MOSAIC.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       ENV["Environment"]

       subgraph "All Same Paradigm"
           A0["Agent 0<br/>(RL)"]
           A1["Agent 1<br/>(RL)"]
           A2["Agent 2<br/>(RL)"]
       end

       ENV -- "obs" --> A0
       ENV -- "obs" --> A1
       ENV -- "obs" --> A2
       A0 -- "action" --> ENV
       A1 -- "action" --> ENV
       A2 -- "action" --> ENV

       style ENV fill:#4a90d9,stroke:#2e5a87,color:#fff
       style A0 fill:#9370db,stroke:#6a0dad,color:#fff
       style A1 fill:#9370db,stroke:#6a0dad,color:#fff
       style A2 fill:#9370db,stroke:#6a0dad,color:#fff

The Operator Protocol
---------------------

Every decision-maker in MOSAIC implements the same ``Protocol``
(structural subtyping -- no base class inheritance required):

.. code-block:: python

   class Operator(Protocol):
       """Any entity that selects actions from observations."""

       @property
       def id(self) -> str: ...

       @property
       def name(self) -> str: ...

       def select_action(
           self,
           observation: Any,
           legal_actions: Optional[list] = None,
       ) -> Any:
           """Return an action given an observation."""
           ...

       def reset(self, seed: Optional[int] = None) -> None:
           """Reset internal state for a new episode."""
           ...

       def on_step_result(
           self,
           observation: Any,
           action: Any,
           reward: float,
           terminated: bool,
           truncated: bool,
       ) -> None:
           """Receive feedback from the environment."""
           ...

Any Python object with these methods is a valid Operator.  The GUI,
experiment runner, and telemetry system call ``select_action(obs)`` and
receive an action -- they never need to know what kind of decision-maker
they are talking to.

Categories
----------

Every operator belongs to one of five categories.  The category
determines how the GUI renders configuration controls, how the
subprocess is launched, and what action source is used:

.. _category-human:

Human
^^^^^

The human operator returns ``None`` from ``select_action()``.  The GUI
intercepts this and injects keyboard input from the user.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       OBS["Observation<br/>(rendered frame)"] --> GUI["GUI<br/>Keyboard Capture"]
       GUI --> ACTION["Action<br/>(key mapping)"]
       ACTION --> ENV["Environment"]

       style OBS fill:#eee,stroke:#999,color:#333
       style GUI fill:#9370db,stroke:#6a0dad,color:#fff
       style ACTION fill:#eee,stroke:#999,color:#333
       style ENV fill:#4a90d9,stroke:#2e5a87,color:#fff

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Worker**
     - ``human_worker``
   * - **Configuration**
     - Key bindings (arrow keys, WASD, custom)
   * - **Use case**
     - Manual play, debugging, recording demonstrations

.. _category-llm:

LLM
^^^

The LLM operator converts the observation into a text prompt, sends it
to a language model API, and parses the response into an action.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       OBS["Observation<br/>(grid / board)"] --> PROMPT["Text Prompt<br/>Generation"]
       PROMPT --> API["LLM API<br/>vLLM / OpenRouter / OpenAI"]
       API --> PARSE["Response<br/>Parsing"]
       PARSE --> ACTION["Action"]

       style OBS fill:#eee,stroke:#999,color:#333
       style PROMPT fill:#ff7f50,stroke:#cc5500,color:#fff
       style API fill:#9370db,stroke:#6a0dad,color:#fff
       style PARSE fill:#ff7f50,stroke:#cc5500,color:#fff
       style ACTION fill:#eee,stroke:#999,color:#333

MOSAIC provides three LLM workers, each serving a different purpose:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Worker
     - Scope
     - Description
   * - ``mosaic_llm_worker``
     - **Multi-agent**
     - MOSAIC's native LLM worker built for multi-agent coordination.
       Features 3-level coordination strategies (emergent, hint-based,
       role-based) and Theory of Mind observation modes.  Supports
       MultiGrid, BabyAI, MiniHack, Crafter, TextWorld, BabaIsAI,
       MeltingPot, and PettingZoo environments.
   * - ``balrog_worker``
     - **Single-agent**
     - Imported BALROG benchmark runner for single-agent LLM evaluation
       on MiniGrid, BabyAI, MiniHack, and Crafter.
   * - ``chess_worker``
     - **Two-player**
     - LLM chess play with multi-turn dialog prompting via the
       llm_chess framework.

**MOSAIC LLM coordination levels:**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Level
     - Name
     - Behavior
   * - 1
     - Emergent
     - Minimal guidance; LLMs discover coordination naturally
   * - 2
     - Basic Hints
     - Cooperation tips without being prescriptive
   * - 3
     - Role-Based
     - Explicit roles (e.g., Forward/Defender) with detailed strategies

**Configuration fields:**
   Provider, model ID, API key, base URL, temperature,
   coordination level, observation mode (egocentric / visible teammates),
   agent strategy (naive, chain-of-thought, few-shot, robust)

.. _category-rl:

RL
^^

The RL operator loads a trained neural network checkpoint and runs a
forward pass to select actions.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       OBS["Observation<br/>(tensor)"] --> NET["Neural Network<br/>Forward Pass"]
       NET --> ACTION["Action<br/>(argmax / sample)"]

       style OBS fill:#eee,stroke:#999,color:#333
       style NET fill:#9370db,stroke:#6a0dad,color:#fff
       style ACTION fill:#eee,stroke:#999,color:#333

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Workers**
     - ``cleanrl_worker`` (PPO, DQN), ``xuance_worker`` (MAPPO, QMIX),
       ``ray_worker`` (PPO, IMPALA)
   * - **Configuration**
     - Policy checkpoint path, algorithm name
   * - **Use case**
     - Evaluating trained RL policies, comparing algorithms,
       ablation studies

.. _category-baseline:

Baseline
^^^^^^^^

Baseline operators provide reference points for comparison.  They use
simple strategies like random action selection or fixed heuristics.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       OBS["Observation<br/>(ignored)"] --> RNG["Random / Heuristic<br/>Action Selection"]
       RNG --> ACTION["Action"]

       style OBS fill:#eee,stroke:#999,color:#333
       style RNG fill:#9370db,stroke:#6a0dad,color:#fff
       style ACTION fill:#eee,stroke:#999,color:#333

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Workers**
     - ``operators_worker``
   * - **Configuration**
     - Max steps, seed
   * - **Use case**
     - Sanity checks, lower-bound performance reference

Single-Worker Pattern
---------------------

In homogeneous setups, one Operator wraps one Worker subprocess.  This
is the most common pattern:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       OBS["observation"] --> OP
       subgraph OP["Operator"]
           SA["select_action()"]
           WK["Worker subprocess"]
           SA --> WK
       end
       OP --> ACT["action"]

       style OP fill:#9370db,stroke:#6a0dad,color:#fff
       style WK fill:#ff7f50,stroke:#cc5500,color:#fff
       style OBS fill:#eee,stroke:#999,color:#333
       style ACT fill:#eee,stroke:#999,color:#333

The Operator translates the protocol call into a JSON command sent over
stdin to the Worker subprocess, and reads the response from stdout.

Configuration
-------------

Single-paradigm operators are created with the ``single_agent`` factory
method:

.. code-block:: python

   # LLM operator for BabyAI
   config = OperatorConfig.single_agent(
       operator_id="llm_1",
       display_name="GPT-4o on BabyAI",
       worker_id="balrog_worker",
       worker_type="llm",
       env_name="babyai",
       task="BabyAI-GoToRedBall-v0",
       settings={
           "client_name": "vllm",
           "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
           "base_url": "http://127.0.0.1:8000/v1",
       },
   )

   # RL operator with trained policy
   config = OperatorConfig.single_agent(
       operator_id="rl_1",
       display_name="PPO on CartPole",
       worker_id="cleanrl_worker",
       worker_type="rl",
       env_name="gymnasium",
       task="CartPole-v1",
       settings={
           "algorithm": "ppo",
           "checkpoint": "runs/ppo_cartpole/model.pt",
       },
   )

   # Random baseline
   config = OperatorConfig.single_agent(
       operator_id="baseline_1",
       display_name="Random Baseline",
       worker_id="operators_worker",
       worker_type="baseline",
       env_name="gymnasium",
       task="CartPole-v1",
       settings={},
   )

Side-by-Side Comparison
-----------------------

Even within the same paradigm, running multiple operators in parallel
is useful for comparing algorithms, hyperparameters, or models:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       ENV1["MiniGrid-DoorKey-6x6-v0<br/>seed=42"]
       ENV2["MiniGrid-DoorKey-6x6-v0<br/>seed=42"]
       ENV3["MiniGrid-DoorKey-6x6-v0<br/>seed=42"]

       OP1["LLM Operator<br/>GPT-4o"]
       OP2["LLM Operator<br/>Claude 3.5"]
       OP3["LLM Operator<br/>Llama 3 70B"]

       ENV1 --> OP1
       ENV2 --> OP2
       ENV3 --> OP3

       style ENV1 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style ENV2 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style ENV3 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style OP1 fill:#9370db,stroke:#6a0dad,color:#fff
       style OP2 fill:#9370db,stroke:#6a0dad,color:#fff
       style OP3 fill:#9370db,stroke:#6a0dad,color:#fff

Each operator gets its own environment instance but shares the same
seed schedule, ensuring identical initial conditions.  The
``MultiOperatorService`` manages this:

.. code-block:: python

   class MultiOperatorService:
       def add_operator(self, config: OperatorConfig) -> None: ...
       def remove_operator(self, operator_id: str) -> None: ...
       def start_all(self) -> None: ...
       def stop_all(self) -> None: ...

GUI Adaptation by Category
--------------------------

The GUI automatically adapts its controls based on the operator
category.  The ``OperatorConfigWidget`` renders different configuration
fields for each type:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Category
     - GUI Controls
     - Component
   * - ``human``
     - Key binding display, action buttons
     - ``OperatorRenderContainer`` (interactive mode)
   * - ``llm``
     - Provider dropdown, model selector, API key field,
       base URL, temperature slider
     - ``OperatorConfigWidget``
   * - ``rl``
     - Policy file picker, algorithm dropdown
     - ``OperatorConfigWidget``
   * - ``baseline``
     - Max steps spinner, seed field
     - ``OperatorConfigWidget``

The ``OperatorRenderContainer`` also adapts its header to show a
color-coded type badge:

- **LLM** -- blue badge
- **RL** -- purple badge
- **Human** -- orange badge
- **Baseline** -- gray badge
