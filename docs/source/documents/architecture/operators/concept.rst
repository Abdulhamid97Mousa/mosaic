What Is an Operator?
====================

Operators answer a single question: **given an observation, what action
should the agent take?**  They are the decision-making layer of MOSAIC,
sitting above the process-level Worker abstraction.

The Problem
-----------

MOSAIC supports many kinds of decision-makers: humans clicking keys,
LLMs reasoning about board states, trained RL policies, and scripted
baselines.  Without a unified abstraction, the GUI would need separate
code paths for each:

.. mermaid::

   graph LR
       subgraph "Without Operators (Fragile)"
           GUI["GUI"]
           GUI -->|"if human"| KB["Keyboard Handler"]
           GUI -->|"if RL"| RL["Policy Loader"]
           GUI -->|"if LLM"| LLM["API Client"]
           GUI -->|"if script"| SC["Script Runner"]
       end

       style GUI fill:#ddd,stroke:#999

The Solution: The Operator Protocol
------------------------------------

MOSAIC defines a ``Protocol`` (structural subtyping) that any
decision-maker must satisfy:

.. code-block:: python

   from typing import Protocol, Any, Optional

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

Any Python object that implements these methods is a valid Operator --
no inheritance required.  This enables maximum flexibility: a human
operator returns ``None`` (the GUI injects keyboard input), an RL
operator runs a forward pass through a neural network, and an LLM
operator calls a language model API.

.. mermaid::

   graph TB
       PROTO["Operator Protocol<br/>select_action(obs) -> action"]

       HUMAN["HumanOperator<br/>returns None<br/>(GUI injects input)"]
       LLM["LLM Operator<br/>calls vLLM / OpenRouter"]
       RL["RL Operator<br/>PyTorch forward pass"]
       BASE["Baseline Operator<br/>random / scripted"]

       PROTO --> HUMAN
       PROTO --> LLM
       PROTO --> RL
       PROTO --> BASE

       style PROTO fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HUMAN fill:#9370db,stroke:#6a0dad,color:#fff
       style LLM fill:#9370db,stroke:#6a0dad,color:#fff
       style RL fill:#9370db,stroke:#6a0dad,color:#fff
       style BASE fill:#9370db,stroke:#6a0dad,color:#fff

Worker vs Operator
------------------

These two concepts serve different purposes and operate at different
levels of the architecture:

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Concept
     - Definition
     - Examples
   * - **Worker**
     - A *process-level* wrapper that manages an RL library's lifecycle,
       configuration, and telemetry.  Workers live in ``3rd_party/`` and
       communicate via gRPC/JSONL.
     - ``cleanrl_worker``, ``xuance_worker``, ``ray_worker``
   * - **Operator**
     - An *agent-level* abstraction that selects actions step-by-step
       via ``select_action(obs) -> action``.  Operators are the
       decision-makers that run inside worker subprocesses.
     - Human Operator, LLM Operator, RL Operator

A **Worker** *contains* one or more **Operators**.  For example:

.. mermaid::

   graph LR
       subgraph Worker["CleanRL Worker (Process)"]
           TRAIN["Training Loop"]
           CKPT["Checkpoint"]
       end

       subgraph Operator["RL Operator (Interface)"]
           SA["select_action(obs)"]
           RESET["reset(seed)"]
       end

       TRAIN -->|"produces"| CKPT
       CKPT -->|"loads into"| Operator

       style Worker fill:#ff7f50,stroke:#cc5500,color:#fff
       style Operator fill:#9370db,stroke:#6a0dad,color:#fff
       style CKPT fill:#f0e68c,stroke:#bdb76b

At **training time**, the Worker drives the loop.  At **evaluation
time**, the trained policy is loaded into an Operator that the GUI
controls step-by-step.

Operator Categories
-------------------

Every operator belongs to one of five categories.  The category
determines which GUI controls are shown and how the subprocess is
launched:

.. list-table::
   :header-rows: 1
   :widths: 15 30 30 25

   * - Category
     - Description
     - Configuration
     - Action Source
   * - ``human``
     - Manual keyboard input
     - None (key bindings)
     - GUI keyboard events
   * - ``llm``
     - Language model inference
     - Provider, model, API key, base URL
     - LLM API call
   * - ``rl``
     - Trained RL policy
     - Policy path, algorithm
     - Neural network forward pass
   * - ``bdi``
     - Belief-Desire-Intention agent
     - Agent configuration
     - BDI reasoning engine
   * - ``baseline``
     - Scripted or random agent
     - Max steps, seed
     - Random / heuristic

OperatorConfig
--------------

Each operator is configured via an ``OperatorConfig`` dataclass that
captures everything needed to launch and control it:

.. code-block:: python

   @dataclass
   class OperatorConfig:
       operator_id: str
       operator_type: str        # "llm", "rl", "human", "baseline"
       worker_id: str            # Which worker subprocess to use
       display_name: str
       env_name: str             # Environment family
       task: str                 # Specific environment ID
       settings: dict            # Type-specific settings
       run_id: str = ""          # Assigned at launch time
       max_steps: int = 0        # 0 = unlimited

**Factory methods** simplify creation:

.. code-block:: python

   # Single-agent operator
   config = OperatorConfig.single_agent(
       operator_id="llm_1",
       operator_type="llm",
       worker_id="barlog_worker",
       display_name="GPT-4o on BabyAI",
       env_name="babyai",
       task="BabyAI-GoToRedBall-v0",
       settings={
           "client_name": "vllm",
           "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
           "base_url": "http://127.0.0.1:8000/v1",
       },
   )

   # Multi-agent operator (e.g., Chess)
   config = OperatorConfig.multi_agent(
       operator_id="chess_match",
       display_name="LLM Chess Match",
       env_name="pettingzoo",
       task="chess_v6",
       worker_assignments={
           "player_0": WorkerAssignment(
               worker_id="chess_worker", worker_type="llm", ...
           ),
           "player_1": WorkerAssignment(
               worker_id="chess_worker", worker_type="llm", ...
           ),
       },
   )

OperatorService
---------------

The ``OperatorService`` provides a central registry for all available
operators:

.. code-block:: python

   class OperatorService:
       def register_operator(
           self,
           operator: Operator,
           descriptor: OperatorDescriptor,
       ) -> None: ...

       def set_active_operator(self, operator_id: str) -> None: ...

       def select_action(self, observation: Any) -> Any: ...

       def seed(self, seed: int) -> None: ...

At startup, MOSAIC registers built-in operators (Human, Worker) and
any discovered via entry points.  The GUI's operator dropdown is
populated from ``OperatorService.get_descriptors()``.

MultiOperatorService
--------------------

For side-by-side comparison, the ``MultiOperatorService`` manages
N operators running in parallel:

.. code-block:: python

   class MultiOperatorService:
       def add_operator(self, config: OperatorConfig) -> None: ...
       def remove_operator(self, operator_id: str) -> None: ...
       def get_active_operators(self) -> list[OperatorConfig]: ...
       def start_all(self) -> None: ...
       def stop_all(self) -> None: ...

Each operator gets its own environment instance, render container,
and subprocess -- but they share the same seed for reproducibility.

Directory Layout
----------------

Operator-related code lives in the MOSAIC core (not ``3rd_party/``):

.. code-block:: text

   gym_gui/
   ├── services/
   │   ├── operator.py                          # Protocol + OperatorService
   │   ├── operator_launcher.py                 # Subprocess spawning
   │   └── operator_script_execution_manager.py # Script mode state machine
   └── ui/widgets/
       ├── operator_config_widget.py            # Configuration UI
       ├── operator_render_container.py         # Per-operator render view
       ├── multi_operator_render_view.py        # Grid of render containers
       └── script_experiment_widget.py          # Script mode UI
