What Is an Operator?
====================

Operators answer a single question: **given an observation, what action
should the agent take?**  They are the decision-making layer of MOSAIC,
sitting above the process-level :doc:`Worker <../workers/index>`
abstraction.

The core interface is simple:

.. code-block:: text

   observation --> [Operator] --> action

Every decision-maker whether it may be a human, LLM, RL policy, or Random
policy,  implements the same ``select_action(obs) -> action``
protocol.  This makes all decision-makers **interchangeable**: the GUI,
the experiment runner, and the telemetry system never need to know what
kind of operator they are talking to.

Operator vs Worker
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Concept
     - Definition
     - Examples
   * - **Operator**
     - The *agent-level interface*, wraps one or more Worker
       subprocesses and presents ``select_action(obs) -> action``
       to the GUI.
     - LLM Operator, RL Operator, Human Operator,
       Chess Operator (wraps 2 workers)
   * - **Worker**
     - A *process-level* execution unit inside an Operator.
       Manages library lifecycle, API calls, or scripted behaviors.
       Lives in ``3rd_party/`` and communicates via stdin/stdout JSON.
     - ``balrog_worker``, ``cleanrl_worker``, ``xuance_worker``,
       ``ray_worker``, ``operators_worker``

Two Modes of Operation
----------------------

MOSAIC supports two fundamentally different operator configurations:

.. mermaid::

   graph LR
       subgraph "Homogeneous"
           H1["RL"]
           H2["RL"]
           H3["RL"]
       end

       subgraph "Hybrid"
           X1["RL"]
           X2["LLM"]
           X3["Human"]
       end

       style H1 fill:#9370db,stroke:#6a0dad,color:#fff
       style H2 fill:#9370db,stroke:#6a0dad,color:#fff
       style H3 fill:#9370db,stroke:#6a0dad,color:#fff
       style X1 fill:#9370db,stroke:#6a0dad,color:#fff
       style X2 fill:#4a90d9,stroke:#2e5a87,color:#fff
       style X3 fill:#ff7f50,stroke:#cc5500,color:#fff

:doc:`Homogeneous Decision-Makers <homogenous_decision_makers/index>`
   All agents use the same paradigm (all RL, all LLM, etc.).
   Covers the Operator Protocol, the five categories (human, llm, rl,
   baseline), the single-worker pattern, and GUI adaptation by
   category.

:doc:`Hybrid Decision-Maker <hybrid_decision_maker/index>`
   Agents use **different** paradigms in the same experiment (e.g., RL +
   LLM as teammates).  Covers the research gap this addresses, the
   WorkerAssignment system, experimental configurations, deterministic
   cross-paradigm evaluation, and the research questions this enables.

OperatorConfig
--------------

Each operator is configured via an ``OperatorConfig`` dataclass:

.. code-block:: python

   @dataclass
   class OperatorConfig:
       operator_id: str
       display_name: str
       env_name: str
       task: str
       workers: Dict[str, WorkerAssignment]
       run_id: str | None = None
       execution_mode: str = "aec"
       max_steps: int | None = None

OperatorService
---------------

The ``OperatorService`` provides a central registry for all available
operators:

.. code-block:: python

   class OperatorService:
       def register_operator(self, operator, descriptor) -> None: ...
       def set_active_operator(self, operator_id: str) -> None: ...
       def select_action(self, observation: Any) -> Any: ...
       def seed(self, seed: int) -> None: ...

At startup, MOSAIC registers built-in operators and any discovered via
entry points.  The GUI's operator dropdown is populated from
``OperatorService.get_descriptors()``.

Directory Layout
----------------

Operator-related code lives in the MOSAIC core (not ``3rd_party/``):

.. code-block:: text

   gym_gui/
       services/
           operator.py                          # Protocol + OperatorService
           operator_launcher.py                 # Subprocess spawning
           operator_script_execution_manager.py # Script mode state machine
       ui/
           widgets/
               operators_tab.py                 # Manual + Script mode tabs
               operator_config_widget.py        # Per-operator config rows
               operator_render_container.py     # Per-operator render view
               multi_operator_render_view.py    # Grid of render containers
               script_experiment_widget.py      # Script mode UI
           panels/
               control_panel_container.py       # Service-to-UI bridge
           config_panels/
               single_agent/                    # Per-game environment configs
               multi_agent/                     # Multi-agent environment configs
