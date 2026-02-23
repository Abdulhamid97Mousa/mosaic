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

   %%{init: {"flowchart": {"curve": "linear"}} }%%
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
       execution_mode: str = "aec"  # "aec" (Agent-Environment Cycle, turn-based) or "parallel" (simultaneous)
       max_steps: int | None = None

WorkerAssignment
----------------

Each agent slot in an operator maps to a ``WorkerAssignment``.  For
single-agent environments there is one assignment; for multi-agent
environments there is one per player:

.. code-block:: python

   @dataclass
   class WorkerAssignment:
       worker_id: str   # e.g. "cleanrl_worker", "operators_worker"
       worker_type: str  # "llm", "vlm", "rl", "human", "baseline"
       settings: Dict[str, Any] = field(default_factory=dict)

The ``worker_type`` controls how the GUI renders configuration fields
and how the ``OperatorLauncher`` builds the subprocess command.  Valid
types:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Type
     - UI Label
     - Description
   * - ``llm``
     - LLM
     - Language model agent.  Settings include ``client_name``,
       ``model_id``, ``api_key``, ``base_url``.
   * - ``vlm``
     - VLM
     - Vision-language model.  Same as LLM plus
       ``max_image_history=1``.
   * - ``rl``
     - RL
     - Trained RL policy.  Settings include ``policy_path``,
       ``algorithm``.
   * - ``human``
     - Human
     - Keyboard-driven.  No subprocess; the GUI captures input.
   * - ``baseline``
     - Random
     - Simple scripted behavior.  Settings include ``behavior``
       (``"random"``, ``"noop"``, ``"cycling"``).  Uses
       ``operators_worker``.

.. note::

   In the Configure Operators widget the user selects **"Random"** from
   the Type dropdown.  The widget maps this to ``worker_type="baseline"``
   and ``worker_id="operators_worker"`` with ``behavior="random"``
   internally.  The distinction between the UI label and the internal
   type is intentional: users think in terms of *what the agent does*
   (random actions), not the internal worker category.

Agent-Level Interface
---------------------

The **agent-level interface** is the core abstraction that sits between
environments and decision-makers.  Every agent slot in a multi-agent
environment is assigned to exactly one decision-maker -- an RL policy,
an LLM, a human, or a random baseline.  The interface is uniform:
regardless of what runs behind it, the environment only ever calls
``select_action(obs) → action``.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       ENV["Environment<br/>(MultiGrid / MeltingPot / PettingZoo)"]

       ENV -- "obs" --> A0
       ENV -- "obs" --> A1
       ENV -- "obs" --> A2

       A0 -- "action" --> ENV
       A1 -- "action" --> ENV
       A2 -- "action" --> ENV

       subgraph AGENTS["Agent-Level Interface (Player Assignments)"]
           A0["agent_0<br/>RL · XuanCe"]
           A1["agent_1<br/>LLM · GPT-4o"]
           A2["agent_2<br/>Random · Baseline"]
       end

       style ENV fill:#4a90d9,stroke:#2e5a87,color:#fff
       style AGENTS fill:#f5f5f5,stroke:#999,color:#333
       style A0 fill:#9370db,stroke:#6a0dad,color:#fff
       style A1 fill:#50c878,stroke:#2e8b57,color:#fff
       style A2 fill:#ff7f50,stroke:#cc5500,color:#fff

This is what makes hybrid teams possible -- each agent slot is
independently configured, yet they all plug into the same environment
through a single protocol.

Player Assignment (the GUI for the Agent-Level Interface)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Player Assignment** is the GUI realization of the agent-level
interface.  The ``PlayerAssignmentPanel`` in the Configure Operators
widget lets the user wire each agent slot to a specific decision-maker
by selecting a **Type** and a **Worker**.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       OCW["OperatorConfigWidget"]

       OCW --> PAP
       OCW --> MGS

       subgraph PAP["PlayerAssignmentPanel"]
           direction TB
           ROW0["PlayerAssignmentRow<br/>agent_0 → RL · XuanCe Worker"]
           ROW1["PlayerAssignmentRow<br/>agent_1 → LLM · GPT-4o"]
       end

       subgraph MGS["Environment-Specific Settings<br/>(MultiGrid / MeltingPot)"]
           direction TB
           OBS["Observation Mode"]
           COORD["Coordination Strategy<br/>(LLM only)"]
           ROLES["Role Assignment<br/>(Level 3 only)"]
       end

       style OCW fill:#4a90d9,stroke:#2e5a87,color:#fff
       style PAP fill:#e8f5e9,stroke:#2e8b57,color:#333
       style ROW0 fill:#ff7f50,stroke:#cc5500,color:#fff
       style ROW1 fill:#ff7f50,stroke:#cc5500,color:#fff
       style MGS fill:#ede7f6,stroke:#6a0dad,color:#333
       style OBS fill:#9370db,stroke:#6a0dad,color:#fff
       style COORD fill:#9370db,stroke:#6a0dad,color:#fff
       style ROLES fill:#9370db,stroke:#6a0dad,color:#fff

Each ``PlayerAssignmentRow`` exposes:

- **Type dropdown** -- ``LLM``, ``RL``, ``Human``, or ``Random``.
  Controls which configuration fields are visible.
- **Worker dropdown** -- populated based on the selected type.  Hidden
  for Human and Random (single worker each).
- **Type-specific settings** -- LLM shows provider/model/API-key fields;
  RL shows policy path and algorithm; Human and Random show nothing
  extra.

The panel emits an ``assignments_changed`` signal whenever any row
changes, which the parent widget uses to:

1. Rebuild the ``OperatorConfig`` via ``get_config()``.
2. Update the visibility of the **Coordination Strategy** selector --
   this dropdown appears only for MultiGrid and MeltingPot environments,
   and only when at least one agent uses an LLM worker (it configures
   the ``mosaic_llm_worker``'s coordination level).  When no agent is
   LLM the entire coordination section is hidden.

.. code-block:: python

   # How the widget builds a multi-agent config (any multi-agent env)
   config = OperatorConfig.multi_agent(
       operator_id="op_0",
       display_name="Hybrid Team",
       env_name="<env_family>",        # e.g. mosaic_multigrid, meltingpot, pettingzoo
       task="<env_id>",                # e.g. Soccer-2v2, predator_prey, chess_v6
       player_workers={
           "agent_0": WorkerAssignment(
               worker_id="xuance_worker",
               worker_type="rl",
               settings={"policy_path": "/path/to/final_train_model.pth"},
           ),
           "agent_1": WorkerAssignment(
               worker_id="operators_worker",
               worker_type="baseline",
               settings={"behavior": "random"},
           ),
       },
       observation_mode="visible_teammates",
       coordination_level=1,
   )

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
