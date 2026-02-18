Operator Lifecycle
==================

Operators have two distinct execution modes: **Manual Mode** for
interactive exploration and **Script Mode** for automated experiments.
Each mode has its own state machine and signal flow.

Execution Modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Description
     - Entry Point
   * - **Manual Mode**
     - User clicks "Reset All" and "Step All" buttons.
       One step per click.
     - Operators Tab > Manual sub-tab
   * - **Script Mode**
     - User loads a Python script defining operators and
       seeds.  Execution runs automatically across N episodes.
     - Operators Tab > Script Experiments sub-tab

.. important::

   Manual Mode and Script Mode are **fully decoupled**, they use
   separate signal paths and state machines.  Running a script does
   not interfere with manual controls, and vice versa.

Manual Mode
-----------

State Machine
~~~~~~~~~~~~~

.. mermaid::

   stateDiagram-v2
       [*] --> IDLE : Operator configured

       IDLE --> LAUNCHING : "Start All" clicked
       LAUNCHING --> RUNNING : Subprocess spawned
       RUNNING --> STEPPING : "Step All" clicked
       STEPPING --> RUNNING : Response received
       RUNNING --> RESETTING : "Reset All" clicked
       RESETTING --> RUNNING : Ready response

       RUNNING --> STOPPED : "Stop All" clicked
       STEPPING --> STOPPED : "Stop All" clicked

       STOPPED --> [*]

Signal Flow
~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       actor User
       participant Panel as ControlPanel
       participant MW as MainWindow
       participant LAUNCH as OperatorLauncher
       participant HANDLE as ProcessHandle
       participant WORKER as Operator Subprocess

       User->>Panel: Click "Start All"
       Panel->>MW: start_operators_requested
       MW->>LAUNCH: launch_operator(config, interactive=True)
       LAUNCH-->>MW: OperatorProcessHandle

       User->>Panel: Click "Reset All"
       Panel->>MW: reset_operators_requested
       MW->>HANDLE: send_reset(seed=42)
       MW->>MW: QTimer.singleShot(100ms, poll)
       WORKER-->>HANDLE: {"type": "ready", ...}
       MW->>MW: Render initial frame

       User->>Panel: Click "Step All"
       Panel->>MW: step_operators_requested
       MW->>HANDLE: send_step()
       MW->>MW: QTimer.singleShot(100ms, poll)
       WORKER-->>HANDLE: {"type": "step", ...}
       MW->>MW: Render frame + update stats

       User->>Panel: Click "Stop All"
       Panel->>MW: stop_operators_requested
       MW->>HANDLE: send_stop()
       MW->>LAUNCH: cleanup subprocess

Script Mode
-----------

Script mode enables automated multi-episode experiments.  The user
loads a Python script that defines operator configurations and seed
lists, then the ``OperatorScriptExecutionManager`` drives execution
automatically.

State Machine
~~~~~~~~~~~~~

.. mermaid::

   stateDiagram-v2
       [*] --> IDLE : Script loaded

       IDLE --> LAUNCHING : "Run Experiment" clicked
       LAUNCHING --> WAITING_READY : Subprocess spawned + reset sent
       WAITING_READY --> STEPPING : "ready" response received

       STEPPING --> STEPPING : "step" response (paced loop)
       STEPPING --> EPISODE_DONE : "episode_end" response

       EPISODE_DONE --> WAITING_READY : Next episode (reset with next seed)
       EPISODE_DONE --> COMPLETED : All episodes finished

       STEPPING --> STOPPED : User clicks "Stop"
       WAITING_READY --> STOPPED : User clicks "Stop"

       COMPLETED --> [*]
       STOPPED --> [*]

OperatorScriptExecutionManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The execution manager is a ``QObject`` state machine that owns the
entire automatic execution lifecycle:

.. code-block:: python

   class OperatorScriptExecutionManager(QObject):
       # Signals sent to MainWindow for subprocess control
       launch_operator = pyqtSignal(str, object, int)   # op_id, config, seed
       reset_operator = pyqtSignal(str, int)             # op_id, seed
       step_operator = pyqtSignal(str)                   # op_id
       stop_operator = pyqtSignal(str)                   # op_id

       # Signals sent to ScriptExperimentWidget for UI updates
       progress_updated = pyqtSignal(int, int, int)      # episode, total, seed
       experiment_completed = pyqtSignal(int)             # num_episodes

The manager coordinates with ``MainWindow`` through signals:

.. mermaid::

   graph LR
       SCRIPT["ScriptExperimentWidget<br/>(UI)"]
       MGR["OperatorScriptExecutionManager<br/>(State Machine)"]
       MW["MainWindow<br/>(Signal Router)"]
       PROC["Operator Subprocess"]

       SCRIPT -->|"start_experiment()"| MGR
       MGR -->|"launch_operator signal"| MW
       MW -->|"spawn + send_reset()"| PROC
       PROC -->|"stdout response"| MW
       MW -->|"on_ready_received()"| MGR
       MGR -->|"step_operator signal"| MW
       MW -->|"send_step()"| PROC
       PROC -->|"stdout response"| MW
       MW -->|"on_step_received()"| MGR
       MGR -->|"progress_updated signal"| SCRIPT

       style SCRIPT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style MGR fill:#50c878,stroke:#2e8b57,color:#fff
       style MW fill:#ff7f50,stroke:#cc5500,color:#fff
       style PROC fill:#9370db,stroke:#6a0dad,color:#fff

Paced Stepping
~~~~~~~~~~~~~~

A critical design detail: after receiving a step response, the manager
does **not** immediately request the next step.  Instead, it uses
``QTimer.singleShot(delay_ms, callback)`` to insert a small delay:

.. code-block:: python

   def on_step_received(self, operator_id: str) -> None:
       if not self._is_running or not self._waiting_for_response:
           return

       def _emit_next_step():
           if self._is_running and self._waiting_for_response:
               self.step_operator.emit(operator_id)

       QTimer.singleShot(self._step_delay_ms, _emit_next_step)

This delay serves a critical purpose: **without it, the step loop runs
so fast that Qt paint events are starved**.  The render view would show
visual jumps (e.g., step 26 to step 37) because frames are logically
rendered but never visually painted by the Qt compositor.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Approach
     - Speed
     - Visual Quality
     - Status
   * - Timer spam
     - Uncontrolled
     - Broken (flooding)
     - Abandoned
   * - Immediate re-emit
     - Maximum
     - Jittery (skips frames)
     - Abandoned
   * - **Paced stepping**
     - **Configurable**
     - **Smooth**
     - **Current**

The step delay is configurable via a UI spinbox:

- **0 ms**: fastest (may skip frames visually)
- **50 ms**: smooth frame-by-frame rendering (default)
- **200 ms**: slow-motion playback for analysis
- **1000 ms**: one step per second

.. warning::

   Never use ``QApplication.processEvents()`` for step pacing.
   It causes reentrancy issues and unpredictable behavior.
   ``QTimer.singleShot`` is the Qt-idiomatic solution.

Episode Advancement
~~~~~~~~~~~~~~~~~~~

When an episode ends, the manager advances to the next seed:

.. code-block:: python

   def on_episode_ended(
       self, operator_id: str, terminated: bool, truncated: bool
   ) -> None:
       self._current_episode += 1

       if self._current_episode >= self._total_episodes:
           self.experiment_completed.emit(self._total_episodes)
           return

       next_seed = self._seeds[self._current_episode]
       self.reset_operator.emit(operator_id, next_seed)

**Seed modes:**

- **Procedural**: different seed per episode (tests generalization)
- **Fixed**: same seed every episode (isolates agent behavior)

Script Format
-------------

Experiment scripts are Python files that define operator configurations
and execution parameters:

.. code-block:: python

   # simple_random_baseline.py

   operators = [
       {
           "id": "random_1",
           "name": "Random Agent",
           "type": "baseline",
           "worker_id": "operators_worker",
           "env_name": "minigrid",
           "task": "MiniGrid-Empty-8x8-v0",
       },
   ]

   execution = {
       "num_episodes": 10,
       "seeds": [1000, 1001, 1002, 1003, 1004,
                 1005, 1006, 1007, 1008, 1009],
       "step_delay_ms": 50,
       "env_mode": "procedural",
   }

Scripts are loaded via ``compile()`` + sandboxed namespace execution
(not ``importlib``, which can cause import chain hangs on Linux):

.. code-block:: python

   with open(script_path) as f:
       code = compile(f.read(), script_path, "exec")
   namespace = {}
   exec(code, namespace)
   operators = namespace["operators"]
   execution = namespace.get("execution", {})

Telemetry Output
----------------

Operator telemetry is written to ``var/operators/telemetry/``:

.. code-block:: text

   var/operators/telemetry/
   ├── random_minigrid_{run_id}_steps.jsonl    # Per-step data
   └── random_minigrid_{run_id}_episodes.jsonl # Per-episode summaries

Each line is a JSON object with step index, action, reward,
termination status, and optional render payload.
