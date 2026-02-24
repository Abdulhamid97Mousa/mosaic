IPC Architecture
================

Operators are the agent-level interface of MOSAIC.  Each Operator wraps
one or more Worker subprocesses and communicates with the MOSAIC GUI
through a lightweight **stdin/stdout JSON protocol**.  Unlike the
training pipeline (which uses the three-tier gRPC architecture),
Operators use direct subprocess IPC for low-latency, step-by-step
control.

MOSAIC provides two evaluation modes that share this IPC architecture:

- **Manual Mode:** Side-by-side visual comparison of up to *N* concurrent
  operators through lock-step execution.  "Step All" advances every operator
  by one environment step simultaneously and "Reset All" synchronizes all
  operators to identical initial conditions via shared seeds, across
  single-agent, sequential (AEC), and simultaneous (POSG) stepping paradigms.

- **Script Mode:** Automated, long-running evaluation driven by Python scripts
  that define operator configurations, worker assignments, seed sequences, and
  episode counts.  Scripts execute deterministically with no manual
  intervention, producing reproducible JSONL telemetry for every step and
  episode.  Scripts support *procedural seeds* (different seed per episode to
  test generalization) and *fixed seeds* (same seed every episode to isolate
  agent behaviour), with configurable step pacing for visual inspection or
  headless batch execution.

Two Communication Modes
-----------------------

MOSAIC supports two IPC patterns for operators, chosen at launch time:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Mechanism
     - Use Case
   * - **Interactive**
     - stdin/stdout JSON lines
     - GUI-controlled stepping (manual and script mode)
   * - **Batch**
     - CLI args + JSONL stdout
     - Autonomous evaluation runs

Interactive mode is the primary mode for the Operator system.  It
enables the GUI to send commands one at a time and receive responses,
creating a synchronized step loop.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       subgraph GUI["MOSAIC GUI  (Main Process)"]
           direction LR
           LAUNCHER["OperatorLauncher"]
           HANDLE["OperatorProcessHandle"]
       end

       LAUNCHER -- "subprocess.Popen()" --> RUNTIME
       HANDLE -- "stdin: JSON command<br/>{'cmd':'step'}" --> RUNTIME
       RUNTIME -- "stdout: JSON response<br/>{'type':'step', ...}" --> HANDLE

       subgraph WORKER["Worker Subprocess  (Operator / Agent-Level Interface)"]
           RUNTIME["InteractiveRuntime<br/>(stdin/stdout loop)"]
           ENV["Environment<br/>(Gymnasium)"]
           POLICY["Policy / LLM / Human"]
           RUNTIME -- "obs = env.step(action)" --> ENV
           RUNTIME -- "action = select_action(obs)" --> POLICY
       end

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style WORKER fill:#50c878,stroke:#2e8b57,color:#fff
       style LAUNCHER fill:#2a6db5,stroke:#1a4d80,color:#fff
       style HANDLE fill:#2a6db5,stroke:#1a4d80,color:#fff
       style RUNTIME fill:#ff7f50,stroke:#cc5500,color:#fff
       style ENV fill:#ddd,stroke:#999,color:#333
       style POLICY fill:#ddd,stroke:#999,color:#333

Interactive JSON Protocol
-------------------------

Commands (GUI to Operator via stdin)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {"cmd": "reset", "seed": 42}
   {"cmd": "step"}
   {"cmd": "stop"}

Responses (Operator to GUI via stdout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ready response** -- after a successful reset:

.. code-block:: json

   {
     "type": "ready",
     "run_id": "op_llm_1_20260115",
     "env_id": "BabyAI-GoToRedBall-v0",
     "seed": 42,
     "observation_shape": [7, 7, 3]
   }

**Step response** -- after each environment step:

.. code-block:: json

   {
     "type": "step",
     "step_index": 5,
     "action": 2,
     "reward": 0.0,
     "terminated": false,
     "truncated": false,
     "episode_reward": 0.0,
     "render_payload": {
       "mode": "rgb",
       "rgb": [[[255, 0, 0], [0, 255, 0]]],
       "width": 64,
       "height": 64
     }
   }

**Episode end** -- when the episode terminates or truncates:

.. code-block:: json

   {
     "type": "episode_end",
     "total_reward": 0.95,
     "episode_length": 15,
     "terminated": true,
     "truncated": false
   }

**Error** -- on failure:

.. code-block:: json

   {
     "type": "error",
     "message": "Failed to load policy: FileNotFoundError"
   }

**Stopped** -- acknowledgement of stop command:

.. code-block:: json

   {"type": "stopped"}

OperatorProcessHandle
---------------------

The ``OperatorProcessHandle`` wraps a subprocess and provides typed
methods for the JSON protocol:

.. code-block:: python

   class OperatorProcessHandle:
       """Manages a single operator subprocess."""

       def __init__(self, process, operator_id, config, run_id): ...

       # Properties
       @property
       def pid(self) -> int: ...
       @property
       def is_running(self) -> bool: ...

       # Commands (write to stdin)
       def send_reset(self, seed: int) -> None: ...
       def send_step(self) -> None: ...
       def send_stop(self) -> None: ...

       # Response reading (read from stdout)
       def try_read_response(self, timeout: float = 0.1) -> dict | None: ...
       def read_response(self, timeout: float = 5.0) -> dict: ...

.. important::

   Operators **must** be launched with ``interactive=True`` for the
   stdin/stdout protocol to work.  Without it, ``send_reset()`` fails
   silently with "Cannot send command to non-interactive operator".

OperatorLauncher
----------------

The ``OperatorLauncher`` spawns operator subprocesses and returns
``OperatorProcessHandle`` instances:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       LAUNCHER["OperatorLauncher"]

       subgraph OP_LLM["LLM Operator"]
           LLM["balrog_worker / chess_worker / operators_worker"]
       end

       subgraph OP_RL["RL Operator"]
           RL["cleanrl_worker --interactive"]
       end

       subgraph OP_BASE["Baseline Operator"]
           BASE["operators_worker --random"]
       end

       subgraph OP_HUMAN["Human Operator"]
           HUMAN["human_worker / keyboard input"]
       end

       LAUNCHER -->|"_build_llm_command()"| OP_LLM
       LAUNCHER -->|"_build_rl_command()"| OP_RL
       LAUNCHER -->|"_build_baseline_command()"| OP_BASE
       LAUNCHER -->|"_build_human_command()"| OP_HUMAN

       style LAUNCHER fill:#50c878,stroke:#2e8b57,color:#fff
       style OP_LLM fill:#9370db,stroke:#6a0dad,color:#fff
       style OP_RL fill:#9370db,stroke:#6a0dad,color:#fff
       style OP_BASE fill:#9370db,stroke:#6a0dad,color:#fff
       style OP_HUMAN fill:#9370db,stroke:#6a0dad,color:#fff
       style LLM fill:#ff7f50,stroke:#cc5500,color:#fff
       style RL fill:#ff7f50,stroke:#cc5500,color:#fff
       style BASE fill:#ff7f50,stroke:#cc5500,color:#fff
       style HUMAN fill:#ff7f50,stroke:#cc5500,color:#fff

**Command dispatch** -- the launcher selects the correct worker
subprocess based on operator type:

.. code-block:: python

   class OperatorLauncher:
       def launch_operator(
           self,
           config: OperatorConfig,
           run_id: str,
           *,
           interactive: bool = False,
       ) -> OperatorProcessHandle:
           """Spawn an operator subprocess."""
           if config.operator_type == "llm":
               cmd = self._build_llm_command(config, run_id, interactive)
           elif config.operator_type == "rl":
               cmd = self._build_rl_command(config, run_id, interactive)
           elif config.operator_type == "baseline":
               cmd = self._build_baseline_command(config, run_id, interactive)
           else:
               cmd = self._build_human_command(config, run_id)

           process = subprocess.Popen(
               cmd,
               stdin=subprocess.PIPE if interactive else None,
               stdout=subprocess.PIPE if interactive else log_file,
               stderr=log_file,
           )
           return OperatorProcessHandle(process, config, run_id)

**LLM worker routing** -- the launcher automatically selects the
correct LLM worker based on the environment:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Environment
     - Worker
     - Reason
   * - PettingZoo Chess
     - ``chess_worker``
     - llm_chess multi-turn prompting
   * - BabyAI / MiniGrid
     - ``barlog_worker``
     - BALROG-style prompting
   * - Other environments
     - ``operators_worker``
     - General-purpose LLM operator

Multi-Agent Operator Launching
------------------------------

For multi-agent environments (chess, soccer, etc.), the launcher
spawns one subprocess **per player**:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph TB
       LAUNCHER["OperatorLauncher"]

       subgraph OP["Chess Operator (Agent-Level Interface)"]
           MULTI["MultiAgentOperatorHandle"]
           subgraph W0["chess_worker (Player 0 -- GPT-4o)"]
               P0_RT["InteractiveRuntime"]
           end
           subgraph W1["chess_worker (Player 1 -- Claude)"]
               P1_RT["InteractiveRuntime"]
           end
       end

       LAUNCHER -->|"launch_multiagent_operator()"| OP
       MULTI --> W0
       MULTI --> W1

       style LAUNCHER fill:#50c878,stroke:#2e8b57,color:#fff
       style OP fill:#9370db,stroke:#6a0dad,color:#fff
       style MULTI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style W0 fill:#ff7f50,stroke:#cc5500,color:#fff
       style W1 fill:#ff7f50,stroke:#cc5500,color:#fff

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15); margin-top:16px;" controls autoplay muted loop playsinline>
     <source src="../../../_static/videos/pettingzoo_chess_v6.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Multi-Agent Operator in Action:</strong> Two LLM workers (GPT-4o vs Claude) each running in a separate subprocess, coordinated by a single <code>MultiAgentOperatorHandle</code> over PettingZoo Chess (chess_v6).
   </p>

.. code-block:: python

   class MultiAgentOperatorHandle:
       """Manages N worker subprocesses for a multi-agent operator."""

       player_handles: dict[str, OperatorProcessHandle]

       def send_init_agents(self) -> None: ...
       def send_select_action(
           self, player_id: str, observation: Any, ...
       ) -> None: ...
       def poll_all_responses(self) -> dict[str, dict]: ...

Environment Variables
---------------------

The launcher sets environment variables for each operator subprocess:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable
     - Example
     - Purpose
   * - ``OPERATOR_ID``
     - ``llm_1``
     - Operator identifier
   * - ``OPERATOR_RUN_ID``
     - ``op_llm_1_20260115``
     - Unique run identifier
   * - ``TELEMETRY_DIR``
     - ``var/operators/telemetry``
     - Output directory for JSONL logs
   * - ``MPI4PY_RC_INITIALIZE``
     - ``0``
     - Prevent MPI hang on import

For vLLM operators, proxy environment variables (``ALL_PROXY``,
``HTTP_PROXY``, etc.) are cleared to prevent SOCKS proxies from
blocking localhost connections.

Response Polling
----------------

After sending a command, the GUI polls for responses using
``QTimer.singleShot``:

.. mermaid::

   sequenceDiagram
       participant GUI as MainWindow
       participant HANDLE as ProcessHandle
       participant WORKER as Operator Subprocess

       GUI->>HANDLE: send_step()
       GUI->>GUI: QTimer.singleShot(100ms, poll)

       Note over WORKER: Processing step...

       GUI->>HANDLE: try_read_response(timeout=0.1)
       HANDLE-->>GUI: None (not ready yet)
       GUI->>GUI: QTimer.singleShot(100ms, poll)

       WORKER-->>HANDLE: {"type": "step", ...}
       GUI->>HANDLE: try_read_response(timeout=0.1)
       HANDLE-->>GUI: {"type": "step", "action": 2, ...}

       GUI->>GUI: Render frame + update UI

This polling pattern keeps the Qt event loop responsive while
waiting for potentially slow operations (LLM API calls can take
seconds).
