Worker Lifecycle
================

Every worker follows a well-defined lifecycle managed by the Trainer
Daemon.  A finite-state machine governs transitions from job submission
to completion (or failure).

State Machine
-------------

.. mermaid::

   stateDiagram-v2
       [*] --> INIT : SubmitRun()

       INIT --> HANDSHAKE : Dispatcher spawns worker
       HANDSHAKE --> READY : RegisterWorker succeeds
       READY --> EXECUTING : First telemetry received
       EXECUTING --> TERMINATED : Worker exits (code 0)
       EXECUTING --> FAULTED : Crash or heartbeat timeout

       INIT --> CANCELLED : User cancels
       HANDSHAKE --> CANCELLED : User cancels
       READY --> CANCELLED : User cancels
       EXECUTING --> CANCELLED : User cancels

       TERMINATED --> [*]
       FAULTED --> [*]
       CANCELLED --> [*]

State Descriptions
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 55 27

   * - State
     - Description
     - Trigger
   * - ``INIT``
     - Run is registered in the RunRegistry (SQLite).
       Waiting for the Dispatcher to pick it up.
     - ``SubmitRun()`` RPC from GUI
   * - ``HANDSHAKE``
     - Worker subprocess has been spawned.  Waiting for it to call
       ``RegisterWorker()`` with its capabilities.
     - Dispatcher spawns process
   * - ``READY``
     - Worker has registered and is ready to emit telemetry.
       Session token has been issued.
     - ``RegisterWorker()`` succeeds
   * - ``EXECUTING``
     - Training or evaluation is in progress.  Telemetry is flowing
       from worker → proxy → daemon → GUI.
     - First ``PublishRunSteps`` received
   * - ``TERMINATED``
     - Worker exited cleanly (return code 0).  GPU slots released.
       Analytics manifest available.
     - Worker process exits normally
   * - ``FAULTED``
     - Worker crashed, was killed, or missed heartbeats for 5 minutes.
       GPU slots released.  Error details logged.
     - Non-zero exit or heartbeat timeout
   * - ``CANCELLED``
     - User manually cancelled the run.  ``SIGTERM`` sent to the
       worker process group.
     - ``CancelRun()`` RPC from GUI

Job Submission Flow
-------------------

The complete journey from clicking "Start Training" to seeing live
telemetry in the GUI:

.. mermaid::

   sequenceDiagram
       actor User
       participant GUI as Qt6 GUI
       participant Client as TrainerClient
       participant Service as TrainerService
       participant Registry as RunRegistry (SQLite)
       participant Dispatch as TrainerDispatcher
       participant Proxy as Telemetry Proxy
       participant Worker as Worker Process

       User->>GUI: Click "Start Training"
       GUI->>Client: submit_run(config)
       Client->>Service: SubmitRun(config_json)
       Service->>Registry: INSERT run (status=INIT)
       Service-->>Client: run_id

       Note over Dispatch: Polls every 2s for INIT runs

       Dispatch->>Registry: SELECT WHERE status=INIT
       Registry-->>Dispatch: [run_record]

       Dispatch->>Dispatch: Write worker config to disk
       Dispatch->>Proxy: spawn subprocess
       Proxy->>Worker: spawn subprocess
       Dispatch->>Registry: UPDATE status=HANDSHAKE

       Worker->>Service: RegisterWorker(capabilities)
       Service->>Registry: UPDATE status=READY
       Service-->>Worker: session_token

       Worker->>Worker: Start training loop

       loop Every step
           Worker->>Proxy: JSONL to stdout
           Proxy->>Service: PublishRunSteps(RunStep)
           Service->>Registry: Persist step
       end

       Note over Registry: status → EXECUTING

       Service->>Client: StreamRunSteps(stream)
       Client->>GUI: Live telemetry update
       GUI->>User: Render frame + charts

       Worker->>Proxy: run_completed event
       Worker->>Worker: Exit (code 0)
       Dispatch->>Registry: UPDATE status=TERMINATED
       Dispatch->>Dispatch: Release GPU slots

Configuration Flow
------------------

When a job is submitted, the Daemon writes a worker-specific config
file to disk.  The worker reads it at startup.

.. mermaid::

   graph LR
       GUI["GUI Form"] -->|"JSON config"| DAEMON["Daemon"]
       DAEMON -->|"write"| DISK["/var/gym_gui/trainer/configs/<br/>worker-{run_id}.json"]
       DISK -->|"read"| WORKER["Worker"]
       DAEMON -->|"env vars"| WORKER

       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff
       style DAEMON fill:#50c878,stroke:#2e8b57,color:#fff
       style WORKER fill:#ff7f50,stroke:#cc5500,color:#fff

**Trainer config** (``config-{run_id}.json``):

.. code-block:: json

   {
     "metadata": {
       "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
       "run_name": "CartPole-PPO-Experiment",
       "worker": {
         "module": "cleanrl_worker.cli",
         "use_grpc": true,
         "grpc_target": "127.0.0.1:50055",
         "worker_id": "worker-001",
         "config": { "env_id": "CartPole-v1", "algo": "ppo" }
       }
     },
     "payload": {
       "resources": {
         "gpus": { "requested": 1, "mandatory": false }
       }
     }
   }

**Worker config** (``worker-{run_id}.json``):

.. code-block:: json

   {
     "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
     "worker_id": "worker-001",
     "env_id": "CartPole-v1",
     "algo": "ppo",
     "total_timesteps": 10000,
     "seed": 42,
     "extras": {
       "learning_rate": 0.0003,
       "cuda": true,
       "tensorboard_dir": "tensorboard"
     }
   }

**Environment variables** set by the Dispatcher:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable
     - Example
     - Purpose
   * - ``RUN_ID``
     - ``01ARZ3ND...``
     - Unique run identifier
   * - ``WORKER_ID``
     - ``worker-001``
     - Worker instance ID
   * - ``CUDA_VISIBLE_DEVICES``
     - ``0,1``
     - GPU allocation
   * - ``GYM_GUI_FASTLANE_ONLY``
     - ``1``
     - Enable FastLane rendering
   * - ``GYM_GUI_FASTLANE_SLOT``
     - ``0``
     - FastLane slot index

Telemetry Flow
--------------

Step and episode data flows from the worker process through five
stages before reaching the user's screen:

.. mermaid::

   graph TB
       W["1. Worker Process<br/>print(json, flush=True)"]
       P["2. Telemetry Proxy<br/>JsonlTailer parses stdout"]
       G["3. gRPC Stream<br/>PublishRunSteps()"]
       D["4. Daemon Ingestion<br/>SQLite persist + fan-out"]
       B["5. RunTelemetryBroadcaster<br/>per-run circular buffer (4096)"]
       H["6. TelemetryAsyncHub<br/>drain loop → Qt signals"]
       UI["7. Live Telemetry Tab<br/>render frame + chart"]

       W --> P --> G --> D --> B --> H --> UI

       style W fill:#87ceeb,stroke:#4682b4
       style P fill:#87ceeb,stroke:#4682b4
       style G fill:#87ceeb,stroke:#4682b4
       style D fill:#87ceeb,stroke:#4682b4
       style B fill:#87ceeb,stroke:#4682b4
       style H fill:#87ceeb,stroke:#4682b4
       style UI fill:#87ceeb,stroke:#4682b4

Each stage adds value:

1. **Worker**: produces raw training data
2. **Proxy**: validates, parses, and converts to protobuf
3. **gRPC**: typed, backpressure-aware transport
4. **Daemon**: persists to SQLite (crash-safe, WAL mode)
5. **Broadcaster**: per-client subscription queues with replay
6. **Hub**: bridges async gRPC → Qt event loop
7. **UI**: renders frames, updates charts, logs metrics

Storage Layout
--------------

All run artifacts are organized under ``/var/gym_gui/``:

.. code-block:: text

   /var/gym_gui/
   ├── trainer/
   │   ├── configs/
   │   │   ├── config-{run_id}.json        # Full trainer config
   │   │   └── worker-{run_id}.json        # Per-worker config
   │   ├── runs/                           # Training run artifacts
   │   │   └── {run_id}/
   │   │       ├── tensorboard/            # TensorBoard logs
   │   │       ├── checkpoints/            # Model checkpoints
   │   │       ├── videos/                 # Recorded episodes
   │   │       ├── logs/
   │   │       │   ├── worker.stdout.log
   │   │       │   └── worker.stderr.log
   │   │       └── analytics.json          # GUI manifest
   │   ├── evals/                          # Evaluation run artifacts (separate from training)
   │   │   └── {run_id}/
   │   │       ├── tensorboard/            # Evaluation TensorBoard logs
   │   │       ├── videos/                 # Evaluation video captures
   │   │       ├── logs/
   │   │       │   ├── cleanrl.stdout.log
   │   │       │   └── cleanrl.stderr.log
   │   │       ├── analytics.json          # GUI manifest
   │   │       └── eval_summary.json       # Evaluation results
   │   ├── registry.db                     # SQLite state store
   │   ├── daemon.pid                      # Daemon PID file
   │   └── daemon.lock                     # Singleton lock
   ├── telemetry/
   │   └── telemetry.sqlite                # Durable telemetry
   └── logs/
       ├── gym_gui.log                     # Application log
       └── trainer_daemon.log              # Daemon log

Training runs and evaluation runs are stored in separate directories.
CleanRL policy evaluations (``extras.mode == "policy_eval"``) and XuanCe
test-mode runs (``test_mode == True``) are routed to ``evals/``. All other
runs go to ``runs/``. The run manager searches both directories for cleanup
and disk usage reporting.
