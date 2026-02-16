IPC Architecture
================

Workers communicate with the MOSAIC core through a layered IPC
(Inter-Process Communication) architecture built on **gRPC** and
**JSONL over stdout**.

Three-Tier Process Model
------------------------

.. mermaid::

   graph TB
       subgraph Tier1["Tier 1 · Main Process"]
           GUI["Qt6 GUI"]
           TC["TrainerClient<br/>(gRPC async client)"]
           GUI --> TC
       end

       subgraph Tier2["Tier 2 · Daemon Process"]
           TS["TrainerService<br/>(gRPC server @ 127.0.0.1:50055)"]
           DISP["TrainerDispatcher<br/>(subprocess manager)"]
           REG["RunRegistry<br/>(SQLite state store)"]
           REB["RunEventBroadcaster<br/>(fan-out queues)"]
           RTB["RunTelemetryBroadcaster<br/>(step/episode fan-out)"]
           TS --> DISP
           TS --> REG
           TS --> REB
           TS --> RTB
           DISP --> REG
       end

       subgraph Tier3["Tier 3 · Worker Processes"]
           PROXY["Telemetry Proxy<br/>(sidecar)"]
           WORKER["Worker Process<br/>(e.g. cleanrl_worker)"]
           PROXY -->|"tail stdout"| WORKER
       end

       TC -- "gRPC over HTTP/2" --> TS
       DISP -- "spawn" --> PROXY
       PROXY -- "spawn" --> WORKER
       PROXY -- "PublishRunSteps<br/>PublishRunEpisodes" --> TS
       RTB -- "StreamRunSteps" --> TC

       style Tier1 fill:#e3f2fd,stroke:#1565c0
       style Tier2 fill:#e8f5e9,stroke:#2e7d32
       style Tier3 fill:#fff3e0,stroke:#e65100

Process Hierarchy
~~~~~~~~~~~~~~~~~

.. code-block:: text

   MOSAIC GUI (PID 12345)
   └── Trainer Daemon (PID 12346)          ← spawned at startup
       ├── Telemetry Proxy (PID 12348)     ← spawned per job
       │   └── Worker Process (PID 12349)
       │       └── CleanRL PPO (in-process)
       └── Telemetry Proxy (PID 12350)     ← another job
           └── Worker Process (PID 12351)

Each worker is launched with ``os.setsid()`` to create a new process
group, allowing the entire tree to be killed with a single ``SIGTERM``.

gRPC Protocol
-------------

MOSAIC uses gRPC (Protocol Buffers over HTTP/2) for all structured
communication between the GUI, Daemon, and workers.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Aspect
     - Value
   * - Transport
     - HTTP/2 over TCP (loopback: ``127.0.0.1:50055``)
   * - Serialization
     - Protocol Buffers v5.27.2
   * - Max Message
     - 64 MB send/receive
   * - Keepalive
     - HTTP/2 pings every 30 s
   * - Security
     - Insecure (loopback only)

RPC Methods
~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
       participant GUI
       participant Daemon
       participant Proxy as Telemetry Proxy
       participant Worker

       GUI->>Daemon: SubmitRun(config_json)
       Daemon-->>GUI: run_id

       Note over Daemon: Status: INIT

       Daemon->>Proxy: spawn subprocess
       Proxy->>Worker: spawn subprocess

       Worker->>Daemon: RegisterWorker(capabilities)
       Daemon-->>Worker: session_token

       Note over Daemon: Status: HANDSHAKE → READY

       Worker->>Proxy: JSONL to stdout
       Proxy->>Daemon: PublishRunSteps(stream)

       Note over Daemon: Status: READY → EXECUTING

       Daemon->>GUI: StreamRunSteps(stream)

       Note over Worker: Training loop...

       Worker->>Proxy: run_completed event
       Proxy->>Daemon: Close stream

       Note over Daemon: EXECUTING → TERMINATED
       Daemon->>GUI: RunUpdate(TERMINATED)

The full RPC surface:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - RPC Method
     - Type
     - Description
   * - ``SubmitRun(config_json)``
     - Unary
     - Submit a new training run; returns ``run_id``
   * - ``CancelRun(run_id)``
     - Unary
     - Cancel a running job
   * - ``ListRuns(filter)``
     - Unary
     - Query runs by status
   * - ``WatchRuns()``
     - Server stream
     - Live stream of run status changes
   * - ``RegisterWorker(capabilities)``
     - Unary
     - Worker handshake; returns ``session_token``
   * - ``PublishRunSteps(stream)``
     - Client stream
     - Worker → Daemon step telemetry
   * - ``PublishRunEpisodes(stream)``
     - Client stream
     - Worker → Daemon episode summaries
   * - ``StreamRunSteps(run_id, since_seq)``
     - Server stream
     - Daemon → GUI live step data
   * - ``StreamRunEpisodes(run_id, since_seq)``
     - Server stream
     - Daemon → GUI live episode data
   * - ``Heartbeat(run_id)``
     - Unary
     - Worker liveness signal (every 60 s)

Telemetry Pipeline
------------------

Workers emit telemetry as **newline-delimited JSON (JSONL)** to
``stdout``.  This is the simplest possible integration: any script that
can ``print()`` can become a MOSAIC worker.

.. mermaid::

   graph LR
       W["Worker<br/>print(json)"] -->|"stdout"| P["Telemetry Proxy<br/>(JsonlTailer)"]
       P -->|"parse + validate"| PB["Protobuf<br/>RunStep / RunEpisode"]
       PB -->|"gRPC stream"| D["Daemon"]
       D -->|"persist"| SQL["SQLite"]
       D -->|"fan-out"| BUS["RunBus"]
       BUS -->|"gRPC stream"| GUI["Qt6 GUI"]

       style W fill:#ff7f50,stroke:#cc5500,color:#fff
       style P fill:#ffd700,stroke:#b8860b
       style D fill:#50c878,stroke:#2e8b57,color:#fff
       style GUI fill:#4a90d9,stroke:#2e5a87,color:#fff

JSONL Event Types
~~~~~~~~~~~~~~~~~

**Step event** — emitted every environment step:

.. code-block:: json

   {
     "event_type": "step",
     "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
     "episode": 0,
     "step_index": 42,
     "action": 1,
     "observation": [0.02, -0.01, 0.03, -0.02],
     "reward": 1.0,
     "terminated": false,
     "truncated": false
   }

**Episode event** — emitted when an episode ends:

.. code-block:: json

   {
     "event_type": "episode",
     "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
     "episode": 0,
     "total_reward": 195.0,
     "steps": 195,
     "terminated": true,
     "truncated": false
   }

**Lifecycle event** — emitted at run boundaries:

.. code-block:: json

   {
     "event": "run_started",
     "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
     "payload": {
       "worker_id": "worker-001",
       "env_id": "CartPole-v1",
       "algo": "ppo"
     }
   }

The Telemetry Proxy
~~~~~~~~~~~~~~~~~~~

Workers never speak gRPC directly.  A **Telemetry Proxy** sidecar
(spawned by the Daemon alongside the worker) reads ``stdout``,
parses each JSON line, validates it against versioned schemas,
converts it to a Protocol Buffer message, and streams it to the
Daemon.

This sidecar architecture provides two benefits:

1. **Workers stay simple** — no gRPC client code, no protobuf imports
2. **Fault isolation** — a malformed log line cannot crash the worker

Protobuf Messages
~~~~~~~~~~~~~~~~~

The Proxy translates JSONL into typed protobuf messages:

.. code-block:: protobuf

   message RunStep {
     string  run_id          = 1;
     uint64  episode_index   = 2;
     uint64  step_index      = 3;
     string  action_json     = 4;
     string  observation_json = 5;
     double  reward          = 6;
     bool    terminated      = 7;
     bool    truncated       = 8;
     string  agent_id        = 13;
     string  render_payload_json = 17;
     uint64  episode_seed    = 18;
     string  worker_id       = 19;
     uint64  seq_id          = 12;
   }

   message RunEpisode {
     string  run_id          = 1;
     uint64  episode_index   = 2;
     double  total_reward    = 3;
     uint64  steps           = 4;
     bool    terminated      = 5;
     bool    truncated       = 6;
     string  metadata_json   = 7;
     uint64  seq_id          = 9;
     string  agent_id        = 10;
     string  worker_id       = 11;
   }

Proto files are located at:

.. code-block:: text

   gym_gui/services/trainer/proto/
   ├── trainer.proto           # Protocol definition
   ├── trainer_pb2.py          # Generated Python code
   └── trainer_pb2_grpc.py     # Generated gRPC stubs

Reliability Mechanisms
----------------------

Heartbeats
~~~~~~~~~~

Workers send a ``Heartbeat`` RPC every 60 seconds.  If the Daemon
receives no heartbeat for 300 seconds (5 minutes), the run transitions
to ``FAULTED`` and GPU resources are released.

Backpressure
~~~~~~~~~~~~

A credit-based system prevents memory exhaustion when workers produce
telemetry faster than the GUI can render:

.. mermaid::

   graph LR
       W["Worker"] -->|"fast"| D["Daemon<br/>Credit Manager"]
       D -->|"stream"| G["GUI"]
       G -->|"refill credits<br/>after rendering"| D

       style W fill:#ff7f50,stroke:#cc5500,color:#fff
       style D fill:#50c878,stroke:#2e8b57,color:#fff
       style G fill:#4a90d9,stroke:#2e5a87,color:#fff

When credits are exhausted, the hub emits ``CONTROL STARVED`` and
slows ingestion.  Once the GUI renders queued frames, credits are
refilled and ``CONTROL RESUMED`` is emitted.

Reconnection
~~~~~~~~~~~~~

The GUI tracks the last received ``seq_id`` for each run.  On
reconnection, it requests ``StreamRunSteps(run_id, since_seq=last_seq)``
to resume from where it left off — no data is lost.

Performance
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Metric
     - Value
     - Notes
   * - gRPC latency
     - ~1–3 ms
     - Localhost unary call
   * - Telemetry throughput
     - 10 k+ steps/s
     - Limited by JSON parsing
   * - Queue depth
     - 4096 steps
     - Per-run circular buffer
   * - Max concurrent runs
     - 100+
     - Limited by GPU / memory
   * - SQLite write speed
     - ~5 k inserts/s
     - WAL mode enabled
   * - Max message size
     - 64 MB
     - gRPC send/receive limit
