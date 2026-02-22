Slow Lane
=========

The slow lane is MOSAIC's **durable** rendering and telemetry path.  It carries
every step and episode event from workers through gRPC, a publish-subscribe bus,
and into SQLite: feeding both the live UI and persistent storage for replay and
analytics.  Where the :doc:`fastlane` optimises for latency, the slow lane
optimises for **completeness**: every event is persisted.

Pipeline Overview
-----------------

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       W["Worker Process<br/>(JSONL stdout)"]
       PROXY["TrainerTelemetryProxy<br/>JSONL → gRPC"]
       DAEMON["TrainerService<br/>(gRPC Daemon)"]
       BUS["RunBus<br/>pub-sub · queue 2 048"]
       LTC["LiveTelemetryController<br/>(background thread)"]
       CM["CreditManager<br/>200 credits/stream"]
       RSR["RenderingSpeedRegulator<br/>100 ms drain · queue 32"]
       LTT["LiveTelemetryTab"]
       SINK["TelemetryDBSink<br/>batch 256"]
       DB[("SQLite / WAL")]

       W --> PROXY --> DAEMON --> BUS
       BUS --> LTC --> CM --> RSR --> LTT
       BUS --> SINK --> DB

       style BUS fill:#fff3e0,stroke:#e65100,color:#333
       style DB fill:#e3f2fd,stroke:#1565c0,color:#333

All queue sizes above are governed by
:doc:`/documents/runtime_logging/constants`: see
``constants_telemetry.py`` and ``constants_telemetry_bus.py``.

Components
----------

TrainerTelemetryProxy
^^^^^^^^^^^^^^^^^^^^^

Lives in ``gym_gui/services/trainer/trainer_telemetry_proxy.py``.  Tails the
worker's JSONL stdout stream and translates each line into a gRPC
``PublishRunSteps`` / ``PublishRunEpisodes`` call on the daemon.  When
:doc:`fastlane` mode is active, the proxy also extracts RGB frames and writes
them to a ``FastLaneWriter``: bridging the two lanes.

RunBus
^^^^^^

An in-process publish-subscribe event bus (``gym_gui/telemetry/run_bus.py``)
that fans telemetry events to all subscribers.  Key topics:

- ``STEP_APPENDED``: a new step arrived.
- ``EPISODE_FINALIZED``: an episode completed.
- ``CONTROL``: pause / resume / stop commands.

Default queue size: ``RUNBUS_DEFAULT_QUEUE_SIZE = 2048``
(``RUNBUS_UI_PATH_QUEUE_SIZE = 512``, ``RUNBUS_DB_PATH_QUEUE_SIZE = 1024``).

LiveTelemetryController
^^^^^^^^^^^^^^^^^^^^^^^

A ``QObject`` (``gym_gui/controllers/live_telemetry_controllers.py``) that
subscribes to the ``RunBus`` on a **background thread** and emits Qt signals
on the main thread.

**Signals:**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Signal
     - Description
   * - ``run_tab_requested(run_id, agent_id, tab_name)``
     - First event for a new agent → create a :doc:`render_tabs`
       ``LiveTelemetryTab``.
   * - ``telemetry_stats_updated(run_id, stats)``
     - Aggregate stats changed (steps, episodes, mean reward).
   * - ``run_completed(run_id)``
     - Training run finished: clean up resources.

**Subscription lifecycle:**

.. code-block:: python

   # 1. Controller subscribes to a run
   controller.subscribe_to_run(run_id)
     → subscribe_to_runbus(run_id)
     → credit_manager.initialize_stream(run_id, "default")

   # 2. First step event triggers tab creation
   controller._process_step_queue()
     → detects first event → emits run_tab_requested

   # 3. Tab registers itself
   controller.register_tab(run_id, agent_id, tab)
     → grant_credits(run_id, agent_id, 200)
     → flush buffered steps/episodes

Queue sizes: ``LIVE_STEP_QUEUE_SIZE = 64``,
``LIVE_EPISODE_QUEUE_SIZE = 64``, ``LIVE_CONTROL_QUEUE_SIZE = 32``.

CreditManager
^^^^^^^^^^^^^

Credit-based backpressure (``gym_gui/telemetry/credit_manager.py``) prevents
the bus from overwhelming the UI thread.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Method
     - Behaviour
   * - ``initialize_stream(run_id, agent_id)``
     - Sets initial credits to ``INITIAL_CREDITS`` (200).
   * - ``consume_credit(run_id, agent_id) → bool``
     - Returns ``True`` and decrements if credit > 0.  Returns ``False`` and
       increments the drop counter otherwise.
   * - ``grant_credits(run_id, agent_id, amount)``
     - Grants credits up to ``initial × 2`` cap.

When credits reach zero the producer pauses the UI rendering path: the
:doc:`/documents/runtime_logging/constants` ``CreditDefaults``
(``initial_credits=200``, ``starvation_threshold=10``) control the thresholds.
Database writes via ``TelemetryDBSink`` are **never** throttled.

RenderingSpeedRegulator
^^^^^^^^^^^^^^^^^^^^^^^

A ``QObject`` (``gym_gui/telemetry/rendering_speed_regulator.py``) that
decouples visual frame rendering from table/telemetry updates.

- Maintains a bounded ``deque`` of render payloads (default max
  ``RENDER_QUEUE_SIZE = 32``).
- Drains at a configurable interval (default ``100 ms`` → ~10 FPS).
- Auto-drops oldest payloads when the queue is full: the GUI always shows
  the freshest available frame.
- Emits ``payload_ready(dict)`` when a frame should be painted.
- Buffers early payloads submitted before ``start()`` is called;
  an auto-start timer (``RENDER_BOOTSTRAP_TIMEOUT_MS = 500 ms``) ensures the
  regulator eventually starts even if no explicit ``start()`` call arrives.

.. code-block:: python

   regulator = RenderingSpeedRegulator(render_delay_ms=100)
   regulator.payload_ready.connect(tab.render_payload)
   regulator.start()
   # Workers submit at arbitrary rate:
   regulator.submit_payload({"rgb": frame_bytes, "reward": 0.5})

Human-Control Path
------------------

When a human plays, the slow lane takes a shorter route:

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       HIC["HumanInputController<br/>keyboard shortcuts"]
       SC["SessionController"]
       TEL["TelemetryService"]
       STOR["StorageRecorderService<br/>JSONL ring"]
       SQL["TelemetrySQLiteStore"]
       DB[("SQLite / WAL")]

       HIC --> SC --> TEL
       TEL --> STOR
       TEL --> SQL --> DB

       style HIC fill:#9370db,stroke:#6a0dad,color:#fff
       style DB fill:#e3f2fd,stroke:#1565c0,color:#333

``HumanInputController`` captures keyboard events, ``SessionController`` emits
``StepRecord`` objects, and ``TelemetryService`` fans them to both the JSONL
ring buffer and SQLite for durable persistence.  The
:doc:`/documents/runtime_logging/log_constants` codes ``LOG401``–``LOG407``
trace session lifecycle and human input events.

Agent-Control Path
------------------

Remote agents stream JSONL through ``trainer_telemetry_proxy.py``, which
calls ``TrainerService.PublishRunSteps`` / ``PublishRunEpisodes``.
``TrainerService`` fans events onto ``RunBus``; ``TelemetryDBSink`` drains
the bus into ``TelemetrySQLiteStore`` with batch writes
(``DB_SINK_BATCH_SIZE = 256``, ``DB_SINK_CHECKPOINT_INTERVAL = 4096``).

Design Principles
-----------------

- **WAL + batching**: SQLite's WAL mode with large batch writes keeps the slow
  lane efficient under high-frequency telemetry.
- **Hot vs cold storage**: ``LiveTelemetryTab`` shows hot data from ``RunBus``;
  SQLite provides cold storage for replay and post-hoc analysis.
- **The GUI never blocks**: all writes are asynchronous.  Credit-based
  backpressure ensures the UI thread stays responsive.
- **Complementary to the fast lane**: the :doc:`fastlane` gives real-time
  visuals while the slow lane guarantees every event is persisted for
  :doc:`/documents/architecture/overview` analytics (W&B, TensorBoard).

See Also
--------

- :doc:`fastlane`: the zero-serialisation rendering path for live training.
- :doc:`render_tabs`: ``LiveTelemetryTab`` is the slow-lane widget that
  receives regulated payloads.
- :doc:`strategies`: the ``RendererRegistry`` decides how each slow-lane
  frame is painted.
- :doc:`/documents/runtime_logging/constants`: all queue sizes, batch sizes,
  and credit thresholds live in the constants package.
