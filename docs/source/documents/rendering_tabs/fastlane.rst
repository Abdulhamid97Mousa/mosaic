Fast Lane
=========

The fast lane is MOSAIC's **zero-serialisation** rendering path.  It streams
live RGB frames from a training worker directly into shared memory, bypassing
the :doc:`slow_lane` gRPC/SQLite pipeline entirely.  The GUI-side
``FastLaneConsumer`` polls the buffer every 16 ms (~60 Hz) and hands the
latest frame to a :doc:`render_tabs` ``FastLaneTab`` for Qt Quick display.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       W["Worker Process"] -->|"publish(frame)"| FLW["FastLaneWriter"]
       FLW -->|"shared memory"| SHM[("SPSC Ring Buffer<br/>magic FLAN · seqlock")]
       SHM -->|"latest_frame()"| FLR["FastLaneReader"]
       FLR --> FLC["FastLaneConsumer<br/>QTimer 16 ms"]
       FLC -->|"frame_ready signal"| FLT["FastLaneTab<br/>QQuickWidget · QML"]

       style SHM fill:#e8f5e9,stroke:#2e8b57,color:#333

SPSC Ring Buffer
----------------

The core is a **Single-Producer, Single-Consumer ring buffer** in shared
memory, inspired by the LMAX Disruptor pattern.  Implementation lives in
``gym_gui/fastlane/buffer.py``.

Data Classes
^^^^^^^^^^^^

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Class
     - Fields
   * - ``FastLaneConfig``
     - ``width``, ``height``, ``channels=3``, ``pixel_format="RGB"``,
       ``capacity=128``, ``metadata_size=0``
   * - ``FastLaneFrame``
     - ``data: bytes``, ``width``, ``height``, ``channels``,
       ``metrics: FastLaneMetrics``, ``metadata: bytes | None``
   * - ``FastLaneMetrics``
     - ``last_reward: float``, ``rolling_return: float``,
       ``step_rate_hz: float``

All three are frozen dataclasses.

Binary Layout
^^^^^^^^^^^^^

.. code-block:: text

   ┌──────────────────── Header (72 bytes) ──────────────────┐
   │ MAGIC(4s) VERSION(I) WIDTH(I) HEIGHT(I) CHANNELS(I)     │
   │ PIXEL_FMT(I) CAPACITY(I) SLOT_SIZE(I) META_SIZE(I)      │
   │ FLAGS(I) HEAD(Q) TAIL(Q)                                 │
   │ LAST_REWARD(d) ROLLING_RETURN(d) STEP_RATE_HZ(d)        │
   └──────────────────────────────────────────────────────────┘
   ┌── Slot 0 ──┐  ┌── Slot 1 ──┐  ...  ┌── Slot N-1 ──┐
   │ seq(Q)      │  │ seq(Q)      │       │ seq(Q)        │
   │ payload_len │  │ payload_len │       │ payload_len   │
   │ reserved    │  │ reserved    │       │ reserved      │
   │ [RGB bytes] │  │ [RGB bytes] │       │ [RGB bytes]   │
   └─────────────┘  └─────────────┘       └───────────────┘

Key constants:

- ``MAGIC = b"FLAN"``  ·  ``VERSION = 1``
- ``_HEADER_SIZE = 72``  (struct ``< 4sIIIIIIIIIQQddd``)
- ``_SLOT_META_SIZE = 16``  (struct ``< QII``)
- ``FLAG_INVALIDATED = 0x1``

Seqlock Mechanism
^^^^^^^^^^^^^^^^^

The writer and reader coordinate without locks using a **seqlock** protocol:

1. **Write path**: ``FastLaneWriter.publish(frame, *, metrics, metadata) → int``:

   - Computes ``slot = head % capacity``.
   - Writes ``seq = head * 2`` into the slot header (*odd* = in-flight).
   - Copies RGB payload bytes.
   - Writes ``seq = head * 2 + 2`` (*even* = committed).
   - Atomically advances ``head`` in the shared header.

2. **Read path**: ``FastLaneReader.latest_frame() → FastLaneFrame | None``:

   - Reads ``seq1`` from the slot header.
   - If ``seq1 % 2 == 1`` → write in progress, skip.
   - Copies the payload bytes.
   - Reads ``seq2`` and verifies ``seq1 == seq2`` → data is consistent.
   - On mismatch → retry or return ``None``.

3. **Metrics path**: ``FastLaneReader.metrics() → FastLaneMetrics``:
   reads ``last_reward``, ``rolling_return``, ``step_rate_hz`` directly from
   the header doubles.

Factory Methods
^^^^^^^^^^^^^^^

.. code-block:: python

   # Worker side
   writer = FastLaneWriter.create(
       run_id,
       FastLaneConfig(width=84, height=84, channels=3, capacity=128),
   )
   seq = writer.publish(frame_bytes, metrics=FastLaneMetrics(...))

   # GUI side
   reader = FastLaneReader.attach(run_id)
   frame  = reader.latest_frame()

Design Rules
^^^^^^^^^^^^

1. **SPSC only**: one writer, one reader, no mutexes.
2. **Lossy**: the consumer always jumps to the latest sequence; old frames are
   silently overwritten.
3. **Batch-friendly**: no frame debt; the reader skips ahead.
4. **Simple payload**: tight-packed RGB(A) bytes; HUD scalars in the header.
5. **Invalidation**: ``FLAG_INVALIDATED`` tells the reader that the writer
   has exited and the buffer should be re-attached.

Frame Tiling
------------

When a worker uses vectorized environments,
``tile_frames(frames: Sequence[np.ndarray]) → np.ndarray`` composites *N*
sub-environment frames into a near-square grid (``rows = ceil(sqrt(N))``,
``cols = ceil(N / rows)``).  This mirrors Stable-Baselines3 ``VecEnv`` tiling
and allows streaming multiple environments in a single fast-lane slot.

Worker Integration Helpers
--------------------------

``apply_fastlane_environment()`` injects canonical environment variables into
a worker's subprocess launch dict:

.. list-table::
   :widths: 38 62
   :header-rows: 1

   * - Environment Variable
     - Description
   * - ``MOSAIC_FASTLANE_ONLY``
     - ``"1"`` or ``"0"``: skip :doc:`slow_lane` telemetry
   * - ``MOSAIC_FASTLANE_SLOT``
     - Which vectorized-env index feeds the writer
   * - ``MOSAIC_FASTLANE_VIDEO_MODE``
     - ``"single"`` | ``"grid"`` | ``"off"``
   * - ``MOSAIC_FASTLANE_GRID_LIMIT``
     - Max environments composited in grid mode (default 4)

.. code-block:: python

   def apply_fastlane_environment(
       env: Dict[str, Any],
       *,
       fastlane_only: bool,
       fastlane_slot: int,
       video_mode: str = "SINGLE",
       grid_limit: int = 4,
   ) -> Dict[str, Any]: ...

FastLaneConsumer
----------------

``FastLaneConsumer`` (``gym_gui/ui/fastlane_consumer.py``) is a ``QObject``
that bridges shared memory to Qt signals.

**Polling loop**: a ``QTimer`` fires every **16 ms**:

1. If not connected → attempt ``FastLaneReader.attach(run_id)``.
2. Check ``FLAG_INVALIDATED`` → trigger reconnection.
3. Validate header (``capacity > 0``, ``slot_size > 0``).
4. Read ``reader.latest_frame()`` → convert bytes to ``QImage``
   (``Format_RGB888`` or ``Format_RGBA8888``).
5. Emit ``frame_ready(FastLaneFrameEvent)`` with the ``QImage`` and a
   HUD string: ``"reward: {:.2f}\nreturn: {:.2f}\nstep/sec: {:.1f}"``.

**Signals:**

- ``frame_ready(FastLaneFrameEvent)``: image + HUD text + optional metadata.
- ``status_changed(str)``: ``"connected"`` | ``"reconnecting"`` |
  ``"fastlane-unavailable"``.

FastLaneTab
-----------

``FastLaneTab`` (``gym_gui/ui/widgets/fastlane_tab.py``) hosts a
``QQuickWidget`` loading ``FastLaneView.qml`` for GPU-accelerated rendering.
See :doc:`render_tabs` for how it plugs into the central tab widget.

.. code-block:: python

   FastLaneTab(
       run_id: str,
       agent_id: str,
       *,
       mode_label: str | None = None,   # default "Fast lane"
       run_mode: str | None = None,     # "train" | "policy_eval"
       parent: QWidget | None = None,
   )

**Modes:**

- ``"train"`` (default): live frames + reward / step-rate HUD.
- ``"policy_eval"``: adds an evaluation summary overlay that reloads
  ``eval_summary.json`` every 1 s (batch count, episodes, avg/min/max return).

Directory Layout
----------------

.. code-block:: text

   gym_gui/
     fastlane/
       __init__.py           # Public API re-exports
       buffer.py             # SPSC shared-memory ring buffer
       tiling.py             # tile_frames() for multi-env compositing
       worker_helpers.py     # apply_fastlane_environment()
     ui/
       fastlane_consumer.py  # FastLaneConsumer (QTimer → QImage)
       widgets/
         fastlane_tab.py     # FastLaneTab (QQuickWidget host)

See Also
--------

- :doc:`slow_lane`: the durable gRPC/SQLite telemetry path that complements
  the fast lane.
- :doc:`render_tabs`: ``FastLaneTab`` is dynamically added to ``RenderTabs``
  by worker presenters.
- :doc:`/documents/architecture/workers/index`: the worker subprocess layer
  that produces fast-lane frames.
- :doc:`/documents/architecture/workers/integrated_workers/CleanRL_Worker/index`
 : CleanRL's ``FastLaneTelemetryWrapper`` integration.
- :doc:`/documents/runtime_logging/constants`: ``RenderDefaults`` and
  ``BufferDefaults`` for queue-size tuning.
