Training Views
==============

Individual view widgets that are hosted inside :doc:`render_tabs`.  Each view
handles one specific rendering concern: high-speed live frames, slow-lane
telemetry, single-operator rendering, or side-by-side multi-agent evaluation.

LiveTelemetryTab
----------------

``LiveTelemetryTab`` (``gym_gui/ui/widgets/live_telemetry_tab.py``) displays
a live :doc:`slow_lane` telemetry stream for a specific
``(run_id, agent_id)`` pair.

.. code-block:: python

   LiveTelemetryTab(
       run_id: str,
       agent_id: str,
       *,
       game_id: GameId | None = None,
       buffer_size: int = 100,            # DEFAULT_TELEMETRY_BUFFER_SIZE
       episode_buffer_size: int = 10,     # DEFAULT_EPISODE_BUFFER_SIZE
       render_throttle_interval: int = 1, # UI_RENDERING_THROTTLE_MIN
       render_delay_ms: int = 100,        # DEFAULT_RENDER_DELAY_MS
       live_render_enabled: bool = True,
       renderer_registry: RendererRegistry | None = None,
       parent: QWidget | None = None,
   )

**Data flow:**

1. ``LiveTelemetryController`` (see :doc:`slow_lane`) emits step/episode events.
2. Events are queued in bounded ``Deque`` buffers (oldest dropped on overflow):
   ``_step_buffer`` (maxlen=100), ``_episode_buffer`` (maxlen=10).
3. ``RenderingSpeedRegulator`` drains the render queue at ~10 FPS
   (``render_delay_ms=100``).
4. The tab delegates to a :doc:`strategies` ``RendererStrategy`` (resolved via
   ``RendererRegistry``) for the visual frame, and updates a stats table
   independently.
5. ``_render_throttle_interval`` controls which steps trigger a render
   (every *N*-th step).

FastLaneTab
-----------

``FastLaneTab`` (``gym_gui/ui/widgets/fastlane_tab.py``) renders frames from
the :doc:`fastlane` shared-memory ring buffer using **Qt Quick / QML**.

.. code-block:: python

   FastLaneTab(
       run_id: str,
       agent_id: str,
       *,
       mode_label: str | None = None,   # default "Fast lane"
       run_mode: str | None = None,     # "train" | "policy_eval"
       parent: QWidget | None = None,
   )

**Architecture:**

- Owns a ``FastLaneConsumer`` that polls shared memory every 16 ms (~60 Hz).
- ``FastLaneConsumer`` emits ``frame_ready(FastLaneFrameEvent)`` with a
  ``QImage`` + HUD text.
- The tab hosts a ``QQuickWidget`` loading ``FastLaneView.qml`` for
  GPU-accelerated rendering.
- A status label shows connection state (``connected``, ``reconnecting``,
  ``fastlane-unavailable``).

**Modes:**

- ``"train"`` (default): live training display with reward / step-rate HUD.
- ``"policy_eval"``: adds an evaluation summary overlay that reloads
  ``eval_summary.json`` every 1 s (batch count, episodes, avg/min/max return).

OperatorRenderContainer
-----------------------

``OperatorRenderContainer``
(``gym_gui/ui/widgets/operator_render_container.py``) wraps a single
:doc:`/documents/architecture/operators/index` rendering area during
multi-operator evaluation.

.. code-block:: python

   OperatorRenderContainer(
       config: OperatorConfig,
       renderer_registry: RendererRegistry | None = None,
       parent: QWidget | None = None,
   )

**Layout:**

- **Header**: operator name + type badge (``llm`` / ``vlm`` / ``rl`` /
  ``human``) with status-coloured indicator (pending / loaded / running /
  stopped / error).
- **Render area**: hosts a :doc:`strategies` ``RendererStrategy``
  (Grid, RGB, or BoardGame).
- **Stats bar**: compact step / episode / reward counters.
- **Human interaction**: detected via
  ``operator_type == "human" or any(worker.type == "human")``.

**Signals:**

- ``status_changed(str, str)``: operator_id, new_status
- ``human_action_submitted(str, int)``: operator_id, action_index
- ``board_game_move_made(str, str, str)``: operator_id, from_sq, to_sq
- ``chess_move_button_clicked(str, str)``: operator_id, uci_move

MultiOperatorRenderView
-----------------------

``MultiOperatorRenderView``
(``gym_gui/ui/widgets/multi_operator_render_view.py``) arranges *N*
``OperatorRenderContainer`` widgets in a dynamic grid layout:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Operators
     - Auto-layout
     - Manual override
   * - 1
     - Full width (1 col)
     - "Auto", "1 Column"
   * - 2
     - 1 x 2
     - "2 Columns"
   * - 3 to 4
     - 2 x 2
     - "3 Columns"
   * - 5 to 6
     - 2 x 3
     - (auto or override)
   * - 7 to 9
     - 3 x 3
     - (auto or override)
   * - 10+
     - 4 columns
     - (auto or override)

This is the primary view for side-by-side
:doc:`/documents/architecture/paradigms` comparison during evaluation: e.g.,
watching an RL :doc:`/documents/architecture/operators/index` and an LLM
operator solve the same environment simultaneously.

**Signals forwarded:**

- ``operator_status_changed(str, str)``
- ``human_action_submitted(str, int)``
- ``board_game_move_made(str, str, str)``
- ``chess_move_button_clicked(str, str)``

See Also
--------

- :doc:`render_tabs`: the ``RenderTabs`` QTabWidget that hosts all views, and
  the worker presenters that create them per run.
- :doc:`fastlane`: ``FastLaneConsumer`` and shared-memory ring buffer
  details used by ``FastLaneTab``.
- :doc:`slow_lane`: ``LiveTelemetryController`` and ``RenderingSpeedRegulator``
  that feed ``LiveTelemetryTab``.
- :doc:`strategies`: the ``RendererStrategy`` protocol and ``RendererRegistry``
  used by both ``LiveTelemetryTab`` and ``OperatorRenderContainer``.
- :doc:`/documents/architecture/operators/index`: the operator system that
  populates ``MultiOperatorRenderView``.
