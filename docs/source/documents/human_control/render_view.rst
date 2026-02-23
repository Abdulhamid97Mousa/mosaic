Render View System
==================

The render view displays environment observations as visual output in
the MOSAIC GUI.  It is built on the **strategy pattern**: a
``RendererRegistry`` selects the correct ``RendererStrategy`` for the
current render mode, and the strategy converts raw observation data into
a Qt widget.

Strategy Pattern
----------------

RendererStrategy Protocol
^^^^^^^^^^^^^^^^^^^^^^^^^

Every renderer implements the ``RendererStrategy`` protocol defined in
``gym_gui.rendering.interfaces``:

- ``mode``: the ``RenderMode`` this strategy handles.
- ``widget``: the Qt widget that displays the rendered output.
- ``render(payload, context)``: draw a single frame from the given
  observation payload.
- ``supports(payload)``: return ``True`` if this strategy can handle
  the payload.
- ``reset()``: clear state between episodes.

RendererContext
^^^^^^^^^^^^^^

A ``RendererContext`` dataclass carries metadata needed by strategies:

- ``game_id``:  is  ``GameId`` enum identifying the current environment.
- ``square_size``:  pixel size per grid tile (used by the grid renderer).

RendererRegistry
^^^^^^^^^^^^^^^^

``RendererRegistry`` maps ``RenderMode`` values to strategy factory
functions:

- ``register(mode, factory)``: add a new mode-to-strategy binding.
- ``create(mode, parent)``: instantiate the strategy for the given mode.
- ``is_registered(mode)``: check whether a mode has a registered factory.
- ``supported_modes()``: list all registered modes.

The convenience function ``create_default_renderer_registry()``
pre-populates the registry with the ``GRID`` and ``RGB_ARRAY``
strategies.

Strategy Selection Flow
^^^^^^^^^^^^^^^^^^^^^^^

.. mermaid::

   graph LR
       OBS[Environment Observation] --> ADAPT[Adapter]
       ADAPT --> |payload dict| REG[RendererRegistry]
       REG --> |RenderMode.GRID| GRID[GridRendererStrategy]
       REG --> |RenderMode.RGB_ARRAY| RGB[RgbRendererStrategy]
       GRID --> |QGraphicsView| TAB[Render Tab]
       RGB --> |QPainter widget| TAB

       style OBS fill:#9370db,stroke:#6a0dad,color:#fff
       style ADAPT fill:#9370db,stroke:#6a0dad,color:#fff
       style REG fill:#ff7f50,stroke:#cc5500,color:#fff
       style GRID fill:#ff7f50,stroke:#cc5500,color:#fff
       style RGB fill:#ff7f50,stroke:#cc5500,color:#fff
       style TAB fill:#4a90d9,stroke:#2e5a87,color:#fff

Render Modes
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - RenderMode
     - Description
   * - ``ANSI``
     - Plain-text rendering (terminal style).
   * - ``ASCII``
     - Character-grid rendering.
   * - ``GRID``
     - Tile-based QGraphicsView rendering for toy-text environments.
   * - ``RGB_ARRAY``
     - Pixel-array rendering via QPainter for most visual environments.
   * - ``SURFACE``
     - Pygame surface passthrough (used internally by some environment
       wrappers).

RGB Renderer
------------

``RgbRendererStrategy`` (in ``gym_gui.rendering.strategies.rgb``)
handles the most common case: the environment returns a NumPy RGB array
and MOSAIC displays it as a scaled image.

Rendering Pipeline
^^^^^^^^^^^^^^^^^^

1. Receive a ``(H, W, 3)`` NumPy array from the adapter payload.
2. Convert to ``QImage`` (``Format_RGB888``).
3. Convert to ``QPixmap``.
4. Paint in ``paintEvent`` with aspect-ratio-preserving scaling using
   ``Qt.KeepAspectRatio``.

Mouse Interaction Modes
^^^^^^^^^^^^^^^^^^^^^^^

The RGB renderer supports three optional mouse interaction modes for
environments that need pointer input:

**FPS Mouse Capture (ViZDoom):** ``set_mouse_capture_enabled()`` locks the cursor to the render widget.
Mouse deltas are forwarded via ``set_mouse_delta_callback()`` for
first-person look control.  Press **Esc** to release the capture.

**Grid Click (Jumanji Tetris / Minesweeper):** ``set_grid_click_callback(rows, cols, grid_rect)`` overlays a logical
grid on the rendered image.  Clicks are translated to ``(row, col)``
coordinates and forwarded to the environment as discrete actions.

**Scroll Wheel (Tetris Rotation):** ``set_scroll_callback()`` forwards mouse wheel events, typically used
for piece rotation in puzzle games.

Grid Renderer
-------------

``GridRendererStrategy`` (in ``gym_gui.rendering.strategies.grid``)
renders toy-text environments (FrozenLake, Taxi, CliffWalking) as
tile-based maps using Qt's ``QGraphicsScene`` and ``QGraphicsView``.

Asset System
^^^^^^^^^^^^

Each toy-text environment has a dedicated asset class that maps
observation states to tile pixmaps:

- ``FrozenLakeAssets``: ice, hole, start, goal, and elf sprites.
- ``TaxiAssets``: road grid, passenger, destination, taxi sprites.
- ``CliffWalkingAssets``: safe tiles, cliff tiles, agent sprite.

All asset classes use the ``AssetManager`` singleton, which provides
pixmap caching and lazy loading to avoid reloading sprites on every
frame.

Tile Layers
^^^^^^^^^^^

The grid renderer uses a two-layer system:

1. **Base layer**: static background tiles (ice, road, cliff edge).
2. **Overlay layer**: dynamic elements (agent position, items).

This allows the agent sprite to move over the background without
re-rendering static tiles.

Board Game Renderer
-------------------

``BoardGameRendererStrategy`` (in
``gym_gui.rendering.strategies.board_game``) provides interactive board
displays for turn-based games from PettingZoo Classic and OpenSpiel.

Supported Games
^^^^^^^^^^^^^^^

The ``SUPPORTED_GAMES`` frozenset includes:

- Chess
- Go
- Connect Four
- TicTacToe
- Checkers
- Sudoku

User Interaction Signals
^^^^^^^^^^^^^^^^^^^^^^^^^

Board game renderers emit Qt signals when the human player makes a move:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Signal
     - Trigger
   * - ``chess_move_made``
     - Player drags a piece to a valid square.
   * - ``connect_four_column_clicked``
     - Player clicks a column to drop a disc.
   * - ``go_intersection_clicked``
     - Player clicks a board intersection to place a stone.
   * - ``go_pass_requested``
     - Player clicks the pass button.

These signals are connected to the ``SessionController``, which
translates the board-level move into the environment's action encoding
and calls ``_apply_action()``.
