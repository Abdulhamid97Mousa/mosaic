Rendering Strategies
====================

MOSAIC uses a **Strategy + Registry** pattern to render different payload types
(pixel frames, grid worlds, board games) through a single interface.
Strategies are resolved at runtime by :doc:`render_tabs` and
:doc:`slow_lane` ``LiveTelemetryTab``, which delegate all visual painting to
whichever strategy matches the current payload.

.. mermaid::

   %%{init: {"flowchart": {"curve": "linear"}} }%%
   graph LR
       LTT["LiveTelemetryTab<br/>(slow lane)"]
       REG["RendererRegistry"]
       RGB["RgbRendererStrategy<br/>mouse capture · scroll · grid-click"]
       GRID["GridRendererStrategy<br/>FrozenLake · Taxi · CliffWalking"]
       BG["BoardGameRendererStrategy<br/>Chess · Go · Connect4 · Sudoku · Checkers · TicTacToe"]

       LTT -->|"create(mode)"| REG
       REG -->|"RGB_ARRAY"| RGB
       REG -->|"GRID"| GRID
       REG -->|"GRID (board)"| BG

       style REG fill:#fff3e0,stroke:#e65100,color:#333

RendererStrategy Protocol
-------------------------

Every rendering strategy implements the ``RendererStrategy`` protocol
(``gym_gui/rendering/interfaces.py``):

.. code-block:: python

   class RendererStrategy(Protocol):
       mode: RenderMode

       @property
       def widget(self) -> QWidget: ...

       def render(self, payload: Mapping[str, object],
                  *, context: RendererContext | None = None) -> None: ...

       def supports(self, payload: Mapping[str, object]) -> bool: ...

       def reset(self) -> None: ...

The companion ``RendererContext`` dataclass carries optional metadata:

.. code-block:: python

   @dataclass(slots=True)
   class RendererContext:
       game_id: GameId | None = None
       square_size: int | None = None   # tile / square display size

``RenderMode`` (from ``gym_gui/core/enums.py``) has two values:
``RGB_ARRAY`` and ``GRID``.

RendererRegistry
----------------

The ``RendererRegistry`` (``gym_gui/rendering/registry.py``) maps
``RenderMode`` values to factory callables.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Method
     - Description
   * - ``register(mode, factory)``
     - Binds a ``RenderMode`` to a ``Callable[..., RendererStrategy]``.
   * - ``create(mode, parent)``
     - Instantiates the strategy registered for *mode*.
   * - ``is_registered(mode)``
     - Check if a mode has a factory.
   * - ``supported_modes()``
     - Returns all registered ``RenderMode`` values.

``create_default_renderer_registry()`` pre-populates two built-in modes:

.. list-table::
   :widths: 22 25 53
   :header-rows: 1

   * - RenderMode
     - Strategy
     - Description
   * - ``RGB_ARRAY``
     - ``RgbRendererStrategy``
     - Pixel-based environments (Atari, MiniGrid, Crafter, ViZDoom, …)
   * - ``GRID``
     - ``GridRendererStrategy``
     - Text/grid environments (FrozenLake, Taxi, CliffWalking)

``BoardGameRendererStrategy`` shares the ``GRID`` mode but is instantiated
directly by :doc:`render_tabs` when the payload contains board-game keys
(``fen``, ``connect_four``, ``go_board``, etc.).

RgbRendererStrategy
-------------------

Renders RGB array payloads into a scrollable Qt widget (``_RgbView``) with
aspect-ratio preservation.

**Key features:**

Mouse capture (FPS-style)
   For environments requiring continuous mouse input (e.g., ViZDoom).
   ``set_mouse_delta_callback(callback)`` delivers ``(delta_x, delta_y)``
   in degrees; ESC releases capture.  ``set_mouse_delta_scale(scale)``
   controls degrees per pixel (default ``0.5``).

   Signal: ``mouse_capture_changed(bool)`` on the internal ``_RgbView``.

Discrete turn actions
   ``set_mouse_action_callback(callback)`` accumulates mouse deltas and
   triggers ``turn_left`` / ``turn_right`` actions at a threshold.
   ``set_turn_action_indices(turn_left, turn_right)`` configures the
   action indices.

Grid-click mode
   ``set_grid_click_callback(callback, rows, cols, grid_rect)`` normalises
   click coordinates to ``(row, col)`` within a defined rectangle: used
   for Tetris, Minesweeper, and similar tile-action environments.

Scroll-wheel support
   ``set_scroll_callback(callback)`` maps wheel events to discrete actions.

Tooltip overlay
   Overlays environment-specific metadata on hover via the render payload.

GridRendererStrategy
--------------------

Renders grid-based environments using ``QGraphicsScene`` with tiled sprite
assets.  The internal ``_GridRenderer`` draws at **120 px per tile** and
auto-detects the game type from ``RendererContext.game_id`` or the payload's
``game_id`` field, defaulting to ``GameId.FROZEN_LAKE``.

**Supported environments:**

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Environment
     - Rendering approach
   * - **FrozenLake**
     - Ice, hole, cracked-hole, goal, and stool tiles with directional elf
       agent sprites.
   * - **Taxi**
     - Cab sprites (front / rear / left / right), depot colours
       (R / G / Y / B), passenger state, median walls between cells.
   * - **CliffWalking**
     - Mountain background, cliff / stool / cookie overlays with layered
       compositing and directional actor sprites.

**Asset management**: ``AssetManager`` singleton
(``gym_gui/rendering/assets.py``) caches ``QPixmap`` objects from
``gym_gui/assets/toy_text_images/``.  Per-game asset classes
(``FrozenLakeAssets``, ``TaxiAssets``, ``CliffWalkingAssets``) enumerate
every sprite.

BoardGameRendererStrategy
-------------------------

A unified interactive renderer for turn-based board games.  Internally wraps
a ``_BoardGameWidget`` (``QStackedWidget``) that delegates to game-specific
renderers.

**Game detection** (``_detect_game``):

1. Check ``context.game_id`` (if provided).
2. Check payload ``"game_id"`` field (``GameId`` enum or string).
3. Fall back to payload-key heuristics: ``"chess"`` / ``"fen"`` → Chess,
   ``"connect_four"`` → Connect Four, ``"go"`` / ``"go_board"`` → Go,
   ``"sudoku"`` → Sudoku, ``"checkers"`` / ``"draughts"`` → Checkers.

**Supported games and signals:**

.. list-table::
   :widths: 16 20 64
   :header-rows: 1

   * - Game
     - Payload key
     - Interaction signals
   * - **Chess**
     - ``fen``
     - ``chess_move_made(from_sq: str, to_sq: str)``
   * - **Connect Four**
     - ``connect_four``
     - ``connect_four_column_clicked(col: int)``
   * - **Go**
     - ``go_board``
     - ``go_intersection_clicked(row, col)``,
       ``go_pass_requested()``
   * - **Tic-Tac-Toe**
     - ``tictactoe``
     - ``tictactoe_cell_clicked(row, col)``
   * - **Sudoku**
     - ``sudoku``
     - ``sudoku_cell_selected(row, col)``,
       ``sudoku_digit_entered(row, col, digit)``,
       ``sudoku_cell_cleared(row, col)``
   * - **Checkers**
     - ``checkers``
     - ``checkers_cell_clicked(row, col)``

**Internal renderers**: ``_ChessBoardRenderer``, ``_ConnectFourBoardRenderer``,
``_GoBoardRenderer``, ``_TicTacToeBoardRenderer``, ``_SudokuBoardRenderer``,
``_CheckersBoardRenderer``.  Each paints its own ``QWidget`` and emits
game-specific signals that are forwarded through :doc:`render_tabs` so
controllers can react to player moves without knowing which strategy is active.

Directory Layout
----------------

.. code-block:: text

   gym_gui/
     rendering/
       __init__.py
       interfaces.py       # RendererContext, RendererStrategy protocol
       registry.py          # RendererRegistry factory
       assets.py            # AssetManager singleton (QPixmap cache)
       strategies/
         rgb.py             # RgbRendererStrategy + _RgbView widget
         grid.py            # GridRendererStrategy + _GridRenderer
         board_game.py      # BoardGameRendererStrategy (6 games)
     assets/
       toy_text_images/     # Sprite sheets for grid environments

See Also
--------

- :doc:`render_tabs`: ``RenderTabs`` hosts the strategy widgets and forwards
  all board-game signals to controllers.
- :doc:`fastlane`: the fast lane bypasses strategies entirely and renders
  raw RGB via Qt Quick / QML.
- :doc:`/documents/runtime_logging/constants`: ``RenderDefaults``
  (queue sizes, FPS caps) and ``SliderDefaults`` for render-delay sliders.
