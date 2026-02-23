Classic
=======

Turn-based board games using PettingZoo's AEC (Alternating Environment Cycle) API.
In MOSAIC, these run with ``SEQUENTIAL`` stepping — each agent observes, decides,
and acts before the turn passes to the next player.

:Install: ``pip install -e ".[pettingzoo]"``
:Paradigm: Multi-agent (turn-based)
:Stepping: ``SEQUENTIAL``
:Docs: `pettingzoo.farama.org/environments/classic/ <https://pettingzoo.farama.org/environments/classic/>`_

Chess
-----

.. image:: https://pettingzoo.farama.org/_images/classic_chess.gif
   :alt: Chess environment
   :width: 200px
   :align: right

Standard chess with two players.  MOSAIC provides a built-in Stockfish opponent
so you can play Human vs AI or pit RL agents against the engine.

- **ID**: ``chess_v6``
- **Players**: 2
- **Action space**: Discrete (legal moves)
- **Observation**: 8×8×111 binary planes

.. note::

   Requires the Stockfish binary: ``sudo apt install stockfish`` (Linux)
   or ``brew install stockfish`` (macOS).

|

Go
--

.. image:: https://pettingzoo.farama.org/_images/classic_go.gif
   :alt: Go environment
   :width: 200px
   :align: right

The ancient board game of Go.  Supports 9×9, 13×13, and 19×19 board sizes.

- **ID**: ``go_v5``
- **Players**: 2
- **Action space**: Discrete (board_size² + 1 for pass)
- **Observation**: board_size × board_size × 17 planes

|

Connect Four
------------

.. image:: https://pettingzoo.farama.org/_images/classic_connect_four.gif
   :alt: Connect Four environment
   :width: 200px
   :align: right

Classic 4-in-a-row game on a 6×7 board.  A simple environment often used for
testing multi-agent algorithms.

- **ID**: ``connect_four_v3``
- **Players**: 2
- **Action space**: Discrete(7) — column selection
- **Observation**: 6×7×2 binary planes

|

Tic-Tac-Toe
------------

.. image:: https://pettingzoo.farama.org/_images/classic_tictactoe.gif
   :alt: Tic-Tac-Toe environment
   :width: 200px
   :align: right

3×3 Tic-Tac-Toe.  The simplest PettingZoo Classic environment — ideal for
testing MOSAIC's sequential stepping pipeline.

- **ID**: ``tictactoe_v3``
- **Players**: 2
- **Action space**: Discrete(9) — cell selection
- **Observation**: 3×3×2 binary planes

|

.. tip::

   In MOSAIC, you can assign **different agent types** to each player.
   For example, ``player_0`` as Human keyboard control and ``player_1``
   as a CleanRL-trained PPO agent:

   .. code-block:: python

      policy_service.bind_agent_policy("player_0", "human_keyboard")
      policy_service.bind_agent_policy("player_1", "cleanrl_ppo")
