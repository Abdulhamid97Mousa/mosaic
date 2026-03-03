Chess LLM Worker
=================

.. figure:: /images/workers/llm_chess_banner.png
   :alt: LLM Chess Architecture
   :align: center
   :width: 80%

   LLM Chess multi-turn agent interaction architecture
   (`Saplin, 2025 <https://arxiv.org/abs/2512.01992>`_).

.. raw:: html

   <br>

Overview
--------

The **Chess LLM Worker** wraps the
`llm_chess <https://github.com/maxim-saplin/llm_chess>`_ library to enable
LLM-driven chess play inside MOSAIC's PettingZoo Chess environment
(``chess_v6``).  The worker implements a **multi-turn dialog** protocol: on
each move the LLM can query for the current board state and legal moves
before committing to a UCI move.  If the model produces an invalid move, the
worker retries with corrective feedback and, after exhausting retries, falls
back to a random legal move.

Key features:

- **OpenAI-compatible API:** works with vLLM (local), OpenAI, and Anthropic
  backends via a single ``client_name`` switch.
- **Multi-turn reasoning:** the LLM can issue ``get_current_board``,
  ``get_legal_moves``, or ``make_move <uci>`` commands across multiple
  dialog turns before the worker returns an action.
- **Token tracking:** every move records ``input_tokens`` and
  ``output_tokens`` for cost analysis.
- **Graceful fallback:** invalid moves trigger retries; on max retries a
  random legal move is selected automatically.

Architecture
------------

The Chess LLM Worker follows the standard MOSAIC
:doc:`shim pattern <../../concept>`:

.. code-block:: text

   3rd_party/chess_worker/
   ├── chess_worker/
   │   ├── __init__.py      # package metadata & get_worker_metadata()
   │   ├── cli.py           # CLI entry point (chess-worker command)
   │   ├── config.py        # ChessWorkerConfig dataclass
   │   └── runtime.py       # ChessWorkerRuntime (multi-turn LLM loop)
   ├── llm_chess/            # upstream library (git submodule, unmodified)
   ├── tests/
   └── pyproject.toml

Components
^^^^^^^^^^

``cli.py``
   Parses command-line arguments (``--run-id``, ``--client-name``,
   ``--model-id``, ``--base-url``, ``--temperature``, ``--max-tokens``,
   ``--max-retries``, ``--max-dialog-turns``, ``--telemetry-dir``) and
   launches the interactive runtime.

``config.py``
   ``ChessWorkerConfig`` dataclass holding all parameters:

   - **LLM settings:** ``client_name`` (vllm / openai / anthropic),
     ``model_id``, ``base_url``, ``api_key``
   - **Generation:** ``temperature`` (default 0.3), ``max_tokens`` (256)
   - **Chess-specific:** ``max_retries`` (3), ``max_dialog_turns`` (10)
   - **Environment:** ``env_name`` ("pettingzoo"), ``task`` ("chess_v6")

``runtime.py``
   ``ChessWorkerRuntime`` implements the multi-turn dialog loop:

   1. Receive ``init_agent`` with ``player_id`` → set player color and
      system prompt.
   2. Receive ``select_action`` with board state and legal moves.
   3. Build observation message, enter multi-turn LLM conversation
      (up to ``max_dialog_turns``).
   4. Parse LLM response for UCI move (``make_move <uci>``).
   5. Validate against legal moves, retry on invalid.
   6. Return action with reasoning and token statistics.

Supported Models
----------------

Any model accessible through an OpenAI-compatible API:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Backend
     - Flag
     - Example Models
   * - **vLLM** (local)
     - ``--client-name vllm``
     - Qwen2.5-1.5B-Instruct, Llama-3, Mistral
   * - **OpenAI**
     - ``--client-name openai``
     - GPT-4o, GPT-4o-mini, o1
   * - **Anthropic**
     - ``--client-name anthropic``
     - Claude 3.5 Sonnet, Claude 3 Opus

Action Protocol
---------------

The LLM communicates through structured commands in its text output:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``get_current_board``
     - Request the current board state (ASCII diagram)
   * - ``get_legal_moves``
     - Request the list of legal UCI moves
   * - ``make_move <uci>``
     - Submit a chess move in UCI notation (e.g. ``make_move e2e4``)

The worker parses these commands from the LLM response using regex and
responds accordingly within the multi-turn loop.

JSON IPC Protocol
-----------------

The worker communicates with the MOSAIC GUI via stdin/stdout JSON messages:

**Commands:**

.. code-block:: json

   {"command": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}

.. code-block:: json

   {"command": "select_action", "observation": "...", "legal_moves": ["e2e4", "d2d4", "..."], "board_str": "..."}

.. code-block:: json

   {"command": "stop"}

**Response:**

.. code-block:: json

   {
     "action_str": "e2e4",
     "action_index": null,
     "reasoning": "Opening with king's pawn to control the centre.",
     "input_tokens": 45,
     "output_tokens": 12,
     "success": true
   }

Configuration
-------------

CLI arguments:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--run-id``
     -
     - Unique run identifier
   * - ``--client-name``
     - ``vllm``
     - LLM backend (vllm, openai, anthropic)
   * - ``--model-id``
     - ``Qwen/Qwen2.5-1.5B-Instruct``
     - Model identifier
   * - ``--base-url``
     - ``http://127.0.0.1:8000/v1``
     - API base URL
   * - ``--temperature``
     - ``0.3``
     - Sampling temperature
   * - ``--max-tokens``
     - ``256``
     - Max output tokens per LLM call
   * - ``--max-retries``
     - ``3``
     - Max invalid move retries before random fallback
   * - ``--max-dialog-turns``
     - ``10``
     - Max conversation turns per move
   * - ``--telemetry-dir``
     -
     - Directory for telemetry output

Installation
------------

.. code-block:: bash

   # From the MOSAIC root
   pip install -e "3rd_party/chess_worker[chess]"

   # This installs: python-chess, pettingzoo[classic], openai

Worker Capabilities
-------------------

.. list-table::
   :widths: 30 70

   * - **Worker type**
     - chess
   * - **Supported paradigms**
     - self_play, human_vs_ai
   * - **Max agents**
     - 2
   * - **GPU required**
     - No (LLM inference is remote)
   * - **Estimated RAM**
     - ~512 MB
