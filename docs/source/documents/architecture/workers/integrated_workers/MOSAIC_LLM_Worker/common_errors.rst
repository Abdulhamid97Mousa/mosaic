Common Errors
=============

OPENROUTER_API_KEY not set
--------------------------

.. code-block:: text

   ValueError: OPENROUTER_API_KEY environment variable must be set

**Fix:** Set the key in ``.env`` or export it:

.. code-block:: bash

   export OPENROUTER_API_KEY=sk-or-v1-your_key_here

Or enter it directly in the Operator Config panel in the GUI.

Rate limiting (429) on free models
----------------------------------

.. code-block:: text

   openai.RateLimitError: Error code: 429

Free-tier models on OpenRouter have limits: **50 requests/day,
20 requests/minute**. Some providers (e.g., Venice) rate-limit more
aggressively.

**Fix:** Use models hosted on their own infrastructure (less rate-limiting):

- ``nvidia/nemotron-3-nano-30b-a3b:free``: NVIDIA-hosted, ~1.6s latency
- ``arcee-ai/trinity-large-preview:free``: Arcee-hosted, ~2.4s latency
- ``upstage/solar-pro-3:free``: Upstage-hosted, ~4s latency

LLM returns invalid action
---------------------------

.. code-block:: text

   MOSAIC: Failed to parse action from LLM output, defaulting to 'still': ...

The LLM returned text that doesn't match any known action name.

**Fix:** This is handled automatically, the worker defaults to ``still``
(do nothing) for Legacy environments or ``done`` for INI environments.
Consider using the ``robust_cot`` or ``robust_naive`` agent type for better
action parsing:

.. code-block:: bash

   llm-worker --agent-type robust_cot ...

Timeout on LLM API call
------------------------

.. code-block:: text

   Failed to execute api_call after 3 retries.

The LLM API did not respond within the timeout period (default: 60s).

**Fix:** Increase the timeout or switch to a faster model:

.. code-block:: bash

   llm-worker --timeout 120 ...

   # Or use a faster model
   llm-worker --model nvidia/nemotron-3-nano-30b-a3b:free ...

Environment not initialized
----------------------------

.. code-block:: text

   Environment not initialized. Send reset first.

In interactive mode, a ``reset`` command must be sent before ``step``.

**Fix:** Send ``{"cmd": "reset", "seed": 42}`` before any ``{"cmd": "step"}``
commands.

Agent not initialized (action-selector mode)
---------------------------------------------

.. code-block:: text

   Agent not initialized. Send init_agent or reset first.

In action-selector mode (PettingZoo games), ``init_agent`` must be sent
before ``select_action``.

**Fix:** Send ``{"cmd": "init_agent", "game_name": "chess_v6", "player_id": "player_0"}``
before ``{"cmd": "select_action", ...}``.

omegaconf not installed
-----------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'omegaconf'

The agent factory uses OmegaConf for configuration management.

**Fix:**

.. code-block:: bash

   pip install omegaconf

vLLM server not running
------------------------

.. code-block:: text

   openai.APIConnectionError: Connection error.

When using ``--client vllm``, the vLLM server must be running locally.

**Fix:** Start the server first:

.. code-block:: bash

   vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key EMPTY --port 8000

Then set:

.. code-block:: bash

   llm-worker --client vllm --base-url http://localhost:8000/v1 ...
