Common Errors
=============

The VLM Worker shares the same API clients as the
:doc:`LLM Worker <../MOSAIC_LLM_Worker/index>`, so most errors and fixes are
identical. Below are the VLM-specific issues plus the shared ones.

Image encoding errors
---------------------

.. code-block:: text

   ValueError: Cannot encode observation as image

The environment returned an observation that cannot be converted to an RGB
image for VLM input.

**Fix:** Ensure the environment supports ``render_mode="rgb_array"`` and
that ``--max-image-history`` is set to ``0`` for text-only environments
like TextWorld:

.. code-block:: bash

   # Text-only fallback (no images)
   vlm-worker --max-image-history 0 ...

   # With image observations
   vlm-worker --max-image-history 1 --render-mode rgb_array ...

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
20 requests/minute**. VLM requests with images consume more tokens and may
hit limits faster than text-only requests.

**Fix:** Use models hosted on their own infrastructure (less rate-limiting):

- ``nvidia/nemotron-3-nano-30b-a3b:free``: NVIDIA-hosted, ~1.6s latency
- ``arcee-ai/trinity-large-preview:free``: Arcee-hosted, ~2.4s latency

VLM returns invalid action
---------------------------

.. code-block:: text

   MOSAIC: Failed to parse action from VLM output, defaulting to 'still': ...

The VLM returned text that doesn't match any known action name.

**Fix:** This is handled automatically — the worker defaults to ``still``
(do nothing). Consider using the ``robust_cot`` or ``robust_naive`` agent
type for better action parsing:

.. code-block:: bash

   vlm-worker --agent-type robust_cot ...

Timeout on VLM API call
------------------------

.. code-block:: text

   Failed to execute api_call after 3 retries.

VLM requests with images take longer than text-only LLM requests due to
image encoding and processing overhead.

**Fix:** Increase the timeout:

.. code-block:: bash

   vlm-worker --timeout 120 ...

vLLM server not running
------------------------

.. code-block:: text

   openai.APIConnectionError: Connection error.

When using ``--client vllm``, the vLLM server must be running locally with
a vision-capable model.

**Fix:** Start the server with a VLM model first:

.. code-block:: bash

   vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
       --host 127.0.0.1 --port 8100 \
       --max-model-len 4096 --trust-remote-code

Then point the worker at it:

.. code-block:: bash

   vlm-worker --client vllm --base-url http://localhost:8100/v1 ...

omegaconf not installed
-----------------------

.. code-block:: text

   ModuleNotFoundError: No module named 'omegaconf'

The agent factory uses OmegaConf for configuration management.

**Fix:**

.. code-block:: bash

   pip install omegaconf
