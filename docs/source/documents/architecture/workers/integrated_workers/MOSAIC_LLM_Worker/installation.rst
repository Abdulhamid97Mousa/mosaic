Installation
============

The MOSAIC LLM Worker is installed as an editable Python package from the
project's ``3rd_party/mosaic/llm_worker`` directory.

.. tabs::

   .. tab:: pip (API-only, recommended)

      .. code-block:: bash

         pip install -e 3rd_party/mosaic/llm_worker[all]

      This installs both the ``openai`` and ``anthropic`` SDK clients.
      OpenRouter and Google Gemini work through these same SDKs.

   .. tab:: pip (minimal)

      .. code-block:: bash

         pip install -e 3rd_party/mosaic/llm_worker

      Install only the base package. Add clients individually:

      .. code-block:: bash

         pip install openai      # For OpenRouter, OpenAI, vLLM
         pip install anthropic   # For Anthropic Claude

   .. tab:: pip (vLLM local inference)

      .. code-block:: bash

         pip install -e 3rd_party/mosaic/llm_worker[all,vllm]

      Requires a CUDA-capable GPU for local model inference.

API Key Setup
-------------

Configure API keys in the ``.env`` file or export them as environment
variables:

**OpenRouter (recommended, unified access to all providers):**

.. code-block:: bash

   # In .env
   OPENROUTER_API_KEY=sk-or-v1-your_key_here

   # Or export directly
   export OPENROUTER_API_KEY=sk-or-v1-your_key_here

Get your key from https://openrouter.ai/keys

**Other providers:**

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY=sk-...

   # Anthropic
   export ANTHROPIC_API_KEY=sk-ant-...

   # Google Gemini
   export GOOGLE_API_KEY=AI...

   # vLLM (local, no key needed)
   # Just start the server:
   vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key EMPTY

Verify Installation
-------------------

.. code-block:: bash

   # Check CLI is available
   llm-worker --version

   # Quick test with a free model (no credits needed)
   export OPENROUTER_API_KEY=sk-or-v1-your_key_here
   llm-worker --run-id verify-install \
       --env babyai \
       --task BabyAI-GoToRedBall-v0 \
       --client openrouter \
       --model nvidia/nemotron-3-nano-30b-a3b:free \
       --num-episodes 1 --max-steps 10

Dependencies
------------

Core (always installed):

- ``httpx >= 0.27.0``

Optional (via extras):

- ``openai >= 1.0.0``: OpenAI, OpenRouter, vLLM, NVIDIA clients
- ``anthropic >= 0.18.0``: Anthropic Claude client
- ``vllm >= 0.6.0``: Local model inference server
- ``omegaconf``: Configuration management (used by agent factory)
- ``google-genai``: Google Gemini client