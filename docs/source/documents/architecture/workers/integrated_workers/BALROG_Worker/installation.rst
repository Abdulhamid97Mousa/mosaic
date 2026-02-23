Installation
============

.. tabs::

   .. tab:: API-based (no GPU)

      .. code-block:: bash

         pip install -e ".[balrog]"

   .. tab:: Local inference (vLLM)

      .. code-block:: bash

         pip install -e ".[balrog,vllm]"

Set API credentials for your chosen backend:

.. code-block:: bash

   # OpenAI
   export OPENAI_API_KEY="sk-..."

   # Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."

   # Google Gemini
   export GOOGLE_API_KEY="..."

Verify the installation:

.. code-block:: bash

   python -c "import balrog_worker; print('BALROG worker ready')"

.. note::

   NetHack and MiniHack require the NetHack Learning Environment (NLE).
   On some systems this needs system packages:
   ``sudo apt-get install -y build-essential libncurses-dev``
