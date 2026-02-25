BALROG Worker
=============

.. figure:: /images/workers/balrog_banner.png
   :alt: BALROG Banner
   :align: center
   :width: 80%

   BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games
   (`Paglieri et al., 2024 <https://arxiv.org/abs/2411.13543>`_).

.. raw:: html

   <br>

The BALROG worker is MOSAIC's **LLM/VLM agentic evaluation** integration.
It wraps `BALROG <https://github.com/balrog-ai/BALROG>`, A benchmark
framework for evaluating large language models and vision-language models
as agents on complex, long-horizon interactive tasks behind the standard
:doc:`shim pattern <../../concept>`.  

.. list-table::
   :widths: 25 75

   * - **Paradigm**
     - Single-agent, LLM/VLM evaluation
   * - **Task Type**
     - Long-horizon interactive decision-making
   * - **Model Support**
     - API clients (OpenAI, Anthropic, Google Gemini) and local inference (vLLM)
   * - **Environments**
     - NetHack, MiniHack, BabyAI, Crafter, TextWorld, MiniGrid
   * - **Execution**
     - Subprocess (parallel workers)
   * - **GPU required**
     - No (API-based) / Optional (vLLM local inference)
   * - **Source**
     - ``3rd_party/balrog_worker/balrog_worker/``
   * - **Upstream**
     - `github.com/balrog-ai/BALROG <https://github.com/balrog-ai/BALROG>`_
   * - **Paper**
     - `arXiv:2411.13543 <https://arxiv.org/abs/2411.13543>`_

Overview
--------

BALROG benchmarks agentic LLM and VLM reasoning on reinforcement learning games, environments
that demand long sequences of decisions, partial observability, and
adaptive behaviour.  Unlike standard RL workers that train neural
policies, the BALROG worker **drives pre-trained language models** through
game environments and records performance on the BALROG benchmark suite.

Key features:

- Dual support for **text-only LLMs** and **vision-language models (VLMs)**
- Pluggable API backends — OpenAI, Anthropic Claude, Google Gemini, or any
  OpenAI-compatible endpoint (vLLM)
- Configurable **history windows** and **interaction modes**
- Parallel evaluation across multiple workers
- JSONL telemetry streamed back to the MOSAIC Trainer Daemon

Architecture
------------

The diagram below shows the BALROG evaluation pipeline from the original paper:

.. figure:: /images/workers/concepts/balrog_diagram.png
   :alt: BALROG evaluation pipeline showing env_wrapper, client, evaluator, and agent
   :align: center
   :width: 90%

   BALROG evaluation pipeline (`Paglieri et al., 2024 <https://arxiv.org/abs/2411.13543>`_):
   ``env_wrapper.py``, ``client.py``, ``evaluator.py``, and ``agent.py`` collaborate
   to drive LLM/VLM agents through game environments.

The BALROG worker follows the standard MOSAIC :doc:`shim pattern <../../concept>`.

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Backend
     - Models
     - Notes
   * - **OpenAI API**
     - GPT-4o, GPT-4-turbo, GPT-3.5-turbo
     - Requires ``OPENAI_API_KEY``
   * - **Anthropic API**
     - Claude 3 Opus/Sonnet/Haiku
     - Requires ``ANTHROPIC_API_KEY``
   * - **Google Gemini**
     - Gemini 1.5 Pro/Flash
     - Requires ``GOOGLE_API_KEY``
   * - **vLLM (local)**
     - Any HuggingFace-compatible model
     - Self-hosted inference server

Installation
------------

.. tabs::

   .. tab:: pip (API-only)

      .. code-block:: bash

         pip install -e ".[balrog]"

   .. tab:: pip (vLLM local inference)

      .. code-block:: bash

         pip install -e ".[balrog,vllm]"

Configuration
-------------

The BALROG worker is configured via the MOSAIC GUI training form or
directly via JSON:

.. code-block:: json

   {
     "worker": "balrog",
     "model": "claude-3-5-sonnet-20241022",
     "backend": "anthropic",
     "environment": "MiniHack-River-v0",
     "num_episodes": 100,
     "max_steps": 1000,
     "history_window": 4,
     "parallel_workers": 4
   }

References
----------

- **GitHub**: `github.com/balrog-ai/BALROG <https://github.com/balrog-ai/BALROG>`_
- **Paper**: `BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games <https://arxiv.org/abs/2411.13543>`_
- **Leaderboard**: `balrogai.com <https://balrogai.com>`_

.. toctree::
   :maxdepth: 1

   installation
   common_errors
