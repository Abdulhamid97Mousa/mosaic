Workers
=======

MOSAIC uses external "workers" for training and inference.
Each worker is a separate process that communicates via gRPC.

Worker Types
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Worker
     - Paradigm
     - Algorithms
     - Use Case
   * - CleanRL
     - SINGLE_AGENT
     - PPO, DQN, SAC, TD3, DDPG
     - Simple RL training
   * - XuanCe
     - SIMULTANEOUS
     - MAPPO, QMIX, MADDPG, VDN
     - Multi-agent RL
   * - RLlib
     - SIMULTANEOUS
     - PPO, IMPALA, APPO, SAC
     - Distributed training
   * - Jason BDI
     - HIERARCHICAL
     - AgentSpeak plans
     - Goal-driven agents
   * - SPADE BDI
     - HIERARCHICAL
     - Python BDI
     - Python-native BDI
   * - LLM
     - HIERARCHICAL
     - GPT, Claude, Llama
     - Language model agents

CleanRL Worker
--------------

Single-file RL implementations for quick experimentation.

.. code-block:: bash

   pip install -e ".[cleanrl]"

**Supported algorithms**: PPO, DQN, SAC, TD3, DDPG, C51

**Configuration**:

.. code-block:: python

   config = {
       "algorithm": "ppo",
       "learning_rate": 3e-4,
       "total_timesteps": 1_000_000,
       "num_envs": 4,
       "capture_video": True,
   }

XuanCe Worker
-------------

Multi-agent RL library with comprehensive algorithm support.

.. code-block:: bash

   pip install -e ".[xuance]"

**Supported algorithms**: MAPPO, QMIX, MADDPG, VDN, COMA, IPPO, IQL

**Configuration**:

.. code-block:: python

   config = {
       "algorithm": "mappo",
       "learning_rate": 5e-4,
       "batch_size": 256,
       "backend": "torch",
   }

RLlib Worker
------------

Ray-based distributed RL for large-scale training.

.. code-block:: bash

   pip install -e ".[ray-rllib]"

**Supported algorithms**: PPO, IMPALA, APPO, SAC, DQN

**Configuration**:

.. code-block:: python

   config = {
       "algorithm": "PPO",
       "num_workers": 8,
       "num_envs_per_worker": 4,
       "framework": "torch",
   }

Jason BDI Worker
----------------

AgentSpeak agents via Java bridge.

**Configuration**:

.. code-block:: python

   config = {
       "agent_file": "agent.asl",
       "mas_file": "project.mas2j",
       "debug_mode": False,
   }

Adding a New Worker
-------------------

1. Create directory structure:

   .. code-block:: text

      3rd_party/myworker/
      ├── pyproject.toml
      ├── myworker/
      │   └── __init__.py
      └── original_lib/  (optional)

2. Create requirements file: ``requirements/myworker.txt``

3. Add to ``pyproject.toml`` optional dependencies

4. Register worker capabilities
