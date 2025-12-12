MOSAIC: Multi-Agent Orchestration System
========================================

.. raw:: html

   <a href="https://github.com/Abdulhamid97Mousa/MOSAIC">
        <img alt="GitHub" src="https://img.shields.io/github/stars/Abdulhamid97Mousa/MOSAIC?style=social">
   </a>
   <a href="https://github.com/Abdulhamid97Mousa/MOSAIC/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/Abdulhamid97Mousa/MOSAIC">
   </a>
   <a href="https://www.python.org/downloads/">
        <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg">
   </a>
   <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-red">
   </a>
   <a href="https://www.gymlibrary.dev/">
        <img alt="Gymnasium" src="https://img.shields.io/badge/gymnasium-%3E%3D0.28.1-blue">
   </a>
   <a href="https://pettingzoo.farama.org/">
        <img alt="PettingZoo" src="https://img.shields.io/badge/PettingZoo-%3E%3D1.24.0-blue">
   </a>

.. raw:: html

   <br><br>

**MOSAIC** (Multi-Agent Orchestration System with Adaptive Intelligent Control) is a unified platform
for heterogeneous agent workloads. It provides a visual-first interface for configuring, running,
and analyzing reinforcement learning experiments across multiple paradigms.

| **GitHub**: `https://github.com/Abdulhamid97Mousa/MOSAIC <https://github.com/Abdulhamid97Mousa/MOSAIC>`_

Why MOSAIC?
-----------

MOSAIC addresses a fundamental challenge in multi-agent reinforcement learning:
**existing platforms are locked to single stepping paradigms**, forcing researchers
to choose between frameworks like RLlib (POSG) or PettingZoo (AEC).

MOSAIC provides:

- **Multi-Paradigm Support**: Seamlessly switch between Single-Agent, Sequential (AEC), Simultaneous (POSG), and Hierarchical paradigms
- **Heterogeneous Agent Composition**: Mix Human, RL (CleanRL, RLlib, XuanCe), BDI (Jason, SPADE), and LLM agents in the same environment
- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface
- **PolicyMappingService**: Per-agent policy binding with paradigm awareness
- **Extensible Worker Architecture**: Add new training backends without modifying core code

Key Features
------------

**Multi-Paradigm Orchestration**

* :class:`SINGLE_AGENT` - Standard Gymnasium environments
* :class:`SEQUENTIAL` - PettingZoo AEC (turn-based games like Chess)
* :class:`SIMULTANEOUS` - PettingZoo Parallel / RLlib (cooperative/competitive continuous control)
* :class:`HIERARCHICAL` - BDI agents with goal-driven behavior

**Supported Workers**

* **CleanRL** - Single-file RL implementations (PPO, DQN, SAC, TD3)
* **XuanCe** - Multi-agent algorithms (MAPPO, QMIX, MADDPG)
* **RLlib** - Distributed training with Ray
* **Jason BDI** - AgentSpeak agents via Java bridge
* **SPADE BDI** - Python-native BDI agents
* **LLM** - Language model agents (planned)

Quick Example
-------------

.. code-block:: python

   from mosaic.services import PolicyMappingService
   from mosaic.core.enums import SteppingParadigm

   # Configure heterogeneous agents for a Chess game
   policy_service = PolicyMappingService()
   policy_service.set_paradigm(SteppingParadigm.SEQUENTIAL)

   # Player 0: Human control
   policy_service.bind_agent_policy("player_0", "human_keyboard")

   # Player 1: Trained RL policy
   policy_service.bind_agent_policy("player_1", "cleanrl_ppo")

Architecture Overview
---------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        MOSAIC Platform                          │
   ├─────────────────────────────────────────────────────────────────┤
   │  ┌───────────────────────────────────────────────────────────┐  │
   │  │                    Visual Layer (PyQt6)                    │  │
   │  │  MainWindow │ ControlPanel │ RenderTabs │ AdvancedConfig  │  │
   │  └───────────────────────────────────────────────────────────┘  │
   │                              │                                   │
   │  ┌───────────────────────────────────────────────────────────┐  │
   │  │                    Service Layer                           │  │
   │  │  PolicyMappingService │ ActorService │ TelemetryService   │  │
   │  └───────────────────────────────────────────────────────────┘  │
   │                              │                                   │
   │  ┌───────────────────────────────────────────────────────────┐  │
   │  │                    Adapter Layer                           │  │
   │  │  ParadigmAdapter │ EnvironmentAdapter │ PettingZooAdapter │  │
   │  └───────────────────────────────────────────────────────────┘  │
   │                              │                                   │
   └──────────────────────────────┼───────────────────────────────────┘
                                  │ gRPC / IPC
   ┌──────────────────────────────┼───────────────────────────────────┐
   │                        3rd Party Workers                         │
   │  CleanRL │ XuanCe │ RLlib │ Jason BDI │ SPADE │ LLM │ ViZDoom  │
   └──────────────────────────────────────────────────────────────────┘

Who Is MOSAIC For?
------------------

MOSAIC is designed for:

- **Researchers** exploring multi-agent RL with heterogeneous agents
- **Developers** building RL applications with visual configuration
- **Students** learning about different RL paradigms and agent architectures
- **AI practitioners** interested in combining symbolic AI (BDI) with neural methods (RL)

.. raw:: html

   <br><hr>

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   documents/tutorials/installation
   documents/tutorials/quickstart
   documents/tutorials/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   documents/architecture/overview
   documents/architecture/paradigms
   documents/architecture/policy_mapping
   documents/architecture/workers

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   documents/api/core
   documents/api/services
   documents/api/adapters

.. toctree::
   :hidden:
   :caption: Development

   GitHub <https://github.com/Abdulhamid97Mousa/MOSAIC>
   documents/contributing
   documents/changelog
