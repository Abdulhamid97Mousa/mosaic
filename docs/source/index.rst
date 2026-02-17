MOSAIC: Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous Agent Workloads
=============================================================================================================

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

**MOSAIC** (Multi-Agent Orchestration System with Adaptive Intelligent Control for Heterogeneous
Agent Workloads) is a unified platform that orchestrates diverse agents, paradigms, and workers
to create cohesive intelligent systems — like tiles in a mosaic forming a complete picture.
It provides a visual-first interface for configuring, running, and analyzing reinforcement
learning experiments across multiple paradigms.

.. image:: images/Platform_Main_View.png
   :alt: MOSAIC Platform - Main View
   :align: center
   :width: 100%

.. raw:: html

   <br>

| **GitHub**: `https://github.com/Abdulhamid97Mousa/MOSAIC <https://github.com/Abdulhamid97Mousa/MOSAIC>`_

Why MOSAIC?
-----------

Today's AI landscape offers powerful but **fragmented** tools: RL frameworks (CleanRL, RLlib, XuanCe),
language models (GPT, Claude), robotics simulators (MuJoCo), and
3D game engines (Godot). Each excels in isolation, but **no platform bridges them together**
under a unified, visual-first interface.

MOSAIC provides:

- **Unified Framework Bridge**: Connect RL, LLM, Robotics, and 3D Simulation in a single platform
- **Visual-First Design**: Configure experiments through an intuitive PyQt6 interface — no code required
- **Heterogeneous Agent Mixing**: Run Human + RL + LLM agents in the same environment
- **Resource Management & Quotas**: GPU allocation, queue limits, credit-based backpressure, health monitoring
- **Per-Agent Policy Binding**: Route each agent to different workers via ``PolicyMappingService``
- **Worker Lifecycle Orchestration**: Subprocess management with heartbeat monitoring and graceful termination

Supported Environment Families
------------------------------

MOSAIC supports **26 environment families** spanning single-agent, multi-agent,
and cooperative/competitive paradigms.  See the full
:doc:`Environment Families <documents/environments/index>` reference for
installation instructions, environment lists, and academic citations.

.. list-table::
   :widths: 28 42 30
   :header-rows: 1

   * - Family
     - Description
     - Example Environments
   * - **Gymnasium**
     - Standard single-agent RL (Toy Text, Classic Control, Box2D, MuJoCo)
     - CartPole, LunarLander, Ant, FrozenLake
   * - **Atari / ALE**
     - 128 classic Atari 2600 games
     - Breakout, Pong, SpaceInvaders
   * - **MiniGrid**
     - Procedural grid-world navigation
     - Empty, DoorKey, LavaGap, MultiRoom
   * - **BabyAI**
     - Language-grounded instruction following
     - GoTo, Open, Pickup, Unlock, BossLevel
   * - **ViZDoom**
     - Doom-based first-person visual RL
     - Basic, DeadlyCorridor, Deathmatch
   * - **MiniHack / NetHack**
     - Roguelike dungeon crawling (NLE)
     - Room, MazeWalk, NetHackChallenge
   * - **Crafter**
     - Open-world survival benchmark
     - CrafterReward, CrafterNoReward
   * - **Procgen**
     - 16 procedurally generated environments
     - CoinRun, StarPilot, Maze, Heist
   * - **TextWorld**
     - Text-based interactive fiction
     - CoinCollector, TreasureHunter, Cooking
   * - **BabaIsAI**
     - Rule-manipulation puzzles
     - BabaIsAI-Default
   * - **Jumanji**
     - JAX-accelerated logic/routing/packing (25 envs)
     - Game2048, Tetris, PacMan, Snake
   * - **PyBullet Drones**
     - Quadcopter physics simulation
     - HoverAviary, MultiHoverAviary
   * - **PettingZoo Classic**
     - Turn-based board games (AEC)
     - Chess, Go, Connect Four, TicTacToe
   * - **OpenSpiel**
     - Board games + draughts variants (AEC)
     - Checkers, International Draughts
   * - **MOSAIC MultiGrid**
     - Competitive team sports (view_size=3)
     - Soccer 2v2, Collect, Basketball 3v3
   * - **INI MultiGrid**
     - Cooperative exploration (view_size=7)
     - Empty, LockedHallway, RedBlueDoors
   * - **Melting Pot**
     - Social multi-agent scenarios (up to 16 agents)
     - CleanUp, Territory, Cooking, PrisonersDilemma
   * - **Overcooked**
     - Cooperative cooking (2 agents)
     - CrampedRoom, CoordinationRing
   * - **SMAC**
     - StarCraft Multi-Agent Challenge (hand-designed maps)
     - 3m, 8m, 2s3z, MMM2
   * - **SMACv2**
     - StarCraft Multi-Agent Challenge v2 (procedural units)
     - 10gen_terran, 10gen_protoss, 10gen_zerg
   * - **RWARE**
     - Cooperative warehouse delivery
     - tiny/small/medium/large (2–8 agents)

Supported Workers
-----------------

* **CleanRL** -- Single-file RL implementations (PPO, DQN, SAC, TD3, DDPG, C51)
* **XuanCe** -- Multi-agent algorithms (MAPPO, QMIX, MADDPG, VDN, COMA)
* **RLlib** -- Distributed training with Ray (PPO, IMPALA, APPO)
* **BALROG** -- Single-agent LLM benchmarking (MiniGrid, BabyAI, MiniHack, Crafter)
* **MOSAIC LLM** -- Multi-agent LLM with coordination strategies and Theory of Mind (MultiGrid, BabyAI, MeltingPot, PettingZoo)
* **Chess LLM** -- LLM chess play with multi-turn dialog (PettingZoo Chess)

Quick Example
-------------

.. code-block:: python

   from gym_gui.services import PolicyMappingService
   from gym_gui.core.enums import SteppingParadigm

   # Configure heterogeneous agents for a Chess game
   policy_service = PolicyMappingService()
   policy_service.set_paradigm(SteppingParadigm.SEQUENTIAL)

   # Player 0: Human control
   policy_service.bind_agent_policy("player_0", "human_keyboard")

   # Player 1: Trained RL policy
   policy_service.bind_agent_policy("player_1", "cleanrl_ppo")

Architecture Overview
---------------------

.. mermaid::

   graph LR
       A[PyQt6 GUI] --> B[Services]
       B --> C[Adapters]
       C <--> D[Workers]

       style A fill:#4a90d9,stroke:#2e5a87,color:#fff
       style B fill:#50c878,stroke:#2e8b57,color:#fff
       style C fill:#ff7f50,stroke:#cc5500,color:#fff
       style D fill:#9370db,stroke:#6a0dad,color:#fff

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Layer
     - Components
   * - **GUI**
     - MainWindow, ControlPanel, RenderTabs, AdvancedConfig
   * - **Services**
     - PolicyMappingService, ActorService, TelemetryService
   * - **Adapters**
     - ParadigmAdapter, PettingZooAdapter, ViZDoomAdapter
   * - **Workers**
     - CleanRL, XuanCe, RLlib, LLM

Core Features
-------------

**Multi-Paradigm Support**
   Seamlessly switch between single-agent, multi-agent (AEC/Parallel), and hybrid environments
   without changing your agent code.

**Agent Integration**
   Human, RL (CleanRL, Ray), and future LLM agents in the same framework.

**Policy Mapping**
   Assign different policies to different agents with flexible configuration through the
   PolicyMappingService.

**3D Engine Support**
   MuJoCo MPC for robotics, Godot for game environments, with AirSim planned for drone/vehicle
   simulation.

**Real-time Visualization**
   Interactive render view with the MOSAIC space animation, live telemetry, and episode replay.

Who Is MOSAIC For?
------------------

MOSAIC is designed for:

- **Researchers** exploring multi-agent RL with heterogeneous agents
- **Developers** building RL applications with visual configuration
- **Students** learning about different RL paradigms and agent architectures
- **AI practitioners** interested in combining language models (LLM) with neural methods (RL)
- **Game developers** training AI agents in custom 3D environments

.. raw:: html

   <br><hr>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   documents/tutorials/installation/index
   documents/tutorials/quickstart
   documents/tutorials/basic_usage

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Environments

   documents/environments/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Architecture

   documents/architecture/overview
   documents/architecture/paradigms
   documents/architecture/policy_mapping
   documents/architecture/workers/index
   documents/architecture/operators/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Human Control

   documents/human_control/index

.. toctree::
   :hidden:
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
