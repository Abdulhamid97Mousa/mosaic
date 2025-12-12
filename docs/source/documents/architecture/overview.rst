Architecture Overview
=====================

MOSAIC is built on a layered architecture that separates concerns
and enables extensibility.

System Layers
-------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                     VISUAL LAYER (PyQt6)                        │
   │  MainWindow │ ControlPanel │ RenderTabs │ AdvancedConfigTab    │
   └─────────────────────────────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────────────────────────────────────────┐
   │                     SERVICE LAYER                               │
   │  PolicyMappingService │ ActorService │ TelemetryService        │
   └─────────────────────────────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────────────────────────────────────────┐
   │                     CONTROLLER LAYER                            │
   │  SessionController │ HumanInputController │ EnvironmentMgr     │
   └─────────────────────────────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────────────────────────────────────────┐
   │                     ADAPTER LAYER                               │
   │  ParadigmAdapter │ EnvironmentAdapter │ PettingZooAdapter      │
   └─────────────────────────────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────────────────────────────────────────┐
   │                     WORKER LAYER (3rd Party)                    │
   │  CleanRL │ XuanCe │ RLlib │ Jason BDI │ SPADE │ LLM            │
   └─────────────────────────────────────────────────────────────────┘

Visual Layer
------------

The visual layer provides the user interface built with PyQt6:

- **MainWindow**: Application shell with tab management
- **ControlPanel**: Environment selection and actor configuration
- **RenderTabs**: Display environment renders (RGB, ASCII, etc.)
- **AdvancedConfigTab**: Fine-grained experiment configuration

Service Layer
-------------

Services provide business logic independent of the UI:

- **PolicyMappingService**: Per-agent policy binding with paradigm awareness
- **ActorService**: Actor registration and action selection
- **TelemetryService**: Metrics collection and export (TensorBoard, W&B)
- **SeedManager**: Reproducibility management

Controller Layer
----------------

Controllers coordinate between services and handle state:

- **SessionController**: Manages the training/evaluation loop
- **HumanInputController**: Captures keyboard/mouse input
- **EnvironmentManager**: Environment lifecycle management

Adapter Layer
-------------

Adapters provide unified interfaces to different environment types:

- **ParadigmAdapter**: Abstract base for stepping paradigms
- **EnvironmentAdapter**: Gymnasium environments
- **PettingZooAdapter**: PettingZoo multi-agent environments
- **ViZDoomAdapter**: ViZDoom FPS environments

Worker Layer
------------

External training backends communicate via gRPC/IPC:

- **CleanRL**: Single-agent RL (PPO, DQN, SAC, TD3)
- **XuanCe**: Multi-agent RL (MAPPO, QMIX, MADDPG)
- **RLlib**: Distributed RL with Ray
- **Jason BDI**: AgentSpeak agents
- **SPADE BDI**: Python BDI agents
