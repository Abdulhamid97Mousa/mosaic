Developing a Worker
===================

This guide walks through creating a new MOSAIC worker from scratch.
A complete worker has two sides:

.. mermaid::

   graph LR
       subgraph "Backend (3rd_party/)"
           CFG["Config<br/>config.py"]
           RT["Runtime<br/>runtime.py"]
           CLI["CLI<br/>cli.py"]
           EP["Entry Point<br/>pyproject.toml"]
       end

       subgraph "Frontend (gym_gui/ui/)"
           FORM["Train Form<br/>widgets/"]
           FAC["Form Factory<br/>forms/"]
           CAT["Worker Catalog<br/>worker_catalog/"]
           PRES["Presenter<br/>presenters/"]
       end

       subgraph "Bridge"
           HANDLER["TrainingFormHandler"]
           CLIENT["TrainerClient (gRPC)"]
       end

       FORM -->|"get_config()"| HANDLER
       HANDLER -->|"submit_run()"| CLIENT
       CLIENT -->|"gRPC"| RT
       EP -.->|"discovery"| CFG
       FAC -.->|"creates"| FORM
       CAT -.->|"metadata"| FAC

       style CFG fill:#ff7f50,stroke:#cc5500,color:#fff
       style RT fill:#ff7f50,stroke:#cc5500,color:#fff
       style CLI fill:#ff7f50,stroke:#cc5500,color:#fff
       style EP fill:#ff7f50,stroke:#cc5500,color:#fff
       style FORM fill:#4a90d9,stroke:#2e5a87,color:#fff
       style FAC fill:#4a90d9,stroke:#2e5a87,color:#fff
       style CAT fill:#4a90d9,stroke:#2e5a87,color:#fff
       style PRES fill:#4a90d9,stroke:#2e5a87,color:#fff
       style HANDLER fill:#50c878,stroke:#2e8b57,color:#fff
       style CLIENT fill:#50c878,stroke:#2e8b57,color:#fff

- **Backend:** the worker package under ``3rd_party/``. Handles training
  logic, telemetry, and the CLI entry point that the Daemon spawns.
- **Frontend:** the Qt6 UI integration under ``gym_gui/ui/``. Handles
  the training form dialog, worker catalog entry, presenter, and how
  user configuration reaches the backend via gRPC.

.. toctree::
   :hidden:
   :maxdepth: 2

   backend
   frontend
