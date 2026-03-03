Changelog
=========

All notable changes to MOSAIC will be documented here.

[1.0.0] - 2026-02-24
---------------------

Added
^^^^^

- **Operator Abstraction**: Unified agent-level interface for RL, LLM, VLM, and Human decision-makers
- **IPC Worker Protocol**: stdin/stdout JSON protocol making any worker interchangeable
- **Two Evaluation Modes**: Manual Mode (lock-step side-by-side) and Script Mode (automated batch)
- **8 Integrated Workers**: CleanRL, XuanCe, Ray RLlib, BALROG, MOSAIC LLM, Chess LLM, Human Worker, Random Worker
- **26 Environment Families**: Gymnasium, Atari, MiniGrid, BabyAI, ViZDoom, NetHack, Crafter, Procgen, BabaIsAI, Jumanji, PyBullet Drones, PettingZoo, MOSAIC MultiGrid, INI MultiGrid, Melting Pot, Overcooked, SMAC, SMACv2, RWARE, MuJoCo
- **Heterogeneous Decision-Maker**: Mix RL, LLM, Human, and Random agents in the same multi-agent environment
- **Homogeneous Decision-Maker**: Deploy teams of identical paradigm (all-RL, all-LLM, all-Human)
- **Multi-Keyboard Support**: Linux evdev-based per-keyboard routing for multi-human play
- **Deterministic Cross-Paradigm Evaluation**: Shared seed schedules for reproducible comparison
- **Script Mode Configs**: Declarative Python scripts for automated batch evaluation
- **Curriculum Training**: CleanRL DoorKey progression with environment wrappers
- **XuanCe Solo Training Configs**: Soccer and Basketball (blue/green teams)
- **Visual-First GUI**: PyQt6 interface for experiment configuration, rendering, and telemetry
- **Resource Management**: GPU allocation, queue limits, health monitoring
- **Per-Agent Policy Binding**: ``PolicyMappingService`` for routing agents to workers
- **Runtime Logging**: JSONL telemetry per step and episode
- **Sphinx Documentation**: Full docs with video demos, architecture diagrams, and API reference
