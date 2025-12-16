Changelog
=========

All notable changes to MOSAIC will be documented here.

[Unreleased]
------------

Added
^^^^^

- Multi-paradigm orchestration (SINGLE_AGENT, SEQUENTIAL, SIMULTANEOUS, HIERARCHICAL)
- PolicyMappingService for per-agent policy binding
- Visual-first interface with PyQt6
- Support for heterogeneous agents (Human, RL, BDI, LLM)
- CleanRL worker integration
- XuanCe worker integration (planned)
- RLlib worker integration (planned)
- Jason BDI worker integration
- PettingZoo environment support
- Gymnasium environment support
- ViZDoom environment support

Changed
^^^^^^^

- Refactored adapter layer for paradigm awareness
- Renamed ``ui/environments/`` to ``ui/config_panels/``

Fixed
^^^^^

- Large file handling for GitHub (godot binary excluded)

[0.1.0] - 2024-XX-XX
--------------------

Initial release of MOSAIC.
