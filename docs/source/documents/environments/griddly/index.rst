Griddly
=======

.. figure:: /images/envs/griddly/griddly.gif
   :alt: Griddly environments showcase
   :align: center
   :width: 80%

   Griddly: High-performance grid-based research platform

Overview
--------

**Griddly** is a high-performance grid world research platform with a C++ backend
and Vulkan GPU rendering capable of 30,000+ FPS in headless training mode.

:Install: ``pip install -e ".[griddly]"``
:Paper: `Bamford et al. (2021) <https://arxiv.org/abs/2011.06363>`_
:Repo: https://github.com/Bam4d/Griddly
:Docs: https://griddly.readthedocs.io

Key Features
------------

- **High Performance**: 30,000+ FPS headless training with Vulkan GPU rendering
- **Flexible Design**: Define custom games using GDY (Griddly Description YAML)
- **Rich Environment Suite**: 30+ pre-built environments spanning puzzles, RTS, and multi-agent games
- **Partial Observability**: Built-in support for limited field-of-view observations
- **Multi-Agent Support**: Both cooperative and competitive multi-agent scenarios

Environment Categories
----------------------

**Single-Agent Puzzles**
   - Zelda, Sokoban, Clusters, Bait, Zen Puzzle, Labyrinth, Cook Me Pasta
   - Spiders, Spider Nest, Butterflies and Spiders, Random Butterflies
   - Eyeball, Drunk Dwarf, Doggo

**Multi-Agent RTS**
   - GriddlyRTS, Push Mania, Kill The King, Heal Or Die

**Multi-Agent Cooperative/Competitive**
   - Robot Tag (4v4, 8v8, 12v12), Foragers

Partial Observability
----------------------

Many Griddly environments support partial observability with limited field-of-view:

- ``GDY-Partially-Observable-Zelda-v0``
- ``GDY-Partially-Observable-Sokoban---2-v0``
- ``GDY-Partially-Observable-Clusters-v0``
- ``GDY-Partially-Observable-Bait-v0``
- ``GDY-Partially-Observable-Zen-Puzzle-v0``
- ``GDY-Partially-Observable-Labyrinth-v0``
- ``GDY-Partially-Observable-Cook-Me-Pasta-v0``

Available Environments
----------------------

MOSAIC supports all Griddly environments through the ``GriddlyAdapter``:

.. code-block:: python

   from gym_gui.core.enums import GameId
   from gym_gui.core.factories.adapters import create_adapter

   # Single-agent puzzle
   adapter = create_adapter(GameId.GRIDDLY_ZELDA)

   # Multi-agent RTS
   adapter = create_adapter(GameId.GRIDDLY_KILL_THE_KING)

   # Multi-agent cooperative
   adapter = create_adapter(GameId.GRIDDLY_FORAGERS)

Performance Characteristics
---------------------------

Griddly's C++ backend and Vulkan rendering make it ideal for:

- **Large-scale training**: 30,000+ FPS enables rapid policy iteration
- **Curriculum learning**: Fast environment resets support complex curricula
- **Multi-agent research**: Efficient parallel execution of multiple agents
- **Procedural generation**: Quick generation of diverse training scenarios

System Requirements
-------------------

**Required:**
   - Vulkan-compatible GPU drivers
   - Python 3.10+

**Installation:**

.. code-block:: bash

   # Install Griddly support
   pip install -e ".[griddly]"

   # Verify Vulkan drivers
   vulkaninfo | grep "Vulkan Instance Version"

Common Issues
-------------

**Vulkan not found**
   Install Vulkan drivers for your GPU:

   .. code-block:: bash

      # Ubuntu/Debian
      sudo apt-get install vulkan-tools libvulkan1

      # Verify installation
      vulkaninfo

**Multi-agent action space errors**
   Some Griddly multi-agent environments use ``MultiAgentActionSpace`` which is not
   compatible with MOSAIC's single-agent adapter. Use the environments listed above
   which have been verified to work with MOSAIC.

Citation
--------

If you use Griddly in your research, please cite:

.. code-block:: bibtex

   @inproceedings{bamford2021griddly,
     title={Griddly: A Platform for AI Research in Games},
     author={Bamford, Chris and Bignell, Simon and Lucas, Simon M},
     booktitle={2021 IEEE Conference on Games (CoG)},
     pages={1--8},
     year={2021},
     organization={IEEE}
   }

See Also
--------

- :doc:`../minigrid/index` - Similar grid-based environments with different focus
- :doc:`../mosaic_multigrid/index` - Multi-agent grid worlds for team coordination
- :doc:`../procgen/index` - Procedurally generated environments
