Griddly
=======

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://private-user-images.githubusercontent.com/80536675/560458741-ae51195e-04f2-453d-b2c0-9110fc5c1688.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <br><br>

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
   - Full human keyboard control supported: each player commands their units
     with WASD / arrow keys; MOSAIC automatically resolves the target unit
     via the Griddly game state API

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

**Multi-agent action space**
   All Griddly multi-agent environments are fully supported, including RTS-style
   games (Push Mania, GriddlyRTS, Kill The King, Heal Or Die) and cooperative
   games (Foragers, Robot Tag).  Connect two USB keyboards, assign each to a
   player in the *Keyboard Assignment* panel, and use WASD / arrow keys to play.

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
