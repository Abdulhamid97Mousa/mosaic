Quickstart
==========

This guide will get you running your first MOSAIC experiment in minutes.

Launching the Interface
-----------------------

.. code-block:: bash

   python -m gym_gui

This opens the MOSAIC visual interface with:

- **Environment Selection**: Choose from Gymnasium, PettingZoo, ViZDoom, etc.
- **Agent Configuration**: Assign policies to agents
- **Training Controls**: Start, pause, and monitor training

Running a Single-Agent Environment
----------------------------------

1. Select **Single Agent** tab
2. Choose **Gymnasium** > **Classic Control** > **CartPole-v1**
3. Click **Load Environment**
4. Select **Human Keyboard** as the actor
5. Click **Play** and use arrow keys to control

Running a Multi-Agent Environment
---------------------------------

1. Select **Multi Agent** tab
2. Choose **PettingZoo** > **Classic** > **Chess**
3. Click **Load Environment**
4. Configure agents:
   - Player 0: Human Keyboard
   - Player 1: Random Policy
5. Click **Play**

Using the Advanced Tab
----------------------

The Advanced tab provides fine-grained control:

1. **Environment Selector**: Choose environment with paradigm auto-detection
2. **Agent Config Table**: Per-agent policy and worker assignment
3. **Worker Config Panel**: Worker-specific settings (learning rate, etc.)
4. **Run Mode Selector**: Interactive, Headless, or Evaluation mode
