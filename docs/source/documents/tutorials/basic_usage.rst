Basic Usage
===========

This guide covers the fundamental concepts for using MOSAIC.

Understanding Paradigms
-----------------------

MOSAIC supports four stepping paradigms:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Paradigm
     - Description
     - Use Cases
   * - SINGLE_AGENT
     - One agent interacts with environment
     - CartPole, Atari, MuJoCo
   * - SEQUENTIAL
     - Agents take turns (AEC)
     - Chess, Go, Turn-based games
   * - SIMULTANEOUS
     - All agents act together (POSG)
     - MPE, Cooperative control

Policy Mapping
--------------

Use ``PolicyMappingService`` to assign policies to agents:

.. code-block:: python

   from gym_gui.services import PolicyMappingService
   from gym_gui.core.enums import SteppingParadigm

   # Create service
   policy_service = PolicyMappingService()

   # Set paradigm
   policy_service.set_paradigm(SteppingParadigm.SEQUENTIAL)

   # Bind policies
   policy_service.bind_agent_policy("player_0", "human")
   policy_service.bind_agent_policy("player_1", "trained_ppo")

Available Actors
----------------

MOSAIC provides several built-in actors:

- **Human Keyboard**: Manual control via keyboard
- **Human Mouse**: Manual control via mouse clicks
- **Random Policy**: Uniform random action selection
- **CleanRL Worker**: Trained RL policies (PPO, DQN, SAC, TD3)
- **Scripted**: Custom Python logic

Loading Trained Policies
------------------------

.. code-block:: python

   # Load a trained CleanRL model
   policy_service.bind_agent_policy(
       "player_1",
       "cleanrl_ppo",
       config={
           "model_path": "var/policies/cartpole_ppo.pt",
           "deterministic": True
       }
   )
