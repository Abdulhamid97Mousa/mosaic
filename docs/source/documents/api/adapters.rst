Adapters API
============

Adapter classes for different environment types.  Adapters bridge the
gap between raw environment APIs and MOSAIC's unified operator
interface, translating observations and actions so that any agent type
(LLM, RL, Human, or Random) can interact with any supported
environment.

EnvironmentAdapter
------------------

.. autoclass:: gym_gui.core.adapters.base.EnvironmentAdapter
   :members:
   :undoc-members:
   :no-index:

PettingZooAdapter
-----------------

.. autoclass:: gym_gui.core.adapters.pettingzoo.PettingZooAdapter
   :members:
   :undoc-members:
   :no-index:

ParadigmAdapter
---------------

.. autoclass:: gym_gui.core.adapters.paradigm.ParadigmAdapter
   :members:
   :undoc-members:
