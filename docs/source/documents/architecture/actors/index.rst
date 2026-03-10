Actors
======

An **Actor** is the inference-time counterpart to a Worker.  Where Workers run
training pipelines in isolated subprocesses, Actors live inside the GUI process
and are responsible for selecting actions during evaluation and interactive
play, without starting any training loop.

.. toctree::
   :maxdepth: 2
   :caption: Actor Documentation

   concept
   architecture
   lifecycle
   actor_types
