RWARE (Robotic Warehouse)
=========================

Cooperative multi-agent shelf delivery in a simulated warehouse.

:Install: ``pip install -e 3rd_party/robotic-warehouse/``
:Paradigm: Multi-agent (simultaneous)
:Stepping: ``SIMULTANEOUS``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - rware-tiny-2ag/4ag-v2
     - Tiny warehouse (2 or 4 agents)
   * - rware-small-2ag/4ag-v2
     - Small warehouse
   * - rware-medium-2ag/4ag-v2
     - Medium warehouse (also easy/hard reward variants)
   * - rware-large-4ag/8ag-v2
     - Large warehouse (also hard reward variants)

Citation
--------

.. code-block:: bibtex

   @inproceedings{papoudakis2021rware,
     author       = {Georgios Papoudakis and Filippos Christianos and Lukas Sch{\"a}fer and Stefano V. Albrecht},
     title        = {Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
     booktitle    = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
     year         = {2021},
   }
