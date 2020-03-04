.. _fermipy_jobs:

fermipy.jobs subpackage
=======================

The fermipy.jobs sub-package is a light-weight, largely standalone, package
to manage data analysis pipelines.   It allows the user to build up
increasingly complex analysis pipelines from single applications
that are callable either from inside python or from the unix command line.


Subpackage contents
-------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   fermipy_jobs_tools
   fermipy_jobs_multiple_ROIs


Module contents
---------------

.. automodule:: fermipy.jobs
    :members:
    :undoc-members:
    :show-inheritance:


Link class and trivial sub-classes
----------------------------------

.. autoclass:: fermipy.jobs.link.Link
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.jobs.gtlink
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.app_link.AppLink
    :members:
    :undoc-members:
    :show-inheritance:


ScatterGather class
-------------------

.. autoclass:: fermipy.jobs.scatter_gather.ScatterGather
    :members:
    :undoc-members:
    :show-inheritance:

    
Chain class
-----------

.. autoclass:: fermipy.jobs.chain.Chain
    :members:
    :undoc-members:
    :show-inheritance:


High-level analysis classes
---------------------------

These are `Link` sub-classes that implement `fermipy` analyses, 
or perform tasks related to `fermipy` analyses, such as plotting
or collecting results for a set of simulations.

.. autoclass:: fermipy.jobs.target_analysis.AnalyzeROI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_analysis.AnalyzeSED
    :members:
    :undoc-members:
    :show-inheritance:

  
.. autoclass:: fermipy.jobs.target_collect.CollectSED
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.jobs.target_sim.CopyBaseROI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_sim.RandomDirGen
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_sim.SimulateROI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_plotting.PlotCastro
    :members:
    :undoc-members:
    :show-inheritance:


    
High-level analysis job dispatch
--------------------------------

These are `ScatterGather` sub-classes that invoke the `Link` sub-classes
listed above.

.. autoclass:: fermipy.jobs.target_analysis.AnalyzeROI_SG
    :members:
    :undoc-members:
    :show-inheritance:


.. autoclass:: fermipy.jobs.target_analysis.AnalyzeSED_SG
    :members:
    :undoc-members:
    :show-inheritance:

    
.. autoclass:: fermipy.jobs.target_collect.CollectSED_SG
    :members:
    :undoc-members:
    :show-inheritance:
    
.. autoclass:: fermipy.jobs.target_sim.CopyBaseROI_SG
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_sim.RandomDirGen_SG
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_sim.SimulateROI_SG
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fermipy.jobs.target_plotting.PlotCastro_SG
    :members:
    :undoc-members:
    :show-inheritance:


Batch and System Interfaces
---------------------------

.. automodule:: fermipy.jobs.sys_interface
    :members:
    :undoc-members:
    :show-inheritance:
   
.. automodule:: fermipy.jobs.native_impl
    :members:
    :undoc-members:
    :show-inheritance:
   
.. automodule:: fermipy.jobs.slac_impl
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: fermipy.jobs.batch
    :members:
    :undoc-members:
    :show-inheritance:

File Archive module
-------------------

.. automodule:: fermipy.jobs.file_archive
    :members:
    :undoc-members:
    :show-inheritance:

    
Job Archive module
------------------

.. automodule:: fermipy.jobs.job_archive
    :members:
    :undoc-members:
    :show-inheritance:

