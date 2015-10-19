.. _extension:

Extension Analysis
==================

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.extension` method executes
a source extension analysis for a given source by computing a
likelihood ratio test with respect to the no-extension (point-source)
hypothesis and a best-fit model for extension.  The best-fit extension
is evaluated by a likelihood profile scan over the source width.
Currently this method supports two models for extension: a 2D Gaussian
(*GaussianSource*) or a 2D disk (*DiskSource*).

The configuration of
:py:meth:`~fermipy.gtanalysis.GTAnalysis.extension` can be controlled
with the *extension* block of the configuration file:

.. code-block:: yaml
   
   extension:
     spatial_model : GaussianSource
     width_min : 0.01
     width_max : 1.0
     width_nstep : 21
     
At runtime the default settings for the extension analysis can be
overriden by supplying one or more *kwargs* when executing
:py:meth:`~fermipy.gtanalysis.GTAnalysis.extension`:

.. code-block:: python
   
   # Run analysis with default settings
   gta.extension('sourceA')

   # Override spatial model
   gta.extension('sourceA',spatial_model='DiskSource')

By default the extension method will refit all background parameters
that were free when the method was executed.  One can optionally fix
all background parameters with the *fix_background* parameter:

.. code-block:: python
   
   # Free a nearby source that maybe be partially degenerate with the
   # source of interest
   gta.free_source_norm('sourceB')

   # Normalization of SourceB will be refit when testing the extension
   # of sourceA
   gta.extension('sourceA')

   # Fix all background parameters when testing the extension
   # of sourceA
   gta.extension('sourceA',fix_background=True)

The results of the extension analysis are written to a dictionary
which is the return argument of the extension method.  This dictionary
is also written to the *extension* dictionary of the corresponding
source and thus will also be contained in the output file generated
with :py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.
   
.. code-block:: python
   
   ext = gta.extension('sourceA')

   print ext.keys()
   ['fit', 'ext_ul95', 'ext_err_lo', 'dlogLike', 'ts_ext', 'ext', 'logLike', 'ext_err_hi', 'width']
   
The contents of the output dictionary are described in the following table:

========== =================================================================
Key        Description
========== =================================================================
dlogLike   Sequence of delta-log-likelihood values for each point
           in the profile likelihood scan.
logLike    Sequence of likelihood values for each point in the profile likelihood scan.
ext        Best-fit extension in degrees.
ext_err_hi Upper (1 sigma) error on the best-fit extension in degrees.
ext_err_lo Lower (1 sigma) error on the best-fit extension in degrees.
ext_ul95   95% CL upper limit on the extension in degrees.
width      List of width parameters.
ts_ext     Test statistic for the extension hypothesis.
fit        Sequence of source dictionaries with best-fit source
           parameters for each point in the likelihood scan.
config     Copy of the input parameters to this method.
========== =================================================================


Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.extension
   :noindex:


