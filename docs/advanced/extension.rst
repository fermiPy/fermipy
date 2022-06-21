.. _extension:

Extension Fitting
=================

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.extension` method
executes a source extension analysis for a given source by computing a
likelihood ratio test with respect to the no-extension (point-source)
hypothesis and a best-fit model for extension.  The best-fit extension
is found by performing a likelihood profile scan over the source width
(68% containment) and fitting for the extension that maximizes the
model likelihood.  Currently this method supports two models for
extension: a 2D Gaussian (*RadialGaussian*) or a 2D disk
(*RadialDisk*).

At runtime the default settings for the extension analysis can be
overriden by passing one or more *kwargs* when executing
:py:meth:`~fermipy.gtanalysis.GTAnalysis.extension`:

.. code-block:: python
   
   # Run extension fit of sourceA with default settings
   >>> gta.extension('sourceA')

   # Override default spatial model
   >>> gta.extension('sourceA', spatial_model='RadialDisk')

By default the method will fix all background parameters before
performing the extension fit.  One can leave background parameters
free by setting ``free_background=True``:

.. code-block:: python
   
   # Free a nearby source that maybe be partially degenerate with the
   # source of interest.  The normalization of SourceB will be refit
   # when testing the extension of sourceA   
   gta.free_norm('sourceB')
   gta.extension('sourceA', free_background=True)

   # Fix all background parameters when testing the extension
   # of sourceA
   gta.extension('sourceA', free_background=False)

   # Free normalizations of sources within 2 degrees of sourceA
   gta.extension('sourceA', free_radius=2.0)

.. warning::

    If the best-fit extension differs significantly from the nominal value, the user is recommended to re-optimize the ROI.

Energy-dependent Extension Fit
------------------------------

Use the option ``fit_ebin=True`` to perform separate exention scans
in each energy bin (in addition to the joint fit):

.. code-block:: python

   gta.extension('sourceA', fit_ebin=True, loge_bins=np.linspace(3, 6.5, 15) )

By default, the method will use the energy bins of the underlying
analysis.  The ``loge_bins`` keyword argument can be used to override
the default binning with the restriction that the SED energy bins
must align with the analysis bins. The bins used in the analysis can be
found with ``gta.log_energies``. For example if in the analysis
8 energy bins per decade are considered and you want to make the SED in 4 bins
per decade you can specify ``loge_bins=gta.log_energies[::2]``.


User-defined Fit Range
----------------------

The default extension scan covers 21 points between 0.01 deg and 1.0 deg,
evenly spaced in log space. The user may overwrite the fit range
(``width_min`` and ``width_max`` keywords) and/or the number of
points for the scan (``width_nstep``) or directly supply a list
of extensions to be tested (``width`` keyword).

The point source hypothesis is always tested as well. If the extension
range includes the nominal extension value used during ROI optimization,
this value will be explicitly scanned as well.

.. warning::
    The wider the range of extension values to be fit over, the more important it is to consider freeing the normalization of the background sources to avoid artifacts.


Extension Dictionary
--------------------

The results of the extension analysis are written to a dictionary
which is the return value of the extension method.  
   
.. code-block:: python
   
   ext = gta.extension('sourceA', write_npy=True, write_fits=True)
   
The contents of the output dictionary are given in the following table:

.. csv-table:: *extension* Output Dictionary
   :header:    Key, Type, Description
   :file: ../config/extension_output.csv
   :delim: tab
   :widths: 10,10,80


Configuration
-------------

The default configuration of the method is controlled with the
:ref:`config_extension` section of the configuration file.  The default
configuration can be overriden by passing the option as a *kwargs*
argument to the method.

.. csv-table:: *extension* Options
   :header:    Option, Default, Description
   :file: ../config/extension.csv
   :delim: tab
   :widths: 10,10,80
            
Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.extension
   :noindex:


