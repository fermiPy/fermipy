.. _findsources:

Source Finding
==============

:py:meth:`~fermipy.gtanalysis.GTAnalysis.find_sources` is an iterative
source-finding algorithm that uses peak detection on a TS map to find
new source candidates.  The procedure for adding new sources at each
iteration is as follows:

* Generate a TS map for the test source model defined with the ``model``
  argument.
  
* Identify peaks with sqrt(TS) > ``sqrt_ts_threshold`` and an angular
  distance of at least ``min_separation`` from a higher amplitude peak
  in the map.

* Order the peaks by TS and add a source at each peak starting from
  the highest TS peak.  Set the source position by fitting a 2D
  parabola to the log-likelihood surface around the peak maximum.
  After adding each source, re-fit its spectral parameters.

* Add sources at the N highest peaks up to N = ``sources_per_iter``.

Source finding is repeated up to ``max_iter`` iterations or until no
peaks are found in a given iteration.  Sources found by the method are
added to the model and given designations *PS JXXXX.X+XXXX* according
to their position in celestial coordinates.
  
Examples
--------

.. code-block:: python

   model = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}
   srcs = gta.find_sources(model=model, sqrt_ts_threshold=5.0,
                           min_separation=0.5)

The method for generating the TS maps can be controlled with the
``tsmap_fitter`` option.  TS maps can be generated with either
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` or
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tscube`.


Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.find_sources
   :noindex:
