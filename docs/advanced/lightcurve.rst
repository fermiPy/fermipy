.. _lightcurve:

Light Curves
============

:py:meth:`~fermipy.gtanalysis.GTAnalysis.lightcurve` can be used to
fit a source in a sequence of time bins.

Examples
--------

.. code-block:: python
   
   # Generate a lightcurve with two bins
   lc = gta.lightcurve('sourceA', nbins=2)

   # Generate a lightcurve with 1-week binning
   lc = gta.lightcurve('sourceA', binsz=86400.*7.0)
   

Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.lightcurve
   :noindex:

