.. _advanced:

##################################
Advanced Analysis Methods
##################################

This page documents some of the more advanced methods and features
available in Fermipy:

* :ref:`psmap`: Generate a PS map for a new source
  centered at each spatial bin in the ROI using the algorithm described in
  Bruel P. (2021), A&A, 656, A81. (`doi:10.1051/0004-6361/202141553 <https://arxiv.org/pdf/2109.07443.pdf>`_).

* :ref:`tsmap`: Generate a test statistic (TS) map for a new source
  centered at each spatial bin in the ROI.

* :ref:`tscube`: Generate a TS map using the `gttscube` ST
  application.  In addition to generating a TS map this method can
  also extract a test source likelihood profile as a function of
  energy and position over the whole ROI.

* :ref:`residmap`: Generate a residual map by evaluating the
  difference between smoothed data and model maps (residual) at each
  spatial bin in the ROI.

* :ref:`findsources`: Find new sources using an iterative
  source-finding algorithim.  Adds new sources to the ROI by looking
  for peaks in the TS map.

* :ref:`sed`: Extract the spectral energy distribution of a source
  with the `~fermipy.gtanalysis.GTAnalysis.sed` method.  This method
  fits the source amplitude in a sequence of energy bins.

* :ref:`curvature`: Quickly test for spectral curvature with the
  `~fermipy.gtanalysis.GTAnalysis.curvature` method.

* :ref:`lightcurve`: Extract the lightcurve of a source with the
  `~fermipy.gtanalysis.GTAnalysis.lightcurve` method.  This method
  fits the source amplitude in a sequence of time bins.
  
* :ref:`extension`: Fit the angular extension of a source with the
  `~fermipy.gtanalysis.GTAnalysis.extension` method.

* :ref:`localization`: Find the best-fit position of a source with the
  `~fermipy.gtanalysis.GTAnalysis.localize` method.

* :ref:`phased`: Instructions for performing a phased-selected analysis.
  
* :ref:`sensitivity`: Scripts and classes for estimating sensitivity.

.. toctree::
   :hidden:
   :maxdepth: 1

   sed
   curvature
   lightcurve
   extension
   psmap
   tsmap
   tscube
   residmap
   detection
   localization
   phased
   sensitivity
