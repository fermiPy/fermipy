.. _advanced:

##################################
Advanced Analysis Methods
##################################

This page documents some of the more advanced methods and features
available in Fermipy:

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

* :ref:`sed`: Compute the spectral energy distribution of a source by
  fitting its amplitude in a sequence of energy bins.

* :ref:`lightcurve`: Compute the lightcurve of a source by fitting its
  amplitude in a sequence of time bins.
  
* :ref:`extension`: Fit the angular extension of a source.

* :ref:`localization`: Find the best-fit position of a source.

* :ref:`phased`: Instructions for performing a phased-selected analysis.
  
* :ref:`sensitivity`: Scripts and classes for estimating sensitivity.

.. toctree::
   :hidden:
   :maxdepth: 1

   sed
   lightcurve
   extension
   tsmap
   tscube
   residmap
   detection
   localization
   phased
   sensitivity
