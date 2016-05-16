.. _changelog:

Changelog
=========

This page is a changelog for releases of Fermipy.

0.7.1 (May 15, 2016)
---------------------

* Added new variables to source dictionary:
  
  * Likelihood scan of source normalization (``dloglike_scan``, ``eflux_scan``, ``flux_scan``).
  * Source localization errors (``pos_sigma``,
    ``pos_sigma_semimajor``, ``pos_sigma_semiminor``, ``pos_r68``,
    ``pos_r95``, ``pos_r99``, ``pos_angle``).  These are automatically
    filled when running `~fermipy.gtanalysis.GTAnalysis.localize` or
    `~fermipy.gtanalysis.GTAnalysis.find_sources`.
    
* Removed camel-case in some source variable names.
* Add option to disable caching FT1 files (``cacheft1``).
* Support FITS file format for preliminary releases of the 4FGL
  catalog.
* Add ``__future__`` statements throughout to ensure
  forward-compatibility with python3.
* Reorganize utility modules including those for manipulation of WCS
  and healpix images.
* Various improvements and refactoring in
  `~fermipy.gtanalysis.GTAnalysis.localize`.  This method moved to
  `~fermipy.sourcefind` module.
* Add new global parameter ``llscan_pts`` to define the number of
  likelihood evaluation points.
* Write output of `~fermipy.gtanalysis.GTAnalysis.sed` to a FITS file
  in the Likelihood SED format.  More information about the
  Likelihood SED format is available on this `page
  <http://gamma-astro-data-formats.readthedocs.io/en/latest/results/binned_likelihoods/index.html>`_.
* Write ROI model to a FITS file when calling
  `~fermipy.gtanalysis.GTAnalysis.write_roi`.  This file contains a
  BINTABLE with one row per source and uses the same column names as
  the 3FGL catalog file to describe spectral parameterizations.  Note
  that this file only contains a subset of the information available
  in the numpy output file.
* Reorganize classes and methods in `~fermipy.sed` for manipulating
  and fitting bin-by-bin likelihoods.  Spectral functions moved to a
  dedicated `~fermipy.spectrum` module.
  
  
0.7.0 (April 19, 2016)
----------------------

* some features
