.. _changelog:

Changelog
=========

This page is a changelog for releases of Fermipy.  You can also browse
releases on `Github <https://github.com/fermiPy/fermipy/releases>`_.

0.13.0 (01/16/2017)
-------------------
* Rewrite LTCube class to add support for fast LT cube generation.
  The ``gtlike.use_local_ltcube`` option can be used to enable the
  python-based LT cube calculation in lieu of ``gtltcube``.
* Bug fixes and improvements to lightcurve method (see `#102
  <https://github.com/fermiPy/fermipy/pull/102>`_).  Python-based LT
  cube generation is now enabled by default resulting in much faster
  execution time when generating light curves over long time spans.
* Add ``fit_position`` option to
  `~fermipy.gtanalysis.GTAnalysis.extension` that can be used to
  enable a joint fit of extension and position.
* New scheme for auto-generating parameter docstrings.
* Add new `~fermipy.gtanalysis.GTAnalysis.set_source_morphology`
  method to update the spatial model of a source at runtime.
* Major refactoring of `~fermipy.gtanalysis.GTAnalysis.extension` and
  `~fermipy.gtanalysis.GTAnalysis.localize` (see `#106
  <https://github.com/fermiPy/fermipy/pull/106>`_ and `#110
  <https://github.com/fermiPy/fermipy/pull/110>`_).
* Pulled in many new modules and scripts for diffuse all-sky analysis
  (see `#105 <https://github.com/fermiPy/fermipy/pull/105>`_).

0.12.0 (11/20/2016)
-------------------
* Add support for phased analysis (`#87
  <https://github.com/fermiPy/fermipy/pull/87>`_). ``gtlike.expscale``
  and ``gtlike.src_expscale`` can be used to apply a constant exposure
  correction to a whole component or individual sources within a
  component.  See :ref:`phased` for examples.
* Add script and tools for calculating flux sensitivity (`#88
  <https://github.com/fermiPy/fermipy/pull/88>`_ and `#95
  <https://github.com/fermiPy/fermipy/pull/95>`_).  The
  ``fermipy-flux-sensitivity`` script evaluates both the differential
  and integral flux sensitivity for a given TS threshold and minimum
  number of detected counts.  See :ref:`sensitivity` for examples.
* Add ``fermipy-healview`` script for generating images of healpix
  maps and cubes.
* Improvements to HPX-related classes and utilities.
* Refactoring in ``irfs`` module to support development of new
  validation tools.
* Improvements to configuration handling to allow parameter validation
  when updating configuration at runtime.
* Add lightcurve method (`#80
  <https://github.com/fermiPy/fermipy/pull/80>`_).  See
  :ref:`lightcurve` for documentation.
* Change convention for flux arrays in source object.  Values and
  uncertainties are now stored in separate arrays (e.g. ``flux`` and
  ``flux_err``).  
* Add :ref:`Docker-based installation <dockerinstall>` instructions.
  This can be used to run the RHEL6 SLAC ST builds on any machine that
  supports Docker (e.g. OSX Yosemite or later).
* Adopt changes to column name conventions in SED format.  All column
  names are now lowercase.

0.11.0 (08/24/2016)
-------------------
* Add support for weighted likelihood fits (supported in ST
  11-03-00 or later).  Weights maps can be specified with the ``wmap``
  parameter in :ref:`config_gtlike`.
* Implemented performance improvements in
  `~fermipy.gtanalysis.GTAnalysis.tsmap` including switching to
  newton's method for step-size calculation and masking of empty
  pixels (see `#79 <https://github.com/fermiPy/fermipy/pull/79>`_).
* Ongoing development and refactoring of classes for dealing with
  CastroData (binned likelihood profiles).
* Added `~fermipy.gtanalysis.GTAnalysis.reload_sources` method for
  faster recomputation of source maps.
* Fixed sign error in localization plotting method that gave wrong
  orientation for error ellipse..
* Refactored classes in `~fermipy.spectrum` and simplified interface
  for doing spectral fits (see `#69
  <https://github.com/fermiPy/fermipy/pull/69>`_).
* Added DMFitFunction spectral model class in
  `~fermipy.spectrum` (see `#66
  <https://github.com/fermiPy/fermipy/pull/66>`_).  This uses the same
  lookup tables as the ST DMFitFunction class but provides a pure
  python implementation which can be used independently of the STs.
  
0.10.0 (07/03/2016)
-------------------

* Implement support for more spectral models
  (DMFitFunction, EblAtten, FileFunction, Gaussian).
* New options (``outdir_regex`` and ``workdir regex``) for
  fine-grained control over input/output file staging.
* Add ``offset_roi_edge`` to source dictionary.  Defined as the
  distance from the source position to the edge of the ROI (< 0 =
  inside the ROI, > 0 = outside the ROI).
* Add new variables in `~fermipy.gtanalysis.GTAnalysis.fit` output
  (``edm``, ``fit_status``).
* Add new package scripts (``fermipy-collect-sources``,
  ``fermipy-cluster-sources``).
* Various refactoring and improvements in code for dealing with castro
  data.
* Add ``MODEL_FLUX`` and ``PARAMS`` HDUs to SED FITS file.  Many new
  elements added SED output dictionary.
* Support NEWTON fitter with the same interface as MINUIT and
  NEWMINUIT.  Running `~fermipy.gtanalysis.GTAnalysis.fit` with
  ``optimizer`` = NEWTON will use the NEWTON fitter where applicable
  (only free norms) and MINUIT otherwise.  The ``optimizer`` argument
  to `~fermipy.gtanalysis.GTAnalysis.sed`,
  `~fermipy.gtanalysis.GTAnalysis.extension`, and
  `~fermipy.gtanalysis.GTAnalysis.localize` can be used to override
  the default optimizer at runtime.  Note that the NEWTON fitter is
  only supported by ST releases *after* 11-01-01.

  
0.9.0 (05/25/2016)
------------------

* Bug fixes and various refactoring in TSCube and CastroData.  Classes
  for reading and manipulating bin-by-bin likelihoods are now moved to
  the `~fermipy.castro` module.
* Rationalized naming conventions for energy-related variables.
  Properties and method arguments with units of the logarithm of the
  energy now consistently contain ``log`` in the name.

  * `~fermipy.gtanalysis.GTAnalysis.energies` now returns bin energies
    in MeV (previously it returned logarithmic energies).
    `~fermipy.gtanalysis.GTAnalysis.log_energies` can be used to
    access logarithmic bin energies.
  * Changed ``erange`` parameter to ``loge_bounds`` in the methods
    that accept an energy range.
  * Changed the units of ``emin``, ``ectr``, and ``emax`` in the sed
    output dictionary to MeV.
    
* Add more columns to the FITS source catalog file generated by
  `~fermipy.gtanalysis.GTAnalysis.write_roi`.  All float and string
  values in the source dictionary are now automatically included in
  the FITS file.  Parameter values, errors, and names are written to
  the ``param_values``, ``param_errors``, and ``param_names`` vector
  columns.

* Add package script for dispatching batch jobs to LSF (``fermipy-dispatch``).

* Fixed some bugs related to handling of unicode strings.

  
0.8.0 (05/18/2016)
------------------

* Added new variables to source dictionary:
  
  * Likelihood scan of source normalization (``dloglike_scan``,
    ``eflux_scan``, ``flux_scan``).
  * Source localization errors (``pos_sigma``,
    ``pos_sigma_semimajor``, ``pos_sigma_semiminor``, ``pos_r68``,
    ``pos_r95``, ``pos_r99``, ``pos_angle``).  These are automatically
    filled when running `~fermipy.gtanalysis.GTAnalysis.localize` or
    `~fermipy.gtanalysis.GTAnalysis.find_sources`.
    
* Removed camel-case in some source variable names.
* Add ``cacheft1`` option to :ref:`config_data` disable caching FT1
  files.  Cacheing is still enabled by default.
* Support FITS file format for preliminary releases of the 4FGL
  catalog.
* Add ``__future__`` statements throughout to ensure
  forward-compatibility with python3.
* Reorganize utility modules including those for manipulation of WCS
  and healpix images.
* Various improvements and refactoring in
  `~fermipy.gtanalysis.GTAnalysis.localize`.  This method now moved to
  the `~fermipy.sourcefind` module.
* Add new global parameter ``llscan_pts`` in :ref:`config_gtlike` to
  define the number of likelihood evaluation points.
* Write output of `~fermipy.gtanalysis.GTAnalysis.sed` to a FITS file
  in the Likelihood SED format.  More information about the
  Likelihood SED format is available on this `page
  <http://gamma-astro-data-formats.readthedocs.io/en/latest/results/binned_likelihoods/index.html>`_.
* Write ROI model to a FITS file when calling
  `~fermipy.gtanalysis.GTAnalysis.write_roi`.  This file contains a
  BINTABLE with one row per source and uses the same column names as
  the 3FGL catalog file to describe spectral parameterizations.  Note
  that this file currently only contains a subset of the information
  available in the numpy output file.
* Reorganize classes and methods in `~fermipy.sed` for manipulating
  and fitting bin-by-bin likelihoods.  Spectral functions moved to a
  dedicated `~fermipy.spectrum` module.
* Write return dictionary to a numpy file in
  `~fermipy.gtanalysis.GTAnalysis.residmap` and
  `~fermipy.gtanalysis.GTAnalysis.tsmap`.
  
  
