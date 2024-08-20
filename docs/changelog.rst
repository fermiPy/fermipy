.. _changelog:

Changelog
=========

This page is a changelog for releases of Fermipy.  You can also browse
releases on `Github <https://github.com/fermiPy/fermipy/releases>`_.

1.3.1 (08/20/2024)
------------------
* Requirements of astropy and gammapy updated

1.3.0 (08/20/2024)
------------------
* Added `tutorial notebooks <https://fermipy.readthedocs.io/en/master/tutorials.html>`_ to the documentation
* Published `organization chart <https://fermipy.readthedocs.io/en/master/team.html>`_
* Added PS map implementing code from Philippe Bruel in `~fermipy.gtanalysis.GTAnalysis.psmap`
* Added PS map visualition in `~fermipy.gtanalysis.plotter.make_psmap_plots`
* Added `PS map <https://fermipy.readthedocs.io/en/master/advanced/psmap.html>`_ to the documentation

1.2.2 (01/21/2024)
------------------
* fix the dependence of scipy due to gammapy

1.2.1 (12/08/2023)
------------------
* Small bug fixes.
* pinned astropy<6

1.2 (09/21/2022)
----------------
* Small bug fixes.
* No more glaring issues remain open.

1.1.6 (07/19/2022)
------------------
* Allow user-defined energy bins for energy-dependent extension fit.
* Make sure the last time bin in the light curve ends at ``tmax``.

1.1.5 (07/18/2022)
------------------
* Small fixes in lightcurve code (see `#472 <https://github.com/fermiPy/fermipy/issues/472>`_ and `#467 <https://github.com/fermiPy/fermipy/issues/467>`_)
* Install ``fermitools`` release version 2.2.0 or higher.
* Minor other fixes.

1.1.4 (06/24/2022)
------------------
* Compatibility with numpy 1.23

1.1.3 (06/04/2022)
------------------
* Update interface to astropy.fits.writeto for compatibility w/ astropy 5.1

1.1.2 (05/26/2022)
------------------
* Minor bug fixes & doc updates.
* Implemented new super-exponential cutoff PL for curvature test.

1.1.1 (05/13/2022)
------------------
* More bug fixes for lightcurve module

1.1 (05/10/2022)
----------------
* Fixes to work with recent astropy and matplotlib releases
* Bug fixes for lightcurve module

1.0.1 (03/12/2021)
------------------
* Switch primaty installation method to conda
* Fixes to work with recent gammapy and yaml releases
* Remove old installation scripts
* Switch to github actions for testing

1.0.0 (10/1/2020)
-----------------
* First working python3 version, very close to 0.20.0

0.20.0 (08/24/2020)
-------------------
* Upgrade to P8R3_XXX_V3 irfs in testing
* Added improvements to extension fitting and residual maps

0.19.0 (03/25/2020)
-------------------
* Switch installation procedure and to using fermitools-data for diffuse emission models
* Update docs to reflect changers in intallation procedure
* Added new version of 4FGL catalog
* Updated automatic testing to work with 4FGL and new diffuse emisison models

0.18.1 (03/10/2020)
-------------------
* Changes to improve automated builds for testing
* Added documentation for fermipy.jobs and femripy.diffuse modules
* Some changes to support different size pixels in different components
* Added gtpsf and gtdrm support to gtanalysis

0.18.0 (10/18/2019)
-------------------
* Added tools to use new version of energy dispersion from fermitools.
* Changed travis testing and conda scripts to work with conda release of fermitools.


0.17.4 (3/13/2019)
------------------
* Including 4FGL, interstellar emission model and isotropic templates files.
* Changing catalog.py to use 4FGL catalog
* Minor fixes to target_collect.py, lightcurve.py and gta.simulate()


0.17.3 (7/12/2018)
------------------

* Added fitting code to fermipy.diffuse module
* Improved fermipy.jobs to deal with analyses with multiple components
* Added capability to plot global minimum in castro plots
* Added spectra for dark matter decay to DMFitFunction
* Added code to split simulations into smaller batch jobs
* Added fixed shape lightcurve to correct TS_var computation


0.17.2 (5/30/2018)
------------------

* Added lots of documentation for the fermipy.jobs module.
* Minor changes to the fermipy.jobs module to work with the
  dark matter analysis pipeline (dmpipe). 


0.17.1 (5/23/2018)
------------------

* Patch release to get versioning working with GitHub release system.


0.17.0 (5/22/2018)
------------------

* The LogParabola, PowerLawSuperExponetial and Dark Matter SEDs have been added to the sensitivity.py script.
* There are a lot of additions to perform a stacking analysis. This can be applied for instance for the search of dark matter with a stacking analysis of Milky Way dSphs, Galaxy Clusters or other galaxies.
* It contains scripts to send jobs to SLAC Batch Farm and collect the results.
* It includes scripts and functions to perform all sky fits.
* It also fixes a few issues with glon and glat in the localization (#225), and the wrong orientation of residual and TS maps (#216)


0.16.0 (12/27/2017)
-------------------

* Improvements and refactoring in the internals of the ``lightcurve``
  method (see `#156 <https://github.com/fermiPy/fermipy/pull/156>`_,
  `#157 <https://github.com/fermiPy/fermipy/pull/157>`_, `#160
  <https://github.com/fermiPy/fermipy/pull/160>`_, `#161
  <https://github.com/fermiPy/fermipy/pull/161>`_, `#162
  <https://github.com/fermiPy/fermipy/pull/162>`_).  Resolve fit
  stability issues that were arising when the source of interest was
  not significantly detected in a given time bin.  Added options to
  speed up source map calculation by rescaling source maps (enabled
  with ``use_scaled_srcmap=True``) and split the lightcurve
  calculation across N cores (enabled with ``multithread=True`` and
  ``nthread=N``).  Add calculation of ``TS_var`` to test for
  variability using method from the 2FGL.
* Updates to validation tools.  Added MeritSkimmer script
  (``fermipy-merit-skimmer``) for skimming ROOT merit tuples either
  locally or on xrootd.

0.15.0 (09/05/2017)
-------------------

* Bug fix related to restoring analysis state for phased analysis
  (scaled exposure).
* Many improvements and feature additions to senstivity tools (see e.g. `#148
  <https://github.com/fermiPy/fermipy/pull/148>`_, `#149
  <https://github.com/fermiPy/fermipy/pull/149>`_, and `#152
  <https://github.com/fermiPy/fermipy/pull/152>`_).
* Various updates to support DM pipeline package (`#146
  <https://github.com/fermiPy/fermipy/pull/146>`_).
* Improve robustness of algorithms for extracting peak and
  uncertainty ellipse from 2D likelihood surface.
* Added `~fermipy.gtanalysis.GTAnalysis.curvature` method for testing a
  source for spectral curvature.
* Added ``fix_shape`` option to
  `~fermipy.gtanalysis.GTAnalysis.extension` and
  `~fermipy.gtanalysis.GTAnalysis.localize` to fix spectral shape
  parameters.  Spectral shape parameters of the source of interest are
  now free by default when localizing or fitting extension.
  

0.14.0 (03/28/2017)
-------------------
* Refactoring and improvements in
  `~fermipy.gtanalysis.GTAnalysis.localize` and
  `~fermipy.gtanalysis.GTAnalysis.extension` (see `#124
  <https://github.com/fermiPy/fermipy/pull/124>`_).  Cleanup of
  columns in `~fermipy.gtanalysis.GTAnalysis.localize`.  Add new
  columns for 1-sigma errors projected in CEL and GAL coordinates as
  well as associated covariance and correlation matrices.  Add
  positional errors when running
  `~fermipy.gtanalysis.GTAnalysis.extension` with
  ``fit_position=True``.
* Add ``free_radius`` option to
  `~fermipy.gtanalysis.GTAnalysis.localize`,
  `~fermipy.gtanalysis.GTAnalysis.extension`, and
  `~fermipy.gtanalysis.GTAnalysis.sed`.  This can be used to free
  background sources within a certain distance of the analyzed source.
* Relocalize point-source hypothesis when testing extension of
  extended sources.
* Improve speed and accuracy of source map calculation (see `#123
  <https://github.com/fermiPy/fermipy/pull/123>`_).  Exposures are now
  extracted directly from the exposure map.
* Write analysis configuration to ``CONFIG`` header keyword of all
  FITS output files.
* Add ``jobs`` and ``diffuse`` submodules (see `#120
  <https://github.com/fermiPy/fermipy/pull/120>`_ and `#122
  <https://github.com/fermiPy/fermipy/pull/120>`_).  These contain
  functionality for peforming all-sky diffuse analysis and setting up
  automated analysis pipelines.  More detailed documentation on these
  features to be provided in a future release.
  
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
  
  
