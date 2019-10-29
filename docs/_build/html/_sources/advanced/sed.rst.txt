.. _sed:

SED Analysis
============

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` method computes a
spectral energy distribution (SED) by performing independent fits for
the flux normalization of a source in bins of energy.  The
normalization in each bin is fit using a power-law spectral
parameterization with a fixed index.  The value of this index can be
set with the ``bin_index`` parameter or allowed to vary over the
energy range according to the local slope of the global spectral model
(with the ``use_local_index`` parameter).

The ``free_background``, ``free_radius``, and ``cov_scale`` parameters
control how nuisance parameters are dealt with in the fit.  By default
the method will fix the parameters of background components ROI when
fitting the source normalization in each energy bin
(``free_background=False``).  Setting ``free_background=True`` will
profile the normalizations of all background components that were free
when the method was executed.  In order to minimize overfitting,
background normalization parameters are constrained with priors taken
from the global fit.  The strength of the priors is controlled with
the ``cov_scale`` parameter.  A larger (smaller) value of
``cov_scale`` applies a weaker (stronger) constraint on the background
amplitude.  Setting ``cov_scale=None`` performs an unconstrained fit
without priors.

Examples
--------

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` method is executed
by passing the name of a source in the ROI as a single argument.
Additional keyword argument can also be provided to override the
default configuration of the method:

.. code-block:: python
   
   # Run analysis with default energy binning
   sed = gta.sed('sourceA')

   # Override the energy binning and the assumed power-law index
   # within the bin   
   sed = gta.sed('sourceA', loge_bins=[2.0,2.5,3.0,3.5,4.0,4.5,5.0], bin_index=2.3)

   # Profile background normalization parameters with prior scale of 5.0
   sed = gta.sed('sourceA', free_background=True, cov_scale=5.0)
   
By default the method will use the energy bins of the underlying
analysis.  The ``loge_bins`` keyword argument can be used to override
the default binning with the restriction that the SED energy bins
must align with the analysis bins. The bins used in the analysis can be
found with ``gta.log_energies``. For example if in the analysis
8 energy bins per decade are considered and you want to make the SED in 4 bins 
per decade you can specify ``loge_bins=gta.log_energies[::2]``.


The return value of :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` is a
dictionary with the results of the analysis.  The following example
shows how to extract values from the output dictionary and load the
SED data from the output FITS file:
   
.. code-block:: python
   
   # Get the sed results from the return argument
   sed = gta.sed('sourceA', outfile='sed.fits')

   # Print the SED flux values
   print(sed['flux'])

   # Reload the SED table from the output FITS file
   from astropy.table import Table
   sed_tab = Table.read('sed.fits')   

The contents of the FITS file and output dictionary are documented in
:ref:`sed_fits` and :ref:`sed_dict`.
   
.. _sed_fits:
                
SED FITS File
-------------

The following table describes the contents of the FITS file written by
:py:meth:`~fermipy.gtanalysis.GTAnalysis.sed`.  The ``SED`` HDU uses
that data format specification for SEDs documented `here
<https://gamma-astro-data-formats.readthedocs.io/en/latest/results/flux_points/index.html>`_.

.. csv-table:: *sed* Output Dictionary
   :header:    HDU, Column Name, Description
   :file: ../config/sed_fits_output.csv
   :delim: tab
   :widths: 10,10,80

.. _sed_dict:
            
SED Dictionary
--------------
   
The following table describes the contents of the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` output dictionary:

.. csv-table:: *sed* Output Dictionary
   :header:    Key, Type, Description
   :file: ../config/sed_output.csv
   :delim: tab
   :widths: 10,10,80


Configuration
-------------

The default configuration of the method is controlled with the
:ref:`config_sed` section of the configuration file.  The default
configuration can be overriden by passing the option as a *kwargs*
argument to the method.

.. csv-table:: *sed* Options
   :header:    Option, Default, Description
   :file: ../config/sed.csv
   :delim: tab
   :widths: 10,10,80
            
Reference/API
-------------

.. automethod:: fermipy.gtanalysis.GTAnalysis.sed
   :noindex:


