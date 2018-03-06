.. _tscube:

TS Cube
=======

.. warning:: This method requires Fermi Science Tools version 11-04-00
   or later.

:py:meth:`~fermipy.gtanalysis.GTAnalysis.tscube` can be used generate
both test statistic (TS) maps and bin-by-bin scans of the test source
likelihood as a function of spatial pixel and energy bin (likelihood cubes).
The implemention is based on the `gttscube` ST application which uses
an efficient newton optimization algorithm for fitting the test source at
each pixel in the ROI.

The TS map output has the same format as TS maps produced by
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` (see :ref:`tsmap` for
further details).  However while
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` fixes the background
model, :py:meth:`~fermipy.gtanalysis.GTAnalysis.tscube` can also fit
background normalization parameters when scanning the test source
likelihood.  This method makes no approximations in the
evaluation of the likelihood and may be somewhat slower than
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` depending on the ROI
dimensions and energy bounds.

For each spatial bin the method calculates the maximum likelihood test
statistic given by

.. math::

   \mathrm{TS} = 2 \sum_{k} \ln L(\mu,\hat{\theta}|n_{k}) - \ln L(0,\hat{\hat{\theta}}|n_{k})

where the summation index *k* runs over both spatial and energy bins,
μ is the test source normalization parameter, and θ represents the
parameters of the background model.  Normalization parameters of the
background model are refit at every test source position if they are
free in the model.  All other spectral parameters (indices etc.) are
kept fixed.

Examples
--------

The method is executed by providing a `model` dictionary argument that
defines the spectrum and spatial morphology of the test source:

.. code-block:: python
   
   # Generate TS cube for a power-law point source with Index=2.0
   model = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}
   cube = gta.tscube('fit1',model=model)

   # Generate TS cube for a power-law point source with Index=2.0 and
   # restricting the analysis to E > 3.16 GeV
   model = {'Index' : 2.0, 'SpatialModel' : 'PointSource'}
   cube = gta.tscube('fit1_emin35',model=model,erange=[3.5,None])

   # Generate TS cubes for a power-law point source with Index=1.5, 2.0, and 2.5
   model={'SpatialModel' : 'PointSource'}
   cubes = []
   for index in [1.5,2.0,2.5]:
       model['Index'] = index
       cubes += [gta.tsmap('fit1',model=model)]

In addition to generating a TS map, this method can also extract a
test source likelihood profile as a function of energy at every
position in the ROI (likelihood cube).  This information is saved to
the ``SCANDATA`` HDU of the output FITS file:

.. code-block:: python

   from astropy.table import Table
   cube = gta.tscube('fit1',model=model, do_sed=True)
   tab_scan = Table.read(cube['file'],'SCANDATA')
   tab_ebounds = Table.read(cube['file'],'EBOUNDS')

   eflux_scan = tab_ebounds['REF_EFLUX'][None,:,None]*tab_scan['norm_scan']
   
   # Plot likelihood for pixel 400 and energy bin 2
   plt.plot(eflux_scan[400,2],tab_scan['dloglike_scan'][400,2])
   
The likelihood profile cube can be used to evaluate the likelihood for
a test source with an arbitrary spectral model at any position in the
ROI.  The `~fermipy.castro.TSCube` and `~fermipy.castro.CastroData`
classes can be used to analyze a TS cube:

.. code-block:: python

   from fermipy.castro import TSCube                
   tscube = TSCube.create_from_fits('tscube.fits')
   cd = tscube.castroData_from_ipix(400)

   # Fit the likelihoods at pixel 400 with different spectral models
   cd.test_spectra()

Configuration
-------------

The default configuration of the method is controlled with the
:ref:`config_tscube` section of the configuration file.  The default
configuration can be overriden by passing the option as a *kwargs*
argument to the method.

.. csv-table:: *tscube* Options
   :header:    Option, Default, Description
   :file: ../config/tscube.csv
   :delim: tab
   :widths: 10,10,80
   
Reference/API
-------------
   
.. automethod:: fermipy.gtanalysis.GTAnalysis.tscube
   :noindex:
