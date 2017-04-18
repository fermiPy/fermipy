.. _quickstart:

Quickstart Guide
================

This page walks through the steps to setup and perform a basic
spectral analysis of a source.  For additional fermipy tutorials see
the `IPython Notebook Tutorials`_.  To more easily follow along with
this example a directory containing pre-generated input files (FT1,
source maps, etc.) is available from the following link:

.. code-block:: bash
                
   $ curl -OL https://raw.githubusercontent.com/fermiPy/fermipy-extras/master/data/mkn421.tar.gz
   $ tar xzf mkn421.tar.gz
   $ cd mkn421
   
Creating a Configuration File
-----------------------------

The first step is to compose a configuration file that defines the
data selection and analysis parameters.  Complete documentation on the
configuration file and available options is given in the :ref:`config`
page.  fermiPy uses the `YAML format <http://yaml.org/>`_ for its
configuration files.  The configuration file has a hierarchical
organization that groups related parameters into separate
dictionaries.  In this example we will compose a configuration file
for a SOURCE-class analysis of Markarian 421 with FRONT+BACK event
types (evtype=3):

.. code-block:: yaml
   
   data:
     evfile : ft1.lst
     scfile : ft2.fits
     ltcube : ltcube.fits
     
   binning:
     roiwidth   : 10.0    
     binsz      : 0.1 
     binsperdec : 8   

   selection :
     emin : 100
     emax : 316227.76
     zmax    : 90
     evclass : 128
     evtype  : 3
     tmin    : 239557414
     tmax    : 428903014
     filter  : null
     target : 'mkn421'
     
   gtlike:
     edisp : True
     irfs : 'P8R2_SOURCE_V6'
     edisp_disable : ['isodiff','galdiff']

   model:
     src_roiwidth : 15.0
     galdiff  : '$FERMI_DIFFUSE_DIR/gll_iem_v06.fits'
     isodiff  : 'iso_P8R2_SOURCE_V6_v06.txt'
     catalogs : ['3FGL']

The *data* section defines the input data set and spacecraft file for
the analysis.  Here ``evfile`` points to a list of FT1 files that
encompass the chosen ROI, energy range, and time selection.  The
parameters in the *binning* section define the dimensions of the ROI
and the spatial and energy bin size.  The *selection* section defines
parameters related to the data selection (energy range, zmax cut, and
event class/type).  The ``target`` parameter in this section defines
the ROI center to have the same coordinates as the given source.  The
*model* section defines parameters related to the ROI model definition
(diffuse templates, point sources).

Fermipy gives the user the option to combine multiple data selections
into a joint likelihood with the *components* section.  The components
section contains a list of dictionaries with the same hierarchy as the
root analysis configuration.  Each element of the list defines the
analysis parameters for an independent sub-selection of the data.  Any
parameters not defined within the component dictionary default to the
value defined in the root configuration.  The following example shows
the *components* section that could be appended to the previous
configuration to define a joint analysis with four PSF event types:

.. code-block:: yaml
   
   components:
     - { selection : { evtype : 4  } } # PSF0
     - { selection : { evtype : 8  } } # PSF1
     - { selection : { evtype : 16 } } # PSF2
     - { selection : { evtype : 32 } } # PSF3

Any configuration parameter can be changed with this mechanism.  The
following example is a configuration in which a different zmax
selection and isotropic template is used for each of the four PSF
event types:

.. code-block:: yaml

   components:
     - model: {isodiff: isotropic_source_psf0_4years_P8V3.txt}
       selection: {evtype: 4, zmax: 70}
     - model: {isodiff: isotropic_source_psf1_4years_P8V3.txt}
       selection: {evtype: 8, zmax: 75}
     - model: {isodiff: isotropic_source_psf2_4years_P8V3.txt}
       selection: {evtype: 16, zmax: 85}
     - model: {isodiff: isotropic_source_psf3_4years_P8V3.txt}
       selection: {evtype: 32, zmax: 90}


Creating an Analysis Script
---------------------------

Once the configuration file has been composed, the analysis is
executed by creating an instance of
:py:class:`~fermipy.gtanalysis.GTAnalysis` with the configuration file
as its argument and calling its analysis methods.
:py:class:`~fermipy.gtanalysis.GTAnalysis` serves as a wrapper over
the underlying pyLikelihood classes and provides methods to fix/free
parameters, add/remove sources from the model, and perform a fit to
the ROI.  For a complete documentation of the available methods you
can refer to the :ref:`fermipy` page.

In the following python examples we show how to initialize and run a
basic analysis of a source.  First we instantiate a
:py:class:`~fermipy.gtanalysis.GTAnalysis` object with the path to the
configuration file and run
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup`.

.. code-block:: python

   from fermipy.gtanalysis import GTAnalysis
           
   gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})
   gta.setup()

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` method performs
the data preparation and response calculations needed for the analysis
(selecting the data, creating counts and exposure maps, etc.).
Depending on the data selection and binning of the analysis this will
often be the slowest step in the analysis sequence.  The output of
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` is cached in the
analysis working directory so subsequent calls to
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` will run much faster.

Before running any other analysis methods it is recommended to first
run :py:meth:`~fermipy.gtanalysis.GTAnalysis.optimize`:

.. code-block:: python

   gta.optimize()

This will loop over all model components in the ROI and fit their
normalization and spectral shape parameters.  This method also
computes the TS of all sources which can be useful for identifying
weak sources that could be fixed or removed from the model.  We can
check the results of the optimization step by calling
:py:meth:`~fermipy.gtanalysis.GTAnalysis.print_roi`:

.. code-block:: python

   gta.print_roi()
    
.. Once the *GTAnalysis* object is initialized we can define which
.. source parameters will be free in the fit.

By default all models parameters are initially fixed.  The
:py:meth:`~fermipy.gtanalysis.GTAnalysis.free_source` and
:py:meth:`~fermipy.gtanalysis.GTAnalysis.free_sources` methods can be
use to free or fix parameters of the model.  In the following example
we free the normalization of catalog sources within 3 deg of the ROI
center and free the galactic and isotropic components by name.

.. code-block:: python

   # Free Normalization of all Sources within 3 deg of ROI center
   gta.free_sources(distance=3.0,pars='norm')

   # Free all parameters of isotropic and galactic diffuse components 
   gta.free_source('galdiff')
   gta.free_source('isodiff')

The ``minmax_ts`` and ``minmax_npred`` arguments to
:py:meth:`~fermipy.gtanalysis.GTAnalysis.free_sources` can be used to
free or fixed sources on the basis of their current TS or Npred
values:

.. code-block:: python

   # Free sources with TS > 10
   gta.free_sources(minmax_ts=[10,None],pars='norm')

   # Fix sources with TS < 10
   gta.free_sources(minmax_ts=[None,10],free=False,pars='norm')

   # Fix sources with 10 < Npred < 100
   gta.free_sources(minmax_npred=[10,100],free=False,pars='norm')
   
When passing a source name argument both case and whitespace are
ignored.  When using a FITS catalog file a source can also be referred
to by any of its associations.  When using the 3FGL catalog, the
following calls are equivalent ways of freeing the parameters of Mkn
421:

.. code-block:: python

   # These calls are equivalent
   gta.free_source('mkn421')
   gta.free_source('Mkn 421')
   gta.free_source('3FGL J1104.4+3812')
   gta.free_source('3fglj1104.4+3812')

After freeing parameters of the model we can execute a fit by calling
:py:meth:`~fermipy.gtanalysis.GTAnalysis.fit`.  The will maximize the
likelihood with respect to the model parameters that are currently
free.

.. code-block:: python

   gta.fit()

After the fitting is complete we can write the current state of the
model with `~fermipy.gtanalysis.GTAnalysis.write_roi`:

.. code-block:: python

   gta.write_roi('fit_model')

This will write several output files including an XML model file and
an ROI dictionary file.  The names of all output files will be
prepended with the ``prefix`` argument to
:py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.

Once we have optimized our model for the ROI we can use the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.residmap` and
:py:meth:`~fermipy.gtanalysis.GTAnalysis.tsmap` methods to assess the
fit quality and look for new sources.

.. code-block:: python

   # Dictionary defining the spatial/spectral parameters of the test source
   model = {'SpatialModel' : 'PointSource', 'Index' : 2.0,
            'SpectrumType' : 'PowerLaw'}

   # Both methods return a dictionary with the maps
   m0 = gta.residmap('fit_model', model=model, make_plots=True)
   m1 = gta.tsmap('fit_model', model=model, make_plots=True)

More documentation on these methods is available in
the :ref:`tsmap` and :ref:`residmap` pages.

By default, calls to :py:meth:`~fermipy.gtanalysis.GTAnalysis.fit` will
execute a global spectral fit over the entire energy range of the
analysis.  To extract a bin-by-bin flux spectrum (i.e. a SED) you can
call :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed` method with the
name of the source:

.. code-block:: python

   gta.sed('mkn421', make_plots=True)

More information about :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed`
method can be found in the :ref:`sed` page.


Extracting Analysis Results
---------------------------

Results of the analysis can be extracted from the dictionary file
written by :py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.  This
method writes information about the current state of the analysis to a
python dictionary.  More documentation on the contents of the output
file are available in the :ref:`output` page.

By default the output dictionary is written to a file in the `numpy
format <http://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ and
can be loaded from a python session after your analysis is complete.
The following demonstrates how to load the analysis dictionary that
was written to *fit_model.npy* in the Mkn421 analysis example:

.. code-block:: python
   
   >>> # Load analysis dictionary from a npy file
   >>> import np
   >>> c = np.load('fit_model.npy').flat[0]
   >>> list(c.keys())
   ['roi', 'config', 'sources', 'version']

The output dictionary contains the following top-level elements:

.. csv-table:: File Dictionary
   :header:    Key, Description
   :file: config/file_output.csv
   :delim: tab
   :widths: 10,10,80

Each source dictionary collects the properties of the given source
(TS, NPred, best-fit parameters, etc.) computed up to that point in
the analysis.

.. code-block:: python
   
   >>> list(c['sources'].keys())
   ['3FGL J1032.7+3735',
   '3FGL J1033.2+4116',
   ...
   '3FGL J1145.8+4425',
   'galdiff',
   'isodiff']
   >>> c['sources']['3FGL J1104.4+3812']['ts']
   87455.9709683
   >>> c['sources']['3FGL J1104.4+3812']['npred']
   31583.7166495
    
Information about individual sources in the ROI is also saved to a
catalog FITS file with the same string prefix as the dictionary file.
This file can be loaded with the `astropy.io.fits` or
`astropy.table.Table` interface:
    
.. code-block:: python
   
   >>> # Load the source catalog file
   >>> from astropy.table import Table
   >>> tab = Table.read('fit_model.fits')
   >>> tab[['name','class','ts','npred','flux']]
       name       class       ts           npred                    flux [2]               
                                                                  1 / (cm2 s)              
   ----------------- ----- -------------- ------------- --------------------------------------
   3FGL J1104.4+3812   BLL  87455.9709683 31583.7166495 2.20746290445e-07 .. 1.67062058528e-09
   3FGL J1109.6+3734   bll    42.34511826 93.7971922425  5.90635786943e-10 .. 3.6620894143e-10
   ...
   3FGL J1136.4+3405  fsrq  4.78089819776 261.427034151 1.86805869704e-08 .. 8.62638727067e-09
   3FGL J1145.8+4425  fsrq  3.78006883967 237.525501441 7.25611442299e-08 .. 3.77056557247e-08

The FITS file contains columns for all scalar and vector elements of
the source dictionary.  Spectral fit parameters are contained in the
``param_names``, ``param_values``, and ``param_errors`` columns:

.. code-block:: python
                
   >>> tab[['param_names','param_values','param_errors']][0]
   <Row 0 of table
    values=(['Prefactor', 'Index', 'Scale', '', '', ''],
            [2.1301351784512767e-11, -1.7716399431228638, 1187.1300048828125, nan, nan, nan],
            [1.6126233510314277e-13, nan, nan, nan, nan, nan])
    dtype=[('param_names', 'S32', (6,)),
           ('param_values', '>f8', (6,)),
           ('param_errors', '>f8', (6,))]>
   
Reloading from a Previous State
-------------------------------

One can reload an analysis instance that was saved with
:py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi` by calling either
the :py:meth:`~fermipy.gtanalysis.GTAnalysis.create` or
:py:meth:`~fermipy.gtanalysis.GTAnalysis.load_roi` methods.  The
:py:meth:`~fermipy.gtanalysis.GTAnalysis.create` method can be used to
construct an entirely new instance of
:py:class:`~fermipy.gtanalysis.GTAnalysis` from a previously saved
results file:

.. code-block:: python
   
   from fermipy.gtanalysis import GTAnalysis
   gta = GTAnalysis.create('fit_model.npy')

   # Continue running analysis starting from the previously saved
   # state 
   gta.fit()

where the argument is the path to an output file produced with
:py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.  This function
will instantiate a new analysis object, run the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.setup` method, and load the
state of the model parameters at the time that
:py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi` was called.

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.load_roi` method can be
used to reload a previous state of the analysis to an existing
instance of :py:class:`~fermipy.gtanalysis.GTAnalysis`.

.. code-block:: python
   
   from fermipy.gtanalysis import GTAnalysis

   gta = GTAnalysis('config.yaml')
   gta.setup()

   gta.write_roi('prefit_model')

   # Fit a source
   gta.free_source('mkn421')
   gta.fit()

   # Restore the analysis to its prior state before the fit of mkn421
   # was executed
   gta.load_roi('prefit_model')
   
Using :py:meth:`~fermipy.gtanalysis.GTAnalysis.load_roi` is generally
faster than :py:meth:`~fermipy.gtanalysis.GTAnalysis.create` when an
analysis instance already exists.

IPython Notebook Tutorials
--------------------------

Additional tutorials with more detailed examples are available as
IPython notebooks in the `notebooks
<https://github.com/fermiPy/fermipy-extra/tree/master/notebooks/>`_
directory of the `fermipy-extra
<https://github.com/fermiPy/fermipy-extra>`_ respository.  These
notebooks can be browsed as `static web pages
<http://nbviewer.jupyter.org/github/fermiPy/fermipy-extra/blob/master/notebooks/index.ipynb>`_
or run interactively by downloading the fermipy-extra repository and
running ``jupyter notebook`` in the notebooks directory:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy-extra.git    
   $ cd fermipy-extra/notebooks
   $ jupyter notebook index.ipynb

Note that this will require you to have both ipython and jupyter
installed in your python environment.  These can be installed in a
conda- or pip-based installation as follows:

.. code-block:: bash

   # Install with conda
   $ conda install ipython jupyter

   # Install with pip
   $ pip install ipython jupyter

One can also run the notebooks from a docker container following the
:ref:`dockerinstall` instructions:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy-extra.git    
   $ cd fermipy-extra
   $ docker pull fermipy/fermipy
   $ docker run -it --rm -p 8888:8888 -v $PWD:/workdir -w /workdir fermipy/fermipy

After launching the notebook server, paste the URL that appears into
your web browser and navigate to the *notebooks* directory.
