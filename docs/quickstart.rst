.. _quickstart:

Quickstart Guide
================

This page walks through the steps to setup and perform a basic
spectral analysis of a source.


Creating a Configuration File
-------------------------------

The first step is to compose a configuration file that defines the
basic analysis parameters.  Complete documentation for the configuration
file format and parameters is given in the :ref:`config` page.
fermiPy accepts configuration files in the YAML format.  The following
example is a configuration file for a SOURCE-class analysis of
Markarian 421 with all event types combined (evtype=3).

.. code-block:: yaml
   
   data:
     evfile : ft1.lst
     scfile : ft2.fits
     
   binning:
     roiwidth   : 10.0    
     binsz      : 0.1 
     binsperdec : 8   

   selection :
     emin : 100
     emax : 10000
     zmax    : 90
     evclass : 128
     evtype  : 3
     target : 'mkn421'

   gtlike:
     edisp : True
     irfs : 'P8R2_SOURCE_V6'
     edisp_disable : ['isodiff','galdiff']

   model:
     src_roiwidth : 10.0
     galdiff  : '$FERMI_DIFFUSE_DIR/template_4years_P8_V2_scaled.fits'
     isodiff  : '$FERMI_DIFFUSE_DIR/isotropic_source_4years_P8V3.txt'
     catalogs : 
       - 'gll_psc_v14.fit'

The configuration file is divided into blocks that group together
related options.  The *data* block defines the FT1 and FT2 files.
Here *evfile* points to a list of FT1 files that encompass the chosen
ROI, energy range, and time selection.  The parameters in the
*binning* block define the dimensions of the ROI and the spatial and energy
bin size.  The *selection* block defines parameters related to the
data selection (energy range, zmax cut, and event class/type).  The
*target* parameter in this block defines the ROI center to have the
same coordinates as the given source.   The *model*
block defines all parameters related to the ROI model definition (diffuse
templates, point sources).  

fermiPy allows the user to combine multiple data selections into a
joint likelihood with the *components* block.  The components block
contains a list of dictionaries with the same hierarchy as the root
analysis configuration.  Each element of the list defines the analysis
parameters for an independent sub-selection of the data.  Any
parameters not defined within the component dictionary default to the
value defined in the root configuration.  The following example shows
the components block that could be appended to the previous
configuration to define a joint analysis with four PSF event types:

.. code-block:: yaml
   
   components:
     - { selection : { evtype : 4  } } # PSF0
     - { selection : { evtype : 8  } } # PSF1
     - { selection : { evtype : 16 } } # PSF2
     - { selection : { evtype : 32 } } # PSF3

Any configuration parameter can be changed with this mechanism.  The
following example shows how one can use a different zmax selection and
isotropic template for each of the four PSF event types:

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

.. These classes are also directly exposed

Once the configuration file has been composed, the analysis is
executed by creating an instance of
:py:class:`~fermipy.gtanalysis.GTAnalysis` with this configuration and
calling its associated methods.
:py:class:`~fermipy.gtanalysis.GTAnalysis` provides a similar
functionality to the underlying BinnedAnalysis/UnbinnedAnalysis
classes with methods to fix/free parameters, add/remove sources from
the model, and perform a fit to the ROI.

In the following example we lay out the sequence of python calls that
could be run interactively or in a script to setup and run an
analysis.  First we instantiate :py:class:`~fermipy.gtanalysis.GTAnalysis` with the chosen
configuration.

.. code-block:: python

   from fermipy.gtanalysis import GTAnalysis
           
   gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})
   gta.setup()

The :py:meth:`~fermipy.gtanalysis.GTAnalysis.setup`. method performs
all the prepratory steps for the analysis (selecting the data,
creating counts and exposure maps, etc.).  It should be noted that
depending on the parameters of the analysis this will often be the
slowest step in the analysis sequence.

Once the *GTAnalysis* object is initialized we can control which
sources and source parameters will be free in the fit.  By default all
models parameters are initially fixed.  In the following example we
free the normalization of catalog sources within 3 deg of the ROI
center and free the galactic and isotropic components by name.

.. code-block:: python

   # Free Normalization of all Sources within 3 deg of ROI center
   gta.free_sources(distance=3.0,pars='norm')

   # Free all parameters of isotropic and galactic diffuse components 
   gta.free_source('galdiff')
   gta.free_source('isodiff')

Note that when passing a source name argument both case and whitespace
are ignored.  A source can also be identified by the name of any of
its source associations.  Thus the following calls are equivalent ways
of freeing the parameters of Mkn 421:

.. code-block:: python

   # These calls are equivalent
   gta.free_source('mkn421')
   gta.free_source('Mkn 421')
   gta.free_source('3FGL J1104.4+3812')
   gta.free_source('3fglj1104.4+3812')

After freeing the parameters of one or more sources we can execute a
fit by calling :py:meth:`~fermipy.gtanalysis.GTAnalysis.fit`.  The will
maximize the likelihood with respect to the model parameters that are
currently free in the model.

.. code-block:: python

   gta.fit()

After the fitting is complete we can write the current state of the
best-fit model with the
:py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi` method:

.. code-block:: python

   gta.write_roi('fit_model')

This will write both an XML model file and a YAML results file with a
variety of information about each source.  By default the fit to each
source is performed with a global spectral model that spans the entire
analysis energy range.  To extract a bin-by-bin flux spectrum (i.e. a
SED) you can call :py:meth:`~fermipy.gtanalysis.GTAnalysis.sed`
method with the name of the source:

.. code-block:: python

   gta.sed('mkn421')


Extracting Analysis Results
---------------------------

Results of the analysis can be extracted from the dictionary file
written by :py:meth:`~fermipy.gtanalysis.GTAnalysis.write_roi`.  This
method writes the current ROI model to both an XML model file and a
results dictionary.  The results dictionary is written in both npy and
yaml formats and can be loaded from a python session after your
analysis is complete.  The following example demonstrates how to load
the dictionary from either format:

.. code-block:: python
   
   >>> # Load from yaml
   >>> import yaml
   >>> c = yaml.load(open('fit_model.yaml'))
   >>>
   >>> # Load from npy
   >>> import np
   >>> c = np.load('fit_model.npy').flat[0]
   >>>
   >>> print c.keys()
   ['roi', 'config', 'sources']

The results dictionary is split into three top-level dictionaries:

roi 
   A dictionary containing information about the ROI as a whole.

config   
   The configuration dictionary of the
   :py:class:`~fermipy.gtanalysis.GTAnalysis` instance.

sources
   A dictionary containing information for individual
   sources (diffuse and point-like).  Each element of this dictionary
   maps to a single source in the ROI model.

Each source dictionary collects the properties of the given source
(TS, NPred, best-fit parameters, etc.) computed up to that point in
the analysis.

.. code-block:: python
   
   >>> print c['sources'].keys()
   ['3FGL J0954.2+4913',
    '3FGL J0957.4+4728',
    '3FGL J1006.7+3453',

    ...

    '3FGL J1153.4+4932',
    '3FGL J1159.5+2914',
    '3FGL J1203.2+3847',
    '3FGL J1209.4+4119',
    'galdiff',
    'isodiff']
