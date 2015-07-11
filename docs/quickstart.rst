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
     galdiff  : '$(DIFFUSEDIR)/template_4years_P8_V2_scaled.fits'
     isodiff  : '$(DIFFUSEDIR)/isotropic_source_4years_P8V3.txt'
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

Once a configuration file has been composed, the user executes their
analysis by creating an instance of
:py:class:`fermipy.gtanalysis.GTAnalysis` with this configuration and
calling its associated methods.
:py:class:`fermipy.gtanalysis.GTAnalysis` provides a similar
functionality to the underlying BinnedAnalysis/UnbinnedAnalysis
classes with methods to fix/free parameters, add/remove sources from
the model, and perform a fit to the ROI.

In the following example we lay out the sequence of python calls that
could be run interactively or in a script to setup and run an
analysis.  First we instantiate *GTAnalysis* with the chosen
configuration.

.. code-block:: python

   from fermipy.gtanalysis import GTAnalysis
           
   gta = GTAnalysis('config.yaml',logging={'verbosity' : 3})
   gta.setup()

The *setup* method performs all the prepratory steps for the analysis
(selecting the data, creating counts and exposure maps, etc.).  It
should be noted that depending on the parameters of the analysis this
will often be the slowest step in the analysis sequence.

Once the *GTAnalysis* object is initialized we can control which
sources and source parameters will be free in the fit.  By default all
parameters of the model start as fixed.  In the following example we
free catalog sources within 3 deg of the ROI center and free the
galactic and isotropic components by name.

.. code-block:: python

   # Free Sources
   gta.free_sources(distance=3.0)
   gta.free_source('galdiff')
   gta.free_source('isodiff')

.. code-block:: python

   gta.fit()
   gta.write_xml('fit_model.xml')
   gta.sed('mkn421')

   # Write results yaml file
   gta.write_roi('fit_model')



.. code-block:: bash
   
   >>>
   >>>


Extracting Analysis Results
---------------------------



