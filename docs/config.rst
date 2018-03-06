.. _config:

Configuration
=============

This page describes the configuration management scheme used within
the Fermipy package and documents the configuration parameters
that can be set in the configuration file.


##################################
Class Configuration
##################################

Classes in the Fermipy package own a configuration state dictionary
that is initialized when the class instance is created.  Elements of
the configuration dictionary can be scalars (str, int, float) or
dictionaries containing groups of parameters.  The settings in this
dictionary are used to control the runtime behavior of the class.

When creating a class instance, the configuration is initialized by
passing either a configuration dictionary or configuration file path
to the class constructor.  Keyword arguments can be passed to the
constructor to override configuration parameters in the input
dictionary.  In the following example the *config* dictionary defines
values for the parameters *emin* and *emax*.  By passing a dictionary
for the *selection* keyword argument, the value of *emax* in the
keyword argument (10000) overrides the value of *emax* in the input
dictionary.

.. code-block:: python
   
   config = { 
   'selection' : { 'emin' : 100, 
                   'emax' : 1000 }   
   }

   gta = GTAnalysis(config,selection={'emax' : 10000})
   
The first argument can also be the path to a YAML configuration file
rather than a dictionary:

.. code-block:: python
   
   gta = GTAnalysis('config.yaml',selection={'emax' : 10000})


##################################
Configuration File
##################################

Fermipy uses `YAML <http://yaml.org/>`_ files to read and write its
configuration in a persistent format.  The configuration file has a
hierarchical structure that groups parameters into dictionaries that
are keyed to a section name (*data*, *binning*, etc.).

.. code-block:: yaml
   :caption: Sample Configuration

   data:
     evfile : ft1.lst
     scfile : ft2.fits
     ltfile : ltcube.fits
     
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
                          
The configuration file has the same structure as the configuration
dictionary such that one can read/write configurations using the
load/dump methods of the yaml module:

.. code-block:: python

   import yaml
   # Load a configuration
   config = yaml.load(open('config.yaml'))
   # Update a parameter and write a new configuration
   config['selection']['emin'] = 1000.
   yaml.dump(config, open('new_config.yaml','w'))
   
Most of the configuration parameters are optional and if not set
explicitly in the configuration file will be set to a default value.
The parameters that can be set in each section are described below.

.. _config_binning:

binning
-------

Options in the *binning* section control the spatial and spectral binning of the data.

.. code-block:: yaml
   :caption: Sample *binning* Configuration
                
   binning:

     # Binning
     roiwidth   : 10.0
     npix       : null
     binsz      : 0.1 # spatial bin size in deg
     binsperdec : 8   # nb energy bins per decade
     projtype   : WCS

.. csv-table:: *binning* Options
   :header:    Option, Default, Description
   :file: config/binning.csv
   :delim: tab
   :widths: 10,10,80

.. _config_components:

components
----------

The *components* section can be used to define analysis configurations
for independent subselections of the data.  Each subselection will
have its own binned likelihood instance that is combined in a global
likelihood function for the ROI (implemented with the SummedLikelihood
class in pyLikelihood).  The *components* section is optional and when
set to null (the default) only a single likelihood component will be
created with the parameters of the root analysis configuration.

The component section is defined as a list of dictionaries where each
element sets analysis parameters for a different subcomponent of the
analysis.  The component configurations follow the same structure and
accept the same parameters as the root analysis configuration.
Parameters not defined in a given element will default to the values
set in the root analysis configuration.

The following example illustrates how to define a Front/Back analysis
with two components.  Files associated to each component will be given
a suffix according to their order in the list (e.g. file_00.fits,
file_01.fits, etc.).

.. code-block:: yaml

   # Component section for Front/Back analysis
     - { selection : { evtype : 1 } } # Front
     - { selection : { evtype : 2 } } # Back

.. _config_data:
     
data
----

The *data* section defines the input data files for the analysis (FT1,
FT2, and livetime cube).  ``evfile`` and ``scfile`` can either be 
individual files or group of files.  The optional ``ltcube`` option can
be used to choose a pre-generated livetime cube.  If ``ltcube`` is
null a livetime cube will be generated at runtime with ``gtltcube``.  

.. code-block:: yaml
   :caption: Sample *data* Configuration

   data :
     evfile : ft1.lst
     scfile : ft2.fits 
     ltcube : null

.. csv-table:: *data* Options
   :header:    Option, Default, Description
   :file: config/data.csv
   :delim: tab
   :widths: 10,10,80

.. _config_extension:
            
extension
---------

The options in *extension* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.extension` method.  For more information
about using this method see the :ref:`extension` page.

.. csv-table:: *extension* Options
   :header:    Option, Default, Description
   :file: config/extension.csv
   :delim: tab
   :widths: 10,10,80

.. _config_fileio:
            
fileio
------

The *fileio* section collects options related to file bookkeeping.
The ``outdir`` option sets the root directory of the analysis instance
where all output files will be written.  If ``outdir`` is null then the
output directory will be automatically set to the directory in which
the configuration file is located.  Enabling the ``usescratch`` option
will stage all output data files to a temporary scratch directory
created under ``scratchdir``.

.. code-block:: yaml                
   :caption: Sample *fileio* Configuration
           
   fileio:
      outdir : null
      logfile : null
      usescratch : False
      scratchdir  : '/scratch'

.. csv-table:: *fileio* Options
   :header:    Option, Default, Description
   :file: config/fileio.csv
   :delim: tab
   :widths: 10,10,80


.. _config_gtlike:
            
gtlike
------

Options in the *gtlike* section control the setup of the likelihood
analysis include the IRF name (``irfs``).

.. csv-table:: *gtlike* Options
   :header:    Option, Default, Description
   :file: config/gtlike.csv
   :delim: tab
   :widths: 10,10,80

.. _config_lightcurve:
            
lightcurve
----------

The options in *lightcurve* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.lightcurve` method.  For more information
about using this method see the :ref:`lightcurve` page.

.. csv-table:: *lightcurve* Options
   :header:    Option, Default, Description
   :file: config/lightcurve.csv
   :delim: tab
   :widths: 10,10,80
            
.. _config_model:

model
-----

The *model* section collects options that control the inclusion of
point-source and diffuse components in the model.  ``galdiff`` and
``isodiff`` set the templates for the Galactic IEM and isotropic
diffuse respectively.  ``catalogs`` defines a list of catalogs that
will be merged to form a master analysis catalog from which sources
will be drawn.  Valid entries in this list can be FITS files or XML
model files.  ``sources`` can be used to insert additional
point-source or extended components beyond those defined in the master
catalog.  ``src_radius`` and ``src_roiwidth`` set the maximum distance
from the ROI center at which sources in the master catalog will be
included in the ROI model.

.. code-block:: yaml
   :caption: Sample *model* Configuration
                
   model :
   
     # Diffuse components
     galdiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v06.fits'
     isodiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/iso_P8R2_SOURCE_V6_v06.txt'

     # List of catalogs to be used in the model.
     catalogs : 
       - '3FGL'
       - 'extra_sources.xml'

     sources :
       - { 'name' : 'SourceA', 'ra' : 60.0, 'dec' : 30.0, 'SpectrumType' : PowerLaw }
       - { 'name' : 'SourceB', 'ra' : 58.0, 'dec' : 35.0, 'SpectrumType' : PowerLaw }

     # Include catalog sources within this distance from the ROI center
     src_radius  : null

     # Include catalog sources within a box of width roisrc.
     src_roiwidth : 15.0

.. csv-table:: *model* Options
   :header:    Option, Default, Description
   :file: config/model.csv
   :delim: tab
   :widths: 10,10,80
            
.. _config_optimizer:
            
optimizer
---------

.. csv-table:: *optimizer* Options
   :header:    Option, Default, Description
   :file: config/optimizer.csv
   :delim: tab
   :widths: 10,10,80

.. _config_plotting:
            
plotting
--------

.. csv-table:: *plotting* Options
   :header:    Option, Default, Description
   :file: config/plotting.csv
   :delim: tab
   :widths: 10,10,80

.. _config_residmap:
            
residmap
--------

The options in *residmap* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.residmap` method.  For more
information about using this method see the :ref:`residmap` page.

.. csv-table:: *residmap* Options
   :header:    Option, Default, Description
   :file: config/residmap.csv
   :delim: tab
   :widths: 10,10,80

.. _config_roiopt:

roiopt
------

The options in *roiopt* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.optimize` method.  For more
information about using this method see the :ref:`fitting` page.

.. csv-table:: *roiopt* Options
   :header:    Option, Default, Description
   :file: config/roiopt.csv
   :delim: tab
   :widths: 10,10,80
            
.. _config_sed:
            
sed
---

The options in *sed* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.sed` method.  For more information
about using this method see the :ref:`sed` page.

.. csv-table:: *sed* Options
   :header:    Option, Default, Description
   :file: config/sed.csv
   :delim: tab
   :widths: 10,10,80

.. _config_selection:

selection
---------

The *selection* section collects parameters related to the data
selection and target definition.  The majority of the parameters in
this section are arguments to *gtselect* and *gtmktime*.  The ROI
center can be set with the *target* parameter by providing the name of
a source defined in one of the input catalogs (defined in the *model*
section).  Alternatively the ROI center can be defined by giving
explicit sky coordinates with *ra* and *dec* or *glon* and *glat*.

.. code-block:: yaml

   selection:

     # gtselect parameters
     emin    : 100
     emax    : 100000
     zmax    : 90
     evclass : 128
     evtype  : 3
     tmin    : 239557414
     tmax    : 428903014 

     # gtmktime parameters
     filter : 'DATA_QUAL>0 && LAT_CONFIG==1'
     roicut : 'no'

     # Set the ROI center to the coordinates of this source
     target : 'mkn421'

.. csv-table:: *selection* Options
   :header:    Option, Default, Description
   :file: config/selection.csv
   :delim: tab
   :widths: 10,10,80

.. _config_sourcefind:
            
sourcefind
----------

The options in *sourcefind* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.find_sources` method.  For more information
about using this method see the :ref:`findsources` page.

.. csv-table:: *sourcefind* Options
   :header:    Option, Default, Description
   :file: config/sourcefind.csv
   :delim: tab
   :widths: 10,10,80

.. _config_tsmap:
            
tsmap
-----

The options in *tsmap* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.tsmap` method.  For more information
about using this method see the :ref:`tsmap` page.

.. csv-table:: *tsmap* Options
   :header:    Option, Default, Description
   :file: config/tsmap.csv
   :delim: tab
   :widths: 10,10,80

.. _config_tscube:

tscube
------

The options in *tscube* control the default behavior of the
`~fermipy.gtanalysis.GTAnalysis.tscube` method.  For more information
about using this method see the :ref:`tscube` page.

.. csv-table:: *tscube* Options
   :header:    Option, Default, Description
   :file: config/tscube.csv
   :delim: tab
   :widths: 10,10,80
            

            

