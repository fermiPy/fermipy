.. _config:

Configuration
=============

This page describes the configuration management scheme used within
the fermiPy package and the documents the analysis options that can be
controlled with the configuration file.


##################################
Class Configuration
##################################

Classes in the fermiPy package follow a common convention by which the
the runtime behavior of a class instance can be controlled.
Internally every class instance has a dictionary that defines its
configuration state.  Elements of this dictionary can be scalars (str,
int ,float) or dictionaries defining nested blocks of the
configuration.

The class configuration dictionary is set at the time of object
creation by passing a dictionary or a YAML file containing a
dictionary to the class constructor.  Optional kwargs arguments can be
used to override options in the input dictionary.  For instance in the
following example the *config* dictionary defines values for the
parameters *emin* and *emax*.  By passing an additional dictionary for
the selection block, the value of emax in the kwargs argument (1000)
overrides the value in the config dictionary.

.. code-block:: python
   
   config = { 
   'selection' : { 'emin' : 100, 
                   'emax' : 1000 }   
   }

   gta = GTAnalysis(config,selection={'emax' : 10000})
   
Alternatively the config argument can be set to the path to a YAML
configuration file:

.. code-block:: python
   
   gta = GTAnalysis('config.yaml',selection={'emax' : 10000})


##################################
Configuration File
##################################

fermiPy uses YAML-format configuration files which define a structured
hierarchy of parameters organized in sections that mirrors the layout of
the configuration dictionary.  Each configuration section groups a set
of related options.  The following sub-sections describe the options
that can be set in each section.

fileio
------

The *fileio* section collects options related to file bookkeeping.  The
*outdir* option sets the root directory of the analysis instance where
all output files will be written.  If *outdir* is null then the output
directory will be set to the directory of the configuration file.
Enabling the *usescratch* option will stage all output data files to
a temporary scratch directory created under *scratchdir*.

.. code-block:: yaml

   fileio:

     # Set the output directory
     outdir : null

     # Enable staging analysis output to an intermediate scratch directory
     usescratch : False

     # Set the root directory under which the scratch directory should
     # be written
     scratchdir  : '/scratch'

data
----

The *data* section defines the input data files for the analysis (FT1,
FT2, and livetime cube).  *evfile* and *scfile* can either be an
individual file or group of files.  The optional *ltcube* option can
be used to choose a pre-generated livetime cube.  If this parameter is
null a livetime cube will be generated at runtime.

.. code-block:: yaml

   data :
     evfile : ft1.lst
     scfile : ft2.fits 
     ltcube : null

model
-----

The *model* section collects options that control the inclusion of
point-source and diffuse components in the model.  *galdiff* and
*isodiff* set the templates for the Galactic IEM and isotropic diffuse
respectively.  *catalogs* defines a list of catalogs that will be
merged to form a master analysis catalog from which sources will be
drawn.  Valid entries in this list can be FITS files or XML model
files.  *sources* can be used to insert additional point-source or
extended components beyond those defined in the master catalog.
*src_radius* and *src_roiwidth* set the maximum distance from the ROI
center at which sources in the master catalog will be included in the
ROI model.

.. code-block:: yaml

   model :
   
     # Diffuse components
     galdiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/gll_iem_v06.fits'
     isodiff  : '$FERMI_DIR/refdata/fermi/galdiffuse/iso_P8R2_SOURCE_V6_v06.txt'

     # List of catalogs to be used in the model.
     catalogs : 
       - 'gll_psc_v14.fit'
       - 'extra_sources.xml'

     sources :
       - { 'name' : 'SourceA', 'ra' : 60.0, 'dec' : 30.0, 'SpectrumType' : PowerLaw }
       - { 'name' : 'SourceB', 'ra' : 58.0, 'dec' : 35.0, 'SpectrumType' : PowerLaw }

     # Include catalog sources within this distance from the ROI center
     src_radius  : null

     # Include catalog sources within a box of width roisrc.
     src_roiwidth : 15.0

binning
-------

.. code-block:: yaml

   binning:

     # Binning
     roiwidth   : 10.0
     npix       : null
     binsz      : 0.1 # spatial bin size in deg
     binsperdec : 8   # nb energy bins per decade


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


components
----------

The *components* section is used to define a joint analysis formed by
the product of likelihoods for different subselection of the data
(implemented with the SummedLikelihood class in pyLikelihood).  This
section is optional and when set to null (the default) fermiPy will
construct a single likelihood using the parameters of the root
analysis configuration.

The component section can be defined as either a list or dictionary of
dictionary elements where each element sets analysis parameters for a
different subcomponent of the analysis.  Dictionary elements have the
same hierarchy of parameters as the root analysis configuration.
Parameters not defined in a given element will default to the values
set in the root analysis configuration.

The following example illustrates how to define a Front/Back analysis
with the a list of dictionaries.  In this case files associated to
each component will be named according to their order in the list
(e.g. file_00.fits, file_01.fits, etc.).

.. code-block:: yaml

   # Component section for Front/Back analysis with list style
   components:
     - { selection : { evtype : 1 } } # Front
     - { selection : { evtype : 2 } } # Back

This example illustrates how to define the components as a dictionary
of dictionaries.  In this case the files of a component will be
appended with its corresponding key (e.g. file_front.fits,
file_back.fits).

.. code-block:: yaml

   # Component section for Front/Back analysis with dictionary style
   components:
     front : { selection : { evtype : 1 } } # Front
     back  : { selection : { evtype : 2 } } # Back


