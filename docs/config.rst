.. _config:

Configuration
=============

This page documents the configuration scheme used by the fermiPy
package and the valid options for the configuration file.

.. The fermiPy package is controlled through a yaml-format
.. configuration file.

##################################
Class Configuration
##################################

Classes in the fermiPy package follow a common convention by which the
the runtime behavior of a class instance can be controlled.
Internally every class instance has a dictionary that defines its
configuration state.  Elements of this dictionary can be scalars (str,
int ,float) or dictionaries defining nested blocks of the
configuration.

The configuration dictionary is set at the time of object creation by
passing a dictionary object to the class constructor.  Optional kwargs
arguments can be used to override options in the input dictionary.
For instance in the following example the *config* dictionary defines
values for the parameters *emin* and *emax*.  By passing an additional
dictionary for the selection block the value of emax in config is
overriden to 10000.

.. code-block:: python
   
   config = { 
   'selection' : { 'emin' : 100, 
                   'emax' : 1000 }   
   }

   gta = GTAnalysis(config,selection={'emax' : 10000})



##################################
Configuration File
##################################

fermiPy uses YAML-format configuration files which define a structured
hierarchy of parameters organized in blocks that mirrors the layout of
the configuration dictionary.  Each configuration block groups a set
of related options.  The following sub-sections describe the options
that can be set in each block.

fileio
------

The *fileio* block collects options related to file bookkeeping.  The
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

The *data* block defines the input data files for the analysis (FT1,
FT2, and livetime cube).  *evfile* and *scfile* can either be an
individual file or group of files.  The optional *ltcube*

.. code-block:: yaml

   data :
     evfile : ft1.lst
     scfile : ft2.fits 
     ltcube : null

model
-----
The *model* block collects options related to the ROI model.

.. code-block:: yaml

   model :
   
     # Include catalog sources within this distance from the ROI center
     src_radius  : null

     # Include catalog sources within a box of width roisrc.
     src_roiwidth : 15.0

     galdiff  : '/Users/mdwood/fermi/diffuse/v5r0/gll_iem_v06.fits'
     isodiff  : '/Users/mdwood/fermi/diffuse/v5r0/iso_P8R2_SOURCE_V6_v06.txt'
     limbdiff : null

     # List of catalogs to be used in the model.
     catalogs : 
       - 'gll_psc_v14.fit'

binning
-------

.. code-block:: yaml

   binning:


selection
---------

.. code-block:: yaml

   selection:

     # Data selections
     emin    : 100
     emax    : 100000
     zmax    : 90
     evclass : 128
     evtype  : 3
     tmin    : 239557414
     tmax    : 428903014 # 6 years

     # Set the ROI center to the coordinates of this source
     target : 'mkn421'

components
----------

The *components* block defines a set of configurations for
subcomponents of the analysis.  These configurations can either be
defined as a list or a dictionary.  This block is optional.




