.. _config:

Configuration
=============

This page documents the configuration scheme adopted in the fermiPy
package and the valid options for the configuration file.

.. The fermiPy package is controlled through a yaml-format
.. configuration file.

##################################
Class Configuration
##################################

The fermiPy classes follow a common convention by which the
configuration of the class can be controlled.  Internally every class
instance owns a configuration dictionary that controls its behavior.
Elements of this dictionary can be scalars (str, int ,float) or
dictionaries defining nested blocks of the configuration.

The configuration dictionary is set at the time of object
creation by passing a dictionary object to the class constructor.  An
optional set of kwargs arguments can be used to override any options
in the input dictionary.  For instance in the following example the
*config* dictionary defines values for the parameters emin and emax.
By passing an additional dictionary for the common block the value of
emax in config is overriden to 10000.

.. code-block:: python
   
   config = { 
   'common' : { 'emin' : 100, 
                'emax' : 1000 }   
   }

   gta = GTAnalysis(config,common={'emax' : 10000})



##################################
Configuration File
##################################

The fermiPy configuration file uses the YAML format which defines a
structure hierarchy of parameters.  Each block of the configuration
groups together a set of related options.  This section describes the
options that can be defined in each block.

fileio
------

The *fileio* block defines options related to file bookkeeping.  The
*outdir* option sets the root directory of the analysis instance.  All
output of the analysis scripts will be written to this directory.  If
*outdir* is null then the output directory will be set to the
directory of the configuration file.  The *stageoutput* option will 

.. code-block:: yaml

   fileio:

     # Set the output directory
     outdir : null

     # Enable staging analysis output to an intermediate scratch directory
     stageoutput : False

     # Set the root directory under which the scratch directory should
     # be written
     scratchdir  : '/scratch'

data
----

The *data* block defines the input data files for the analysis (FT1,
FT2, and livetime cube).  evfile and scfile can either be an
individual file or group of files.

.. code-block:: yaml

   data :
     evfile : ft1.lst
     scfile : ft2.fits 
     ltcube : null

model
-----
The *model* block 

.. code-block:: yaml

   model :

binning
-------

selection
---------

components
----------

The *components* block defines a set of configurations for
subcomponents of the analysis.  These configurations can either be
defined as a list or a dictionary.  This block is optional.




