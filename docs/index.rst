.. Fermipy documentation master file, created by
   sphinx-quickstart on Fri Mar 20 15:40:47 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fermipy's documentation!
===================================

##################################
Introduction
##################################

This is the Fermipy documentation page.  Fermipy is a python package
that facilitates analysis of data from the Large Area Telescope (LAT)
with the `Fermi Science Tools
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/>`_.  For
more information about the Fermi mission and the LAT instrument please
refer to the `Fermi Science Support Center
<http://fermi.gsfc.nasa.gov/ssc/>`_.

The Fermipy package is built on the pyLikelihood interface of the
Fermi Science Tools and provides a set of high-level tools for
performing common analysis tasks:

* Data and model preparation with the gt-tools (gtselect, gtmktime,
  etc.).

* Extracting a spectral energy distribution (SED) of a source.

* Generating TS and residual maps for a region of interest.

* Finding new source candidates.

* Localizing a source or fitting its spatial extension.

Fermipy uses a configuration-file driven workflow in which the
analysis parameters (data selection, IRFs, and ROI model) are defined
in a YAML configuration file.  Analysis is executed through a python
script that calls the methods of `~fermipy.gtanalysis.GTAnalysis` to
perform different analysis operations.

For instructions on installing Fermipy see the :ref:`install` page.
For a short introduction to using Fermipy see the :ref:`quickstart`.

Getting Help
------------

If you have questions about using Fermipy please open a `GitHub Issue
<https://github.com/fermiPy/fermipy/issues>`_ or email the `Fermipy
developers <mailto:fermipy.developers@gmail.com>`_.

Acknowledging Fermipy
---------------------

To acknowledge Fermipy in a publication please cite `Wood et al. 2017
<http://adsabs.harvard.edu/abs/2017arXiv170709551W>`_.


Documentation Contents
----------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   install
   quickstart
   config
   output
   fitting
   model
   advanced/index
   validation/index
   fermipy
   fermipy_jobs
   fermipy_diffuse
   changelog

Indices and tables
==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

