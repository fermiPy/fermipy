.. Fermipy documentation master file, created by
   sphinx-quickstart on Fri Mar 20 15:40:47 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fermipy's documentation!
===================================

##################################
Introduction
##################################

This is the Fermipy documentation page.  Fermipy is a set of python
modules and scripts that automate analysis with the `Fermi Science
Tools <http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/>`_.
fermipy provides a configuration-file driven workflow in which the
analysis parameters (data selection, IRFs, and ROI model) are defined
in a user-specified YAML file.  The analysis is controlled with a set
of python classes that provide methods to execute various analysis
tasks.  For instruction on installing Fermipy see the :ref:`install`
page.  For a short introduction to using Fermipy see the
:ref:`quickstart`.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 3

   install
   quickstart
   config
   output
   fitting
   model
   advanced/index
   fermipy

Indices and tables
==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

