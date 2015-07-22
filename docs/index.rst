.. Fermipy documentation master file, created by
   sphinx-quickstart on Fri Mar 20 15:40:47 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Fermipy's documentation!
===================================


   
##################################
Introduction
##################################

This is the fermiPy documentation page.  fermiPy is a set of python
modules and scripts that automate analysis with the Fermi Science
Tools.  fermipy provides a configuration-file driven workflow in which
the analysis parameters (data selection, IRFs, and ROI model) are
defined in a user-specified YAML file.  The analysis is controlled
with a set of python classes that provide methods to execute various
analysis tasks.  For a short introduction to using fermiPy see the
:ref:`quickstart` page.


Installation
------------

.. note:: 

   It is recommended to only use the fermipy package with ST v10r0p5
   or later.

These instructions assume that you already have a local installation
of the Fermi STs.  Instructions for downloading and installing the STs
are provided through the `FSSC
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.

Before starting the installation process, you will need to determine
whether you have setuptools and pip installed in your local python
environment.  You may need to install these packages if you are
running with the binary version of the Fermi Science Tools distributed
by the FSSC.  The following command will install both packages in your
local environment:

.. code-block:: bash

   >>> curl https://bootstrap.pypa.io/get-pip.py | python -

Next install the ez_setup module with pip (required by pywcsgrid2):

.. code-block:: bash

   >>> pip install ez_setup

Download the latest version of fermiPy from the github repository:

.. code-block:: bash

   >>> git clone https://github.com/fermiPy/fermipy.git

Run the setup.py script.  This will install the fermiPy package itself
and its dependencies in your local python environment:

.. code-block:: bash

   >>> cd fermipy
   >>> python setup.py install

Note that if you are running in an environment in which you do not have write
access to your python installation, you will need to run both pip and
setup.py with the *user* flag:

.. code-block:: bash

   >>> pip install ez_setup --user
   >>> python setup.py install --user


Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   config
   quickstart
   model
   fermipy


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

