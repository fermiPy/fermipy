.. _install:

Installation
============

.. note:: 

   It is recommended to use fermiPy with ST v10r0p5 or later.

These instructions assume that you already have a local installation
of the Fermi STs.  Instructions for downloading and installing the STs
are provided through the `FSSC
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.

Before starting the installation process, you will need to determine
whether you have setuptools and pip installed in your local python
environment.  You may need to install these packages if you are
running with the binary version of the Fermi Science Tools distributed
by the FSSC.  

Verify that you're running the python from the Science Tools

.. code-block:: bash

   >>> which python

If this doesn't point to the python in your Science Tools install
(i.e. it returnes /usr/bin/python or /usr/local/bin/python) then the
Science Tools are not properly setup.

Then, install pip. The following command will install both packages in
your local environment:

.. code-block:: bash

   >>> curl https://bootstrap.pypa.io/get-pip.py | python -

Check if pip is correctly installed:

.. code-block:: bash

   >>> which pip

Once again, if this isn't the pip in the Science Tools, something went
wrong.

Next install the ez_setup module with pip (required by pywcsgrid2):

.. code-block:: bash

   >>> pip install ez_setup

Then install several needed packages.  Note that these should be
installed properly by the setup.py script below but sometimes this
fails since they might not be setup correctly within the Science Tools.

.. code-block:: bash

   >>> pip install ez_setup
   >>> pip install --upgrade numpy
   >>> pip install --upgrade matplotlib	
   >>> pip install --upgrade scipy
   >>> pip install --upgrade astropy	
   >>> pip install --upgrade pyyaml
   >>> pip install --upgrade healpy
   >>> pip install --upgrade pywcsgrid2
   >>> pip install --upgrade ipython
   >>> pip install --upgrade jupyter

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

Finally, check that fermipy imports:

.. code-block:: bash

   >>> python
   Python 2.7.8 (default, Aug 20 2015, 11:36:15)
   [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.56)] on darwin
   Type "help", "copyright", "credits" or "license" for more information. 
   >>> from fermipy.gtanalysis import GTAnalysis
   >>> help(GTAnalysis)