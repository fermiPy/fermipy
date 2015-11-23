.. _install:

Installation
============

.. note:: 

   It is recommended to use fermiPy with ST v10r0p5 or later.

These instructions assume that you already have a local installation
of the Fermi STs.  Instructions for downloading and installing the STs
are provided through the `FSSC
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.  If you
are running at SLAC you can follow the instructions under `Running at
SLAC`_.

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


Download the latest version of fermiPy from the github repository:

.. code-block:: bash

   >>> git clone https://github.com/fermiPy/fermipy.git

Run the setup.py script.  This will install the fermiPy package itself
and its dependencies in your local python environment:

.. code-block:: bash

   >>> cd fermipy
   >>> python setup.py install --user

Running pip and setup.py with the *user* flag is recommended if you do not
have write access to your python installation (for instance if you are
running in a UNIX/Linux environment with a shared python
installation).  To install fermipy into the common package directory
of your python installation the *user* flag should be ommitted.

Finally, check that fermipy imports:

.. code-block:: bash

   >>> python
   Python 2.7.8 (default, Aug 20 2015, 11:36:15)
   [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.56)] on darwin
   Type "help", "copyright", "credits" or "license" for more information. 
   >>> from fermipy.gtanalysis import GTAnalysis
   >>> help(GTAnalysis)

Issues
------

If you get an error about importing matplotlib (specifically something
about the macosx backend) you might change your default backend to get
it working.  The [customizing matplotlib
page](http://matplotlib.org/users/customizing.html) details the
instructions to modify your default matplotlibrc file (you can pick
GTK or WX as an alternative).

In some cases the setup.py script will fail to properly install the
fermipy package dependecies.  If installation fails you can try
running a forced upgrade of these packages with `pip install --upgrade`:

.. code-block:: bash

   >>> pip install --upgrade numpy
   >>> pip install --upgrade matplotlib	
   >>> pip install --upgrade scipy
   >>> pip install --upgrade astropy	
   >>> pip install --upgrade pyyaml
   >>> pip install --upgrade healpy
   >>> pip install --upgrade wcsaxes
   >>> pip install --upgrade ipython
   >>> pip install --upgrade jupyter

Running at SLAC
---------------

This section provides specific installation instructions for running
on the SLAC cluster.  First source the `slacsetup.sh` script in the
fermipy directory:

.. code-block:: bash

   >>> source slacsetup.sh

This will setup your GLAST_EXT path and source the setup script for
one of the pre-built ST installations (default is 10-01-01).  

Then install fermipy with the package setup script:

.. code-block:: bash

   >>> git clone https://github.com/fermiPy/fermipy.git
   >>> cd fermipy
   >>> python setup.py install --user
