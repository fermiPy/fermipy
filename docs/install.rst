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

Running pip and setup.py with the ``user`` flag is recommended if you do not
have write access to your python installation (for instance if you are
running in a UNIX/Linux environment with a shared python
installation).  To install fermipy into the common package directory
of your python installation the ``user`` flag should be ommitted.

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
it working.  The `customizing matplotlib
page <http://matplotlib.org/users/customizing.html>`_ details the
instructions to modify your default matplotlibrc file (you can pick
GTK or WX as an alternative).

In some cases the setup.py script will fail to properly install the
fermipy package dependecies.  If installation fails you can try
running a forced upgrade of these packages with ``pip install --upgrade``:

.. code-block:: bash

   >>> pip install --upgrade --user numpy matplotlib scipy astropy pyyaml healpy wcsaxes ipython jupyter

Installing with Anaconda Python
-------------------------------

.. note:: 

   The following instructions have only been verified to work with
   binary Linux distributions of the Fermi STs.  If you are using OSX
   or you have installed the STs from source you should follow the
   installation thread above.

These instructions explain how to use fermipy with a new or existing
conda python installation.  These instructions assume that you have
already downloaded and installed the Fermi STs from the FSSC and you
have set the ``FERMI_DIR`` environment variable to point to the location
of this installation.

The ``condainstall.sh`` script can be used to install fermipy into an
existing conda python installation or to create a minimal conda
installation from scratch.  In either case clone the fermipy git
repository and run the ``condainstall.sh`` installation script from
within the fermipy directory:

.. code-block:: bash

   >>> git clone https://github.com/fermiPy/fermipy.git; cd fermipy
   >>> bash condainstall.sh

If you do not already have anaconda python installed on your system
this script will create an installation under ``$HOME/miniconda``.  If
you already have conda installed (i.e. if the conda command is already
in your path) it will use your existing installation.  The script will
create a separate environment for your fermipy installation called
*fermi-env*.

Once fermipy is installed you can initialize the fermi environment by
running ``condasetup.sh``:

.. code-block:: bash

   >>> source condasetup.sh

This will both activate the fermi-env environment and set up your
shell environment to run the Fermi Science Tools.  The *fermi-env*
python environment can be exited by running:

.. code-block:: bash

   >>> source deactivate


Running at SLAC
---------------

This section provides specific installation instructions for running
on the SLAC cluster.  First checkout the fermipy git repository:

.. code-block:: bash

   >>> git clone https://github.com/fermiPy/fermipy.git
   >>> Cd fermipy

Then source the ``slacsetup.sh`` script in the fermipy directory and run
the ``slacsetup`` function:

.. code-block:: bash

   >>> source slacsetup.sh
   >>> slacsetup

This will setup your GLAST_EXT path and source the setup script for
one of the pre-built ST installations (default is 10-01-01).  To
manually override the ST version you can optionally provide the
release tag as an argument to the ``slacsetup`` function:

.. code-block:: bash

   >>> slacsetup 10-XX-XX

After setting up the STs environment, install fermipy with pip:

.. code-block:: bash

   >>> pip install fermipy --user

This will install fermipy under the ``$HOME/.local`` directory.  You
can verify that the installation has succeeded by importing
`~fermipy.gtanalysis.GTAnalysis`:

.. code-block:: bash

   >>> python
   Python 2.7.8 |Anaconda 2.1.0 (64-bit)| (default, Aug 21 2014, 18:22:21) 
   [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   Anaconda is brought to you by Continuum Analytics.
   Please check out: http://continuum.io/thanks and https://binstar.org
   >>> from fermipy.gtanalysis import GTAnalysis

For Developers
--------------

You have the option of either installing a tagged release
(recommended) or the latest commit on the master branch:

.. code-block:: bash

   # See the list of tags
   >>> git tag
   0.4.0
   0.5.0
   0.5.1
   0.5.2
   0.5.3
   0.5.4
   0.6.0
   0.6.1
   
   # Checkout a specific release tag (usually the latest one)
   >>> git checkout X.X.X 
   >>> python setup.py install --user 

.. code-block:: bash

   # Install the HEAD of the master branch
   >>> python setup.py install --user 

.. code-block:: bash

   # See the list of tags
   >>> git tag
   # Checkout a specific release tag (usually the latest one)
   >>> git checkout X.X.X 
   >>> python setup.py install --user 
