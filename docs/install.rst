.. _install:

Installation
============

.. note:: 

   Fermipy is only compatible with Science Tools v10r0p5 or later.  If
   you are using an earlier version, you will need to download and
   install the latest version from the `FSSC
   <http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.  Note
   that it is recommended to use the *non-ROOT* binary distributions
   of the Science Tools.

These instructions assume that you already have a local installation
of the Fermi Science Tools (STs).  For more information about
installing and setting up the STs see :ref:`stinstall`.  If you are
running at SLAC you can follow the `Running at SLAC`_ instructions.
For Unix/Linux users we currently recommend following the
:ref:`condainstall` instructions.  For OSX users we recommend
following the :ref:`pipinstall` instructions.

.. _stinstall:

Installing the Fermi Science Tools
----------------------------------

The Fermi STs are a prerequisite for fermipy.  To install the STs we
recommend using one of the non-ROOT binary distributions available
from the `FSSC
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.  The
following example illustrates how to install the binary distribution
on a Linux machine running Ubuntu Trusty:

.. code-block:: bash

   $ curl -OL http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/tar/ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA.tar.gz
   $ tar xzf ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA.tar.gz
   $ export FERMI_DIR=ScienceTools-v10r0p5-fssc-20150518-x86_64-unknown-linux-gnu-libc2.19-10-without-rootA/x86_64-unknown-linux-gnu-libc2.19-10
   $ source $FERMI_DIR/fermi-init.sh

More information about installing the STs as well as the complete list
of the available binary distributions is available on the `FSSC
software page
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.
   
.. _pipinstall:

Installing with pip
-------------------

These instructions cover installation with the ``pip`` package
management tool.  This method will install fermipy and its
dependencies into the python distribution that comes with the Fermi
Science Tools.  First verify that you're running the python from the
Science Tools

.. code-block:: bash

   $ which python

If this doesn't point to the python in your Science Tools install
(i.e. it returns /usr/bin/python or /usr/local/bin/python) then the
Science Tools are not properly setup.

Before starting the installation process, you will need to determine
whether you have setuptools and pip installed in your local python
environment.  You may need to install these packages if you are
running with the binary version of the Fermi Science Tools distributed
by the FSSC.  The following command will install both packages in your
local environment:

.. code-block:: bash

   $ curl https://bootstrap.pypa.io/get-pip.py | python -

Check if pip is correctly installed:

.. code-block:: bash

   $ which pip

Once again, if this isn't the pip in the Science Tools, something went
wrong.  Now install fermipy by running

.. code-block:: bash

   $ pip install fermipy

To run the ipython notebook examples you will also need to install
jupyter notebook:
   
.. code-block:: bash

   $ pip install jupyter

.. Running pip and setup.py with the ``user`` flag is recommended if you do not
.. have write access to your python installation (for instance if you are
.. running in a UNIX/Linux environment with a shared python
.. installation).  To install fermipy into the common package directory
.. of your python installation the ``user`` flag should be ommitted.

Finally, check that fermipy imports:

.. code-block:: bash

   $ python
   Python 2.7.8 (default, Aug 20 2015, 11:36:15)
   [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.56)] on darwin
   Type "help", "copyright", "credits" or "license" for more information. 
   >>> from fermipy.gtanalysis import GTAnalysis
   >>> help(GTAnalysis)

.. _condainstall:
   
Installing with Anaconda Python
-------------------------------

.. note:: 

   The following instructions have only been verified to work with
   binary Linux distributions of the Fermi STs.  If you are using OSX
   or you have installed the STs from source you should follow the
   :ref:`pipinstall` thread above.

These instructions cover how to use fermipy with a new or existing
conda python installation.  These instructions assume that you have
already downloaded and installed the Fermi STs from the FSSC and you
have set the ``FERMI_DIR`` environment variable to point to the location
of this installation.

The ``condainstall.sh`` script can be used to install fermipy into an
existing conda python installation or to create a minimal conda
installation from scratch.  In either case download and run the
``condainstall.sh`` installation script from the fermipy repository:

.. code-block:: bash

   $ curl -OL https://raw.githubusercontent.com/fermiPy/fermipy/master/condainstall.sh
   $ bash condainstall.sh

If you do not already have anaconda python installed on your system
this script will create a new installation under ``$HOME/miniconda``.
If you already have conda installed and the ``conda`` command is
in your path the script will use your existing installation.
The script will create a separate environment for your fermipy
installation called *fermi-env*.

Once fermipy is installed you can initialize the fermi environment by
running ``condasetup.sh``:

.. code-block:: bash

   $ curl -OL https://raw.githubusercontent.com/fermiPy/fermipy/master/condasetup.sh 
   $ source condasetup.sh

This will both activate the *fermi-env* environment and set up your
shell environment to run the Fermi Science Tools.  The *fermi-env*
python environment can be exited by running:

.. code-block:: bash

   $ source deactivate


Running at SLAC
---------------

This section provides specific installation instructions for running
in the SLAC computing environment.  First download and source the
``slacsetup.sh`` script:

.. code-block:: bash

   $ wget https://raw.githubusercontent.com/fermiPy/fermipy/master/slacsetup.sh -O slacsetup.sh
   $ source slacsetup.sh
   
To initialize the ST environment run the ``slacsetup`` function:

.. code-block:: bash

   $ slacsetup

This will setup your ``GLAST_EXT`` path and source the setup script
for one of the pre-built ST installations (the current default is
10-01-01).  To manually override the ST version you can optionally
provide the release tag as an argument to ``slacsetup``:

.. code-block:: bash

   $ slacsetup XX-XX-XX

Because users don't have write access to the ST python installation
all pip commands that install or uninstall packages must be executed
with the ``--user`` flag.  After initializing the STs environment,
install fermipy with pip:

.. code-block:: bash

   $ pip install fermipy --user

This will install fermipy in ``$HOME/.local``.  You can verify that
the installation has succeeded by importing
`~fermipy.gtanalysis.GTAnalysis`:

.. code-block:: bash

   $ python
   Python 2.7.8 |Anaconda 2.1.0 (64-bit)| (default, Aug 21 2014, 18:22:21) 
   [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   Anaconda is brought to you by Continuum Analytics.
   Please check out: http://continuum.io/thanks and https://binstar.org
   >>> from fermipy.gtanalysis import GTAnalysis

.. _upgrade:
   
Upgrading
---------

By default installing fermipy with ``pip`` will get the latest tagged
released available on the `PyPi <https://pypi.python.org/pypi>`_
package respository.  You can check your currently installed version
of fermipy with ``pip show``:

.. code-block:: bash

   $ pip show fermipy
   ---
   Metadata-Version: 2.0
   Name: fermipy
   Version: 0.6.7
   Summary: A Python package for analysis of Fermi-LAT data
   Home-page: https://github.com/fermiPy/fermipy
   Author: The Fermipy developers
   Author-email: fermipy.developers@gmail.com
   License: BSD
   Location: /home/vagrant/miniconda/envs/fermi-env/lib/python2.7/site-packages
   Requires: wcsaxes, astropy, matplotlib, healpy, scipy, numpy, pyyaml

To upgrade your fermipy installation to the latest version run the pip
installation command with ``--upgrade --no-deps``:
   
.. code-block:: bash
   
   $ pip install fermipy --upgrade --no-deps
   Collecting fermipy
   Installing collected packages: fermipy
     Found existing installation: fermipy 0.6.6
       Uninstalling fermipy-0.6.6:
         Successfully uninstalled fermipy-0.6.6
   Successfully installed fermipy-0.6.7

   
.. _gitinstall:
   
Building from Source
--------------------

These instructions describe how to install fermipy from its git source
code repository using ``setup.py``.  Installing from source is
necessary if you want to do local development or test features in an
untagged release.  Note that for non-expert users it is recommended to
install fermipy with ``pip`` following the instructions above.  First
clone the fermipy repository:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy.git
   $ cd fermipy

To install the head of the master branch run ``setup.py install`` from
the root of the source tree:

.. code-block:: bash

   # Install the latest version
   $ git checkout master
   $ python setup.py install --user 

A useful option if you are doing active code development is to install
your working copy as the local installation.  This can be done by
running ``setup.py develop``:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop --user 

You can later remove the link to your working copy by running the same
command with the ``--uninstall`` flag:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop --user --uninstall
   
You also have the option of installing a previous release tag.  To see
the list of release tags use ``git tag``:

.. code-block:: bash

   $ git tag
   0.4.0
   0.5.0
   0.5.1
   0.5.2
   0.5.3
   0.5.4
   0.6.0
   0.6.1

To install a specific release tag, run ``git checkout`` with the tag
name followed by ``setup.py install``:
   
.. code-block:: bash
   
   # Checkout a specific release tag
   $ git checkout X.X.X 
   $ python setup.py install --user 


   
Issues
------

If you get an error about importing matplotlib (specifically something
about the macosx backend) you might change your default backend to get
it working.  The `customizing matplotlib page
<http://matplotlib.org/users/customizing.html>`_ details the
instructions to modify your default matplotlibrc file (you can pick
GTK or WX as an alternative).  Specifically the ``TkAgg`` and
``macosx`` backends currently do not work on OSX if you upgrade
matplotlib to the version required by fermipy.  To get around this
issue you can enable the ``Agg`` backend at runtime:

.. code-block:: bash

   >>> import matplotlib
   >>> matplotlib.use('Agg')

However this backend does not support interactive plotting.

In some cases the setup.py script will fail to properly install the
fermipy package dependecies.  If installation fails you can try
running a forced upgrade of these packages with ``pip install --upgrade``:

.. code-block:: bash

   $ pip install --upgrade --user numpy matplotlib scipy astropy pyyaml healpy wcsaxes ipython jupyter
