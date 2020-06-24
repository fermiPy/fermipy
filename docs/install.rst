.. _install:

Installation
============

.. note:: 

   From version 0.19.0 fermipy is only compatible with
   fermitools version 1.2.23 or later.  If you are using an earlier
   version, you will need to download and
   install the latest version from the `FSSC
   <http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.  

These instructions assume that you already have a local installation
of the fermitools.  For more information about
installing and setting up the fermitools see :ref:`stinstall`.  
For we currently recommend following the :ref:`condainstall`
instructions.  However the :ref:`pipinstall` instructions should
also work.   The :ref:`dockerinstall` instructions can be used to
install the fermitools on OSX and Linux machines that are new enough to support Docker.  To
install the development version of Fermipy follow the
:ref:`devinstall` instructions.


.. _condainstall_script:

The condainstall.sh script
---------------------------

The recommended way to install fermipy and the fermitools is by using
the condainstall.sh script included in the package.   This script
properly handles a rather complicated set of interdependencies between
fermipy, the fermitools and packages they depend on.

.. code-block:: bash

   $ curl -OL https://raw.githubusercontent.com/fermiPy/fermipy/master/condainstall.sh
   $ export CONDA_PATH=<path to your conda installation>
   $ source condainstall.sh
   
This script optionally uses a number of other environmental variarbles
to control how the installtion is set up.    The important ones and
their default values are listed below.   Unless you want to override
some of these values you can leave them as is:

.. code-block:: bash

   $ export PYTHON_VERSION=2.7
   $ export CONDA_DEPS="scipy matplotlib pyyaml numpy astropy gammapy healpy"
   $ # This should point at your conda installation, or at the place you would like to install conda
   $ export CONDA_PATH="$HOME/minconda"
   $ # This is the name that will be given to the conda environment created for fermipy
   $ export FERMIPY_CONDA_ENV="fermipy"      
   $ # This is the command used to install the fermitools.
   $ # Set it to an empty string if you do not want to install the fermitools
   $ # of if you have already installed them.
   $ export ST_INSTALL="conda install -y --name $FERMIPY_CONDA_ENV $FERMI_CONDA_CHANNELS -c $CONDA_CHANNELS fermitools"
   $ # This is the command used install fermipy.
   $ # If you want to install for source or use a different version of
   $ # fermipy you should change this
   $ export INSTALL_CMD="conda install -y --name  $FERMIPY_CONDA_ENV -c $CONDA_CHANNELS fermipy"


.. _dev_install_script:

The dev_install.sh script
-------------------------

If you want to install fermipy from source, you can use the
'dev_install.sh' script included in the package.  This script sets the
values of the environmental variables listed above to values that are
suitable for installing from source

.. code-block:: bash

   $ git clone 
   $ cd fermipy
   $ <edit dev_install.sh to set CONDA_PATH and FERMIPY_CONDA_ENV>
   $ . dev_install.sh
   $ py.test # to test



.. _stinstall:

Installing the fermitools
-------------------------

.. note:: 

    If you used the condainstall.sh script, it should have already 
    installed the fermitools.   This example is if you want to
    install the fermitools without using that script.

The fermitools are a prerequisite for fermipy.  The
following example illustrates how the fermitools in an existing
anaconda installation.   

.. code-block:: bash

   $ conda create --name fermipy -y python=$PYTHON_VERSION
   $ conda activate fermipy
   $ conda install -y --name fermipy -c conda-forge/label/cf201901 -c
   fermi -c conda-forge fermitools"

More information about installing the fermitools is available on the `FSSC
software page
<http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.   More
information about setting up an anaconda installation is included in
the :ref:`condainstall` instructions below.


The diffuse emission models
------------------------------

Starting with fermipy version 0.19.0, we are using the diffuse and
istoropic emission model from the fermitools-data package rather
than including them in fermipy.    However, for working on older
analyses created with earlier version of fermipy you can set the
FERMI_DIFFUSE_DIR environmental variable to point at a directory
that include the version of the models that you wish to use.


.. _pipinstall:

Installing with pip
-------------------

These instructions cover installation with the ``pip`` package
management tool.  This will install fermipy and its dependencies into
the conda distribution that contains the fermitools.   We will assume
that you have installed the fermitools in a conda environment called "fermi".
First verify that you've installed from the fermitools

.. code-block:: bash

   $ conda activate fermi
   $ which girfs

If this doesn't point to the gtirfs in your fermitools install then the
fermitools are not properly set up.

Until the fermitools moves to python 3, we recommend making sure
that this environment includes python and pip

.. code-block:: bash

   $ conda activate fermi
   $ which girfs
   $ which pip

Both the gtirfs and pip should point to the versions installed in the
fermi environment.

Because of some issues with the dependendies in fermitoolts and
gammapy we recommend installing the dependedcies using conda.

.. code-block:: bash
		
   $ conda install -n fermi -y -c conda-forge scipy matplotlib pyyaml numpy astropy gammapy healpy
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
   
Installing Anaconda Python
--------------------------

These instructions cover how to use fermipy with a new or existing
anaconda python installation. 
   
If you do not have an anaconda installation, the ``condainstall.sh``
script can be used to create a minimal anaconda installation from
scratch.  First download and source the ``condainstall.sh`` script
from the fermipy repository:

.. code-block:: bash

   $ curl -OL https://raw.githubusercontent.com/fermiPy/fermipy/master/condainstall.sh
   $ source condainstall.sh

If you do not already have anaconda python installed on your system
this script will create a new installation under ``$HOME/miniconda``.
If you already have anaconda installed and the ``conda`` command is in
your path the script will use your existing installation.


.. _dockerinstall:

Installing with Docker
----------------------

.. note::

   This method for installing the STs is currently experimental
   and has not been fully tested on all operating systems.  If you
   encounter issues please try either the pip- or anaconda-based
   installation instructions.

Docker is a virtualization tool that can be used to deploy software in
portable containers that can be run on any operating system that
supports Docker.  Before following these instruction you should first
install docker on your machine following the `installation instructions
<https://docs.docker.com/engine/installation/>`_ for your operating
system.  Docker is currently supported on the following operating
systems:

* macOS 10.10.3 Yosemite or later
* Ubuntu Precise 12.04 or later
* Debian 8.0 or later
* RHEL7 or later
* Windows 10 or later

Note that Docker is not supported by RHEL6 or its variants (CentOS6,
Scientific Linux 6).

These instructions describe how to create a docker-based ST
installation that comes preinstalled with anaconda python and fermipy.
The installation is fully contained in a docker image that is roughly
2GB in size.  To see a list of the available images go to the `fermipy
Docker Hub page <https://hub.docker.com/r/fermipy/fermipy/tags/>`_.
Images are tagged with the release version of the STs that was used to
build the image (e.g. 11-05-00).  The *latest* tag points to the image
for the most recent ST release.

To install the *latest* image first download the image file:

.. code-block:: bash

   $ docker pull fermipy/fermipy
   
Now switch to the directory where you plan to run your analysis and execute
the following command to launch a docker container instance:

.. code-block:: bash
   
   $ docker run -it --rm -p 8888:8888 -v $PWD:/workdir -w /workdir fermipy/fermipy

This will start an ipython notebook server that will be attached to
port 8888.  Once you start the server it will print a URL that you can
use to connect to it with the web browser on your host machine.  The
`-v $PWD:/workdir` argument mounts the current directory to the
working area of the container.  Additional directories may be mounted
by adding more volume arguments ``-v`` with host and container paths
separated by a colon.

The same docker image may be used to launch python, ipython, or a bash
shell by passing the command as an argument to ``docker run``:

.. code-block:: bash
   
   $ docker run -it --rm -v $PWD:/workdir -w /workdir fermipy/fermipy ipython
   $ docker run -it --rm -v $PWD:/workdir -w /workdir fermipy/fermipy python
   $ docker run -it --rm -v $PWD:/workdir -w /workdir fermipy/fermipy /bin/bash

By default interactive graphics will not be enabled.  The following
commands can be used to enable X11 forwarding for interactive graphics
on an OSX machine.  This requires you to have installed XQuartz 2.7.10
or later.  First enable remote connections by default and start the X
server:

.. code-block:: bash
                
   $ defaults write org.macosforge.xquartz.X11 nolisten_tcp -boolean false
   $ open -a XQuartz

Now check that the X server is running and listening on port 6000:

.. code-block:: bash
                
   $ lsof -i :6000

If you don't see X11 listening on port 6000 then try restarting XQuartz.

Once you have XQuartz configured you can enable forwarding by setting
DISPLAY environment variable to the IP address of the host machine:

.. code-block:: bash

   $ export HOST_IP=`ifconfig en0 | grep "inet " | cut -d " " -f2`
   $ xhost +local:
   $ docker run -it --rm -e DISPLAY=$HOST_IP:0 -v $PWD:/workdir -w /workdir fermipy ipython


.. _devinstall:



Installing From Source
----------------------

The instructions describe how to install development versions of
Fermipy from source code.  Before installing a development version we recommend first
installing a tagged release following the :ref:`pipinstall` or
:ref:`condainstall` instructions above.

.. code-block:: bash
                
   $ git clone https://github.com/fermiPy/fermipy.git
   $ cd fermipy
   $ export INSTALL_CMD=" "
   $ source condainstall.sh
   $ # Consider using python setup.py develop
   $ # if you are doing active development
   $ python setup.py install 

   
   
Upgrading
---------

By default installing fermipy with ``pip`` or ``conda`` will get the latest tagged
released available on the `PyPi <https://pypi.python.org/pypi>`_
package respository.  You can check your currently installed version
of fermipy with ``pip show``:

.. code-block:: bash

   $ pip show fermipy

or ``conda info``:

.. code-block:: bash

   $ conda info fermipy
   
To upgrade your fermipy installation to the latest version run the pip
installation command with ``--upgrade --no-deps`` (remember to also
include the ``--user`` option if you're running at SLAC):
   
.. code-block:: bash
   
   $ pip install fermipy --upgrade --no-deps
   Collecting fermipy
   Installing collected packages: fermipy
     Found existing installation: fermipy 0.6.6
       Uninstalling fermipy-0.6.6:
         Successfully uninstalled fermipy-0.6.6
   Successfully installed fermipy-0.6.7

If you installed fermipy with ``conda`` the equivalent command is:

.. code-block:: bash

   $ conda update fermipy
   
   
.. _gitinstall:


Developer Installation
----------------------

These instructions describe how to install fermipy from its git source
code repository using the ``setup.py`` script.  Installing from source
can be useful if you want to make your own modifications to the
fermipy source code.  Note that non-developers are recommended to
install a tagged release of fermipy following the :ref:`pipinstall` or
:ref:`condainstall` instructions above.

First clone the fermipy git repository and cd to the root directory of
the repository:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy.git
   $ cd fermipy
   $ export INSTALL_CMD=" "
   $ source condainstall.sh

   
To install the latest commit in the master branch run ``setup.py
install`` from the root directory:

.. code-block:: bash

   # Install the latest commit
   $ git checkout master
   $ python setup.py install --user 

A useful option if you are doing active code development is to install
your working copy of the package.  This will create an installation in
your python distribution that is linked to the copy of the code in
your local repository.  This allows you to run with any local
modifications without having to reinstall the package each time you
make a change.  To install your working copy of fermipy run with the
``develop`` argument:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop --user 

You can later remove the link to your working copy by running the same
command with the ``--uninstall`` flag:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop --user --uninstall
   

Specific release tags can be installed by running ``git checkout``
before running the installation command:
   
.. code-block:: bash
   
   # Checkout a specific release tag
   $ git checkout X.X.X 
   $ python setup.py install --user 

To see the list of available release tags run ``git tag``.
   
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
issue you can switch to the ``Agg`` backend at runtime before
importing fermipy:

.. code-block:: bash

   >>> import matplotlib
   >>> matplotlib.use('Agg')

However note that this backend does not support interactive plotting.

If you are running OSX El Capitan or newer you may see errors like the following:

.. code-block:: bash
                
   dyld: Library not loaded

In this case you will need to disable the System Integrity Protections
(SIP).  See `here
<http://www.macworld.com/article/2986118/security/how-to-modify-system-integrity-protection-in-el-capitan.html>`_
for instructions on disabling SIP on your machine.

In some cases the setup.py script will fail to properly install the
fermipy package dependecies.  If installation fails you can try
running a forced upgrade of these packages with ``pip install --upgrade``:

.. code-block:: bash

   $ pip install --upgrade --user numpy matplotlib scipy astropy pyyaml healpy wcsaxes ipython jupyter
