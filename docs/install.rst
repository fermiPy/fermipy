.. _install:

Installation
============

.. note:: 

   From version 1.0.1 fermipy is only compatible with
   fermitools version 2 or later, and with python version 3.7 or
   higher.
   If you are using an earlier version, you will need to download and
   install the latest version from the `FSSC
   <http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`_.  

These instructions will install fermipy as well as it's dependecies.


.. _conda_installation:

Conda-based installation
------------------------

The recommended way to install fermipy and the fermitools by using conda.

Conda properly handles a rather complicated set of interdependencies between
fermipy, the fermitools and packages they depend on.

.. code-block:: bash

   $ conda create --name fermipy -c conda-forge -c fermi fermipy
   

.. _installing from source:

Installing from source
----------------------

If you want to install fermipy from source you can use the
environment.yml file to install the dependecies. Installing from source
can be useful if you want to make your own modifications to the
fermipy source code.  Note that non-developers are recommended to
install a tagged release of fermipy following the
:ref:`conda_installation` instructions above.

To set up a conda environment with the dependencies

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy.git
   $ cd fermipy
   $ conda create --name fermipy -f environment.yml
   
To install the latest commit in the master branch run ``setup.py
install`` from the root directory:

.. code-block:: bash

   # Install the latest commit
   $ git checkout master
   $ python setup.py install 

A useful option if you are doing active code development is to install
your working copy of the package.  This will create an installation in
your python distribution that is linked to the copy of the code in
your local repository.  This allows you to run with any local
modifications without having to reinstall the package each time you
make a change.  To install your working copy of fermipy run with the
``develop`` argument:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop

You can later remove the link to your working copy by running the same
command with the ``--uninstall`` flag:

.. code-block:: bash

   # Install a link to your source code installation
   $ python setup.py develop --uninstall
   

Specific release tags can be installed by running ``git checkout``
before running the installation command:
   
.. code-block:: bash
   
   # Checkout a specific release tag
   $ git checkout X.X.X 
   $ python setup.py install
   
To see the list of available release tags run ``git tag``.




The diffuse emission models
------------------------------

Starting with fermipy version 0.19.0, we are using the diffuse and
istoropic emission model from the fermitools-data package rather
than including them in fermipy.    However, for working on older
analyses created with earlier version of fermipy you can set the
FERMI_DIFFUSE_DIR environmental variable to point at a directory
that include the version of the models that you wish to use.


   
   
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
   
