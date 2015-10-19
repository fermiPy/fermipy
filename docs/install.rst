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
