.. _tutorials:

Tutorials
=========

This page collects the IPython notebook tutorials with more detailed 
examples 
are available as IPython notebooks in the `notebooks
<https://github.com/fermiPy/fermipy-extra/tree/master/notebooks/>`_
directory of the `fermipy-extra
<https://github.com/fermiPy/fermipy-extra>`_ respository.  These
notebooks can be browsed as `static web pages
<http://nbviewer.jupyter.org/github/fermiPy/fermipy-extra/blob/master/notebooks/index.ipynb>`_
or run interactively by downloading the fermipy-extra repository and
running ``jupyter notebook`` in the notebooks directory:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy-extra.git    
   $ cd fermipy-extra/notebooks
   $ jupyter notebook index.ipynb

Note that this will require you to have both ipython and jupyter
installed in your python environment.  These can be installed in a
conda- or pip-based installation as follows:

.. code-block:: bash

   # Install with conda
   $ conda install ipython jupyter

   # Install with pip
   $ pip install ipython jupyter

One can also run the notebooks from a docker container following the
:ref:`dockerinstall` instructions:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy-extra.git    
   $ cd fermipy-extra
   $ docker pull fermipy/fermipy
   $ docker run -it --rm -p 8888:8888 -v $PWD:/workdir -w /workdir fermipy/fermipy

After launching the notebook server, paste the URL that appears into
your web browser and navigate to the *notebooks* directory.

.. toctree::
   :titlesonly:
   
   notebooks/SMC.ipynb