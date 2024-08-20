.. _tutorials:

Tutorials
=========

This page collects various python notebooks that demonstrate how to use
fermipy. These notebooks are available in the `notebooks
<https://github.com/fermiPy/fermipy-extra/tree/master/notebooks/>`_
directory of the `fermipy-extra
<https://github.com/fermiPy/fermipy-extra>`_ repository. These
notebooks can be run interactively by downloading the fermipy-extra 
repository and running ``jupyter notebook`` in the notebooks directory:

.. code-block:: bash

   $ git clone https://github.com/fermiPy/fermipy-extra.git    
   $ cd fermipy-extra/notebooks
   $ jupyter notebook index.ipynb

Note that this will require you to have both ipython and jupyter
installed in your python environment. These can be installed in a
conda- or pip-based installation as follows:

.. code-block:: bash

   # Install with conda
   $ conda install ipython jupyter

   # Install with pip
   $ pip install ipython jupyter

Here is a list of available notebooks:

.. toctree::
   :titlesonly:
   
   notebooks/SMC.ipynb
   notebooks/pg1553.ipynb
   notebooks/optimize_model.ipynb
   notebooks/gtools_customize.ipynb
   notebooks/file_function_examples.ipynb
   notebooks/phase_analysis.ipynb
   notebooks/ic443.ipynb
   notebooks/draco.ipynb