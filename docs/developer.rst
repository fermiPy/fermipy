.. _developer:

Developer Notes
===============


Adding a spectral model
-----------------------

One of the most common changes to the underlying fermitools code is to add
a new spectral model.   To be able to use that model in fermipy will
require a few changes, depending on how exactly you would like you
use the model.

1. At a minimum, the model, and default values and bounds for the parameters need to
   be added to ``fermipy/data/models.yaml``
2. If you want to be able to use functions that free the source-shape
   parameters, fit the SED, you will want to modify the
   ``norm_parameters`` and ``shape_parameters`` blocks at the top of
   the ``fermipy/gtanalysis.py`` file to include the new spectral
   model.
3. If you want to be able to include the spectral model in an xml
   'catalog' of sources that you use to define an ROI, you will have
   to update the ``spectral_pars_from_catalog`` and ``get_catalog_dict``
   functions in ``fermipy/roi_model.py`` to include the spectral
   model.
4. If the spectral model is included in a new source catalog, and you
   want to support that catalog, see the section below on supporting new
   catalogs.
5. If you want to use the spectral to do more complicated things, like
   vectorizing call to evalute the spectrum because you are using it
   in sensitivity studies, then you will have to add it the the
   ``fermipy/spectrum.py`` file.   That is pretty much expert territory.


Supporting a new catalog
-------------------------

To support a new catalog will require some changes in the
``fermipy/catalog.py`` file.   In short

1. Define a class to manage the catalog.   This will have to handle
   converting the parameters in the FITS file to the format that
   fermipy expects.  It should inherit from the ``Catalog`` class.
2. Update the ``Catalog.create`` function to have a hook to create a
   class of the correct type.
3. For now we are also maintaining the catalog files in the
   ``fermipy/data/catalogs`` area, so the catalog files should be
   added to that area.

Installing development versions of fermitools
---------------------------------------------

To test fermipy against future versions of fermitools, use the ``rc`` (release candidate) or ``dev`` (development) versions in the respective conda channels, e.g. ``mamba install -c fermi/label/rc fermitools=2.1.36``.

Installing development versions of fermipy
------------------------------------------

You can use `pip` to install `fermipy` directly from a git branch or from a local directory. Inside an environment that should already contain fermitools and its dependencies, use ``pip install -U git+https://github.com/fermiPy/fermipy@branch_name``.

Installing a local version:

.. code-block:: bash

   $ cd $MY_FERMIY_DIR
   $ git clone git@github.com:fermiPy/fermipy.git
   $ cd ./fermipy
   $ git checkout -b "new_feature_branch"
   [make local changes]
   $ pip install .


Running tests locally
---------------------

Note: The tests create a number of temporary files. It is recommended to run them inside an empty directory and remove the directory afterwards.

.. code-block:: bash
  
  $ pip install pytest
  $ pytest --pyargs fermipy



Creating a New Release
----------------------

The following are steps for creating a new release:

1. Update the Changelog page (in ``docs/changelog.rst``) with notes
   for the release and commit those changes.
2. Update documentation tables by running ``make_tables.py`` inside
   the ``docs`` subdirectory and commit any resulting changes to the
   configuration table files under ``docs/config``.
3. Checkout ``master`` and ensure that you have pulled all commits from origin.
4. Create the release tag and push it to GitHub.
   
.. code-block:: bash

   $ git tag -a XX.YY.ZZ -m "version XX.YY.ZZ"
   $ git push --tags

5. Upload the release to pypi.
   
.. code-block:: bash

   $ pip install build twine
   [inside the fermipy directory:]
   $ python -m build
   $ python -m twine upload dist/fermipy-1.1.2*
   Username: __token__
   Password: [secret token from pypi.org]

Or alternatively:

.. code-block:: bash

   $ python setup.py sdist upload -r pypi
   $ twine upload dist/fermipy-XX.YY.ZZ.tar.gz

6. Create a new release on conda-forge by opening a PR on the
   `fermipy-feedstock
   <https://github.com/conda-forge/fermipy-feedstock>`_ repo.  There
   is a fork of ``fermipy-feedstock`` in the fermipy organization that
   you can use for this purpose.  Edit ``recipe/meta.yaml`` by
   entering the new package version and updating the sha256 hash to
   the value copied from the `pypi download
   <https://pypi.org/project/fermipy/#files>`_ page.  Update the
   package dependencies as necessary in the ``run`` section of
   ``requirements``.  Verify that ``entry_points`` contains the
   desired set of command-line scripts.  Generally this section should
   match the contents ``entry_points`` in ``setup.py``.  Before
   merging the PR confirm that all tests have successfully passed.
