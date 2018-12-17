.. _developer:

Developer Notes
===============


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

   $ git tag -a XX.YY.ZZ -m ""
   $ git push --tags

5. Upload the release to pypi.
   
.. code-block:: bash

   $ python setup.py sdist upload -r pypi

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
