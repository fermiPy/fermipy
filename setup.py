#!/usr/bin/env python

from setuptools import setup
import os
import sys

from fermipy.version import get_git_version

##Check to make sure we're using the python in the STs.
if 'GLAST_EXT' in os.environ:
    print("Looks like you're using the SLAC version of the tools.")
    print("Not checking for correct python.")
elif 'FERMI_DIR' in os.environ:
    print("Looks like you're using the FSSC version of the tools.")
    fermi_dir = os.getenv('FERMI_DIR')
    os_file = os.__file__
    if fermi_dir not in os_file:
        print("Python executable is not the one in $FERMI_DIR/bin.  Exiting.")
        sys.exit(1)
else:
    print("Looks like the Fermi Science Tools are not setup.  Exiting.")
    sys.exit(1)

setup(name='fermipy',
      version=get_git_version(),
      author='The Fermipy developers',
      license='BSD',
      packages=['fermipy'],
      package_data = {
        '' : ['*yaml','*xml','*fit'],
        'fermipy' : ['catalogs/Extended_archive_v14/*fits',
                     'catalogs/Extended_archive_v14/*xml',
                     'catalogs/Extended_archive_v14/*/*fits',
                     'catalogs/Extended_archive_v14/*/*xml',
                     'catalogs/Extended_archive_v15/*fits',
                     'catalogs/Extended_archive_v15/*xml',
                     'catalogs/Extended_archive_v15/*/*fits',
                     'catalogs/Extended_archive_v15/*/*xml']},
      include_package_data = True, 
      url = "https://github.com/fermiPy/fermipy",
      scripts = [],
      data_files=[('fermipy',['fermipy/_version.py'])],
      install_requires=['numpy >= 1.6.1',
                        'matplotlib >= 1.1.0',
                        'astropy >= 0.4',
                        'pyyaml',
                        'healpy',
                        'ez_setup',
                        'wcsaxes',
                        'scipy >= 0.13'])
