#!/usr/bin/env python

from setuptools import setup
import re

from fermipy.version import get_git_version

setup(name='fermipy',
      version=get_git_version(),
      author='The Fermipy developers',
      license='BSD',
      packages=['fermipy','fermipy.config','fermipy.catalogs'],
      package_data = {
        '' : ['*yaml','*xml','*fit'],
        'fermipy.catalogs' : ['Extended_archive_v14/*fits',
                              'Extended_archive_v14/*xml',
                              'Extended_archive_v14/*/*fits',
                              'Extended_archive_v14/*/*xml',
                              'Extended_archive_v15/*fits',
                              'Extended_archive_v15/*xml',
                              'Extended_archive_v15/*/*fits',
                              'Extended_archive_v15/*/*xml']},
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
