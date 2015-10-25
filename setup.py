#!/usr/bin/env python

from setuptools import setup
import re

VERSIONFILE="fermipy/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='fermipy',
      version=verstr,
      author='The Fermipy developers',
      packages=['fermipy','fermipy.config','fermipy.catalogs'],
      package_data = {
        '' : ['*yaml','*xml','*fit','*fits'],
        'fermipy.catalogs' : ['Extended_archive_v14/*fits',
                              'Extended_archive_v14/*xml',
                              'Extended_archive_v14/*/*fits',
                              'Extended_archive_v14/*/*xml']},
      include_package_data = True, 
      url = "https://github.com/fermiPy/fermipy",
      scripts = [],
      data_files=[],
      install_requires=['numpy >= 1.6.1',
                        'matplotlib >= 1.1.0',
                        'astropy >= 0.4',
                        'pyyaml',
                        'healpy',
                        'ez_setup',
                        'pywcsgrid2 >= 1.0-b3',
                        'scipy >= 0.13'])
