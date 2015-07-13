#!/usr/bin/env python

from setuptools import setup

setup(name='fermipy',
      version='0.2.0',
      author='The Fermipy developers',
      packages=['fermipy','fermipy.config','fermipy.catalogs'],
      package_data = { '' : ['*yaml','*xml','*fit'] },
      include_package_data = True, 
      url = "https://github.com/fermiPy/fermipy",
      scripts = [],
      data_files=[],
      install_requires=['numpy >= 1.6.1',
                        'matplotlib >= 1.1.0',
                        'astropy >= 0.4',
                        'pyyaml',
                        'pywcsgrid2',
                        'scipy >= 0.13'])
