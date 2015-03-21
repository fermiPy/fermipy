#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name='fermipy',
      version='0.1.0',
      author='The Fermipy developers',
      packages=['fermipy'],
      url = "https://github.com/fermiPy/fermipy",
      scripts = [],
      data_files=[],
      install_requires=['numpy >= 1.8.2',
                        'matplotlib >= 1.2.0',
                        'astropy >= 0.3',
                        'scipy >= 0.13'])
