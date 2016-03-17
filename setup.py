#!/usr/bin/env python

from setuptools import setup, find_packages
import os
import sys

from fermipy.version import get_git_version

##Check to make sure we're using the python in the STs.
#if 'GLAST_EXT' in os.environ:
#    print("Looks like you're using the SLAC version of the tools.")
#    print("Not checking for correct python.")
#elif 'FERMI_DIR' in os.environ:
#    print("Looks like you're using the FSSC version of the tools.")
#    fermi_dir = os.getenv('FERMI_DIR')
#    os_file = os.__file__
#    if fermi_dir not in os_file:
#        print("Python executable is not the one in $FERMI_DIR/bin.  Exiting.")
#        sys.exit(1)
#else:
#    print("Looks like the Fermi Science Tools are not setup.  Exiting.")
#    sys.exit(1)

setup(name='fermipy',
      version=get_git_version(),
      author='The Fermipy developers',
      author_email='fermipy.developers@gmail.com',
      description='A Python package for analysis of Fermi-LAT data',
      license='BSD',
      packages=find_packages(exclude='tests'),
      include_package_data = True, 
      url = "https://github.com/fermiPy/fermipy",
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 4 - Beta',
      ],
      scripts = [],
      install_requires=['numpy >= 1.6.1',
                        'matplotlib >= 1.4.0',
                        'scipy >= 0.14',
                        'astropy >= 1.0',
                        'pyyaml',
                        'healpy',
                        'wcsaxes'])

