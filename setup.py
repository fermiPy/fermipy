#!/usr/bin/env python
from setuptools import setup, find_packages
from fermipy.version import get_git_version

setup(
    name='fermipy',
    version=get_git_version(),
    author='The Fermipy developers',
    author_email='fermipy.developers@gmail.com',
    description='A Python package for analysis of Fermi-LAT data',
    license='BSD',
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/fermiPy/fermipy",
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
    scripts=[],
    entry_points={'console_scripts': [
        'fermipy-dispatch = fermipy.scripts.dispatch:main',
        'fermipy-clone-configs = fermipy.scripts.clone_configs:main',
        'fermipy-collect-sources = fermipy.scripts.collect_sources:main',
        'fermipy-cluster-sources = fermipy.scripts.cluster_sources:main',
        'fermipy-flux-sensitivity = fermipy.scripts.flux_sensitivity:main',
        'fermipy-run-tempo = fermipy.scripts.run_tempo:main',
        'fermipy-select = fermipy.scripts.select_data:main',
        'fermipy-validate = fermipy.scripts.validate:main',
        'fermipy-coadd = fermipy.merge_utils:main',
    ]},
    install_requires=[
        'numpy >= 1.6.1',
        'astropy >= 1.2.1',
        'matplotlib >= 1.5.0',
        'scipy >= 0.14',
        'pyyaml',
        'healpy',
        'wcsaxes',
    ],
    extras_require=dict(
        all=[],
    ),
)
