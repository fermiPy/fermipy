# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Analysis framework for all-sky diffuse emission fitting
"""
from __future__ import absolute_import, division, print_function

# Options for diffuse analysis
diffuse = {
    'binning_yaml': (None, 'Path to yaml file defining binning.', str),
    'dataset_yaml': (None, 'Path to yaml file defining dataset.', str),
    'diffuse_comp_yaml': (None, 'Path to yaml file defining diffuse components.', str),
    'catalog_comp_yaml': (None, 'Path to yaml file defining catalog components.', str),
    'hpx_order_ccube': (9, 'Maximum HEALPIX order for binning counts data.', int),
    'hpx_order_expcube': (6, 'Maximum HEALPIX order for exposure cubes.', int), 
    'hpx_order_fitting': (7, 'Maximum HEALPIX order for model fitting.', int),
    'coordsys': ('GAL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'irf_ver': ('V6', 'Version of IRFs to use.', str),
    }

# Options for residual cosmic-ray analysis
residual_cr = {
    'ft1file' : (None, 'Path to list of input FT1 files', str), 
    'binning_yaml' : (None, 'Path to yaml file defining binning.', str),
    'dataset_clean_yaml' : (None, 'Path to yaml file defining dataset for "clean" selection', str),
    'dataset_dirty_yaml' : (None, 'Path to yaml file defining dataset for "dirty" selection', str),
    'hpx_order_binning': (6, 'Maximum HEALPIX order for binning counts data.', int),
    'hpx_order_fitting' : (4, 'HEALPIX order for analysis', int),
    'coordsys' : ('GAL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'irf_ver': ('V6', 'Version of IRFs to use.', str),
    'dry_run' : (False, 'Print commands but do not run them', bool),
}
