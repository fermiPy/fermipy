# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Analysis framework for all-sky diffuse emission fitting
"""
from __future__ import absolute_import, division, print_function

# Options for diffuse analysis
diffuse = {
    'comp': (None, 'Path to yaml file defining binning.', str),
    'data': (None, 'Path to yaml file defining dataset.', str),
    'ft1file' : ('P8_P302_8years_source_zmax105.lst', 'Path to list of input FT1 files', str),
    'diffuse': (None, 'Path to yaml file defining diffuse components.', str),
    'sources': (None, 'Path to yaml file defining catalog components.', str),
    'hpx_order_ccube': (9, 'Maximum HEALPIX order for binning counts data.', int),
    'hpx_order_expcube': (6, 'Maximum HEALPIX order for exposure cubes.', int),
    'hpx_order_fitting': (7, 'Maximum HEALPIX order for model fitting.', int),
    'coordsys': ('GAL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'irf_ver': ('V6', 'Version of IRFs to use.', str),
    'make_xml' : (True, 'Make XML files.', bool),
    'dry_run' : (False, 'Print commands but do not run them', bool),
    }

# Options for residual cosmic-ray analysis
residual_cr = {
    'dataset_yaml': (None, 'Path to yaml file defining dataset.', str),
    'comp' : ('config/binning_4_residualCR.yaml', 'Path to yaml file defining binning.', str),
    'ft1file' : ('P8_P302_8years_source_zmax105.lst', 'Path to list of input FT1 files', str),
    'ft2file' : ('ft2_files/ft2_8years_moon.lst', 'Path to FT2 file', str),
    'hpx_order_binning': (6, 'Maximum HEALPIX order for binning counts data.', int),
    'hpx_order_fitting' : (4, 'HEALPIX order for analysis', int),
    'coordsys' : ('GAL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'irf_ver': ('V6', 'Version of IRFs to use.', str),
    'dry_run' : (False, 'Print commands but do not run them', bool),
    'full_output' : (False, 'Include diagnostic output', bool),
    }

# Options for residual cosmic-ray analysis
sun_moon = {
    'binning_yaml' : (None, 'Path to yaml file defining binning.', str),
    'dataset_yaml' : (None, 'Path to yaml file defining dataset.', str),
    'irf_ver': ('V6', 'Version of IRFs to use.', str),
    'sourcekeys' : (None, "Keys for sources to make template for", list),
    }

# Options relating to gtapps
gtopts = {
    'irfs' : ('CALDB', 'Instrument response functions', str),
    'evtype' : (None, 'Event type selections', int),
    'expcube' : (None, 'Input Livetime cube file', str),
    'cmap' : (None, 'Input counts cube file', str),
    'srcmaps' : (None, 'Input source maps file', str),
    'bexpmap' : (None, 'Input binned exposure map file', str),
    'srcmdl' : (None, 'Input source model xml file', str),
    'outfile' : (None, 'Output file', str),
    'coordsys' : ('GAL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'hpx_order': (6, 'HEALPIX order parameter', int),
    }

