# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Analysis framework for all-sky diffuse emission fitting
"""
from __future__ import absolute_import, division, print_function

# Options for diffuse analysis
diffuse = {
    'config': (None, 'Config yaml file', str),
    'comp': ('config/binning.yaml', 'Path to yaml file defining binning.', str),
    'data': ('config/dataset_source.yaml', 'Path to yaml file defining dataset.', str),
    'ft1file': ('P8_P305_8years_source_zmax105.lst', 'Path to list of input FT1 files', str),
    'ft2file': ('ft2_8years.lst', 'Path to list of input FT2 files', str),
    'library': ('models/library.yaml', 'Path to yaml file defining model components.', str),
    'models': ('models/modellist.yaml', 'Path to yaml file defining models.', str),
    'hpx_order_ccube': (9, 'Maximum HEALPIX order for binning counts data.', int),
    'hpx_order_expcube': (6, 'Maximum HEALPIX order for exposure cubes.', int),
    'hpx_order_fitting': (7, 'Maximum HEALPIX order for model fitting.', int),
    'mktimefilter': (None, 'Key for gtmktime selection', str),
    'do_ltsum': (False, 'Run gtltsum on inputs', bool),
    'make_xml': (True, 'Make XML files.', bool),
    'dry_run': (False, 'Print commands but do not run them', bool),
    'scratch': (None, 'Path to scratch area.', str),
}

# Options for residual cosmic-ray analysis
residual_cr = {
    'ccube_dirty': (None, 'Input counts cube for dirty event class.', str),
    'ccube_clean': (None, 'Input counts cube for clean event class.', str),
    'bexpcube_dirty': (None, 'Input exposure cube for dirty event class.', str),
    'bexpcube_clean': (None, 'Input exposure cube for clean event class.', str),
    'clean': ('ultracleanveto', 'Clean event class', str),
    'dirty': ('source', 'Dirty event class', str),
    'select_factor': (5.0, 'Pixel selection factor for Aeff Correction', float),
    'mask_factor': (2.0, 'Pixel selection factor for output mask', float),
    'sigma': (3.0, 'Width of gaussian to smooth output maps [degrees]', float),
    'full_output': (False, 'Include diagnostic output', bool),
}


# Options for residual cosmic-ray analysis
sun_moon = {
    'sourcekeys': (None, "Keys for sources to make template for", list),
}

# Options relating to gtapps
gtopts = {
    'emin': (100., 'Minimum energy [MeV]', float),
    'emax': (100000., 'Maximum energy [MeV]', float),
    'enumbins': (16, 'Number of energy bins', int),
    'zmax': (100., 'Maximum zenith angle [degrees]', float),
    'irfs': ('CALDB', 'Instrument response functions', str),
    'evtype': (None, 'Event type selections', int),
    'evclass': (None, 'Event Class', int),
    'expcube': (None, 'Input Livetime cube file', str),
    'cmap': (None, 'Input counts cube file', str),
    'srcmaps': (None, 'Input source maps file', str),
    'bexpmap': (None, 'Input binned exposure map file', str),
    'srcmdl': (None, 'Input source model xml file', str),
    'infile': (None, 'Input file', str),
    'evfile': (None, 'Input FT1 eventfile', str),
    'outfile': (None, 'Output file', str),
    'hpx_order': (6, 'HEALPIX order parameter', int),
    'coordsys': ('GAL', "Coordinate system", str),
    'pfiles': (None, "PFILES directory", str),
    'clobber': (False, "Overwrite files", bool),
}
