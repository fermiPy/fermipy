# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Analysis defaults options for stacking pipeline analysis
"""
from __future__ import absolute_import, division, print_function

from fermipy.jobs import defaults as base_defaults

generic = {
    'limitfile': (None, 'Path to file with limits.', str),
}
generic.update(base_defaults.generic)

common = {
    'roster': (None, 'Name of a stacking target roster.', str),
    'rosters': ([], 'Name of a stacking target roster.', list),
    'rosterlist': (None, 'Path to the roster list.', str),
    'alias_dict': (None, 'File to rename target version keys.', str),
    'specconfig': (None, 'Path to yaml file defining stacking spectra of interest.', str),
    'specfile': (None, 'Path to spectrum file.', str),
    'astro_value_file': (None, 'Path to yaml file with target stacking normalizaiton', str),
    'astro_prior': (None, 'Types of Prior on stacking normalization', str),
    'astro_priors': ([], 'Types of Prior on stacking normalization', list),
    'spatial_models': ([], 'Types of spatial models to use', list),
    'specs': ([], 'Spectra to consider', list),
    'spec': ('powerlaw', 'Spectral model', str),
    'index': (2.0, 'Spectral index', float),
    'spec_type': ('eflux', 'Type of flux to consider', str),
    'global_min': (False, 'Use global min for castro plots.', bool),
    'clobber': (False, 'Overwrite existing files.', bool),
}
common.update(base_defaults.common)

sims = {}
sims.update(base_defaults.sims)

collect = {
    'bands': (None, 'Name of file with expected limit bands.', str),
}
collect.update(base_defaults.collect)
