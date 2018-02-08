# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
from collections import OrderedDict
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
import fermipy.skymap
from fermipy.data_struct import MutableNamedTuple


def make_default_dict(d):

    o = {}
    for k, v in d.items():
        o[k] = copy.deepcopy(v[0])

    return o


def make_default_tuple(d):
    vals = [(k, copy.deepcopy(v[0])) for k, v in d.items()]
    return MutableNamedTuple(vals)


def make_attrs_class(typename, d):

    import attr

    vals = {}
    for k, v in d.items():
        if v[2] == float:
            vals[k] = attr.ib(
                default=v[0], validator=attr.validators.instance_of(v[2]))
        else:
            vals[k] = attr.ib(default=v[0])
    C = attr.make_class(typename, vals)
    return C()


DIFF_FLUX_UNIT = ':math:`\mathrm{cm}^{-2}~\mathrm{s}^{-1}~\mathrm{MeV}^{-1}`'
FLUX_UNIT = ':math:`\mathrm{cm}^{-2}~\mathrm{s}^{-1}`'
ENERGY_FLUX_UNIT = ':math:`\mathrm{MeV}~\mathrm{cm}^{-2}~\mathrm{s}^{-1}`'

# Options that are common to several sections
common = {
    'multithread': (False, 'Split the calculation across number of processes set by nthread option.', bool),
    'nthread': (None, 'Number of processes to create when multithread is True.  If None then one process '
                'will be created for each available core.', int),
    'model': (None, 'Dictionary defining the spatial/spectral properties of the test source. '
              'If model is None the test source will be a PointSource with an Index 2 power-law spectrum.', dict),
    'free_background': (False, 'Leave background parameters free when performing the fit. If True then any '
                        'parameters that are currently free in the model will be fit simultaneously '
                        'with the source of interest.', bool),
    'fix_shape': (False, 'Fix spectral shape parameters of the source of interest. If True then only '
                  'the normalization parameter will be fit.', bool),
    'free_radius': (None, 'Free normalizations of background sources within this angular distance in degrees '
                    'from the source of interest.  If None then no sources will be freed.', float),
    'make_plots': (False, 'Generate diagnostic plots.', bool),
    'use_weights' : (False, 'Used weighted version of maps in making plots.', bool),
    'write_fits': (True, 'Write the output to a FITS file.', bool),
    'write_npy': (True, 'Write the output dictionary to a numpy file.', bool),
    'loge_bounds':  (None, 'Restrict the analysis to an energy range (emin,emax) in '
                     'log10(E/MeV) that is a subset of the analysis energy range. '
                     'By default the full analysis energy range will be used.  If '
                     'either emin/emax are None then only an upper/lower bound on '
                     'the energy range wil be applied.', list),
}

# Options for defining input data files
data = {
    'evfile': (None, 'Path to FT1 file or list of FT1 files.', str),
    'scfile': (None, 'Path to FT2 (spacecraft) file.', str),
    'ltcube': (None, 'Path to livetime cube.  If none a livetime cube will be generated with ``gtmktime``.', str),
    'cacheft1': (True, 'Cache FT1 files when performing binned analysis.  If false then only the counts cube is retained.', bool),
}

# Options for data selection.
selection = {
    'emin': (None, 'Minimum Energy (MeV)', float),
    'emax': (None, 'Maximum Energy (MeV)', float),
    'logemin': (None, 'Minimum Energy (log10(MeV))', float),
    'logemax': (None, 'Maximum Energy (log10(MeV))', float),
    'tmin': (None, 'Minimum time (MET).', int),
    'tmax': (None, 'Maximum time (MET).', int),
    'zmax': (None, 'Maximum zenith angle.', float),
    'evclass': (None, 'Event class selection.', int),
    'evtype': (None, 'Event type selection.', int),
    'convtype': (None, 'Conversion type selection.', int),
    'phasemin': (None, 'Minimum pulsar phase', float),
    'phasemax': (None, 'Maximum pulsar phase', float),
    'target': (None, 'Choose an object on which to center the ROI.  '
                     'This option takes precendence over ra/dec or glon/glat.', str),
    'ra': (None, '', float),
    'dec': (None, '', float),
    'glat': (None, '', float),
    'glon': (None, '', float),
    'radius': (None, 'Radius of data selection.  If none this will be automatically set from the ROI size.', float),
    'filter': (None, 'Filter string for ``gtmktime`` selection.', str),
    'roicut': ('no', '', str)
}

# Options for ROI model.
model = {
    'src_radius':
        (None,
         'Radius of circular region in degrees centered on the ROI that selects '
         'sources for inclusion in the model.  If this parameter is none then no '
         'selection is applied.  This selection is ORed with the ``src_roiwidth`` selection.',
         float),
    'src_roiwidth':
        (None,
         'Width of square region in degrees centered on the ROI that selects '
         'sources for inclusion in the model.  If this parameter is none then no '
         'selection is applied.  This selection will be ORed with the ``src_radius`` selection.', float),
    'src_radius_roi':
        (None,
         'Half-width of ``src_roiwidth`` selection.  This parameter can be used in '
         'lieu of ``src_roiwidth``.',
         float),
    'isodiff': (None, 'Set the path to one or more isotropic templates.  A separate component will be '
                'generated for each item in this list.', list),
    'galdiff': (None, 'Set the path to one or more galactic IEM mapcubes.  A separate component will be '
                'generated for each item in this list.', list),
    'limbdiff': (None, '', list),
    'diffuse': (None, '', list),
    'sources': (None, '', list),
    'extdir': (None, 'Set a directory that will be searched for extended source FITS templates.  Template files in this directory '
               'will take precendence over catalog source templates with the same name.', str),
    'diffuse_dir': (None, '', list),
    'catalogs': (None, '', list),
    'merge_sources':
        (True, 'Merge properties of sources that appear in multiple '
         'source catalogs.  If merge_sources=false then subsequent sources with '
         'the same name will be ignored.', bool),
    'assoc_xmatch_columns':
        (['3FGL_Name'], 'Choose a set of association columns on which to '
         'cross-match catalogs.', list),
    'extract_diffuse': (
        False, 'Extract a copy of all mapcube components centered on the ROI.',
        bool)
}

# Options for configuring likelihood analysis
gtlike = {
    'irfs': (None, 'Set the IRF string.', str),
    'edisp': (True, 'Enable the correction for energy dispersion.', bool),
    'edisp_disable': (None,
                      'Provide a list of sources for which the edisp '
                      'correction should be disabled.',
                      list),
    'minbinsz': (0.05, 'Set the minimum bin size used for resampling diffuse maps.', float),
    'rfactor': (2, '', int),
    'convolve': (True, '', bool),
    'resample': (True, '', bool),
    'srcmap': (None, 'Set the source maps file.  When defined this file will be used instead of the '
               'local source maps file.', str),
    'bexpmap': (None, '', str),
    'bexpmap_roi': (None, '', str),
    'srcmap_base': (None, 'Set the baseline source maps file.  This will be used to generate a scaled source map.', str),
    'bexpmap_base': (None, 'Set the basline all-sky expoure map file.  This will be used to generate a scaled source map.', str),
    'bexpmap_roi_base': (None, 'Set the basline ROI expoure map file.  This will be used to generate a scaled source map.', str),
    'use_external_srcmap': (False, 'Use an external precomputed source map file.', bool),
    'use_scaled_srcmap': (False, 'Generate source map by scaling an external srcmap file.', bool),
    'wmap': (None, 'Likelihood weights map.', str),
    'llscan_npts': (20, 'Number of evaluation points to use when performing a likelihood scan.', int),
    'src_expscale': (None, 'Dictionary of exposure corrections for individual sources keyed to source name.  The exposure '
                     'for a given source will be scaled by this value.  A value of 1.0 corresponds to the nominal exposure.', dict),
    'expscale': (None, 'Exposure correction that is applied to all sources in the analysis component.  '
                 'This correction is superseded by `src_expscale` if it is defined for a source.', float),
}

# Options for generating livetime cubes
ltcube = {
    'binsz': (1.0, 'Set the angular bin size for generating livetime cubes.', float),
    'phibins': (0, 'Set the number of phi bins for generating livetime cubes.', int),
    'dcostheta': (0.025, 'Set the inclination angle binning represented as the cosine of the off-axis angle.', float),
    'use_local_ltcube': (False, 'Generate a livetime cube in the vicinity of the ROI using interpolation. '
                         'This option disables LT cube generation with gtltcube.', bool),
}

# Options for binning.
binning = {
    'projtype': ('WCS', 'Projection mode (WCS or HPX).', str),
    'proj': ('AIT', 'Spatial projection for WCS mode.', str),
    'coordsys': ('CEL', 'Coordinate system of the spatial projection (CEL or GAL).', str),
    'npix':
        (None,
         'Number of pixels.  If none then this will be set from ``roiwidth`` '
         'and ``binsz``.', int),
    'roiwidth': (10.0,
                 'Width of the ROI in degrees.  The number of pixels in each spatial dimension will be set from ``roiwidth`` / ``binsz`` (rounded up).',
                 float),
    'binsz': (0.1, 'Spatial bin size in degrees.', float),
    'binsperdec': (8, 'Number of energy bins per decade.', float),
    'enumbins': (
        None,
        'Number of energy bins.  If none this will be inferred from energy '
        'range and ``binsperdec`` parameter.', int),
    'hpx_ordering_scheme': ('RING', 'HEALPix Ordering Scheme', str),
    'hpx_order': (10, 'Order of the map (int between 0 and 12, included)', int),
    'hpx_ebin': (True, 'Include energy binning', bool)
}

# Options related to I/O and output file bookkeeping
fileio = {
    'outdir': (None, 'Path of the output directory.  If none this will default to the directory containing the configuration file.', str),
    'scratchdir': ('/scratch', 'Path to the scratch directory.  If ``usescratch`` is True then a temporary working directory '
                   'will be created under this directory.', str),
    'workdir': (None, 'Path to the working directory.', str),
    'logfile': (None, 'Path to log file.  If None then log will be written to fermipy.log.', str),
    'savefits': (True, 'Save intermediate FITS files.', bool),
    'workdir_regex': (['\.fits$|\.fit$|\.xml$|\.npy$'],
                      'Stage files to the working directory that match at least one of the regular expressions in this list.  '
                      'This option only takes effect when ``usescratch`` is True.', list),
    'outdir_regex': (['\.fits$|\.fit$|\.xml$|\.npy$|\.png$|\.pdf$|\.yaml$'],
                     'Stage files to the output directory that match at least one of the regular expressions in this list.  '
                     'This option only takes effect when ``usescratch`` is True.', list),
    'usescratch': (
        False, 'Run analysis in a temporary working directory under ``scratchdir``.', bool),
}

logging = {
    'prefix': ('', 'Prefix that will be appended to the logger name.', str),
    'chatter': (3, 'Set the chatter parameter of the STs.', int),
    'verbosity': (3, '', int)
}

# Options related to likelihood optimizer
optimizer = {
    'optimizer':
        ('MINUIT', 'Set the optimization algorithm to use when maximizing the '
                   'likelihood function.', str),
    'tol': (1E-3, 'Set the optimizer tolerance.', float),
    'max_iter': (100, 'Maximum number of iterations for the Newtons method fitter.', int),
    'init_lambda': (1E-4, 'Initial value of damping parameter for step size calculation '
                    'when using the NEWTON fitter.  A value of zero disables damping.', float),
    'retries': (3, 'Set the number of times to retry the fit when the fit quality is less than ``min_fit_quality``.', int),
    'min_fit_quality': (2, 'Set the minimum fit quality.', int),
    'verbosity': (0, '', int)
}

fit_output = {
    'edm': (None, 'Estimated distance to maximum of log-likelihood function.', float),
    'fit_status': (None, 'Optimizer return code (0 = ok).', int, 'int'),
    'fit_quality': (None, 'Fit quality parameter for MINUIT and NEWMINUIT optimizers (3 - Full accurate covariance matrix, '
                    '2 - Full matrix, but forced positive-definite (i.e. not accurate), '
                    '1 - Diagonal approximation only, not accurate, '
                    '0 - Error matrix not calculated at all)', int, 'int'),
    'covariance': (None, 'Covariance matrix between free parameters of the fit.', np.ndarray),
    'correlation': (None, 'Correlation matrix between free parameters of the fit.', np.ndarray),
    'dloglike': (None, 'Improvement in log-likehood value.', float),
    'loglike': (None, 'Post-fit log-likehood value.', float),
    'values': (None, 'Vector of best-fit parameter values (unscaled).', np.ndarray),
    'errors': (None, 'Vector of parameter errors (unscaled).', np.ndarray),
    'config': (None, 'Copy of input configuration to this method.', dict),
}

# MC options
mc = {
    'seed': (None, '', int)
}

# ROI Optimization
roiopt = {
    'npred_threshold': (1.0, '', float),
    'npred_frac': (0.95, '', float),
    'shape_ts_threshold':
        (25.0, 'Threshold on source TS used for determining the sources '
         'that will be fit in the third optimization step.', float),
    'max_free_sources':
        (5, 'Maximum number of sources that will be fit simultaneously in '
         'the first optimization step.', int),
    'skip':
        (None, 'List of str source names to skip while optimizing.', list)
}

roiopt_output = {
    'loglike0': (None, 'Pre-optimization log-likelihood value.', float),
    'loglike1': (None, 'Post-optimization log-likelihood value.', float),
    'dloglike': (None, 'Improvement in log-likehood value.', float),
    'config': (None, 'Copy of input configuration to this method.', dict),
}

# Residual Maps
residmap = {
    'model': common['model'],
    'exclude': (None, 'List of sources that will be removed from the model when '
                'computing the residual map.', list),
    'loge_bounds': common['loge_bounds'],
    'make_plots': common['make_plots'],
    'use_weights': common['use_weights'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# TS Map
tsmap = {
    'model': common['model'],
    'exclude': (None, 'List of sources that will be removed from the model when '
                'computing the TS map.', list),
    'multithread': common['multithread'],
    'nthread': common['nthread'],
    'max_kernel_radius': (3.0, 'Set the maximum radius of the test source kernel.  Using a '
                          'smaller value will speed up the TS calculation at the loss of '
                          'accuracy.', float),
    'loge_bounds': common['loge_bounds'],
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# TS Cube
tscube = {
    'model': common['model'],
    'do_sed': (True, 'Compute the energy bin-by-bin fits', bool),
    'nnorm': (10, 'Number of points in the likelihood v. normalization scan', int),
    'norm_sigma': (5.0, 'Number of sigma to use for the scan range ', float),
    'cov_scale_bb': (-1.0, 'Scale factor to apply to global fitting '
                     'cov. matrix in broadband fits. ( < 0 -> no prior ) ', float),
    'cov_scale': (-1.0, 'Scale factor to apply to broadband fitting cov. '
                  'matrix in bin-by-bin fits ( < 0 -> fixed ) ', float),
    'tol': (1E-3, 'Critetia for fit convergence (estimated vertical distance to min < tol )', float),
    'max_iter': (30, 'Maximum number of iterations for the Newtons method fitter.', int),
    'tol_type': (0, 'Absoulte (0) or relative (1) criteria for convergence.', int),
    'remake_test_source': (False, 'If true, recomputes the test source image (otherwise just shifts it)', bool),
    'st_scan_level': (0, 'Level to which to do ST-based fitting (for testing)', int),
    'init_lambda': (0, 'Initial value of damping parameter for newton step size calculation.   A value of zero disables damping.', float),
}

# Options for Source Finder
sourcefind = {
    'model': common['model'],
    'min_separation': (1.0,
                       'Minimum separation in degrees between sources detected in each '
                       'iteration. The source finder will look for the maximum peak '
                       'in the TS map within a circular region of this radius.', float),
    'sqrt_ts_threshold': (5.0, 'Source threshold in sqrt(TS).  Only peaks with sqrt(TS) '
                          'exceeding this threshold will be used as seeds for new '
                          'sources.', float),
    'max_iter': (5, 'Maximum number of source finding iterations.  The source '
                 'finder will continue adding sources until no additional '
                 'peaks are found or the number of iterations exceeds this '
                 'number.', int),
    'sources_per_iter': (4, 'Maximum number of sources that will be added in each '
                         'iteration.  If the number of detected peaks in a given '
                         'iteration is larger than this number, only the N peaks with '
                         'the largest TS will be used as seeds for the current '
                         'iteration.', int),
    'tsmap_fitter': ('tsmap', 'Set the method for generating the TS map.  Valid options are tsmap or tscube.', str),
    'free_params': (None, '', list),
    'multithread': common['multithread'],
    'nthread': common['nthread'],
}

# Options for lightcurve analysis
lightcurve = {
    'outdir': (None, r'Store all data in this directory (e.g. "30days"). If None then use current directory.', str),
    'use_local_ltcube': (True, 'Generate a fast LT cube.', bool),
    'use_scaled_srcmap': (False, 'Generate approximate source maps for each time bin by scaling '
                          'the current source maps by the exposure ratio with respect to that time bin.', bool),
    'save_bin_data': (True, 'Save analysis directories for individual time bins.  If False then only '
                      'the analysis results table will be saved.', bool),
    'binsz': (86400.0, 'Set the lightcurve bin size in seconds.', float),
    'shape_ts_threshold': (16.0, 'Set the TS threshold at which shape parameters of '
                           'sources will be freed.  If a source is detected with TS less than this '
                           'value then its shape parameters will be fixed to values derived from the '
                           'analysis of the full time range.', float),
    'nbins': (None, 'Set the number of lightcurve bins.  The total time range will be evenly '
              'split into this number of time bins.', int),
    'time_bins': (None, 'Set the lightcurve bin edge sequence in MET.  This option '
                  'takes precedence over binsz and nbins.', list),
    'free_background': common['free_background'],
    'free_radius': common['free_radius'],
    'free_sources': (None, 'List of sources to be freed.  These sources will be added to the list of sources '
                     'satisfying the free_radius selection.', list),
    'free_params': (None, 'Set the parameters of the source of interest that will be re-fit in each time bin. '
                    'If this list is empty then all parameters will be freed.', list),
    'max_free_sources':
        (5, 'Maximum number of sources that will be fit simultaneously with the source of interest.', int),
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
    'multithread': common['multithread'],
    'nthread': common['nthread'],
    'systematic': (0.02, 'Systematic correction factor for TS:subscript:`var`. See Sect. 3.6 in 2FGL for details.', float),
}

# Output for lightcurve Analysis
lightcurve_output = OrderedDict((
    ('name', (None, 'Name of Source'', ', str)),
    ('tmin', (None, 'Lower edge of time bin in MET.', np.ndarray)),
    ('tmax', (None, 'Upper edge of time bin in MET.', np.ndarray)),
    ('fit_success', (None, 'Did the likelihood fit converge? True if yes.',
                     np.ndarray)),
    ('config', ({}, 'Copy of the input configuration to this method.', dict)),
    ('ts_var', (None, r'TS of variability. Should be distributed as :math:`\chi^2` with '
                ':math:`n-1` degrees of freedom, where :math:`n` is the number of time bins.', float)),
))

# Options for SED analysis
sed = {
    'bin_index': (2.0, 'Spectral index that will be use when fitting the energy distribution within an energy bin.', float),
    'use_local_index': (False, 'Use a power-law approximation to the shape of the global spectrum in '
                        'each bin.  If this is false then a constant index set to `bin_index` '
                        'will be used.', bool),
    'free_background': common['free_background'],
    'free_radius': common['free_radius'],
    'free_pars': (None, 'Set the parameters of the source of interest that will be freed when performing '
                  'the global fit.  By default all parameters will be freed.', list),
    'ul_confidence': (0.95, 'Confidence level for flux upper limit.',
                      float),
    'cov_scale': (3.0, 'Scale factor that sets the strength of the prior on nuisance '
                  'parameters that are free.  Setting this to None disables the prior.', float),
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# Output for SED analysis
sed_output = OrderedDict((
    ('loge_min', (None, 'Lower edges of SED energy bins (log10(E/MeV)).',
                  np.ndarray)),
    ('loge_max', (None, 'Upper edges of SED energy bins (log10(E/MeV)).',
                  np.ndarray)),
    ('loge_ctr', (None, 'Centers of SED energy bins (log10(E/MeV)).',
                  np.ndarray)),
    ('loge_ref', (None, 'Reference energies of SED energy bins (log10(E/MeV)).',
                  np.ndarray)),
    ('e_min', (None, 'Lower edges of SED energy bins (MeV).',
               np.ndarray)),
    ('e_max', (None, 'Upper edges of SED energy bins (MeV).',
               np.ndarray)),
    ('e_ctr', (None, 'Centers of SED energy bins (MeV).', np.ndarray)),
    ('e_ref', (None, 'Reference energies of SED energy bins (MeV).',
               np.ndarray)),
    ('ref_flux', (None, 'Flux of the reference model in each bin (%s).' %
                  FLUX_UNIT,  np.ndarray)),
    ('ref_eflux', (None, 'Energy flux of the reference model in each bin (%s).' %
                   ENERGY_FLUX_UNIT,  np.ndarray)),
    ('ref_dnde', (None, 'Differential flux of the reference model evaluated at the bin center (%s)' %
                  DIFF_FLUX_UNIT,  np.ndarray)),
    ('ref_dnde_e_min', (None, 'Differential flux of the reference model evaluated at the lower bin edge (%s)' %
                        DIFF_FLUX_UNIT,  np.ndarray)),
    ('ref_dnde_e_max', (None, 'Differential flux of the reference model evaluated at the upper bin edge (%s)' %
                        DIFF_FLUX_UNIT,  np.ndarray)),
    ('ref_e2dnde', (None, 'E^2 x the differential flux of the reference model evaluated at the bin center (%s)' %
                    ENERGY_FLUX_UNIT,  np.ndarray)),
    ('ref_npred', (None, 'Number of predicted counts in the reference model in each bin.',
                   np.ndarray)),
    ('norm', (None, 'Normalization in each bin in units of the reference model.',
              np.ndarray)),
    ('flux', (None, 'Flux in each bin (%s).' %
              FLUX_UNIT, np.ndarray)),
    ('eflux', (None, 'Energy flux in each bin (%s).' %
               ENERGY_FLUX_UNIT, np.ndarray)),
    ('dnde', (None, 'Differential flux in each bin (%s).' %
              DIFF_FLUX_UNIT, np.ndarray)),
    ('e2dnde', (None, 'E^2 x the differential flux in each bin (%s).' %
                ENERGY_FLUX_UNIT, np.ndarray)),
    ('dnde_err', (None, '1-sigma error on dnde evaluated from likelihood curvature.',
                  np.ndarray)),
    ('dnde_err_lo', (None, 'Lower 1-sigma error on dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray)),
    ('dnde_err_hi', (None, 'Upper 1-sigma error on dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray)),
    ('dnde_ul95', (None, '95% CL upper limit on dnde evaluated from the profile likelihood (MINOS errors).',
                   np.ndarray)),
    ('dnde_ul', (None, 'Upper limit on dnde evaluated from the profile likelihood using a CL = ``ul_confidence``.',
                 np.ndarray)),
    ('e2dnde_err', (None, '1-sigma error on e2dnde evaluated from likelihood curvature.',
                    np.ndarray)),
    ('e2dnde_err_lo', (None, 'Lower 1-sigma error on e2dnde evaluated from the profile likelihood (MINOS errors).',
                       np.ndarray)),
    ('e2dnde_err_hi', (None, 'Upper 1-sigma error on e2dnde evaluated from the profile likelihood (MINOS errors).',
                       np.ndarray)),
    ('e2dnde_ul95', (None, '95% CL upper limit on e2dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray)),
    ('e2dnde_ul', (None, 'Upper limit on e2dnde evaluated from the profile likelihood using a CL = ``ul_confidence``.',
                   np.ndarray)),
    ('ts', (None, 'Test statistic.', np.ndarray)),
    ('loglike', (None, 'Log-likelihood of model for the best-fit amplitude.',
                 np.ndarray)),
    ('npred', (None, 'Number of model counts.', np.ndarray)),
    ('fit_quality', (None, 'Fit quality parameter for MINUIT and NEWMINUIT optimizers (3 - Full accurate covariance matrix, '
                     '2 - Full matrix, but forced positive-definite (i.e. not accurate), '
                     '1 - Diagonal approximation only, not accurate, '
                     '0 - Error matrix not calculated at all).', np.ndarray)),
    ('fit_status', (None, 'Fit status parameter (0=ok).', np.ndarray)),
    ('index', (None, 'Spectral index of the power-law model used to fit this bin.',
               np.ndarray)),
    ('norm_scan', (None, 'Array of NxM normalization values for the profile likelihood scan in N '
                   'energy bins and M scan points.  A row-wise multiplication with '
                   'any of ``ref`` columns can be used to convert this matrix to the '
                   'respective unit.',
                   np.ndarray)),
    ('dloglike_scan', (None, 'Array of NxM delta-loglikelihood values for the profile likelihood '
                       'scan in N energy bins and M scan points.', np.ndarray)),
    ('loglike_scan', (None, 'Array of NxM loglikelihood values for the profile likelihood scan '
                      'in N energy bins and M scan points.', np.ndarray)),
    ('param_covariance', (None, 'Covariance matrix for the best-fit spectral parameters of the source.',
                          np.ndarray)),
    ('param_names', (None, 'Array of names for the parameters in the global spectral parameterization of this source.',
                     np.ndarray)),
    ('param_values', (None, 'Array of parameter values.', np.ndarray)),
    ('param_errors', (None, 'Array of parameter errors.', np.ndarray)),
    ('model_flux', (None, 'Dictionary containing the differential flux uncertainty '
                    'band of the best-fit global spectral parameterization for the '
                    'source.', dict)),
    ('config', (None, 'Copy of input configuration to this method.', dict)),
))

# Options for extension analysis
extension = {
    'spatial_model': ('RadialGaussian', 'Spatial model that will be used to test the source'
                      'extension.  The spatial scale parameter of the '
                      'model will be set such that the 68% containment radius of '
                      'the model is equal to the width parameter.', str),
    'width': (None, 'Sequence of values in degrees for the likelihood scan over spatial extension '
              '(68% containment radius).  If this argument is None then the scan points will '
              'be determined from width_min/width_max/width_nstep.', list),
    'fit_position': (False, 'Perform a simultaneous fit to the source position and extension.', bool),
    'width_min': (0.01, 'Minimum value in degrees for the likelihood scan over spatial extent.', float),
    'width_max': (1.0, 'Maximum value in degrees for the likelihood scan over spatial extent.', float),
    'width_nstep': (21, 'Number of scan points between width_min and width_max. '
                    'Scan points will be spaced evenly on a logarithmic scale '
                    'between `width_min` and `width_max`.', int),
    'free_background': common['free_background'],
    'fix_shape': common['fix_shape'],
    'free_radius': common['free_radius'],
    'fit_ebin': (False, 'Perform a fit for the angular extension in each analysis energy bin.', bool),
    'update': (False, 'Update this source with the best-fit model for spatial '
               'extension if TS_ext > ``tsext_threshold``.', bool),
    'save_model_map': (False, 'Save model counts cubes for the best-fit model of extension.', bool),
    'sqrt_ts_threshold': (None, 'Threshold on sqrt(TS_ext) that will be applied when ``update`` is True.  If None then no'
                          'threshold is applied.', float),
    'psf_scale_fn': (None, 'Tuple of two vectors (logE,f) defining an energy-dependent PSF scaling function '
                     'that will be applied when building spatial models for the source of interest.  '
                     'The tuple (logE,f) defines the fractional corrections f at the sequence of energies '
                     'logE = log10(E/MeV) where f=0 corresponds to no correction.  The correction function f(E) is evaluated '
                     'by linearly interpolating the fractional correction factors f in log(E).  The '
                     'corrected PSF is given by P\'(x;E) = P(x/(1+f(E));E) where x is the angular separation.',
                     tuple),
    'make_tsmap': (True, 'Make a TS map for the source of interest.', bool),
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# Options for localization analysis
localize = {
    'nstep': (5, 'Number of steps in longitude/latitude that will be taken '
              'when refining the source position.  The bounds of the scan '
              'range are set to the 99% positional uncertainty as '
              'determined from the TS map peak fit.  The total number of '
              'sampling points will be nstep**2.', int),
    'dtheta_max': (0.5, 'Half-width of the search region in degrees used for the first pass of the localization search.', float),
    'free_background': common['free_background'],
    'fix_shape': common['fix_shape'],
    'free_radius': common['free_radius'],
    'update': (True, 'Update the source model with the best-fit position.', bool),
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# Output for localization analysis
localize_output = OrderedDict((

    ('name', (None, 'Name of source.', str)),
    ('file', (None, 'Name of output FITS file.', str)),
    ('config', ({}, 'Copy of the input configuration to this method.', dict)),

    # Position
    ('ra', (np.nan, 'Right ascension of best-fit position (deg).', float)),
    ('dec', (np.nan, 'Declination of best-fit position (deg).', float)),
    ('glon', (np.nan, 'Galactic Longitude of best-fit position (deg).', float)),
    ('glat', (np.nan, 'Galactic Latitude of best-fit position (deg).', float)),
    ('xpix', (np.nan, 'Longitude pixel coordinate of best-fit position.', float)),
    ('ypix', (np.nan, 'Latitude pixel coordinate of best-fit position.', float)),
    ('deltax', (np.nan, 'Longitude offset from old position (deg).', float)),
    ('deltay', (np.nan, 'Latitude offset from old position (deg).', float)),
    ('skydir', (None, '', astropy.coordinates.SkyCoord,
                '`~astropy.coordinates.SkyCoord`')),
    ('ra_preloc', (np.nan, 'Right ascension of pre-localization position (deg).', float)),
    ('dec_preloc', (np.nan, 'Declination of pre-localization position (deg).', float)),
    ('glon_preloc', (np.nan,
                     'Galactic Longitude of pre-localization position (deg).', float)),
    ('glat_preloc', (np.nan,
                     'Galactic Latitude of pre-localization position (deg).', float)),

    # Positional Errors
    ('ra_err', (np.nan, 'Std. deviation of positional uncertainty in right ascension (deg).', float)),
    ('dec_err', (np.nan, 'Std. deviation of positional uncertainty in declination (deg).', float)),
    ('glon_err', (np.nan, 'Std. deviation of positional uncertainty in galactic longitude (deg).', float)),
    ('glat_err', (np.nan, 'Std. deviation of positional uncertainty in galactic latitude (deg).', float)),
    ('pos_offset', (np.nan, 'Angular offset (deg) between the old and new (localized) source positions.', float)),
    ('pos_err', (np.nan, '1-sigma positional uncertainty (deg).', float)),
    ('pos_r68', (np.nan, '68% positional uncertainty (deg).', float)),
    ('pos_r95', (np.nan, '95% positional uncertainty (deg).', float)),
    ('pos_r99', (np.nan, '99% positional uncertainty (deg).', float)),
    ('pos_err_semimajor', (np.nan,
                           '1-sigma uncertainty (deg) along major axis of uncertainty ellipse.', float)),
    ('pos_err_semiminor', (np.nan,
                           '1-sigma uncertainty (deg) along minor axis of uncertainty ellipse.', float)),
    ('pos_angle', (np.nan, 'Position angle of uncertainty ellipse with respect to major axis.', float)),
    ('pos_ecc', (np.nan,
                 'Eccentricity of uncertainty ellipse defined as sqrt(1-b**2/a**2).', float)),
    ('pos_ecc2', (np.nan,
                  'Eccentricity of uncertainty ellipse defined as sqrt(a**2/b**2-1).', float)),
    ('pos_gal_cov', (np.nan * np.ones((2, 2)),
                     'Covariance matrix of positional uncertainties in local projection in galactic coordinates.',
                     np.ndarray)),
    ('pos_gal_corr', (np.nan * np.ones((2, 2)),
                      'Correlation matrix of positional uncertainties in local projection in galactic coordinates.',
                      np.ndarray)),
    ('pos_cel_cov', (np.nan * np.ones((2, 2)),
                     'Covariance matrix of positional uncertainties in local projection in celestial coordinates.',
                     np.ndarray)),
    ('pos_cel_corr', (np.nan * np.ones((2, 2)),
                      'Correlation matrix of positional uncertainties in local projection in celestial coordinates.',
                      np.ndarray)),

    # Maps
    ('tsmap', (None, '', fermipy.skymap.Map)),
    ('tsmap_peak', (None, '', fermipy.skymap.Map)),

    # Miscellaneous
    ('loglike_init', (np.nan, 'Log-Likelihood of model before localization.', float)),
    ('loglike_base', (np.nan, 'Log-Likelihood of model after initial spectral fit.', float)),
    ('loglike_loc', (np.nan, 'Log-Likelihood of model after localization.', float)),
    ('dloglike_loc', (np.nan,
                      'Difference in log-likelihood before and after localization.', float)),
    ('fit_success', (True, '', bool)),
    ('fit_inbounds', (True, '', bool)),
    ('fit_init', (None, '', dict)),
    ('fit_scan', (None, '', dict)),

))

# Output for extension analysis
extension_output = OrderedDict((

    ('name', (None, 'Name of source.', str)),
    ('file', (None, 'Name of output FITS file.', str)),
    ('config', ({}, 'Copy of the input configuration to this method.', dict)),

    # Extension
    ('width', (None, 'Vector of width (intrinsic 68% containment radius) values (deg).',
               np.ndarray)),
    ('dloglike', (None, 'Delta-log-likelihood values for each point in the profile likelihood scan.',
                  np.ndarray)),
    ('loglike', (None, 'Log-likelihood values for each point in the scan over the spatial extension.',
                 np.ndarray)),
    ('loglike_ptsrc', (np.nan,
                       'Log-Likelihood value of the best-fit point-source model.', float)),
    ('loglike_ext', (np.nan, 'Log-Likelihood of the best-fit extended source model.', float)),
    ('loglike_init', (np.nan, 'Log-Likelihood of model before extension fit.', float)),
    ('loglike_base', (np.nan, 'Log-Likelihood of model after initial spectral fit.', float)),
    ('ext', (np.nan, 'Best-fit extension (68% containment radius) (deg).', float)),
    ('ext_err_hi', (np.nan,
                    'Upper (1-sigma) error on the best-fit extension (deg).', float)),
    ('ext_err_lo', (np.nan,
                    'Lower (1-sigma) error on the best-fit extension (deg).', float)),
    ('ext_err', (np.nan, 'Symmetric (1-sigma) error on the best-fit extension (deg).', float)),
    ('ext_ul95', (np.nan, '95% CL upper limit on the spatial extension (deg).', float)),
    ('ts_ext', (np.nan, 'Test statistic for the extension hypothesis.', float)),

    # Extension vs. Energy
    ('ebin_e_min', (None, '', np.ndarray)),
    ('ebin_e_ctr', (None, '', np.ndarray)),
    ('ebin_e_max', (None, '', np.ndarray)),
    ('ebin_ext', (None, 'Best-fit extension as measured in each energy bin (intrinsic 68% containment radius) (deg).',
                  np.ndarray)),
    ('ebin_ext_err', (None,
                      'Symmetric (1-sigma) error on best-fit extension in each energy bin (deg).',
                      np.ndarray)),
    ('ebin_ext_err_hi', (None,
                         'Upper (1-sigma) error on best-fit extension in each energy bin (deg).',
                         np.ndarray)),
    ('ebin_ext_err_lo', (None,
                         'Lower (1-sigma) error on best-fit extension in each energy bin (deg).',
                         np.ndarray)),
    ('ebin_ext_ul95', (None,
                       '95% CL upper limit on best-fit extension in each energy bin (deg).',
                       np.ndarray)),
    ('ebin_ts_ext', (None,
                     'Test statistic for extension hypothesis in each energy bin.',
                     np.ndarray)),
    ('ebin_dloglike', (None, 'Delta-log-likelihood values for scan over the spatial extension in each energy bin.',
                       np.ndarray)),
    ('ebin_loglike', (None, 'Log-likelihood values for scan over the spatial extension in each energy bin.',
                      np.ndarray)),
    ('ebin_loglike_ptsrc', (None,
                            'Log-Likelihood value of the best-fit point-source model in each energy bin.',
                            np.ndarray)),
    ('ebin_loglike_ext', (None, 'Log-Likelihood value of the best-fit extended source model in each energy bin.',
                          np.ndarray)),

    # Position
    ('ra', localize_output['ra']),
    ('dec', localize_output['dec']),
    ('glon', localize_output['glon']),
    ('glat', localize_output['glat']),
    ('ra_err', localize_output['ra_err']),
    ('dec_err', localize_output['dec_err']),
    ('glon_err', localize_output['glon_err']),
    ('glat_err', localize_output['glat_err']),
    ('pos_offset', localize_output['pos_offset']),
    ('pos_err', localize_output['pos_err']),
    ('pos_r68', localize_output['pos_r68']),
    ('pos_r95', localize_output['pos_r95']),
    ('pos_r99', localize_output['pos_r99']),
    ('pos_err_semimajor', localize_output['pos_err_semimajor']),
    ('pos_err_semiminor', localize_output['pos_err_semiminor']),
    ('pos_angle', localize_output['pos_angle']),

    # Maps
    ('tsmap', (None, '', fermipy.skymap.Map)),
    ('ptsrc_tot_map', (None, '', fermipy.skymap.Map)),
    ('ptsrc_src_map', (None, '', fermipy.skymap.Map)),
    ('ptsrc_bkg_map', (None, '', fermipy.skymap.Map)),
    ('ext_tot_map', (None, '', fermipy.skymap.Map)),
    ('ext_src_map', (None, '', fermipy.skymap.Map)),
    ('ext_bkg_map', (None, '', fermipy.skymap.Map)),

    # Miscellaneous
    ('source_fit', ({}, 'Dictionary with parameters of the best-fit extended source model.', dict)),
))

# Options for plotting
plotting = {
    'loge_bounds': (None, '', list),
    'catalogs': (None, '', list),
    'graticule_radii': (None, 'Define a list of radii at which circular graticules will be drawn.', list),
    'format': ('png', '', str),
    'cmap': ('magma', 'Set the colormap for 2D plots.', str),
    'cmap_resid': ('RdBu_r', 'Set the colormap for 2D residual plots.', str),
    'figsize': ([8.0, 6.0], 'Set the default figure size.', list),
    'label_ts_threshold':
        (0., 'TS threshold for labeling sources in sky maps.  If None then no sources will be labeled.', float),
    'interactive': (False, 'Enable interactive mode.  If True then plots will be drawn after each plotting command.', bool),
}

# Source dictionary


source_meta_output = OrderedDict((
    ('name', (None, 'Name of the source.', str)),
    ('Source_Name', (None, 'Name of the source.', str)),
    ('SpatialModel', (None, 'Spatial model.', str)),
    ('SpatialWidth', (None, 'Spatial size parameter.', float)),
    ('SpatialType', (None, 'Spatial type string.  This corresponds to the type attribute of '
                     'the spatialModel component in the XML model.', str)),
    ('SourceType', (None, 'Source type string (PointSource or DiffuseSource).', str)),
    ('SpectrumType', (None, 'Spectrum type string.  This corresponds to the type attribute of '
                      'the spectrum component in the XML model (e.g. PowerLaw, LogParabola, etc.).', str)),
    ('Spatial_Filename',
     (None, 'Path to spatial template associated to this source.', str)),
    ('Spectrum_Filename', (None,
                           'Path to file associated to the spectral model of this source.', str)),
    ('correlation', ({}, 'Dictionary of correlation coefficients.', dict)),
    ('model_counts', (None, 'Vector of predicted counts for this source in each analysis energy bin.',
                      np.ndarray)),
    ('model_counts_wt', (None, 'Vector of predicted counts for this source in each analysis energy bin.',
                         np.ndarray)),
    ('sed', (None, 'Output of SED analysis.  See :ref:`sed` for more information.', dict)),
))

source_pos_output = OrderedDict((
    ('ra', (np.nan, 'Right ascension of the source (deg).', float)),
    ('dec', (np.nan, 'Declination of the source (deg).', float)),
    ('glon', (np.nan, 'Galactic longitude of the source (deg).', float)),
    ('glat', (np.nan, 'Galactic latitude of the source (deg).', float)),
    ('ra_err', localize_output['ra_err']),
    ('dec_err', localize_output['dec_err']),
    ('glon_err', localize_output['glon_err']),
    ('glat_err', localize_output['glat_err']),
    ('pos_err', localize_output['pos_err']),
    ('pos_r68', localize_output['pos_r68']),
    ('pos_r95', localize_output['pos_r95']),
    ('pos_r99', localize_output['pos_r99']),
    ('pos_err_semimajor', localize_output['pos_err_semimajor']),
    ('pos_err_semiminor', localize_output['pos_err_semiminor']),
    ('pos_angle', localize_output['pos_angle']),
    ('pos_gal_cov', localize_output['pos_gal_cov']),
    ('pos_gal_corr', localize_output['pos_gal_corr']),
    ('pos_cel_cov', localize_output['pos_cel_cov']),
    ('pos_cel_corr', localize_output['pos_cel_corr']),
    ('offset_ra', (np.nan, 'Right ascension offset from ROI center in local celestial projection (deg).', float)),
    ('offset_dec', (np.nan, 'Declination offset from ROI center in local celestial projection (deg).', float)),
    ('offset_glon', (np.nan, 'Galactic longitude offset from ROI center in local galactic projection (deg).', float)),
    ('offset_glat', (np.nan, 'Galactic latitude offset from ROI center in local galactic projection (deg).', float)),
    ('offset_roi_edge', (np.nan, 'Distance from the edge of the ROI (deg).  Negative (positive) values '
                         'indicate locations inside (outside) the ROI.', float)),
    ('offset', (np.nan, 'Angular offset from ROI center (deg).', float)),
))

source_flux_output = OrderedDict((
    ('param_names', (np.zeros(10, dtype='S32'),
                     'Names of spectral parameters.', np.ndarray)),
    ('param_values', (np.empty(10, dtype=float) * np.nan,
                      'Spectral parameter values.', np.ndarray)),
    ('param_errors', (np.empty(10, dtype=float) * np.nan,
                      'Spectral parameters errors.', np.ndarray)),
    ('ts', (np.nan, 'Source test statistic.', float)),
    ('loglike', (np.nan, 'Log-likelihood of the model evaluated at the best-fit normalization of the source.', float)),
    ('loglike_scan', (np.array(
        [np.nan]), 'Log-likelihood values for scan of source normalization.', np.ndarray)),
    ('dloglike_scan', (np.array(
        [np.nan]), 'Delta Log-likelihood values for scan of source normalization.', np.ndarray)),
    ('eflux_scan', (np.array(
        [np.nan]), 'Energy flux values for scan of source normalization.', np.ndarray)),
    ('flux_scan', (np.array(
        [np.nan]), 'Flux values for scan of source normalization.', np.ndarray)),
    ('norm_scan', (np.array(
        [np.nan]), 'Normalization parameters values for scan of source normalization.', np.ndarray)),
    ('npred', (np.nan, 'Number of predicted counts from this source integrated over the analysis energy range.', float)),
    ('npred_wt', (np.nan, 'Number of predicted counts from this source integrated over the analysis energy range.', float)),
    ('pivot_energy', (np.nan, 'Decorrelation energy in MeV.', float)),
    ('flux', (np.nan, 'Photon flux (%s) integrated over analysis energy range' % FLUX_UNIT,
              float)),
    ('flux100', (np.nan, 'Photon flux (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                 float)),
    ('flux1000', (np.nan, 'Photon flux (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                  float)),
    ('flux10000', (np.nan, 'Photon flux (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                   float)),
    ('flux_err', (np.nan, 'Photon flux uncertainty (%s) integrated over analysis energy range' % FLUX_UNIT,
                  float)),
    ('flux100_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                     float)),
    ('flux1000_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                      float)),
    ('flux10000_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                       float)),
    ('flux_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated over analysis energy range' % FLUX_UNIT,
                   float)),
    ('flux100_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                      float)),
    ('flux1000_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                       float)),
    ('flux10000_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                        float)),
    ('eflux', (np.nan, 'Energy flux (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
               float)),
    ('eflux100', (np.nan, 'Energy flux (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                  float)),
    ('eflux1000', (np.nan, 'Energy flux (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                   float)),
    ('eflux10000', (np.nan, 'Energy flux (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                    float)),
    ('eflux_err', (np.nan, 'Energy flux uncertainty (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
                   float)),
    ('eflux100_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                      float)),
    ('eflux1000_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                       float)),
    ('eflux10000_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                        float)),
    ('eflux_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
                    float)),
    ('eflux100_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                       float)),
    ('eflux1000_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                        float)),
    ('eflux10000_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                         float)),
    ('dnde', (np.nan, 'Differential photon flux (%s) evaluated at the pivot energy.' % DIFF_FLUX_UNIT,
              float)),
    ('dnde100', (np.nan, 'Differential photon flux (%s) evaluated at 100 MeV.' % DIFF_FLUX_UNIT,
                 float)),
    ('dnde1000', (np.nan, 'Differential photon flux (%s) evaluated at 1 GeV.' % DIFF_FLUX_UNIT,
                  float)),
    ('dnde10000', (np.nan, 'Differential photon flux (%s) evaluated at 10 GeV.' % DIFF_FLUX_UNIT,
                   float)),
    ('dnde_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at the pivot energy.' % DIFF_FLUX_UNIT,
                  float)),
    ('dnde100_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 100 MeV.' % DIFF_FLUX_UNIT,
                     float)),
    ('dnde1000_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 1 GeV.' % DIFF_FLUX_UNIT,
                      float)),
    ('dnde10000_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 10 GeV.' % DIFF_FLUX_UNIT,
                       float)),
    ('dnde_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at the pivot energy.',
                    float)),
    ('dnde100_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at 100 MeV.',
                       float)),
    ('dnde1000_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated evaluated at 1 GeV.',
                        float)),
    ('dnde10000_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at 10 GeV.',
                         float)),
))

source_output = OrderedDict(list(source_meta_output.items()) +
                            list(source_pos_output.items()) +
                            list(source_flux_output.items()))

# Top-level dictionary for output file
file_output = OrderedDict((
    ('roi', (None, 'A dictionary containing information about the ROI as a whole.', dict)),
    ('sources', (None, 'A dictionary containing information about individual sources in the model (diffuse and point-like).  '
                 'Each element of this dictionary maps to a single source in the ROI model.', dict)),
    ('config', (None, 'The configuration dictionary of the :py:class:`~fermipy.gtanalysis.GTAnalysis` instance.', dict)),
    ('version', (None, 'The version of the Fermipy package that was used to run the analysis.  This is automatically generated from the git release tag.', str))
))
