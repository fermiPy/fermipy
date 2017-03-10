# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
from collections import OrderedDict
import numpy as np


def make_default_dict(d):

    o = {}
    for k, v in d.items():
        o[k] = copy.deepcopy(v[0])

    return o

DIFF_FLUX_UNIT = ':math:`\mathrm{cm}^{-2}~\mathrm{s}^{-1}~\mathrm{MeV}^{-1}`'
FLUX_UNIT = ':math:`\mathrm{cm}^{-2}~\mathrm{s}^{-1}`'
ENERGY_FLUX_UNIT = ':math:`\mathrm{MeV}~\mathrm{cm}^{-2}~\mathrm{s}^{-1}`'

# Options that are common to several sections
common = {
    'model': (None, 'Dictionary defining the spatial/spectral properties of the test source. '
              'If model is None the test source will be a PointSource with an Index 2 power-law spectrum.', dict),
    'free_background': (False, 'Leave background parameters free when performing the fit. If True then any '
                        'parameters that are currently free in the model will be fit simultaneously '
                        'with the source of interest.', bool),
    'free_radius': (None, 'Free normalizations of background sources within this angular distance in degrees '
                    'from the source of interest.  If None then no sources will be freed.', float),
    'make_plots': (False, 'Generate diagnostic plots.', bool),
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
    'use_external_srcmap': (False, 'Use an external precomputed source map file.', bool),
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
    'edm': (None, 'Estimated distance to maximum of log-likelihood function.', float, 'float'),
    'fit_status': (None, 'Optimizer return code (0 = ok).', int, 'int'),
    'fit_quality': (None, 'Fit quality parameter for MINUIT and NEWMINUIT optimizers (3 - Full accurate covariance matrix, '
                    '2 - Full matrix, but forced positive-definite (i.e. not accurate), '
                    '1 - Diagonal approximation only, not accurate, '
                    '0 - Error matrix not calculated at all)', int, 'int'),
    'covariance': (None, 'Covariance matrix between free parameters of the fit.', np.ndarray, '`~numpy.ndarray`'),
    'correlation': (None, 'Correlation matrix between free parameters of the fit.', np.ndarray, '`~numpy.ndarray`'),
    'dloglike': (None, 'Improvement in log-likehood value.', float, 'float'),
    'loglike': (None, 'Post-fit log-likehood value.', float, 'float'),
    'values': (None, 'Vector of best-fit parameter values (unscaled).', np.ndarray, '`~numpy.ndarray`'),
    'errors': (None, 'Vector of parameter errors (unscaled).', np.ndarray, '`~numpy.ndarray`'),
    'config': (None, 'Copy of input configuration to this method.', dict, 'dict'),
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
    'loglike0': (None, 'Pre-optimization log-likelihood value.', float, 'float'),
    'loglike1': (None, 'Post-optimization log-likelihood value.', float, 'float'),
    'dloglike': (None, 'Improvement in log-likehood value.', float, 'float'),
    'config': (None, 'Copy of input configuration to this method.', dict, 'dict'),
}

# Residual Maps
residmap = {
    'model': common['model'],
    'exclude': (None, 'List of sources that will be removed from the model when '
                'computing the residual map.', list),
    'loge_bounds': common['loge_bounds'],
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# TS Map
tsmap = {
    'model': common['model'],
    'exclude': (None, 'List of sources that will be removed from the model when '
                'computing the TS map.', list),
    'multithread': (False, 'Split the calculation across all available cores.', bool),
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
    'multithread': (False, 'Split the calculation across all available cores.', bool),
}

# Options for lightcurve analysis
lightcurve = {
    'use_local_ltcube': (True, '', bool),
    'binsz': (86400.0, 'Set the lightcurve bin size in seconds.', float),
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
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# Output for lightcurve Analysis
lightcurve_output = OrderedDict((
    ('name', (None, 'Name of Source'', ', str, 'str')),
    ('plottimes', (None, 'Center of Time Bin in MJD', np.ndarray, '`~numpy.ndarray`')),
    ('model', (None, 'Best fit model to the source', str, 'str')),
    ('IntFlux', (None, 'Integral Flux in user defined energy range',
                 np.ndarray, '`~numpy.ndarray`')),
    ('IntFluxErr', (None, 'Error on Integral Flux, if 0 this means IntFlux is an Upperlimit',
                    np.ndarray, '`~np.ndarray`')),
    ('Index1', (None, 'Spectral Index', np.ndarray, '`~np.ndarray`')),
    ('Index1Err', (None, 'Error on Spectral Index', np.ndarray, '`~np.ndarray`')),
    ('Index2', (None, 'Spectral Index', np.ndarray, '`~np.ndarray`')),
    ('Index2Err', (None, 'Error on Spectral Index', np.ndarray, '`~np.ndarray`')),
    ('TS', (None, 'Test Statistic', np.ndarray, '`~np.ndarray`')),
    ('retCode', (None, 'Did the likelihood fit converge? 0 if yes, anything else means no',
                 np.ndarray, '`~np.ndarray`')),
    ('npred', (None, 'Number of Predicted photons in time bin from source',
               np.ndarray, '`~np.ndarray`')),
    ('config', ({}, 'Copy of the input configuration to this method.', dict, 'dict')),
))

# Options for SED analysis
sed = {
    'bin_index': (2.0, 'Spectral index that will be use when fitting the energy distribution within an energy bin.', float),
    'use_local_index': (False, 'Use a power-law approximation to the shape of the global spectrum in '
                        'each bin.  If this is false then a constant index set to `bin_index` '
                        'will be used.', bool),
    'free_background': common['free_background'],
    'free_radius': common['free_radius'],
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
                  np.ndarray, '`~numpy.ndarray`')),
    ('loge_max', (None, 'Upper edges of SED energy bins (log10(E/MeV)).',
                  np.ndarray, '`~numpy.ndarray`')),
    ('loge_ctr', (None, 'Centers of SED energy bins (log10(E/MeV)).',
                  np.ndarray, '`~numpy.ndarray`')),
    ('loge_ref', (None, 'Reference energies of SED energy bins (log10(E/MeV)).',
                  np.ndarray, '`~numpy.ndarray`')),
    ('e_min', (None, 'Lower edges of SED energy bins (MeV).',
               np.ndarray, '`~numpy.ndarray`')),
    ('e_max', (None, 'Upper edges of SED energy bins (MeV).',
               np.ndarray, '`~numpy.ndarray`')),
    ('e_ctr', (None, 'Centers of SED energy bins (MeV).', np.ndarray, '`~numpy.ndarray`')),
    ('e_ref', (None, 'Reference energies of SED energy bins (MeV).',
               np.ndarray, '`~numpy.ndarray`')),
    ('ref_flux', (None, 'Flux of the reference model in each bin (%s).' %
                  FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_eflux', (None, 'Energy flux of the reference model in each bin (%s).' %
                   ENERGY_FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_dnde', (None, 'Differential flux of the reference model evaluated at the bin center (%s)' %
                  DIFF_FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_dnde_e_min', (None, 'Differential flux of the reference model evaluated at the lower bin edge (%s)' %
                        DIFF_FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_dnde_e_max', (None, 'Differential flux of the reference model evaluated at the upper bin edge (%s)' %
                        DIFF_FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_e2dnde', (None, 'E^2 x the differential flux of the reference model evaluated at the bin center (%s)' %
                    ENERGY_FLUX_UNIT,  np.ndarray, '`~numpy.ndarray`')),
    ('ref_npred', (None, 'Number of predicted counts in the reference model in each bin.',
                   np.ndarray, '`~numpy.ndarray`')),
    ('norm', (None, 'Normalization in each bin in units of the reference model.',
              np.ndarray, '`~numpy.ndarray`')),
    ('flux', (None, 'Flux in each bin (%s).' %
              FLUX_UNIT, np.ndarray, '`~numpy.ndarray`')),
    ('eflux', (None, 'Energy flux in each bin (%s).' %
               ENERGY_FLUX_UNIT, np.ndarray, '`~numpy.ndarray`')),
    ('dnde', (None, 'Differential flux in each bin (%s).' %
              DIFF_FLUX_UNIT, np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde', (None, 'E^2 x the differential flux in each bin (%s).' %
                ENERGY_FLUX_UNIT, np.ndarray, '`~numpy.ndarray`')),
    ('dnde_err', (None, '1-sigma error on dnde evaluated from likelihood curvature.',
                  np.ndarray, '`~numpy.ndarray`')),
    ('dnde_err_lo', (None, 'Lower 1-sigma error on dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray, '`~numpy.ndarray`')),
    ('dnde_err_hi', (None, 'Upper 1-sigma error on dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray, '`~numpy.ndarray`')),
    ('dnde_ul95', (None, '95% CL upper limit on dnde evaluated from the profile likelihood (MINOS errors).',
                   np.ndarray, '`~numpy.ndarray`')),
    ('dnde_ul', (None, 'Upper limit on dnde evaluated from the profile likelihood using a CL = ``ul_confidence``.',
                 np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde_err', (None, '1-sigma error on e2dnde evaluated from likelihood curvature.',
                    np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde_err_lo', (None, 'Lower 1-sigma error on e2dnde evaluated from the profile likelihood (MINOS errors).',
                       np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde_err_hi', (None, 'Upper 1-sigma error on e2dnde evaluated from the profile likelihood (MINOS errors).',
                       np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde_ul95', (None, '95% CL upper limit on e2dnde evaluated from the profile likelihood (MINOS errors).',
                     np.ndarray, '`~numpy.ndarray`')),
    ('e2dnde_ul', (None, 'Upper limit on e2dnde evaluated from the profile likelihood using a CL = ``ul_confidence``.',
                   np.ndarray, '`~numpy.ndarray`')),
    ('ts', (None, 'Test statistic.', np.ndarray, '`~numpy.ndarray`')),
    ('loglike', (None, 'Log-likelihood of model for the best-fit amplitude.',
                 np.ndarray, '`~numpy.ndarray`')),
    ('npred', (None, 'Number of model counts.', np.ndarray, '`~numpy.ndarray`')),
    ('fit_quality', (None, 'Fit quality parameter for MINUIT and NEWMINUIT optimizers (3 - Full accurate covariance matrix, '
                     '2 - Full matrix, but forced positive-definite (i.e. not accurate), '
                     '1 - Diagonal approximation only, not accurate, '
                     '0 - Error matrix not calculated at all).', np.ndarray, '`~numpy.ndarray`')),
    ('fit_status', (None, 'Fit status parameter (0=ok).', np.ndarray, '`~numpy.ndarray`')),
    ('index', (None, 'Spectral index of the power-law model used to fit this bin.',
               np.ndarray, '`~numpy.ndarray`')),
    ('norm_scan', (None, 'Array of NxM normalization values for the profile likelihood scan in N '
                   'energy bins and M scan points.  A row-wise multiplication with '
                   'any of ``ref`` columns can be used to convert this matrix to the '
                   'respective unit.',
                   np.ndarray, '`~numpy.ndarray`')),
    ('dloglike_scan', (None, 'Array of NxM delta-loglikelihood values for the profile likelihood '
                       'scan in N energy bins and M scan points.', np.ndarray, '`~numpy.ndarray`')),
    ('loglike_scan', (None, 'Array of NxM loglikelihood values for the profile likelihood scan '
                      'in N energy bins and M scan points.', np.ndarray, '`~numpy.ndarray`')),
    ('param_covariance', (None, 'Covariance matrix for the best-fit spectral parameters of the source.',
                          np.ndarray, '`~numpy.ndarray`')),
    ('param_names', (None, 'Array of names for the parameters in the global spectral parameterization of this source.',
                     np.ndarray, '`~numpy.ndarray`')),
    ('param_values', (None, 'Array of parameter values.', np.ndarray, '`~numpy.ndarray`')),
    ('param_errors', (None, 'Array of parameter errors.', np.ndarray, '`~numpy.ndarray`')),
    ('model_flux', (None, 'Dictionary containing the differential flux uncertainty '
                    'band of the best-fit global spectral parameterization for the '
                    'source.', dict, 'dict')),
    ('config', (None, 'Copy of input configuration to this method.', dict, 'dict')),
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
    'free_radius': common['free_radius'],
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
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

extension_output = OrderedDict((
    ('width', (None, 'Vector of width values.', np.ndarray, '`~numpy.ndarray`')),
    ('dloglike', (None, 'Sequence of delta-log-likelihood values for each point in the profile likelihood scan.',
                  np.ndarray, '`~numpy.ndarray`')),
    ('loglike', (None, 'Sequence of likelihood values for each point in the scan over the spatial extension.',
                 np.ndarray, '`~numpy.ndarray`')),
    ('loglike_ptsrc', (np.nan,
                       'Model log-Likelihood value of the best-fit point-source model.', float, 'float')),
    ('loglike_ext', (np.nan, 'Model log-Likelihood value of the best-fit extended source model.', float, 'float')),
    ('loglike_base', (np.nan,
                      'Model log-Likelihood value of the baseline model.', float, 'float')),
    ('ext', (np.nan, 'Best-fit extension in degrees.', float, 'float')),
    ('ext_err_hi', (np.nan, 'Upper (1 sigma) error on the best-fit extension in degrees.', float, 'float')),
    ('ext_err_lo', (np.nan, 'Lower (1 sigma) error on the best-fit extension in degrees.', float, 'float')),
    ('ext_err', (np.nan, 'Symmetric (1 sigma) error on the best-fit extension in degrees.', float, 'float')),
    ('ext_ul95', (np.nan, '95% CL upper limit on the spatial extension in degrees.', float, 'float')),
    ('ts_ext', (np.nan, 'Test statistic for the extension hypothesis.', float, 'float')),
    ('source_fit', ({}, 'Dictionary with parameters of the best-fit extended source model.', dict, 'dict')),
    ('config', ({}, 'Copy of the input configuration to this method.', dict, 'dict')),
))

# Options for localization analysis
localize = {
    'nstep': (5, 'Number of steps in longitude/latitude that will be taken '
              'when refining the source position.  The bounds of the scan '
              'range are set to the 99% positional uncertainty as '
              'determined from the TS map peak fit.  The total number of '
              'sampling points will be nstep**2.', int),
    'dtheta_max': (0.5, 'Half-width of the search region in degrees used for the first pass of the localization search.', float),
    'free_background': common['free_background'],
    'free_radius': common['free_radius'],
    'update': (True, 'Update the source model with the best-fit position.', bool),
    'make_plots': common['make_plots'],
    'write_fits': common['write_fits'],
    'write_npy': common['write_npy'],
}

# Output for localization analysis
localize_output = OrderedDict((
    ('ra', (np.nan, 'Right ascension of best-fit position in deg.', float, 'float')),
    ('dec', (np.nan, 'Declination of best-fit position in deg.', float, 'float')),
    ('glon', (np.nan, 'Galactic Longitude of best-fit position in deg.', float, 'float')),
    ('glat', (np.nan, 'Galactic Latitude of best-fit position in deg.', float, 'float')),
    ('offset', (np.nan, 'Angular offset in deg between the old and new (localized) source positions.', float, 'float')),
    ('sigma', (np.nan, '1-sigma positional uncertainty in deg.', float, 'float')),
    ('r68', (np.nan, '68% positional uncertainty in deg.', float, 'float')),
    ('r95', (np.nan, '95% positional uncertainty in deg.', float, 'float')),
    ('r99', (np.nan, '99% positional uncertainty in deg.', float, 'float')),
    ('sigmax', (np.nan, '1-sigma uncertainty in deg in longitude.', float, 'float')),
    ('sigmay', (np.nan, '1-sigma uncertainty in deg in latitude.', float, 'float')),
    ('sigma_semimajor', (np.nan,
                         '1-sigma uncertainty in deg along major axis of uncertainty ellipse.', float, 'float')),
    ('sigma_semiminor', (np.nan,
                         '1-sigma uncertainty in deg along minor axis of uncertainty ellipse.', float, 'float')),
    ('xpix', (np.nan, 'Longitude pixel coordinate of best-fit position.', float, 'float')),
    ('ypix', (np.nan, 'Latitude pixel coordinate of best-fit position.', float, 'float')),
    ('theta', (np.nan, 'Position angle of uncertainty ellipse.', float, 'float')),
    ('eccentricity', (np.nan,
                      'Eccentricity of uncertainty ellipse defined as sqrt(1-b**2/a**2).', float, 'float')),
    ('eccentricity2', (np.nan,
                       'Eccentricity of uncertainty ellipse defined as sqrt(a**2/b**2-1).', float, 'float')),
    ('config', (None, 'Copy of the input parameters to this method.', dict, 'dict')),
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
}

# Source dictionary


source_meta_output = OrderedDict((
    ('name', (None, 'Name of the source.', str, 'str')),
    ('Source_Name', (None, 'Name of the source.', str, 'str')),
    ('SpatialModel', (None, 'Spatial model.', str, 'str')),
    ('SpatialWidth', (None, 'Spatial size parameter.', float, 'float')),
    ('SpatialType', (None, 'Spatial type string.  This corresponds to the type attribute of the spatialModel component in the XML model.', str, 'str')),
    ('SourceType', (None, 'Source type string (PointSource or DiffuseSource).', str, 'str')),
    ('SpectrumType', (None, 'Spectrum type string.  This corresponds to the type attribute of the spectrum component in the XML model (e.g. PowerLaw, LogParabola, etc.).', str, 'str')),
    ('Spatial_Filename',
     (None, 'Path to spatial template associated to this source.', str, 'str')),
    ('Spectrum_Filename', (None,
                           'Path to file associated to the spectral model of this source.', str, 'str')),
    ('correlation', ({}, 'Dictionary of correlation coefficients.', dict, 'dict')),
    ('model_counts', (None, 'Vector of predicted counts for this source in each analysis energy bin.',
                      np.ndarray, '`~numpy.ndarray`')),
    ('sed', (None, 'Output of SED analysis.  See :ref:`sed` for more information.', dict, 'dict')),
))

source_pos_output = OrderedDict((
    ('ra', (np.nan, 'Right ascension of the source in deg.', float, 'float')),
    ('dec', (np.nan, 'Declination of the source in deg.', float, 'float')),
    ('glon', (np.nan, 'Galactic Longitude of the source in deg.', float, 'float')),
    ('glat', (np.nan, 'Galactic Latitude of the source in deg.', float, 'float')),
    ('offset_ra', (np.nan, 'Angular offset from ROI center along RA.', float, 'float')),
    ('offset_dec', (np.nan, 'Angular offset from ROI center along DEC', float, 'float')),
    ('offset_glon', (np.nan, 'Angular offset from ROI center along GLON.', float, 'float')),
    ('offset_glat', (np.nan, 'Angular offset from ROI center along GLAT.', float, 'float')),
    ('offset_roi_edge', (np.nan, 'Distance from the edge of the ROI in deg.  Negative (positive) values '
                         'indicate locations inside (outside) the ROI.', float, 'float')),
    ('offset', (np.nan, 'Angular offset from ROI center.', float, 'float')),
    ('pos_sigma', (np.nan, '1-sigma uncertainty (deg) on the source position.', float, 'float')),
    ('pos_sigma_semimajor', (np.nan,
                             '1-sigma uncertainty (deg) on the source position along major axis.', float, 'float')),
    ('pos_sigma_semiminor', (np.nan,
                             '1-sigma uncertainty (deg) on the source position along minor axis.', float, 'float')),
    ('pos_angle', (np.nan, 'Position angle (deg) of the positional uncertainty ellipse.', float, 'float')),
    ('pos_r68', (np.nan, '68% uncertainty (deg) on the source position.', float, 'float')),
    ('pos_r95', (np.nan, '95% uncertainty (deg) on the source position.', float, 'float')),
    ('pos_r99', (np.nan, '99% uncertainty (deg) on the source position.', float, 'float')),
))

source_flux_output = OrderedDict((
    ('param_names', (np.zeros(10, dtype='S32'),
                     'Names of spectral parameters.', np.ndarray, '`~numpy.ndarray`')),
    ('param_values', (np.empty(10, dtype=float) * np.nan,
                      'Spectral parameter values.', np.ndarray, '`~numpy.ndarray`')),
    ('param_errors', (np.empty(10, dtype=float) * np.nan,
                      'Spectral parameters errors.', np.ndarray, '`~numpy.ndarray`')),
    ('ts', (np.nan, 'Source test statistic.', float, 'float')),
    ('loglike', (np.nan, 'Log-likelihood of the model evaluated at the best-fit normalization of the source.', float, 'float')),
    ('loglike_scan', (np.array(
        [np.nan]), 'Log-likelihood values for scan of source normalization.', np.ndarray, '`~numpy.ndarray`')),
    ('dloglike_scan', (np.array(
        [np.nan]), 'Delta Log-likelihood values for scan of source normalization.', np.ndarray, '`~numpy.ndarray`')),
    ('eflux_scan', (np.array(
        [np.nan]), 'Energy flux values for scan of source normalization.', np.ndarray, '`~numpy.ndarray`')),
    ('flux_scan', (np.array(
        [np.nan]), 'Flux values for scan of source normalization.', np.ndarray, '`~numpy.ndarray`')),
    ('norm_scan', (np.array(
        [np.nan]), 'Normalization parameters values for scan of source normalization.', np.ndarray, '`~numpy.ndarray`')),
    ('npred', (np.nan, 'Number of predicted counts from this source integrated over the analysis energy range.', float, 'float')),
    ('pivot_energy', (np.nan, 'Decorrelation energy in MeV.', float, 'float')),
    ('flux', (np.nan, 'Photon flux (%s) integrated over analysis energy range' % FLUX_UNIT,
              float, 'float')),
    ('flux100', (np.nan, 'Photon flux (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                 float, 'float')),
    ('flux1000', (np.nan, 'Photon flux (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                  float, 'float')),
    ('flux10000', (np.nan, 'Photon flux (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                   float, 'float')),
    ('flux_err', (np.nan, 'Photon flux uncertainty (%s) integrated over analysis energy range' % FLUX_UNIT,
                  float, 'float')),
    ('flux100_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                     float, 'float')),
    ('flux1000_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                      float, 'float')),
    ('flux10000_err', (np.nan, 'Photon flux uncertainty (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                       float, 'float')),
    ('flux_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated over analysis energy range' % FLUX_UNIT,
                   float, 'float')),
    ('flux100_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 100 MeV to 316 GeV.' % FLUX_UNIT,
                      float, 'float')),
    ('flux1000_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 1 GeV to 316 GeV.' % FLUX_UNIT,
                       float, 'float')),
    ('flux10000_ul95', (np.nan, '95%' + ' CL upper limit on the photon flux (%s) integrated from 10 GeV to 316 GeV.' % FLUX_UNIT,
                        float, 'float')),
    ('eflux', (np.nan, 'Energy flux (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
               float, 'float')),
    ('eflux100', (np.nan, 'Energy flux (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                  float, 'float')),
    ('eflux1000', (np.nan, 'Energy flux (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                   float, 'float')),
    ('eflux10000', (np.nan, 'Energy flux (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                    float, 'float')),
    ('eflux_err', (np.nan, 'Energy flux uncertainty (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
                   float, 'float')),
    ('eflux100_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                      float, 'float')),
    ('eflux1000_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                       float, 'float')),
    ('eflux10000_err', (np.nan, 'Energy flux uncertainty (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                        float, 'float')),
    ('eflux_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated over analysis energy range' % ENERGY_FLUX_UNIT,
                    float, 'float')),
    ('eflux100_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 100 MeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                       float, 'float')),
    ('eflux1000_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 1 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                        float, 'float')),
    ('eflux10000_ul95', (np.nan, '95%' + ' CL upper limit on the energy flux (%s) integrated from 10 GeV to 316 GeV.' % ENERGY_FLUX_UNIT,
                         float, 'float')),
    ('dnde', (np.nan, 'Differential photon flux (%s) evaluated at the pivot energy.' % DIFF_FLUX_UNIT,
              float, 'float')),
    ('dnde100', (np.nan, 'Differential photon flux (%s) evaluated at 100 MeV.' % DIFF_FLUX_UNIT,
                 float, 'float')),
    ('dnde1000', (np.nan, 'Differential photon flux (%s) evaluated at 1 GeV.' % DIFF_FLUX_UNIT,
                  float, 'float')),
    ('dnde10000', (np.nan, 'Differential photon flux (%s) evaluated at 10 GeV.' % DIFF_FLUX_UNIT,
                   float, 'float')),
    ('dnde_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at the pivot energy.' % DIFF_FLUX_UNIT,
                  float, 'float')),
    ('dnde100_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 100 MeV.' % DIFF_FLUX_UNIT,
                     float, 'float')),
    ('dnde1000_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 1 GeV.' % DIFF_FLUX_UNIT,
                      float, 'float')),
    ('dnde10000_err', (np.nan, 'Differential photon flux uncertainty (%s) evaluated at 10 GeV.' % DIFF_FLUX_UNIT,
                       float, 'float')),
    ('dnde_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at the pivot energy.',
                    float, 'float')),
    ('dnde100_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at 100 MeV.',
                       float, 'float')),
    ('dnde1000_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated evaluated at 1 GeV.',
                        float, 'float')),
    ('dnde10000_index', (np.nan, 'Logarithmic slope of the differential photon spectrum evaluated at 10 GeV.',
                         float, 'float')),
))

source_output = OrderedDict(list(source_meta_output.items()) +
                            list(source_pos_output.items()) +
                            list(source_flux_output.items()))

# Top-level dictionary for output file
file_output = OrderedDict((
    ('roi', (None, 'A dictionary containing information about the ROI as a whole.', dict, 'dict')),
    ('sources', (None, 'A dictionary containing information for individual sources in the model (diffuse and point-like).  Each element of this dictionary maps to a single source in the ROI model.', dict, 'dict')),
    ('config', (None, 'The configuration dictionary of the :py:class:`~fermipy.gtanalysis.GTAnalysis` instance.', dict, 'dict')),
    ('version', (None, 'The version of the fermiPy package that was used to run the analysis.  This is automatically generated from the git release tag.', str, 'str'))
))
