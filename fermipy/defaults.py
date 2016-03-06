
# Options for defining input data files
data = {
    'evfile': (None, 'Path to FT1 file or list of FT1 files.', str),
    'scfile': (None, 'Path to FT2 (spacecraft) file.', str),
    'ltcube': (None, 'Path to livetime cube.  If none a livetime cube will be generated with ``gtmktime``.', str),
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
    'target': (None, 'Choose an object on which to center the ROI.  '
                     'This option takes precendence over ra/dec.', str),
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
        (None, 'Set the maximum distance for inclusion of sources in the ROI '
               'model.  Selects all sources within a circle of this radius '
               'centered '
               'on the ROI.  If none then no selection is applied.  This '
               'selection '
               'will be ORed with sources passing the cut on src_roiwidth.',
         float),
    'src_roiwidth':
        (None, 'Select sources within a box of RxR centered on the ROI.  If '
               'none then no cut is applied.', float),
    'src_radius_roi':
        (None,
         'Half-width of the ROI selection.  This parameter can be used in '
         'lieu of src_roiwidth.',
         float),
    'isodiff': (None, 'Set the isotropic template.', list),
    'galdiff': (None, 'Set the galactic IEM mapcube.', list),
    'limbdiff': (None, '', list),
    'diffuse': (None, '', list),
    'sources': (None, '', list),
    'extdir': ('Extended_archive_v15', '', str),
    'catalogs': (None, '', list),
#    'min_ts': (None, '', float),
#    'min_flux': (None, '', float),
    'merge_sources' :
        (True, 'Merge properties of sources that appear in multiple '
         'source catalogs.  If merge_sources=false then subsequent sources with '
         'the same name will be ignored.', bool),
    'assoc_xmatch_columns' :
        (['3FGL_Name'],'Choose a set of association columns on which to '
         'cross-match catalogs.',list),
#    'remove_duplicates': (False, 'Remove duplicate catalog sources.', bool),
    'extract_diffuse': (
        False, 'Extract a copy of all mapcube components centered on the ROI.',
        bool)
}

# Options for configuring likelihood analysis
gtlike = {
    'irfs': (None, '', str),
    'edisp': (True, 'Enable the correction for energy dispersion.', bool),
    'edisp_disable': (None,
                      'Provide a list of sources for which the edisp '
                      'correction should be disabled.',
                      list),
#    'likelihood': ('binned', '', str),
    'minbinsz': (0.05, 'Set the minimum bin size used for resampling diffuse maps.', float),
    'rfactor': (2, '', int),
    'convolve': (True, '', bool),
    'resample': (True, '', bool),
    'srcmap': (None, '', str),
    'bexpmap': (None, '', str),
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
    'scratchdir': ('/scratch', 'Path to the scratch directory.', str),
    'workdir': (None, 'Override the working directory.', str),
    'logfile': (None, 'Path to log file.  If None then log will be written to fermipy.log.', str),
    'savefits': (True, 'Save intermediate FITS files.', bool),
    'usescratch': (
        False, 'Run analysis in a temporary directory under ``scratchdir``.', bool),
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
    'tol': (1E-4, 'Set the optimizer tolerance.', float),
    'retries': (3, 'Set the number of times to retry the fit.', int),
    'min_fit_quality': (3, 'Set the minimum fit quality.', int),
    'verbosity': (0, '', int)
}

# MC options
mc = {
    'seed' : (None, '', int)
}

# ROI Optimization
roiopt = {
    'npred_threshold': (1.0, '', float),
    'npred_frac': (0.95, '', float),
    'shape_ts_threshold': (100.0, '', float)
}

# Residual Maps
residmap = {
    'model': (None, '', dict),
    'erange': (None, '', list),
}

# TS Map
tsmap = {
    'model': (None, '', dict),
    'multithread': (False, '', bool),
    'max_kernel_radius': (3.0, '', float),
    'erange': (None, '', list),
}

# TS Cube
tscube = {
    'model': (None, '', dict),
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
}

# Options for Source Finder
sourcefind = {
    'model': (None, 'Set the source model dictionary.', dict),
    'min_separation': (1.0, 'Set the minimum separation in deg for sources added in each iteration.', float),
    'sqrt_ts_threshold': (5.0, 'Set the threshold on sqrt(TS).', float),
    'max_iter': (3, 'Set the number of search iterations.', int),
    'sources_per_iter': (3, '', int),
    'tsmap_fitter': ('tsmap', '', str)
}

# Options for SED analysis
sed = {
    'bin_index': (2.0, '', float),
    'use_local_index': (False, '', bool),
    'fix_background': (True, 'Fix background parameters when fitting the '
                       'source flux in each energy bin.', bool),
    'ul_confidence': (0.95, 'Confidence level for upper limit calculation.',
                      float)
}

# Options for extension analysis
extension = {
    'spatial_model': ('GaussianSource', 'Spatial model use for extension test.', str),
    'width': (None, 'Parameter vector for scan over spatial extent.  If none then the parameter '
              'vector will be set from ``width_min``, ``width_max``, and ``width_nstep``.', str),
    'width_min': (0.01, '', float),
    'width_max': (1.0, '', float),
    'width_nstep': (21, '', int),
    'save_templates': (False, '', bool),
    'fix_background': (False, 'Fix any background parameters that are currently free in the model when '
                       'performing the likelihood scan over extension.', bool),
    'save_model_map': (False, '', bool),
}

# Options for localization analysis
localize = {
    'nstep': (5, '', int),
    'dtheta_max': (0.3, '', float),
    'update': (False, '', bool)
}

# Options for plotting
plotting = {
    'erange': (None, '', list),
    'catalogs': (None, '', list),
    'draw_radii': (None, '', list),
    'format': ('png', '', str),
    'cmap': ('magma', '', str),
    
}
