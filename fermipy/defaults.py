# def build_defaults():
#    path = os.path.join(fermipy.PACKAGE_ROOT,'config','defaults.yaml')
#    with open(path,'r') as f: config = yaml.load(f)

# Options for defining input data files
data = {
    'evfile': (None, 'Input FT1 file.', str),
    'scfile': (None, 'Input FT2 file.', str),
    'ltcube': (None, 'Input LT cube file (optional).', str),
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
    'radius': (None, '', float),
    'filter': (None, '', str),
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
    'min_ts': (None, '', float),
    'min_flux': (None, '', float),
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
    'edisp': (True, '', bool),
    'edisp_disable': (None,
                      'Provide a list of sources for which the edisp '
                      'correction should be disabled.',
                      list),
    'likelihood': ('binned', '', str),
    'minbinsz': (0.05, '', float),
    'rfactor': (2, '', int),
    'convolve': (True, '', bool),
    'resample': (True, '', bool),
    'srcmap': (None, '', str),
    'bexpmap': (None, '', str),
}

# Options for binning.
binning = {
    'projtype': ('WCS', '', str),
    'proj': ('AIT', '', str),
    'coordsys': ('CEL', '', str),
    'npix':
        (None,
         'Number of spatial bins.  If none this will be inferred from roiwidth '
         'and binsz.', int),
    'roiwidth': (10.0,
                 'Set the width of the ROI in degrees.  If both roiwidth and '
                 'binsz are given the roiwidth will be rounded up to be a '
                 'multiple of binsz.',
                 float),
    'binsz': (0.1, 'Set the bin size in degrees.', float),
    'binsperdec': (8, 'Set the number of energy bins per decade.', float),
    'enumbins': (
        None,
        'Number of energy bins.  If none this will be inferred from energy '
        'range and binsperdec parameter.', int),
    'hpx_ordering_scheme': ('RING', 'HEALPix Ordering Scheme', str),
    'hpx_order': (10, 'Order of the map (int between 0 and 12, included)', int),
    'hpx_ebin': (True, 'Include energy binning', bool)
}

# Options related to I/O and output file bookkeeping
fileio = {
    'outdir': (None, 'Set the name of the output directory.', str),
    'scratchdir': ('/scratch', 'Set the path to the scratch directory.', str),
    'workdir': (None, 'Override the working directory.', str),
    'logfile': (None, '', str),
    'savefits': (True, 'Save intermediate FITS data products.', bool),
    'usescratch': (
        False, 'Perform analysis in a temporary working directory.', bool),
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
mc = {}

#
roiopt = {
    'npred_threshold': (1.0, '', float),
    'npred_frac': (0.95, '', float),
    'shape_ts_threshold': (100.0, '', float)
}

#
residmap = {
    'models': (None, '', list),
    'model': (None, '', dict),
    'erange': (None, '', list),
}

#
tsmap = {
    'model': (None, '', dict),
    'multithread': (False, '', bool),
    'max_kernel_radius': (3.0, '', float),
    'erange': (None, '', list),
}

#
tscube = {
    'model': (None, '', dict),
}

#
sourcefind = {
    'model': (None, '', dict),
    'min_separation': (1.0, '', float),
    'sqrt_ts_threshold': (5.0, '', float),
    'max_iter': (3, '', int),
    'sources_per_iter': (3, '', int),
}

# Options for SED analysis
sed = {
    'bin_index': (2.0, '', float),
    'use_local_index': (False, '', bool)
}

# Options for extension analysis
extension = {
    'spatial_model': ('GaussianSource', '', str),
    'width': (None, '', str),
    'width_min': (0.01, '', float),
    'width_max': (1.0, '', float),
    'width_nstep': (21, '', int),
    'save_templates': (False, '', bool),
    'fix_background': (False, '', bool),
    'save_model_map': (False, '', bool),
}

# Options for localization analysis
localize = {
    'nstep': (10, '', int),
    'dtheta_max': (0.05, '', float),
    'update': (False, '', bool)
}

# Options for anlaysis
run = {
    'sed': (None, '', list),
    'extension': (None, '', list)
}

# Options for plotting
plotting = {
    'erange': (None, '', list),
    'format': ('png', '', str)
}
