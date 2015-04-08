

# Options for defining input data files
inputs = {
    'evfile'    : (None,'Input FT1 file.'),
    'scfile'    : (None,'Input FT2 file.'),
    'ltcube'    : (None,'Input LT cube file (optional).'),
    }

# Options for data selection.
selection = {
    'emin'    : (None,'Minimum Energy'),
    'emax'    : (None,'Maximum Energy'),
    'tmin'    : (None,'Minimum time (MET).'),
    'tmax'    : (None,'Maximum time (MET).'),
    'zmax'    : (None,'Maximum zenith angle.'),
    'evclass' : (None,'Event class selection.'),
    'evtype'  : (None,'Event type selection.'),
    'convtype': (None,'Conversion type selection.'),
    'target'  : (None,'Choose a target object at which the ROI will be '
                 'centered.  This option takes precendence over ra/dec.'),
    'ra'      : (None,''),
    'dec'     : (None,''),
    'glat'    : (None,''),
    'glon'    : (None,''),
    'radius'  : (None,''),
    'filter'  : ('DATA_QUAL>0 && LAT_CONFIG==1',''),
    }

# Options for ROI model.
roi = {
    'radius'        :
        (None,'Set the maximum distance for inclusion of sources in the ROI model.  '
         'Selects all sources within a circle of this radius centered on the ROI.  '
         'If none then no selection is applied.  This selection will be ORed with '
         'sources passing the cut on roisize.'),
    'roisize'        :
        (None,'Select sources within a box of RxR centered on the ROI.  If none then no '
         'cut is applied.'),
    
    'isodiff'       : (None,''),
    'galdiff'       : (None,''),
    'limbdiff'      : (None,''),
    'extdir'        : ('Extended_archive_v14',''),
    'catalogs'      : (None,'',None,list)
    }

irfs = {
    'irfs'          : (None,''),
    'enable_edisp'  : (False,'')
    }

# Options for binning.
binning = {
    'proj'       : ('AIT',''),
    'coordsys'   : ('CEL',''),        
    'npix'       : (None,'Number of spatial bins.  If none this will be inferred from roi_width and binsz.'),        
    'roi_width'  : (10.0,'Set the width of the ROI in degrees.'),
    'binsz'      : (0.1,'Set the bin size in degrees.'),
    'binsperdec' : (8,'Set the number of energy bins per decade.'),
    'enumbins'   : (None,'Number of energy bins.  If none this will be inferred from energy range and binsperdec.')
    }

# Options related to I/O and output file bookkeeping
fileio = {
    'base'        : (None,'Set the name of the output'),
    'scratchdir'  : (None,''),
    'savedir'     : (None,'Override the output directory.'),
    'workdir'     : (None,'Override the working directory.'),
    'logfile'     : (None,''),
    'saveoutput'  : (True,'Save intermediate FITS data products.')
    }

# Options related to likelihood optimizer
optimizer = {
    'optimizer'   : ('MINUIT','Set the optimizer name.'),
    'tol'         : (1E-4,'Set the optimizer tolerance.'),
    'retries'     : (3,'Number of times to retry the fit.')
    }
