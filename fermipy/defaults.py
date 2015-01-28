class GTAnalysisDefaults(object):

    # Options for defining input data files
    defaults_inputs = {
        'evfile'    : (None,'Input FT1 file.'),
        'scfile'    : (None,'Input FT2 file.'),
        'ltcube'    : (None,'Input LT cube file (optional).'),
        }

    # Options for data selection.
    defaults_selection = {
        'emin'    : (None,'Minimum Energy'),
        'emax'    : (None,'Maximum Energy'),
        'tmin'    : (None,'Minimum time (MET).'),
        'tmax'    : (None,'Maximum time (MET).'),
        'zmax'    : (None,'Maximum zenith angle.'),
        'evclass' : (None,'Event class selection.'),
        'evtype'  : (None,'Event type selection.'),
        'target'  : (None,'Choose a target object at which the ROI will be '
                     'centered.  This option takes precendence over ra/dec.'),
        'ra'      : (None,''),
        'dec'     : (None,''),
        'radius'  : (None,''),
        }

    # Options for ROI model.
    defaults_roi = {
        'radius'        : (None,'Set the radius of the ROI.  This determines what sources '
                           'will be included in the ROI model.  If none then no cut is applied.'),
        'isodiff'       : (None,''),
        'galdiff'       : (None,''),
        'limbdiff'      : (None,''),
        'catalogs'      : (None,'',None,list)
        }

    defaults_irfs = {
        'irfs'          : (None,''),
        'enable_edisp'  : (False,'')
        }
    
    # Options for binning.
    defaults_binning = {
        'proj'       : ('AIT',''),
        'coordsys'   : ('GAL',''),        
        'npix'       : (None,'Number of spatial bins.  If none this will be inferred from roi_width and binsz.'),        
        'roi_width'  : (10.0,'Set the width of the ROI in degrees.'),
        'binsz'      : (0.1,'Set the bin size in degrees.'),
        'binsperdec' : (8,'Set the number of energy bins per decade.'),
        'enumbins'   : (None,'Number of energy bins.  If none this will be inferred from energy range and binsperdec.')
        }

    # Options related to I/O and output file bookkeeping
    defaults_fileio = {
        'base'        : (None,'Set the name of the output'),
        'scratchdir'  : (None,''),
        'savedir'     : (None,'Override the output directory.'),
        'workdir'     : (None,'Override the working directory.'),
        'saveoutput'  : (True,'Save intermediate FITS data products.')
        }
    
