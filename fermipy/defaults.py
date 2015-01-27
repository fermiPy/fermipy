class GTAnalysisDefaults(object):

    defaults_selection = {'emin'    : (None,'Minimum Energy'),
                          'emax'    : (None,'Maximum Energy'),
                          'zmax'    : (None,'Maximum zenith angle.'),
                          'evclass' : (None,'Event class selection.'),
                          'evtype'  : (None,'Event type selection.'),
                          }

    defaults_roi = {'isodiff'       : (None,''),
                    'galdiff'       : (None,''),
                    'limbdiff'      : (None,''),
                    'catalogs'      : (None,'',None,list) }

    defaults_binning = {'binsz'      : (0.1,''),
                        'binsperdec' : (8,'')}
