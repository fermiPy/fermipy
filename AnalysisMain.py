
import pprint
from utils import *
from Logger import Logger
from Logger import logLevel as ll


class AnalysisBase(object):
    """The base class provides common facilities like configuration
    parsing and saving state. """
    def __init__(self,config,**kwargs):
        self._config = self.get_config()
        update_dict(self._config,config)
        update_dict(self._config,kwargs)

    @classmethod
    def get_config(cls):
        # Load defaults
        return load_config(cls.defaults)

    @property
    def config(self):
        """Return the internal configuration state of this class."""
        return self._config

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

class GTAnalysis(AnalysisBase):
    """High-level analysis interface that internally manages a set of
    analysis component objects.  Most of the interactive functionality
    of the fermiPy package is provided through the methods of this class."""

    defaults = {'common' : GTAnalysisDefaults.defaults_selection,
                'verbosity' : (0,'')}

    def __init__(self,config,**kwargs):
        super(GTAnalysis,self).__init__(config,**kwargs)

        # Destination directory for output data products
        self._savedir = None

        # Working directory (can be the same as savedir)
        self._workdir = None

        self.logger = Logger('test', self.__class__.__name__,
                             ll(self.config['verbosity'])).get()

        # Setup the ROI definition
        self._roi = ROIManager(config['roi'])
        pprint.pprint(self._roi.config)

        self._components = []
        for k,v in config['components'].iteritems():

            cfg = self.config['common']
            update_dict(cfg,v)

            self.logger.info("Creating Analysis Component: " + k)
            comp = GTAnalysisComponent(cfg)
            self._components.append(comp)

        # Instantiate pyLikelihood objects here and tie common parameters

    def create_components(self,analysis_type):
        """Auto-generate a set of components given an analysis type flag."""
        # Lookup a pregenerated config file for the desired analysis setup
        pass

    def setup(self):
        """Run pre-processing step for each analysis component.  This
        will run everything except the likelihood optimization: data
        selection (gtselect, gtmktime), counts maps generation
        (gtbin), model generation (gtexpcube2,gtsrcmaps,gtdiffrsp)."""
        for i, c in enumerate(self._components):

            self.logger.info("Performing setup for Analysis Component %i"%i)
            c.setup()

    def free_source(self):
        """Free/Fix all parameters of a source."""
        pass

    def free_norm(self):
        """Free/Fix normalization of a source."""
        pass

    def free_index(self):
        """Free/Fix index of a source."""
        pass

    def fit(self):
        """Run likelihood optimization."""
        pass

    def load_xml(self):
        """Load model definition from XML."""
        pass

    def write_xml(self):
        """Save current model definition as XML file."""
        pass

    def write_results(self):
        """Write out parameters of current model as yaml file."""
        pass

class GTAnalysisComponent(AnalysisBase):

    defaults = dict(GTAnalysisDefaults.defaults_selection.items()+
                    GTAnalysisDefaults.defaults_roi.items())


    def __init__(self,config,**kwargs):
        super(GTAnalysisComponent,self).__init__(config,**kwargs)



    def setup(self):
        """Run pre-processing step."""

        # Run gtselect

        # Run gtmktime

        # Run gtltcube

        # Create BinnedObs

        # Create BinnedAnalysis

        pass
    
class ROIManager(AnalysisBase):
    """This class is responsible for reading and writing XML model
    files."""

    defaults = dict(GTAnalysisDefaults.defaults_roi.items())

    def __init__(self,config,**kwargs):
        super(ROIManager,self).__init__(config,**kwargs)

        for c in self.config['catalogs']:
            self.load_catalog(c)

    def load_catalog(self,catalog_file):
        """Load sources from a FITS catalog file."""
        pass

    def load_xml(self):
        """Load sources from an XML file."""

    def write_xml(self):
        """Save current model definition as XML file."""
        pass
