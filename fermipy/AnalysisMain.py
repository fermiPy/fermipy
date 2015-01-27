
import pprint
from utils import *
from defaults import *
from Logger import Logger
from Logger import logLevel as ll






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
    
