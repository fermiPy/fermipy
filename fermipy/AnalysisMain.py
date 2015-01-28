
import pprint
import os
import sys
import copy
from utils import *
from defaults import *
from roi_manager import *
from Logger import Logger
from Logger import logLevel as ll

# pylikelihood
from GtApp import GtApp
from BinnedAnalysis import BinnedObs,BinnedAnalysis
from UnbinnedAnalysis import UnbinnedObs, UnbinnedAnalysis
from pyLikelihood import ParameterVector
#from Composite2 import *
from SummedLikelihood import SummedLikelihood

def filter_dict(d,val):
    for k, v in d.items():
        if v == val: del d[k]

        

class GTAnalysis(AnalysisBase):
    """High-level analysis interface that internally manages a set of
    analysis component objects.  Most of the interactive functionality
    of the fermiPy package is provided through the methods of this class."""

    defaults = {'common' :
                    dict(GTAnalysisDefaults.defaults_selection.items() +
                         GTAnalysisDefaults.defaults_fileio.items() +
                         GTAnalysisDefaults.defaults_binning.items() +
                         GTAnalysisDefaults.defaults_irfs.items() +
                         GTAnalysisDefaults.defaults_inputs.items()),
                'roi' : GTAnalysisDefaults.defaults_roi,
                'verbosity' : (0,'')}

    def __init__(self,config,**kwargs):
        super(GTAnalysis,self).__init__(config,**kwargs)

        rootdir = os.getcwd()
        
                
        # Destination directory for output data products

        if self.config['common']['base'] is not None:
#            self._savedir = os.path.abspath(config['common']['savedir'])
#        elif config['common']['name'] is not None:
            self._savedir = os.path.join(rootdir,
                                         self.config['common']['base'])
            mkdir(self._savedir)
        else:
            raise Exception('Save directory not defined.')
            
        # Working directory (can be the same as savedir)
        if self.config['common']['scratchdir'] is not None:
            self._workdir = mkdtemp(prefix=os.environ['USER'] + '.',
                                    dir=self.config['common']['scratchdir'])
        else:
            self._workdir = self._savedir

        self.logger = Logger(os.path.join(self._savedir,
                                          self.config['common']['base']),
                             self.__class__.__name__,
                             ll(self.config['verbosity'])).get()

        # Setup the ROI definition
#        self._roi_mgr = ROIManager(config['roi'])
#        self._roi_mgr.load()
#        pprint.pprint(self._roi_mgr.config)

        self._roi = ROIManager.create_roi_from_source(self.config['common']['target'],
                                                      self.config['roi'])

        for s in self._roi:
            print s.name
        
        self._components = []
        for i,k in enumerate(sorted(config['components'].keys())):

            cfg = self.config['common']
            cfg['roi'] = self.config['roi']
            update_dict(cfg,config['components'][k])

            roi = copy.deepcopy(self._roi)
            roi.configure(cfg['roi'])
            roi.load_diffuse_srcs()
            
            self.logger.info("Creating Analysis Component: " + k)
            comp = GTBinnedAnalysis(cfg,roi,
                                    logger=self.logger,
                                    file_suffix='_' + k,
                                    savedir=self._savedir,
                                    workdir=self._workdir)

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

class GTBinnedAnalysis(AnalysisBase):

    defaults = dict(GTAnalysisDefaults.defaults_selection.items()+
                    GTAnalysisDefaults.defaults_binning.items()+
                    GTAnalysisDefaults.defaults_irfs.items()+
                    GTAnalysisDefaults.defaults_inputs.items()+
                    GTAnalysisDefaults.defaults_fileio.items(),
                    roi=GTAnalysisDefaults.defaults_roi,
                    file_suffix=('',''))


    def __init__(self,config,roi,logger=None,**kwargs):
        super(GTBinnedAnalysis,self).__init__(config,**kwargs)

        pprint.pprint(self.config)

        if logger is not None:
            self._logger = logger
        
        savedir = self.config['savedir']
        self._roi = roi
        
        from os.path import join

        self._ft1_file=join(savedir,
                            'ft1%s.fits'%self.config['file_suffix'])        
        self._ccube_file=join(savedir,
                             'ccube%s.fits'%self.config['file_suffix'])
        self._mcube_file=join(savedir,
                              'mcube%s.fits'%self.config['file_suffix'])
        self._srcmap_file=join(savedir,
                               'srcmap%s.fits'%self.config['file_suffix'])
        self._bexpmap_file=join(savedir,
                                'bexpmap%s.fits'%self.config['file_suffix'])
        self._srcmdl_file=join(savedir,
                               'srcmdl%s.xml'%self.config['file_suffix'])

        self.enumbins = np.round(self.config['binsperdec']*np.log10(self.config['emax']/self.config['emin']))
        self.enumbins = int(self.enumbins)

        if self.config['npix'] is None:
            self.npix = int(np.round(self.config['roi_width']/self.config['binsz']))
        else:
            self.npix = self.config['npix']
            
    @property
    def roi(self):
        return self._roi
        
    def setup(self):
        """Run pre-processing step."""

        # Write ROI XML
        self._roi.write_xml(self._srcmdl_file)
        roi_center = self._roi.radec
        
        # Run gtselect
        kw = dict(infile=self.config['evfile'],
                  outfile=self._ft1_file,
                  ra=roi_center[0], dec=roi_center[1],
                  rad=self.config['radius'],
                  tmin=self.config['tmin'], tmax=self.config['tmax'],
                  emin=self.config['emin'], emax=self.config['emax'],
                  zmax=self.config['zmax'])
#                  chatter=self.config['chatter'])

        filter_dict(kw,None)
        pprint.pprint(kw)

        if not os.path.isfile(self._ft1_file):
            gtselect=GtApp('gtselect','gtselect')
            gtselect.run(**kw)
        else:
            self._logger.info('Skipping gtselect')
        
        # Run gtmktime

        # Run gtltcube
        if self.config['ltcube'] is not None:
            self._ltcube = self.config['ltcube']
            
        # Run gtbin
        kw = dict(algorithm='ccube',
                  nxpix=self.npix, nypix=self.npix,
                  binsz=self.config['binsz'],
                  evfile=self._ft1_file,
                  outfile=self._ccube_file,
                  scfile=self.config['scfile'],
                  xref=self.roi.radec[0], yref=self.roi.radec[1], axisrot=0,
                  proj=self.config['proj'],
                  ebinalg='LOG', emin=self.config['emin'], emax=self.config['emax'],
                  enumbins=self.enumbins,
                  coordsys=self.config['coordsys'])
#                  chatter=self.config['chatter']

        filter_dict(kw,None)
        pprint.pprint(kw)
        
        if not os.path.isfile(self._ccube_file):
            gtbin=GtApp('gtbin','gtbin')
            gtbin.run(**kw)
        else:
            self._logger.info('Skipping gtbin')

        # Run gtexpcube2
        kw = dict(infile=self._ltcube,
                  cmap=self._ccube_file,
                  ebinalg='LOG',
                  emin=self.config['emin'], emax=self.config['emax'],
                  enumbins=self.enumbins,
                  outfile=self._bexpmap_file, proj='CAR',
                  nxpix=360, nypix=180, binsz=1,
                  irfs=self.config['irfs'],
                  coordsys=self.config['coordsys'])
#                  chatter=self.config['chatter'])
        
        filter_dict(kw,None)
        pprint.pprint(kw)

        if not os.path.isfile(self._bexpmap_file):
            gtexpcube=GtApp('gtexpcube2','gtexpcube2')
            gtexpcube.run(**kw)
        else:
            print 'Skipping gtexpcube'

        # Run gtsrcmaps
        kw = dict(scfile=self.config['scfile'],
                  expcube=self._ltcube,
                  cmap=self._ccube_file,
                  srcmdl=self._srcmdl_file,
                  bexpmap=self._bexpmap_file,
                  outfile=self._srcmap_file,
                  irfs=self.config['irfs'],
#                   rfactor=self.config['rfactor'],
#                   resample=self.config['resample'],
#                   minbinsz=self.config['minbinsz'],
#                   chatter=self.config['chatter'],
                  emapbnds='no' ) 

        if not os.path.isfile(self._srcmap_file):
            gtsrcmaps=GtApp('gtsrcmaps','gtsrcmaps')
            gtsrcmaps.run(**kw)
        else:
            print 'Skipping gtsrcmaps'

        # Create BinnedObs
        print 'Creating BinnedObs'
        obs=BinnedObs(srcMaps=self._srcmap_file,expCube=self._ltcube,
                      binnedExpMap=self._bexpmap_file,irfs=self.config['irfs'])

        # Create BinnedAnalysis
        print 'Creating BinnedAnalysis'
        self.like = BinnedAnalysis(binnedData=obs,srcModel=self._srcmdl_file,
                                   optimizer='MINUIT')#self.config['optimizer'])

        if self.config['enable_edisp']:
            print 'Enabling energy dispersion'
            self.like.logLike.set_edisp_flag(True)
#            os.environ['USE_BL_EDISP'] = 'true'

            




    
