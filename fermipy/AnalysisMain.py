
import pprint
import os
import sys
import copy
import yaml
from utils import *
import defaults
import fermipy
from roi_manager import *
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll
import logging

# pylikelihood
import pyLikelihood as pyLike
import GtApp
import BinnedAnalysis as ba
import UnbinnedAnalysis as uba
import Composite2
import SummedLikelihood
import FluxDensity

from LikelihoodState import LikelihoodState
from UpperLimits import UpperLimits

def run_gtapp(appname,logger,kw):

    logger.info('Running %s'%appname)
#    logger.debug('\n' + yaml.dump(kw))
    filter_dict(kw,None)
    gtapp=GtApp.GtApp(appname)

    for k,v in kw.items(): gtapp[k] = v
    logger.info(gtapp.command())
    stdin, stdout = gtapp.runWithOutput(print_command=False)

    for line in stdout:
        logger.info(line.strip())

    # Capture return code?

def filter_dict(d,val):
    for k, v in d.items():
        if v == val: del d[k]

def gtlike_spectrum_to_dict(spectrum):
    """ Convert a pyLikelihood object to a python 
        dictionary which can be easily saved to a file. """
    parameters=pyLike.ParameterVector()
    spectrum.getParams(parameters)
    d = dict(spectrum_type = spectrum.genericName())
    for p in parameters:

        pname = p.getName()
        d[pname]= p.getTrueValue()
        d['%s_err' % pname]= p.error()*p.getScale() if p.isFree() else np.nan
        if d['spectrum_type'] == 'FileFunction': 
            ff=pyLike.FileFunction_cast(spectrum)
            d['file']=ff.filename()
    return d
        

class GTAnalysis(AnalysisBase):
    """High-level analysis interface that internally manages a set of
    analysis component objects.  Most of the interactive functionality
    of the fermiPy package is provided through the methods of this class."""

    defaults = {'common' :
                    dict(defaults.selection.items() +
                         defaults.fileio.items() +
                         defaults.binning.items() +
                         defaults.irfs.items() +
                         defaults.optimizer.items() +
                         defaults.inputs.items(),
                         roi=defaults.roi),
                'verbosity' : (0,''),
                'components' : (None,'')}

    def __init__(self,config,**kwargs):
        super(GTAnalysis,self).__init__(config,**kwargs)

        # Setup directories
        self._rootdir = os.getcwd()
                        
        # Destination directory for output data products
        if self.config['common']['base'] is not None:
#            self._savedir = os.path.abspath(config['common']['savedir'])
#        elif config['common']['name'] is not None:
            self._savedir = os.path.join(self._rootdir,
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


        # put pfiles into savedir
        os.environ['PFILES']= \
            self._savedir+';'+os.environ['PFILES'].split(';')[-1]

        logfile = os.path.join(self._savedir,'fermipy')

        self.logger = Logger.get(self.__class__.__name__,logfile,
                                 ll(self.config['verbosity']))

        self.logger.info('\n' + '-'*80 + '\n' + "This is fermipy version {}.".
                         format(fermipy.__version__))
        self.print_config(self.logger)
        
        # Setup the ROI definition
        self._roi = \
            ROIManager.create_roi_from_source(self.config['common']['target'],
                                              self.config['common']['roi'])


        self._like = SummedLikelihood.SummedLikelihood()
        self._components = []
        configs = self.create_component_configs()

        for cfg in configs:
            comp = self._create_component(cfg,logfile)
            self._components.append(comp)

        energies = np.zeros(0)
        for c in self.components:
            energies = np.concatenate((energies,c.energies))
            
        self._ebin_edges = np.sort(np.unique(energies.round(5)))
        self._enumbins = len(self._ebin_edges)-1
        self._roi_model = {}
            
    @property
    def like(self):
        """Return the global likelihood object."""
        return self._like

    @property
    def components(self):
        """Return the list of analysis components."""
        return self._components

    @property
    def energies(self):
        return self._ebin_edges

    @property
    def enumbins(self):
        return self._enumbins    
    
    def create_component_configs(self):
        configs = []

        components = self.config['components']
        
        if components is None:
            cfg = copy.copy(self.config['common'])
            cfg['file_suffix'] = '_00'
            cfg['name'] = '00'      
            configs.append(self.config['common'])
        elif isinstance(components,dict):            
            for i,k in enumerate(sorted(components.keys())):
                cfg = copy.copy(self.config['common'])                
                cfg = merge_dict(cfg,components[k])
                cfg['file_suffix'] = '_' + k
                cfg['name'] = k
                configs.append(cfg)
        elif isinstance(components,list):
            for i,c in enumerate(components):
                cfg = copy.copy(self.config['common'])                
                cfg = merge_dict(cfg,c)
                cfg['file_suffix'] = '_%02i'%i
                cfg['name'] = '%02i'%i
                configs.append(cfg)
        else:
            raise Exception('Invalid type for component block.')

        return configs
                
    def init_components(self):
        self._components = []        
    
    def _create_component(self,cfg,logfile):
        roi = copy.deepcopy(self._roi)
        roi.configure(cfg['roi'])
        roi.load_diffuse_srcs()
            
        self.logger.info("Creating Analysis Component: " + cfg['name'])
        comp = GTBinnedAnalysis(cfg,roi,
                                logfile=logfile,
                                savedir=self._savedir,
                                workdir=self._workdir,
                                verbosity=self.config['verbosity'])

        return comp

    def setup(self):
        """Run pre-processing step for each analysis component.  This
        will run everything except the likelihood optimization: data
        selection (gtselect, gtmktime), counts maps generation
        (gtbin), model generation (gtexpcube2,gtsrcmaps,gtdiffrsp)."""

        # Run data selection step

        self._like = SummedLikelihood.SummedLikelihood()
        for i, c in enumerate(self._components):

            self.logger.info("Performing setup for Analysis Component: " +
                             c.name)
            c.setup()
            self._like.addComponent(c.like)

        for name in self.like.sourceNames():
            self._roi_model[name] = {'sed' : None}
            
            
    def generate_model(self,model_name=None):
        """Generate model maps for all components.  model_name should
        be a unique identifier for the model.  If model_name is None
        then the model maps will be generated using the current
        parameters of the ROI."""

        for i, c in enumerate(self._components):
            c.generate_model(model_name=model_name)

        # If all model maps have the same spatial/energy binning we
        # could generate a co-added model map here

    def setEnergyRange(self,emin,emax):
        """Set the energy range of the analysis."""
        for c in self.components:
            c.setEnergyRange(emin,emax)
            
    def modelCountsSpectrum(self,name,emin,emax):
        """Return the predicted number of model counts versus energy
        for a given source and energy range."""

        cs = []
        for c in self.components: 
            cs += [c.modelCountsSpectrum(name,emin,emax)]
        return cs
            
    def free_sources(self,free=True,pars=None,radius=None):
        """Free all sources within a certain radius of the given sky
        coordinate.

        Parameters
        ----------

        free : bool        
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list        
            Set a list of parameters to be freed/fixed for this source.  If
            none then all source parameters will be freed/fixed with the
            exception of those defined in the skip_pars list.

        radius : float        
            Distance out to which sources should be freed or fixed.
            If none then all sources will be selected.
        
        """

        rsrc, srcs = self._roi.get_sources_by_position(self._roi.radec[0],
                                                       self._roi.radec[1],
                                                       radius)
        
        for r, s in zip(rsrc,srcs):
            self.free_source(s.name,free=free,pars=pars)

        for s in self._roi._diffuse_srcs:
            self.free_source(s.name,free=free,pars=pars)
            
    def free_source(self,name,free=True,pars=None,skip_pars=['Scale']):
        """Free/Fix parameters of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool        
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list        
            Set a list of parameters to be freed/fixed for this source.  If
            none then all source parameters will be freed/fixed with the
            exception of those defined in the skip_pars list.
            
        """

        # Find the source
        if not name in ['isodiff','galdiff','limbdiff']:
            name = self._roi.get_source_by_name(name).name
            
        # Deduce here the names of all parameters from the spectral type
        src_par_names = pyLike.StringVector()
        self.like[name].src.spectrum().getParamNames(src_par_names)

        par_indices = []
        par_names = []
        for p in src_par_names:
            if p in skip_pars: continue
            if pars is not None and not p in pars: continue
            par_indices.append(self.like.par_index(name,p))
            par_names.append(p)
            
        for (idx,par_name) in zip(par_indices,par_names):

            if free:
                self.logger.debug('Freeing parameter %s for source %s'
                                  %(par_name,name))
            else:
                self.logger.debug('Fixing parameter %s for source %s'
                                  %(par_name,name))
                
            self.like[idx].setFree(free)
        self.like.syncSrcParams(name)
                
#        freePars = self.like.freePars(name)
#        normPar = self.like.normPar(name).getName()
#        idx = self.like.par_index(name, normPar)
        
#        if not free:
#            self.like.setFreeFlag(name, freePars, False)
#        else:
#            self.like[idx].setFree(True)

        
    def free_norm(self,name,free=True):
        """Free/Fix normalization of a source.

        Parameters
        ----------

        name : str
            Source name.

        free : bool        
            Choose whether to free (free=True) or fix (free=False).
        
        """

        if free: self.logger.debug('Freeing norm for ' + name)
        else: self.logger.debug('Fixing norm for ' + name)
        
        normPar = self.like.normPar(name).getName()
        par_index = self.like.par_index(name,normPar)
        self.like[par_index].setFree(free)
        self.like.syncSrcParams(name)

    def free_index(self,name,free=True):
        """Free/Fix index of a source."""
        pass
    
    def sed(self,name,compute_lnlprofile=True):
        
        # Find the source
        if not name in ['isodiff','galdiff','limbdiff']:
            name = self._roi.get_source_by_name(name).name
        
        saved_state = LikelihoodState(self.like)
        
        energies = self.energies
        nbins = self.enumbins

        o = {'emin' : energies[:-1],
             'emax' : energies[1:],
             'ecenter' : 0.5*(energies[:-1]+energies[1:]),
             'flux' : np.zeros(nbins),
             'eflux' : np.zeros(nbins),
             'flux_err' : np.zeros(nbins),
             'eflux_err' : np.zeros(nbins),
             'npred' : np.zeros(nbins),
             'ts' : np.zeros(nbins),
             'lnlprofile' : []
             }
        
        for i, (emin,emax) in enumerate(zip(energies[:-1],energies[1:])):
            saved_state.restore()
            self.free_sources(free=False)
            self.free_norm(name)            
            self.logger.debug('Fitting %s SED from %.0fMeV to %.0fMeV' % (name,10**emin,10**emax))
            self.setEnergyRange(float(10**emin)+1, float(10**emax)-1)
            self.fit()
            
            o['flux'][i] = self.like[name].flux(10**emin, 10**emax)
            o['eflux'][i] = self.like[name].energyFlux(10**emin, 10**emax)
            o['flux_err'][i] = self.like.fluxError(name,10**emin, 10**emax)
            o['eflux_err'][i] = self.like.energyFluxError(name,10**emin, 10**emax)

            cs = self.modelCountsSpectrum(name,emin,emax)
            for c in cs: o['npred'][i] = np.sum(c)            
            o['ts'][i] = self.like.Ts(name,reoptimize=False)

            if compute_lnlprofile:
                o['lnlprofile'] += [self.profile(name,emin=emin,emax=emax)]
            
#            nobs.append(self.gtlike.nobs[i])
        saved_state.restore()

        self.setEnergyRange(float(10**energies[0])+1, float(10**energies[-1])-1)
        
        src_model = self._roi_model.get(name,{})
        src_model['sed'] = copy.deepcopy(o)        
        return o
            
    def profile(self, name, emin=None,emax=None, reoptimize=False,xvals=None,npts=None):
#                  flux_min=0, flux_max=10, emin=None, emax=None, 
#                  npts=None, fix_src_pars=False, verbosity=0, log=True, 
#                  **kwargs):
        """ Profile the likelihood for the given source and parameter.  
        """
        # Find the source
        if not name in ['isodiff','galdiff','limbdiff']:
            name = self._roi.get_source_by_name(name).name

        par = self.like.normPar(name)
        parName = self.like.normPar(name).getName()
        idx = self.like.par_index(name,parName)
        scale = float(self.like.model[idx].getScale())
        bounds = self.like.model[idx].getBounds()

        emin = min(self.energies) if emin is None else emin
        emax = max(self.energies) if emax is None else emax

        saved_state = LikelihoodState(self.like)
        
        self.setEnergyRange(float(10**emin)+1, float(10**emax)-1)
        
        logLike0 = self.like()
        print parName, idx, scale, bounds, par.getValue(), par.error()

        if xvals is None:

            err = par.error()
            val = par.getValue()
            if err <= 0 or val <= 3*err:
                xvals = np.linspace(-2.0,1.0,41)
                xvals = err*10**xvals
            else:
                xvals = np.linspace(0,1,21)
                xvals = np.concatenate((-1.0*xvals[1:][::-1],xvals))
                xvals = val*10**xvals

        self.like[idx].setBounds(xvals[0],xvals[-1])

        o = {'xvals'    : xvals,
             'npred'    : np.zeros(len(xvals)),
             'fluxes'   : np.zeros(len(xvals)),
             'efluxes'  : np.zeros(len(xvals)),
             'dlogLike' : np.zeros(len(xvals)) }
                     
        for i, x in enumerate(xvals):
            
            self.like[idx] = x
            self.like.syncSrcParams(name)

            if self.like.logLike.getNumFreeParams() > 1 and reoptimize:
                # Only reoptimize if not all frozen                
                self.like.freeze(idx)
                self.like.optimize(0, **kwargs)
                self.like.thaw(idx)
                
            logLike1 = self.like()
            o['dlogLike'][i] = logLike0 - logLike1
            o['fluxes'][i] = self.like[name].flux(10**emin, 10**emax)
            o['efluxes'][i] = self.like[name].energyFlux(10**emin, 10**emax)

            cs = self.modelCountsSpectrum(name,emin,emax)
            for c in cs: o['npred'][i] += np.sum(c)
            
#            if verbosity:
#                print "%-10i%-12.5g%-12.5g%-12.5g%-12.5g%-12.5g"%(i,x,npred[-1],fluxes[-1],
#                                                                  efluxes[-1],dlogLike[-1])
#        if len(self.like.model.srcs) == 1 and fluxes[0] == 0:
#            # Likelihood is undefined with one source and no flux, hack it..
#            dlogLike[0] = dlogLike[1]

        # Restore model parameters to original values
        saved_state.restore()
        self.like[idx].setBounds(*bounds)
#        print parName, idx, scale, bounds, par.getValue(), par.error()
        
        return o
    
    def initOptimizer(self):
        pass        

    def create_optObject(self):
        """ Make MINUIT or NewMinuit type optimizer object """

        optimizer = self.config['common']['optimizer']
        if optimizer.upper() == 'MINUIT':
            optObject = pyLike.Minuit(self.like.logLike)
        elif optimizer.upper == 'NEWMINUIT':
            optObject = pyLike.NewMinuit(self.like.logLike)
        else:
            optFactory = pyLike.OptimizerFactory_instance()
            optObject = optFactory.create(optimizer, self.like.logLike)
        return optObject
    
    def fit(self):
        """Run likelihood optimization."""

        if not self.like.logLike.getNumFreeParams(): 
            self.logger.info("Skipping fit.  No free parameters.")
            return
        
        saved_state = LikelihoodState(self.like)
        kw = dict(optObject = self.create_optObject(),
                  covar=True,verbosity=0)
#tol=1E-4
#                  optimizer='DRMNFB')
        
# if 'verbosity' not in kwargs: kwargs['verbosity'] = max(self.config['chatter'] - 1, 0)
        niter = 0; max_niter = self.config['common']['retries']
        try: 
            while niter < max_niter:
                self.logger.info("Fit iteration: %i"%niter)
                niter += 1
                self.like.fit(**kw)
                if isinstance(self.like.optObject,pyLike.Minuit) or \
                        isinstance(self.like.optObject,pyLike.NewMinuit):
                    quality = self.like.optObject.getQuality()
                    if quality > 2: return
                else: return
            raise Exception("Failed to converge with %s"%self.like.optimizer)
        except Exception, message:
            self.logger.error('Likelihood optimization failed.', exc_info=True)
            saved_state.restore()
        

    def fitDRM(self):
        
        kw = dict(optObject = None, #pyLike.Minuit(self.like.logLike),
                  covar=True,#tol=1E-4
                  optimizer='DRMNFB')

        

        
#        self.MIN.tol = float(self.likelihoodConf['mintol'])
        
        
        try:
            self.like.fit(**kw)
        except Exception, message:
            print message
            print "Failed to converge with DRMNFB"

        kw = dict(optObject = pyLike.Minuit(self.like.logLike),
                  covar=True)

        self.like.fit(**kw)
        
    def load_xml(self,xmlfile):
        """Load model definition from XML."""
        raise NotImplementedError()

    def write_xml(self,model_name):
        """Save current model definition as XML file.

        Parameters
        ----------

        model_name : str
            Name of the output model.

        """

        for i, c in enumerate(self._components):
            c.write_xml(model_name)

        # Write a common XML file?

    def write_roi(self,outfile=None):
        """Write out parameters of current model as yaml file."""
        # extract the results in a convenient format

        if outfile is None:
            outfile = os.path.join(self._savedir,'results.yaml')
        else:
            outfile, ext = os.path.splitext(outfile)
            if not ext:
                outfile = os.path.join(self._savedir,outfile + '.yaml')
            else:
                outfile = outfile + ext
                        
        o = self.get_roi_dict()
                
        # Get the subset of sources with free parameters
            
        yaml.dump(tolist(o),open(outfile,'w'))

    def get_roi_dict(self):
        """Populate a dictionary with the current parameters of the
        ROI model as extracted from the pylikelihood object."""

        # Should we skip extracting fit results for sources that
        # weren't free in the last fit?

        # Determine what sources had at least one free parameter?
        gf = {}        
        for name in self.like.sourceNames():
            
            source = self.like[name].src
            spectrum = source.spectrum()

            src_dict = gtlike_spectrum_to_dict(spectrum)
            
            # Should we update the TS values at the end of fitting?
            src_dict['ts'] = self.like.Ts(name,reoptimize=False)

            # Get NPred
            src_dict['npred'] = self.like.NpredValue(name)
            
            # Extract covariance matrix
            src_dict['covar'] = None
            
            try:
                 fd = FluxDensity.FluxDensity(self.like,name)
                 src_dict['covar'] = fd.covar
            except RuntimeError, ex:
                pass
#                 if ex.message == 'Covariance matrix has not been computed.':
#                      pass
#                 elif 
#                      raise ex
                 
            # Extract bowtie            
            gf[name] = src_dict

        self._roi_model = merge_dict(self._roi_model,gf,add_new_keys=True) 
        return copy.deepcopy(self._roi_model)        
            
class GTBinnedAnalysis(AnalysisBase):

    defaults = dict(defaults.selection.items()+
                    defaults.binning.items()+
                    defaults.irfs.items()+
                    defaults.inputs.items()+
                    defaults.fileio.items(),
                    roi=defaults.roi,
                    name=('00',''),
                    file_suffix=('',''),
                    verbosity=(0,''))

    def __init__(self,config,roi,**kwargs):
        super(GTBinnedAnalysis,self).__init__(config,**kwargs)

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['logfile'],
                                 ll(self.config['verbosity']))

        self.print_config(self.logger,loglevel=logging.DEBUG)
        
        savedir = self.config['savedir']
        self._roi = roi
        self._name = self.config['name']
        
        from os.path import join

        self._ft1_file=join(savedir,
                            'ft1%s.fits'%self.config['file_suffix'])
        self._ft1_filtered_file=join(savedir,
                                     'ft1_filtered%s.fits'%self.config['file_suffix'])        
        self._ltcube=join(savedir,
                          'ltcube%s.fits'%self.config['file_suffix'])
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

        self._enumbins = np.round(self.config['binsperdec']*
                                 np.log10(self.config['emax']/self.config['emin']))
        self._enumbins = int(self._enumbins)
        self._ebin_edges = np.linspace(np.log10(self.config['emin']),
                                       np.log10(self.config['emax']),
                                       self._enumbins+1)
        self._ebin_center = 0.5*(self._ebin_edges[1:] + self._ebin_edges[:-1])
        
        if self.config['npix'] is None:
            self.npix = int(np.round(self.config['roi_width']/self.config['binsz']))
        else:
            self.npix = self.config['npix']
            
    @property
    def roi(self):
        return self._roi

    @property
    def like(self):
        return self._like

    @property
    def name(self):
        return self._name

    @property
    def energies(self):
        return self._ebin_edges

    def setEnergyRange(self,emin,emax):
        self.like.setEnergyRange(emin,emax)

    def modelCountsSpectrum(self,name,emin,emax):
        cs = np.array(self.like.logLike.modelCountsSpectrum(name))
        imin = valToBinBounded(self.energies,emin+1E-7)[0]
        imax = valToBinBounded(self.energies,emax-1E-7)[0]+1

        print emin, emax, imin, imax
        
        if imax <= imin: raise Exception('Invalid energy range.')        
        return cs[imin:imax]
        
    def setup(self):
        """Run pre-processing step."""

        # Write ROI XML
        self._roi.write_xml(self._srcmdl_file)
        roi_center = self._roi.radec
        
        # Run gtselect and gtmktime
        kw_gtselect = dict(infile=self.config['evfile'],
                           outfile=self._ft1_file,
                           ra=roi_center[0], dec=roi_center[1],
                           rad=self.config['radius'],
                           convtype=self.config['convtype'],
                           evtype=self.config['evtype'],
                           evclass=self.config['evclass'],
                           tmin=self.config['tmin'], tmax=self.config['tmax'],
                           emin=self.config['emin'], emax=self.config['emax'],
                           zmax=self.config['zmax'],
                           chatter=self.config['verbosity'])

        kw_gtmktime = dict(evfile=self._ft1_file,
                           outfile=self._ft1_filtered_file,
                           scfile=self.config['scfile'],
                           roicut='no',
                           filter=self.config['filter'])

        if not os.path.isfile(self._ft1_file):
            run_gtapp('gtselect',self.logger,kw_gtselect)
            run_gtapp('gtmktime',self.logger,kw_gtmktime)
            os.system('mv %s %s'%(self._ft1_filtered_file,self._ft1_file))
        else:
            self.logger.info('Skipping gtselect')
            
        # Run gtltcube
        kw = dict(evfile=self._ft1_file,
                  scfile=self.config['scfile'],
                  outfile=self._ltcube,
                  zmax=self.config['zmax'])
        
        if self.config['ltcube'] is not None:
            self._ltcube = self.config['ltcube']
        elif not os.path.isfile(self._ltcube):             
            run_gtapp('gtltcube',self.logger,kw)
        else:
            self.logger.info('Skipping gtltcube')
            
        # Run gtbin
        kw = dict(algorithm='ccube',
                  nxpix=self.npix, nypix=self.npix,
                  binsz=self.config['binsz'],
                  evfile=self._ft1_file,
                  outfile=self._ccube_file,
                  scfile=self.config['scfile'],
                  xref=float(self.roi.radec[0]),
                  yref=float(self.roi.radec[1]),
                  axisrot=0,
                  proj=self.config['proj'],
                  ebinalg='LOG',
                  emin=self.config['emin'],
                  emax=self.config['emax'],
                  enumbins=self._enumbins,
                  coordsys=self.config['coordsys'],
                  chatter=self.config['verbosity'])
        
        if not os.path.isfile(self._ccube_file):
            run_gtapp('gtbin',self.logger,kw)            
        else:
            self.logger.info('Skipping gtbin')

        evtype = self.config['evtype']
            
        if self.config['irfs'] == 'CALDB':
            cmap = self._ccube_file
        else:
            cmap = 'none'
            
        # Run gtexpcube2
        kw = dict(infile=self._ltcube,cmap=cmap,
                  ebinalg='LOG',
                  emin=self.config['emin'], emax=self.config['emax'],
                  enumbins=self._enumbins,
                  outfile=self._bexpmap_file, proj='CAR',
                  nxpix=360, nypix=180, binsz=1,
                  xref=0.0,yref=0.0,
                  evtype=evtype,
                  irfs=self.config['irfs'],
                  coordsys=self.config['coordsys'],
                  chatter=self.config['verbosity'])

        if not os.path.isfile(self._bexpmap_file):
            run_gtapp('gtexpcube2',self.logger,kw)              
        else:
            self.logger.info('Skipping gtexpcube')

        # Run gtsrcmaps
        kw = dict(scfile=self.config['scfile'],
                  expcube=self._ltcube,
                  cmap=self._ccube_file,
                  srcmdl=self._srcmdl_file,
                  bexpmap=self._bexpmap_file,
                  outfile=self._srcmap_file,
                  irfs=self.config['irfs'],
                  evtype=evtype,
#                   rfactor=self.config['rfactor'],
#                   resample=self.config['resample'],
#                   minbinsz=self.config['minbinsz'],
                  chatter=self.config['verbosity'],
                  emapbnds='no' ) 

        if not os.path.isfile(self._srcmap_file):
            run_gtapp('gtsrcmaps',self.logger,kw)             
        else:
            self.logger.info('Skipping gtsrcmaps')

        # Create BinnedObs
        self.logger.info('Creating BinnedObs')
        kw = dict(srcMaps=self._srcmap_file,expCube=self._ltcube,
                  binnedExpMap=self._bexpmap_file,
                  irfs=self.config['irfs'])
        self.logger.info(kw)
        
        self._obs=ba.BinnedObs(**kw)

        # Create BinnedAnalysis
        self.logger.info('Creating BinnedAnalysis')
        self._like = ba.BinnedAnalysis(binnedData=self._obs,
                                       srcModel=self._srcmdl_file,
                                       optimizer='MINUIT')

        if self.config['enable_edisp']:
            self.logger.info('Enabling energy dispersion')
            self.like.logLike.set_edisp_flag(True)
            
    def generate_model(self,model_name=None,outfile=None):
        """Generate a counts model map.

        Parameters
        ----------

        model_name : str
        
            Name of the model.  If no name is given it will default to
            the seed model.

        outfile : str

            Override the name of the output model file.
            
        """


        
        if model_name is None: srcmdl = self._srcmdl_file
        else: srcmdl = self.get_model_path(model_name)

        if not os.path.isfile(srcmdl):
            raise Exception("Model file does not exist: %s"%srcmdl)
        
#        if outfile is None: outfile = self._mcube_file
        outfile = os.path.join(self.config['savedir'],
                               'mcube_%s%s.fits'%(model_name,
                                                  self.config['file_suffix']))
        
        # May consider generating a custom source model file

        if not os.path.isfile(outfile):

            kw = dict(srcmaps = self._srcmap_file,
                      srcmdl  = srcmdl,
                      bexpmap = self._bexpmap_file,
                      outfile = outfile,
                      expcube = self._ltcube,
                      irfs    = self.config['irfs'],
                      evtype  = self.config['evtype'],
                      # edisp   = bool(self.config['enable_edisp']),
                      outtype = 'ccube',
                      chatter = self.config['verbosity'])
            
            run_gtapp('gtmodel',self.logger,kw)       
        else:
            self.logger.info('Skipping gtmodel')
            

    def write_xml(self,model_name):
        """Write the XML model for this analysis component."""
        
        xmlfile = self.get_model_path(model_name)            
        self.logger.info('Writing %s...'%xmlfile)
        self.like.writeXml(xmlfile)

    def get_model_path(self,name):
        """Infer the path to the XML model name."""
        
        name, ext = os.path.splitext(name)
        if not ext: ext = '.xml'
        xmlfile = name + self.config['file_suffix'] + ext

        if os.path.commonprefix([self.config['savedir'],xmlfile]) \
                != self.config['savedir']:        
            xmlfile = os.path.join(self.config['savedir'],xmlfile)

        return xmlfile
