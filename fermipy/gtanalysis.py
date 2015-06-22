
import os
import sys
import copy
import glob
import shutil
import yaml
from utils import *
import defaults
import fermipy
from fermipy.roi_model import *
from fermipy.logger import Logger, StreamLogger
from fermipy.logger import logLevel as ll
import logging
import tempfile

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


norm_parameters = {
    'ConstantValue' : ['Value'],
    'PowerLaw' : ['Prefactor'],
    'PowerLaw2' : ['Integral'],
    'BrokenPowerLaw' : ['Prefactor'],    
    'LogParabola' : ['norm'],
    'PLSuperExpCutoff' : ['Prefactor'],
    'ExpCutoff' : ['Prefactor'],
    'FileFunction' : ['Normalization'],
    }

shape_parameters = {
    'ConstantValue' : [],
    'PowerLaw' : ['Index'],
    'PowerLaw2' : ['Index'],
    'BrokenPowerLaw' : ['Index1','Index2'],    
    'LogParabola' : ['alpha','beta','Eb'],    
    'PLSuperExpCutoff' : ['Index1','Index2','Cutoff'],
    'ExpCutoff' : ['Index1','Cutoff'],
    'FileFunction' : [],
    }

index_parameters = {
    'ConstantValue' : [],
    'PowerLaw' : ['Index'],
    'PowerLaw2' : ['Index'],
    'BrokenPowerLaw' : ['Index1','Index2'],    
    'LogParabola' : ['alpha','beta'],    
    'PLSuperExpCutoff' : ['Index1','Index2'],
    'ExpCutoff' : ['Index1'],
    'FileFunction' : [],
    }
             
def cl_to_dlnl(cl):
    import scipy.special as spfn    
    alpha = 1.0-cl    
    return 0.5*np.power(np.sqrt(2.)*spfn.erfinv(1-2*alpha),2.)    

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
        pval = p.getTrueValue()
        perr = abs(p.error()*p.getScale()) if p.isFree() else np.nan        
        d[pname]= np.array([pval,perr])
        
        if d['spectrum_type'] == 'FileFunction': 
            ff=pyLike.FileFunction_cast(spectrum)
            d['file']=ff.filename()
    return d
        
class GTAnalysis(AnalysisBase):
    """High-level analysis interface that internally manages a set of
    analysis component objects.  Most of the functionality of the
    fermiPy package is provided through the methods of this class.
    The class constructor accepts a dictionary that defines the
    configuration for the analysis.  Keyword arguments provided as
    **kwargs can be used to override parameter values in the
    configuration dictionary."""

    defaults = {'logging'    : defaults.logging,
                'fileio'     : defaults.fileio,
                'optimizer'  : defaults.optimizer,
                'binning'    : defaults.binning,
                'selection'  : defaults.selection,
                'model'      : defaults.model,
                'data'       : defaults.data,
                'gtlike'     : defaults.gtlike,
                'mc'         : defaults.mc,
#                'roiopt'     : defaults.roiopt,
                'components' : (None,'')}

    def __init__(self,config,**kwargs):
        super(GTAnalysis,self).__init__(config,**kwargs)

        # Setup directories
        self._rootdir = os.getcwd()
                        
        # Destination directory for output data products
        if self.config['fileio']['outdir'] is not None:
            self._savedir = os.path.join(self._rootdir,
                                         self.config['fileio']['outdir'])
            mkdir(self._savedir)
        else:
            raise Exception('Save directory not defined.')

        # put pfiles into savedir
        os.environ['PFILES']= \
            self._savedir+';'+os.environ['PFILES'].split(';')[-1]

        if self.config['fileio']['logfile'] is None:
            self._config['fileio']['logfile'] = os.path.join(self._savedir,'fermipy')
            
        self.logger = Logger.get(self.__class__.__name__,self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))

        self.logger.info('\n' + '-'*80 + '\n' + "This is fermipy version {}.".
                         format(fermipy.__version__))
        self.print_config(self.logger)
        
        # Working directory (can be the same as savedir)
#        if self.config['fileio']['scratchdir'] is not None:
        if self.config['fileio']['stageoutput']:
            self._workdir = tempfile.mkdtemp(prefix=os.environ['USER'] + '.',
                                             dir=self.config['fileio']['scratchdir'])
            self.logger.info('Created working directory: %s'%self._workdir)
            self.stage_input()
        else:
            self._workdir = self._savedir
        
        # Setup the ROI definition
        self._roi = \
            ROIModel.create_from_source(self.config['selection']['target'],
                                          self.config['model'],
                                          logfile=self.config['fileio']['logfile'],
                                          logging=self.config['logging'])

        self._like = SummedLikelihood.SummedLikelihood()
        self._components = []
        configs = self.create_component_configs()

        for cfg in configs:
            comp = self._create_component(cfg)
            self._components.append(comp)

        energies = np.zeros(0)
        for c in self.components:
            energies = np.concatenate((energies,c.energies))
            
        self._ebin_edges = np.sort(np.unique(energies.round(5)))
        self._enumbins = len(self._ebin_edges)-1
        self._roi_model = {'roi' : {'logLike' : np.nan}}

    def __del__(self):
        self.stage_output()
        self.cleanup()
        
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
        """Return the number of energy bins."""
        return self._enumbins    
    
    def create_component_configs(self):
        configs = []

        components = self.config['components']

        common_config = GTBinnedAnalysis.get_config()
        common_config = merge_dict(common_config,self.config)
        
        if components is None:
            cfg = copy.copy(common_config)
            cfg['file_suffix'] = '_00'
            cfg['name'] = '00'      
            configs.append(cfg)
        elif isinstance(components,dict):            
            for i,k in enumerate(sorted(components.keys())):
                cfg = copy.copy(common_config)                
                cfg = merge_dict(cfg,components[k])
                cfg['file_suffix'] = '_' + k
                cfg['name'] = k
                configs.append(cfg)
        elif isinstance(components,list):
            for i,c in enumerate(components):
                cfg = copy.copy(common_config)                
                cfg = merge_dict(cfg,c)
                cfg['file_suffix'] = '_%02i'%i
                cfg['name'] = '%02i'%i
                configs.append(cfg)
        else:
            raise Exception('Invalid type for component block.')

        return configs
                
    def _create_component(self,cfg):
            
        self.logger.info("Creating Analysis Component: " + cfg['name'])

        cfg['fileio']['workdir'] = self._workdir
        
        comp = GTBinnedAnalysis(cfg,
                                logging=self.config['logging'])

        return comp

    def stage_output(self):
        """Copy data products to final output directory."""

        if self._workdir == self._savedir:
            return
        elif os.path.isdir(self._workdir):
            self.logger.info('Staging output data products to %s'%self._savedir)
            for f in glob.glob(os.path.join(self._workdir,'*.fits')):
                shutil.copy(f,self._savedir)
        else:
            self.logger.error('Working directory does not exist.')

    def stage_input(self):
        """Copy data products to intermediate working directory."""

        if self._workdir == self._savedir:
            return
        elif os.path.isdir(self._workdir):
            for f in glob.glob(os.path.join(self._savedir,'*')):
                shutil.copy(f,self._workdir)
        else:
            self.logger.error('Working directory does not exist.')
            
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

            src = self._roi.get_source_by_name(name)
            
            src_model = {'sed' : None}

            if isinstance(src,Source):
                src_model['RA'] = src['RAJ2000']
                src_model['DEC'] = src['DEJ2000']

            src_model.update(self.get_src_model(name,True))            
            self._roi_model[name] = src_model

        self._roi_model['roi']['nobs'] = np.zeros(self.enumbins)

    def cleanup(self):

        if self._workdir == self._savedir: return
        elif os.path.isdir(self._workdir):
            self.logger.info('Deleting working directory: ' + self._workdir)
            shutil.rmtree(self._workdir)
            
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
            
    def modelCountsSpectrum(self,name,emin,emax,summed=False):
        """Return the predicted number of model counts versus energy
        for a given source and energy range.  If summed=True return
        the counts spectrum summed over all components otherwise
        return a list of model spectra."""
        
        if summed:
            cs = np.zeros(self.enumbins)
            imin = valToBinBounded(self.energies,emin+1E-7)[0]
            imax = valToBinBounded(self.energies,emax-1E-7)[0]+1

            for c in self.components:
                ecenter = 0.5*(c.energies[:-1]+c.energies[1:])
                counts = c.modelCountsSpectrum(name,self.energies[0],
                                               self.energies[-1])

                cs += np.histogram(ecenter,
                                   weights=counts,
                                   bins=self.energies)[0]

            return cs[imin:imax]
        else:        
            cs = []
            for c in self.components: 
                cs += [c.modelCountsSpectrum(name,emin,emax)]            
            return cs

    def get_sources(self,cuts=None,distance=None,roilike=False):
        """Retrieve list of sources satisfying the given selections."""
        rsrc, srcs = self._roi.get_sources_by_position(self._roi.radec[0],
                                                       self._roi.radec[1],
                                                       distance,roilike=roilike)
        o = []
        if cuts is None: cuts = []        
        for s,r in zip(srcs,rsrc):
            if not s.check_cuts(cuts): continue            
            o.append(s)

        return o
        
    
    def delete_sources(self,cuts=None,distance=None,roilike=False):
        """Delete sources within the ROI."""
        
        srcs = self.get_sources(cuts,distance,roilike)
        self._roi.delete_sources(srcs)    
        for c in self.components:
            c.delete_sources(srcs)
            
    def free_sources(self,free=True,pars=None,cuts=None,
                     distance=None,roilike=False):
        """Free/Fix sources within the ROI.

        Parameters
        ----------

        free : bool        
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list        
            Set a list of parameters to be freed/fixed for this
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        distance : float        
            Distance out to which sources should be freed or fixed.
            If none then all sources will be selected.

        roilike : bool        
            Apply an ROI-like selection on the maximum distance in
            either X or Y in projected cartesian coordinates.        
        
        """
        rsrc, srcs = self._roi.get_sources_by_position(self._roi.radec[0],
                                                       self._roi.radec[1],
                                                       distance,roilike=roilike)
        
        if cuts is None: cuts = []        
        for s,r in zip(srcs,rsrc):
            if not s.check_cuts(cuts): continue            
            self.free_source(s.name,free=free,pars=pars)

        for s in self._roi._diffuse_srcs:
#            if not s.check_cuts(cuts): continue
            self.free_source(s.name,free=free,pars=pars)
                                        
    def free_sources_by_position(self,free=True,pars=None,distance=None,roilike=False):
        """Free/Fix all sources within a certain distance of the given sky
        coordinate.  By default it will use the ROI center.

        Parameters
        ----------

        free : bool        
            Choose whether to free (free=True) or fix (free=False)
            source parameters.

        pars : list        
            Set a list of parameters to be freed/fixed for this
            source.  If none then all source parameters will be
            freed/fixed.  If pars='norm' then only normalization
            parameters will be freed.

        distance : float        
            Distance out to which sources should be freed or fixed.
            If none then all sources will be selected.

        roilike : bool        
            Apply an ROI-like selection on the maximum distance in
            either X or Y in projected cartesian coordinates.        
        """

        self.free_sources(free,pars,cuts=None,distance=distance,roilike=roilike)

    def set_edisp_flag(self,name,flag=True):

        src = self._roi.get_source_by_name(name)
        name = src.name
        
        for c in self.components:
            c.like[name].src.set_edisp_flag(flag)        
        
    def free_source(self,name,free=True,pars=None):
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
        src = self._roi.get_source_by_name(name)
        name = src.name
        
        if pars is None:
            pars = []
            pars += norm_parameters[src['SpectrumType']]
            pars += shape_parameters[src['SpectrumType']]
        elif pars == 'norm':
            pars = []
            pars += norm_parameters[src['SpectrumType']]
        elif pars == 'shape':
            pars = []
            pars += shape_parameters[src['SpectrumType']]            
        else:
            raise Exception('Invalid parameter list.')
            
        # Deduce here the names of all parameters from the spectral type
        src_par_names = pyLike.StringVector()
        self.like[name].src.spectrum().getParamNames(src_par_names)

        par_indices = []
        par_names = []
        for p in src_par_names:
            if pars is not None and not p in pars: continue
            par_indices.append(self.like.par_index(name,p))
            par_names.append(p)

        if free:
            self.logger.info('Freeing parameters for %-20s: %s'
                             %(name,par_names))
        else:
            self.logger.info('Fixing parameters for %-20s: %s'
                             %(name,par_names))
            
        for (idx,par_name) in zip(par_indices,par_names):

            
                
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

        name = self._roi.get_source_by_name(name).name
        
        if free: self.logger.debug('Freeing norm for ' + name)
        else: self.logger.debug('Fixing norm for ' + name)
        
        normPar = self.like.normPar(name).getName()
        par_index = self.like.par_index(name,normPar)
        self.like[par_index].setFree(free)
        self.like.syncSrcParams(name)

    def free_index(self,name,free=True):
        """Free/Fix index of a source."""
        src = self._roi.get_source_by_name(name)

        self.free_source(src.name,free=free,
                         pars=index_parameters[src['SpectrumType']])
        
    def free_shape(self,name,free=True):
        """Free/Fix shape parameters of a source."""
        src = self._roi.get_source_by_name(name)

        self.free_source(src.name,free=free,
                         pars=shape_parameters[src['SpectrumType']])


    def get_free_source_params(self,name):
        src = self._roi.get_source_by_name(name)
        spectrum = self.like[name].src.spectrum()
        parNames = pyLike.StringVector()
        spectrum.getFreeParamNames(parNames)
        return [str(p) for p in parNames]
    
    def sed(self,name,profile=True,energies=None):
        
        # Find the source
        name = self._roi.get_source_by_name(name).name

        self.logger.info('Computing SED for %s'%name)
        saved_state = LikelihoodState(self.like)

        if energies is None: energies = self.energies
        else: energies = np.array(energies)
        
        nbins = len(energies)-1

        o = {'emin' : energies[:-1],
             'emax' : energies[1:],
             'ecenter' : 0.5*(energies[:-1]+energies[1:]),
             'flux' : np.zeros(nbins),
             'e2flux' : np.zeros(nbins),
             'flux_err' : np.zeros(nbins),
             'e2flux_err' : np.zeros(nbins),
             'flux_ul95' : np.zeros(nbins)*np.nan,
             'e2flux_ul95' : np.zeros(nbins)*np.nan,
             'flux_err_lo' : np.zeros(nbins)*np.nan,
             'e2flux_err_lo' :  np.zeros(nbins)*np.nan,
             'flux_err_hi' : np.zeros(nbins)*np.nan,
             'e2flux_err_hi' :  np.zeros(nbins)*np.nan,
             'Npred' : np.zeros(nbins),
             'ts' : np.zeros(nbins),
             'fit_quality' : np.zeros(nbins),
             'lnlprofile' : []
             }
        
        for i, (emin,emax) in enumerate(zip(energies[:-1],energies[1:])):
            saved_state.restore()
            self.free_sources(free=False)
            self.free_norm(name)
            self.logger.info('Fitting %s SED from %.0f MeV to %.0f MeV' %
                             (name,10**emin,10**emax))
            self.setEnergyRange(float(10**emin)+1, float(10**emax)-1)
            o['fit_quality'][i] = self.fit(update=False)

            ecenter = 0.5*(emin+emax)
            deltae = 10**emax - 10**emin
            flux = self.like[name].flux(10**emin, 10**emax)
            flux_err = self.like.fluxError(name,10**emin, 10**emax)
            
            o['flux'][i] = flux/deltae 
            o['e2flux'][i] = flux/deltae*10**(2*ecenter)
            o['flux_err'][i] = flux_err/deltae
            o['e2flux_err'][i] = flux_err/deltae*10**(2*ecenter)

            cs = self.modelCountsSpectrum(name,emin,emax,summed=True)
            o['Npred'][i] = np.sum(cs)            
            o['ts'][i] = self.like.Ts(name,reoptimize=False)
            if profile:

                lnlp = self.profile_norm(name,emin=emin,emax=emax)                
                o['lnlprofile'] += [lnlp]
                
                imax = np.argmax(lnlp['dlogLike'])
                lnlmax = lnlp['dlogLike'][imax]
                dlnl = lnlp['dlogLike']-lnlmax
                                
                o['flux_ul95'][i] = np.interp(cl_to_dlnl(0.95),-dlnl[imax:],lnlp['flux'][imax:])
                o['e2flux_ul95'][i] = np.interp(cl_to_dlnl(0.95),-dlnl[imax:],lnlp['e2flux'][imax:])

                o['flux_err_hi'][i] = np.interp(0.5,-dlnl[imax:],lnlp['flux'][imax:]) - lnlp['flux'][imax]
                o['e2flux_err_hi'][i] = np.interp(0.5,-dlnl[imax:],lnlp['e2flux'][imax:]) - lnlp['e2flux'][imax] 

                if dlnl[0] < -0.5:
                    o['flux_err_lo'][i] = lnlp['flux'][imax] - np.interp(0.5,-dlnl[:imax][::-1],
                                                                         lnlp['flux'][:imax][::-1]) 
                    o['e2flux_err_lo'][i] = lnlp['e2flux'][imax] - np.interp(0.5,-dlnl[:imax][::-1],
                                                                             lnlp['e2flux'][:imax][::-1])
                
#            nobs.append(self.gtlike.nobs[i])

        self.setEnergyRange(float(10**energies[0])+1, float(10**energies[-1])-1)
        saved_state.restore()        
        src_model = self._roi_model.get(name,{})
        src_model['sed'] = copy.deepcopy(o)        
        return o

    def profile_norm(self,name, emin=None,emax=None, reoptimize=False,xvals=None,npts=None):
        """
        Profile the normalization of a source.
        """
        
        # Find the source
        name = self._roi.get_source_by_name(name).name

        par = self.like.normPar(name)
        parName = self.like.normPar(name).getName()
        idx = self.like.par_index(name,parName)
        bounds = self.like.model[idx].getBounds()
        emin = min(self.energies) if emin is None else emin
        emax = max(self.energies) if emax is None else emax

        npred = np.sum(self.modelCountsSpectrum(name,emin,emax,summed=True))
        
        if xvals is None:

            err = par.error()
            val = par.getValue()

            if npred < 10:
                val *= 1./min(1.0,npred)
                xvals = val*10**np.linspace(-2.0,2.0,101)
                xvals = np.insert(xvals,0,0.0)
            else:
                xvals = np.linspace(0,1,51)
                xvals = np.concatenate((-1.0*xvals[1:][::-1],xvals))
                xvals = val*10**xvals
        
        return self.profile(name,parName,emin=emin,emax=emax,reoptimize=reoptimize,xvals=xvals)
    
    def profile(self, name, parName, emin=None,emax=None, reoptimize=False,xvals=None,npts=None):
        #                  **kwargs):
        """ Profile the likelihood for the given source and parameter.  
        """
        # Find the source
        name = self._roi.get_source_by_name(name).name

        par = self.like.normPar(name)
        parName = self.like.normPar(name).getName()
        idx = self.like.par_index(name,parName)
        scale = float(self.like.model[idx].getScale())
        bounds = self.like.model[idx].getBounds()

        emin = min(self.energies) if emin is None else emin
        emax = max(self.energies) if emax is None else emax

        ecenter = 0.5*(emin+emax)
        deltae = 10**emax - 10**emin
        npred = np.sum(self.modelCountsSpectrum(name,emin,emax,summed=True))
        
        saved_state = LikelihoodState(self.like)
        
        self.setEnergyRange(float(10**emin)+1, float(10**emax)-1)
        
        logLike0 = self.like()
#        print parName, idx, scale, bounds, par.getValue(), par.error()

        if xvals is None:

            err = par.error()
            val = par.getValue()
            if err <= 0 or val <= 3*err:                
                xvals = 10**np.linspace(-2.0,2.0,51)
                if val < xvals[0]: xvals = np.insert(xvals,val,0)
            else:
                xvals = np.linspace(0,1,25)
                xvals = np.concatenate((-1.0*xvals[1:][::-1],xvals))
                xvals = val*10**xvals

        self.like[idx].setBounds(xvals[0],xvals[-1])

        o = {'xvals'    : xvals,
             'Npred'    : np.zeros(len(xvals)),
             'flux'   : np.zeros(len(xvals)),
             'e2flux'  : np.zeros(len(xvals)),
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

            flux = self.like[name].flux(10**emin, 10**emax)
            
            o['dlogLike'][i] = logLike0 - logLike1
            o['flux'][i] = flux/deltae
            o['e2flux'][i] = flux/deltae*10**(2*ecenter)
#self.like[name].energyFlux(10**emin, 10**emax)

            cs = self.modelCountsSpectrum(name,emin,emax,summed=True)
            o['Npred'][i] += np.sum(cs)
            
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

        optimizer = self.config['optimizer']['optimizer']
        if optimizer.upper() == 'MINUIT':
            optObject = pyLike.Minuit(self.like.logLike)
        elif optimizer.upper == 'NEWMINUIT':
            optObject = pyLike.NewMinuit(self.like.logLike)
        else:
            optFactory = pyLike.OptimizerFactory_instance()
            optObject = optFactory.create(optimizer, self.like.logLike)
        return optObject

    def _run_fit(self,**kwargs):

        try:
            self.like.fit(**kwargs)            
        except Exception, message:
            self.logger.error('Likelihood optimization failed.', exc_info=True)

        if isinstance(self.like.optObject,pyLike.Minuit) or \
                isinstance(self.like.optObject,pyLike.NewMinuit):
            quality = self.like.optObject.getQuality()
        else:
            quality = 3

        return quality

    def fit(self,update=True,**kwargs):
        """Run likelihood optimization."""
        
        if not self.like.logLike.getNumFreeParams(): 
            self.logger.info("Skipping fit.  No free parameters.")
            return

        verbosity = kwargs.get('verbosity',self.config['optimizer']['verbosity'])
        covar = kwargs.get('covar',True)
        tol = kwargs.get('tol',self.config['optimizer']['tol'])
        
        saved_state = LikelihoodState(self.like)
        kw = dict(optObject = self.create_optObject(),
                  covar=covar,verbosity=verbosity,tol=tol)
#                  optimizer='DRMNFB')

        quality=0
        niter = 0; max_niter = self.config['optimizer']['retries']
        while niter < max_niter:
            self.logger.info("Fit iteration: %i"%niter)
            niter += 1
            quality = self._run_fit(**kw)
            if quality > 2: break
            
#        except Exception, message:
#            print self.like.optObject.getQuality()
#            self.logger.error('Likelihood optimization failed.', exc_info=True)
#            saved_state.restore()
#            return quality

        if quality < self.config['optimizer']['min_fit_quality']:
            self.logger.error("Failed to converge with %s"%self.like.optimizer)
            saved_state.restore()
            return quality
        elif not update:
            return quality

        for name in self.like.sourceNames():
            freePars = self.get_free_source_params(name)                
            if len(freePars) == 0: continue
            self._roi_model[name] = self.get_src_model(name)

        self._roi_model['roi']['logLike'] = self.like()
        self._roi_model['roi']['fit_quality'] = quality

        return quality
        
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

    def write_xml(self,xmlfile,save_model_map=True):
        """Save current model definition as XML file.

        Parameters
        ----------

        model_name : str
            Name of the output model.

        """

        model_name = os.path.splitext(xmlfile)[0]
        
        for i, c in enumerate(self._components):
            c.write_xml(xmlfile)
            if save_model_map:
                c.generate_model_map(model_name)

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
                        
        o = self.get_roi_model()
                
        # Get the subset of sources with free parameters
            
        yaml.dump(tolist(o),open(outfile,'w'))
                    
    def bowtie(self,fd,energies=None):
        
        if energies is None:
            emin = self.energies[0]
            emax = self.energies[-1]        
            energies = np.linspace(emin,emax,50)
        
        
        flux = [fd.value(10**x) for x in energies]
        flux_err = [fd.error(10**x) for x in energies]

        flux = np.array(flux)
        flux_err = np.array(flux_err)
        fhi = flux*(1.0 + flux_err/flux)
        flo = flux/(1.0 + flux_err/flux)

        return {'ecenter' : energies, 'flux' : flux,
                'fluxlo' : flo, 'fluxhi' : fhi }
        
    def get_roi_model(self):
        """Populate a dictionary with the current parameters of the
        ROI model as extracted from the pylikelihood object."""

        # Should we skip extracting fit results for sources that
        # weren't free in the last fit?

        # Determine what sources had at least one free parameter?
        gf = {}        
        for name in self.like.sourceNames():
            
#            source = self.like[name].src
#            spectrum = source.spectrum()

            gf[name] = self.get_src_model(name)

        self._roi_model = merge_dict(self._roi_model,gf,add_new_keys=True) 
        return copy.deepcopy(self._roi_model)        

    def get_src_model(self,name,paramsonly=False):
        source = self.like[name].src
        spectrum = source.spectrum()

        src_dict = { }

        src_dict['params'] = gtlike_spectrum_to_dict(spectrum)

        # Get NPred
        src_dict['Npred'] = self.like.NpredValue(name)

        if not self.get_free_source_params(name) or paramsonly:
            return src_dict
        
        # Should we update the TS values at the end of fitting?
        src_dict['ts'] = self.like.Ts(name,reoptimize=False)
            
        # Extract covariance matrix
        fd = None            
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
        if fd and len(src_dict['covar']) and src_dict['covar'].ndim >= 1:
            src_dict['model_flux'] = self.bowtie(fd)

        return src_dict
    
class GTBinnedAnalysis(AnalysisBase):

    defaults = dict(selection=defaults.selection,
                    binning=defaults.binning,
                    gtlike=defaults.gtlike,
                    data=defaults.data,
                    model=defaults.model,
                    logging=defaults.logging,
                    fileio=defaults.fileio,
                    name=('00',''),
                    file_suffix=('',''))

    def __init__(self,config,**kwargs):
        super(GTBinnedAnalysis,self).__init__(config,**kwargs)

        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))

        self._roi = \
            ROIModel.create_from_source(config['selection']['target'],
                                        config['model'],
                                        logfile=self.config['fileio']['logfile'],
                                        logging=self.config['logging'])

        
        
        workdir = self.config['fileio']['workdir']
        self._name = self.config['name']
        
        from os.path import join

        self._ft1_file=join(workdir,
                            'ft1%s.fits'%self.config['file_suffix'])
        self._ft1_filtered_file=join(workdir,
                                     'ft1_filtered%s.fits'%self.config['file_suffix'])        
        self._ltcube=join(workdir,
                          'ltcube%s.fits'%self.config['file_suffix'])
        self._ccube_file=join(workdir,
                             'ccube%s.fits'%self.config['file_suffix'])
        self._mcube_file=join(workdir,
                              'mcube%s.fits'%self.config['file_suffix'])
        self._srcmap_file=join(workdir,
                               'srcmap%s.fits'%self.config['file_suffix'])
        self._bexpmap_file=join(workdir,
                                'bexpmap%s.fits'%self.config['file_suffix'])
        self._srcmdl_file=join(workdir,
                               'srcmdl%s.xml'%self.config['file_suffix'])

        self._enumbins = np.round(self.config['binning']['binsperdec']*
                                 np.log10(self.config['selection']['emax']/
                                          self.config['selection']['emin']))
        self._enumbins = int(self._enumbins)
        self._ebin_edges = np.linspace(np.log10(self.config['selection']['emin']),
                                       np.log10(self.config['selection']['emax']),
                                       self._enumbins+1)
        self._ebin_center = 0.5*(self._ebin_edges[1:] + self._ebin_edges[:-1])
        
        if self.config['binning']['npix'] is None:
            self.npix = int(np.round(self.config['binning']['roiwidth']/self.config['binning']['binsz']))
        else:
            self.npix = self.config['binning']['npix']

        if self.config['selection']['radius'] is None:
            self._config['selection']['radius'] = np.sqrt(2.)*0.5*self.npix*self.config['binning']['binsz']+0.5
            self.logger.info('Automatically setting selection radius to %s deg'%self.config['radius'])

        self._like = None
            
        self.print_config(self.logger,loglevel=logging.DEBUG)
            
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

    @property
    def enumbins(self):
        return len(self._ebin_edges)-1

    def add_source(self,name,radec=None):

        if radec is None:
            radec = self.roi.radec
        
        src = pyLike.PointSource(0, 0,
                                 self.like.logLike.observation())
        pl = pyLike.SourceFactory_funcFactory().create("PowerLaw")
        pl.setParamValues((1, -2, 100))
        indexPar = pl.getParam("Index")
        indexPar.setBounds(-3.5, -1)
        pl.setParam(indexPar)
        prefactor = pl.getParam("Prefactor")
        prefactor.setBounds(1e-10, 1e3)
        prefactor.setScale(1e-9)
        pl.setParam(prefactor)
        src.setSpectrum(pl)
        src.setName(name)
        src.setDir(radec[0],radec[1],True,False)
        self.like.addSource(src)

    def delete_source(self,name):
        
        self.like.deleteSource(name)
        
    def delete_sources(self,srcs):
        for s in srcs:
            if self.like: self.like.deleteSource(s.name)
        self._roi.delete_sources(srcs)

    def set_edisp_flag(self,name,flag=True):
        src = self._roi.get_source_by_name(name)
        name = src.name        
        self.like[name].src.set_edisp_flag(flag)         
        
    def setEnergyRange(self,emin,emax):
        self.like.setEnergyRange(emin,emax)

    def countsMap(self):

        z = self.like.logLike.countsMap().data()
        z = np.array(z).reshape(self.enumbins,self.npix,self.npix).swapaxes(0,2)
        return z
        
    def modelCountsMap(self,name=None):
        """Return the model counts map for a single source or for the
        sum of all sources in the ROI."""
        
        v = pyLike.FloatVector(self.npix**2*self.enumbins)
        if name is None:
            for name in self.like.sourceNames():
                model = self.like.logLike.getSourceMap(name)
                self.like.logLike.updateModelMap(v,model)
        else:
            model = self.like.logLike.getSourceMap(name)
            self.like.logLike.updateModelMap(v,model)
        
        retVals = np.array(v).reshape(self.enumbins,
                                      self.npix,self.npix).swapaxes(0,2)
        return retVals
        
    def modelCountsSpectrum(self,name,emin,emax):
        cs = np.array(self.like.logLike.modelCountsSpectrum(name))
        imin = valToBinBounded(self.energies,emin+1E-7)[0]
        imax = valToBinBounded(self.energies,emax-1E-7)[0]+1
        
        if imax <= imin: raise Exception('Invalid energy range.')        
        return cs[imin:imax]
        
    def setup(self):
        """Run pre-processing step."""

        # Write ROI XML
        self._roi.write_xml(self._srcmdl_file)
        roi_center = self._roi.radec
        
        # Run gtselect and gtmktime
        kw_gtselect = dict(infile=self.config['data']['evfile'],
                           outfile=self._ft1_file,
                           ra=roi_center[0], dec=roi_center[1],
                           rad=self.config['selection']['radius'],
                           convtype=self.config['selection']['convtype'],
                           evtype=self.config['selection']['evtype'],
                           evclass=self.config['selection']['evclass'],
                           tmin=self.config['selection']['tmin'],
                           tmax=self.config['selection']['tmax'],
                           emin=self.config['selection']['emin'],
                           emax=self.config['selection']['emax'],
                           zmax=self.config['selection']['zmax'],
                           chatter=self.config['logging']['chatter'])

        kw_gtmktime = dict(evfile=self._ft1_file,
                           outfile=self._ft1_filtered_file,
                           scfile=self.config['data']['scfile'],
                           roicut=self.config['selection']['roicut'],
                           filter=self.config['selection']['filter'])

        if not os.path.isfile(self._ft1_file):
            run_gtapp('gtselect',self.logger,kw_gtselect)
            run_gtapp('gtmktime',self.logger,kw_gtmktime)
            os.system('mv %s %s'%(self._ft1_filtered_file,self._ft1_file))
        else:
            self.logger.info('Skipping gtselect')
            
        # Run gtltcube
        kw = dict(evfile=self._ft1_file,
                  scfile=self.config['data']['scfile'],
                  outfile=self._ltcube,
                  zmax=self.config['selection']['zmax'])
        
        if self.config['data']['ltcube'] is not None:
            self._ltcube = self.config['data']['ltcube']
        elif not os.path.isfile(self._ltcube):             
            run_gtapp('gtltcube',self.logger,kw)
        else:
            self.logger.info('Skipping gtltcube')
            
        # Run gtbin
        kw = dict(algorithm='ccube',
                  nxpix=self.npix, nypix=self.npix,
                  binsz=self.config['binning']['binsz'],
                  evfile=self._ft1_file,
                  outfile=self._ccube_file,
                  scfile=self.config['data']['scfile'],
                  xref=float(self.roi.radec[0]),
                  yref=float(self.roi.radec[1]),
                  axisrot=0,
                  proj=self.config['binning']['proj'],
                  ebinalg='LOG',
                  emin=self.config['selection']['emin'],
                  emax=self.config['selection']['emax'],
                  enumbins=self._enumbins,
                  coordsys=self.config['binning']['coordsys'],
                  chatter=self.config['logging']['chatter'])
        
        if not os.path.isfile(self._ccube_file):
            run_gtapp('gtbin',self.logger,kw)            
        else:
            self.logger.info('Skipping gtbin')

        evtype = self.config['selection']['evtype']
            
        if self.config['gtlike']['irfs'] == 'CALDB':
            cmap = self._ccube_file
        else:
            cmap = 'none'
            
        # Run gtexpcube2
        kw = dict(infile=self._ltcube,cmap=cmap,
                  ebinalg='LOG',
                  emin=self.config['selection']['emin'],
                  emax=self.config['selection']['emax'],
                  enumbins=self._enumbins,
                  outfile=self._bexpmap_file, proj='CAR',
                  nxpix=360, nypix=180, binsz=1,
                  xref=0.0,yref=0.0,
                  evtype=evtype,
                  irfs=self.config['gtlike']['irfs'],
                  coordsys=self.config['binning']['coordsys'],
                  chatter=self.config['logging']['chatter'])

        if not os.path.isfile(self._bexpmap_file):
            run_gtapp('gtexpcube2',self.logger,kw)              
        else:
            self.logger.info('Skipping gtexpcube')

        # Run gtsrcmaps
        kw = dict(scfile=self.config['data']['scfile'],
                  expcube=self._ltcube,
                  cmap=self._ccube_file,
                  srcmdl=self._srcmdl_file,
                  bexpmap=self._bexpmap_file,
                  outfile=self._srcmap_file,
                  irfs=self.config['gtlike']['irfs'],
                  evtype=evtype,
#                   rfactor=self.config['rfactor'],
#                   resample=self.config['resample'],
#                   minbinsz=self.config['minbinsz'],
                  chatter=self.config['logging']['chatter'],
                  emapbnds='no' ) 

        if not os.path.isfile(self._srcmap_file):
            run_gtapp('gtsrcmaps',self.logger,kw)             
        else:
            self.logger.info('Skipping gtsrcmaps')

        # Create BinnedObs
        self.logger.info('Creating BinnedObs')
        kw = dict(srcMaps=self._srcmap_file,expCube=self._ltcube,
                  binnedExpMap=self._bexpmap_file,
                  irfs=self.config['gtlike']['irfs'])
        self.logger.info(kw)
        
        self._obs=ba.BinnedObs(**kw)

        # Create BinnedAnalysis
        self.logger.info('Creating BinnedAnalysis')
        self._like = ba.BinnedAnalysis(binnedData=self._obs,
                                       srcModel=self._srcmdl_file,
                                       optimizer='MINUIT')
        
        if self.config['gtlike']['edisp']:
            self.logger.info('Enabling energy dispersion')
            self.like.logLike.set_edisp_flag(True)

        for s in self.config['gtlike']['edisp_disable']: 
            self.logger.info('Disabling energy dispersion for %s'%s)
            self.set_edisp_flag(s,False)
                       
        self.logger.info('Finished setup')

    def generate_model_map(self,model_name=None):
        
        if model_name is None: suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s'%(model_name,self.config['file_suffix'])
        
        outfile = os.path.join(self.config['fileio']['workdir'],'mcube%s.fits'%(suffix))
        
        h = pyfits.open(self._ccube_file)
        
        counts = self.modelCountsMap()        
        hdu_image = pyfits.PrimaryHDU(counts.T,header=h[0].header)
        hdulist = pyfits.HDUList([hdu_image,h['GTI'],h['EBOUNDS']])        
        hdulist.writeto(outfile,clobber=True)
        
    def generate_model(self,model_name=None,outfile=None):
        """Generate a counts model map from an XML model file using
        gtmodel.

        Parameters
        ----------

        model_name : str
        
            Name of the model.  If no name is given it will use 
            the baseline model.

        outfile : str

            Override the name of the output model file.
            
        """

        if model_name is not None:
            model_name = os.path.splitext(model_name)[0]
        
        if model_name is None or model_name == '': srcmdl = self._srcmdl_file
        else: srcmdl = self.get_model_path(model_name)

        if not os.path.isfile(srcmdl):
            raise Exception("Model file does not exist: %s"%srcmdl)

        if model_name is None: suffix = self.config['file_suffix']
        else:
            suffix = '_%s%s'%(model_name,self.config['file_suffix'])
        
        outfile = os.path.join(self.config['fileio']['workdir'],'mcube%s.fits'%(suffix))
        
        # May consider generating a custom source model file
        if not os.path.isfile(outfile):

            kw = dict(srcmaps = self._srcmap_file,
                      srcmdl  = srcmdl,
                      bexpmap = self._bexpmap_file,
                      outfile = outfile,
                      expcube = self._ltcube,
                      irfs    = self.config['gtlike']['irfs'],
                      evtype  = self.config['selection']['evtype'],
                      edisp   = bool(self.config['gtlike']['edisp']),
                      outtype = 'ccube',
                      chatter = self.config['logging']['chatter'])
            
            run_gtapp('gtmodel',self.logger,kw)       
        else:
            self.logger.info('Skipping gtmodel')
            

    def write_xml(self,xmlfile):
        """Write the XML model for this analysis component."""
        
        xmlfile = self.get_model_path(xmlfile)            
        self.logger.info('Writing %s...'%xmlfile)
        self.like.writeXml(xmlfile)

    def get_model_path(self,name):
        """Infer the path to the XML model name."""
        
        name, ext = os.path.splitext(name)
        if not ext: ext = '.xml'
        xmlfile = name + self.config['file_suffix'] + ext

        if os.path.commonprefix([self.config['fileio']['workdir'],xmlfile]) \
                != self.config['fileio']['workdir']:        
            xmlfile = os.path.join(self.config['fileio']['workdir'],xmlfile)

        return xmlfile
