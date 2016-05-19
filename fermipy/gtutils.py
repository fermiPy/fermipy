# pylikelihood
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import copy

import numpy as np
import pyLikelihood as pyLike
from SrcModel import SourceModel
from AnalysisBase import AnalysisBase
from LikelihoodState import LikelihoodState
import pyIrfLoader

pyIrfLoader.Loader_go()

_funcFactory = pyLike.SourceFactory_funcFactory()

import BinnedAnalysis 
import SummedLikelihood

import fermipy.utils as utils

evtype_string = {
    4 : 'PSF0',
    8 : 'PSF1',
    16 : 'PSF2',
    32 : 'PSF3'
    }

def bitmask_to_bits(mask):

    bits = []    
    for i in range(32):
        if mask&(2**i): bits += [2**i]

    return bits

DEFAULT_SCALE_DICT =  {'value': 1000.0, 'scale' : None, 'min': 0.001, 'max': 1000.0}
DEFAULT_NORM_DICT = {'value': 1E-12, 'scale' : None, 'min': 1E-5, 'max': 100.0}
DEFAULT_INTEGRAL_DICT = {'value': 1E-6, 'scale' : None, 'min': 1E-5, 'max': 100.0}
DEFAULT_INDEX_DICT = {'value': 2.0, 'scale': -1.0, 'min': 0.0, 'max': 5.0 }

FUNCTION_NORM_PARS = {}
FUNCTION_PAR_NAMES = {}
FUNCTION_DEFAULT_PARS = {
    'PowerLaw': { 'Index': DEFAULT_INDEX_DICT,
                  'Scale': DEFAULT_SCALE_DICT,
                  'Prefactor' : DEFAULT_NORM_DICT },
    'PowerLaw2': { 'Index': DEFAULT_INDEX_DICT,
                   'LowerLimit': {'value': 100.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
                   'UpperLimit': {'value': 100000.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
                   'Integral' : DEFAULT_INTEGRAL_DICT },
    'BrokenPowerLaw': { 'Index1': DEFAULT_INDEX_DICT,
                        'Index2': DEFAULT_INDEX_DICT,
                        'BreakValue' : DEFAULT_SCALE_DICT,
                        'Prefactor' : DEFAULT_NORM_DICT },
    'BrokenPowerLaw2': { 'Index1': DEFAULT_INDEX_DICT,
                         'Index2': DEFAULT_INDEX_DICT,
                         'LowerLimit': {'value': 100.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
                         'UpperLimit': {'value': 100000.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
                         'BreakValue' : DEFAULT_SCALE_DICT,
                         'Integral' : DEFAULT_INTEGRAL_DICT },
    'BPLExpCutoff': { 'Index1': DEFAULT_INDEX_DICT,
                      'Index2': DEFAULT_INDEX_DICT,
                      'BreakValue' : DEFAULT_SCALE_DICT,
                      'Prefactor' : DEFAULT_NORM_DICT },
    'SmoothBrokenPowerLaw': { 'Index1': DEFAULT_INDEX_DICT,
                              'Index2': DEFAULT_INDEX_DICT,
                              'BreakValue' : DEFAULT_SCALE_DICT,
                              'Prefactor' : DEFAULT_NORM_DICT,
                              'Beta' : {'value': 0.2, 'scale': 1.0, 'min': 0.01, 'max': 10.0} },
    'PLSuperExpCutoff' : { 'Cutoff': DEFAULT_SCALE_DICT,
                           'Index1': {'value': 2.0, 'scale': -1.0, 'min': 0.0, 'max': 5.0},
                           'Index2': {'value': 1.0, 'scale': 1.0, 'min': 0.0, 'max': 2.0},
                           'Prefactor' : DEFAULT_NORM_DICT,
                           },
    'LogParabola' : {'norm' : DEFAULT_NORM_DICT,
                     'alpha': {'value': 2.0, 'scale': 1.0, 'min': -5.0, 'max': 5.0},
                     'beta' : {'value': 0.0, 'scale': 1.0, 'min': -10.0, 'max': 10.0},
                     'Eb' : DEFAULT_SCALE_DICT },
    'ConstantValue' : {'norm' : {'value': 1.0, 'scale' : 1.0, 'min': 1E-5, 'max': 100.0} }

    }

def init_function_pars():
    
    global FUNCTION_PAR_NAMES
    global FUNCTION_NORM_PARS
    global FUNCTION_DEFAULT_PARS
    
    FUNCTION_PAR_NAMES = {}
    FUNCTION_NORM_PARS = {}
    
    funcFactory = pyLike.SourceFactory_funcFactory()
    
    names = pyLike.StringVector()
    funcFactory.getFunctionNames(names)
        
    for fname in names:

        FUNCTION_DEFAULT_PARS.setdefault(fname,{})
        FUNCTION_PAR_NAMES.setdefault(fname,[])
        
        fn = funcFactory.create(fname)        
        try:
            FUNCTION_NORM_PARS[fname] = fn.normPar().getName()
        except Exception as e:
            FUNCTION_NORM_PARS[fname] = 'Prefactor'

        params = pyLike.ParameterVector()
        fn.getParams(params)

        for i, p in enumerate(params):

            pname = p.getName()
            FUNCTION_PAR_NAMES[fname] += [pname]
            
            if pname == 'Scale':
                FUNCTION_DEFAULT_PARS[fname].setdefault(pname,DEFAULT_SCALE_DICT)
            elif pname == 'Prefactor':
                FUNCTION_DEFAULT_PARS[fname].setdefault(pname,DEFAULT_NORM_DICT)
            else:
                FUNCTION_DEFAULT_PARS[fname].setdefault(pname,{})

            bounds = p.getBounds()
            par_dict = dict(name = pname,
                            value = p.getValue(),
                            min = bounds[0],
                            max = bounds[1],
                            free = False)

            par_dict.update(copy.deepcopy(FUNCTION_DEFAULT_PARS[fname][pname]))
            par_dict['name'] = pname            
            FUNCTION_DEFAULT_PARS[fname][pname] = par_dict
        

def get_function_par_names(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()

    if not function_type in FUNCTION_PAR_NAMES.keys():
        raise Exception('Invalid Function Type: %s'%function_type)
    
    return copy.deepcopy(FUNCTION_PAR_NAMES[function_type])
            
def get_function_norm_par_name(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()
            
    return FUNCTION_NORM_PARS[function_type]

def get_function_pars_dict(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()
            
    return copy.deepcopy(FUNCTION_DEFAULT_PARS[function_type])

def make_parameter_dict(pdict, fixed_par=False, rescale=True):
    """
    Prepare a parameter dictionary.  This function will automatically
    set the parameter scale and bounds if they are not defined.
    Bounds are also adjusted to ensure that they encompass the
    parameter value.
    """
    o = copy.deepcopy(pdict)

    if 'scale' not in o or o['scale'] is None:

        if rescale:        
            value, scale = utils.scale_parameter(o['value'])
        else:
            value, scale = o['value'], 1.0
            
        o['value'] = value
        o['scale'] = scale
        if 'error' in o:
            o['error'] /= np.abs(scale)

    if 'min' not in o:
        o['min'] = o['value']*1E-3

    if 'max' not in o:
        o['max'] = o['value']*1E3
        
    if fixed_par:
        o['min'] = o['value']
        o['max'] = o['value']

    if float(o['min']) > float(o['value']):
        o['min'] = o['value']

    if float(o['max']) < float(o['value']):
        o['max'] = o['value']

#    for k, v in o.items():
#        o[k] = str(v)

    return o


def create_spectral_pars_dict(spectrum_type,spectral_pars=None):

    pars_dict = get_function_pars_dict(spectrum_type)

    if spectral_pars is None:
        spectral_pars = {}
    else:
        spectral_pars = copy.deepcopy(spectral_pars)

    for k, v in spectral_pars.items():

        if not k in pars_dict:
            continue
        
        if not isinstance(v,dict):
            spectral_pars[k] = {'name' : k, 'value' : v}
    
    pars_dict = utils.merge_dict(pars_dict,spectral_pars)

    for k, v in pars_dict.items():
        pars_dict[k] = make_parameter_dict(v)

    return pars_dict

def create_spectrum_from_dict(spectrum_type,spectral_pars=None):
    """Create a Function object from a parameter dictionary.

    Parameters
    ----------
    spectrum_type : str
       String identifying the spectrum type (e.g. PowerLaw).

    spectral_pars : dict
       Dictionary of spectral parameters.

    """

    pars = create_spectral_pars_dict(spectrum_type,spectral_pars)
    fn = pyLike.SourceFactory_funcFactory().create(str(spectrum_type))

    for k, v in pars.items():

        v = make_parameter_dict(v)
        
        par = fn.getParam(str(k))

        vmin = min(float(v['value']), float(v['min']))
        vmax = max(float(v['value']), float(v['max']))
        
        par.setValue(float(v['value']))
        par.setBounds(vmin, vmax)
        par.setScale(float(v['scale']))

        if 'free' in v and int(v['free']) != 0:
            par.setFree(True)
        else:
            par.setFree(False)
        fn.setParam(par)

    return fn


def get_spatial_type(spatial_model):
    """Translate a spatial model string to a spatial type."""
    
    if spatial_model in ['SkyDirFunction', 'PointSource',
                         'Gaussian', 'PSFSource']:
        return 'SkyDirFunction'
    elif spatial_model in ['GaussianSource', 'DiskSource', 'SpatialMap']:
        return 'SpatialMap'
    elif spatial_model in ['RadialGaussian','RadialDisk']:
        if hasattr(pyLike,'RadialGaussian'):
            return spatial_model
        else:
            return 'SpatialMap'
    else:
        return spatial_model

    
def get_source_type(spatial_type):
    """Translate a spatial type string to a source type."""
    
    if spatial_type == 'SkyDirFunction':
        return 'PointSource'
    else:
        return 'DiffuseSource'

    
def gtlike_spectrum_to_dict(spectrum):
    """ Convert a pyLikelihood object to a python dictionary which can
        be easily saved to a file."""
    parameters = pyLike.ParameterVector()
    spectrum.getParams(parameters)
    d = dict(spectrum_type=spectrum.genericName())
    for p in parameters:

        pname = p.getName()
        pval = p.getTrueValue()
        perr = abs(p.error() * p.getScale()) if p.isFree() else np.nan
        d[pname] = np.array([pval, perr])

        if d['spectrum_type'] == 'FileFunction':
            ff = pyLike.FileFunction_cast(spectrum)
            d['file'] = ff.filename()
    return d

def get_pars_dict_from_source(src):

    pars_dict = {}

    par_names = pyLike.StringVector()
    src.spectrum().getParamNames(par_names)

    for pname in par_names:

        par = src.spectrum().getParam(pname)
        bounds = par.getBounds()
        perr = par.error() if par.isFree() else np.nan
        pars_dict[pname] = dict(name = pname,
                                value = par.getValue(),
                                error = perr,
                                min = bounds[0],
                                max = bounds[1],
                                free = par.isFree(),
                                scale = par.getScale())

    return pars_dict

def cast_pars_dict(pars_dict):

    o = {}

    for pname, pdict in pars_dict.items():

        o[pname] = {}
        
        for k,v in pdict.items():

            if k == 'free':
                o[pname][k] = bool(v)
            elif k == 'name':
                o[pname][k] = v
            else:
                o[pname][k] = float(v)

    return o

class SummedLikelihood(SummedLikelihood.SummedLikelihood):

    def Ts2(self, srcName, reoptimize=False, approx=True,
           tol=None, MaxIterations=10, verbosity=0):

        srcName = str(srcName)
        
        if verbosity > 0:
            print("*** Start Ts_dl ***")
        source_attributes = self.components[0].getExtraSourceAttributes()
        self.syncSrcParams()
        freeParams = pyLike.DoubleVector()
        self.components[0].logLike.getFreeParamValues(freeParams)
        logLike1 = -self()
        for comp in self.components:            
            comp.scaleSource(srcName,1E-10)
            comp._ts_src = comp.logLike.getSource(srcName)
            free_flag = comp._ts_src.spectrum().normPar().isFree()

            if reoptimize:
                comp._ts_src.spectrum().normPar().setFree(False)
                self.syncSrcParams()

        logLike0 = -self()
        if tol is None:
            tol = self.tol
        if reoptimize:
            if verbosity > 0:
                print("** Do reoptimize")
            optFactory = pyLike.OptimizerFactory_instance()
            myOpt = optFactory.create(self.optimizer, self.composite)
            Niter = 1
            while Niter <= MaxIterations:
                try:
                    myOpt.find_min(0, tol)
                    break
                except RuntimeError as e:
                    print(e)
                if verbosity > 0:
                    print("** Iteration :",Niter)
                Niter += 1
        else:
            if approx:
                try:
                    self._renorm()
                except ZeroDivisionError:
                    pass
        self.syncSrcParams()
        logLike0 = max(-self(), logLike0)
        Ts_value = 2*(logLike1 - logLike0)
        for comp in self.components:
            comp.scaleSource(srcName, 1E10)
            if reoptimize:
                comp._ts_src.spectrum().normPar().setFree(free_flag)
            self.syncSrcParams(srcName)
            comp.logLike.setFreeParamValues(freeParams)
            comp.model = SourceModel(comp.logLike)
            for src in source_attributes:
                comp.model[src].__dict__.update(source_attributes[src])
        self.model = self.components[0].model

        return Ts_value

    def _renorm(self, factor=None):
        
        if factor is None:
            freeNpred, totalNpred = self._npredValues()
            deficit = self.total_nobs() - totalNpred
            self.renormFactor = 1. + deficit/freeNpred
        else:
            self.renormFactor = factor
        if self.renormFactor < 1:
            self.renormFactor = 1
        srcNames = self.sourceNames()
        for src in srcNames:

            if src == self.components[0]._ts_src.getName():
                continue
            
            parameter = self.normPar(src)
            if (parameter.isFree() and 
                self.components[0]._isDiffuseOrNearby(src)):
                oldValue = parameter.getValue()
                newValue = oldValue*self.renormFactor
                # ensure new value is within parameter bounds
                xmin, xmax = parameter.getBounds()
                if xmin <= newValue and newValue <= xmax:
                    parameter.setValue(newValue)
    
class BinnedAnalysis(BinnedAnalysis.BinnedAnalysis):

    def __init__(self, binnedData, srcModel=None, optimizer='Drmngb',
                 use_bl2=False, verbosity=0, psfcorr=True,convolve=True,
                 resample=True,resamp_fact=2,minbinsz=0.1):
        AnalysisBase.__init__(self)
        if srcModel is None:
            srcModel, optimizer = self._srcDialog()
        self.binnedData = binnedData
        self.srcModel = srcModel
        self.optimizer = optimizer
        if use_bl2:
            self.logLike = pyLike.BinnedLikelihood2(binnedData.countsMap,
                                                    binnedData.observation,
                                                    binnedData.srcMaps,
                                                    True, psfcorr, convolve,
                                                    resample,
                                                    resamp_fact,
                                                    minbinsz)
        else:
            self.logLike = pyLike.BinnedLikelihood(binnedData.countsMap,
                                                   binnedData.observation,
                                                   binnedData.srcMaps,
                                                   True, psfcorr, convolve,
                                                   resample,
                                                   resamp_fact,
                                                   minbinsz)
        self.verbosity = verbosity
        self.logLike.initOutputStreams()
        self.logLike.readXml(srcModel, _funcFactory, False, True, False)
        self.model = SourceModel(self.logLike, srcModel)
        self.energies = np.array(self.logLike.energies())
        self.e_vals = np.sqrt(self.energies[:-1]*self.energies[1:])
        self.nobs = self.logLike.countsSpectrum()
        self.sourceFitPlots = []
        self.sourceFitResids  = []
        
    def scaleSource(self,srcName,scale):
        src = self.logLike.getSource(srcName)
        old_scale = src.spectrum().normPar().getScale()
        src.spectrum().normPar().setScale(old_scale*scale)
        self.logLike.syncParams()
        
    def Ts2(self, srcName, reoptimize=False, approx=True,
            tol=None, MaxIterations=10, verbosity=0):

        '''Computes the TS value for a source indicated by "srcName."
        If "reoptimize=True" is selected this function will reoptimize
        the model up to "MaxIterations" given the tolerance "tol"
        (default is the tolerance selected for the overall fit).  If
        "appox=True" is selected (the default) it will renormalize the
        model (see _renorm).'''
        
        saved_state = LikelihoodState(self)
        if verbosity > 0:
            print("*** Start Ts_dl ***")
        source_attributes = self.getExtraSourceAttributes()
        self.logLike.syncParams()
        src = self.logLike.getSource(srcName)
        self._ts_src = src
        freeParams = pyLike.DoubleVector()
        self.logLike.getFreeParamValues(freeParams)
        logLike1 = self.logLike.value()
        self.scaleSource(srcName,1E-10)
        logLike0 = self.logLike.value()
        if tol is None:
            tol = self.tol
        if reoptimize:
            if verbosity > 0:
                print("** Do reoptimize")
            optFactory = pyLike.OptimizerFactory_instance()
            myOpt = optFactory.create(self.optimizer, self.logLike)
            Niter = 1
            while Niter <= MaxIterations:
                try:
                    myOpt.find_min(0, tol)
                    break
                except RuntimeError,e:
                    print(e)
                if verbosity > 0:
                    print("** Iteration :",Niter)
                Niter += 1
        else:
            if approx:
                try:
                    self._renorm()
                except ZeroDivisionError:
                    pass
        self.logLike.syncParams()
        logLike0 = max(self.logLike.value(), logLike0)
        Ts_value = 2*(logLike1 - logLike0)
        self.scaleSource(srcName,1E10)
        
        self.logLike.setFreeParamValues(freeParams)
        self.model = SourceModel(self.logLike)
        for src in source_attributes:
            self.model[src].__dict__.update(source_attributes[src])
        saved_state.restore()
        self.logLike.value()
        return Ts_value

    def _isDiffuseOrNearby(self, srcName):
        if (self[srcName].src.getType() == 'Diffuse' or 
            self._ts_src.getType() == 'Diffuse'):
            return True
        elif self._separation(self._ts_src, self[srcName].src) < self.maxdist:
            return True
        return False
