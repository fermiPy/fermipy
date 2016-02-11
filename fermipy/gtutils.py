# pylikelihood

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
                     'alpha': {'value': 0.0, 'scale': 1.0, 'min': -5.0, 'max': 5.0},
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
                            free = '0')

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

class SummedLikelihood(SummedLikelihood.SummedLikelihood):

    def Ts2(self, srcName, reoptimize=False, approx=True,
           tol=None, MaxIterations=10, verbosity=0):

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
                    print "** Iteration :",Niter
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
#            print src, parameter.getName(), parameter.isFree(), self.components[0]._isDiffuseOrNearby(src)
            
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
            print "*** Start Ts_dl ***"
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
                print "** Do reoptimize"
            optFactory = pyLike.OptimizerFactory_instance()
            myOpt = optFactory.create(self.optimizer, self.logLike)
            Niter = 1
            while Niter <= MaxIterations:
                try:
                    myOpt.find_min(0, tol)
                    break
                except RuntimeError,e:
                    print e
                if verbosity > 0:
                    print "** Iteration :",Niter
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
