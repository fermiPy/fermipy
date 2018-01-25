# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
from functools import wraps
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

from fermipy import utils
from fermipy import model_utils

evtype_string = {
    4: 'PSF0',
    8: 'PSF1',
    16: 'PSF2',
    32: 'PSF3'
}


def bitmask_to_bits(mask):

    bits = []
    for i in range(32):
        if mask & (2**i):
            bits += [2**i]

    return bits


DEFAULT_SCALE_DICT = {'value': 1000.0,
                      'scale': 1.0, 'min': 0.001, 'max': 1000.0}
DEFAULT_NORM_DICT = {'value': 1E-12, 'scale': 1.0, 'min': 1E-5, 'max': 1000.0}
DEFAULT_INTEGRAL_DICT = {'value': 1E-6,
                         'scale': 1.0, 'min': 1E-5, 'max': 1000.0}
DEFAULT_INDEX_DICT = {'value': 2.0, 'scale': -1.0, 'min': 0.0, 'max': 5.0}

FUNCTION_NORM_PARS = {}
FUNCTION_PAR_NAMES = {}
FUNCTION_DEFAULT_PARS = {
    'PowerLaw': {
        'Index': DEFAULT_INDEX_DICT,
        'Scale': DEFAULT_SCALE_DICT,
        'Prefactor': DEFAULT_NORM_DICT},
    'PowerLaw2': {
        'Index': DEFAULT_INDEX_DICT,
        'LowerLimit': {'value': 100.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
        'UpperLimit': {'value': 100000.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
        'Integral': DEFAULT_INTEGRAL_DICT},
    'BrokenPowerLaw': {
        'Index1': DEFAULT_INDEX_DICT,
        'Index2': DEFAULT_INDEX_DICT,
        'BreakValue': DEFAULT_SCALE_DICT,
        'Prefactor': DEFAULT_NORM_DICT},
    'BrokenPowerLaw2': {
        'Index1': DEFAULT_INDEX_DICT,
        'Index2': DEFAULT_INDEX_DICT,
        'LowerLimit': {'value': 100.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
        'UpperLimit': {'value': 100000.0, 'scale': 1.0, 'min': 20.0, 'max': 1000000.},
        'BreakValue': DEFAULT_SCALE_DICT,
        'Integral': DEFAULT_INTEGRAL_DICT},
    'BPLExpCutoff': {
        'Index1': DEFAULT_INDEX_DICT,
        'Index2': DEFAULT_INDEX_DICT,
        'BreakValue': DEFAULT_SCALE_DICT,
        'Prefactor': DEFAULT_NORM_DICT},
    'SmoothBrokenPowerLaw': {
        'Index1': DEFAULT_INDEX_DICT,
        'Index2': DEFAULT_INDEX_DICT,
        'BreakValue': DEFAULT_SCALE_DICT,
        'Prefactor': DEFAULT_NORM_DICT,
        'Beta': {'value': 0.2, 'scale': 1.0, 'min': 0.01, 'max': 10.0}},
    'PLSuperExpCutoff': {
        'Cutoff': DEFAULT_SCALE_DICT,
        'Index1': {'value': 2.0, 'scale': -1.0, 'min': 0.0, 'max': 5.0},
        'Index2': {'value': 1.0, 'scale': 1.0, 'min': 0.0, 'max': 2.0},
        'Prefactor': DEFAULT_NORM_DICT,
    },
    'LogParabola': {
        'norm': DEFAULT_NORM_DICT,
        'alpha': {'value': 2.0, 'scale': 1.0, 'min': -5.0, 'max': 5.0},
        'beta': {'value': 0.0, 'scale': 1.0, 'min': -2.0, 'max': 2.0},
        'Eb': DEFAULT_SCALE_DICT},
    'SpatialMap': {
        'Prefactor': {'value': 1.0, 'scale': 1.0, 'min': 1.0, 'max': 1.0}},
    'ConstantValue': {
        'Normalization': {'value': 1.0, 'scale': 1.0, 'min': 1E-5, 'max': 1000.0}},
    'FileFunction': {
        'Normalization': {'value': 1.0, 'scale': 1.0, 'min': 1E-5, 'max': 1000.0}},
    'Gaussian': {
        'Mean': {'value': 1000.0, 'scale': 1.0, 'min': 1E-5, 'max': 1E5},
        'Sigma': {'value': 100.0, 'scale': 1.0, 'min': 10., 'max': 1E5},
        'Prefactor': DEFAULT_NORM_DICT},
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

        pars = FUNCTION_DEFAULT_PARS.setdefault(fname, {})
        par_names = FUNCTION_PAR_NAMES.setdefault(fname, [])

        if 'EblAtten' in fname and fname[len('EblAtten::'):] in FUNCTION_DEFAULT_PARS:
            pars.update(FUNCTION_DEFAULT_PARS[fname[len('EblAtten::'):]])

        fn = funcFactory.create(fname)
        try:
            FUNCTION_NORM_PARS[fname] = fn.normPar().getName()
        except Exception:
            FUNCTION_NORM_PARS[fname] = None

        params = pyLike.ParameterVector()
        fn.getParams(params)

        for i, p in enumerate(params):

            pname = p.getName()
            par_names += [pname]

            if pname == 'Scale':
                pars.setdefault(pname, DEFAULT_SCALE_DICT)
            elif pname == 'Prefactor':
                pars.setdefault(pname, DEFAULT_NORM_DICT)
            else:
                pars.setdefault(pname, {})

            bounds = p.getBounds()
            par_dict = dict(name=pname,
                            value=p.getValue(),
                            min=bounds[0],
                            max=bounds[1],
                            scale=1.0,
                            free=False)

            par_dict.update(copy.deepcopy(pars[pname]))
            par_dict['name'] = pname
            pars[pname] = par_dict


def get_function_par_names(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()

    if not function_type in FUNCTION_PAR_NAMES.keys():
        raise Exception('Invalid Function Type: %s' % function_type)

    return copy.deepcopy(FUNCTION_PAR_NAMES[function_type])


def get_function_norm_par_name(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()

    return FUNCTION_NORM_PARS[function_type]


def get_function_defaults(function_type):

    if not FUNCTION_NORM_PARS:
        init_function_pars()

    return copy.deepcopy(FUNCTION_DEFAULT_PARS[function_type])


def create_spectrum_from_dict(spectrum_type, spectral_pars, fn=None):
    """Create a Function object from a parameter dictionary.

    Parameters
    ----------
    spectrum_type : str
       String identifying the spectrum type (e.g. PowerLaw).

    spectral_pars : dict
       Dictionary of spectral parameters.

    """

    if fn is None:
        fn = pyLike.SourceFactory_funcFactory().create(str(spectrum_type))

    for k, v in spectral_pars.items():

        v.setdefault('scale', 1.0)
        v.setdefault('min', v['value'] * 1E-3)
        v.setdefault('max', v['value'] * 1E3)

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


def gtlike_spectrum_to_vectors(spectrum):
    """ Convert a pyLikelihood object to a python dictionary which can
        be easily saved to a file."""

    o = {'param_names': np.zeros(10, dtype='S32'),
         'param_values': np.empty(10, dtype=float) * np.nan,
         'param_errors': np.empty(10, dtype=float) * np.nan,
         }

    parameters = pyLike.ParameterVector()
    spectrum.getParams(parameters)
    for i, p in enumerate(parameters):
        o['param_names'][i] = p.getName()
        o['param_values'][i] = p.getTrueValue()
        perr = abs(p.error() * p.getScale()) if p.isFree() else np.nan
        o['param_errors'][i] = perr

    return o


def get_function_pars_dict(fn):

    pars = get_function_pars(fn)
    pars_dict = {p['name']: p for p in pars}
    return pars_dict


def get_function_pars(fn):
    """Extract the parameters of a pyLikelihood function object
    (value, scale, bounds).

    Parameters
    ----------

    fn : pyLikelihood.Function

    Returns
    -------

    pars : list

    """

    pars = []
    par_names = pyLike.StringVector()
    fn.getParamNames(par_names)

    for pname in par_names:

        par = fn.getParam(pname)
        bounds = par.getBounds()
        perr = par.error() if par.isFree() else np.nan
        pars += [dict(name=pname,
                      value=par.getValue(),
                      error=perr,
                      min=bounds[0],
                      max=bounds[1],
                      free=par.isFree(),
                      scale=par.getScale())]

    return pars


def get_params_dict(like):

    params = get_params(like)

    params_dict = {}
    for p in params:
        params_dict.setdefault(p['src_name'], [])
        params_dict[p['src_name']] += [p]

    return params_dict


def get_params(like):

    params = []
    for src_name in like.sourceNames():

        src = like[src_name].src
        spars, ppars = get_source_pars(src)

        for p in spars:
            p['src_name'] = src_name
            params += [p]

        for p in ppars:
            p['src_name'] = src_name
            params += [p]

    return params


def get_priors(like):
    """Extract priors from a likelihood object."""

    npar = len(like.params())

    vals = np.ones(npar)
    errs = np.ones(npar)
    has_prior = np.array([False] * npar)

    for i, p in enumerate(like.params()):

        prior = like[i].log_prior()

        if prior is None:
            continue

        par_names = pyLike.StringVector()
        prior.getParamNames(par_names)

        if not 'Mean' in par_names:
            raise Exception('Failed to find Mean in prior parameters.')

        if not 'Sigma' in par_names:
            raise Exception('Failed to find Sigma in prior parameters.')

        for t in par_names:

            if t == 'Mean':
                vals[i] = prior.parameter(t).getValue()

            if t == 'Sigma':
                errs[i] = prior.parameter(t).getValue()

        has_prior[i] = True

    return vals, errs, has_prior


def get_source_pars(src):
    """Extract the parameters associated with a pyLikelihood Source object.

    """

    fnmap = src.getSrcFuncs()

    keys = fnmap.keys()

    if 'Position' in keys:
        ppars = get_function_pars(src.getSrcFuncs()[str('Position')])
    elif 'SpatialDist' in keys:
        ppars = get_function_pars(src.getSrcFuncs()[str('SpatialDist')])
    else:
        raise Exception('Failed to extract spatial parameters.')

    fn = src.getSrcFuncs()[str('Spectrum')]
    spars = get_function_pars(fn)

    for i, p in enumerate(ppars):
        ppars[i]['is_norm'] = False

    for i, p in enumerate(spars):

        if fn.normPar().getName() == p['name']:
            spars[i]['is_norm'] = True
        else:
            spars[i]['is_norm'] = False

    return spars, ppars


def savefreestate(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        free_params = self.get_free_param_vector()
        o = func(self, *args, **kwargs)
        self.set_free_param_vector(free_params)
        return o
    return wrapper


def savestate(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        saved_state = LikelihoodState(self.like)
        o = func(self, *args, **kwargs)
        saved_state.restore()
        return o
    return wrapper


class FreeParameterState(object):

    def __init__(self, gta):
        self._gta = gta
        self._free = gta.get_free_param_vector()

    def restore(self):
        self._gta.set_free_param_vector(self._free)


class SourceMapState(object):

    def __init__(self, like, names):

        self._srcmaps = {}
        self._like = like

        for name in names:
            self._srcmaps[name] = []
            for c in self._like.components:
                self._srcmaps[name] += [c.logLike.sourceMap(str(name)).model()]

    def restore(self):
        for name in self._srcmaps.keys():
            for i, c in enumerate(self._like.components):
                c.logLike.setSourceMapImage(str(name),
                                            self._srcmaps[name][i])


class SummedLikelihood(SummedLikelihood.SummedLikelihood):

    def nFreeParams(self):
        """Count the number of free parameters in the active model."""
        nF = 0
        pars = self.params()
        for par in pars:
            if par.isFree():
                nF += 1
        return nF

    def optimize(self, verbosity=3, tol=None, optimizer=None, optObject=None):
        self._syncParams()
        if optimizer is None:
            optimizer = self.optimizer
        if tol is None:
            tol = self.tol
        if optObject is None:
            optFactory = pyLike.OptimizerFactory_instance()
            myOpt = optFactory.create(optimizer, self.logLike)
        else:
            myOpt = optObject
        myOpt.find_min_only(verbosity, tol, self.tolType)
        self.saveBestFit()

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
            comp.scaleSource(srcName, 1E-10)
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
                    print("** Iteration :", Niter)
                Niter += 1
        else:
            if approx:
                try:
                    self._renorm()
                except ZeroDivisionError:
                    pass
        self.syncSrcParams()
        logLike0 = max(-self(), logLike0)
        Ts_value = 2 * (logLike1 - logLike0)
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
            self.renormFactor = 1. + deficit / freeNpred
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
                newValue = oldValue * self.renormFactor
                # ensure new value is within parameter bounds
                xmin, xmax = parameter.getBounds()
                if xmin <= newValue and newValue <= xmax:
                    parameter.setValue(newValue)


class BinnedAnalysis(BinnedAnalysis.BinnedAnalysis):

    def __init__(self, binnedData, srcModel=None, optimizer='Drmngb',
                 use_bl2=False, verbosity=0, psfcorr=True, convolve=True,
                 resample=True, resamp_fact=2, minbinsz=0.1, wmap=None):
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
            if wmap is None or wmap == "none":
                self.logLike = pyLike.BinnedLikelihood(binnedData.countsMap,
                                                       binnedData.observation,
                                                       binnedData.srcMaps,
                                                       True, psfcorr, convolve,
                                                       resample,
                                                       resamp_fact,
                                                       minbinsz)
                self._wmap = None
            else:
                self._wmap = pyLike.WcsMapLibrary.instance().wcsmap(wmap, "SKYMAP")
                self._wmap.setInterpolation(False)
                self._wmap.setExtrapolation(True)
                self.logLike = pyLike.BinnedLikelihood(binnedData.countsMap,
                                                       self._wmap,
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
        self.e_vals = np.sqrt(self.energies[:-1] * self.energies[1:])
        self.nobs = self.logLike.countsSpectrum()
        self.sourceFitPlots = []
        self.sourceFitResids = []

    def scaleSource(self, srcName, scale):
        src = self.logLike.getSource(srcName)
        old_scale = src.spectrum().normPar().getScale()
        src.spectrum().normPar().setScale(old_scale * scale)
        self.logLike.syncParams()

    def Ts2(self, srcName, reoptimize=False, approx=True,
            tol=None, MaxIterations=10, verbosity=0):
        """Computes the TS value for a source indicated by "srcName."

        If "reoptimize=True" is selected this function will reoptimize
        the model up to "MaxIterations" given the tolerance "tol"
        (default is the tolerance selected for the overall fit).  If
        "appox=True" is selected (the default) it will renormalize the
        model (see _renorm).
        """

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
        self.scaleSource(srcName, 1E-10)
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
                except RuntimeError as e:
                    print(e)
                if verbosity > 0:
                    print("** Iteration :", Niter)
                Niter += 1
        else:
            if approx:
                try:
                    self._renorm()
                except ZeroDivisionError:
                    pass
        self.logLike.syncParams()
        logLike0 = max(self.logLike.value(), logLike0)
        Ts_value = 2 * (logLike1 - logLike0)
        self.scaleSource(srcName, 1E10)

        self.logLike.setFreeParamValues(freeParams)
        self.model = SourceModel(self.logLike)
        for src in source_attributes:
            self.model[src].__dict__.update(source_attributes[src])
        saved_state.restore()
        self.logLike.value()
        return Ts_value

    def _isDiffuseOrNearby(self, srcName):
        if (self[srcName].src.getType() in ['Diffuse','Composite'] or 
            self._ts_src.getType() in ['Diffuse','Composite']):
            return True
        elif self._separation(self._ts_src, self[srcName].src) < self.maxdist:
            return True
        return False
