# pylikelihood

import glob
import numpy as np
import healpy as hp

import pyLikelihood as pyLike

import astropy.io.fits as pyfits

from SrcModel import SourceModel
from AnalysisBase import AnalysisBase, _quotefn, _null_file, num
import pyLikelihood as pyLike
from LikelihoodState import LikelihoodState
import pyIrfLoader

pyIrfLoader.Loader_go()

_funcFactory = pyLike.SourceFactory_funcFactory()

import BinnedAnalysis 
import UnbinnedAnalysis 
import SummedLikelihood

from fermipy.utils import edge_to_center
from fermipy.utils import edge_to_width

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


    
    
class SummedLikelihood(SummedLikelihood.SummedLikelihood):

    def Ts2(self, srcName, reoptimize=False, approx=True,
           tol=None, MaxIterations=10, verbosity=0):

        if verbosity > 0:
            print "*** Start Ts_dl ***"
        source_attributes = self.components[0].getExtraSourceAttributes()
        self.syncSrcParams()
        freeParams = pyLike.DoubleVector()
        self.components[0].logLike.getFreeParamValues(freeParams)
        logLike1 = -self()
        for comp in self.components:            
            comp.scaleSource(srcName,1E-10)
            comp._ts_src = comp.logLike.getSource(srcName)
            free_flag = comp._ts_src.spectrum().normPar().isFree()
            #comp._ts_src.spectrum().normPar().setFree(False)
            
        logLike0 = -self()
        if tol is None:
            tol = self.tol
        if reoptimize:
            if verbosity > 0:
                print "** Do reoptimize"
            optFactory = pyLike.OptimizerFactory_instance()
            myOpt = optFactory.create(self.optimizer, self.composite)
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
        self.syncSrcParams()
        logLike0 = max(-self(), logLike0)
        Ts_value = 2*(logLike1 - logLike0)
        for comp in self.components:
            comp.scaleSource(srcName,1E10)
            #comp._ts_src.spectrum().normPar().setFree(free_flag)
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
        self.energies = num.array(self.logLike.energies())
        self.e_vals = num.sqrt(self.energies[:-1]*self.energies[1:])
        self.nobs = self.logLike.countsSpectrum()
        self.sourceFitPlots = []
        self.sourceFitResids  = []

    def scaleSource(self,srcName,scale):
        src = self.logLike.getSource(srcName)
        old_scale = src.spectrum().normPar().getScale()
        src.spectrum().normPar().setScale(old_scale*scale)
        
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
