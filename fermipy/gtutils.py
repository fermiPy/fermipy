# pylikelihood

from SrcModel import SourceModel
from AnalysisBase import AnalysisBase, _quotefn, _null_file, num
import pyLikelihood as pyLike

_funcFactory = pyLike.SourceFactory_funcFactory()

import BinnedAnalysis as ba
import UnbinnedAnalysis as uba

class BinnedAnalysis(ba.BinnedAnalysis):

    def __init__(self, binnedData, srcModel=None, optimizer='Drmngb',
                 use_bl2=False, verbosity=0, psfcorr=True,resamp_fact=2,
                 minbinsz=0.1):
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
                                                    True, psfcorr, True, True,
                                                    resamp_fact,
                                                    minbinsz)
        else:
            self.logLike = pyLike.BinnedLikelihood(binnedData.countsMap,
                                                   binnedData.observation,
                                                   binnedData.srcMaps,
                                                   True, psfcorr, True, True,
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
