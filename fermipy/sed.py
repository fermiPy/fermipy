#!/usr/bin/env python
#

# Description
"""
Utilities for dealing with SEDs

Many parts of this code are taken from dsphs/like/lnlfn.py by 
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""

import copy
import logging

import numpy as np
from scipy.interpolate import UnivariateSpline, splrep, splev
import scipy.optimize as opt
import scipy.special as spf
from scipy.integrate import quad
import scipy
import astropy.io.fits as pf
from astropy.coordinates import SkyCoord

import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
from fermipy.utils import read_energy_bounds, read_spectral_data
from fermipy.fits_utils import read_map_from_fits
from fermipy.logger import Logger
from fermipy.logger import logLevel
from fermipy.sourcefind import find_peaks, refine_peak

from LikelihoodState import LikelihoodState

# Some useful functions


def alphaToDeltaLogLike_1DOF(alpha):
    """return the delta log-likelihood corresponding to a particular
    C.L. of (1-alpha)%

    """
    dlnl = pow(np.sqrt(2.)*spf.erfinv(1-2*alpha),2.)/2.  
    return dlnl


FluxTypes = ['NORM','FLUX','EFLUX','NPRED','DIF_FLUX','DIF_EFLUX']

PAR_NAMES = {"PowerLaw":["Prefactor","Index"],
             "LogParabola":["norm","alpha","beta"],
             "PLExpCutoff":["Prefactor","Index1","Cutoff"]}

#class SEDGenerator(fermipy.config.Configurable):
class SEDGenerator(object):
    """Mixin class which provides SED functionality to
    `~fermipy.gtanalysis.GTAnalysis`."""
    
    def sed(self, name, profile=True, energies=None, **kwargs):
        """Generate a spectral energy distribution (SED) for a source.  This
        function will fit the normalization of the source in each
        energy bin.  By default the SED will be generated with the
        analysis energy bins but a custom binning can be defined with
        the ``energies`` parameter.  

        Parameters
        ----------
        name : str
            Source name.

        prefix : str
           Optional string that will be prepended to all output files
           (FITS and rendered images).
            
        profile : bool
            Profile the likelihood in each energy bin.

        energies : `~numpy.ndarray`
            Sequence of energies in log10(E/MeV) defining the edges of
            the energy bins.  If this argument is None then the
            analysis energy bins will be used.  The energies in this
            sequence must align with the bin edges of the underyling
            analysis instance.

        bin_index : float
            Spectral index that will be use when fitting the energy
            distribution within an energy bin.

        use_local_index : bool
            Use a power-law approximation to the shape of the global
            spectrum in each bin.  If this is false then a constant
            index set to `bin_index` will be used.

        fix_background : bool
            Fix background components when fitting the flux
            normalization in each energy bin.  If fix_background=False
            then all background parameters that are currently free in
            the fit will be profiled.  By default fix_background=True.

        ul_confidence : float
            Set the confidence level that will be used for the
            calculation of flux upper limits in each energy bin.

        cov_scale : float
            Scaling factor that will be applied when setting the
            gaussian prior on the normalization of free background
            sources.  If this parameter is None then no gaussian prior
            will be applied.

        Returns
        -------
        sed : dict 
           Dictionary containing output of the SED analysis.  This
           dictionary is also saved to the 'sed' dictionary of the
           `~fermipy.roi_model.Source` instance.

        """

        name = self.roi.get_source_by_name(name, True).name

        prefix = kwargs.get('prefix','')
        
        self.logger.info('Computing SED for %s' % name)
        
        o = self._make_sed(name,profile,energies,**kwargs)
        
        self._plotter.make_sed_plot(self, name, **kwargs)

        self.logger.info('Finished SED')
        
        return o
    
    def _make_sed(self, name, profile=True, energies=None, **kwargs):

        # Extract options from kwargs
        config = copy.deepcopy(self.config['sed'])
        config.update(kwargs)

        bin_index = config['bin_index']
        use_local_index = config['use_local_index']
        fix_background = config['fix_background']
        ul_confidence = config['ul_confidence']
        cov_scale = config['cov_scale']        

        if energies is None:
            energies = self.energies
        else:
            energies = np.array(energies)

        nbins = len(energies) - 1
        max_index = 5.0
        min_flux = 1E-30
        erange = self.erange
        
        # Output Dictionary
        o = {'emin': energies[:-1],
             'emax': energies[1:],
             'ecenter': 0.5 * (energies[:-1] + energies[1:]),
             'flux': np.zeros(nbins),
             'eflux': np.zeros(nbins),
             'dfde': np.zeros(nbins),
             'e2dfde': np.zeros(nbins),
             'flux_err': np.zeros(nbins),
             'eflux_err': np.zeros(nbins),
             'dfde_err': np.zeros(nbins),
             'e2dfde_err': np.zeros(nbins),
             'flux_ul95': np.zeros(nbins) * np.nan,
             'eflux_ul95': np.zeros(nbins) * np.nan,
             'dfde_ul95': np.zeros(nbins) * np.nan,
             'e2dfde_ul95': np.zeros(nbins) * np.nan,
             'flux_ul': np.zeros(nbins) * np.nan,
             'eflux_ul': np.zeros(nbins) * np.nan,
             'dfde_ul': np.zeros(nbins) * np.nan,
             'e2dfde_ul': np.zeros(nbins) * np.nan,
             'dfde_err_lo': np.zeros(nbins) * np.nan,
             'e2dfde_err_lo': np.zeros(nbins) * np.nan,
             'dfde_err_hi': np.zeros(nbins) * np.nan,
             'e2dfde_err_hi': np.zeros(nbins) * np.nan,
             'index': np.zeros(nbins),
             'Npred': np.zeros(nbins),
             'ts': np.zeros(nbins),
             'fit_quality': np.zeros(nbins),
             'lnlprofile': [],
             'correlation' : {},
             'model_flux' : {},
             'config': config
             }
        
        saved_state = LikelihoodState(self.like)

        # Perform global spectral fit
        self._latch_free_params()
        self.free_sources(False,pars='shape')
        self.free_source(name)
        self.fit(loglevel=logging.DEBUG,update=False)
        o['model_flux'] = self.bowtie(name)
        
        self._restore_free_params()

        # Setup background parameters for SED
        self.free_sources(False,pars='shape')
        self.free_norm(name)

        if fix_background:
            self.free_sources(free=False)
        elif cov_scale is not None:
            self._latch_free_params()
            self.zero_source(name)
            self.fit(loglevel=logging.DEBUG,update=False)            
            srcNames = list(self.like.sourceNames())
            srcNames.remove(name)
            self.constrain_norms(srcNames, cov_scale)
            self.unzero_source(name)
            self._restore_free_params()
                                    
        # Precompute fluxes in each bin from global fit
        gf_bin_flux = []
        gf_bin_index = []
        for i, (emin, emax) in enumerate(zip(energies[:-1], energies[1:])):

            delta = 1E-5
            f = self.like[name].flux(10 ** emin, 10 ** emax)
            f0 = self.like[name].flux(10 ** emin * (1 - delta),
                                      10 ** emin * (1 + delta))
            f1 = self.like[name].flux(10 ** emax * (1 - delta),
                                      10 ** emax * (1 + delta))

            if f0 > min_flux:
                g = 1 - np.log10(f0 / f1) / np.log10(10 ** emin / 10 ** emax)
                gf_bin_index += [g]
                gf_bin_flux += [f]
            else:
                gf_bin_index += [max_index]
                gf_bin_flux += [min_flux]

        source = self.components[0].like.logLike.getSource(name)
        old_spectrum = source.spectrum()
        self.like.setSpectrum(name, 'PowerLaw')
        self.free_parameter(name, 'Index', False)
        self.set_parameter(name, 'Prefactor', 1.0, scale=1E-13,
                          true_value=False,
                          bounds=[1E-10, 1E10],
                          update_source=False)

        src_norm_idx = -1        
        free_params = self.get_params(True)
        for j, p in enumerate(free_params):
            if not p['is_norm']:
                continue
            if p['is_norm'] and p['src_name'] == name:
                src_norm_idx = j
            
            o['correlation'][p['src_name']] =  np.zeros(nbins) * np.nan
        
        for i, (emin, emax) in enumerate(zip(energies[:-1], energies[1:])):

            ecenter = 0.5 * (emin + emax)
            self.set_parameter(name, 'Scale', 10 ** ecenter, scale=1.0,
                               bounds=[1, 1E6], update_source=False)

            if use_local_index:
                o['index'][i] = -min(gf_bin_index[i], max_index)
            else:
                o['index'][i] = -bin_index
                
            self.set_parameter(name, 'Index', o['index'][i], scale=1.0,
                               update_source=False)

            normVal = self.like.normPar(name).getValue()
            flux_ratio = gf_bin_flux[i] / self.like[name].flux(10 ** emin,
                                                               10 ** emax)
            newVal = max(normVal * flux_ratio, 1E-10)
            self.set_norm(name, newVal)
            
            self.like.syncSrcParams(name)
            self.free_norm(name)
            self.logger.debug('Fitting %s SED from %.0f MeV to %.0f MeV' %
                              (name, 10 ** emin, 10 ** emax))
            self.setEnergyRange(emin, emax)

            fit_output = self.fit(loglevel=logging.DEBUG,update=False)
            free_params = self.get_params(True)
            for j, p in enumerate(free_params):
                
                if not p['is_norm']:
                    continue
                
                o['correlation'][p['src_name']][i] = \
                    fit_output['correlation'][src_norm_idx,j]
            
            o['fit_quality'][i] = fit_output['fit_quality']

            prefactor = self.like[self.like.par_index(name, 'Prefactor')]

            flux = self.like[name].flux(10 ** emin, 10 ** emax)
            flux_err = self.like.fluxError(name, 10 ** emin, 10 ** emax)
            eflux = self.like[name].energyFlux(10 ** emin, 10 ** emax)
            eflux_err = self.like.energyFluxError(name, 10 ** emin, 10 ** emax)
            dfde = prefactor.getTrueValue()
            dfde_err = dfde * flux_err / flux
            e2dfde = dfde * 10 ** (2 * ecenter)
            
            o['flux'][i] = flux
            o['eflux'][i] = eflux
            o['dfde'][i] = dfde
            o['e2dfde'][i] = e2dfde
            o['flux_err'][i] = flux_err
            o['eflux_err'][i] = eflux_err
            o['dfde_err'][i] = dfde_err
            o['e2dfde_err'][i] = dfde_err * 10 ** (2 * ecenter)

            cs = self.model_counts_spectrum(name, emin, emax, summed=True)
            o['Npred'][i] = np.sum(cs)
            o['ts'][i] = max(self.like.Ts2(name, reoptimize=False), 0.0)

            if profile:
                lnlp = self.profile_norm(name, emin=emin, emax=emax,
                                        savestate=False, reoptimize=True,
                                        npts=20)
                o['lnlprofile'] += [lnlp]

                ul_data = utils.get_parameter_limits(lnlp['flux'], lnlp['dlogLike'])
                
                o['flux_ul95'][i] = ul_data['ul']
                o['eflux_ul95'][i] = ul_data['ul']*(lnlp['eflux'][-1]/lnlp['flux'][-1])
                o['dfde_ul95'][i] = ul_data['ul']*(lnlp['dfde'][-1]/lnlp['flux'][-1])
                o['e2dfde_ul95'][i] = o['dfde_ul95'][i] * 10 ** (2 * ecenter)
                o['dfde_err_hi'][i] = ul_data['err_hi']*(lnlp['dfde'][-1]/lnlp['flux'][-1])
                o['e2dfde_err_hi'][i] = o['dfde_err_hi'][i] * 10 ** (2 * ecenter)
                o['dfde_err_lo'][i] = ul_data['err_lo']*(lnlp['dfde'][-1]/lnlp['flux'][-1])
                o['e2dfde_err_lo'][i] = o['dfde_err_lo'][i] * 10 ** (2 * ecenter)
                
                ul_data = utils.get_parameter_limits(lnlp['flux'], lnlp['dlogLike'], 
                                                     ul_confidence=ul_confidence)

                o['flux_ul'][i] = ul_data['ul']
                o['eflux_ul'][i] = ul_data['ul']*(lnlp['eflux'][-1]/lnlp['flux'][-1])
                o['dfde_ul'][i] = ul_data['ul']*(lnlp['dfde'][-1]/lnlp['flux'][-1])
                o['e2dfde_ul'][i] = o['dfde_ul'][i] * 10 ** (2 * ecenter)


        self.setEnergyRange(erange[0], erange[1])
        self.like.setSpectrum(name, old_spectrum)
        saved_state.restore()

        if cov_scale is not None:
            self.remove_priors()
        
        src = self.roi.get_source_by_name(name, True)
        src.update_data({'sed': copy.deepcopy(o)})

        return o
        
class Interpolator(object):
    """ Helper class for interpolating a 1-D function from a
    set of tabulated values.  

    Safely deals with overflows and underflows
    """
    def __init__(self,x,y):
        """ C'tor, take input array of x and y value         
        """
        x = np.squeeze(np.array(x,ndmin=1))
        y = np.squeeze(np.array(y,ndmin=1))
        
        msk = np.isfinite(y)
        x = x[msk]
        y = y[msk]

        y -= np.max(y)

        self._x = x
        self._y = y
        self._xmin = x[0]
        self._xmax = x[-1]
        self._ymin = y[0]
        self._ymax = y[-1]
        self._dydx_lo = (y[1]-y[0])/(x[1]-x[0])
        self._dydx_hi = (y[-1]-y[-2])/(x[-1]-x[-2])

        self._fn = UnivariateSpline(x,y,s=0,k=2)        
        self._sp = splrep(x,y,k=2,s=0)
        
    @property
    def xmin(self):
        """ return the minimum value over which the spline is defined
        """
        return self._xmin

    @property
    def xmax(self):
        """ return the maximum value over which the spline is defined
        """       
        return self._xmax

    @property
    def x(self):
        """ return the x values used to construct the split
        """       
        return self._x

    @property
    def y(self):
        """ return the y values used to construct the split
        """         
        return self._y

    def derivative(self,x,der=1):
        """ return the derivative a an array of input values

        x   : the inputs        
        der : the order of derivative         
        """ 
        return splev(x,self._sp,der=der)

    def __call__(self,x):
        """ Return the interpolated values for an array of inputs

        x : the inputs        

        Note that if any x value is outside the interpolation ranges
        this will return a linear extrapolation based on the slope
        at the endpoint
        """         
        x = np.array(x,ndmin=1)

        below_bounds = x < self._xmin
        above_bounds = x > self._xmax
        
        dxhi = np.array(x-self._xmax)
        dxlo = np.array(x-self._xmin)

        # UnivariateSpline will only accept 1-D arrays so this
        # passes a flattened version of the array.
        y = self._fn(x.ravel())
        y.resize(x.shape)
            
        y[above_bounds] = (self._ymax + dxhi[above_bounds]*self._dydx_hi)
        y[below_bounds] = (self._ymin + dxlo[below_bounds]*self._dydx_lo)
        return y
  

class LnLFn(object):
    """
    Helper class for interpolating a 1-D log-likelihood function from a
    set of tabulated values.  
    """
    def __init__(self,x,y,fluxType=0):
        """ C'tor, take input array of x and y value   

        fluxType :  code specifying the quantity used for the flux 
           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
           4: Differential flux of the test source ( ph cm^-2 s^-1 MeV^-1 )
           5: Differential energy flux of the test source ( MeV cm^-2 s^-1 MeV^-1 )           
        """
        self._interp = Interpolator(x,y)
        self._mle = None
        self._fluxType = fluxType

    @property
    def interp(self):
        """ return the underlying Interpolator object
        """ 
        return self._interp

    @property
    def fluxType(self):
        """ return a code specifying the quantity used for the flux 

           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
           4: Differential flux of the test source ( ph cm^-2 s^-1 MeV^-1 )
           5: Differential energy flux of the test source ( MeV cm^-2 s^-1 MeV^-1 )           
        """
        return self._fluxType        

    def _compute_mle(self):
        """ compute the maximum likelihood estimate, using the scipy.optimize.brentq method
        """
        if self._interp.y[0] == np.max(self._interp.y):
            self._mle = self._interp.x[0]
        else:
            ix0 = max(np.argmax(self._interp.y)-4,0)
            ix1 = min(np.argmax(self._interp.y)+4,len(self._interp.x)-1)            

            while np.sign(self._interp.derivative(self._interp.x[ix0])) == np.sign(self._interp.derivative(self._interp.x[ix1])):
                ix0 += 1
            
            self._mle = scipy.optimize.brentq(self._interp.derivative,
                                              self._interp.x[ix0], self._interp.x[ix1],
                                              xtol=1e-10*np.median(self._interp.x))    

    def mle(self):
        """ return the maximum likelihood estimate 

        This will return the cached value, if it exists
        """
        if self._mle is None:
            self._compute_mle()
        
        return self._mle
        

    def fn_mle(self):
        """ return the function value at the maximum likelihood estimate """
        return self._interp(self.mle())

    def TS(self):
        """ return the Test Statistic """
        return 2. * (self._interp(self.mle()) - self._interp(0.))

    
    def getLimit(self,alpha,upper=True):
        """ Evaluate the limits corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha :  limit confidence level.
        upper :  upper or lower limits.
        """
        dlnl = alphaToDeltaLogLike_1DOF(alpha)
        lnl_max = self.fn_mle()

        # This ultra-safe code to find an absolute maximum
        #fmax = self.fn_mle()
        #m = (fmax-self.interp.y > 0.1+dlnl) & (self.interp.x>self._mle)

        #if sum(m) == 0:
        #    xmax = self.interp.x[-1]*10
        #else:
        #    xmax = self.interp.x[m][0]

        # Matt has found that this is use an interpolator than an actual root-finder to 
        # find the root probably b/c of python overhead       
        #rf = lambda x: self._interp(x)+dlnl-lnl_max
        if upper:
            x = np.linspace(self._mle,self._interp.xmax,100)  
            #return opt.brentq(rf,self._mle,self._interp.xmax,xtol=1e-10*np.abs(self._mle))
        else:
            x = np.linspace(self._mle,self._interp.xmin,100) 
            #return opt.brentq(rf,self._interp.xmin,self._mle,xtol=1e-10*np.abs(self._mle))
            
        retVal =  np.interp(dlnl,lnl_max-self.interp(x),x)
        return retVal


    def getInterval(self,alpha):
        """ Evaluate the interval corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha : limit confidence level.
        """
        lo_lim = self.getLimit(alpha,upper=False)
        hi_lim = self.getLimit(alpha,upper=True)
        return (lo_err,hi_err)


class SpecData(object):
    """ This class wraps spectral data, e.g., energy bin definitions, flux values and number of predicted photons
    """
    def __init__(self,ebins,fluxes,npreds):
        """
        """
        self._ebins = ebins
        self._log_ebins = np.log10(self._ebins)
        self._evals = np.sqrt(self._ebins[0:-1]*self._ebins[1:])
        self._bin_widths = self._ebins[1:] - self._ebins[0:-1]
        self._fluxes = fluxes
        self._efluxes = self._ebins * self._fluxes
        self._npreds = npreds
        self._ne = len(ebins)-1


    @property
    def log_ebins(self):
        """ return the log10 of the energy bin edges
        """ 
        return self._log_ebins

    @property
    def ebins(self):
        """ return the energy bin edges
        """
        return self._ebins
    
    @property
    def bin_widths(self):
        """ return the energy bin widths
        """
        return self._bin_widths

    @property
    def evals(self):
        """ return the energy centers
        """
        return self._evals

    @property
    def fluxes(self):
        """ return the flux values
        """
        return self._fluxes

    @property
    def efluxes(self):
        """ return the energy flux values
        """
        return self._efluxes

    @property
    def npreds(self):
        """ return the number of predicted events
        """
        return self._npreds

    @property
    def nE(self):
        """ return the number of energy bins
        """
        return self._ne


class CastroData(object):
    """This class wraps the data needed to make a "Castro" plot, namely
        the log-likelihood as a function of normalization for a series
        of energy bins.
    """
    def __init__(self,norm_vals,nll_vals,specData,fluxType):
        """ C'tor

        Parameters
        ----------
        norm_vals   : The normalization values ( nEBins X N array, where N is the number of sampled values for each bin )
        nll_vals    : The log-likelihood values ( nEBins X N array, where N is the number of sampled values for each bin )
        specData    : The specData object
        fluxType :  code specifying the quantity used for the flux 
           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
           4: Differential flux of the test source ( ph cm^-2 s^-1 MeV^-1 )
           5: Differential energy flux of the test source ( MeV cm^-2 s^-1 MeV^-1 )           
        """
        self._norm_vals = norm_vals
        self._nll_vals = nll_vals
        self._specData = specData
        self._fluxType = fluxType
        self._loglikes = []
        self._ne = self._specData.nE
        self._nll_null = 0.0
      
        if fluxType == 0:
            factors = np.ones((self._specData.nE))
        elif fluxType == 1:
            factors = np.sqrt(self._specData.fluxes[0:-1]*self._specData.fluxes[1:]) * self._specData.bin_widths
        elif fluxType == 2:
            factors = np.sqrt(self._specData.efluxes[0:-1]*self._specData.efluxes[1:]) * self._specData.bin_widths
        elif fluxType == 3:
            factors = self._specData.npreds
        elif fluxType == 4:
            factors = np.sqrt(self._specData.fluxes[0:-1]*self._specData.fluxes[1:]) 
        elif fluxType == 5:
            factors = np.sqrt(self._specData.efluxes[0:-1]*self._specData.efluxes[1:]) 
        else:
            raise Exception('Unknown flux type: %s.  Options are 0-5'%fluxType)
         
        for ie in range(self._specData.nE):            
            nvv = factors[ie]*self._norm_vals[ie]
            nllfunc = LnLFn(nvv,self._nll_vals[ie],self._fluxType)
            self._nll_null -= self._nll_vals[ie][0]
            self._loglikes.append(nllfunc)
            pass


    @property
    def specData(self):
        """ Return the Spectral Data object """
        return self._specData

    @property
    def fluxType(self):
        """ Return the Flux type flag """ 
        return self._fluxType
    

    @property
    def nll_null(self):
        """ Return the negative log-likelihood for the null-hypothesis """ 
        return self._nll_null
    

    def __getitem__(self,i):
        """ return the LnLFn object for the ith energy bin
        """
        return self._loglikes[i]


    def __call__(self,x):
        """ return the log-like for an array of values, summed over the energy bins

        Parameters
        ----------
        x  : `~numpy.ndarray`  
           Array of nEbins x M values

        Returns
        -------
        nll_val : `~numpy.ndarray`          
           Array of negative log-likelihood values.
        """
        nll_val = 0.
        # crude hack to force the fitter away from unphysical values
        if ( x < 0 ).any():
            return 1000.

        for i,xv in enumerate(x):            
            nll_val -= self._loglikes[i].interp(xv)
        return nll_val
        

    def derivative(self,x,der=1):
        """ return the derivate of the log-like summed over the energy bins

        Parameters
        ----------
        x   : `~numpy.ndarray`  
           Array of nEbins x M values
           
        der : int
           Order of the derivate

        Returns
        -------
        der_val : `~numpy.ndarray`  
           Array of negative log-likelihood values.
        """
        der_val = 0.
        for i,xv in enumerate(x):
            der_val -= self._loglikes[i].interp.derivative(xv,der=der)
            pass
        return der_val
       

    def mles(self):
        """ return the maximum likelihood estimates for each of the energy bins
        """
        mle_vals = np.ndarray((self._ne))
        
        for i in range(self._ne):
            mle_vals[i] = self._loglikes[i].mle()
            pass
        return mle_vals


    def fn_mles(self):
        """ returns the summed likelihood at the maximum likelihood estimate

        Note that simply sums the maximum likelihood values at each bin, and 
        does not impose any sort of constrain between bins
        """
        mle_vals = self.mles()
        return self(mle_vals)


    def ts_vals(self):
        """ returns test statistic values for each energy bin
        """ 
        ts_vals = np.ndarray((self._ne))
        for i in range(self._ne):
            ts_vals[i] = self._loglikes[i].TS()
            pass
        return ts_vals 

    
    def getLimits(self,alpha,upper=True):
        """ Evaluate the limits corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha :  limit confidence level.
        upper :  upper or lower limits.

        returns an array of values, one for each energy bin
        """
        limit_vals = np.ndarray((self._ne))
        
        for i in range(self._ne):
            limit_vals[i] = self._loglikes[i].getLimit(alpha,upper)
            pass
        return limit_vals

    
    def fitNormalization(self,specVals,xlims):
        """ Fit the normalization given a set of spectral values that define a spectral shape

        This version is faster, and solves for the root of the derivatvie

        Parameters
        ----------
        specVals :  an array of (nebin values that define a spectral shape
        xlims    :  fit limits     

        returns the best-fit normalization value
        """
        fDeriv = lambda x : self.derivative(specVals*x)        
        try:
            result = scipy.optimize.brentq(fDeriv,xlims[0],xlims[1])
        except:
            if self.__call__(specVals*xlims[0]) < self.__call__(specVals*xlims[1]):
                return xlims[0]
            else:
                return xlims[1]
        return result
       
    
    def fitNorm_v2(self,specVals):
        """ Fit the normalization given a set of spectral values that define a spectral shape

        This version uses scipy.optimize.fmin
        
        Parameters
        ----------
        specVals :  an array of (nebin values that define a spectral shape
        xlims    :  fit limits     

        returns the best-fit normalization value
        """
        fToMin = lambda x : self.__call__(specVals*x)
        result = scipy.optimize.fmin(fToMin,0.,disp=False,xtol=1e-6)   
        return result
       
    
    def fit_spectrum(self,specFunc,initPars):
        """ Fit for the free parameters of a spectral function
        
        Parameters
        ----------
        specFunc :  The Spectral Function
        initPars :  The initial values of the parameters     


        Returns
        -------
        result   : tuple
           The output of scipy.optimize.fmin
        spec_out : `~numpy.ndarray`
           The best-fit spectral values
        TS_spec  : float
           The TS of the best-fit spectrum
        """
        fToMin = lambda x : self.__call__(specFunc(x))
        result = scipy.optimize.fmin(fToMin,initPars,disp=False,xtol=1e-6)   
        spec_out = specFunc(result)
        TS_spec = self.TS_spectrum(spec_out)
        return result,spec_out,TS_spec


    def TS_spectrum(self,spec_vals):
        """ Calculate and the TS for a given set of spectral values
        """        
        return 2. * (self._nll_null - self.__call__(spec_vals))



    def test_spectra(self,spec_types=["PowerLaw","LogParabola","PLExpCutoff"]):
        """
        """
        retDict = {}
        for specType in spec_types:            
            spec_func,init_pars,scaleEnergy = self.buildTestSpectrumFunction(specType)
            fit_result,fit_spec,fit_ts = self.fit_spectrum(spec_func,init_pars)
            # tweak the fit result to account for the flux type
            if self._fluxType == 0:
                fit_result[0] *= self._specData.fluxes[0] * self._specData.bin_widths[0]
                fit_result[1] -= 2.
            elif self._fluxType == 1:
                #fit_result[0] *= 1.
                fit_result[1] -= 1.
            elif self._fluxType == 2:
                fit_result[0] /= self._specData.ebins[0]
                fit_result[1] -= 2.
            elif self._fluxType == 3:
                fit_result[0] *= self._specData.fluxes[0] * self._specData.bin_widths[0] / self._specData.npreds[0]
                fit_result[1] -= 1.
            elif self._fluxType == 4:
                fit_result[0] *= self._specData.bin_widths[0]
            elif self._fluxType == 5:
                fit_result[0] *= self._specData.bin_widths[0] / self._specData.ebins[0]
                fit_result[1] -= 1.
       
            specDict = {"Function":spec_func,
                        "Result":fit_result,
                        "Spectrum":fit_spec,
                        "ScaleEnergy":scaleEnergy,
                        "TS":fit_ts}

            retDict[specType] = specDict
            pass
        return retDict


    def buildTestSpectrumFunction(self,specType):
        """
        """
        scaleEnergy = self._specData.ebins[0]
        cutoffEnergy = 10.*scaleEnergy

        # The initial parameters depend how the flux is expressed        
        if self._fluxType == 0:
            initPars = np.array([1e-3,0.0,0.0])
            initPars_pc = np.array([1e-3,0.0,cutoffEnergy])    
        elif self._fluxType == 1:
            initPars = np.array([1e-12,-1.0,0.0])
            initPars_pc = np.array([1e-12,-1.0,cutoffEnergy])
        elif self._fluxType == 2:
            initPars = np.array([1e-7,0.0,0.0])
            initPars_pc = np.array([1e-7,0.0,cutoffEnergy])       
        elif self._fluxType == 3:
            initPars = np.array([1.0,-2.0,0.0])
            initPars_pc = np.array([1.0,-2.0,cutoffEnergy])
        elif self._fluxType == 4:
            initPars = np.array([1e-17,-2.0,0.0])
            initPars_pc = np.array([1e-17,-2.0,cutoffEnergy])     
        elif self._fluxType == 5:
            initPars = np.array([1e-12,-1.0,0.0])
            initPars_pc = np.array([1e-12,-1.0,cutoffEnergy])

        # Build a function, and return it and the correct initial parameters
        if specType == "PowerLaw":
            return (PowerLaw(self._specData.evals,scaleEnergy),initPars[0:2],scaleEnergy)
        elif specType == "LogParabola":
            return (LogParabola(self._specData.evals,scaleEnergy),initPars,scaleEnergy)
        elif specType == "PLExpCutoff":
            return (PLExpCutoff(self._specData.evals,scaleEnergy),initPars_pc,scaleEnergy)
        else:
            print "Did not recognize test specturm type %s"%specType
        return None

    
class TSCube(object):
    """ 
    """
    def __init__(self,tsmap,normmap,tscube,norm_vals,nll_vals,specData,fluxType):
        """ C'tor


        Parameters
        ----------
        tsmap       : `~fermipy.utils.Map`
           A Map object with the TestStatistic values in each pixel
           
        tscube      : `~fermipy.utils.Map`
           A Map object with the TestStatistic values in each pixel & energy bin
           
        norm_vals   : `~numpy.ndarray`        
           The normalization values ( nEBins X N array, where N is the
           number of sampled values for each bin )
           
        nll_vals    : `~numpy.ndarray`        
           The log-likelihood values ( nEBins X N array, where N is
           the number of sampled values for each bin )
           
        specData    : `~fermipy.sed.SpecData`
           The specData object
           
        fluxType :
           code specifying the quantity used for the flux 
           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
           4: Differential flux of the test source ( ph cm^-2 s^-1 MeV^-1 )
           5: Differential energy flux of the test source ( MeV cm^-2 s^-1 MeV^-1 )           
       """
        self._tsmap = tsmap
        self._normmap = normmap
        self._tscube = tscube
        self._ts_cumul = tscube.sum_over_energy()
        self._specData = specData
        self._norm_vals = norm_vals
        self._nll_vals = nll_vals
        self._nE = self._specData.nE
        self._nN = 10
        self._castro_shape = (self._nN,self._nE)
        self._fluxType = fluxType

    @property
    def tsmap(self):
        """ return the Map of the TestStatistic value """
        return self._tsmap
    
    @property
    def normmap(self):
        """ return the Map of the Best-fit normalization value """
        return self._normmap

    @property
    def tscube(self):
        """ return the Cube of the TestStatistic value per pixel / energy bin """
        return self._tscube

    @property
    def ts_cumul(self):
        """ return the Map of the cumulative TestStatistic value per pixel (summed over energy bin) """
        return self._ts_cumul   

    @property
    def specData(self):
        """ Return the Spectral Data object """
        return self._specData  

    @property
    def nE(self):
        """ return the number of energy bins """
        return self._nE

    @property
    def nN(self):
        """ return the number of sample points in each energy bin """
        return self._nN

    @staticmethod 
    def create_from_fits(fitsfile,fluxType):
        """ Build a TSCube object from a fits file created by gttscube """
        m,f = read_map_from_fits(fitsfile)
        n,f = read_map_from_fits(fitsfile,"N_MAP")
        c,f = read_map_from_fits(fitsfile,"TSCUBE")
        log_ebins,fluxes,npreds = read_spectral_data(f["EBOUNDS"])
        ebins = np.power(10.,log_ebins)
        specData = SpecData(ebins,fluxes,npreds)
        cube_data_hdu = f["SCANDATA"]
        nll_vals = cube_data_hdu.data.field("NLL_SCAN")
        norm_vals = cube_data_hdu.data.field("NORMSCAN")
        return TSCube(m,n,c,norm_vals,nll_vals,specData,fluxType)


    def castroData_from_ipix(self,ipix,colwise=False):
        """ Build a CastroData object for a particular pixel """
        # pix = utils.skydir_to_pix
        if colwise:
            ipix = self._tsmap.ipix_swap_axes(ipix,colwise)
        norm_d = self._norm_vals[ipix].reshape(self._castro_shape).swapaxes(0,1)
        nll_d = self._nll_vals[ipix].reshape(self._castro_shape).swapaxes(0,1)
        return CastroData(norm_d,nll_d,self._specData,self._fluxType)
     

    def castroData_from_pix_xy(self,xy,colwise=False):
        """ Build a CastroData object for a particular pixel """
        ipix = self._tsmap.xy_pix_to_ipix(xy,colwise)
        return self.castroData_from_ipix(ipix)


    def find_and_refine_peaks(self,threshold,min_separation=1.0,use_cumul=False):
        """
        """
        if use_cumul: 
            theMap = self._ts_cumul
        else:
            theMap = self._tsmap
                    
        peaks = find_peaks(theMap,threshold,min_separation)
        for peak in peaks:
            o =  utils.fit_parabola(theMap.counts,peak['iy'],peak['ix'],dpix=2)
            peak['fit_loc'] = o
            peak['fit_skydir'] = SkyCoord.from_pixel(o['y0'],o['x0'],theMap.wcs)
            if o['fit_success']:            
                skydir = peak['fit_skydir']
            else:
                skydir = peak['skydir']
            pass
        return peaks


    def test_spectra_of_peak(self,peak,spec_types=["PowerLaw","LogParabola","PLExpCutoff"]):
        """
        """
        castro = self.castroData_from_pix_xy(xy=(peak['ix'],peak['iy']),colwise=False)
        test_dict = castro.test_spectra(spec_types)
        return (castro,test_dict)


    def find_sources(self,threshold,
                     min_separation=1.0,
                     use_cumul=False,
                     output_peaks=False,
                     output_castro=False,
                     output_specInfo=False,
                     output_src_dicts=False,
                     output_srcs=False):
        """
        """        
        srcs = []
        src_dicts = []
        castros = []
        specInfo = []
        names = []
        peaks = self.find_and_refine_peaks(threshold,min_separation,use_cumul=True)
        for i,peak in enumerate(peaks):
            (castro,test_dict) = self.test_spectra_of_peak(peak,["PowerLaw"])
            src_name = utils.create_source_name(peak['fit_skydir'])
            src_dict = build_source_dict(src_name,peak,test_dict,"PowerLaw")
            names.append(src_dict["name"])
            if output_castro:
                castros.append(castro)
            if output_specInfo:
                specInfo.append(test_dict)
            if output_src_dicts:
                src_dicts.append(src_dict)
            if output_srcs:
                src = roi_model.Source.create_from_dict(src_dict)
                srcs.append(src)            
            pass
        retDict = {"Names":names}
        if output_peaks:
            retDict["Peaks"]=peaks
        if output_castro:
            retDict["Castro"]=castros
        if output_specInfo:
            retDict["Spectral"]=specInfo
        if output_src_dicts:
            retDict["SrcDicts"]=src_dicts
        if output_srcs:
            retDict["Sources"]=srcs
        return retDict



def PowerLaw(evals,scale):
    """
    """
    evals_scaled = evals/scale
    return lambda x : x[0] * np.power(evals_scaled,x[1])


def LogParabola(evals,scale):
    """
    """
    evals_scaled = evals/scale
    log_evals_scaled = np.log(evals_scaled)
    return lambda x : x[0] * np.power(evals_scaled,x[1]-x[2]*log_evals_scaled);


def PLExpCutoff(evals,scale):
    """
    """
    evals_scaled = evals/scale
    evals_diff = scale - evals
    return lambda x : x[0] * np.power(evals_scaled,x[1]) * np.exp(evals_diff/x[2])


def build_source_dict(src_name,peak_dict,spec_dict,spec_type):
    """
    """
    spec_results = spec_dict[spec_type]
    src_dir = peak_dict['fit_skydir']

    src_dict = dict(name=src_name,
                    Source_Name=src_name,
                    SpatialModel='PointSource',
                    SpectrumType=spec_type,
                    ts=spec_results["TS"][0],
                    ra=src_dir.icrs.ra.deg,               
                    dec=src_dir.icrs.dec.deg,
                    Prefactor=spec_results["Result"][0],
                    Index=-1.*spec_results["Result"][1],
                    Scale=spec_results["ScaleEnergy"])
    return src_dict
                    


        
if __name__ == "__main__":


    from fermipy import sed
    from fermipy import roi_model
    import fermipy.utils as utils
    import xml.etree.cElementTree as ElementTree
    import sys

    if len(sys.argv) == 1:
        flux_type = 0
    elif sys.argv[1] == "norm":
        flux_type = 0
    elif sys.argv[1] == "flux":
        flux_type = 1
    elif sys.argv[1] == "eflux":
        flux_type = 2
    elif sys.argv[1] == "npred":
        flux_type = 3
    elif sys.argv[1] == "d_flux":
        flux_type = 4
    elif sys.argv[1] == "d_eflux":
        flux_type = 5
    else:
        print "Didn't reconginize flux type %s, choose from norm | flux | eflux | npred"%sys.argv[1]


    tscube = sed.TSCube.create_from_fits("tscube_test.fits",flux_type)

    resultDict = tscube.find_sources(10.0,1.0,use_cumul=True,
                                     output_peaks=True,
                                     output_specInfo=True)
    figList = []
    peaks = resultDict["Peaks"]
    specInfos = resultDict["Spectral"]
    sources = resultDict["Sources"]
    
    root = ElementTree.Element('source_library')
    root.set('title', 'source_library')
      
    for src in sources:
        src.write_xml(root)

        """
        result_pl = test_dict["PowerLaw"]["Result"]
        result_lp = test_dict["LogParabola"]["Result"]
        result_pc = test_dict["PLExpCutoff"]["Result"]
        ts_pl = test_dict["PowerLaw"]["TS"]
        ts_lp = test_dict["LogParabola"]["TS"]
        ts_pc = test_dict["PLExpCutoff"]["TS"]

        print "TS for PL index = 2:  %.1f"%max_ts
        print "Cumulative TS:        %.1f"%castro.ts_vals().sum()
        print "TS for PL index free: %.1f (Index = %.2f)"%(ts_pl[0],idx_off-result_pl[1])
        print "TS for LogParabola:   %.1f (Index = %.2f, Beta = %.2f)"%(ts_lp[0],idx_off-result_lp[1],result_lp[2])
        print "TS for PLExpCutoff:   %.1f (Index = %.2f, E_c = %.2f)"%(ts_pc[0],idx_off-result_pc[1],result_pc[2])
        """

    output_file = open("sed_sources.xml", 'w!')
    output_file.write(utils.prettify_xml(root))

