#!/usr/bin/env python
#

# Description
"""
Utilities for dealing with SEDs

Many parts of this code are taken from dsphs/like/lnlfn.py by 
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, splrep, splev
import scipy.optimize as opt
import scipy.special as spf
from scipy.integrate import quad
import scipy
import astropy.io.fits as pf

from fermipy import utils
from utils import read_energy_bounds, read_spectral_data
from fermipy.fits_utils import read_map_from_fits


# Some useful functions


def alphaToDeltaLogLike_1DOF(alpha):
    """ return the delta log-likelihood corresponding to a particular C.L. of (1-alpha)%
    """
    dlnl = pow(np.sqrt(2.)*spf.erfinv(1-2*alpha),2.)/2.  
    return dlnl


FluxTypes = ['NORM','FLUX','EFLUX','NPRED']


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
    """ This class wraps the data needed to make a "Castro" plot, namely 
        the log-likelihood as a function of normalization for a series of energy bins
    """
    def __init__(self,norm_vals,nll_vals,specData,fluxType):
        """ C'tor

        norm_vals   : The normalization values ( nEBins X N array, where N is the number of sampled values for each bin )
        nll_vals    : The log-likelihood values ( nEBins X N array, where N is the number of sampled values for each bin )
        specData    : The specData object
        fluxType :  code specifying the quantity used for the flux 
           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
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

        x  : Array of nEbins x M values

        returns an array of M values
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

        x   : Array of nEbins x M values
        der : Order of the derivate

        returns an array of M values        
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


          Returns (result,spec_out,TS_spec)
       ----------          
        result   : The output of scipy.optimize.fmin
        spec_out : The best-fit spectral values
        TS_spec  : The TS of the best-fit spectrum
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



class TSCube(object):
    """ 
    """
    def __init__(self,tsmap,norm_vals,nll_vals,specData,fluxType):
        """ C'tor

        tsmap       : A Map object with the TestStatistic values in each pixel
        norm_vals   : The normalization values ( nEBins X N array, where N is the number of sampled values for each bin )
        nll_vals    : The log-likelihood values ( nEBins X N array, where N is the number of sampled values for each bin )
        specData    : The specData object
        fluxType :  code specifying the quantity used for the flux 
           0: Normalization w.r.t. to test source
           1: Flux of the test source ( ph cm^-2 s^-1 )
           2: Energy Flux of the test source ( MeV cm^-2 s^-1 )
           3: Number of predicted photons
       """
        self._tsmap = tsmap
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
        log_ebins,fluxes,npreds = read_spectral_data(f["EBOUNDS"])
        ebins = np.power(10.,log_ebins)
        specData = SpecData(ebins,fluxes,npreds)
        cube_data_hdu = f["SCANDATA"]
        nll_vals = cube_data_hdu.data.field("NLL_SCAN")
        norm_vals = cube_data_hdu.data.field("NORMSCAN")

        return TSCube(m,norm_vals,nll_vals,specData,fluxType)


    def castroData_from_ipix(self,ipix):
        """ Build a CastroData object for a particular pixel """
        # pix = utils.skydir_to_pix
        norm_d = self._norm_vals[ipix].reshape(self._castro_shape).swapaxes(0,1)
        nll_d = self._nll_vals[ipix].reshape(self._castro_shape).swapaxes(0,1)
        return CastroData(norm_d,nll_d,self._specData,self._fluxType)
     



def Powerlaw(evals,scale):
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


def PlExpCutoff(evals,scale):
    """
    """
    evals_scaled = evals/scale
    evals_diff = scale - evals
    return lambda x : x[0] * np.power(evals_scaled,x[1]) * np.exp(evals_diff/x[2])



        
if __name__ == "__main__":

    fluxType = 0
    xlims = (0.,1.)

    tscube = TSCube.create_from_fits("tscube_test.fits",fluxType)
    ts_map = tscube.tsmap.counts
    max_ts_pix = np.argmax(ts_map)
    max_ts = ts_map.flat[max_ts_pix]
    xpix = max_ts_pix/80
    ypix = max_ts_pix%80
    ipix = 80*ypix + xpix
 
    castro = tscube.castroData_from_ipix(ipix)    
    nll = castro[0]

    specVals = np.ones((castro.specData.nE))

    result = castro.fitNormalization(specVals,xlims)
    result2 = castro.fitNorm_v2(specVals)
 
    initPars = np.array([1e-3,0.0,0.0])
    initPars_pc = np.array([1e-3,0.0,1000.0])

    pl = Powerlaw(castro.specData.evals,1000)
    lp = LogParabola(castro.specData.evals,1000)
    pc = PlExpCutoff(castro.specData.evals,1000)

    fx_1 = pl(initPars[0:2])
    fx_2 = pl(initPars)
    fx_3 = pl(initPars_pc)
    
        
    
