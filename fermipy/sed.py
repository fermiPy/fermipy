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
from utils import read_energy_bounds


# Some useful functions


def alphaToDeltaLogLike_1DOF(alpha):
    """ return the delta log-likelihood corresponding to a particular C.L. of (1-alpha)%
    """
    dlnl = pow(np.sqrt(2.)*spf.erfinv(1-2*alpha),2.)/2.  
    return dlnl


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
    def __init__(self,x,y):
        """ C'tor, take input array of x and y value     
        """
        self._interp = Interpolator(x,y)
        self._mle = None

    @property
    def interp(self):
        """ return the underlying Interpolator object
        """ 
        return self._interp

    def _compute_mle(self):
        """ compute the maximum likelihood estimate, using the scipy.optimize.brentq method
        """
        if self._interp.y[0] == np.max(self._interp.y):
            self._mle = self._interp.x[0]
        else:
            ix0 = max(np.argmax(self._interp.y)-4,0)
            ix1 = min(np.argmax(self._interp.y)+4,len(self._interp.x)-1)            

            while np.sign(self._interp.derivative(self._interp.x[ix0])) == np.sign(self._interp.derivative(self._interp.x[ix1])):
                print ix0,ix1
                print np.sign(self._interp.derivative(self._interp.x[ix0]))
                print np.sign(self._interp.derivative(self._interp.x[ix1]))
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
            x = np.linspace(self._interp.xmin,self._mle,100)   
            #return opt.brentq(rf,self._interp.xmin,self._mle,xtol=1e-10*np.abs(self._mle))
            
        return np.interp(dlnl,lnl_max-self.interp(x),x)


    def getInterval(self,alpha):
        """ Evaluate the interval corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha : limit confidence level.
        """
        lo_lim = self.getLimit(alpha,upper=False)
        hi_lim = self.getLimit(alpha,upper=True)
        return (lo_err,hi_err)



class CastroData(object):
    """ This class wraps the data needed to make a "Castro" plot, namely 
        the log-likelihood as a function of normalization for a series of energy bins
    """
    def __init__(self,norm_vals,nll_vals,ebins):
        """ C'tor

        norm_vals   : The normalization values ( nEBins X N array, where N is the number of sampled values for each bin )
        nll_vals    : The log-likelihood values ( nEBins X N array, where N is the number of sampled values for each bin )
        ebins       : The energy bin edges ( array of nEBins+1 values )        
        """
        self._norm_vals = norm_vals
        self._nll_vals = nll_vals
        self._ebins = ebins
        self._ne = len(ebins)-1

        self._loglikes = []
        for ie in range(self._ne):
            nllfunc = LnLFn(self._norm_vals[ie],self._nll_vals[ie])
            self._loglikes.append(nllfunc)
            pass

    @property
    def nE(self):
        """ return the number of energy bins
        """
        return self._ne

    @property
    def ebins(self):
        """ return the energy bin edges
        """ 
        return self._ebins    

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
        for i,xv in enumerate(x):
            nll_val += self._loglikes[i](xv)
            pass
        return nll_val
        

    def derivative(self,x,der=1):
        """ return the derivate of the log-like summed over the energy bins

        x   : Array of nEbins x M values
        der : Order of the derivate

        returns an array of M values        
        """
        der_val = 0.
        for i,xv in enumerate(x):
            der_val += self._loglikes[i].derivative(xv,der=der)
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


    

class TSCube(object):
    """ 
    """
    def __init__(self,tsmap,ebins,ref_spec,norm_data,nll_data):
        """ C'tor

        tsmap     : A Map object with the TestStatistic values in each pixel
        ebins     : The energy bin edges
        ref_spec  : The references spectrum 
        norm_data : The normalization values ( nEBins X N array, where N is the number of sampled values for each bin )
        nll_data  : The log-likelihood values ( nEBins X N array, where N is the number of sampled values for each bin )
        """
        self._tsmap = tsmap
        self._ebins = ebins
        self._ref_spec = ref_spec
        self._norm_data = norm_data
        self._nll_data = nll_data
        self._nE = 20
        self._nN = 10
        self._castro_shape = (self._nN,self._nE)

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
    def create_from_fits(fitsfile):
        """ Build a TSCube object from a fits file created by gttscube """
        m,f = utils.read_map_from_fits(fitsfile)
        ebins = read_energy_bounds(f["EBOUNDS"])
        ref_spec = None
        cube_data_hdu = f["SCANDATA"]
        return TSCube(m,ebins,ref_spec,
                      #cube_data_hdu.data.field("NORM"),
                      #cube_data_hdu.data.field("DELTA_NLL"))
                      cube_data_hdu.data.field("NORMSCAN"),
                      cube_data_hdu.data.field("NLL_SCAN"))


    def castroData_from_ipix(self,ipix):
        """ Build a CastroData object for a particular pixel """
        # pix = utils.skydir_to_pix
        norm_d = self._norm_data[ipix].reshape(self._castro_shape).swapaxes(0,1)
        nll_d = self._nll_data[ipix].reshape(self._castro_shape).swapaxes(0,1)
       
        #print "Norms:",norm_d 
        #print "NLLs:",nll_d
        
        return CastroData(norm_d,nll_d,self._ebins)
        
        
if __name__ == "__main__":

    
    tscube = TSCube.create_from_fits("tscube_test2.fits")
    castro = tscube.castroData_from_ipix(55)
    nll = castro[0]

    
        
        
        
    
