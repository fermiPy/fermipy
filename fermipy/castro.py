# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for dealing with 'castro data', i.e., 2D table of
likelihood values.

Castro data can be tabluated in terms of a variety of variables. The
most common example is probably a simple SED, where we have the
likelihood as a function of Energy and Energy Flux.

However, we could easily convert to the likelihood as a function of
other variables, such as the Flux normalization and the spectral
index, or the mass and cross-section of a putative dark matter
particle.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy

from astropy.table import Table, Column
import astropy.units as u
from fermipy.wcs_utils import wcs_add_energy_axis
from fermipy.skymap import read_map_from_fits, Map
from fermipy.sourcefind_utils import fit_error_ellipse
from fermipy.sourcefind_utils import find_peaks
from fermipy.spectrum import SpectralFunction
from fermipy.utils import cl_to_dlnl

FluxTypes = ['NORM', 'FLUX', 'EFLUX', 'NPRED', 'DFDE', 'EDFDE']

PAR_NAMES = {
    "PowerLaw": ["Prefactor", "Index"],
    "LogParabola": ["norm", "alpha", "beta"],
    "PLExpCutoff": ["Prefactor", "Index1", "Cutoff"],
}


class Interpolator(object):
    """ Helper class for interpolating a 1-D function from a
    set of tabulated values.

    Safely deals with overflows and underflows
    """

    def __init__(self, x, y):
        """ C'tor, take input array of x and y value         
        """
        from scipy.interpolate import UnivariateSpline, splrep

        x = np.squeeze(np.array(x, ndmin=1))
        y = np.squeeze(np.array(y, ndmin=1))

        msk = np.isfinite(y)
        x = x[msk]
        y = y[msk]

        self._x = x
        self._y = y
        self._xmin = x[0]
        self._xmax = x[-1]
        self._ymin = y[0]
        self._ymax = y[-1]
        self._dydx_lo = (y[1] - y[0]) / (x[1] - x[0])
        self._dydx_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])

        self._fn = UnivariateSpline(x, y, s=0, k=1)
        self._sp = splrep(x, y, k=1, s=0)

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

    def derivative(self, x, der=1):
        """ return the derivative a an array of input values

        x   : the inputs        
        der : the order of derivative         
        """
        from scipy.interpolate import splev
        return splev(x, self._sp, der=der)

    def __call__(self, x):
        """ Return the interpolated values for an array of inputs

        x : the inputs        

        Note that if any x value is outside the interpolation ranges
        this will return a linear extrapolation based on the slope
        at the endpoint
        """
        x = np.array(x, ndmin=1)

        below_bounds = x < self._xmin
        above_bounds = x > self._xmax

        dxhi = np.array(x - self._xmax)
        dxlo = np.array(x - self._xmin)

        # UnivariateSpline will only accept 1-D arrays so this
        # passes a flattened version of the array.
        y = self._fn(x.ravel())
        y.resize(x.shape)

        y[above_bounds] = (self._ymax + dxhi[above_bounds] * self._dydx_hi)
        y[below_bounds] = (self._ymin + dxlo[below_bounds] * self._dydx_lo)
        return y


class LnLFn(object):
    """Helper class for interpolating a 1-D log-likelihood function from a
    set of tabulated values.
    """

    def __init__(self, x, y, norm_type=0):
        """C'tor, takes input arrays of x and y values

        Parameters
        ----------
        x : array-like
           Set of values of the free parameter

        y : array-like
           Set of values for the _negative_ log-likelhood

        norm_type :  str
           String specifying the type of quantity used for the `x`
           parameter.

        Notes
        -----
        Note that class takes and returns the _negative log-likelihood
        as fitters typically minimize rather than maximize.

        """
        self._interp = Interpolator(x, y)
        self._mle = None
        self._norm_type = norm_type

    @property
    def interp(self):
        """ return the underlying Interpolator object
        """
        return self._interp

    @property
    def norm_type(self):
        """Return a string specifying the quantity used for the normalization.
        This isn't actually used in this class, but it is carried so
        that the class is self-describing.  The possible values are
        open-ended.  
        """
        return self._norm_type

    def _compute_mle(self):
        """Compute the maximum likelihood estimate.

        Calls `scipy.optimize.brentq` to find the roots of the derivative.
        """
        if self._interp.y[0] == np.min(self._interp.y):
            self._mle = self._interp.x[0]
        else:
            ix0 = max(np.argmin(self._interp.y) - 4, 0)
            ix1 = min(np.argmin(self._interp.y) + 4, 
                      len(self._interp.x) - 1)

            while np.sign(self._interp.derivative(self._interp.x[ix0])) == \
                    np.sign(self._interp.derivative(self._interp.x[ix1])):
                ix0 += 1

            self._mle = scipy.optimize.brentq(self._interp.derivative,
                                              self._interp.x[ix0], 
                                              self._interp.x[ix1],
                                              xtol=1e-10 *
                                              np.median(self._interp.x))

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
        return 2. * (self._interp(0.) - self._interp(self.mle()))

    def getLimit(self, alpha, upper=True):
        """ Evaluate the limits corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha :  limit confidence level.
        upper :  upper or lower limits.
        """
        dlnl = cl_to_dlnl(1.0 - alpha)
        mle_val = self.mle()
        # A little bit of paranoia to avoid zeros
        if mle_val <= 0.:
            mle_val = self._interp.xmin
        if mle_val <= 0.:
            mle_val = self._interp.x[1]
        log_mle = np.log10(mle_val)
        lnl_max = self.fn_mle()

        # This ultra-safe code to find an absolute maximum
        # fmax = self.fn_mle()
        # m = (fmax-self.interp.y > 0.1+dlnl) & (self.interp.x>self._mle)

        # if sum(m) == 0:
        #    xmax = self.interp.x[-1]*10
        # else:
        #    xmax = self.interp.x[m][0]

        # Matt has found that it is faster to use an interpolator 
        # than an actual root-finder to find the root, 
        # probably b/c of python overhead.
        # That would be something like this:
        # rf = lambda x: self._interp(x)+dlnl-lnl_max
        # return opt.brentq(rf,self._mle,self._interp.xmax,
        #                   xtol=1e-10*np.abs(self._mle))
        if upper:
            x = np.logspace(log_mle, np.log10(self._interp.xmax), 100)
            # Old version.  This doesn't work if the interpolations
            # is defined over a huge range
            # x = np.linspace(self._mle, self._interp.xmax, 100)
        else:
            x = np.linspace(self._interp.xmin, self._mle, 100) 

        retVal = np.interp(dlnl, self.interp(x)-lnl_max, x)
        return retVal

    def getInterval(self, alpha):
        """ Evaluate the interval corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha : limit confidence level.
        """
        lo_lim = self.getLimit(alpha, upper=False)
        hi_lim = self.getLimit(alpha, upper=True)
        return (lo_lim, hi_lim)


class ReferenceSpec(object):
    """This class wraps data for a reference spectrum.
    
    Parameters
    ----------
    ne : `int`             
        Number of energy bins

    ebins : `~numpy.ndarray`  
        Array of bin edges.

    emin : `~numpy.ndarray`  
        Array of lower bin edges.
    
    emax : `~numpy.ndarray`  
        Array of upper bin edges.
    
    bin_widths : `~numpy.ndarray`  
        Array of energy bin widths.

    eref : `~numpy.ndarray`  
        Array of reference energies. Typically these are the geometric
        mean of the energy bins
    
    ref_dfde : `~numpy.ndarray`  
        Array of differential photon flux values.

    ref_flux : `~numpy.ndarray`  
        Array of integral photon flux values.

    ref_eflux : `~numpy.ndarray` 
        Array of integral energy flux values.

    ref_npred : `~numpy.ndarray` 
        Array of predicted number of photons in each energy bin.

    """
    def __init__(self, emin, emax, ref_dfde, ref_flux, ref_eflux, ref_npred, eref=None):
        """ C'tor from energy bin edges and refernce fluxes
        """
        self._ebins = np.append(emin, emax[-1])
        self._ne = len(self.ebins) - 1
        self._emin = emin
        self._emax = emax
        if eref is None:
            self._eref = np.sqrt(self.emin * self.emax)
        else:
            self._eref = eref
        self._log_ebins = np.log10(self._ebins)

        self._bin_widths = self._ebins[1:] - self._ebins[0:-1]
        self._ref_dfde = ref_dfde
        self._ref_flux = ref_flux
        self._ref_eflux = ref_eflux
        self._ref_npred = ref_npred

    @property
    def nE(self):
        return self._ne

    @property
    def log_ebins(self):
        return self._log_ebins

    @property
    def ebins(self):
        return self._ebins

    @property
    def emin(self):
        return self._emin

    @property
    def emax(self):
        return self._emax

    @property
    def bin_widths(self):
        return self._bin_widths

    @property
    def eref(self):
        return self._eref

    @property
    def ref_dfde(self):
        return self._ref_dfde

    @property
    def ref_flux(self):
        """ return the flux values
        """
        return self._ref_flux

    @property
    def ref_eflux(self):
        """ return the energy flux values
        """
        return self._ref_eflux

    @property
    def ref_npred(self):
        """ return the number of predicted events
        """
        return self._ref_npred

    @staticmethod
    def create_from_table(tab_e):
        """
        """
        emin = np.array(tab_e['E_MIN'])
        emax = np.array(tab_e['E_MAX'])
        try:
            if str(tab_e['E_MIN'].unit) == 'keV':
                emin /= 1000.
        except:
            pass
        try:
            if str(tab_e['E_MAX'].unit) == 'keV':
                emax /= 1000.
        except:
            pass
        
        ne = len(emin)
        try:
            ref_dfde = np.array(tab_e['REF_DFDE'])
        except:
            ret_dfde = np.ones((ne))

        try:
            ref_flux = np.array(tab_e['REF_FLUX'])
        except:
            ret_flux = np.ones((ne))

        try:
            ref_eflux = np.array(tab_e['REF_EFLUX'])
        except:
            ref_eflux = np.ones((ne))

        try:
            ref_npred = np.array(tab_e['REF_NPRED'])
        except:
            ref_npred = np.ones((ne))
            
        refSpec = ReferenceSpec(emin, emax,
                                ref_dfde,ref_flux,ref_eflux,ref_npred)
        return refSpec


    def build_ebound_table(self):
        """ Build and return a table with the encapsulated data
        """
        col_emin = Column(name="E_MIN", dtype=float,
                          shape=self._emin.shape, data=self._emin)
        col_emax = Column(name="E_MAX", dtype=float,
                          shape=self._emax.shape, data=self._emax)      
        col_eref = Column(name="E_REF", dtype=float,
                          shape=self._eref.shape, data=self._eref)      
        col_dfde = Column(name="REF_DFDE", dtype=float,
                          shape=self._ref_dfde.shape, data=self._ref_dfde)      
        col_flux = Column(name="REF_FLUX", dtype=float,
                          shape=self._ref_flux.shape, data=self._ref_flux)      
        col_eflux = Column(name="REF_EFLUX", dtype=float,
                           shape=self._ref_eflux.shape, data=self._ref_eflux)      
        col_npred = Column(name="REF_NPRED", dtype=float,
                           shape=self._ref_npred.shape, data=self._ref_npred)      

        tab = Table(data=[col_emin, col_emax, col_dfde,
                          col_flux, col_eflux, col_npred])
        return tab


class SpecData(ReferenceSpec):
    """ This class wraps spectral data, e.g., energy bin definitions,
    flux values and number of predicted photons
    """

    def __init__(self, ref_spec, norm_dfde, norm_flux, norm_eflux, norm_dfde_err=None):
        """

        Parameters
        ----------

        ref_spec :  `fermipy.castro.ReferenceSpec'
                     Object with energy bin definitions and reference spectra.

        norm_dfde :  `~numpy.ndarray`
                     Array of differential photon flux values, 
                     w.r.t. reference values.
                     

        norm_dfde_err : `~numpy.ndarray`  
                     Array of uncertainties on differential photon flux values, 
                     w.r.t reference values.

        norm_flux : `~numpy.ndarray`   
                    Array of integrated photon flux values,
                    w.r.t. reference values.
 
        norm_flux : `~numpy.ndarray`   
                    Array of integrated energy flux values,
                    w.r.t. reference values.

        flux :  `~numpy.ndarray`  Array of integral photon flux values.

        eflux :  `~numpy.ndarray` Array of integral energy flux values.

        e2dfde :`~numpy.ndarray`  Differential flux values scaled by E^2

        e2dfde_err :`~numpy.ndarray`  Uncertainties on differential 
                     flux values scaled by E^2

        """
        super(SpecData, self).__init__(ref_spec.emin, ref_spec.emax, 
                                       ref_spec.ref_dfde, ref_spec.ref_flux, 
                                       ref_spec.ref_eflux, ref_spec.ref_npred, 
                                       ref_spec.eref)    
        self._norm_dfde = norm_dfde
        self._norm_dfde_err = norm_dfde_err
        self._norm_flux = norm_flux
        self._norm_eflux = norm_eflux
        if self._norm_dfde is not None:
            self._dfde = self._norm_dfde * self._ref_dfde
        if self._norm_dfde_err is not None:
            self._dfde_err = self._norm_dfde_err * self._ref_dfde
        if self._norm_flux is not None:
            self._flux = self._norm_flux * self._ref_flux
        if self._norm_eflux is not None:
            self._eflux = self._norm_eflux * self._ref_eflux       

    @property
    def dfde(self):
        return self._dfde

    @property
    def dfde_err(self):
        return self._dfde_err

    @property
    def flux(self):
        return self._flux

    @property
    def eflux(self):
        return self._eflux

    @property
    def e2dfde(self):
        return self._dfde*self.eref**2

    @property
    def e2dfde_err(self):
        return self._dfde_err*self.evals**2

    @property
    def norm_dfde(self):
        return self._dfde / self.ref_dfde

    @property
    def norm_flux(self):
        return self._flux / self.ref_flux

    @property
    def norm_eflux(self):
        return self._eflux / self.ref_eflux
    
    @staticmethod
    def create_from_table(tab_e):
        """
        """
        rs = ReferenceSpec.create_from_table(tab_e)
        #FIXME

    def build_spec_table(self):
        """
        """
        col_emin = Column(name="E_MIN", dtype=float,
                          shape=self.emin.shape, data=self.emin)
        col_emax = Column(name="E_MAX", dtype=float,
                          shape=self.emax.shape, data=self.emax)      
        col_ref = Column(name="E_REF", dtype=float,
                          shape=self.eref.shape, data=self.emax)      
        col_list = [col_emin,col_emax,col_ref]
        if self.dfde is not None:
            col_list.append(Column(name="DFDE", dtype=float,
                                   shape=self.dfde.shape, data=self.dfde))
        if self.dfde_err is not None:
            col_list.append(Column(name="DFDE", dtype=float,
                                   shape=self.dfde_err.shape, data=self.dfde_err))
        if self.flux is not None:
            col_list.append(Column(name="FLUX", dtype=float,
                                   shape=self.flux.shape, data=self.flux))
        if self.eflux is not None:
            col_list.append(Column(name="EFLUX", dtype=float,
                                   shape=self.eflux.shape, data=self.eflux))

        tab = Table(data=col_list)
        return tab


class CastroData_Base(object):
    """ This class wraps the data needed to make a "Castro" plot,
    namely the log-likelihood as a function of normalization.

    In this case the x-axes and y-axes are generic
    Sub-classes can implement particul axes choices (e.g., EFlux v. Energy)
    """

    def __init__(self, norm_vals, nll_vals, norm_type):
        """C'tor

        Parameters
        ----------
        norm_vals : `~numpy.ndarray`
           The normalization values in an N X M array, where N is the
           number for bins and M number of sampled values for each bin

        nll_vals : `~numpy.ndarray`
           The _negative_ log-likelihood values in an N X M array,
           where N is the number for bins and M number of sampled
           values for each bin

        norm_type : str
           String specifying the quantity used for the normalization,
           value depend on the sub-class details

        """
        self._norm_vals = norm_vals
        self._nll_vals = nll_vals
        self._loglikes = []
        self._nll_null = 0.0
        self._norm_type = norm_type
        self._nx = self._norm_vals.shape[0]
        self._ny = self._norm_vals.shape[1]

        for i, (normv, nllv) in enumerate(zip(self._norm_vals,
                                              self._nll_vals)):
            nllfunc = self._buildLnLFn(normv, nllv)
            self._nll_null += self._nll_vals[i][0]
            self._loglikes.append(nllfunc)

    @property
    def nx(self):
        """ Return the number of profiles """
        return self._nx

    @property
    def ny(self):
        """ Return the number of profiles """
        return self._ny

    @property
    def norm_type(self):
        """ Return the normalization type flag """
        return self._norm_type

    @property
    def nll_null(self):
        """ Return the negative log-likelihood for the null-hypothesis """
        return self._nll_null

    def __getitem__(self, i):
        """ return the LnLFn object for the ith energy bin
        """
        return self._loglikes[i]

    def __call__(self, x):
        """Return the negative log-likelihood for an array of values,
        summed over the energy bins

        Parameters
        ----------
        x  : `~numpy.ndarray`
           Array of N x M values

        Returns
        -------
        nll_val : `~numpy.ndarray`
           Array of negative log-likelihood values.
        """
        if len(x.shape) == 1:
            nll_val = np.zeros((1))
        else:
            nll_val = np.zeros((x.shape[1:]))
        # crude hack to force the fitter away from unphysical values
        if (x < 0).any():
            return 1000.

        for i, xv in enumerate(x):
            nll_val += self._loglikes[i].interp(xv)

        return nll_val


    def norm_derivative(self, spec, norm):
        """
        """
        if isinstance(norm,float):
            der_val = 0.
        elif len(norm.shape) == 1:
            der_val = np.zeros((1))
        else:
            der_val = np.zeros((norm.shape[1:]))

        for i, sv in enumerate(spec):
            der_val += self._loglikes[i].interp.derivative(norm*sv, der=1) * sv 
        return der_val
   

    def derivative(self, x, der=1):
        """Return the derivate of the log-like summed over the energy
        bins

        Parameters
        ----------
        x   : `~numpy.ndarray`
           Array of N x M values

        der : int
           Order of the derivate

        Returns
        -------
        der_val : `~numpy.ndarray`
           Array of negative log-likelihood values.
        """
        if len(x.shape) == 1:
            der_val = np.zeros((1))
        else:
            der_val = np.zeros((x.shape[1:]))

        for i, xv in enumerate(x):
            der_val += self._loglikes[i].interp.derivative(xv, der=der)
        return der_val

    def mles(self):
        """ return the maximum likelihood estimates for each of the energy bins
        """
        mle_vals = np.ndarray((self._nx))
        for i in range(self._nx):
            mle_vals[i] = self._loglikes[i].mle()
        return mle_vals

    def fn_mles(self):
        """returns the summed likelihood at the maximum likelihood estimate

        Note that simply sums the maximum likelihood values at each
        bin, and does not impose any sort of constrain between bins
        """
        mle_vals = self.mles()
        return self(mle_vals)

    def ts_vals(self):
        """ returns test statistic values for each energy bin
        """
        ts_vals = np.ndarray((self._nx))
        for i in range(self._nx):
            ts_vals[i] = self._loglikes[i].TS()

        return ts_vals

    def getLimits(self, alpha, upper=True):
        """ Evaluate the limits corresponding to a C.L. of (1-alpha)%.

        Parameters
        ----------
        alpha :  float
           limit confidence level.
        upper :  bool
           upper or lower limits.

        returns an array of values, one for each energy bin
        """
        limit_vals = np.ndarray((self._nx))

        for i in range(self._nx):
            limit_vals[i] = self._loglikes[i].getLimit(alpha, upper)

        return limit_vals

    def fitNormalization(self, specVals, xlims):
        """Fit the normalization given a set of spectral values that
        define a spectral shape

        This version is faster, and solves for the root of the derivatvie

        Parameters
        ----------
        specVals :  an array of (nebin values that define a spectral shape
        xlims    :  fit limits     

        returns the best-fit normalization value
        """
        from scipy.optimize import brentq
        fDeriv = lambda x : self.norm_derivative(specVals,x)
        try:
            result = brentq(fDeriv, xlims[0], xlims[1])
        except:
            check_underflow = self.__call__(specVals * xlims[0]) < \
                self.__call__(specVals * xlims[1])
            if check_underflow.any():
                return xlims[0]
            else:
                return xlims[1]
        return result

    def fitNorm_v2(self, specVals):
        """Fit the normalization given a set of spectral values 
        that define a spectral shape.

        This version uses `scipy.optimize.fmin`.

        Parameters
        ----------
        specVals :  an array of (nebin values that define a spectral shape
        xlims    :  fit limits     

        Returns
        -------
        norm : float
            Best-fit normalization value
        """
        from scipy.optimize import fmin
        fToMin = lambda x: self.__call__(specVals * x)
        result = fmin(fToMin, 0., disp=False, xtol=1e-6)
        return result

    def fit_spectrum(self, specFunc, initPars):
        """ Fit for the free parameters of a spectral function

        Parameters
        ----------
        specFunc : `~fermipy.spectrum.SpectralFunction`
           The Spectral Function
        initPars : `~numpy.ndarray`
           The initial values of the parameters

        Returns
        -------
        result   : tuple
           The output of scipy.optimize.fmin
        spec_out : `~numpy.ndarray`
           The best-fit spectral values
        TS_spec  : float
           The TS of the best-fit spectrum
        """
        from scipy.optimize import fmin

        def fToMin(x):
            return self.__call__(specFunc(x))

        result = fmin(fToMin, initPars, disp=False, xtol=1e-6)
        spec_out = specFunc(result)
        TS_spec = self.TS_spectrum(spec_out)
        return result, spec_out, TS_spec

    def TS_spectrum(self, spec_vals):
        """Calculate and the TS for a given set of spectral values.
        """
        return 2. * (self._nll_null - self.__call__(spec_vals))

    def build_scandata_table(self):
        """
        """
        shape = self._norm_vals.shape
        col_norm = Column(name="NORM", dtype=float)
        col_normv = Column(name="NORM_SCAN", dtype=float,
                           shape=shape)
        col_dll = Column(name="DLOGLIKE_SCAN", dtype=float,
                         shape=shape)
        tab = Table(data=[col_norm, col_normv, col_dll])
        tab.add_row({"NORM": 1.,
                     "NORM_SCAN": self._norm_vals,
                     "DLOGLIKE_SCAN": -1*self._nll_vals})
        return tab

    @staticmethod
    def stack_nll(shape, components, ylims, weights=None):
        """Combine the log-likelihoods from a number of components.

        Parameters
        ----------
        shape    :  tuple
           The shape of the return array

        components : `~fermipy.castro.CastroData_Base`
           The components to be stacked

        weights : array-like

        Returns
        -------
        norm_vals : 'numpy.ndarray'
           N X M array of Normalization values

        nll_vals  : 'numpy.ndarray'
           N X M array of log-likelihood values
        """
        n_bins = shape[0]
        n_vals = shape[1]

        if weights is None:
            weights = np.ones((len(components)))

        norm_vals = np.zeros(shape)
        nll_vals = np.zeros(shape)
        for i in range(n_bins):
            log_min = np.log10(ylims[0])
            log_max = np.log10(ylims[1])
            norm_vals[i, 1:] = np.logspace(log_min, log_max, n_vals-1)
            check = 0
            for c, w in zip(components, weights):
                check += w*c[i].interp(norm_vals[i, -1])
                nll_vals[i] += w*c[i].interp(norm_vals[i])
                pass
            # reset the zeros
            nll_obj = LnLFn(norm_vals[i], nll_vals[i])
            nll_min = nll_obj.fn_mle()
            nll_vals[i] -= nll_min
            pass

        return norm_vals, nll_vals


class CastroData(CastroData_Base):
    """ This class wraps the data needed to make a "Castro" plot,
    namely the log-likelihood as a function of normalization for a
    series of energy bins.
    """

    def __init__(self, norm_vals, nll_vals, refSpec, norm_type):
        """ C'tor

        Parameters
        ----------
        norm_vals : `~numpy.ndarray`
           The normalization values ( nEBins X N array, where N is the
           number of sampled values for each bin )

        nll_vals : `~numpy.ndarray`
           The log-likelihood values ( nEBins X N array, where N is
           the number of sampled values for each bin )

        refSpec : `~fermipy.sed.ReferenceSpec`
           The object with the reference spectrum details.

        norm_type : str
            Type of normalization to use, options are:

            * NORM : Normalization w.r.t. to test source
            * FLUX : Flux of the test source ( ph cm^-2 s^-1 )
            * EFLUX: Energy Flux of the test source ( MeV cm^-2 s^-1 )
            * NPRED: Number of predicted photons (Not implemented)
            * DFDE : Differential flux of the test source ( ph cm^-2 s^-1
              MeV^-1 )
            * EDFDE: Differential energy flux of the test source ( MeV
              cm^-2 s^-1 MeV^-1 ) (Not Implemented)
     
        """
        super(CastroData, self).__init__(norm_vals, nll_vals, norm_type)
        self._refSpec = refSpec

    @property
    def nE(self):
        """ Return the number of energy bins.  This is also the number of x-axis bins.
        """
        return self._nx

    @property
    def refSpec(self):
        """ Return a `~fermipy.castro.ReferenceSpec` with the spectral data """
        return self._refSpec

    @staticmethod
    def create_from_flux_points(txtfile):
        """Create a Castro data object from a text file containing a
        sequence of differential flux points."""

        tab = Table.read(txtfile, format='ascii.ecsv')
        dfde_unit = u.ph / (u.MeV * u.cm ** 2 * u.s)
        loge = np.log10(np.array(tab['E_REF'].to(u.MeV)))
        norm = np.array(tab['NORM'].to(dfde_unit))
        norm_errp = np.array(tab['NORM_ERRP'].to(dfde_unit))
        norm_errn = np.array(tab['NORM_ERRN'].to(dfde_unit))
        norm_err = 0.5 * (norm_errp + norm_errn)
        dloge = loge[1:] - loge[:-1]
        dloge = np.insert(dloge, 0, dloge[0])
        emin = 10 ** (loge - dloge * 0.5)
        emax = 10 ** (loge + dloge * 0.5)
        ectr = 10 ** loge
        deltae = emax - emin
        flux = norm * deltae
        eflux = norm * deltae * ectr

        ones = np.ones(flux.shape)        
        ref_spec = ReferenceSpec(emin, emax, ones, ones, ones, ones)

        spec_data = SpecData(ref_spec, norm, flux, eflux, norm_err)

        stephi = np.linspace(0, 1, 11)
        steplo = -np.linspace(0, 1, 11)[1:][::-1]

        loscale = 3 * norm_err
        hiscale = 3 * norm_err
        loscale[loscale > norm] = norm[loscale > norm]

        norm_vals_hi = norm[:, np.newaxis] + \
            stephi[np.newaxis, :] * hiscale[:, np.newaxis]
        norm_vals_lo = norm[:, np.newaxis] + \
            steplo[np.newaxis, :] * loscale[:, np.newaxis]

        norm_vals = np.hstack((norm_vals_lo, norm_vals_hi))
        nll_vals = 0.5 * \
            (norm_vals - norm[:, np.newaxis]) ** 2 / \
            norm_err[:, np.newaxis] ** 2

        norm_vals *= flux[:, np.newaxis] / norm[:, np.newaxis]

        return CastroData(norm_vals, nll_vals, spec_data, 'FLUX')

    @staticmethod
    def create_from_tables(norm_type='EFLUX',
                           tab_s="SCANDATA",
                           tab_e="EBOUNDS"):
        """Create a CastroData object from two tables

        Parameters
        ----------
        norm_type : str
            Type of normalization to use.  Valid options are:

            * NORM : Normalization w.r.t. to test source
            * FLUX : Flux of the test source ( ph cm^-2 s^-1 )
            * EFLUX: Energy Flux of the test source ( MeV cm^-2 s^-1 )
            * NPRED: Number of predicted photons (Not implemented)
            * DFDE : Differential flux of the test source ( ph cm^-2 s^-1
              MeV^-1 )
            * EDFDE: Differential energy flux of the test source ( MeV
              cm^-2 s^-1 MeV^-1 ) (Not Implemented)

        tab_s : str
           table scan data

        tab_e : str
           table energy binning and normalization data

        Returns
        -------
        castro : `~fermipy.castro.CastroData`
        """
        if norm_type in ['FLUX', 'EFLUX', 'DFDE']:
            norm_vals = np.array(tab_s['NORM_SCAN'] *
                                 tab_e['REF_%s' % norm_type][:, np.newaxis])
        elif norm_type == "NORM":
            norm_vals = np.array(tab_s['NORM_SCAN'])
        else:
            raise Exception('Unrecognized normalization type: %s' % norm_type)

        nll_vals = -np.array(tab_s['DLOGLIKE_SCAN'])

        rs = ReferenceSpec.create_from_table(tab_e)
        return CastroData(norm_vals, nll_vals, rs, norm_type)

    @staticmethod
    def create_from_fits(fitsfile, norm_type='EFLUX',
                         hdu_scan="SCANDATA",
                         hdu_energies="EBOUNDS",
                         irow=None):
        """Create a CastroData object from a fits file.

        Parameters
        ----------
        fitsfile  : str
            Name of the fits file

        norm_type : str
            Type of normalization to use.  Valid options are:

            * NORM : Normalization w.r.t. to test source
            * FLUX : Flux of the test source ( ph cm^-2 s^-1 )
            * EFLUX: Energy Flux of the test source ( MeV cm^-2 s^-1 )
            * NPRED: Number of predicted photons (Not implemented)
            * DFDE : Differential flux of the test source ( ph cm^-2 s^-1
              MeV^-1 )
            * EDFDE: Differential energy flux of the test source ( MeV
              cm^-2 s^-1 MeV^-1 ) (Not Implemented)

        hdu_scan : str
            Name of the FITS HDU with the scan data

        hdu_energies : str
            Name of the FITS HDU with the energy binning and normalization data

        irow : int or None
            If none, then this assumes that there is a single row in
            the scan data table Otherwise, this specifies which row of
            the table to use

        Returns
        -------
        castro : `~fermipy.castro.CastroData`

        """
        if irow is not None:
            tab_s = Table.read(fitsfile, hdu=hdu_scan)[irow]
        else:
            tab_s = Table.read(fitsfile, hdu=hdu_scan)
        tab_e = Table.read(fitsfile, hdu=hdu_energies)
        return CastroData.create_from_tables(norm_type, tab_s, tab_e)

    @staticmethod
    def create_from_sedfile(fitsfile, norm_type='EFLUX'):
        """Create a CastroData object from an SED fits file

        Parameters
        ----------
        fitsfile  : str 
            Name of the fits file

        norm_type : str
            Type of normalization to use, options are:

            * NORM : Normalization w.r.t. to test source
            * FLUX : Flux of the test source ( ph cm^-2 s^-1 )
            * EFLUX: Energy Flux of the test source ( MeV cm^-2 s^-1 )
            * NPRED: Number of predicted photons (Not implemented)
            * DFDE : Differential flux of the test source ( ph cm^-2 s^-1
              MeV^-1 )
            * EDFDE: Differential energy flux of the test source ( MeV
              cm^-2 s^-1 MeV^-1 ) (Not Implemented)

        Returns
        -------
        castro : `~fermipy.castro.CastroData`
        """
        tab_s = Table.read(fitsfile, hdu=1)

        if norm_type in ['FLUX', 'EFLUX', 'DFDE']:
            norm_vals = np.array(tab_s['NORM_SCAN'] * 
                                 tab_s['REF_%s' % norm_type][:, np.newaxis])
        elif norm_type == "NORM":
            norm_vals = np.array(tab_s['NORM_SCAN'])
        else:
            raise Exception('Unrecognized normalization type: %s' % norm_type)

        nll_vals = -np.array(tab_s['DLOGLIKE_SCAN'])        
        rs = ReferenceSpec.create_from_table(tab_s)

        return CastroData(norm_vals, nll_vals, rs, norm_type)

    @staticmethod
    def create_from_stack(shape, components, ylims, weights=None):
        """  Combine the log-likelihoods from a number of components.

        Parameters
        ----------
        shape    :  tuple
           The shape of the return array

        components : [~fermipy.castro.CastroData_Base]
           The components to be stacked

        weights : array-like

        Returns
        -------
        castro : `~fermipy.castro.CastroData`
        """
        if len(components) == 0:
            return None
        norm_vals, nll_vals = CastroData_Base.stack_nll(
            shape, components, ylims, weights)
        return CastroData(norm_vals, nll_vals, 
                          components[0].refSpec, 
                          components[0].norm_type)

    def _buildLnLFn(self, normv, nllv):
        """
        """
        return LnLFn(normv, nllv, self._norm_type)

    def spectrum_loglike(self, specType, params, scale=1E3):
        """ return the log-likelihood for a particular spectrum

        Parameters
        ----------
        specTypes  : str
            The type of spectrum to try

        params : array-like
            The spectral parameters

        scale : float
            The energy scale or 'pivot' energy
        """
        sfn = self.create_functor(specType, scale)[0]
        return self.__call__(sfn(params))

    def test_spectra(self, spec_types=['PowerLaw', 
                                       'LogParabola', 
                                       'PLExpCutoff']):
        """Test different spectral types against the SED represented by this
        CastroData.

        Parameters
        ----------
        spec_types : [str,...]
           List of spectral types to try

        Returns
        -------
        retDict : dict
           A dictionary of dictionaries.  The top level dictionary is
           keyed by spec_type.  The sub-dictionaries each contain:

           * "Function"    : `~fermipy.spectrum.SpectralFunction`
           * "Result"      : tuple with the output of scipy.optimize.fmin
           * "Spectrum"    : `~numpy.ndarray` with best-fit spectral values
           * "ScaleEnergy" : float, the 'pivot energy' value
           * "TS"          : float, the TS for the best-fit spectrum

        """
        retDict = {}
        for specType in spec_types:
            spec_func, init_pars, scaleEnergy = self.create_functor(specType)
            fit_result, fit_spec, fit_ts = self.fit_spectrum(
                spec_func, init_pars)

            specDict = {"Function": spec_func,
                        "Result": fit_result,
                        "Spectrum": fit_spec,
                        "ScaleEnergy": scaleEnergy,
                        "TS": fit_ts}

            retDict[specType] = specDict

        return retDict

    def create_functor(self, specType, scale=1E3):
        """Create a functor object that computes normalizations in a
        sequence of energy bins for a given spectral model.

        Parameters
        ----------
        specType : str
            The type of spectrum to use.
            'PowerLaw','LogParabola','PLExpCutoff' are implemented.

        scale : float
            The 'pivot energy' or energy scale to use for the spectrum

        Returns
        -------
        fn : `~fermipy.spectrum.SpectralFunction`
            The functor

        initPars : `~numpy.ndarray`
            Default set of initial parameter for this spectral type

        scale : float
            Energy scale (same as input)

        """

        emin = self._refSpec.emin
        emax = self._refSpec.emax

        if specType == 'PowerLaw':
            initPars = np.array([5e-13, -2.0])
        elif specType == 'LogParabola':
            initPars = np.array([5e-13, -2.0, 0.0])
        elif specType == 'PLExpCutoff':
            initPars = np.array([5e-13, -1.0, 1E4])
        else:
            raise Exception('Unknown spectral type: %s' % specType)

        fn = SpectralFunction.create_functor(specType,
                                             self.norm_type,
                                             emin,
                                             emax,
                                             scale=scale)

        return (fn, initPars, scale)


class TSCube(object):
    """A class wrapping a TSCube, which is a collection of CastroData
    objects for a set of directions.  This class wraps a combination
    of:
      * Pixel data,
      * Pixel x Energy bin data, 
      * Pixel x Energy Bin x Normalization scan point data

    """

    def __init__(self, tsmap, normmap, tscube, normcube,
                 norm_vals, nll_vals, refSpec, norm_type):
        """C'tor

        Parameters
        ----------
        tsmap : `~fermipy.skymap.Map`
           A Map object with the TestStatistic values in each pixel

        normmap : `~fermipy.skymap.Map`
           A Map object with the normalization values in each pixel

        tscube : `~fermipy.skymap.Map`
           A Map object with the TestStatistic values in each pixel &
           energy bin

        normcube : `~fermipy.skymap.Map`
           A Map object with the normalization values in each pixel &
           energy bin

        norm_vals : `~numpy.ndarray`
           The normalization values ( nEBins X N array, where N is the
           number of sampled values for each bin )

        nll_vals : `~numpy.ndarray`
           The negative log-likelihood values ( nEBins X N array, where N is
           the number of sampled values for each bin )

        refSpec : `~fermipy.castro.ReferenceSpec`
           The ReferenceSpec object with the reference values.

        norm_type : str
            Type of normalization to use, options are:

            * NORM : Normalization w.r.t. to test source
            * FLUX : Flux of the test source ( ph cm^-2 s^-1 )
            * EFLUX: Energy Flux of the test source ( MeV cm^-2 s^-1 )
            * NPRED: Number of predicted photons (Not implemented)
            * DFDE : Differential flux of the test source ( ph cm^-2 s^-1
              MeV^-1 )
            * EDFDE: Differential energy flux of the test source ( MeV
              cm^-2 s^-1 MeV^-1 ) (Not Implemented)

        """
        self._tsmap = tsmap
        self._normmap = normmap
        self._tscube = tscube
        self._normcube = normcube
        self._ts_cumul = tscube.sum_over_energy()
        self._refSpec = refSpec
        self._norm_vals = norm_vals
        self._nll_vals = nll_vals
        self._nE = self._refSpec.nE
        self._nN = 10
        self._norm_type = norm_type

    @property
    def nvals(self):
        """Return the number of values in the tscube"""
        return self._norm_vals.shape[0]

    @property
    def tsmap(self):
        """ return the Map of the TestStatistic value """
        return self._tsmap

    @property
    def normmap(self):
        """return the Map of the Best-fit normalization value """
        return self._normmap

    @property
    def tscube(self):
        """return the Cube of the TestStatistic value per pixel / energy bin"""
        return self._tscube

    @property
    def normcube(self):
        """return the Cube of the normalization value per pixel / energy bin
        """
        return self._normcube

    @property
    def ts_cumul(self):
        """return the Map of the cumulative TestStatistic value per pixel
        (summed over energy bin)
        """
        return self._ts_cumul

    @property
    def refSpec(self):
        """ Return the Spectral Data object """
        return self._refSpec

    @property
    def nE(self):
        """ return the number of energy bins """
        return self._nE

    @property
    def nN(self):
        """ return the number of sample points in each energy bin """
        return self._nN

    @staticmethod
    def create_from_fits(fitsfile, norm_type='FLUX'):
        """Build a TSCube object from a fits file created by gttscube
        Parameters
        ----------
        fitsfile : str
           Path to the tscube FITS file.

        norm_type : str
           String specifying the quantity used for the normalization

        """
        tsmap, _ = read_map_from_fits(fitsfile)

        tab_e = Table.read(fitsfile, 'EBOUNDS')
        tab_s = Table.read(fitsfile, 'SCANDATA')
        tab_f = Table.read(fitsfile, 'FITDATA')

        emin = np.array(tab_e['E_MIN'])
        emax = np.array(tab_e['E_MAX'])
        try:
            if str(tab_e['E_MIN'].unit) == 'keV':
                emin /= 1000.
        except:
            pass
        try:
            if str(tab_e['E_MAX'].unit) == 'keV':
                emax /= 1000.
        except:
            pass


        nebins = len(tab_e)
        npred = tab_e['REF_NPRED']

        ndim = len(tsmap.counts.shape)

        if ndim == 2:
            cube_shape = (tsmap.counts.shape[0],
                          tsmap.counts.shape[1], nebins)
        elif ndim == 1:
            cube_shape = (tsmap.counts.shape[0], nebins)
        else:
            raise RuntimeError("Counts map has dimension %i" % (ndim))

        refSpec = ReferenceSpec.create_from_table(tab_e)
        nll_vals = -np.array(tab_s["DLOGLIKE_SCAN"])
        norm_vals = np.array(tab_s["NORM_SCAN"])

        wcs_3d = wcs_add_energy_axis(tsmap.wcs, emin)
        tscube = Map(np.rollaxis(tab_s["TS"].reshape(cube_shape), 2, 0),
                     wcs_3d)
        ncube = Map(np.rollaxis(tab_s["NORM"].reshape(cube_shape), 2, 0),
                    wcs_3d)
        nmap = Map(tab_f['FIT_NORM'].reshape(tsmap.counts.shape),
                   tsmap.wcs)

        ref_colname = 'REF_%s' % norm_type
        norm_vals *= tab_e[ref_colname][np.newaxis, :, np.newaxis]

        return TSCube(tsmap, nmap, tscube, ncube,
                      norm_vals, nll_vals, refSpec,
                      norm_type)

    def castroData_from_ipix(self, ipix, colwise=False):
        """ Build a CastroData object for a particular pixel """
        # pix = utils.skydir_to_pix
        if colwise:
            ipix = self._tsmap.ipix_swap_axes(ipix, colwise)
        norm_d = self._norm_vals[ipix]
        nll_d = self._nll_vals[ipix]
        return CastroData(norm_d, nll_d, self._refSpec, self._norm_type)

    def castroData_from_pix_xy(self, xy, colwise=False):
        """ Build a CastroData object for a particular pixel """
        ipix = self._tsmap.xy_pix_to_ipix(xy, colwise)
        return self.castroData_from_ipix(ipix)

    def find_and_refine_peaks(self, threshold, min_separation=1.0,
                              use_cumul=False):
        """Run a simple peak-finding algorithm, and fit the peaks to
        paraboloids to extract their positions and error ellipses.

        Parameters
        ----------
        threshold : float
            Peak threshold in TS.

        min_separation : float
            Radius of region size in degrees.  Sets the minimum allowable
            separation between peaks.

        use_cumul : bool
            If true, used the cumulative TS map (i.e., the TS summed
            over the energy bins) instead of the TS Map from the fit
            to and index=2 powerlaw.

        Returns
        -------
        peaks    : list
            List of dictionaries containing the location and amplitude of
            each peak.  Output of `~fermipy.sourcefind.find_peaks`

        """
        if use_cumul:
            theMap = self._ts_cumul
        else:
            theMap = self._tsmap

        peaks = find_peaks(theMap, threshold, min_separation)
        for peak in peaks:
            o, skydir = fit_error_ellipse(theMap, (peak['ix'], peak['iy']),
                                          dpix=2)
            peak['fit_loc'] = o
            peak['fit_skydir'] = skydir
            if o['fit_success']:
                skydir = peak['fit_skydir']
            else:
                skydir = peak['skydir']

        return peaks

    def test_spectra_of_peak(self, peak, spec_types=None):
        """Test different spectral types against the SED represented by the
        CastroData corresponding to a single pixel in this TSCube

        Parameters
        ----------
        spec_types  : [str,...]
           List of spectral types to try

        Returns
        -------
        castro : `~fermipy.castro.CastroData`
           The castro data object for the pixel corresponding to the peak

        test_dict : dict
           The dictionary returned by `~fermipy.castro.CastroData.test_spectra`

        """

        if spec_types is None:
            spec_types = ["PowerLaw", "LogParabola", "PLExpCutoff"]

        castro = self.castroData_from_pix_xy(
            xy=(peak['ix'], peak['iy']), colwise=False)
        test_dict = castro.test_spectra(spec_types)
        return (castro, test_dict)

    def find_sources(self, threshold,
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
        peaks = self.find_and_refine_peaks(
            threshold, min_separation, use_cumul=use_cumul)
        for peak in peaks:
            (castro, test_dict) = self.test_spectra_of_peak(peak, ["PowerLaw"])
            src_name = utils.create_source_name(peak['fit_skydir'])
            src_dict = build_source_dict(src_name, peak, test_dict, "PowerLaw")
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

        retDict = {"Names": names}
        if output_peaks:
            retDict["Peaks"] = peaks
        if output_castro:
            retDict["Castro"] = castros
        if output_specInfo:
            retDict["Spectral"] = specInfo
        if output_src_dicts:
            retDict["SrcDicts"] = src_dicts
        if output_srcs:
            retDict["Sources"] = srcs
        return retDict


def build_source_dict(src_name, peak_dict, spec_dict, spec_type):
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
                    Index=-1. * spec_results["Result"][1],
                    Scale=spec_results["ScaleEnergy"])

    src_dict['pos_sigma'] = peak_dict['fit_loc']['sigma']
    src_dict['pos_sigma_semimajor'] = peak_dict['fit_loc']['sigma_semimajor']
    src_dict['pos_sigma_semiminor'] = peak_dict['fit_loc']['sigma_semiminor']
    src_dict['pos_r68'] = peak_dict['fit_loc']['r68']
    src_dict['pos_r95'] = peak_dict['fit_loc']['r95']
    src_dict['pos_r99'] = peak_dict['fit_loc']['r99']
    src_dict['pos_angle'] = np.degrees(peak_dict['fit_loc']['theta'])

    return src_dict


if __name__ == "__main__":

    from fermipy import roi_model
    import fermipy.utils as utils
    import sys

    if len(sys.argv) == 1:
        flux_type = "FLUX"
    else:
        flux_type = sys.argv[1]

    # castro_sed = CastroData.create_from_sedfile("sed.fits")
    castro_sed = CastroData.create_from_fits("castro.fits", irow=0)
    test_dict_sed = castro_sed.test_spectra()
    print(test_dict_sed)
 
