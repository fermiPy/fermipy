# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def cast_args(x):

    if isinstance(x, np.ndarray) and x.ndim >= 2:
        return x

    return np.expand_dims(np.array(x, ndmin=1), -1)


def cast_params(params):

    if isinstance(params[0], np.ndarray) and params[0].ndim >= 2:
        return list(params)

    o = []
    for i, p in enumerate(params):
        o += [np.expand_dims(np.array(params[i], ndmin=1), 0)]

    return o


class SEDFunctor(object):
    """Functor that accepts a model parameter vector and computes the
    normalization of a spectral model in a sequence of SED energy
    bins.
    """

    def __init__(self, spectrum, scale, emin, emax):

        self._emin = emin
        self._emax = emax
        self._scale = scale
        self._spectrum = spectrum

    @property
    def emin(self):
        return self._emin

    @property
    def emax(self):
        return self._emax

    @property
    def scale(self):
        return self._scale


class SEDFluxFunctor(SEDFunctor):
    """Functor that computes the flux of a source in a sequence of SED
    energy bins."""

    def __init__(self, spectrum, scale, emin, emax):
        super(SEDFluxFunctor, self).__init__(spectrum, scale, emin, emax)

    def __call__(self, params):

        params = cast_params(params)
        emin = np.expand_dims(self.emin, 1)
        emax = np.expand_dims(self.emax, 1)
        return np.squeeze(self._spectrum.eval_flux(emin, emax,
                                                   params, self.scale))


class SEDEFluxFunctor(SEDFunctor):
    """Functor that computes the energy flux of a source in a sequence
    of SED energy bins."""

    def __init__(self, spectrum, scale, emin, emax):
        super(SEDEFluxFunctor, self).__init__(spectrum, scale, emin, emax)

    def __call__(self, params):

        params = cast_params(params)
        emin = np.expand_dims(self.emin, 1)
        emax = np.expand_dims(self.emax, 1)
        return np.squeeze(self._spectrum.eval_eflux(emin, emax, params,
                                                    self.scale))


class SpectralFunction(object):
    """Base class for spectral model classes."""

    def __init__(self, params, scale=1.0, extra_params=None):
        self._params = params
        self._scale = scale
        self._extra_params = extra_params

    @property
    def params(self):
        """Return parameter vector of the function."""
        return self._params

    @property
    def log_params(self):
        """Return transformed parameter vector in which norm and scale
        parameters are converted to log10."""
        return self.params_to_log(self._params)

    @property
    def scale(self):
        return self._scale

    @property
    def extra_params(self):
        """Dictionary containing additional parameters needed for evaluation
        of the function."""
        return self._extra_params
    
    @staticmethod
    def create_functor(spec_type, func_type, emin, emax, scale=1.0, extra_params=None):

        if func_type.lower() == 'flux':
            return eval(spec_type).create_flux_functor(emin, emax, scale)
        elif func_type.lower() == 'eflux':
            return eval(spec_type).create_eflux_functor(emin, emax, scale)
        else:
            raise Exception("Did not recognize func_type: %s" % func_type)

    @classmethod
    def create_flux_functor(cls, emin, emax, escale=1.0, extra_params=None):
        return SEDFluxFunctor(cls, escale, emin, emax)

    @classmethod
    def create_eflux_functor(cls, emin, emax, escale=1.0, extra_params=None):
        return SEDEFluxFunctor(cls, escale, emin, emax)

    @classmethod
    def eval_e2dfde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dfde(x, params, scale, extra_params) * x**2

    @classmethod
    def eval_edfde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dfde(x, params, scale, extra_params) * x

    @classmethod
    def eval_dfde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dfde(x, params, scale, extra_params)

    @classmethod
    def eval_dfde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dfde_deriv(x, params, scale, extra_params)

    @classmethod
    def eval_edfde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        dfde_deriv = cls._eval_dfde_deriv(x, params, scale, extra_params)
        dfde = cls._eval_dfde(x, params, scale)
        return x*dfde_deriv + dfde

    @classmethod
    def eval_e2dfde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        dfde_deriv = cls._eval_dfde_deriv(x, params, scale, extra_params)
        dfde = cls._eval_dfde(x, params, scale)
        return x**2*dfde_deriv + 2*x*dfde

    @classmethod
    def _integrate(cls, fn, emin, emax, params, scale=1.0, extra_params=None, npt=20):
        """Fast numerical integration method using mid-point rule."""

        emin = np.expand_dims(emin, -1)
        emax = np.expand_dims(emax, -1)

        params = copy.deepcopy(params)
        for i, p in enumerate(params):
            params[i] = np.expand_dims(params[i], -1)

        xedges = np.linspace(0.0, 1.0, npt + 1)
        logx_edge = np.log(emin) + xedges * (np.log(emax) - np.log(emin))
        logx = 0.5 * (logx_edge[..., 1:] + logx_edge[..., :-1])
        xw = np.exp(logx_edge[..., 1:]) - np.exp(logx_edge[..., :-1])
        dfde = fn(np.exp(logx), params, scale, extra_params)
        return np.sum(dfde * xw, axis=-1)

    @classmethod
    def _eval_dfde_deriv(cls, x, params, scale=1.0, extra_params=None, eps=1E-6):
        return (cls._eval_dfde(x+eps, params, scale) -
                cls._eval_dfde(x, params, scale))/eps

    @classmethod
    def eval_flux(cls, emin, emax, params, scale=1.0, extra_params=None):
        emin = cast_args(emin)
        emax = cast_args(emax)
        params = cast_params(params)
        return cls._integrate(cls.eval_dfde, emin, emax, params, scale, extra_params)

    @classmethod
    def eval_eflux(cls, emin, emax, params, scale=1.0, extra_params=None):
        emin = cast_args(emin)
        emax = cast_args(emax)
        params = cast_params(params)
        return cls._integrate(cls.eval_edfde, emin, emax, params, scale,
                              extra_params)

    def dfde(self, x, params=None):
        """Evaluate differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_dfde(x, params, self.scale,
                                         self.extra_params))

    def edfde(self, x, params=None):
        """Evaluate E times differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_edfde(x, params, self.scale,
                                          self.extra_params))

    def e2dfde(self, x, params=None):
        """Evaluate E^2 times differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_e2dfde(x, params, self.scale,
                                           self.extra_params))

    def dfde_deriv(self, x, params=None):
        """Evaluate derivative of the differential flux with respect to E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_dfde_deriv(x, params, self.scale,
                                               self.extra_params))

    def edfde_deriv(self, x, params=None):
        """Evaluate derivative of E times differential flux with respect to
        E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_edfde_deriv(x, params, self.scale,
                                                self.extra_params))

    def e2dfde_deriv(self, x, params=None):
        """Evaluate derivative of E^2 times differential flux with respect to
        E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_e2dfde_deriv(x, params, self.scale,
                                                 self.extra_params))

    def flux(self, emin, emax, params=None):
        """Evaluate the integral flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_flux(emin, emax, params, self.scale,
                                         self.extra_params))

    def eflux(self, emin, emax, params=None):
        """Evaluate the energy flux flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_eflux(emin, emax, params, self.scale,
                                          self.extra_params))


class PowerLaw(SpectralFunction):
    """Class that evaluates a power-law function with the
    parameterization:

    F(x) = p_0 * (x/x_s)^p_1

    where x_s is the scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index (p_1)

    """
    def __init__(self, params, scale=1.0):
        super(PowerLaw, self).__init__(params, scale)

    @staticmethod
    def _eval_dfde(x, params, scale=1.0, extra_params=None):
        return params[0] * (x / scale) ** params[1]

    @classmethod
    def eval_flux(cls, emin, emax, params, scale=1.0, extra_params=None):

        phi0 = np.array(params[0], ndmin=1)
        index = np.array(params[1], ndmin=1)
        x0 = scale

        index1 = index + 1
        m = np.isclose(index1, 0.0)

        iindex0 = np.zeros(index.shape)
        iindex1 = np.zeros(index.shape)
        iindex1[~m] = 1. / index1[~m]
        iindex0[m] = 1.

        v0 = phi0 * x0 ** (-index) * (np.log(emax) - np.log(emin)) * iindex0
        v1 = phi0 * x0 ** (-index) * (emax ** index1 -
                                      emin ** index1) * iindex1

        return v0 + v1

#        y0 = x0 * phi0 * (emin / x0) ** (index + 1) / (index + 1)
#        y1 = x0 * phi0 * (emax / x0) ** (index + 1) / (index + 1)
#        v2 = y1 - y0
#        return v2

    @classmethod
    def eval_eflux(cls, emin, emax, params, scale=1.0, extra_params=None):

        params = copy.deepcopy(params)
        params[1] += 1.0
        return cls.eval_flux(emin, emax, params, scale) * scale

    @staticmethod
    def eval_norm(scale, index, emin, emax, flux):
        return flux / PowerLaw.eval_flux(emin, emax, [1.0, index], scale)


class LogParabola(SpectralFunction):
    """Class that evaluates a function with the parameterization:

    F(x) = p_0 * (x/x_s)^(p_1 - p_2*log(x/x_s) )

    where x_s is a scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index (p_1)
    * params[2] : Curvature (p_2)

    """
    def __init__(self, params, scale=1.0):
        super(LogParabola, self).__init__(params, scale)

    @staticmethod
    def _eval_dfde(x, params, scale=1.0, extra_params=None):
        return (params[0] * (x / scale) **
                (params[1] - params[2] * np.log(x / scale)))


class PLExpCutoff(SpectralFunction):
    """Class that evaluates a function with the parameterization:

    F(x) = p_0 * (x/x_s)^(p_1 - p_2*log(x/x_s) )

    where x_s is the scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index (p_1)
    * params[2] : Curvature (p_2)

    """
    def __init__(self, params, scale=1.0):
        super(PLExpCutoff, self).__init__(params, scale)

    @staticmethod
    def params_to_log(params):
        return [np.log10(params[0]),
                params[1],
                np.log10(params[2])]

    @staticmethod
    def log_to_params(params):
        return [10**params[0],
                params[1],
                10**params[2]]

    @staticmethod
    def _eval_dfde(x, params, scale=1.0, extra_params=None):
        return params[0] * (x / scale) ** (params[1]) * np.exp(-x / params[2])

    @classmethod
    def _eval_dfde_deriv(cls, x, params, scale=1.0, extra_params=None):
        return (cls._eval_dfde(x, params, scale) *
                (params[1]*params[2] - x)/(params[2]*x))


class DMFitFunction(SpectralFunction):
    """Class that evaluates a spectrum for a DM particle of a given mass,
    channel, cross section, and J-factor.  The parameterization is
    given by:

    F(x) = 1 / (8 * pi) * (1/mass^2) * sigmav * J * dN/dE(E,mass,i)

    where the `params` array should be defined with:

    * params[0] : sigmav
    * params[1] : mass

    """    
    def __init__(self, params, chan='bb', jfactor = 1E25, tablepath=None):
        """Constructor.

        Parameters
        ----------
        params : list
            Parameter vector.

        chan : str
            Channel string.  Can be one of cc, bb, tt, tautau, ww, zz,
            mumu, gg, ee, ss, uu, dd.

        jfactor : float
            J-factor of this object.  Note that this needs to be given
            in the same units as the mass parameter.

        tablepath : str
            Path to lookup table with pre-computed DM spectra on a
            grid of energy, mass, and channel.

        """
        
        if tablepath is None:        
            tablepath = os.path.join('$FERMIPY_DATA_DIR',
                                     'gammamc_dif.dat')
        data = np.loadtxt(os.path.expandvars(tablepath))

        # Number of decades in x = log10(E/M)
        ndec = 10.0        
        xedge = np.linspace(0,1.0,251)
        self._x = 0.5*(xedge[1:]+xedge[:-1])*ndec - ndec

        # Lookup table for chan string to table index
        chan_index = { 'cc' : 0,
                       'bb' : 1,
                       'tt' : 2,
                       'tautau' : 3,
                       'ww' : 4,
                       'zz' : 5,
                       'mumu' : 6,
                       'gg' : 7,
                       'ee' : 8,
                       'ss' : 9,
                       'uu' : 10,
                       'dd' : 11 }
        
        ichan = chan_index[chan]
        self._chan = chan
        
        # These are the mass points
        self._mass = np.array([2.0,4.0,6.0,8.0,10.0,
                               25.0,50.0,80.3,91.2,100.0,
                               150.0,176.0,200.0,250.0,350.0,500.0,750.0,
                               1000.0,1500.0,2000.0,3000.0,5000.0,7000.0,1E4])
        self._mass *= 1E3
        self._dndx = data.reshape((12,24,250))
        self._dndx_interp = RegularGridInterpolator([self._mass,self._x],
                                                    self._dndx[ichan,:,:],
                                                    bounds_error=False,
                                                    fill_value=None)
        extra_params = {'dndx_interp' : self._dndx_interp, 'chan' : 'bb',
                        'jfactor' : jfactor }        
        super(DMFitFunction, self).__init__(params, 1.0, extra_params)
            
    @property
    def chan(self):
        return self._chan

    @staticmethod
    def _eval_dfde(x, params, scale=1.0, extra_params=None):

        dndx_interp = extra_params.get('dndx_interp')
        jfactor = extra_params.get('jfactor')
        sigmav = params[0]
        mass = params[1]
        xm = np.log10(x/mass)
        phip = 1./(8.*np.pi)*np.power(mass,-2)*(sigmav*jfactor)
        #dndx = self._dndx_interp[ichan]((np.log10(mass),xm))
        dndx = dndx_interp((mass,xm))
        dndx[xm > 0] = 0        
        return phip*dndx/x
