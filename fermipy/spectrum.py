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
    """Functor object that wraps a
    `~fermipy.spectrum.SpectralFunction` and computes the
    normalization of the model in a sequence of SED energy bins.  The
    evaluation method of this class accepts a single vector for the
    parameters of the model.  This class serves as an object that can
    be passed to likelihood optimizers.
    """

    def __init__(self, sfn, emin, emax):

        self._emin = emin
        self._emax = emax
        self._sfn = sfn

    @property
    def emin(self):
        return self._emin

    @property
    def emax(self):
        return self._emax

    @property
    def spectral_fn(self):
        return self._sfn

    @property
    def scale(self):
        return self._sfn.scale

    @property
    def params(self):
        return self._sfn.params

    @scale.setter
    def scale(self, scale):
        self._sfn.scale = scale

    @params.setter
    def params(self, params):
        self._sfn.params = params


class SEDFluxFunctor(SEDFunctor):
    """Functor that computes the flux of a source in a pre-defined
    sequence of energy bins."""

    def __init__(self, sfn, emin, emax):
        super(SEDFluxFunctor, self).__init__(sfn, emin, emax)

    def __call__(self, params):

        params = cast_params(params)
        emin = np.expand_dims(self.emin, 1)
        emax = np.expand_dims(self.emax, 1)
        return np.squeeze(self._sfn.flux(emin, emax, params))


class SEDEFluxFunctor(SEDFunctor):
    """Functor that computes the energy flux of a source in a
    pre-defined sequence of energy bins."""

    def __init__(self, sfn, emin, emax):
        super(SEDEFluxFunctor, self).__init__(sfn, emin, emax)

    def __call__(self, params):

        params = cast_params(params)
        emin = np.expand_dims(self.emin, 1)
        emax = np.expand_dims(self.emax, 1)
        return np.squeeze(self._sfn.eflux(emin, emax, params))


class SpectralFunction(object):
    """Base class for spectral models.  Spectral models inheriting
    from this class should implement at a minimum an `_eval_dnde`
    method which evaluates the differential flux at a given energy."""

    def __init__(self, params, scale=1.0, extra_params=None):
        self._params = np.array(params)
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

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    @params.setter
    def params(self, params):
        if params is None:
            self._params = np.zeros(self.nparam)
        else:
            self._params = np.array(params)

    @property
    def extra_params(self):
        """Dictionary containing additional parameters needed for
        evaluation of the function."""
        return self._extra_params

    @classmethod
    def create_from_flux(cls, params, emin, emax, flux, scale=1.0):
        """Create a spectral function instance given its flux."""
        params = params.copy()
        params[0] = 1.0
        params[0] = flux / cls.eval_flux(emin, emax, params, scale=scale)
        return cls(params, scale)

    @classmethod
    def create_from_eflux(cls, params, emin, emax, eflux, scale=1.0):
        """Create a spectral function instance given its energy flux."""
        params = params.copy()
        params[0] = 1.0
        params[0] = eflux / cls.eval_eflux(emin, emax, params, scale=scale)
        return cls(params, scale)

    @classmethod
    def create_functor(cls, spec_type, func_type, emin, emax,
                       params=None, scale=1.0, extra_params=None):

        if isinstance(spec_type, SpectralFunction):
            sfn = copy.deepcopy(spec_type)
        else:
            sfn = eval(spec_type)(params, scale, extra_params)

        if func_type.lower() == 'flux':
            return SEDFluxFunctor(sfn, emin, emax)
        elif func_type.lower() == 'eflux':
            return SEDEFluxFunctor(sfn, emin, emax)
        else:
            raise Exception("Did not recognize func_type: %s" % func_type)

    @classmethod
    def create_flux_functor(cls, emin, emax, params=None, scale=1.0,
                            extra_params=None):
        sfn = cls(params, scale, extra_params)
        return SEDFluxFunctor(sfn, emin, emax)

    @classmethod
    def create_eflux_functor(cls, emin, emax, params=None, scale=1.0,
                             extra_params=None):
        sfn = cls(params, scale, extra_params)
        return SEDEFluxFunctor(sfn, emin, emax)

    @classmethod
    def eval_e2dnde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dnde(x, params, scale, extra_params) * x**2

    @classmethod
    def eval_ednde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dnde(x, params, scale, extra_params) * x

    @classmethod
    def eval_dnde(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dnde(x, params, scale, extra_params)

    @classmethod
    def eval_dnde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        return cls._eval_dnde_deriv(x, params, scale, extra_params)

    @classmethod
    def eval_ednde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        dnde_deriv = cls._eval_dnde_deriv(x, params, scale, extra_params)
        dnde = cls._eval_dnde(x, params, scale)
        return x * dnde_deriv + dnde

    @classmethod
    def eval_e2dnde_deriv(cls, x, params, scale=1.0, extra_params=None):
        x = cast_args(x)
        params = cast_params(params)
        dnde_deriv = cls._eval_dnde_deriv(x, params, scale, extra_params)
        dnde = cls._eval_dnde(x, params, scale)
        return x**2 * dnde_deriv + 2 * x * dnde

    @classmethod
    def _integrate(cls, fn, emin, emax, params, scale=1.0, extra_params=None,
                   npt=20):
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
        dnde = fn(np.exp(logx), params, scale, extra_params)
        return np.sum(dnde * xw, axis=-1)

    @classmethod
    def _eval_dnde_deriv(cls, x, params, scale=1.0, extra_params=None,
                         eps=1E-6):
        return (cls._eval_dnde(x + eps, params, scale) -
                cls._eval_dnde(x, params, scale)) / eps

    @classmethod
    def eval_flux(cls, emin, emax, params, scale=1.0, extra_params=None):
        emin = cast_args(emin)
        emax = cast_args(emax)
        params = cast_params(params)
        return cls._integrate(cls.eval_dnde, emin, emax, params, scale,
                              extra_params)

    @classmethod
    def eval_eflux(cls, emin, emax, params, scale=1.0, extra_params=None):
        emin = cast_args(emin)
        emax = cast_args(emax)
        params = cast_params(params)
        return cls._integrate(cls.eval_ednde, emin, emax, params, scale,
                              extra_params)

    def dnde(self, x, params=None):
        """Evaluate differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_dnde(x, params, self.scale,
                                         self.extra_params))

    def ednde(self, x, params=None):
        """Evaluate E times differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_ednde(x, params, self.scale,
                                          self.extra_params))

    def e2dnde(self, x, params=None):
        """Evaluate E^2 times differential flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_e2dnde(x, params, self.scale,
                                           self.extra_params))

    def dnde_deriv(self, x, params=None):
        """Evaluate derivative of the differential flux with respect to E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_dnde_deriv(x, params, self.scale,
                                               self.extra_params))

    def ednde_deriv(self, x, params=None):
        """Evaluate derivative of E times differential flux with respect to
        E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_ednde_deriv(x, params, self.scale,
                                                self.extra_params))

    def e2dnde_deriv(self, x, params=None):
        """Evaluate derivative of E^2 times differential flux with
        respect to E."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_e2dnde_deriv(x, params, self.scale,
                                                 self.extra_params))

    def flux(self, emin, emax, params=None):
        """Evaluate the integral flux."""
        params = self.params if params is None else params
        return np.squeeze(self.eval_flux(emin, emax, params, self.scale,
                                         self.extra_params))

    def eflux(self, emin, emax, params=None):
        """Evaluate the integral energy flux."""
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

    def __init__(self, params=None, scale=1.0, extra_params=None):
        params = (params if params is not None else
                  np.array([5e-13, -2.0]))
        super(PowerLaw, self).__init__(params, scale)

    @staticmethod
    def nparam():
        return 2

    @staticmethod
    def _eval_dnde(x, params, scale=1.0, extra_params=None):
        return params[0] * (x / scale) ** params[1]

    @staticmethod
    def eval_flux(emin, emax, params, scale=1.0, extra_params=None):

        phi0 = np.array(params[0], ndmin=1)
        index = np.array(params[1], ndmin=1)
        x0 = scale

        index1 = index + 1
        m = np.isclose(index1, 0.0)
        iindex0 = np.zeros(index.shape)
        iindex1 = np.zeros(index.shape)
        iindex1[~m] = 1. / index1[~m]
        iindex0[m] = 1.

        xmin = emin / scale
        xmax = emax / scale
        v = phi0 * iindex1 * (emax * xmax**index -
                              emin * xmin**index)
        if np.any(m):
            v += phi0 * x0 * iindex0 * (np.log(emax) - np.log(emin))
        return v

    @classmethod
    def eval_eflux(cls, emin, emax, params, scale=1.0, extra_params=None):

        params = copy.deepcopy(params)
        params[1] += 1.0
        return cls.eval_flux(emin, emax, params, scale) * scale

    @classmethod
    def eval_norm(cls, scale, index, emin, emax, flux):
        return flux / cls.eval_flux(emin, emax, [1.0, index], scale=scale)


class LogParabola(SpectralFunction):
    """Class that evaluates a function with the parameterization:

    F(x) = p_0 * (x/x_s)^(p_1 - p_2*log(x/x_s) )

    where x_s is a scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index (p_1)
    * params[2] : Curvature (p_2)

    """

    def __init__(self, params=None, scale=1.0, extra_params=None):
        params = (params if params is not None else
                  np.array([5e-13, -2.0, 0.0]))
        super(LogParabola, self).__init__(params, scale)

    @staticmethod
    def nparam():
        return 3

    @staticmethod
    def _eval_dnde(x, params, scale=1.0, extra_params=None):
        return (params[0] * (x / scale) **
                (params[1] - params[2] * np.log(x / scale)))


class PLExpCutoff(SpectralFunction):
    """Class that evaluates a function with the parameterization:

    F(x) = p_0 * (x/x_s)^(p_1) * exp(- x/p_2 ) 

    where x_s is the scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index (p_1)
    * params[2] : Cutoff (p_2)

    """

    def __init__(self, params=None, scale=1.0, extra_params=None):
        params = (params if params is not None else
                  np.array([5e-13, -1.0, 1E4]))
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
    def nparam():
        return 3

    @staticmethod
    def _eval_dnde(x, params, scale=1.0, extra_params=None):
        return params[0] * (x / scale) ** (params[1]) * np.exp(-x / params[2])

    @classmethod
    def _eval_dnde_deriv(cls, x, params, scale=1.0, extra_params=None):
        return (cls._eval_dnde(x, params, scale) *
                (params[1] * params[2] - x) / (params[2] * x))


class PLSuperExpCutoff(SpectralFunction):
    """Class that evaluates a function with the parameterization:

    F(x) = p_0 * (x/x_s)^(p_1) * exp(-(x/p2)**p3)

    where x_s is the scale parameter.  The `params` array should be
    defined with:

    * params[0] : Prefactor (p_0)
    * params[1] : Index1 (p_1)
    * params[2] : Curvature (p_2)
    * params[3] : Index2 (p3)

    """

    def __init__(self, params=None, scale=1.0, extra_params=None):
        params = (params if params is not None else
                  np.array([5e-13, -1.0, 1E4, 1.0]))
        super(PLSuperExpCutoff, self).__init__(params, scale)

    @staticmethod
    def params_to_log(params):
        return [np.log10(params[0]),
                params[1],
                np.log10(params[2]),
                params[3]]

    @staticmethod
    def log_to_params(params):
        return [10**params[0],
                params[1],
                10**params[2],
                params[3]]

    @staticmethod
    def nparam():
        return 4

    @staticmethod
    def _eval_dnde(x, params, scale=1.0, extra_params=None):
        return params[0] * (x / scale) ** (params[1]) * np.exp(- (x / params[2]) ** (params[3]))

    @classmethod
    def _eval_dnde_deriv(cls, x, params, scale=1.0, extra_params=None):
        return (cls._eval_dnde(x, params, scale) *
                (params[1] * params[2] - x) / (params[2] * x))


class DMFitFunction(SpectralFunction):
    """Class that evaluates the spectrum for a DM particle of a given
    mass, channel, cross section, and J-factor.  The parameterization
    is given by:

    F(x) = 1 / (8 * pi) * (1/mass^2) * sigmav * J * dN/dE(E,mass,i)

    where the `params` array should be defined with:

    * params[0] : sigmav
    * params[1] : mass

    Note that this class assumes that mass and J-factor are provided
    in units of GeV and GeV^2 cm^-5 while energies are defined in MeV.
    """

    # Mapping between the ST channel codes and the rows in the gammamc
    # file
    channel_index_mapping = {
        1: 8,  # ee
        2: 6,  # mumu
        3: 3,  # tautau
        4: 1,  # bb
        5: 2,  # tt
        6: 7,  # gg
        7: 4,  # ww
        8: 5,  # zz
        9: 0,  # cc
        10: 10,  # uu
        11: 11,  # dd
        12: 9,  # ss
    }

    # Mapping between ST channel codes and string aliases
    channel_name_mapping = {
        1:  ["e+e-", "ee"],
        2:  ["mu+mu-", "mumu", "musrc"],
        3:  ["tau+tau-", "tautau", "tausrc"],
        4:  ["bb-bar", "bb", "bbbar", "bbsrc"],
        5:  ["tt-bar", "tt"],
        6:  ["gluons", "gg"],
        7:  ["W+W-", "w+w-", "ww", "wwsrc"],
        8:  ["ZZ", "zz"],
        9:  ["cc-bar", "cc"],
        10:  ["uu-bar", "uu"],
        11:  ["dd-bar", "dd"],
        12:  ["ss-bar", "ss"]}

    channel_shortname_mapping = {
        1: "ee",
        2: "mumu",
        3: "tautau",
        4: "bb",
        5: "tt",
        6: "gg",
        7: "ww",
        8: "zz",
        9: "cc",
        10: "uu",
        11: "dd",
        12: "ss"}

   

    channel_rev_map = {vv: k for k, v in channel_name_mapping.items()
                       for vv in v}

    def __init__(self, params, chan='bb', jfactor=1E19, tablepath=None):
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
        xedge = np.linspace(0, 1.0, 251)
        self._x = 0.5 * (xedge[1:] + xedge[:-1]) * ndec - ndec

        chan_code = DMFitFunction.channel_rev_map[chan]
        ichan = DMFitFunction.channel_index_mapping[chan_code]
        self._chan = chan
        self._chan_code = chan_code

        # These are the mass points
        self._mass = np.array([2.0, 4.0, 6.0, 8.0, 10.0,
                               25.0, 50.0, 80.3, 91.2, 100.0,
                               150.0, 176.0, 200.0, 250.0, 350.0, 500.0, 750.0,
                               1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 7000.0, 1E4])
        self._dndx = data.reshape((12, 24, 250))
        self._dndx_interp = RegularGridInterpolator([self._mass, self._x],
                                                    self._dndx[ichan, :, :],
                                                    bounds_error=False,
                                                    fill_value=None)
        extra_params = {'dndx_interp': self._dndx_interp,
                        'chan': chan,
                        'jfactor': jfactor}
        super(DMFitFunction, self).__init__(params, 1.0, extra_params)

    @property
    def chan(self):
        """Return the channel string."""
        return self._chan

    @property
    def chan_code(self):
        """Return the channel code."""
        return self._chan_code

    @staticmethod
    def nparam():
        return 2

    @staticmethod
    def channels():
        """ Return all available DMFit channel strings """
        return DMFitFunction.channel_rev_map.keys()

    @staticmethod
    def _eval_dnde(x, params, scale=1.0, extra_params=None):

        dndx_interp = extra_params.get('dndx_interp')
        jfactor = extra_params.get('jfactor')
        sigmav = params[0]
        mass = params[1]
        xm = np.log10(x / mass) - 3.0
        phip = 1. / (8. * np.pi) * np.power(mass, -2) * (sigmav * jfactor)
        #dndx = self._dndx_interp[ichan]((np.log10(mass),xm))
        dndx = dndx_interp((mass, xm))
        dndx[xm > 0] = 0
        return phip * dndx / x

    def set_channel(self, chan):

        if isinstance(chan, int):
            ichan = DMFitFunction.channel_index_mapping[chan]
        else:
            chan_code = DMFitFunction.channel_rev_map[chan]
            ichan = DMFitFunction.channel_index_mapping[chan_code]

        self._dndx_interp = RegularGridInterpolator([self._mass, self._x],
                                                    self._dndx[ichan, :, :],
                                                    bounds_error=False,
                                                    fill_value=None)
        self.extra_params['dndx_interp'] = self._dndx_interp
        self.extra_params['chan'] = chan
