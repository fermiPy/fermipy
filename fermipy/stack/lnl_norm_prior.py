# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to fit dark matter spectra to castro data
"""
from __future__ import absolute_import, division, print_function

from functools import partial
import numpy as np

import scipy.optimize as opt
from scipy.interpolate import splrep, splev

from fermipy import castro


class LnLFn_norm_prior(castro.LnLFn):
    """ A class to add a prior on normalization of a LnLFn object

    L(x,y|z') = L_z(x*y|z')*L_y(y)

    where x is the parameter of interest, y is a nuisance parameter,
    and L_z is a likelihood constraining z = x*y.

    This class can compute:

    The likelikhood:
       L(x,y|z') : i.e., the likelihood given values of x and y

    The 'straight' likelihood:
       L(x)  : i.e., the likelihood without the prior

    The 'profile' likelihood:
       L_prof(x,y=y_min|z')  : where y_min is the value of y
                               that minimizes L for a given x.

    The 'marginal' likelihood:
       L_marg(x) = \int L(x,y|z') L(y) dy

    The posterior:
       P(x) = \int L(x,y|z') L(y) dy / \int L(x,y|z') L(y) dx dy

    The first call to compute the profile or marginal likelihoods
    or the posterior will result in computing and caching a spline
    to interpolate values on subsequent calls.

    The values returned by __call__ is determined by the ret_type parameter.


    Parameters
    ----------
    lnlx : '~fermipy.castro.LnLFn'
       The object wrapping L(x)

    nuis_pdf : '~fermipy.stats_utils.prior_functor'
       The object wrapping L(y)

    ret_type : str
       determine what is returned by __call__
       allowed values are 'straight','profile','marginal','posterior'
    """

    def __init__(self, lnlfn, nuis_pdf, ret_type='profile'):
        """C'tor
        """
        self._lnlfn = lnlfn
        self._nuis_pdf = nuis_pdf
        self._nuis_norm = nuis_pdf.normalization()
        self._nuis_log_norm = np.log(self._nuis_norm)
        self._marg_interp = None
        self._prof_interp = None
        self._post_interp = None
        self._ret_type = None
        self.clear_cached_values()
        init_interp = self.init_return(ret_type)
        self._mle = None
        xvals = init_interp.x
        yvals = init_interp.y
        super(LnLFn_norm_prior, self).__init__(xvals, yvals, lnlfn.norm_type)

    @staticmethod
    def nll_static(lnl, x, t):
        """Return the negative loglikehood """
        return -lnl.loglike(x, t)

    @property
    def ret_type(self):
        """Specifies what is returned by __call__
        """
        return self._ret_type

    @property
    def interp(self):
        """A '~fermipy.castro.Interpolator'

        That will give interoplated values of the type determined by ret_type
        """
        return self._interp

    def init_return(self, ret_type):
        """Specify the return type.

        Note that this will also construct the
        '~fermipy.castro.Interpolator' object
        for the requested return type.
        """
        if self._ret_type == ret_type:
            return None

        ret_val = None
        if ret_type == "straight":
            ret_val = self._lnlfn.interp
        if ret_type == "profile":
            self._profile_loglike_spline(self._lnlfn.interp.x)
            #self._profile_loglike(self._lnlfn.interp.x)
            ret_val = self._prof_interp
        elif ret_type == "marginal":
            self._marginal_loglike(self._lnlfn.interp.x)
            ret_val = self._marg_interp
        elif ret_type == "posterior":
            self._posterior(self._lnlfn.interp.x)
            ret_val = self._post_interp
        else:
            raise ValueError("Did not recognize return type %s" % ret_type)

        self._ret_type = ret_type
        return ret_val

    def clear_cached_values(self):
        """Removes all of the cached values and interpolators
        """
        self._prof_interp = None
        self._marg_interp = None
        self._post_interp = None
        self._interp = None
        self._ret_type = None

    def like(self, x, y):
        """Evaluate the 2-D likelihood in the x/y parameter space.

        The dimension of the two input arrays should be the same.

        Parameters
        ----------
        x : array_like
            Array of coordinates in the `x` parameter.

        y : array_like
            Array of coordinates in the `y` nuisance parameter.
        """
        # This is the negative log-likelihood
        z = self._lnlfn.interp(x * y)
        return np.exp(-z) * self._nuis_pdf(y) / self._nuis_norm

    def loglike(self, x, y):
        """Evaluate the 2-D log-likelihood in the x/y parameter space.

        The dimension of the two input arrays should be the same.

        Parameters
        ----------
        x : array_like
            Array of coordinates in the `x` parameter.

        y : array_like
            Array of coordinates in the `y` nuisance parameter.
        """
        nuis = self._nuis_pdf(y)
        log_nuis = np.where(nuis > 0., np.log(nuis), -1e2)
        vals = -self._lnlfn.interp(x * y) + log_nuis - self._nuis_log_norm
        return vals

    def straight_loglike(self, x):
        """Return the simple log-likelihood, i.e., L(x)
        """
        return self._lnlfn.interp(x)

    def profile_loglike(self, x):
        """Profile log-likelihood.

        Returns ``L_prof(x,y=y_min|z')``  : where y_min is the
                                            value of y that minimizes
                                            L for a given x.

        This will used the cached '~fermipy.castro.Interpolator' object
        if possible, and construct it if needed.
        """
        if self._prof_interp is None:
            # This calculates values and caches the spline
            return self._profile_loglike(x)[1]

        x = np.array(x, ndmin=1)
        return self._prof_interp(x)

    def marginal_loglike(self, x):
        """Marginal log-likelihood.

        Returns ``L_marg(x) = \int L(x,y|z') L(y) dy``

        This will used the cached '~fermipy.castro.Interpolator'
        object if possible, and construct it if needed.
        """
        if self._marg_interp is None:
            # This calculates values and caches the spline
            return self._marginal_loglike(x)

        x = np.array(x, ndmin=1)
        return self._marg_interp(x)

    def posterior(self, x):
        """Posterior function.

         Returns ``P(x) = \int L(x,y|z') L(y) dy / \int L(x,y|z') L(y) dx dy``

        This will used the cached '~fermipy.castro.Interpolator'
        object if possible, and construct it if needed.
        """
        if self._post_interp is None:
            return self._posterior(x)
        x = np.array(x, ndmin=1)
        return self._post_interp(x)

    def _profile_loglike(self, x):
        """Internal function to calculate and cache the profile likelihood
        """
        x = np.array(x, ndmin=1)

        z = []
        y = []

        for xtmp in x:
            #def fn(t):
            #    """Functor to return profile likelihood"""
            #    return -self.loglike(xtmp, t)
            fn = partial(LnLFn_norm_prior.nll_static, self, xtmp)
            ytmp = opt.fmin(fn, 1.0, disp=False)[0]
            ztmp = self.loglike(xtmp, ytmp)
            z.append(ztmp)
            y.append(ytmp)

        prof_y = np.array(y)
        prof_z = np.array(z)
        prof_z = prof_z.max() - prof_z
        self._prof_interp = castro.Interpolator(x, prof_z)
        return prof_y, prof_z

    def _profile_loglike_spline(self, x):
        """Internal function to calculate and cache the profile likelihood
        """
        z = []
        y = []

        yv = self._nuis_pdf.profile_bins()
        nuis_vals = self._nuis_pdf.log_value(yv) - self._nuis_log_norm
        for xtmp in x:
            zv = -1. * self._lnlfn.interp(xtmp * yv) + nuis_vals
            sp = splrep(yv, zv, k=2, s=0)

            #def rf(t):
            #    """Functor for spline evaluation"""
            #    return splev(t, sp, der=1)
            rf = partial(splev, sp=sp, der=1)
            ix = np.argmax(splev(yv, sp))
            imin, imax = max(0, ix - 3), min(len(yv) - 1, ix + 3)
            try:
                y0 = opt.brentq(rf, yv[imin], yv[imax], xtol=1e-10)
            except ValueError:
                y0 = yv[ix]
            z0 = self.loglike(xtmp, y0)
            z.append(z0)
            y.append(y0)

        prof_y = np.array(y)
        prof_z = np.array(z)
        prof_z = prof_z.max() - prof_z

        self._prof_interp = castro.Interpolator(x, prof_z)
        return prof_y, prof_z

    def _marginal_loglike(self, x):
        """Internal function to calculate and cache the marginal likelihood
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yw = yedge[1:] - yedge[:-1]
        yc = 0.5 * (yedge[1:] + yedge[:-1])

        s = self.like(x[:, np.newaxis], yc[np.newaxis, :])

        # This does the marginalization integral
        z = 1. * np.sum(s * yw, axis=1)
        marg_z = np.zeros(z.shape)
        msk = z > 0
        marg_z[msk] = -1 * np.log(z[msk])

        # Extrapolate to unphysical values
        # FIXME, why is this needed
        dlogzdx = (np.log(z[msk][-1]) - np.log(z[msk][-2])) / (x[msk][-1] - x[msk][-2])
        marg_z[~msk] = marg_z[msk][-1] + \
            (marg_z[~msk] - marg_z[msk][-1]) * dlogzdx
        self._marg_interp = castro.Interpolator(x, marg_z)
        return marg_z

    def _posterior(self, x):
        """Internal function to calculate and cache the posterior
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yc = 0.5 * (yedge[1:] + yedge[:-1])
        yw = yedge[1:] - yedge[:-1]

        like_array = self.like(x[:, np.newaxis], yc[np.newaxis, :]) * yw
        like_array /= like_array.sum()

        post = like_array.sum(1)
        self._post_interp = castro.Interpolator(x, post)
        return post

    def __call__(self, x):
        """Evaluate the quantity specified by ret_type parameter

        Parameters
        ----------
        x : array-like
            x value
        """
        return np.squeeze(self._interp(x))

    def _compute_mle(self):
        """Maximum likelihood estimator.
        """
        xmax = self._lnlfn.interp.xmax
        x0 = max(self._lnlfn.mle(), xmax * 1e-5)
        ret = opt.fmin(lambda x: np.where(
            xmax > x > 0, -self(x), np.inf), x0, disp=False)
        mle = float(ret[0])
        return mle
