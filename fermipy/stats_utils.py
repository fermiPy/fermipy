# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to fit dark matter spectra to castro data
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
from fermipy import castro


def norm(x, mu, sigma=1.0):
    """ Scipy norm function """
    return stats.norm(loc=mu, scale=sigma).pdf(x)


def ln_norm(x, mu, sigma=1.0):
    """ Natural log of scipy norm function truncated at zero """
    return np.log(stats.norm(loc=mu, scale=sigma).pdf(x))


def lognorm(x, mu, sigma=1.0):
    """ Log-normal function from scipy """
    return stats.lognorm(sigma, scale=mu).pdf(x)


def log10norm(x, mu, sigma=1.0):
    """ Scale scipy lognorm from natural log to base 10
    x     : input parameter
    mu    : mean of the underlying log10 gaussian
    sigma : variance of underlying log10 gaussian
    """
    return stats.lognorm(sigma * np.log(10), scale=mu).pdf(x)


def ln_log10norm(x, mu, sigma=1.0):
    """ Natural log of base 10 lognormal """
    return np.log(stats.lognorm(sigma * np.log(10), scale=mu).pdf(x))


def gauss(x, mu, sigma=1.0):
    s2 = sigma * sigma
    return 1. / np.sqrt(2 * s2 * np.pi) * np.exp(-(x - mu) * (x - mu) / (2 * s2))


def lngauss(x, mu, sigma=1.0):
    s2 = sigma * sigma
    return -0.5 * np.log(2 * s2 * np.pi) - np.power(x - mu, 2) / (2 * s2)


def lgauss(x, mu, sigma=1.0, logpdf=False):
    """ Log10 normal distribution...

    x     : Parameter of interest for scanning the pdf
    mu    : Peak of the lognormal distribution (mean of the underlying
            normal distribution is log10(mu)
    sigma : Standard deviation of the underlying normal distribution
    """
    x = np.array(x, ndmin=1)

    lmu = np.log10(mu)
    s2 = sigma * sigma

    lx = np.zeros(x.shape)
    v = np.zeros(x.shape)

    lx[x > 0] = np.log10(x[x > 0])

    v = 1. / np.sqrt(2 * s2 * np.pi) * np.exp(-(lx - lmu)**2 / (2 * s2))

    if not logpdf:
        v /= (x * np.log(10.))

    v[x <= 0] = -np.inf

    return v


def lnlgauss(x, mu, sigma=1.0, logpdf=False):

    x = np.array(x, ndmin=1)

    lmu = np.log10(mu)
    s2 = sigma * sigma

    lx = np.zeros(x.shape)
    v = np.zeros(x.shape)

    mask = x > 0
    inv_mask = np.invert(mask)

    lx[mask] = np.log10(x[mask])

    v = -0.5 * np.log(2 * s2 * np.pi) - np.power(lx - lmu, 2) / (2 * s2)
    if not logpdf:
        v -= 2.302585 * lx + np.log(np.log(10.))

    if inv_mask.any():
        v[inv_mask] = -np.inf

    return v


class prior_functor:
    """A functor class that wraps simple functions we use to
       make priors on parameters.
    """

    def __init__(self, funcname, j_ref=1.0):
        self._funcname = funcname
        self._j_ref = j_ref

    def normalization(self):
        """Normalization, i.e. the integral of the function
           over the normalization_range.
        """
        return 1.

    def normalization_range(self):
        """Normalization range.
        """
        return 0, np.inf

    def mean(self):
        """Mean value of the function.
        """
        return 1.

    def sigma(self):
        """The 'width' of the function.
        What this means depend on the function being used.
        """
        raise NotImplementedError(
            "prior_functor.sigma must be implemented by sub-class")

    def j_ref(self):
        """ Reference value for the function """
        return self._j_ref        

    @property
    def funcname(self):
        """A string identifying the function.
        """
        return self._funcname

    def marginalization_bins(self):
        """Binning to use to do the marginalization integrals
        """
        log_mean = np.log10(self.mean())
        # Default is to marginalize over two decades,
        # centered on mean, using 1000 bins
        return np.logspace(-1. + log_mean, 1. + log_mean, 1001)/self._j_ref

    def profile_bins(self):
        """ The binning to use to do the profile fitting
        """
        log_mean = np.log10(self.mean())
        log_half_width = max(5. * self.sigma(), 3.)
        # Default is to profile over +-5 sigma,
        # centered on mean, using 100 bins
        return np.logspace(log_mean - log_half_width,
                           log_mean + log_half_width, 101)/self._j_ref

    def log_value(self, x):
        """
        """
        return np.log(self.__call__(self.j_ref()*x))


class function_prior(prior_functor):
    """
    """

    def __init__(self, funcname, mu, sigma, fn, lnfn=None, j_ref=1.0):
        """
        """
        # FIXME, why doesn't super(function_prior,self) work here?
        prior_functor.__init__(self, funcname, j_ref)
        self._mu = mu
        self._sigma = sigma
        self._fn = fn
        self._lnfn = lnfn

    def normalization(self):
        """ The normalization 
        i.e., the intergral of the function over the normalization_range 
        """
        norm_r = self.normalization_range()        
        return quad(self, norm_r[0]*self._j_ref, norm_r[1]*self._j_ref)[0]

    def mean(self):
        """ The mean value of the function.
        """
        return self._mu

    def sigma(self):
        """ The 'width' of the function.
        What this means depend on the function being used.
        """
        return self._sigma

    def log_value(self, x):
        """
        """
        if self._lnfn is None:
            return np.log(self._fn(x*self.j_ref(), self._mu, self._sigma))
        return self._lnfn(x*self.j_ref(), self._mu, self._sigma)

    def __call__(self, x):
        """ Normal function from scipy """
        return self._fn(x*self.j_ref(), self._mu, self._sigma)


class lognorm_prior(prior_functor):
    """ A wrapper around the lognormal function.

    A note on the highly confusing scipy.stats.lognorm function...
    The three inputs to this function are:
    s           : This is the variance of the underlying 
                  gaussian distribution
    scale = 1.0 : This is the mean of the linear-space 
                  lognormal distribution.
                  The mean of the underlying normal distribution
                  occurs at ln(scale)
    loc = 0     : This linearly shifts the distribution in x (DO NOT USE)

    The convention is different for numpy.random.lognormal
    mean        : This is the mean of the underlying 
                  normal distribution (so mean = log(scale))
    sigma       : This is the standard deviation of the 
                  underlying normal distribution (so sigma = s)

    For random sampling:
    numpy.random.lognormal(mean, sigma, size)
    mean        : This is the mean of the underlying 
                  normal distribution (so mean = exp(scale))
    sigma       : This is the standard deviation of the 
                  underlying normal distribution (so sigma = s)

    scipy.stats.lognorm.rvs(s, scale, loc, size)
    s           : This is the standard deviation of the 
                  underlying normal distribution
    scale       : This is the mean of the generated 
                  random sample scale = exp(mean)

    Remember, pdf in log space is
    plot( log(x), stats.lognorm(sigma,scale=exp(mean)).pdf(x)*x )

    Parameters
    ----------
    mu : float
        Mean value of the function
    sigma : float
        Variance of the underlying gaussian distribution
    """

    def __init__(self, mu, sigma, j_ref=1.0):
        super(lognorm_prior, self).__init__('norm', j_ref)
        self._mu = mu
        self._sigma = sigma
        

    def mean(self):
        """Mean value of the function.
        """
        return self._mu

    def sigma(self):
        """ The 'width' of the function.
        What this means depend on the function being used.
        """
        return self._sigma

    def __call__(self, x):
        """ Log-normal function from scipy """
        return stats.lognorm(self._sigma, scale=self._mu).pdf(x*self.j_ref())


class norm_prior(prior_functor):
    """ A wrapper around the normal function.
    Parameters
    ----------
    mu : float
        Mean value of the function
    sigma : float
        Variance of the underlying gaussian distribution
    """

    def __init__(self, mu, sigma, j_ref=1.0):
        """
        """
        super(norm_prior, self).__init__('norm', j_ref)
        self._mu = mu
        self._sigma = sigma

    def mean(self):
        """Mean value of the function.
        """
        return self._mu

    def sigma(self):
        """ The 'width' of the function.
        What this means depend on the function being used.
        """
        return self._sigma

    def __call__(self, x):
        """Normal function from scipy """
        return stats.norm(loc=self._mu, scale=self._sigma).pdf(x*self.j_ref())


def create_prior_functor(d):
    """Build a prior from a dictionary.

    Parameters
    ----------
    d     :  A dictionary, it must contain:
       d['functype'] : a recognized function type
                       and all of the required parameters for the 
                       prior_functor of the desired type

    Returns
    ----------
    A sub-class of '~fermipy.stats_utils.prior_functor'

    Recognized types are:

    'lognorm'       : Scipy lognormal distribution
    'norm'          : Scipy normal distribution
    'gauss'         : Gaussian truncated at zero
    'lgauss'        : Gaussian in log-space
    'lgauss_like'   : Gaussian in log-space, with arguments reversed. 
    'lgauss_logpdf' : ???
    """
    functype = d.get('functype', 'lgauss_like')
    j_ref = d.get('j_ref', 1.0)
    if functype == 'norm':
        return norm_prior(d['mu'], d['sigma'], j_ref)
    elif functype == 'lognorm':
        return lognorm_prior(d['mu'], d['sigma'], j_ref)
    elif functype == 'gauss':
        return function_prior(functype, d['mu'], d['sigma'], gauss, lngauss, j_ref)
    elif functype == 'lgauss':
        return function_prior(functype, d['mu'], d['sigma'], lgauss, lnlgauss, j_ref)
    elif functype in ['lgauss_like', 'lgauss_lik']:
        def fn(x, y, s): return lgauss(y, x, s)

        def lnfn(x, y, s): return lnlgauss(y, x, s)
        return function_prior(functype, d['mu'], d['sigma'], fn, lnfn, j_ref)
    elif functype == 'lgauss_log':
        def fn(x, y, s): return lgauss(x, y, s, logpdf=True)

        def lnfn(x, y, s): return lnlgauss(x, y, s, logpdf=True)
        return function_prior(functype, d['mu'], d['sigma'], fn, lnfn, j_ref)
    else:
        raise KeyError("Unrecognized prior_functor type %s" % functype)


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
        self._lnlfn = lnlfn
        self._nuis_pdf = nuis_pdf
        self._nuis_norm = nuis_pdf.normalization()
        self._nuis_log_norm = np.log(self._nuis_norm)
        self.clear_cached_values()
        self.init_return(ret_type)
        self._mle = None
        self._norm_type = lnlfn.norm_type

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
            return
        if ret_type == "straight":
            self._interp = self._lnlfn.interp
        if ret_type == "profile":
            self._profile_loglike_spline(self._lnlfn.interp.x)
            #self._profile_loglike(self._lnlfn.interp.x)
            self._interp = self._prof_interp
        elif ret_type == "marginal":
            self._marginal_loglike(self._lnlfn.interp.x)
            self._interp = self._marg_interp
        elif ret_type == "posterior":
            self._posterior(self._lnlfn.interp.x)
            self._interp = self._post_interp
        else:
            raise ValueError("Did not recognize return type %s" % ret_type)

        self._ret_type = ret_type

    def clear_cached_values(self):
        """Removes all of the cached values and interpolators
        """
        self._prof_interp = None
        self._prof_y = None
        self._prof_z = None
        self._marg_interp = None
        self._marg_z = None
        self._post = None
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
        log_nuis = np.where( nuis > 0., np.log(nuis), -1e2)
        vals = -self._lnlfn.interp(x * y ) + \
            log_nuis - \
            self._nuis_log_norm
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
        if self._post is None:
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
            def fn(t): return -self.loglike(xtmp, t)
            ytmp = opt.fmin(fn, 1.0, disp=False)[0]
            ztmp = self.loglike(xtmp, ytmp)
            z.append(ztmp)
            y.append(ytmp)

        self._prof_y = np.array(y)
        self._prof_z = np.array(z)
        self._prof_z = self._prof_z.max() - self._prof_z
        self._prof_interp = castro.Interpolator(x, self._prof_z)
        return self._prof_y, self._prof_z

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

            def rf(t): return splev(t, sp, der=1)
            ix = np.argmax(splev(yv, sp))
            imin, imax = max(0, ix - 3), min(len(yv) - 1, ix + 3)
            try:
                y0 = opt.brentq(rf, yv[imin], yv[imax], xtol=1e-10)
            except:
                y0 = yv[ix]
            z0 = self.loglike(xtmp, y0)
            z.append(z0)
            y.append(y0)

        self._prof_y = np.array(y)
        self._prof_z = np.array(z)
        self._prof_z = self._prof_z.max() - self._prof_z
        
        self._prof_interp = castro.Interpolator(x, self._prof_z)
        return self._prof_y, self._prof_z

    def _marginal_loglike(self, x):
        """Internal function to calculate and cache the marginal likelihood
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yw = yedge[1:] - yedge[:-1]
        yc = 0.5 * (yedge[1:] + yedge[:-1])

        s = self.like(x[:, np.newaxis], yc[np.newaxis, :])

        # This does the marginalization integral
        z = 1. * np.sum(s * yw, axis=1)
        self._marg_z = np.zeros(z.shape)
        msk = z > 0
        self._marg_z[msk] = -1 * np.log(z[msk])

        # Extrapolate to unphysical values
        # FIXME, why is this needed
        dlogzdx = (np.log(z[msk][-1]) - np.log(z[msk][-2])
                   ) / (x[msk][-1] - x[msk][-2])
        self._marg_z[~msk] = self._marg_z[msk][-1] + \
            (self._marg_z[~msk] - self._marg_z[msk][-1]) * dlogzdx
        self._marg_interp = castro.Interpolator(x, self._marg_z)
        return self._marg_z

    def _posterior(self, x):
        """Internal function to calculate and cache the posterior
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yc = 0.5 * (yedge[1:] + yedge[:-1])
        yw = yedge[1:] - yedge[:-1]

        like_array = self.like(x[:, np.newaxis], yc[np.newaxis, :]) * yw
        like_array /= like_array.sum()

        self._post = like_array.sum(1)
        self._post_interp = castro.Interpolator(x, self._post)
        return self._post

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
