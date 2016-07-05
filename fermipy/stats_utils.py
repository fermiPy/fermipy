#!/usr/bin/env python
#

"""
Utilities to fit dark matter spectra to castro data
"""

import numpy as np

import scipy.stats as stats
import scipy.optimize as opt

from fermipy import castro


class prior_functor:
    """ A functor class that wraps simple functions we use to make priors 
    on paramters.    
    """
    def __init__(self):
        pass       

    def normalization(self):
        """ The normalization 
        i.e., the intergral of the function over the normalization_range 
        """
        return 1.

    def normalization_range(self):
        """ The normalization range.
        """
        return (0,np.inf)

    def mean(self):
        """ The mean value of the function.
        """
        return 1.

    def marginalization_bins(self):
        """ The binning to use to do the marginalization integrals
        """ 
        log_mean = np.log10(self.mean())
        # Default is to marginalize over two decades,
        # centered on mean, using 1000 bins
        return np.logspace(-1.+log_mean,1.+log_mean,1001)



class lognorm_prior(prior_functor):
    """ A wrapper around the lognormal function.

    A note on the highly confusing scipy.stats.lognorm function...
    The three inputs to this function are:
    s           : This is the variance of the underlying gaussian distribution
    scale = 1.0 : This is the mean of the linear-space lognormal distribution.
                  The mean of the underlying normal distribution occurs at ln(scale)
    loc = 0     : This linearly shifts the distribution in x (DO NOT USE)

    The convention is different for numpy.random.lognormal
    mean        : This is the mean of the underlying normal distribution (so mean = log(scale))
    sigma       : This is the standard deviation of the underlying normal distribution (so sigma = s)

    For random sampling:
    numpy.random.lognormal(mean, sigma, size)
    mean        : This is the mean of the underlying normal distribution (so mean = exp(scale))
    sigma       : This is the standard deviation of the underlying normal distribution (so sigma = s)

    scipy.stats.lognorm.rvs(s, scale, loc, size)
    s           : This is the standard deviation of the underlying normal distribution
    scale       : This is the mean of the generated random sample scale = exp(mean)

    Remember, pdf in log space is
    plot( log(x), stats.lognorm(sigma,scale=exp(mean)).pdf(x)*x )
    """
    def __init__(self,mu,sigma):
        """ C'tor

        Parameters
        ----------
        mu    :  The mean value of the function
        sigma :  The variance of the underlying gaussian distribution
        """
        self._mu = mu
        self._sigma = sigma

    def mean(self):
        """ .The mean value of the function.
        """
        return self._mu

    def __call__(self,x):
        """ Log-normal function from scipy """
        return stats.lognorm(self._sigma,scale=self._mu).pdf(x)


class norm_prior(prior_functor):
    """ A wrapper around the lormal function.

    Parameters
    ----------
    mu    :  The mean value of the function
    sigma :  The variance of the function
    """
    def __init__(self,mu,sigma):
        """
        """
        self._mu = mu
        self._sigma = sigma

    def mean(self):
        """ .The mean value of the function.
        """
        return self._mu    

    def __call__(self,x):
        """ Normal function from scipy """
        return stats.norm(loc=self._mu,scale=self._sigma).pdf(x)


def create_prior_functor(d):
    """ Build a prior from a dictionary

    Parameters
    ----------
    d     :  A dictionary, it must contain:
       d['functype'] : 'lognorm' or 'norm' 
       and all of the required parameters for the prior_functor of the desired type

    """
    functype = d.pop('functype','lognorm')
    if functype == 'norm':
        return norm_prior(**d)
    elif functype == 'lognorm':
        return lognorm_prior(**d)
    else:
        raise KeyError("Unrecognized prior_functor type %s"%functype)


class LnLFn_norm_prior:
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
       L_prof(x,y=y_min|z')  : where y_min is the value of y that minimizes L for a given x.

    The 'marginal' likelihood:
       L_marg(x) = \int L(x,y|z') L(y) dy

    The posterior:
       P(x) = \int L(x,y|z') L(y) dy / \int L(x,y|z') L(y) dx dy

    The first call to compute the profile or marginal likelihoods or the posterior will 
    result in computing and caching a spline to interpolate values on subsequent calls.

    The values returned by __call__ is determined by the ret_type parameter.
    """
    def __init__(self,lnlfn,nuis_pdf,ret_type='profile'):
        """ C'tor

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
        self._lnlfn = lnlfn
        self._nuis_pdf = nuis_pdf
        self._nuis_norm = nuis_pdf.normalization()
        self._nuis_log_norm = np.log(self._nuis_norm)
        self.clear_cached_values()
        self.init_return(ret_type) 

    @property
    def ret_type(self):
        """ Specifies what is returned by __call__
        """ 
        return self._ret_type

    @property
    def interp(self):
        """ returns a '~fermipy.castro.Interpolator'

        That will give interoplated values of the type determined by ret_type
        """ 
        return self._interp

    def init_return(self,ret_type):
        """ Specify the return type

        Note that this will also construct the '~fermipy.castro.Interpolator' object
        for the request return type.
        """
        if self._ret_type == ret_type:
            return
        if ret_type == "straight":
            self._interp = self._lnlfn.interp
        if ret_type == "profile":
            self._profile_loglike(self._lnlfn.interp.x)
            self._interp = self._prof_interp
        elif ret_type == "marginal":
            self._marginal_loglike(self._lnlfn.interp.x)
            self._interp = self._marg_interp
        elif ret_type == "posterior":
            self._posterior(self._lnlfn.interp.x)
            self._interp = self._post_interp            
        else:
            raise ValueError("Did not recognize return type %s"%ret_type)
        self._ret_type = ret_type


    def clear_cached_values(self):
        """ Removes all of the cached values and interpolators
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


    def like(self,x,y):
        """ Evaluate the 2-D likelihood in the x/y parameter space.
        The dimension of the two input arrays should be the same.

        Parameters
        ----------
        x : array_like
        Array of coordinates in the `x` parameter.

        y : array_like       
        Array of coordinates in the `y` nuisance parameter.
        """        
        # This is the negative log-likelihood
        z = self._lnlfn.interp(x*y)        
        return np.exp(-z)*self._nuis_pdf(y)/self._nuis_norm


    def loglike(self,x,y):
        """ Evaluate the 2-D log-likelihood in the x/y parameter space.
        The dimension of the two input arrays should be the same.

        Parameters
        ----------
        x : array_like
        Array of coordinates in the `x` parameter.

        y : array_like       
        Array of coordinates in the `y` nuisance parameter.
        """        
        vals = -self._lnlfn.interp(x*y) + np.log(self._nuis_pdf(y)) - self._nuis_log_norm
        return vals


    def straight_loglike(self,x):
        """ Return the simple log-likelihood, i.e., L(x)
        """
        return self._lnlfn.interp(x)

    def profile_loglike(self,x):
        """ Return the profile log-likelihood, 
            i.e., L_prof(x,y=y_min|z')  : where y_min is the value of y that minimizes L for a given x.

            This will used the cached '~fermipy.castro.Interpolator' object if possible, 
            and construct it if needed.
        """
        if self._prof_interp is None:
            # This calculates values and caches the spline 
            return self._profile_loglike(x)[1]            

        x = np.array(x,ndmin=1)
        return self._prof_interp(x)

    def marginal_loglike(self,x):
        """ Return the marginal log-likelihood, 
            i.e., L_marg(x) = \int L(x,y|z') L(y) dy

            This will used the cached '~fermipy.castro.Interpolator' object if possible,
            and construct it if needed.
        """
        if self._marg_interp is None:
            # This calculates values and caches the spline 
            return self._marginal_loglike(x)

        x = np.array(x,ndmin=1)
        return self._marg_interp(x)


    def posterior(self,x):
        """ Return the posterior function
            i.e., P(x) = \int L(x,y|z') L(y) dy / \int L(x,y|z') L(y) dx dy

            This will used the cached '~fermipy.castro.Interpolator' object if possible,
            and construct it if needed.
        """
        if self._post is None:
            return self._posterior(x)
        x = np.array(x,ndmin=1)
        return self._post_interp(x)


    def _profile_loglike(self,x):
        """ Internal function to calculate and cache the profile likelihood
        """
        x = np.array(x,ndmin=1)

        z = []
        y = []

        for xtmp in x:

            fn = lambda t: -self.loglike(xtmp,t)
            ytmp = opt.fmin(fn,1.0,disp=False)[0]
            ztmp = self.loglike(xtmp,ytmp)
            z.append(ztmp)
            y.append(ytmp)

        self._prof_y = np.array(y)
        self._prof_z = np.array(z)
        self._prof_z = self._prof_z.max() - self._prof_z
        self._prof_interp = castro.Interpolator(x,self._prof_z)
        return self._prof_y,self._prof_z


    def _marginal_loglike(self,x):
        """ Internal function to calculate and cache the marginal likelihood
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yw = yedge[1:]-yedge[:-1]
        yc = 0.5*(yedge[1:]+yedge[:-1])

        s = self.like(x[:,np.newaxis],yc[np.newaxis,:])

        # This does the marginalization integral
        z = 1.*np.sum(s*yw,axis=1)
        self._marg_z = np.zeros(z.shape)
        msk = z>0
        self._marg_z[msk] = -1*np.log(z[msk])

        # Extrapolate to unphysical values
        # FIXME, why is this needed
        dlogzdx = (np.log(z[msk][-1]) - np.log(z[msk][-2]))/(x[msk][-1]-x[msk][-2])
        self._marg_z[~msk] = self._marg_z[msk][-1] + (self._marg_z[~msk] - self._marg_z[msk][-1])*dlogzdx
        self._marg_interp = castro.Interpolator(x,self._marg_z)
        return self._marg_z


    def _posterior(self,x):
        """ Internal function to calculate and cache the posterior
        """
        yedge = self._nuis_pdf.marginalization_bins()
        yc = 0.5*(yedge[1:]+yedge[:-1])
        yw = yedge[1:]-yedge[:-1]


        like_array = self.like(x[:,np.newaxis],yc[np.newaxis,:])*yw 
        like_array /= like_array.sum()

        self._post = like_array.sum(1)
        self._post_interp = castro.Interpolator(x,self._post)
        return self._post


    def __call__(self,x):
        """ Evaluate the quantity specified by ret_type parameter

           x : array-like
        """
        return np.squeeze(self._interp(x))


    def mle(self):
        """ Maximum likelihood estimator """

        xmax = self._lnlfn.interp.xmax
        x0 = max(self._lnlfn.mle(),xmax*1e-5)
        ret = opt.fmin(lambda x: np.where(xmax>x>0, -self(x), np.inf), x0, disp=False)                       
        mle = float(ret[0])
        return ret





if __name__ == "__main__":

    pass
