# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pyLikelihood as pyLike
from fermipy import utils
from fermipy import gtutils


def get_fitcache_pars(fitcache):

    pars = pyLike.FloatVector()
    cov = pyLike.FloatVector()

    pyLike.Vector_Hep_to_Stl(fitcache.currentPars(), pars)
    pyLike.Matrix_Hep_to_Stl(fitcache.currentCov(), cov)

    pars = np.array(pars)
    cov = np.array(cov)

    npar = len(pars)

    cov = cov.reshape((npar, npar))
    err = np.sqrt(np.diag(cov))

    return pars, err, cov


class FitCache(object):

    def __init__(self, like, params, tol, max_iter, init_lambda, use_reduced):

        self._like = like
        self._init_fitcache(params, tol, max_iter, init_lambda, use_reduced)

    @property
    def fitcache(self):
        return self._fitcache

    @property
    def params(self):
        return self._cache_params

    def _init_fitcache(self, params, tol, max_iter, init_lambda, use_reduced):

        free_params = [p for p in params if p['free'] is True]
        free_norm_params = [p for p in free_params if p['is_norm'] is True]

        for p in free_norm_params:
            bounds = self._like[p['idx']].getBounds()
            self._like[p['idx']].setBounds(*utils.update_bounds(1.0, bounds))
            self._like[p['idx']] = 1.0
        self._like.syncSrcParams()

        self._fs_wrapper = pyLike.FitScanModelWrapper_Summed(
            self._like.logLike)
        self._fitcache = pyLike.FitScanCache(self._fs_wrapper,
                                             str('fitscan_testsource'),
                                             tol, max_iter, init_lambda,
                                             use_reduced, False, False)

        for p in free_norm_params:
            self._like[p['idx']] = p['value']
            self._like[p['idx']].setBounds(p['min'], p['max'])
        self._like.syncSrcParams()

        self._all_params = gtutils.get_params_dict(self._like)
        self._params = params
        self._cache_params = free_norm_params
        self._cache_param_idxs = [p['idx'] for p in self._cache_params]
        npar = len(self.params)
        self._prior_vals = np.ones(npar)
        self._prior_errs = np.ones(npar)
        self._prior_cov = np.ones((npar, npar))
        self._has_prior = np.array([False] * npar)

    def get_pars(self):
        return get_fitcache_pars(self.fitcache)

    def check_params(self, params):

        if len(params) != len(self._params):
            return False

        free_params = [p for p in params if p['free'] is True]
        free_norm_params = [p for p in free_params if p['is_norm'] is True]
        cache_src_names = np.array(self.fitcache.templateSourceNames())

        for i, p in enumerate(free_norm_params):
            if p['idx'] not in self._cache_param_idxs:
                return False

        # Check if any fixed parameters changed
        for i, p in enumerate(self._params):

            if p['free']:
                continue

            if p['src_name'] in cache_src_names:
                continue

            if not np.isclose(p['value'], params[i]['value']):
                return False

            if not np.isclose(p['scale'], params[i]['scale']):
                return False

        return True

    def update_source(self, name):

        src_names = self.fitcache.templateSourceNames()
        if not name in src_names:
            return

        self.fitcache.updateTemplateForSource(str(name))

    def refactor(self):

        npar = len(self.params)
        self._free_pars = [True] * npar
        par_scales = np.ones(npar)
        ref_vals = np.array(self.fitcache.refValues())
        cache_src_names = np.array(self.fitcache.templateSourceNames())

        update_sources = []
        all_params = gtutils.get_params_dict(self._like)

        for src_name in cache_src_names:

            pars0 = all_params[src_name]
            pars1 = self._all_params[src_name]
            for i, (p0, p1) in enumerate(zip(pars0, pars1)):

                if p0['is_norm']:
                    continue

                if (np.isclose(p0['value'], p1['value']) and
                        np.isclose(p0['scale'], p1['scale'])):
                    continue

                update_sources += [src_name]

        for i, p in enumerate(self.params):
            self._free_pars[i] = self._like[p['idx']].isFree()
            par_scales[i] = self._like[p['idx']].getValue() / ref_vals[i]

        for src_name in update_sources:
            norm_val = self._like.normPar(src_name).getValue()
            par_name = self._like.normPar(src_name).getName()
            bounds = self._like.normPar(src_name).getBounds()
            idx = self._like.par_index(src_name, par_name)
            self._like[idx].setBounds(*utils.update_bounds(1.0, bounds))
            self._like[idx] = 1.0
            self.fitcache.updateTemplateForSource(str(src_name))
            self._like[idx] = norm_val
            self._like[idx].setBounds(*bounds)

        self._all_params = all_params
        self.fitcache.refactorModel(self._free_pars, par_scales, False)

        # Set priors
        self._set_priors_from_like()
        if np.any(self._has_prior):
            self._build_priors()

    def update(self, params, tol, max_iter, init_lambda, use_reduced):

        try:
            self.fitcache.update()
            self.fitcache.setInitLambda(init_lambda)
            self.fitcache.setTolerance(tol)
            self.fitcache.setMaxIter(max_iter)
        except Exception:
            print('ERROR')
            if not self.check_params(params):
                self._init_fitcache(params, tol, max_iter,
                                    init_lambda, use_reduced)
            self.refactor()

    def _set_priors_from_like(self):

        prior_vals, prior_errs, has_prior = gtutils.get_priors(self._like)
        if not np.any(has_prior):
            self._has_prior.fill(False)
            return

        for i, p in enumerate(self.params):
            self._prior_vals[i] = prior_vals[p['idx']]
            self._prior_errs[i] = prior_errs[p['idx']]
            self._has_prior[i] = True  # has_prior[p['idx']]

            if not has_prior[p['idx']]:
                self._prior_errs[i] = 1E3

        self._prior_cov = np.diag(np.array(self._prior_errs)**2)

    def set_priors(self, vals, err):

        self._prior_vals = np.array(vals, ndmin=1)
        self._prior_errs = err
        self._prior_cov = np.diag(np.array(err)**2)
        self._has_prior = np.array([True] * len(vals))

    def _build_priors(self):

        free_pars = np.array(self._free_pars)
        ref_vals = np.array(self.fitcache.refValues())
        pars = self._prior_vals / ref_vals
        cov = np.ravel(self._prior_cov / np.outer(ref_vals, ref_vals))

        pars = pars[free_pars]
        cov = cov[np.ravel(np.outer(free_pars, free_pars))]
        has_prior = self._has_prior[free_pars]

        self.fitcache.buildPriorsFromExternal(pars, cov, has_prior.tolist())

    def fit(self, verbose=0):

        try:
            return self.fitcache.fitCurrent(3, verbose)
        except Exception:
            return self.fitcache.fitCurrent(bool(np.any(self._has_prior)),
                                            False, verbose)
