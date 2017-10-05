# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for dealing with SEDs

Many parts of this code are taken from dsphs/like/lnlfn.py by
  Matthew Wood <mdwood@slac.stanford.edu>
  Alex Drlica-Wagner <kadrlica@slac.stanford.edu>
"""
from __future__ import absolute_import, division, print_function
import copy
import logging
import os
import json

import numpy as np

from astropy.io import fits
from astropy.table import Table, Column

import fermipy.config
from fermipy import utils
from fermipy import gtutils
from fermipy import fits_utils
from fermipy import roi_model
from fermipy.config import ConfigSchema
from fermipy.timing import Timer
from fermipy import model_utils

from LikelihoodState import LikelihoodState
import pyLikelihood as pyLike


class SEDGenerator(object):
    """Mixin class that provides SED functionality to
    `~fermipy.gtanalysis.GTAnalysis`."""

    def sed(self, name, **kwargs):
        """Generate a spectral energy distribution (SED) for a source.  This
        function will fit the normalization of the source in each
        energy bin.  By default the SED will be generated with the
        analysis energy bins but a custom binning can be defined with
        the ``loge_bins`` parameter.

        Parameters
        ----------
        name : str
            Source name.

        prefix : str
            Optional string that will be prepended to all output files
            (FITS and rendered images).

        loge_bins : `~numpy.ndarray`
            Sequence of energies in log10(E/MeV) defining the edges of
            the energy bins.  If this argument is None then the
            analysis energy bins will be used.  The energies in this
            sequence must align with the bin edges of the underyling
            analysis instance.

        {options}

        optimizer : dict
            Dictionary that overrides the default optimizer settings.

        Returns
        -------
        sed : dict
            Dictionary containing output of the SED analysis.

        """
        timer = Timer.create(start=True)
        name = self.roi.get_source_by_name(name).name

        # Create schema for method configuration
        schema = ConfigSchema(self.defaults['sed'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('prefix', '')
        schema.add_option('outfile', None, '', str)
        schema.add_option('loge_bins', None, '', list)
        config = utils.create_dict(self.config['sed'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)

        self.logger.info('Computing SED for %s' % name)

        o = self._make_sed(name, **config)

        self.logger.info('Finished SED')

        outfile = config.get('outfile', None)
        if outfile is None:
            outfile = utils.format_filename(self.workdir, 'sed',
                                            prefix=[config['prefix'],
                                                    name.lower().replace(' ', '_')])
        else:
            outfile = os.path.join(self.workdir,
                                   os.path.splitext(outfile)[0])

        o['file'] = None
        if config['write_fits']:
            o['file'] = os.path.basename(outfile) + '.fits'
            self._make_sed_fits(o, outfile + '.fits', **config)

        if config['write_npy']:
            np.save(outfile + '.npy', o)

        if config['make_plots']:
            self._plotter.make_sed_plots(o, **config)

        self.logger.info('Execution time: %.2f s', timer.elapsed_time)
        return o

    def _make_sed_fits(self, sed, filename, **kwargs):

        # Write a FITS file
        cols = [Column(name='e_min', dtype='f8', data=sed['e_min'], unit='MeV'),
                Column(name='e_ref', dtype='f8',
                       data=sed['e_ref'], unit='MeV'),
                Column(name='e_max', dtype='f8',
                       data=sed['e_max'], unit='MeV'),
                Column(name='ref_dnde_e_min', dtype='f8',
                       data=sed['ref_dnde_e_min'], unit='ph / (MeV cm2 s)'),
                Column(name='ref_dnde_e_max', dtype='f8',
                       data=sed['ref_dnde_e_max'], unit='ph / (MeV cm2 s)'),
                Column(name='ref_dnde', dtype='f8',
                       data=sed['ref_dnde'], unit='ph / (MeV cm2 s)'),
                Column(name='ref_flux', dtype='f8',
                       data=sed['ref_flux'], unit='ph / (cm2 s)'),
                Column(name='ref_eflux', dtype='f8',
                       data=sed['ref_eflux'], unit='MeV / (cm2 s)'),
                Column(name='ref_npred', dtype='f8', data=sed['ref_npred']),
                Column(name='dnde', dtype='f8',
                       data=sed['dnde'], unit='ph / (MeV cm2 s)'),
                Column(name='dnde_err', dtype='f8',
                       data=sed['dnde_err'], unit='ph / (MeV cm2 s)'),
                Column(name='dnde_errp', dtype='f8',
                       data=sed['dnde_err_hi'], unit='ph / (MeV cm2 s)'),
                Column(name='dnde_errn', dtype='f8',
                       data=sed['dnde_err_lo'], unit='ph / (MeV cm2 s)'),
                Column(name='dnde_ul', dtype='f8',
                       data=sed['dnde_ul'], unit='ph / (MeV cm2 s)'),
                Column(name='e2dnde', dtype='f8',
                       data=sed['e2dnde'], unit='MeV / (cm2 s)'),
                Column(name='e2dnde_err', dtype='f8',
                       data=sed['e2dnde_err'], unit='MeV / (cm2 s)'),
                Column(name='e2dnde_errp', dtype='f8',
                       data=sed['e2dnde_err_hi'], unit='MeV / (cm2 s)'),
                Column(name='e2dnde_errn', dtype='f8',
                       data=sed['e2dnde_err_lo'], unit='MeV / (cm2 s)'),
                Column(name='e2dnde_ul', dtype='f8',
                       data=sed['e2dnde_ul'], unit='MeV / (cm2 s)'),
                Column(name='norm', dtype='f8', data=sed['norm']),
                Column(name='norm_err', dtype='f8', data=sed['norm_err']),
                Column(name='norm_errp', dtype='f8', data=sed['norm_err_hi']),
                Column(name='norm_errn', dtype='f8', data=sed['norm_err_lo']),
                Column(name='norm_ul', dtype='f8', data=sed['norm_ul95']),
                Column(name='ts', dtype='f8', data=sed['ts']),
                Column(name='loglike', dtype='f8', data=sed['loglike']),
                Column(name='norm_scan', dtype='f8', data=sed['norm_scan']),
                Column(name='dloglike_scan', dtype='f8',
                       data=sed['dloglike_scan']),

                ]

        tab = Table(cols)
        tab.meta['UL_CONF'] = 0.95
        hdu_sed = fits.table_to_hdu(tab)
        hdu_sed.name = 'SED'

        columns = fits.ColDefs([])

        columns.add_col(fits.Column(name=str('energy'), format='E',
                                    array=sed['model_flux']['energies'],
                                    unit='MeV'))
        columns.add_col(fits.Column(name=str('dnde'), format='E',
                                    array=sed['model_flux']['dnde'],
                                    unit='ph / (MeV cm2 s)'))
        columns.add_col(fits.Column(name=str('dnde_lo'), format='E',
                                    array=sed['model_flux']['dnde_lo'],
                                    unit='ph / (MeV cm2 s)'))
        columns.add_col(fits.Column(name=str('dnde_hi'), format='E',
                                    array=sed['model_flux']['dnde_hi'],
                                    unit='ph / (MeV cm2 s)'))
        columns.add_col(fits.Column(name=str('dnde_err'), format='E',
                                    array=sed['model_flux']['dnde_err'],
                                    unit='ph / (MeV cm2 s)'))
        columns.add_col(fits.Column(name=str('dnde_ferr'), format='E',
                                    array=sed['model_flux']['dnde_ferr']))

        hdu_f = fits.BinTableHDU.from_columns(columns, name='MODEL_FLUX')

        columns = fits.ColDefs([])

        npar = len(sed['param_names'])
        columns.add_col(fits.Column(name=str('name'),
                                    format='A32',
                                    array=sed['param_names']))
        columns.add_col(fits.Column(name=str('value'), format='E',
                                    array=sed['param_values']))
        columns.add_col(fits.Column(name=str('error'), format='E',
                                    array=sed['param_errors']))
        columns.add_col(fits.Column(name=str('covariance'),
                                    format='%iE' % npar,
                                    dim=str('(%i)' % npar),
                                    array=sed['param_covariance']))
        columns.add_col(fits.Column(name=str('correlation'),
                                    format='%iE' % npar,
                                    dim=str('(%i)' % npar),
                                    array=sed['param_correlation']))

        hdu_p = fits.BinTableHDU.from_columns(columns, name='PARAMS')

        hdus = [fits.PrimaryHDU(), hdu_sed, hdu_f, hdu_p]
        hdus[0].header['CONFIG'] = json.dumps(sed['config'])
        hdus[1].header['CONFIG'] = json.dumps(sed['config'])

        fits_utils.write_hdus(hdus, filename,
                              keywords={'SRCNAME': sed['name']})

    def _make_sed(self, name, **config):

        bin_index = config['bin_index']
        use_local_index = config['use_local_index']
        free_background = config['free_background']
        free_radius = config['free_radius']
        ul_confidence = config['ul_confidence']
        cov_scale = config['cov_scale']
        loge_bins = config['loge_bins']

        if not loge_bins or loge_bins is None:
            loge_bins = self.log_energies
        else:
            loge_bins = np.array(loge_bins)

        nbins = len(loge_bins) - 1
        max_index = 5.0
        min_flux = 1E-30
        npts = self.config['gtlike']['llscan_npts']
        loge_bounds = self.loge_bounds

        # Output Dictionary
        o = {'name': name,
             'loge_min': loge_bins[:-1],
             'loge_max': loge_bins[1:],
             'loge_ctr': 0.5 * (loge_bins[:-1] + loge_bins[1:]),
             'loge_ref': 0.5 * (loge_bins[:-1] + loge_bins[1:]),
             'e_min': 10 ** loge_bins[:-1],
             'e_max': 10 ** loge_bins[1:],
             'e_ctr': 10 ** (0.5 * (loge_bins[:-1] + loge_bins[1:])),
             'e_ref': 10 ** (0.5 * (loge_bins[:-1] + loge_bins[1:])),
             'ref_flux': np.zeros(nbins),
             'ref_eflux': np.zeros(nbins),
             'ref_dnde': np.zeros(nbins),
             'ref_dnde_e_min': np.zeros(nbins),
             'ref_dnde_e_max': np.zeros(nbins),
             'ref_e2dnde': np.zeros(nbins),
             'ref_npred': np.zeros(nbins),
             'norm': np.zeros(nbins),
             'flux': np.zeros(nbins),
             'eflux': np.zeros(nbins),
             'dnde': np.zeros(nbins),
             'e2dnde': np.zeros(nbins),
             'index': np.zeros(nbins),
             'npred': np.zeros(nbins),
             'ts': np.zeros(nbins),
             'loglike': np.zeros(nbins),
             'norm_scan': np.zeros((nbins, npts)),
             'dloglike_scan': np.zeros((nbins, npts)),
             'loglike_scan': np.zeros((nbins, npts)),
             'fit_quality': np.zeros(nbins),
             'fit_status': np.zeros(nbins),
             'correlation': {},
             'model_flux': {},
             'config': config
             }

        for t in ['norm', 'flux', 'eflux', 'dnde', 'e2dnde']:
            o['%s_err' % t] = np.zeros(nbins) * np.nan
            o['%s_err_hi' % t] = np.zeros(nbins) * np.nan
            o['%s_err_lo' % t] = np.zeros(nbins) * np.nan
            o['%s_ul95' % t] = np.zeros(nbins) * np.nan
            o['%s_ul' % t] = np.zeros(nbins) * np.nan

        saved_state = LikelihoodState(self.like)
        source = self.components[0].like.logLike.getSource(str(name))

        # Perform global spectral fit
        self._latch_free_params()
        self.free_sources(False, pars='shape', loglevel=logging.DEBUG)
        self.free_source(name, pars=config.get('free_pars', None),
                         loglevel=logging.DEBUG)
        fit_output = self.fit(loglevel=logging.DEBUG, update=False,
                              min_fit_quality=2)
        o['model_flux'] = self.bowtie(name)
        spectral_pars = gtutils.get_function_pars_dict(source.spectrum())
        o['SpectrumType'] = self.roi[name]['SpectrumType']
        o.update(model_utils.pars_dict_to_vectors(o['SpectrumType'],
                                                  spectral_pars))

        param_names = gtutils.get_function_par_names(o['SpectrumType'])
        npar = len(param_names)
        o['param_covariance'] = np.empty((npar, npar), dtype=float) * np.nan

        pmask0 = np.empty(len(fit_output['par_names']), dtype=bool)
        pmask0.fill(False)
        pmask1 = np.empty(npar, dtype=bool)
        pmask1.fill(False)
        for i, pname in enumerate(param_names):

            for j, pname2 in enumerate(fit_output['par_names']):
                if name != fit_output['src_names'][j]:
                    continue
                if pname != pname2:
                    continue
                pmask0[j] = True
                pmask1[i] = True

        src_cov = fit_output['covariance'][pmask0, :][:, pmask0]
        o['param_covariance'][np.ix_(pmask1, pmask1)] = src_cov
        o['param_correlation'] = utils.cov_to_correlation(
            o['param_covariance'])

        for i, pname in enumerate(param_names):
            o['param_covariance'][i, :] *= spectral_pars[pname]['scale']
            o['param_covariance'][:, i] *= spectral_pars[pname]['scale']

        self._restore_free_params()

        self.logger.info('Fitting SED')

        # Setup background parameters for SED
        self.free_sources(False, pars='shape')
        self.free_norm(name)

        if not free_background:
            self.free_sources(free=False, loglevel=logging.DEBUG)

        if free_radius is not None:
            diff_sources = [s.name for s in self.roi.sources if s.diffuse]
            skydir = self.roi[name].skydir
            free_srcs = [s.name for s in
                         self.roi.get_sources(skydir=skydir,
                                              distance=free_radius,
                                              exclude=diff_sources)]
            self.free_sources_by_name(free_srcs, pars='norm',
                                      loglevel=logging.DEBUG)

        if cov_scale is not None:
            self._latch_free_params()
            self.zero_source(name)
            self.fit(loglevel=logging.DEBUG, update=False)
            srcNames = list(self.like.sourceNames())
            srcNames.remove(name)
            self.constrain_norms(srcNames, cov_scale)
            self.unzero_source(name)
            self._restore_free_params()

        # Precompute fluxes in each bin from global fit
        gf_bin_flux = []
        gf_bin_index = []
        for i, (logemin, logemax) in enumerate(zip(loge_bins[:-1],
                                                   loge_bins[1:])):

            emin = 10 ** logemin
            emax = 10 ** logemax
            delta = 1E-5
            f = self.like[name].flux(emin, emax)
            f0 = self.like[name].flux(emin * (1 - delta), emin * (1 + delta))
            f1 = self.like[name].flux(emax * (1 - delta), emax * (1 + delta))

            if f0 > min_flux and f1 > min_flux:
                g = 1 - np.log10(f0 / f1) / np.log10(emin / emax)
                gf_bin_index += [g]
                gf_bin_flux += [f]
            else:
                gf_bin_index += [max_index]
                gf_bin_flux += [min_flux]

        old_spectrum = source.spectrum()
        old_pars = copy.deepcopy(self.roi[name].spectral_pars)
        old_type = self.roi[name]['SpectrumType']

        spectrum_pars = {
            'Prefactor':
                {'value': 1.0, 'scale': 1E-13, 'min': 1E-10,
                    'max': 1E10, 'free': True},
            'Index':
                {'value': 2.0, 'scale': -1.0, 'min': 0.0, 'max': 5.0, 'free': False},
            'Scale':
                {'value': 1E3, 'scale': 1.0, 'min': 1., 'max': 1E6, 'free': False},
        }

        self.set_source_spectrum(str(name), 'PowerLaw',
                                 spectrum_pars=spectrum_pars,
                                 update_source=False)

        src_norm_idx = -1
        free_params = self.get_params(True)
        for j, p in enumerate(free_params):
            if not p['is_norm']:
                continue
            if p['is_norm'] and p['src_name'] == name:
                src_norm_idx = j

            o['correlation'][p['src_name']] = np.zeros(nbins) * np.nan

        self._fitcache = None

        for i, (logemin, logemax) in enumerate(zip(loge_bins[:-1],
                                                   loge_bins[1:])):

            logectr = 0.5 * (logemin + logemax)
            emin = 10 ** logemin
            emax = 10 ** logemax
            ectr = 10 ** logectr
            ectr2 = ectr**2

            saved_state_bin = LikelihoodState(self.like)
            if use_local_index:
                o['index'][i] = -min(gf_bin_index[i], max_index)
            else:
                o['index'][i] = -bin_index

            self.set_norm(name, 1.0, update_source=False)
            self.set_parameter(name, 'Index', o['index'][i], scale=1.0,
                               update_source=False)
            self.like.syncSrcParams(str(name))

            ref_flux = self.like[name].flux(emin, emax)

            o['ref_flux'][i] = self.like[name].flux(emin, emax)
            o['ref_eflux'][i] = self.like[name].energyFlux(emin, emax)
            o['ref_dnde'][i] = self.like[name].spectrum()(pyLike.dArg(ectr))
            o['ref_dnde_e_min'][i] = self.like[
                name].spectrum()(pyLike.dArg(emin))
            o['ref_dnde_e_max'][i] = self.like[
                name].spectrum()(pyLike.dArg(emax))
            o['ref_e2dnde'][i] = o['ref_dnde'][i] * ectr2
            cs = self.model_counts_spectrum(
                name, logemin, logemax, summed=True)
            o['ref_npred'][i] = np.sum(cs)

            normVal = self.like.normPar(name).getValue()
            flux_ratio = gf_bin_flux[i] / ref_flux
            newVal = max(normVal * flux_ratio, 1E-10)
            self.set_norm(name, newVal, update_source=False)
            self.set_norm_bounds(name, [newVal * 1E-6, newVal * 1E4])

            self.like.syncSrcParams(str(name))
            self.free_norm(name)
            self.logger.debug('Fitting %s SED from %.0f MeV to %.0f MeV' %
                              (name, emin, emax))
            self.set_energy_range(logemin, logemax)

            fit_output = self._fit(**config['optimizer'])
            free_params = self.get_params(True)
            for j, p in enumerate(free_params):

                if not p['is_norm']:
                    continue

                o['correlation'][p['src_name']][i] = \
                    fit_output['correlation'][src_norm_idx, j]

            o['fit_quality'][i] = fit_output['fit_quality']
            o['fit_status'][i] = fit_output['fit_status']

            flux = self.like[name].flux(emin, emax)
            eflux = self.like[name].energyFlux(emin, emax)
            dnde = self.like[name].spectrum()(pyLike.dArg(ectr))

            o['norm'][i] = flux / o['ref_flux'][i]
            o['flux'][i] = flux
            o['eflux'][i] = eflux
            o['dnde'][i] = dnde
            o['e2dnde'][i] = dnde * ectr2

            cs = self.model_counts_spectrum(name, logemin,
                                            logemax, summed=True)
            o['npred'][i] = np.sum(cs)
            o['loglike'][i] = fit_output['loglike']

            lnlp = self.profile_norm(name, logemin=logemin, logemax=logemax,
                                     savestate=True, reoptimize=True,
                                     npts=npts, optimizer=config['optimizer'])

            o['ts'][i] = max(
                2.0 * (fit_output['loglike'] - lnlp['loglike'][0]), 0.0)
            o['loglike_scan'][i] = lnlp['loglike']
            o['dloglike_scan'][i] = lnlp['dloglike']
            o['norm_scan'][i] = lnlp['flux'] / ref_flux

            ul_data = utils.get_parameter_limits(
                lnlp['flux'], lnlp['dloglike'])

            o['norm_err_hi'][i] = ul_data['err_hi'] / ref_flux
            o['norm_err_lo'][i] = ul_data['err_lo'] / ref_flux

            if np.isfinite(ul_data['err_lo']):
                o['norm_err'][i] = 0.5 * (ul_data['err_lo'] +
                                          ul_data['err_hi']) / ref_flux
            else:
                o['norm_err'][i] = ul_data['err_hi'] / ref_flux

            o['norm_ul95'][i] = ul_data['ul'] / ref_flux

            ul_data = utils.get_parameter_limits(lnlp['flux'],
                                                 lnlp['dloglike'],
                                                 cl_limit=ul_confidence)
            o['norm_ul'][i] = ul_data['ul'] / ref_flux

            saved_state_bin.restore()

        for t in ['flux', 'eflux', 'dnde', 'e2dnde']:

            o['%s_err' % t] = o['norm_err'] * o['ref_%s' % t]
            o['%s_err_hi' % t] = o['norm_err_hi'] * o['ref_%s' % t]
            o['%s_err_lo' % t] = o['norm_err_lo'] * o['ref_%s' % t]
            o['%s_ul95' % t] = o['norm_ul95'] * o['ref_%s' % t]
            o['%s_ul' % t] = o['norm_ul'] * o['ref_%s' % t]

        self.set_energy_range(loge_bounds[0], loge_bounds[1])
        self.set_source_spectrum(str(name), old_type,
                                 spectrum_pars=old_pars,
                                 update_source=False)

        saved_state.restore()
        self._sync_params(name)

        if cov_scale is not None:
            self.remove_priors()

        return o


if __name__ == "__main__":

    pass
