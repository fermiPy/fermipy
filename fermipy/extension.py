# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import json
import pprint
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import fermipy.config
from fermipy import utils
from fermipy import defaults
from fermipy.config import ConfigSchema
from fermipy.gtutils import SourceMapState, FreeParameterState
from fermipy.timing import Timer
from fermipy.data_struct import MutableNamedTuple
from fermipy import fits_utils
from LikelihoodState import LikelihoodState


class ExtensionFit(object):
    """Mixin class which provides extension fitting to
    `~fermipy.gtanalysis.GTAnalysis`."""

    def extension(self, name, **kwargs):
        """Test this source for spatial extension with the likelihood
        ratio method (TS_ext).  This method will substitute an
        extended spatial model for the given source and perform a
        one-dimensional scan of the spatial extension parameter over
        the range specified with the width parameters.  The 1-D
        profile likelihood is then used to compute the best-fit value,
        upper limit, and TS for extension.  The nuisance parameters
        that will be simultaneously fit when performing the spatial
        scan can be controlled with the ``fix_shape``,
        ``free_background``, and ``free_radius`` options.  By default
        the position of the source will be fixed to its current
        position.  A simultaneous fit to position and extension can be
        performed by setting ``fit_position`` to True.

        Parameters
        ----------
        name : str
            Source name.

        {options}

        optimizer : dict
            Dictionary that overrides the default optimizer settings.

        Returns
        -------
        extension : dict
            Dictionary containing results of the extension analysis.  The same
            dictionary is also saved to the dictionary of this source under
            'extension'.
        """
        timer = Timer.create(start=True)
        name = self.roi.get_source_by_name(name).name

        schema = ConfigSchema(self.defaults['extension'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('prefix', '')
        schema.add_option('outfile', None, '', str)
        config = utils.create_dict(self.config['extension'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)

        self.logger.info('Running extension fit for %s', name)

        free_state = FreeParameterState(self)
        ext = self._extension(name, **config)
        free_state.restore()

        self.logger.info('Finished extension fit.')

        if config['make_plots']:
            self._plotter.make_extension_plots(ext, self.roi,
                                               prefix=config['prefix'])

        outfile = config.get('outfile', None)
        if outfile is None:
            outfile = utils.format_filename(self.workdir, 'ext',
                                            prefix=[config['prefix'],
                                                    name.lower().replace(' ', '_')])
        else:
            outfile = os.path.join(self.workdir,
                                   os.path.splitext(outfile)[0])

        if config['write_fits']:
            self._make_extension_fits(ext, outfile + '.fits')

        if config['write_npy']:
            np.save(outfile + '.npy', dict(ext))

        self.logger.info('Execution time: %.2f s', timer.elapsed_time)
        return ext

    def _extension(self, name, **kwargs):

        spatial_model = kwargs['spatial_model']
        width_min = kwargs['width_min']
        width_max = kwargs['width_max']
        width_nstep = kwargs['width_nstep']
        width = kwargs['width']
        free_background = kwargs['free_background']
        free_radius = kwargs.get('free_radius', None)
        fix_shape = kwargs.get('fix_shape', False)
        make_tsmap = kwargs.get('make_tsmap', False)
        update = kwargs['update']
        sqrt_ts_threshold = kwargs['sqrt_ts_threshold']

        if kwargs['psf_scale_fn']:
            def psf_scale_fn(t): return 1.0 + np.interp(np.log10(t),
                                                        kwargs['psf_scale_fn'][0],
                                                        kwargs['psf_scale_fn'][1])
        else:
            psf_scale_fn = None

        saved_state = LikelihoodState(self.like)
        loglike_init = -self.like()
        self.logger.debug('Initial Model Log-Likelihood: %f', loglike_init)

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

        # Fit baseline model
        self.free_source(name, loglevel=logging.DEBUG)
        if fix_shape:
            self.free_source(name, free=False, pars='shape',
                             loglevel=logging.DEBUG)

        fit_output = self._fit(loglevel=logging.DEBUG, **kwargs['optimizer'])
        src = self.roi.copy_source(name)

        # Save likelihood value for baseline fit
        saved_state_base = LikelihoodState(self.like)
        loglike_base = fit_output['loglike']
        self.logger.debug('Baseline Model Log-Likelihood: %f', loglike_base)

        if not width:
            width = np.logspace(np.log10(width_min), np.log10(width_max),
                                width_nstep)

        width = np.array(width)
        width = width[width > 0]
        width = np.concatenate(([0.0], np.array(width)))

        o = defaults.make_default_tuple(defaults.extension_output)
        o.name = name
        o.width = width
        o.dloglike = np.zeros(len(width) + 1)
        o.loglike = np.zeros(len(width) + 1)
        o.loglike_base = loglike_base
        o.loglike_init = loglike_init
        o.config = kwargs
        o.ebin_ext = np.ones(self.enumbins) * np.nan
        o.ebin_ext_err = np.ones(self.enumbins) * np.nan
        o.ebin_ext_err_lo = np.ones(self.enumbins) * np.nan
        o.ebin_ext_err_hi = np.ones(self.enumbins) * np.nan
        o.ebin_ext_ul95 = np.ones(self.enumbins) * np.nan
        o.ebin_ts_ext = np.ones(self.enumbins) * np.nan
        o.ebin_loglike = np.ones((self.enumbins, len(width))) * np.nan
        o.ebin_dloglike = np.ones((self.enumbins, len(width))) * np.nan
        o.ebin_loglike_ptsrc = np.ones(self.enumbins) * np.nan
        o.ebin_loglike_ext = np.ones(self.enumbins) * np.nan
        o.ebin_e_min = self.energies[:-1]
        o.ebin_e_max = self.energies[1:]
        o.ebin_e_ctr = np.sqrt(o.ebin_e_min * o.ebin_e_max)

        self.logger.debug('Width scan vector:\n %s', width)

        if kwargs['fit_position']:
            ext_fit = self._fit_extension_full(name,
                                               spatial_model=spatial_model,
                                               optimizer=kwargs['optimizer'])
        else:
            ext_fit = self._fit_extension(name,
                                          spatial_model=spatial_model,
                                          optimizer=kwargs['optimizer'],
                                          psf_scale_fn=psf_scale_fn)

        o.update(ext_fit)

        # Fit with the best-fit extension model
        self.logger.info('Fitting extended-source model.')

        self.set_source_morphology(name, spatial_model=spatial_model,
                                   spatial_pars={'ra': o['ra'], 'dec': o['dec'],
                                                 'SpatialWidth': o['ext']},
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        # Perform scan over width parameter
        o.loglike = self._scan_extension(name,
                                         spatial_model=spatial_model,
                                         width=width,
                                         optimizer=kwargs['optimizer'],
                                         psf_scale_fn=psf_scale_fn)

        self.set_source_morphology(name, spatial_model=spatial_model,
                                   spatial_pars={'ra': o['ra'], 'dec': o['dec'],
                                                 'SpatialWidth': o['ext']},
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        fit_output = self._fit(loglevel=logging.DEBUG, update=False,
                               **kwargs['optimizer'])

        o.source_fit = self.get_src_model(name, reoptimize=True,
                                          optimizer=kwargs['optimizer'])
        o.loglike_ext = fit_output['loglike']

        if kwargs['fit_ebin']:
            self._fit_extension_ebin(name, o, **kwargs)

        if kwargs['save_model_map']:
            o.ext_tot_map = self.model_counts_map()
            o.ext_src_map = self.model_counts_map(name)
            o.ext_bkg_map = self.model_counts_map(exclude=[name])

        if make_tsmap:
            tsmap_model = {'SpatialModel': 'RadialDisk',
                           'SpatialWidth': 0.1 * 0.8246211251235321}
            tsmap_model.update(src.spectral_pars)
            self.logger.info('Generating TS map.')
            tsmap = self.tsmap(model=tsmap_model,
                               map_skydir=SkyCoord(
                                   o['ra'], o['dec'], unit='deg'),
                               map_size=max(1.0, 4.0 * o['ext']),
                               exclude=[name],
                               write_fits=False,
                               write_npy=False,
                               use_pylike=False,
                               make_plots=False,
                               loglevel=logging.DEBUG)
            o.tsmap = tsmap['ts']

        self.logger.info('Testing point-source model.')
        # Test point-source hypothesis
        self.set_source_morphology(name, spatial_model='PointSource',
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        # Fit a point-source
        saved_state_base.restore()
        self.logger.debug('Fitting point-source model.')
        fit_output = self._fit(loglevel=logging.DEBUG, **kwargs['optimizer'])

        if src['SpatialModel'] == 'PointSource' and kwargs['fit_position']:
            loc = self.localize(name, update=False)
            o.loglike_ptsrc = loc['loglike_loc']
        else:
            o.loglike_ptsrc = fit_output['loglike']

        o.dloglike = o.loglike - o.loglike_ptsrc
        o.ts_ext = 2 * (o.loglike_ext - o.loglike_ptsrc)
        self.logger.debug('Point-Source Model Likelihood: %f', o.loglike_ptsrc)

        if kwargs['save_model_map']:
            o.ptsrc_tot_map = self.model_counts_map()
            o.ptsrc_src_map = self.model_counts_map(name)
            o.ptsrc_bkg_map = self.model_counts_map(exclude=[name])

        if update and (sqrt_ts_threshold is None or
                       np.sqrt(o['ts_ext']) > sqrt_ts_threshold):
            src = self.delete_source(name, loglevel=logging.DEBUG)
            src.set_spatial_model(spatial_model,
                                  {'ra': o.ra, 'dec': o.dec,
                                   'SpatialWidth': o.ext})
            # FIXME: Issue with source map cache with source is
            # initialized as fixed.
            self.add_source(name, src, free=True, loglevel=logging.DEBUG)
            self.free_source(name, loglevel=logging.DEBUG)
            if fix_shape:
                self.free_source(name, free=False, pars='shape',
                                 loglevel=logging.DEBUG)
            fit_output = self.fit(loglevel=logging.DEBUG,
                                  **kwargs['optimizer'])
            o.loglike_ext = fit_output['loglike']

            src = self.roi.get_source_by_name(name)
            if kwargs['fit_position']:
                for k in ['ra_err', 'dec_err', 'glon_err', 'glat_err',
                          'pos_err', 'pos_err_semimajor', 'pos_err_semiminor',
                          'pos_r68', 'pos_r95', 'pos_r99', 'pos_angle']:
                    src[k] = o[k]

        else:
            self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                       spatial_pars=src.spatial_pars,
                                       update_source=False)
            # Restore ROI to previous state
            saved_state.restore()
            self._sync_params(name)
            self._update_roi()

        self.logger.info('Best-fit extension: %6.4f + %6.4f - %6.4f'
                         % (o['ext'], o['ext_err_hi'], o['ext_err_lo']))
        self.logger.info('TS_ext:        %.3f' % o['ts_ext'])
        self.logger.info('Extension UL: %6.4f' % o['ext_ul95'])
        self.logger.info('LogLike: %12.3f DeltaLogLike: %12.3f',
                         o.loglike_ext, o.loglike_ext - o.loglike_init)

        if kwargs['fit_position']:
            self.logger.info('Position:\n'
                             '(  ra, dec) = (%10.4f +/- %8.4f,%10.4f +/- %8.4f)\n'
                             '(glon,glat) = (%10.4f +/- %8.4f,%10.4f +/- %8.4f)\n'
                             'offset = %8.4f r68 = %8.4f r95 = %8.4f r99 = %8.4f',
                             o.ra, o.ra_err, o.dec, o.dec_err,
                             o.glon, o.glon_err, o.glat, o.glat_err,
                             o.pos_offset, o.pos_r68, o.pos_r95, o.pos_r99)

        return o

    def _make_extension_fits(self, ext, filename, **kwargs):

        maps = {'EXT_TOT_MAP': ext['ext_tot_map'],
                'EXT_SRC_MAP': ext['ext_src_map'],
                'EXT_BKG_MAP': ext['ext_bkg_map'],
                'PTSRC_TOT_MAP': ext['ptsrc_tot_map'],
                'PTSRC_SRC_MAP': ext['ptsrc_src_map'],
                'PTSRC_BKG_MAP': ext['ptsrc_bkg_map']}

        hdu_images = []
        for k, v in sorted(maps.items()):
            if v is None:
                continue
            hdu_images += [v.make_hdu(k)]

        tab = fits_utils.dict_to_table(ext)
        hdu_data = fits.table_to_hdu(tab)
        hdu_data.name = 'EXT_DATA'

        if ext.get('tsmap'):
            hdus = [ext['tsmap'].make_hdu(hdu='PRIMARY')]
        else:
            hdus = [fits.PrimaryHDU()]

        hdus += [hdu_data] + hdu_images
        hdus[0].header['CONFIG'] = json.dumps(utils.tolist(ext['config']))
        hdus[1].header['CONFIG'] = json.dumps(utils.tolist(ext['config']))
        fits_utils.write_hdus(hdus, filename,
                              keywords={'SRCNAME': ext['name']})

    def _fit_extension_ebin(self, name, o, **kwargs):

        optimizer = kwargs.get('optimizer', {})
        spatial_model = kwargs.get('spatial_model')
        psf_scale_fn = kwargs.pop('psf_scale_fn', None)
        reoptimize = kwargs.pop('reoptimize', True)

        src = self.roi.copy_source(name)
        self.set_source_morphology(name, spatial_model='PointSource',
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        for i, (logemin, logemax) in enumerate(zip(self.log_energies[:-1],
                                                   self.log_energies[1:])):

            self.set_energy_range(logemin, logemax)
            o.ebin_loglike_ptsrc[i] = -self.like()

        self.set_energy_range(self.log_energies[0], self.log_energies[-1])
        self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                   spatial_pars=src.spatial_pars,
                                   psf_scale_fn=psf_scale_fn,
                                   use_pylike=False)

        o.ebin_loglike = self._scan_extension_fast_ebin(name,
                                                        spatial_model=spatial_model,
                                                        width=o.width,
                                                        optimizer=kwargs[
                                                            'optimizer'],
                                                        psf_scale_fn=psf_scale_fn,
                                                        reoptimize=False)

        for i, (logemin, logemax) in enumerate(zip(self.log_energies[:-1],
                                                   self.log_energies[1:])):
            ul_data = utils.get_parameter_limits(o.width, o.ebin_loglike[i])
            o.ebin_ext[i] = max(ul_data['x0'], 10**-2.5)
            o.ebin_ext_err[i] = ul_data['err']
            o.ebin_ext_err_lo[i] = ul_data['err_lo']
            o.ebin_ext_err_hi[i] = ul_data['err_hi']
            o.ebin_ext_ul95[i] = ul_data['ul']
            o.ebin_loglike_ext[i] = ul_data['lnlmax']
            o.ebin_ts_ext[i] = 2.0 * \
                (o.ebin_loglike_ext[i] - o.ebin_loglike_ptsrc[i])

        o.ebin_dloglike = o.ebin_loglike - o.ebin_loglike_ptsrc[:, None]
        self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                   spatial_pars=src.spatial_pars,
                                   use_pylike=False)

    def _scan_extension(self, name, **kwargs):

        saved_state = LikelihoodState(self.like)

        if not hasattr(self.components[0].like.logLike, 'setSourceMapImage'):
            loglike = self._scan_extension_pylike(name, **kwargs)
        else:
            loglike = self._scan_extension_fast(name, **kwargs)

        saved_state.restore()

        return loglike

    def _scan_extension_fast(self, name, **kwargs):

        state = SourceMapState(self.like, [name])

        self.free_norm(name)

        optimizer = kwargs.get('optimizer', {})
        width = kwargs.get('width')
        spatial_model = kwargs.get('spatial_model')
        skydir = kwargs.pop('skydir', self.roi[name].skydir)
        psf_scale_fn = kwargs.pop('psf_scale_fn', None)
        reoptimize = kwargs.pop('reoptimize', True)

        src = self.roi.copy_source(name)
        spatial_pars = {'ra': skydir.ra.deg, 'dec': skydir.dec.deg}

        loglike = []
        for i, w in enumerate(width):

            spatial_pars['SpatialWidth'] = max(w, 0.00316)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars=spatial_pars,
                                       use_pylike=False,
                                       psf_scale_fn=psf_scale_fn)
            if reoptimize:
                fit_output = self._fit(loglevel=logging.DEBUG, **optimizer)
                loglike += [fit_output['loglike']]
            else:
                loglike += [-self.like()]

        state.restore()

        return np.array(loglike)

    def _scan_extension_fast_ebin(self, name, **kwargs):

        state = SourceMapState(self.like, [name])

        self.free_norm(name)
        optimizer = kwargs.get('optimizer', {})
        width = kwargs.get('width')
        spatial_model = kwargs.get('spatial_model')
        skydir = kwargs.pop('skydir', self.roi[name].skydir)
        psf_scale_fn = kwargs.pop('psf_scale_fn', None)
        reoptimize = kwargs.pop('reoptimize', True)

        src = self.roi.copy_source(name)
        spatial_pars = {'ra': skydir.ra.deg, 'dec': skydir.dec.deg}

        loglike = np.ones((self.enumbins, len(width)))
        for i, w in enumerate(width):

            spatial_pars['SpatialWidth'] = max(w, 0.00316)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars=spatial_pars,
                                       use_pylike=False,
                                       psf_scale_fn=psf_scale_fn)

            for j, (logemin, logemax) in enumerate(zip(self.log_energies[:-1],
                                                       self.log_energies[1:])):
                self.set_energy_range(logemin, logemax)
                if reoptimize:
                    fit_output = self._fit(loglevel=logging.DEBUG, **optimizer)
                    loglike[j, i] = fit_output['loglike']
                else:
                    loglike[j, i] = -self.like()
            self.set_energy_range(self.log_energies[0], self.log_energies[-1])

        state.restore()
        return loglike

    def _scan_extension_pylike(self, name, **kwargs):

        optimizer = kwargs.get('optimizer', {})
        width = kwargs.get('width')
        spatial_model = kwargs.get('spatial_model')
        skydir = kwargs.pop('skydir', self.roi[name].skydir)

        src = self.roi.copy_source(name)

        self._fitcache = None

        spatial_pars = {'ra': skydir.ra.deg, 'dec': skydir.dec.deg}

        loglike = []
        for i, w in enumerate(width):

            spatial_pars['SpatialWidth'] = max(w, 0.00316)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars=spatial_pars,
                                       use_pylike=False)
            fit_output = self._fit(loglevel=logging.DEBUG, **optimizer)
            loglike += [fit_output['loglike']]

        self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                   spatial_pars=src.spatial_pars)

        return np.array(loglike)

    def _fit_extension(self, name, **kwargs):

        spatial_model = kwargs.get('spatial_model', 'RadialGaussian')
        optimizer = kwargs.get('optimizer', {})
        fit_position = kwargs.get('fit_position', False)
        skydir = kwargs.get('skydir', self.roi[name].skydir)
        psf_scale_fn = kwargs.get('psf_scale_fn', None)
        reoptimize = kwargs.get('reoptimize', True)

        src = self.roi.copy_source(name)

        # If the source is extended split the likelihood scan into two
        # parts centered on the best-fit value -- this ensures better
        # fit stability
        if (src['SpatialModel'] in ['RadialGaussian', 'RadialDisk'] and
                src['SpatialWidth'] > 0.1):
            width_lo = np.logspace(-2.0, np.log10(src['SpatialWidth']), 11)
            width_hi = np.logspace(np.log10(src['SpatialWidth']), 0.5, 11)
            loglike_lo = self._scan_extension(name, spatial_model=spatial_model,
                                              width=width_lo[::-1],
                                              optimizer=optimizer,
                                              skydir=skydir,
                                              psf_scale_fn=psf_scale_fn,
                                              reoptimize=reoptimize)[::-1]
            loglike_hi = self._scan_extension(name, spatial_model=spatial_model,
                                              width=width_hi,
                                              optimizer=optimizer,
                                              skydir=skydir,
                                              psf_scale_fn=psf_scale_fn,
                                              reoptimize=reoptimize)
            width = np.concatenate((width_lo, width_hi[1:]))
            loglike = np.concatenate((loglike_lo, loglike_hi[1:]))
        else:
            width = np.logspace(-2.0, 0.5, 21)
            width = np.concatenate(([0.0], width))
            loglike = self._scan_extension(name, spatial_model=spatial_model,
                                           width=width, optimizer=optimizer,
                                           skydir=skydir,
                                           psf_scale_fn=psf_scale_fn,
                                           reoptimize=reoptimize)

        ul_data = utils.get_parameter_limits(width, loglike,
                                             bounds=[10**-3.0, 10**0.5])

        if not np.isfinite(ul_data['err']):
            ul_data['x0'] = width[np.argmax(loglike)]
            ul_data['err'] = ul_data['x0']
            ul_data['err_lo'] = ul_data['x0']
            ul_data['err_hi'] = ul_data['x0']

        imax = np.argmax(loglike)
        err = max(10**-2.0, ul_data['err'])
        lolim = max(min(ul_data['x0'], width[imax]) - 2.0 * err, 0)

        if np.isfinite(ul_data['ul']):
            hilim = 1.5 * ul_data['ul']
        else:
            hilim = ul_data['x0'] + 2.0 * err

        nstep = max(11, int((hilim - lolim) / err))
        width2 = np.linspace(lolim, hilim, nstep)

        loglike2 = self._scan_extension(name, spatial_model=spatial_model,
                                        width=width2, optimizer=optimizer,
                                        skydir=skydir,
                                        psf_scale_fn=psf_scale_fn,
                                        reoptimize=reoptimize)
        ul_data2 = utils.get_parameter_limits(width2, loglike2,
                                              bounds=[10**-3.0, 10**0.5])

        self.logger.debug('width: %s', width)
        self.logger.debug('loglike: %s', loglike - np.max(loglike))
        self.logger.debug('ul_data:\n %s', pprint.pformat(ul_data))
        self.logger.debug('width2: %s', width2)
        self.logger.debug('loglike2: %s', loglike2 - np.max(loglike2))
        self.logger.debug('ul_data2:\n %s', pprint.pformat(ul_data2))

        return MutableNamedTuple(
            ext=max(ul_data2['x0'], 10**-2.5),
            ext_ul95=ul_data2['ul'],
            ext_err_lo=ul_data2['err_lo'],
            ext_err_hi=ul_data2['err_hi'],
            ext_err=ul_data2['err'],
            loglike_ext=ul_data2['lnlmax'],
            ra=skydir.ra.deg,
            dec=skydir.dec.deg,
            glon=skydir.galactic.l.deg,
            glat=skydir.galactic.b.deg,
            pos_offset=0.0)

    def _fit_extension_full(self, name, **kwargs):

        skydir = self.roi[name].skydir.copy()
        src = self.roi.copy_source(name)
        spatial_model = kwargs.get('spatial_model', 'RadialGaussian')
        loglike = -self.like()

        o = MutableNamedTuple(
            ext=np.nan, ext_ul95=np.nan,
            ext_err_lo=np.nan, ext_err_hi=np.nan,
            ext_err=np.nan,
            loglike_ext=np.nan,
            ra=np.nan, dec=np.nan, glon=np.nan, glat=np.nan,
            ra_err=np.nan, dec_err=np.nan,
            glon_err=np.nan, glat_err=np.nan,
            pos_err=np.nan, pos_r68=np.nan,
            pos_r95=np.nan, pos_r99=np.nan,
            pos_err_semimajor=np.nan, pos_err_semiminor=np.nan,
            pos_angle=np.nan, pos_offset=np.nan)

        t0 = Timer()
        t1 = Timer()

        nstep = 7
        for i in range(4):

            t0.start()
            fit_ext = self._fit_extension(name, skydir=skydir, **kwargs)
            o.update(fit_ext)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars={
                                           'SpatialWidth': max(o.ext, 0.00316)},
                                       use_pylike=False)
            t0.stop()

            t1.start()
            if i == 0 or not np.isfinite(o.pos_r99):
                dtheta_max = max(0.5, 1.5 * o.ext)
            else:
                dtheta_max = max(0.5, o.pos_r99)

            fit_pos0, fit_pos1 = self._fit_position(name, nstep=nstep,
                                                    dtheta_max=dtheta_max,
                                                    zmin=-3.0, use_pylike=False)
            o.update(fit_pos0)
            t1.stop()

            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars={'RA': o['ra'],
                                                     'DEC': o['dec']},
                                       use_pylike=False)

            self.logger.debug('Elapsed Time: %.2f %.2f',
                              t0.elapsed_time, t1.elapsed_time)

            fit_output = self._fit(
                loglevel=logging.DEBUG, **kwargs['optimizer'])

            skydir = fit_pos0['skydir']
            o.pos_offset = skydir.separation(src.skydir).deg
            o.loglike_ext = -self.like()
            dloglike = o.loglike_ext - loglike

            self.logger.info('Iter %i R68 = %8.3f Offset = %8.3f '
                             'LogLikelihood = %10.2f Delta-LogLikelihood = %8.2f',
                             i, o.ext, o.pos_offset, o.loglike_ext,
                             dloglike)

            if i > 0 and dloglike < 0.1:
                break

            loglike = o.loglike_ext

        self.set_source_morphology(name, spatial_pars=src.spatial_pars,
                                   use_pylike=False)
        return o
