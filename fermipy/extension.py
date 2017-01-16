# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import copy
import pprint
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
import fermipy.config
from fermipy import utils
from fermipy import defaults
from fermipy.config import ConfigSchema
from fermipy.gtutils import SourceMapState, FreeParameterState
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
        upper limit, and TS for extension.  Any background parameters
        that are free will also be simultaneously profiled in the
        likelihood scan.

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

        name = self.roi.get_source_by_name(name).name

        schema = ConfigSchema(self.defaults['extension'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('prefix', '')
        config = utils.create_dict(self.config['extension'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)

        self.logger.info('Running extension fit for %s', name)

        free_state = FreeParameterState(self)
        ext = self._extension(name, **config)
        free_state.restore()

        self.logger.info('Finished extension fit.')

        filename = utils.format_filename(self.workdir, 'ext',
                                         prefix=[config['prefix'],
                                                 name.lower().replace(' ', '_')])
        if config['write_fits']:
            self._make_extension_fits(ext, filename + '.fits')

        if config['write_npy']:
            np.save(filename + '.npy', ext)

        return ext

    def _extension(self, name, **kwargs):

        spatial_model = kwargs['spatial_model']
        width_min = kwargs['width_min']
        width_max = kwargs['width_max']
        width_nstep = kwargs['width_nstep']
        width = kwargs['width']
        free_background = kwargs['free_background']
        free_radius = kwargs.get('free_radius', None)
        update = kwargs['update']
        sqrt_ts_threshold = kwargs['sqrt_ts_threshold']

        if kwargs['psf_scale_fn']:
            psf_scale_fn = lambda t: 1.0 + np.interp(np.log10(t),
                                                     kwargs['psf_scale_fn'][0],
                                                     kwargs['psf_scale_fn'][1])
        else:
            psf_scale_fn = None

        saved_state = LikelihoodState(self.like)

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
        self.free_norm(name, loglevel=logging.DEBUG)
        fit_output = self._fit(loglevel=logging.DEBUG, **kwargs['optimizer'])
        src = self.roi.copy_source(name)

        # Save likelihood value for baseline fit
        loglike0 = fit_output['loglike']
        self.logger.debug('Baseline Likelihood: %f', loglike0)

        if not width:
            width = np.logspace(np.log10(width_min), np.log10(width_max),
                                width_nstep)

        width = np.array(width)
        width = width[width > 0]
        width = np.concatenate(([0.0], np.array(width)))

        o = defaults.make_default_dict(defaults.extension_output)
        o['name'] = name
        o['width'] = width
        o['dloglike'] = np.zeros(len(width) + 1)
        o['loglike'] = np.zeros(len(width) + 1)
        o['loglike_base'] = loglike0
        o['config'] = kwargs
        o['ptsrc_tot_map'] = None
        o['ptsrc_src_map'] = None
        o['ptsrc_bkg_map'] = None
        o['ext_tot_map'] = None
        o['ext_src_map'] = None
        o['ext_bkg_map'] = None

        self.set_source_morphology(name, spatial_model='PointSource',
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        # Fit a point-source
        self.logger.debug('Fitting point-source model.')
        fit_output = self._fit(loglevel=logging.DEBUG, **kwargs['optimizer'])
        o['loglike_ptsrc'] = fit_output['loglike']
        self.logger.debug('Point Source Likelihood: %f', o['loglike_ptsrc'])

        if kwargs['save_model_map']:
            o['ptsrc_tot_map'] = self.model_counts_map()
            o['ptsrc_src_map'] = self.model_counts_map(name)
            o['ptsrc_bkg_map'] = self.model_counts_map(exclude=[name])

        # Perform scan over width parameter
        self.logger.debug('Width scan vector:\n %s', width)

        if kwargs['fit_position']:
            ext_fit = self._fit_extension_full(name,
                                               spatial_model=spatial_model,
                                               optimizer=kwargs['optimizer'])
        else:
            ext_fit = self._fit_extension(name,
                                          spatial_model=spatial_model,
                                          optimizer=kwargs['optimizer'])

        o.update(ext_fit)
        o['loglike'] = self._scan_extension(name,
                                            spatial_model=spatial_model,
                                            width=width,
                                            optimizer=kwargs['optimizer'])

        #self.logger.debug('Likelihood: %s',o['loglike'])
        o['dloglike'] = o['loglike'] - o['loglike_ptsrc']
        o['ts_ext'] = 2 * (o['loglike_ext'] - o['loglike_ptsrc'])

        self.logger.info('Best-fit extension: %6.4f + %6.4f - %6.4f'
                         % (o['ext'], o['ext_err_lo'], o['ext_err_hi']))
        self.logger.info('TS_ext:        %.3f' % o['ts_ext'])
        self.logger.info('Extension UL: %6.4f' % o['ext_ul95'])

        # Fit with the best-fit extension model
        self.logger.info('Fitting extended-source model.')

        self.set_source_morphology(name, spatial_model=spatial_model,
                                   spatial_pars={'ra': o['ra'], 'dec': o['dec'],
                                                 'SpatialWidth': o['ext']},
                                   use_pylike=False,
                                   psf_scale_fn=psf_scale_fn)

        fit_output = self._fit(loglevel=logging.DEBUG, update=False,
                               **kwargs['optimizer'])
        self.update_source(name, reoptimize=True,
                           optimizer=kwargs['optimizer'])

        o['source_fit'] = self.get_src_model(name)
        o['loglike_ext'] = fit_output['loglike']

        if kwargs['save_model_map']:
            o['ext_tot_map'] = self.model_counts_map()
            o['ext_src_map'] = self.model_counts_map(name)
            o['ext_bkg_map'] = self.model_counts_map(exclude=[name])

        if update and (sqrt_ts_threshold is None or
                       np.sqrt(o['ts_ext']) > sqrt_ts_threshold):
            src = self.delete_source(name)
            self.add_source(name, src)
            self.fit(loglevel=logging.DEBUG, **kwargs['optimizer'])
        else:
            self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                       spatial_pars=src.spatial_pars)
            # Restore ROI to previous state
            saved_state.restore()
            self._sync_params(name)
            self._update_roi()

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
            hdu_images += [v.create_image_hdu(k)]

        columns = fits.ColDefs([])
        for k, v in sorted(ext.items()):

            if np.isscalar(v) and isinstance(v, float):
                columns.add_col(fits.Column(name=str(k),
                                            format='E',
                                            array=np.array(v, ndmin=1)))
            elif np.isscalar(v) and isinstance(v, str):
                columns.add_col(fits.Column(name=str(k),
                                            format='A32',
                                            array=np.array(v, ndmin=1)))
            elif isinstance(v, np.ndarray):
                columns.add_col(fits.Column(name=str(k),
                                            format='%iE' % len(v),
                                            array=v[None, :]))

        hdu_table = fits.BinTableHDU.from_columns(columns, name='EXTENSION')

        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu_table] + hdu_images)
        for h in hdulist:
            h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
            h.header['STVER'] = fermipy.get_st_version()
        hdulist.writeto(filename, clobber=True)

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

        src = self.roi.copy_source(name)
        spatial_pars = {'ra': skydir.ra.deg, 'dec': skydir.dec.deg}

        loglike = []
        for i, w in enumerate(width):

            spatial_pars['SpatialWidth'] = max(w, 0.00316)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars=spatial_pars,
                                       use_pylike=False)
            fit_output = self._fit(loglevel=logging.DEBUG, **optimizer)
            # self.logger.debug('Fitting width: %10.3f deg LogLike %10.2f',
            #                  w, fit_output['loglike'])
            #print(i, w, fit_output['loglike'])
            loglike += [fit_output['loglike']]

        state.restore()

        return np.array(loglike)

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
            # self.logger.debug('Fitting width: %10.3f deg LogLike %10.2f',
            #                  w, fit_output['loglike'])
            #print(i, w, fit_output['loglike'])
            loglike += [fit_output['loglike']]

        self.set_source_morphology(name, spatial_model=src['SpatialModel'],
                                   spatial_pars=src.spatial_pars)

        return np.array(loglike)

    def _fit_extension(self, name, **kwargs):

        spatial_model = kwargs.get('spatial_model', 'RadialGaussian')
        optimizer = kwargs.get('optimizer', {})
        fit_position = kwargs.get('fit_position', False)
        skydir = kwargs.get('skydir', self.roi[name].skydir)

        width = np.logspace(-2.0, 0.5, 11)
        loglike = self._scan_extension(name, spatial_model=spatial_model,
                                       width=width, optimizer=optimizer,
                                       skydir=skydir)

        ul_data = utils.get_parameter_limits(width, loglike)

        if np.isfinite(ul_data['err_lo']):
            lolim = max(ul_data['x0'] - 2.0 * ul_data['err_lo'], 10**-2.5)
        else:
            lolim = max(ul_data['x0'] - 2.0 * ul_data['err'], 10**-2.5)

        hilim = ul_data['x0'] + 2.0 * ul_data['err_hi']

        width = np.linspace(lolim, hilim, 11)

        loglike = self._scan_extension(name, spatial_model=spatial_model,
                                       width=width, optimizer=optimizer,
                                       skydir=skydir)
        ul_data = utils.get_parameter_limits(width, loglike)

        o = {}
        o['ext'] = ul_data['x0']
        o['ext_ul95'] = ul_data['ul']
        o['ext_err_lo'] = ul_data['err_lo']
        o['ext_err_hi'] = ul_data['err_hi']
        o['ext_err'] = ul_data['err']
        o['loglike_ext'] = ul_data['lnlmax']
        o['ra'] = skydir.ra.deg
        o['dec'] = skydir.dec.deg

        return o

    def _fit_extension_full(self, name, **kwargs):

        skydir = self.roi[name].skydir
        src = self.roi.copy_source(name)

        spatial_model = kwargs.get('spatial_model', 'RadialGaussian')
        loglike = -self.like()

        scan_cdelt = 0.05
        nstep = 6

        for i in range(4):

            fit_ext = self._fit_extension(name, skydir=skydir, **kwargs)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars={'SpatialWidth': max(
                                           fit_ext['ext'], 0.00316)},
                                       use_pylike=False)

            dtheta_max = max(0.5, 2.0 * fit_ext['ext'])
            if i == 0:
                fit_pos = self._fit_position(name, nstep=nstep,
                                             dtheta_max=dtheta_max)
            else:
                fit_pos = self._fit_position_scan(name,
                                                  scan_cdelt=scan_cdelt,
                                                  nstep=nstep)

            scan_cdelt = 2.0 * fit_pos['r95'] / (nstep - 1.0)
            self.set_source_morphology(name,
                                       spatial_model=spatial_model,
                                       spatial_pars={'RA': fit_pos['ra'],
                                                     'DEC': fit_pos['dec']},
                                       use_pylike=False)

            skydir = fit_pos['skydir']
            fit_ext['ra'] = skydir.ra.deg
            fit_ext['dec'] = skydir.dec.deg
            dloglike = fit_ext['loglike_ext'] - loglike

            # print('-----------------------')
            #print('skydir ', skydir.ra.deg, skydir.dec.deg)
            #print(i, fit_ext['ext'], loglike, fit_ext['loglike_ext'], fit_pos['loglike'], dloglike)

            fit_ext['loglike_ext'] = fit_pos['loglike']

            if i > 1 and dloglike < 0.1:
                break

            loglike = fit_ext['loglike_ext']

        self.set_source_morphology(name, spatial_pars=src.spatial_pars,
                                   use_pylike=False)
        return fit_ext
