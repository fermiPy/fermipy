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

        spatial_model : str
            Spatial model that will be used to test the source
            extension.  The spatial scale parameter of the respective
            model will be set such that the 68% containment radius of
            the model is equal to the width parameter.  The following
            spatial models are supported:

            * RadialDisk : Azimuthally symmetric 2D disk.
            * RadialGaussian : Azimuthally symmetric 2D gaussian.

        width_min : float
            Minimum value in degrees for the spatial extension scan.

        width_max : float
            Maximum value in degrees for the spatial extension scan.

        width_nstep : int
            Number of scan points between width_min and width_max.
            Scan points will be spaced evenly on a logarithmic scale
            between log(width_min) and log(width_max).

        width : array-like
            Sequence of values in degrees for the spatial extension
            scan.  If this argument is None then the scan points will
            be determined from width_min/width_max/width_nstep.

        fix_background : bool
            Fix all background sources when performing the extension fit.

        update : bool
            Update this source with the best-fit model for spatial
            extension if TS_ext > ``tsext_threshold``.

        sqrt_ts_threshold : float
            Threshold on sqrt(TS_ext) that will be applied when ``update``
            is true.  If None then no threshold will be applied.

        psf_scale_fn : tuple        
            Tuple of vectors (logE,f) defining an energy-dependent PSF
            scaling function that will be applied when building
            spatial models for the source of interest.  The tuple
            (logE,f) defines the fractional corrections f at the
            sequence of energies logE = log10(E/MeV) where f=0 means
            no correction.  The correction function f(E) is evaluated
            by linearly interpolating the fractional correction
            factors f in log(E).  The corrected PSF is given by
            P\'(x;E) = P(x/(1+f(E));E) where x is the angular
            separation.
        
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
        schema.add_option('write_fits', True)
        schema.add_option('write_npy', True)
        config = utils.create_dict(self.config['extension'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)
        
        spatial_model = config['spatial_model']
        width_min = config['width_min']
        width_max = config['width_max']
        width_nstep = config['width_nstep']
        width = config['width']
        fix_background = config['fix_background']
        update = config['update']
        sqrt_ts_threshold = config['sqrt_ts_threshold']

        if config['psf_scale_fn']:        
            psf_scale_fn = lambda t: 1.0 + np.interp(np.log10(t),
                                                     config['psf_scale_fn'][0],
                                                     config['psf_scale_fn'][1])
        else:
            psf_scale_fn = None
            
        self.logger.info('Starting')
        self.logger.info('Running analysis for %s', name)

        saved_state = LikelihoodState(self.like)

        if fix_background:
            self.free_sources(free=False, loglevel=logging.DEBUG)

        # Fit baseline model
        self.free_norm(name, loglevel=logging.DEBUG)
        fit_output = self._fit(loglevel=logging.DEBUG, **config['optimizer'])
        src = self.roi.copy_source(name)

        # Save likelihood value for baseline fit
        loglike0 = fit_output['loglike']
        self.logger.debug('Baseline Likelihood: %f',loglike0)        
        self.zero_source(name)

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
        o['config'] = config
        o['ptsrc_tot_map'] = None
        o['ptsrc_src_map'] = None
        o['ptsrc_bkg_map'] = None
        o['ext_tot_map'] = None
        o['ext_src_map'] = None
        o['ext_bkg_map'] = None
        
        src_ptsrc = copy.deepcopy(src)
        src_ptsrc.set_name('%s_ptsrc' % (name.lower().replace(' ', '_')))
        src_ptsrc.set_spatial_model('PSFSource')
        src_ptsrc.set_psf_scale_fn(psf_scale_fn)
        
        src_ext = copy.deepcopy(src)
        src_ext.set_name('%s_ext' % (name.lower().replace(' ', '_')))
        src_ext.set_psf_scale_fn(psf_scale_fn)        
        
        # Fit a point-source
        self.logger.debug('Testing point-source model.')
        self.add_source(src_ptsrc.name, src_ptsrc, free=True, init_source=False,
                        loglevel=logging.DEBUG)
        fit_output = self._fit(loglevel=logging.DEBUG, **config['optimizer'])
        o['loglike_ptsrc'] = fit_output['loglike']
        self.logger.debug('Point Source Likelihood: %f',o['loglike_ptsrc'])

        if config['save_model_map']:
            o['ptsrc_tot_map'] = self.model_counts_map()
            o['ptsrc_src_map'] = self.model_counts_map(src_ptsrc.name)
            o['ptsrc_bkg_map'] = self.model_counts_map(exclude=[src_ptsrc.name])

        self.delete_source(src_ptsrc.name, save_template=False,
                           loglevel=logging.DEBUG)

        # Perform scan over width parameter
        self.logger.debug('Width scan vector:\n %s', width)

        if not hasattr(self.components[0].like.logLike, 'setSourceMapImage'):
            o['loglike'] = self._scan_extension_pylike(src_ext, spatial_model,
                                                       width, config['optimizer'])
        else:
            o['loglike'] = self._scan_extension(src_ext, spatial_model,
                                                width, config['optimizer'])

        self.logger.debug('Likelihood: %s',o['loglike'])
        o['loglike'] = np.concatenate(([o['loglike_ptsrc']], o['loglike']))
        o['dloglike'] = o['loglike'] - o['loglike_ptsrc']

        try:

            ul_data = utils.get_parameter_limits(o['width'], o['dloglike'])

            o['ext'] = ul_data['x0']
            o['ext_ul95'] = ul_data['ul']
            o['ext_err_lo'] = ul_data['err_lo']
            o['ext_err_hi'] = ul_data['err_hi']
            o['ts_ext'] = 2 * ul_data['lnlmax']
            o['ext_err'] = ul_data['err']
        except Exception:
            self.logger.error('Upper limit failed.', exc_info=True)

        self.logger.info('Best-fit extension: %6.4f + %6.4f - %6.4f'
                         % (o['ext'], o['ext_err_lo'], o['ext_err_hi']))
        self.logger.info('TS_ext:        %.3f' % o['ts_ext'])
        self.logger.info('Extension UL: %6.4f' % o['ext_ul95'])

        # Fit with the best-fit extension model
        src_ext.set_spatial_model(spatial_model, max(o['ext'], 10**-2.5))

        self.logger.info('Refitting extended model')
        self.add_source(src_ext.name, src_ext, free=True)
        fit_output = self._fit(loglevel=logging.DEBUG, update=False,
                               **config['optimizer'])
        self.update_source(src_ext.name, reoptimize=True,
                           optimizer=config['optimizer'])

        o['source_fit'] = self.get_src_model(src_ext.name)
        o['loglike_ext'] = fit_output['loglike']

        if config['save_model_map']:
            o['ext_tot_map'] = self.model_counts_map()
            o['ext_src_map'] = self.model_counts_map(src_ext.name)
            o['ext_bkg_map'] = self.model_counts_map(exclude=[src_ext.name])
        
        src_ext = self.delete_source(src_ext.name, save_template=False)

        # Restore ROI to previous state
        self.unzero_source(name)
        saved_state.restore()
        self._sync_params(name)
        self._update_roi()

        if update and (sqrt_ts_threshold is None or
                       np.sqrt(o['ts_ext']) > sqrt_ts_threshold):
            src = self.delete_source(name)
            src.set_spectral_pars(src_ext.spectral_pars)
            src.set_spatial_model(src_ext['SpatialModel'],
                                  src_ext['SpatialWidth'])
            src.set_psf_scale_fn(psf_scale_fn)
            self.add_source(name, src, free=True)
            self.fit(loglevel=logging.DEBUG, **config['optimizer'])

        src = self.roi.get_source_by_name(name)
        src['extension'] = copy.deepcopy(o)

        filename = utils.format_filename(self.workdir, 'ext',
                                         prefix=[config['prefix'],
                                                 name.lower().replace(' ', '_')])
        if config['write_fits']:
            self._make_extension_fits(o, filename + '.fits')
                
        if config['write_npy']:
            np.save(filename + '.npy', o)
        
        self.logger.info('Finished')

        return o

    def _make_extension_fits(self, ext, filename, **kwargs):
        
        maps = {'EXT_TOT_MAP': ext['ext_tot_map'],
                'EXT_SRC_MAP': ext['ext_src_map'],
                'EXT_BKG_MAP': ext['ext_bkg_map'],
                'PTSRC_TOT_MAP' : ext['ptsrc_tot_map'],
                'PTSRC_SRC_MAP' : ext['ptsrc_src_map'],
                'PTSRC_BKG_MAP' : ext['ptsrc_bkg_map']}

        hdu_images = []
        for k, v in sorted(maps.items()):
            if v is None:
                continue
            hdu_images += [v.create_image_hdu(k)]

        columns = fits.ColDefs([])
        for k,v in sorted(ext.items()):
            
            if np.isscalar(v) and isinstance(v,float):            
                columns.add_col(fits.Column(name=str(k),
                                            format='E',
                                            array=np.array(v,ndmin=1)))
            elif np.isscalar(v) and isinstance(v,str):            
                columns.add_col(fits.Column(name=str(k),
                                            format='A32',
                                            array=np.array(v,ndmin=1)))
            elif isinstance(v,np.ndarray):
                columns.add_col(fits.Column(name=str(k),
                                            format='%iE'%len(v),
                                            array=v[None,:]))
                
        hdu_table = fits.BinTableHDU.from_columns(columns, name='EXTENSION')

        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu_table] + hdu_images)
        for h in hdulist:
            h.header['CREATOR'] = 'fermipy ' + fermipy.__version__
            h.header['STVER'] = fermipy.get_st_version()
        hdulist.writeto(filename, clobber=True)
        
    def _scan_extension(self, src, spatial_model, width, optimizer):

        src.set_spatial_model('PSFSource', width[-1])
        self.add_source(src.name, src, free=True, init_source=False)
        self._fitcache = None

        loglike = []
        for i, w in enumerate(width[1:]):

            src.set_spatial_model(spatial_model, w)
            self._update_srcmap(src.name, src)
            fit_output = self._fit(**optimizer)
            self.logger.debug('Fitting width: %10.3f deg LogLike %10.2f',
                              w,fit_output['loglike'])
            loglike += [fit_output['loglike']]

        self.delete_source(src.name, save_template=False)

        return np.array(loglike)

    def _scan_extension_pylike(self, src, spatial_model, width, optimizer):

        loglike = []
        for i, w in enumerate(width[1:]):

            src.set_spatial_model(spatial_model, w)
            self.add_source(src.name, src, free=True, init_source=False,
                            loglevel=logging.DEBUG)

            fit_output = self._fit(**optimizer)
            self.logger.debug('Fitting width: %10.3f deg LogLike %10.2f',
                              w,fit_output['loglike'])
            loglike += [fit_output['loglike']]

            self.delete_source(src.name, save_template=False,
                               loglevel=logging.DEBUG)

        return np.array(loglike)
