# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
import pprint
import logging
import numpy as np
from astropy.coordinates import SkyCoord
import fermipy.config
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
from fermipy.sourcefind_utils import fit_error_ellipse
from fermipy.sourcefind_utils import find_peaks
from fermipy.skymap import Map
from fermipy.config import ConfigSchema
from LikelihoodState import LikelihoodState


class SourceFinder(object):
    """Mixin class which provides source-finding functionality to
    `~fermipy.gtanalysis.GTAnalysis`."""

    def find_sources(self, prefix='', **kwargs):
        """An iterative source-finding algorithm.

        Parameters
        ----------

        model : dict
           Dictionary defining the properties of the test source.
           This is the model that will be used for generating TS maps.

        sqrt_ts_threshold : float
           Source threshold in sqrt(TS).  Only peaks with sqrt(TS)
           exceeding this threshold will be used as seeds for new
           sources.

        min_separation : float
           Minimum separation in degrees of sources detected in each
           iteration. The source finder will look for the maximum peak
           in the TS map within a circular region of this radius.

        max_iter : int
           Maximum number of source finding iterations.  The source
           finder will continue adding sources until no additional
           peaks are found or the number of iterations exceeds this
           number.

        sources_per_iter : int
           Maximum number of sources that will be added in each
           iteration.  If the number of detected peaks in a given
           iteration is larger than this number, only the N peaks with
           the largest TS will be used as seeds for the current
           iteration.

        tsmap_fitter : str
           Set the method used internally for generating TS maps.
           Valid options:

           * tsmap
           * tscube

        tsmap : dict
           Keyword arguments dictionary for tsmap method.

        tscube : dict
           Keyword arguments dictionary for tscube method.


        Returns
        -------

        peaks : list
           List of peak objects.

        sources : list
           List of source objects.

        """

        self.logger.info('Starting.')

        schema = ConfigSchema(self.defaults['sourcefind'],
                              tsmap=self.defaults['tsmap'],
                              tscube=self.defaults['tscube'])

        schema.add_option('search_skydir', None, '', SkyCoord)
        schema.add_option('search_minmax_radius', [None, 1.0], '', list)
        
        config = utils.create_dict(self.config['sourcefind'],
                                   tsmap=self.config['tsmap'],
                                   tscube=self.config['tscube'])        
        config = schema.create_config(config, **kwargs)

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpectrumType', 'PowerLaw')
        config['model'].setdefault('SpatialModel', 'PointSource')
        config['model'].setdefault('Prefactor', 1E-13)

        o = {'sources': [], 'peaks': []}

        for i in range(config['max_iter']):
            srcs, peaks = self._find_sources_iterate(prefix, i, **config)

            self.logger.info('Found %i sources in iteration %i.' %
                             (len(srcs), i))

            o['sources'] += srcs
            o['peaks'] += peaks
            if len(srcs) == 0:
                break

        self.logger.info('Done.')

        return o

    def _build_src_dicts_from_peaks(self, peaks, maps, src_dict_template):

        tsmap = maps['ts']
        amp = maps['amplitude']

        src_dicts = []
        names = []

        for p in peaks:

            o, skydir = fit_error_ellipse(tsmap, (p['ix'], p['iy']), dpix=2)
            p['fit_loc'] = o
            p['fit_skydir'] = skydir

            p.update(o)

            if o['fit_success']:
                skydir = p['fit_skydir']
            else:
                skydir = p['skydir']

            name = utils.create_source_name(skydir)
            src_dict = copy.deepcopy(src_dict_template)
            src_dict.update({'Prefactor': amp.counts[p['iy'], p['ix']],
                             'ra': skydir.icrs.ra.deg,
                             'dec': skydir.icrs.dec.deg})

            src_dict['pos_sigma'] = o['sigma']
            src_dict['pos_sigma_semimajor'] = o['sigma_semimajor']
            src_dict['pos_sigma_semiminor'] = o['sigma_semiminor']
            src_dict['pos_r68'] = o['r68']
            src_dict['pos_r95'] = o['r95']
            src_dict['pos_r99'] = o['r99']
            src_dict['pos_angle'] = np.degrees(o['theta'])

            self.logger.info('Found source\n' +
                             'name: %s\n' % name +
                             'ts: %f' % p['amp'] ** 2)

            names.append(name)
            src_dicts.append(src_dict)

        return names, src_dicts

    def _find_sources_iterate(self, prefix, iiter, **kwargs):

        src_dict_template = kwargs.pop('model')

        threshold = kwargs.get('sqrt_ts_threshold')
        min_separation = kwargs.get('min_separation')
        sources_per_iter = kwargs.get('sources_per_iter')
        search_skydir = kwargs.get('search_skydir', None)
        search_minmax_radius = kwargs.get('search_minmax_radius', [None, 1.0])
        tsmap_fitter = kwargs.get('tsmap_fitter','tsmap')
        
        if tsmap_fitter == 'tsmap':
            kw = kwargs.get('tsmap', {})
            kw['model'] = src_dict_template
            m = self.tsmap('%s_sourcefind_%02i' % (prefix, iiter),
                           **kw)
            
        elif tsmap_fitter == 'tscube':
            kw = kwargs.get('tscube', {})
            kw['model'] = src_dict_template
            m = self.tscube('%s_sourcefind_%02i' % (prefix, iiter),
                            **kw)
        else:
            raise Exception(
                'Unrecognized option for fitter: %s.' % tsmap_fitter)

        if tsmap_fitter == 'tsmap':
            peaks = find_peaks(m['sqrt_ts'], threshold, min_separation)
            (names, src_dicts) = \
                self._build_src_dicts_from_peaks(peaks, m, src_dict_template)
        elif tsmap_fitter == 'tscube':
            sd = m['tscube'].find_sources(threshold ** 2, min_separation,
                                          use_cumul=False,
                                          output_src_dicts=True,
                                          output_peaks=True)
            peaks = sd['Peaks']
            names = sd['Names']
            src_dicts = sd['SrcDicts']

        # Loop over the seeds and add them to the model
        new_src_names = []
        for name, src_dict in zip(names, src_dicts):
            # Protect against finding the same source twice
            if self.roi.has_source(name):
                self.logger.info('Source %s found again.  Ignoring it.' % name)
                continue
            # Skip the source if it's outside the search region
            if search_skydir is not None:

                skydir = SkyCoord(src_dict['ra'], src_dict['dec'], unit='deg')
                separation = search_skydir.separation(skydir).deg

                if not utils.apply_minmax_selection(separation,
                                                    search_minmax_radius):
                    self.logger.info('Source %s outside of '
                                     'search region.  Ignoring it.',
                                     name)
                    continue

            self.add_source(name, src_dict, free=True)
            self.free_source(name, False)
            new_src_names.append(name)

            if len(new_src_names) >= sources_per_iter:
                break

        # Re-fit spectral parameters of each source individually
        for name in new_src_names:
            self.logger.info('Performing spectral fit for %s.',name)
            self.logger.debug(pprint.pformat(self.roi[name].params))
            self.free_source(name, True)
            self.fit()
            self.logger.info(pprint.pformat(self.roi[name].params))
            self.free_source(name, False)

        srcs = []
        for name in new_src_names:
            srcs.append(self.roi[name])

        return srcs, peaks

    def localize(self, name, **kwargs):
        """Find the best-fit position of a source.  Localization is
        performed in two steps.  First a TS map is computed centered
        on the source with half-width set by ``dtheta_max``.  A fit is
        then performed to the maximum TS peak in this map.  The source
        position is then further refined by scanning the likelihood in
        the vicinity of the peak found in the first step.  The size of
        the scan region is set to encompass the 99% positional
        uncertainty contour as determined from the peak fit.

        Parameters
        ----------
        name : str
            Source name.

        dtheta_max : float
            Maximum offset in RA/DEC in deg from the nominal source
            position that will be used to define the boundaries of the
            TS map search region.

        nstep : int
            Number of steps in longitude/latitude that will be taken
            when refining the source position.  The bounds of the scan
            range are set to the 99% positional uncertainty as
            determined from the TS map peak fit.  The total number of
            sampling points will be nstep**2.

        fix_background : bool
            Fix background parameters when fitting the source position.

        update : bool
            Update the model for this source with the best-fit
            position.  If newname=None this will overwrite the
            existing source map of this source with one corresponding
            to its new location.

        newname : str
            Name that will be assigned to the relocalized source
            when update=True.  If newname is None then the existing
            source name will be used.

        make_plots : bool
           Generate plots.

        write_fits : bool
           Write the output to a FITS file.

        write_npy : bool
           Write the output dictionary to a numpy file.

        optimizer : dict
            Dictionary that overrides the default optimizer settings.

        Returns
        -------
        localize : dict
            Dictionary containing results of the localization
            analysis.  This dictionary is also saved to the
            dictionary of this source in 'localize'.

        """

        name = self.roi.get_source_by_name(name).name

        schema = ConfigSchema(self.defaults['localize'],
                              optimizer=self.defaults['optimizer'])
        schema.add_option('make_plots', False)
        schema.add_option('write_fits', True)
        schema.add_option('write_npy', True)
        schema.add_option('newname', name)
        schema.add_option('prefix', '')
        config = utils.create_dict(self.config['localize'],
                                   optimizer=self.config['optimizer'])
        config = schema.create_config(config, **kwargs)
        
        nstep = config['nstep']
        dtheta_max = config['dtheta_max']
        update = config['update']
        newname = config['newname']
        prefix = config['prefix']

        self.logger.info('Running localization for %s' % name)

        saved_state = LikelihoodState(self.like)

        if config['fix_background']:
            self.free_sources(free=False, loglevel=logging.DEBUG)

        src = self.roi.copy_source(name)
        skydir = src.skydir
        skywcs = self._skywcs
        src_pix = skydir.to_pixel(skywcs)

        tsmap_fit, tsmap = self._localize_tsmap(name, prefix=prefix,
                                                dtheta_max=dtheta_max)

        self.logger.debug('Completed localization with TS Map.\n'
                          '(ra,dec) = (%10.4f,%10.4f)\n'
                          '(glon,glat) = (%10.4f,%10.4f)',
                          tsmap_fit['ra'], tsmap_fit['dec'],
                          tsmap_fit['glon'], tsmap_fit['glat'])

        # Fit baseline (point-source) model
        self.free_norm(name)
        fit_output = self._fit(loglevel=logging.DEBUG, **config['optimizer'])

        # Save likelihood value for baseline fit
        loglike0 = fit_output['loglike']
        self.logger.debug('Baseline Model Likelihood: %f',loglike0)

        self.zero_source(name)

        o = {'name': name,
             'config': config,
             'fit_success': True,
             'loglike_base': loglike0,
             'loglike_loc' : np.nan,
             'dloglike_loc' : np.nan }

        cdelt0 = np.abs(skywcs.wcs.cdelt[0])
        cdelt1 = np.abs(skywcs.wcs.cdelt[1])
        scan_step = 2.0 * tsmap_fit['r95'] / (nstep - 1.0)

        self.logger.debug('Refining localization search to '
                          'region of width: %.4f deg',
                          tsmap_fit['r95'])

        scan_map = Map.create(SkyCoord(tsmap_fit['ra'],
                                       tsmap_fit['dec'], unit='deg'),
                              scan_step, (nstep, nstep),
                              coordsys=wcs_utils.get_coordsys(skywcs))

        scan_skydir = scan_map.get_pixel_skydirs()

        lnlscan = dict(wcs=scan_map.wcs.to_header().items(),
                       loglike=np.zeros((nstep, nstep)),
                       dloglike=np.zeros((nstep, nstep)),
                       dloglike_fit=np.zeros((nstep, nstep)))

        for i, t in enumerate(scan_skydir):
            model_name = '%s_localize' % (name.replace(' ', '').lower())
            src.set_name(model_name)
            src.set_position(t)
            self.add_source(model_name, src, free=True,
                            init_source=False, save_source_maps=False,
                            loglevel=logging.DEBUG)
            fit_output = self._fit(
                loglevel=logging.DEBUG, **config['optimizer'])

            loglike1 = fit_output['loglike']
            lnlscan['loglike'].flat[i] = loglike1
            self.delete_source(model_name, loglevel=logging.DEBUG)

        lnlscan['dloglike'] = lnlscan['loglike'] - np.max(lnlscan['loglike'])
        scan_tsmap = Map(2.0 * lnlscan['dloglike'].T, scan_map.wcs)
        
        self.unzero_source(name)
        saved_state.restore()
        self._sync_params(name)
        self._update_roi()

        scan_fit, new_skydir = fit_error_ellipse(scan_tsmap, dpix=3)
        o.update(scan_fit)

        o['loglike_loc'] = np.max(lnlscan['loglike'])+0.5*scan_fit['offset']
        o['dloglike_loc'] = o['loglike_loc'] - o['loglike_base']
        
        # lnlscan['dloglike_fit'] = \
        #   utils.parabola(np.linspace(0,nstep-1.0,nstep)[:,np.newaxis],
        #                  np.linspace(0,nstep-1.0,nstep)[np.newaxis,:],
        #                  *scan_fit['popt']).reshape((nstep,nstep))

        o['lnlscan'] = lnlscan

        # Best fit position and uncertainty from fit to TS map
        o['tsmap_fit'] = tsmap_fit

        # Best fit position and uncertainty from pylike scan
        o['scan_fit'] = scan_fit
        pix = new_skydir.to_pixel(skywcs)
        o['xpix'] = float(pix[0])
        o['ypix'] = float(pix[1])
        o['deltax'] = (o['xpix'] - src_pix[0]) * cdelt0
        o['deltay'] = (o['ypix'] - src_pix[1]) * cdelt1

        o['offset'] = skydir.separation(new_skydir).deg

        if o['offset'] > dtheta_max:
            o['fit_success'] = False

        if not o['fit_success']:
            self.logger.error('Localization failed.\n'
                              '(ra,dec) = (%10.4f,%10.4f)\n'
                              '(glon,glat) = (%10.4f,%10.4f)\n'
                              'offset = %8.4f deltax = %8.4f '
                              'deltay = %8.4f',
                              o['ra'], o['dec'], o['glon'], o['glat'],
                              o['offset'], o['deltax'],
                              o['deltay'])
        else:
            self.logger.info('Localization succeeded with '
                             'coordinates:\n'
                             '(ra,dec) = (%10.4f,%10.4f)\n'
                             '(glon,glat) = (%10.4f,%10.4f)\n'
                             'offset = %8.4f r68 = %8.4f',
                             o['ra'], o['dec'],
                             o['glon'], o['glat'],
                             o['offset'], o['r68'])

        self.roi[name]['localize'] = copy.deepcopy(o)

        if config['make_plots']:
            self._plotter.make_localization_plots(o, tsmap, self.roi,
                                                  prefix=prefix,
                                                  skydir=scan_skydir)

        if update and o['fit_success']:
            self.logger.info('Updating source %s '
                             'to localized position.', name)
            src = self.delete_source(name)
            src.set_position(new_skydir)
            src.set_name(newname, names=src.names)

            self.add_source(newname, src, free=True)
            fit_output = self.fit(loglevel=logging.DEBUG)
            o['loglike_loc'] = fit_output['loglike']
            o['dloglike_loc'] = o['loglike_loc'] - o['loglike_base']
            src = self.roi.get_source_by_name(newname)
            self.roi[newname]['localize'] = copy.deepcopy(o)
            self.logger.info('LogLike: %12.3f DeltaLogLike: %12.3f',
                             o['loglike_loc'],o['dloglike_loc'])

        if o['fit_success']:
            src = self.roi.get_source_by_name(newname)
            src['pos_sigma'] = o['sigma']
            src['pos_sigma_semimajor'] = o['sigma_semimajor']
            src['pos_sigma_semiminor'] = o['sigma_semiminor']
            src['pos_r68'] = o['r68']
            src['pos_r95'] = o['r95']
            src['pos_r99'] = o['r99']
            src['pos_angle'] = np.degrees(o['theta'])

        self.logger.info('Finished localization.')
        return o

    def _localize_tscube(self, name, **kwargs):
        """Localize a source from a TS map generated with
        `~fermipy.gtanalysis.GTAnalysis.tscube`. """
        import matplotlib.pyplot as plt
        from fermipy.plotting import ROIPlotter

        prefix = kwargs.get('prefix', '')

        src = self.roi.copy_source(name)
        skydir = src.skydir

        wp = wcs_utils.WCSProj.create(skydir, 0.0125, 20, coordsys='GAL')

        self.zero_source(name)
        tscube = self.tscube(utils.join_strings([prefix,
                                                 name.lower().
                                                replace(' ', '_')]),
                             wcs=wp.wcs, npix=wp.npix,
                             remake_test_source=False)
        self.unzero_source(name)

        tsmap_renorm = copy.deepcopy(tscube['ts'])
        tsmap_renorm._counts -= np.max(tsmap_renorm._counts)

        plt.figure()

        p = ROIPlotter(tsmap_renorm, roi=self.roi)

        p.plot(levels=[-200, -100, -50, -20, -9.21, -5.99, -2.3],
               cmap='BuGn', vmin=-50.0,
               interpolation='bicubic', cb_label='2$\\times\Delta\ln$L')

        plt.savefig('tscube_localize.png')

    def _localize_tsmap(self, name, **kwargs):
        """Localize a source from its TS map."""

        prefix = kwargs.get('prefix', '')
        dtheta_max = kwargs.get('dtheta_max', 0.5)
        write_fits = kwargs.get('write_fits', False)
        write_npy = kwargs.get('write_npy', False)

        src = self.roi.copy_source(name)
        skydir = src.skydir
        skywcs = self._skywcs
        tsmap = self.tsmap(utils.join_strings([prefix, name.lower().
                                              replace(' ', '_')]),
                           model=src.data,
                           map_skydir=skydir,
                           map_size=2.0 * dtheta_max,
                           exclude=[name],
                           write_fits=write_fits,
                           write_npy=write_npy,
                           make_plots=False)

        posfit, skydir = fit_error_ellipse(tsmap['ts'], dpix=2)
        pix = skydir.to_pixel(skywcs)

        o = {}
        o.update(posfit)
        o['xpix'] = float(pix[0])
        o['ypix'] = float(pix[1])
        return o, tsmap

    def _localize_pylike(self, name, **kwargs):
        pass
