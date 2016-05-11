import copy

import numpy as np
import scipy.ndimage
from scipy.ndimage.filters import maximum_filter

from astropy.coordinates import SkyCoord

import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
from fermipy.utils import Map
from fermipy.logger import Logger
from fermipy.logger import logLevel

from LikelihoodState import LikelihoodState

def find_peaks(input_map, threshold, min_separation=0.5):
    """Find peaks in a 2-D map object that have amplitude larger than
    `threshold` and lie a distance at least `min_separation` from another
    peak of larger amplitude.  The implementation of this method uses
    `~scipy.ndimage.filters.maximum_filter`.

    Parameters
    ----------
    input_map : `~fermipy.utils.Map`

    threshold : float

    min_separation : float
       Radius of region size in degrees.  Sets the minimum allowable
       separation between peaks.

    Returns
    -------
    peaks : list
       List of dictionaries containing the location and amplitude of
       each peak.
    """

    data = input_map.counts
    
    cdelt = max(input_map.wcs.wcs.cdelt) 
    min_separation = max(min_separation,2*cdelt)
    
    region_size_pix = int(min_separation/cdelt)
    region_size_pix = max(3,region_size_pix)

    deltaxy = utils.make_pixel_offset(region_size_pix*2+3)
    deltaxy *= max(input_map.wcs.wcs.cdelt)
    region = deltaxy < min_separation

    local_max = maximum_filter(data,footprint=region) == data    
    local_max[data < threshold] = False

    labeled, num_objects = scipy.ndimage.label(local_max)
    slices = scipy.ndimage.find_objects(labeled)

    peaks = []
    for s in slices:
        skydir = SkyCoord.from_pixel(s[1].start, s[0].start,
                                     input_map.wcs)
        peaks.append({'ix': s[1].start,
                      'iy': s[0].start,
                      'skydir': skydir,
                      'amp': data[s[0].start, s[1].start]})

    return sorted(peaks, key=lambda t: t['amp'], reverse=True)


def estimate_pos_and_err_parabolic(tsvals):
    """  Solve for the position and uncertainty of source in one dimension
         assuming that you are near the maximum and the errors are parabolic

    Parameters
    ----------
    tsvals  :  `~numpy.ndarray`
       The TS values at the maximum TS, and for each pixel on either side
    
    Returns
    -------
    The position and uncertainty of the source, in pixel units w.r.t. the center of the maximum pixel         
    """
    a = tsvals[2] - tsvals[0]
    bc =  2.*tsvals[1] - tsvals[0] - tsvals[2]
    s = a / (2*bc)
    err = np.sqrt( 2 / bc )
    return s,err


def refine_peak(tsmap,pix):
    """Solve for the position and uncertainty of source assuming that you
    are near the maximum and the errors are parabolic

    Parameters
    ----------
    tsmap : `~numpy.ndarray`
       Array with the TS data.

    Returns
    -------
    The position and uncertainty of the source, in pixel units w.r.t. the center of the maximum pixel

    """
    # Note the annoying WCS convention
    nx = tsmap.shape[1]
    ny = tsmap.shape[0]
    
    if pix[0] == 0 or pix[0] == (nx-1):
        xval = float(pix[0])
        xerr = -1
    else:
        x_arr = tsmap[pix[1],pix[0]-1:pix[0]+2]
        xval,xerr = estimate_pos_and_err_parabolic(x_arr)
        xval += float(pix[0])

    if pix[1] == 0 or pix[1] == (ny-1):
        yval = float(pix[1])
        yerr = -1        
    else:
        y_arr = tsmap[pix[1]-1:pix[1]+2,pix[0]]
        yval,yerr = estimate_pos_and_err_parabolic(y_arr)
        yval += float(pix[1])

    return (xval,yval),(xerr,yerr)


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
        
        # Extract options from kwargs
        config = copy.deepcopy(self.config['sourcefind'])
        config.update(kwargs) 

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpectrumType', 'PowerLaw')
        config['model'].setdefault('SpatialModel', 'PointSource')
        config['model'].setdefault('Prefactor', 1E-13)
        
        o = {'sources': [], 'peaks' : []}
        
        
        for i in range(config['max_iter']):
            srcs, peaks = self._find_sources_iterate(prefix, i, **config)

            self.logger.info('Found %i sources in iteration %i.'%(len(srcs),i))
            
            o['sources'] += srcs
            o['peaks'] += peaks
            if len(srcs) == 0:
                break

        self.logger.info('Done.')
            
        return o

    def _build_src_dicts_from_peaks(self,peaks,maps,src_dict_template):

        tsmap = maps['ts']
        amp = maps['amplitude']

        src_dicts = []
        names = []        

        for p in peaks:
            o = utils.fit_parabola(tsmap.counts,p['iy'],p['ix'],dpix=2)
            p['fit_loc'] = o
            p['fit_skydir'] = SkyCoord.from_pixel(o['y0'],o['x0'],tsmap.wcs)

            sigmax = 2.0**0.5*o['sigmax']*np.abs(tsmap.wcs.wcs.cdelt[0])
            sigmay = 2.0**0.5*o['sigmay']*np.abs(tsmap.wcs.wcs.cdelt[1])
            sigma = (sigmax*sigmay)**0.5
            p['sigma'] = sigma
            p['sigmax'] = sigmax
            p['sigmay'] = sigmay
            p['r68'] = 2.30**0.5*sigma
            p['r95'] = 5.99**0.5*sigma
            p['r99'] = 9.21**0.5*sigma     
            
            if o['fit_success']:            
                skydir = p['fit_skydir']
            else:
                skydir = p['skydir']
                
            name = utils.create_source_name(skydir)
            src_dict = copy.deepcopy(src_dict_template)
            src_dict.update({'Prefactor': amp.counts[p['iy'], p['ix']],                        
                             'ra': skydir.icrs.ra.deg,
                             'dec': skydir.icrs.dec.deg})

            self.logger.info('Found source\n' +
                             'name: %s\n'%name +
                             'ts: %f'%p['amp']**2)
           
            names.append(name)
            src_dicts.append(src_dict)
            pass

        return names,src_dicts


    def _find_sources_iterate(self, prefix, iiter, **kwargs):

        src_dict_template = kwargs.pop('model')
        
        threshold = kwargs.get('sqrt_ts_threshold')
        min_separation = kwargs.get('min_separation')
        sources_per_iter = kwargs.get('sources_per_iter')
        search_skydir = kwargs.get('search_skydir',None)
        search_minmax_radius = kwargs.get('search_minmax_radius',[None,1.0])
        
        tsmap_fitter = kwargs.get('tsmap_fitter')
        tsmap_kwargs = kwargs.get('tsmap',{})
        tscube_kwargs = kwargs.get('tscube',{})
        
        if tsmap_fitter == 'tsmap':
            m = self.tsmap('%s_sourcefind_%02i'%(prefix,iiter),
                          model=src_dict_template, 
                          **tsmap_kwargs)
        elif tsmap_fitter == 'tscube':
            m = self.tscube('%s_sourcefind_%02i'%(prefix,iiter),
                           model=src_dict_template, 
                          **tscube_kwargs)            
        else:
            raise Exception('Unrecognized option for fitter: %s.'%tsmap_fitter)
            
        amp = m['amplitude']
 
        if tsmap_fitter == 'tsmap':
            peaks = find_peaks(m['sqrt_ts'], threshold, min_separation)
            (names,src_dicts) = self._build_src_dicts_from_peaks(peaks,
                                                                 m,src_dict_template)
        elif tsmap_fitter == 'tscube':
            sd = m['tscube'].find_sources(threshold**2, min_separation,
                                          use_cumul=True,output_src_dicts=True,
                                          output_peaks=True)
            peaks = sd['Peaks']
            names = sd['Names']
            src_dicts = sd['SrcDicts']
            
        # Loop over the seeds and add them to the model
        new_src_names = []
        for name,src_dict in zip(names,src_dicts):    
            # Protect against finding the same source twice
            if self.roi.has_source(name):
                self.logger.info('Source %s found again.  Ignoring it.'%name)
                continue
            # Skip the source if it's outside the search region
            if search_skydir is not None:

                skydir = SkyCoord(src_dict['ra'],src_dict['dec'],unit='deg')                
                separation = search_skydir.separation(skydir).deg
                
                if not utils.apply_minmax_selection(separation,search_minmax_radius):
                    self.logger.info('Source %s outside of search region.  Ignoring it.'%name)
                    continue
                
            self.add_source(name, src_dict, free=True)
            self.free_source(name,False)
            new_src_names.append(name)

            if len(new_src_names) >= sources_per_iter:
                break

        # Re-fit spectral parameters of each source individually
        for name in new_src_names:
            self.free_source(name,True)
            self.fit()
            self.free_source(name,False)
                    
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

        Returns
        -------

        localize : dict
            Dictionary containing results of the localization
            analysis.  This dictionary is also saved to the
            dictionary of this source in 'localize'.

        """

        name = self.roi.get_source_by_name(name, True).name

        # Extract options from kwargs
        config = copy.deepcopy(self.config['localize'])
        config.update(kwargs)
        config.setdefault('newname', name)

        nstep = config['nstep']
        dtheta_max = config['dtheta_max']
        update = config['update']
        newname = config['newname']
        prefix = kwargs.pop('prefix','')
        
        self.logger.info('Running localization for %s' % name)

        saved_state = LikelihoodState(self.like)

        src = self.roi.copy_source(name)
        skydir = src.skydir
        skywcs = self._skywcs
        src_pix = skydir.to_pixel(skywcs)

        tsmap = self.tsmap(utils.join_strings([prefix,name.lower().replace(' ','_')]),
                           model=src.data,
                           map_skydir=skydir,
                           map_size=2.0*dtheta_max,
                           exclude=[name],make_plots=False)

        ix, iy = np.unravel_index(np.argmax(0.5*tsmap['ts'].counts),tsmap['ts'].counts.shape)        
        tsmap_fit = utils.fit_parabola(tsmap['ts'].counts, ix, iy, dpix=2)
                                     
        peak_skydir = SkyCoord.from_pixel(tsmap_fit['y0'],tsmap_fit['x0'],tsmap['ts'].wcs)
        peak_sigmax = 2.0**0.5*tsmap_fit['sigmax']*np.abs(tsmap['ts'].wcs.wcs.cdelt[0])
        peak_sigmay = 2.0**0.5*tsmap_fit['sigmay']*np.abs(tsmap['ts'].wcs.wcs.cdelt[1])
        peak_sigma = (peak_sigmax*peak_sigmay)**0.5
        peak_pix = peak_skydir.to_pixel(skywcs)
        peak_r68 = 2.30**0.5*peak_sigma
        peak_r95 = 5.99**0.5*peak_sigma
        peak_r99 = 9.21**0.5*peak_sigma        
        
        # Fit baseline (point-source) model
        self.free_norm(name)
        self.fit(loglevel=logging.DEBUG,update=False)

        # Save likelihood value for baseline fit
        loglike0 = -self.like()

        self.zero_source(name)

        o = {'config': config,
             'fit_success': True,
             'loglike_base': loglike0 }

        cdelt0 = np.abs(skywcs.wcs.cdelt[0])
        cdelt1 = np.abs(skywcs.wcs.cdelt[1])
        delta_pix = np.linspace(-peak_r99,peak_r99,nstep)/cdelt0
        scan_step = 2.0*peak_r99/(nstep-1.0)
        
        scan_xpix = delta_pix+peak_pix[0]
        scan_ypix = delta_pix+peak_pix[1]
              
        scan_skydir = SkyCoord.from_pixel(np.ravel(np.ones((nstep, nstep)) * scan_xpix[:,np.newaxis]),
                                          np.ravel(np.ones((nstep, nstep)) * scan_ypix[np.newaxis,:]),
                                          skywcs)
                
        lnlscan = dict(xpix=scan_xpix,
                       ypix=scan_ypix,
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
            #self.fit(update=False)
            self.like.optimize(0)
            
            loglike1 = -self.like()
            lnlscan['loglike'].flat[i] = loglike1
            self.delete_source(model_name,loglevel=logging.DEBUG)

        lnlscan['dloglike'] = lnlscan['loglike'] - np.max(lnlscan['loglike'])

        self.unzero_source(name)
        saved_state.restore()
        self._sync_params(name)
        self._update_roi()
        
        ix, iy = np.unravel_index(np.argmax(lnlscan['dloglike']),(nstep,nstep))
        
        scan_fit = utils.fit_parabola(lnlscan['dloglike'], ix, iy, dpix=3)

        sigmax = 2.**0.5*scan_fit['sigmax']*scan_step
        sigmay = 2.**0.5*scan_fit['sigmay']*scan_step
                
        lnlscan['dloglike_fit'] = \
            utils.parabola((np.linspace(0,nstep-1.0,nstep)[:,np.newaxis],
                            np.linspace(0,nstep-1.0,nstep)[np.newaxis,:]),
                           *scan_fit['popt']).reshape((nstep,nstep))
            
        o['lnlscan'] = lnlscan

        # Best fit position and uncertainty from fit to TS map
        o['peak_theta'] = tsmap_fit['theta']
        o['peak_sigmax'] = peak_sigmax
        o['peak_sigmay'] = peak_sigmay
        o['peak_sigma'] = peak_sigma
        o['peak_r68'] = peak_r68
        o['peak_r95'] = peak_r95
        o['peak_r99'] = peak_r99
        o['peak_ra'] = peak_skydir.icrs.ra.deg
        o['peak_dec'] = peak_skydir.icrs.dec.deg
        o['peak_glon'] = peak_skydir.galactic.l.deg
        o['peak_glat'] = peak_skydir.galactic.b.deg
        o['tsmap_fit'] = tsmap_fit
        o['scan_fit'] = scan_fit
        
        # Best fit position and uncertainty from likelihood scan
        o['xpix'] = scan_fit['x0']*scan_step/cdelt0 + scan_xpix[0]
        o['ypix'] = scan_fit['y0']*scan_step/cdelt1 + scan_ypix[0]
        o['deltax'] = (o['xpix']-src_pix[0])*cdelt0
        o['deltay'] = (o['ypix']-src_pix[1])*cdelt1
        o['theta'] = scan_fit['theta']
        o['sigmax'] = sigmax
        o['sigmay'] = sigmay
        o['sigma'] = (o['sigmax']*o['sigmay'])**0.5
        o['r68'] = 2.30**0.5*o['sigma']
        o['r95'] = 5.99**0.5*o['sigma']
        o['r99'] = 9.21**0.5*o['sigma']

        new_skydir = SkyCoord.from_pixel(o['xpix'],o['ypix'],skywcs)

        o['offset'] = skydir.separation(new_skydir).deg
        o['ra'] = new_skydir.icrs.ra.deg
        o['dec'] = new_skydir.icrs.dec.deg
        o['glon'] = new_skydir.galactic.l.deg
        o['glat'] = new_skydir.galactic.b.deg
        
        if o['fit_success'] and o['offset'] > dtheta_max:
            o['fit_success'] = False
            self.logger.error('Best-fit position outside search region:\n '
                              'offset = %.3f deltax = %.3f deltay = %.3f '%(o['offset'],
                                                                            o['deltax'],o['deltay']) +
                              'dtheta_max = %.3f'%(dtheta_max))

        self.roi[name]['localize'] = copy.deepcopy(o)

        try:
            self._plotter.make_localization_plot(self, name, tsmap, prefix=prefix,
                                                 **kwargs)
        except Exception:
            self.logger.error('Plot failed.', exc_info=True)
            
        if update and o['fit_success']:

            self.logger.info(
                'Updating position to: '
                'RA %8.3f DEC %8.3f (offset = %8.3f)' % (o['ra'], o['dec'],
                                                         o['offset']))
            src = self.delete_source(name)
            src.set_position(new_skydir)
            src.set_name(newname, names=src.names)

            self.add_source(newname, src, free=True)
            self.fit(loglevel=logging.DEBUG)
            src = self.roi.get_source_by_name(newname, True)
            self.roi[name]['localize'] = copy.deepcopy(o)
            
        self.logger.info('Finished localization.')
        return o

    def _localize_tsmap(self,name,**kwargs):
        """Localize a source from its TS map."""
        
        prefix = kwargs.get('prefix','')
        dtheta_max = kwargs.get('dtheta_max',0.5)
        
        src = self.roi.copy_source(name)
        skydir = src.skydir
        skywcs = self._skywcs
        src_pix = skydir.to_pixel(skywcs)

        tsmap = self.tsmap(utils.join_strings([prefix,name.lower().replace(' ','_')]),
                           model=src.data,
                           map_skydir=skydir,
                           map_size=2.0*dtheta_max,
                           exclude=[name],make_plots=True)

        ix, iy = np.unravel_index(np.argmax(0.5*tsmap['ts'].counts),tsmap['ts'].counts.shape)        
        tsmap_fit = utils.fit_parabola(tsmap['ts'].counts, ix, iy, dpix=2)        

                
        skydir = SkyCoord.from_pixel(tsmap_fit['y0'],tsmap_fit['x0'],tsmap['ts'].wcs)
        sigmax = 2.0**0.5*tsmap_fit['sigmax']*np.abs(tsmap['ts'].wcs.wcs.cdelt[0])
        sigmay = 2.0**0.5*tsmap_fit['sigmay']*np.abs(tsmap['ts'].wcs.wcs.cdelt[1])
        sigma = (sigmax*sigmay)**0.5
        pix = skydir.to_pixel(skywcs)
        r68 = 2.30**0.5*sigma
        r95 = 5.99**0.5*sigma
        r99 = 9.21**0.5*sigma    

        o = {}
        o['theta'] = tsmap_fit['theta']
        o['sigmax'] = sigmax
        o['sigmay'] = sigmay
        o['sigma'] = sigma
        o['r68'] = r68
        o['r95'] = r95
        o['r99'] = r99
        o['ra'] = skydir.icrs.ra.deg
        o['dec'] = skydir.icrs.dec.deg
        o['glon'] = skydir.galactic.l.deg
        o['glat'] = skydir.galactic.b.deg
        o['fit'] = tsmap_fit
        return o

    def _localize_pylike(self,name,**kwargs):

        scan_dtheta = kwargs.get('scan_dtheta',0.15)
        nstep = kwargs.get('nstep',5)
        
        o = {}
        
        saved_state = LikelihoodState(self.like)

        skywcs = self._skywcs
        
        # Fit baseline (point-source) model
        self.free_norm(name)
        self.fit(loglevel=logging.DEBUG,update=False)

        # Save likelihood value for baseline fit
        loglike0 = -self.like()

        cdelt0 = np.abs(skywcs.wcs.cdelt[0])
        cdelt1 = np.abs(skywcs.wcs.cdelt[1])
        delta_pix = np.linspace(-scan_dtheta,scan_dtheta,nstep)/cdelt0
        scan_step = 2.0*scan_dtheta/(nstep-1.0)
        
        scan_xpix = delta_pix+peak_pix[0]
        scan_ypix = delta_pix+peak_pix[1]
              
        scan_skydir = SkyCoord.from_pixel(np.ravel(np.ones((nstep, nstep)) * scan_xpix[:,np.newaxis]),
                                          np.ravel(np.ones((nstep, nstep)) * scan_ypix[np.newaxis,:]),
                                          skywcs)
                
        lnlscan = dict(xpix=scan_xpix,
                       ypix=scan_ypix,
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
            #self.fit(update=False)
            self.like.optimize(0)
            
            loglike1 = -self.like()
            lnlscan['loglike'].flat[i] = loglike1
            self.delete_source(model_name,loglevel=logging.DEBUG)

        lnlscan['dloglike'] = lnlscan['loglike'] - np.max(lnlscan['loglike'])
        

        self.unzero_source(name)
        saved_state.restore()
        self._sync_params(name)
        self._update_roi()

        ix, iy = np.unravel_index(np.argmax(lnlscan['dloglike']),(nstep,nstep))
        
        scan_fit = utils.fit_parabola(lnlscan['dloglike'], ix, iy, dpix=3)

        sigmax = 2.**0.5*scan_fit['sigmax']*scan_step
        sigmay = 2.**0.5*scan_fit['sigmay']*scan_step
                
        lnlscan['dloglike_fit'] = \
            utils.parabola((np.linspace(0,nstep-1.0,nstep)[:,np.newaxis],
                            np.linspace(0,nstep-1.0,nstep)[np.newaxis,:]),
                           *scan_fit['popt']).reshape((nstep,nstep))
            
        o['lnlscan'] = lnlscan

        # Best fit position and uncertainty from fit to TS map        
        o['scan_fit'] = scan_fit
        o['xpix'] = scan_fit['x0']*scan_step/cdelt0 + scan_xpix[0]
        o['ypix'] = scan_fit['y0']*scan_step/cdelt1 + scan_ypix[0]
        o['deltax'] = (o['xpix']-src_pix[0])*cdelt0
        o['deltay'] = (o['ypix']-src_pix[1])*cdelt1
        o['theta'] = scan_fit['theta']
        o['sigmax'] = sigmax
        o['sigmay'] = sigmay
        o['sigma'] = (o['sigmax']*o['sigmay'])**0.5
        o['r68'] = 2.30**0.5*o['sigma']
        o['r95'] = 5.99**0.5*o['sigma']
        o['r99'] = 9.21**0.5*o['sigma']

        return o
