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
            srcs, peaks = self._iterate(prefix, i, **config)

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


    def _iterate(self, prefix, iiter, **kwargs):

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
            
    
