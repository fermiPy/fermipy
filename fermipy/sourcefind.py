import copy

import numpy as np
import scipy.ndimage
from astropy.coordinates import SkyCoord

import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
from fermipy.utils import Map
from fermipy.logger import Logger
from fermipy.logger import logLevel


def find_peaks(input_map, threshold, min_separation=1.0):
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

#    print region.shape
#    import matplotlib.pyplot as plt
#    plt.figure(); plt.imshow(region,interpolation='nearest')    
#    local_max = scipy.ndimage.filters.maximum_filter(data, region_size) == data
    local_max = scipy.ndimage.filters.maximum_filter(data,
                                                     footprint=region) == data
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


class SourceFinder(fermipy.config.Configurable):

    defaults = dict(defaults.sourcefind.items(),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    def __init__(self, config=None, **kwargs):
        fermipy.config.Configurable.__init__(self, config, **kwargs)
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 logLevel(self.config['logging']['verbosity']))

    def find_sources(self, gta, prefix, **kwargs):
        """
        Find new sources.
        """

        self.logger.info('Starting.')
        
        # Extract options from kwargs
        config = copy.deepcopy(self.config)
        config.update(kwargs) 

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpatialModel', 'PointSource')
        config['model'].setdefault('Prefactor', 1E-13)
        
        o = {'sources': [], 'peaks' : []}
        
        max_iter = kwargs.get('max_iter', self.config['max_iter'])
        for i in range(max_iter):
            srcs, peaks = self._iterate(gta, prefix, i, **config)

            self.logger.info('Found %i sources in iteration %i.'%(len(srcs),i))
            
            o['sources'] += srcs
            o['peaks'] += peaks
            if len(srcs) == 0:
                break

        self.logger.info('Done.')
            
        return o

    def _iterate(self, gta, prefix, iiter, **kwargs):

        src_dict = kwargs.pop('model')
        
        threshold = kwargs.get('sqrt_ts_threshold')
        min_separation = kwargs.get('min_separation')
        sources_per_iter = kwargs.get('sources_per_iter')
        tsmap_fitter = kwargs.get('tsmap_fitter')
        tsmap_kwargs = kwargs.get('tsmap',{})
        tscube_kwargs = kwargs.get('tscube',{})
        
        if tsmap_fitter == 'tsmap':
            m = gta.tsmap('%s_sourcefind_%02i'%(prefix,iiter),
                          model=src_dict, 
                          **tsmap_kwargs)
        elif tsmap_fitter == 'tscube':
            m = gta.tscube('%s_sourcefind_%02i'%(prefix,iiter),
                           model=src_dict, 
                          **tscube_kwargs)            
        else:
            raise Exception('Unrecognized option for fitter: %s.'%tsmap_fitter)
            
        amp = m['amplitude']
        peaks = find_peaks(m['sqrt_ts'], threshold, min_separation)

        names = []
        for i, p in enumerate(peaks[:sources_per_iter]):

            o = utils.fit_parabola(m['ts'].counts,p['iy'],p['ix'],dpix=2)
            peaks[i]['fit_loc'] = o
            peaks[i]['fit_skydir'] = SkyCoord.from_pixel(o['y0'],o['x0'],m['ts'].wcs)
            
            
            if o['fit_success']:            
                skydir = peaks[i]['fit_skydir']
            else:
                skydir = p['skydir']
                
            name = utils.create_source_name(skydir)
            src_dict.update({'Prefactor': amp.counts[p['iy'], p['ix']],                        
                             'ra': skydir.icrs.ra.deg,
                             'dec': skydir.icrs.dec.deg})
            
            self.logger.info('Found source\n' +
                             'name: %s\n'%name +
                             'ts: %f'%p['amp']**2)
            
            names += [name]
            gta.add_source(name, src_dict, free=True)

        for name in names:
            gta.free_source(name,False)

        # Re-fit spectral parameters of each source individually
        for name in names:
            gta.free_source(name,True)
            gta.fit()
            gta.free_source(name,False)
            
        srcs = []
        for name in names:
            srcs.append(gta.roi[name])

        return srcs, peaks
            
    def _fit_source(self, gta, **kwargs):

        localize = kwargs.get('localize',self.config['localize'])
        pass
