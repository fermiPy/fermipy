import copy
import os
import numpy as np
import scipy.signal
import fermipy.config
import fermipy.defaults as defaults
import fermipy.utils as utils
from fermipy.utils import Map
from fermipy.logger import Logger
from fermipy.logger import logLevel as ll


def poisson_lnl(nc, mu):
    nc = np.array(nc, ndmin=1)
    mu = np.array(mu, ndmin=1)

    shape = max(nc.shape, mu.shape)

    lnl = np.zeros(shape)
    mu = mu * np.ones(shape)
    nc = nc * np.ones(shape)

    msk = nc > 0

    lnl[msk] = nc[msk] * np.log(mu[msk]) - mu[msk]
    lnl[~msk] = -mu[~msk]
    return lnl


def convolve_map(m, k, cpix, threshold=0.001,imin=0,imax=None):
    """
    Perform an energy-dependent convolution on a sequence of 2-D spatial maps.

    Parameters
    ----------
    
    m : `~numpy.ndarray`
       3-D map containing a sequence of 2-D spatial maps.  First
       dimension should be energy.
    
    k : `~numpy.ndarray`
       3-D map containing a sequence of convolution kernels (PSF) for
       each slice in m.  This map should have the same dimension as m.

    cpix : list
       Indices of kernel reference pixel in the two spatial dimensions.

    threshold : float
       Kernel amplitude 

    imin : int
       Minimum index in energy dimension.

    imax : int
       Maximum index in energy dimension.

    """
    from scipy import ndimage

    islice = slice(imin,imax)

    o = np.zeros(m[islice,...].shape)

    # Loop over energy
    for i in range(m[islice,...].shape[0]):

        ks = k[islice,...][i,...]
        ms = m[islice,...][i,...]

        mx = ks[cpix[0], :] > ks[cpix[0], cpix[1]] * threshold
        my = ks[:, cpix[1]] > ks[cpix[0], cpix[1]] * threshold

        nx = max(3, np.round(np.sum(mx) / 2.))
        ny = max(3, np.round(np.sum(my) / 2.))

        # Ensure that there is an odd number of pixels in the kernel
        # array
        if cpix[0] + nx + 1 >= ms.shape[0] or cpix[0]-nx < 0:
            nx -= 1        
            ny -= 1

        sx = slice(cpix[0] - nx, cpix[0] + nx + 1)
        sy = slice(cpix[1] - ny, cpix[1] + ny + 1)

        ks = ks[sx, sy]

#        origin = [0, 0]
#        if ks.shape[0] % 2 == 0: origin[0] += 1
#        if ks.shape[1] % 2 == 0: origin[1] += 1
#        o[i,...] = ndimage.convolve(ms, ks, mode='constant',
#                                     origin=origin, cval=0.0)

        o[i,...] = scipy.signal.fftconvolve(ms, ks, mode='same')

    return o

def get_source_kernel(gta, name, kernel=None):
    """Get the PDF for the given source."""

    sm = []
    zs = 0
    for c in gta.components:
        z = c.model_counts_map(name).counts.astype('float')
        if kernel is not None:
            shape = (z.shape[0],) + kernel.shape
            z = np.apply_over_axes(np.sum, z, axes=[1, 2]) * np.ones(
                shape) * kernel[np.newaxis, :, :]
            zs += np.sum(z)
        else:
            zs += np.sum(z)

        sm.append(z)

    sm2 = 0
    for i, m in enumerate(sm):
        sm[i] /= zs
        sm2 += np.sum(sm[i] ** 2)

    for i, m in enumerate(sm):
        sm[i] /= sm2

    return sm

class ResidMapGenerator(fermipy.config.Configurable):
    """This class generates spatial residual maps from the difference
    of data and model maps smoothed with a user-defined
    spatial/spectral template.  The resulting map of source
    significance can be interpreted in the same way as the TS map (the
    likelihood of a source at the given location).  The algorithm
    approximates the best-fit source amplitude that would be derived
    from a least-squares fit to the data."""

    defaults = dict(defaults.residmap.items(),
                    fileio=defaults.fileio,
                    logging=defaults.logging)

    def __init__(self, config=None, **kwargs):
        #        super(ResidMapGenerator,self).__init__(config,**kwargs)
        fermipy.config.Configurable.__init__(self, config, **kwargs)
        self.logger = Logger.get(self.__class__.__name__,
                                 self.config['fileio']['logfile'],
                                 ll(self.config['logging']['verbosity']))

    def run(self, gta, prefix, **kwargs):

        models = kwargs.get('models', self.config['models'])

        if isinstance(models,dict):
            models = [models]

        o = []

        for m in models:
            self.logger.info('Generating Residual map')
            self.logger.info(m)
            o += [self.make_residual_map(gta,prefix,copy.deepcopy(m),**kwargs)]

        return o

    def make_residual_map(self, gta, prefix, src_dict=None, **kwargs):

        exclude = kwargs.get('exclude', None)
        erange = kwargs.get('erange', self.config['erange'])

        if erange is not None:            
            if len(erange) == 0: erange = [None,None]
            elif len(erange) == 1: erange += [None]            
            erange[0] = (erange[0] if erange[0] is not None 
                         else gta.energies[0])
            erange[1] = (erange[1] if erange[1] is not None 
                         else gta.energies[-1])
        else:
            erange = [gta.energies[0],gta.energies[-1]]

        # Put the test source at the pixel closest to the ROI center
        xpix, ypix = (np.round((gta.npix - 1.0) / 2.),
                      np.round((gta.npix - 1.0) / 2.))
        cpix = np.array([xpix, ypix])

        skywcs = gta._skywcs
        skydir = utils.pix_to_skydir(cpix[0], cpix[1], skywcs)

        if src_dict is None: src_dict = {}
        src_dict['ra'] = skydir.ra.deg
        src_dict['dec'] = skydir.dec.deg
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)

        kernel = None

        if src_dict['SpatialModel'] == 'Gaussian':
            kernel = utils.make_gaussian_kernel(src_dict['SpatialWidth'],
                                                cdelt=gta.components[0].binsz,
                                                npix=101)
            kernel /= np.sum(kernel)
            cpix = [50, 50]

        gta.add_source('residmap_testsource', src_dict, free=True,
                       init_source=False,save_source_maps=False)
        src = gta.roi.get_source_by_name('residmap_testsource', True)

        modelname = utils.create_model_name(src)

        enumbins = gta.enumbins
        npix = gta.components[0].npix

        mmst = np.zeros((npix, npix))
        cmst = np.zeros((npix, npix))
        emst = np.zeros((npix, npix))

        sm = get_source_kernel(gta,'residmap_testsource', kernel)
        ts = np.zeros((npix, npix))
        sigma = np.zeros((npix, npix))
        excess = np.zeros((npix, npix))

        gta.delete_source('residmap_testsource')

        for i, c in enumerate(gta.components):

            imin = utils.val_to_edge(c.energies,erange[0])[0]
            imax = utils.val_to_edge(c.energies,erange[1])[0]

            mc = c.model_counts_map(exclude=exclude).counts.astype('float')
            cc = c.counts_map().counts.astype('float')
            ec = np.ones(mc.shape)

            ccs = convolve_map(cc, sm[i], cpix,imin=imin,imax=imax)
            mcs = convolve_map(mc, sm[i], cpix,imin=imin,imax=imax)
            ecs = convolve_map(ec, sm[i], cpix,imin=imin,imax=imax)

            cms = np.sum(ccs, axis=0)
            mms = np.sum(mcs, axis=0)
            ems = np.sum(ecs, axis=0)

            cmst += cms
            mmst += mms
            emst += ems

            cts = 2.0 * (poisson_lnl(cms, cms) - poisson_lnl(cms, mms))
            excess += cms - mms

        ts = 2.0 * (poisson_lnl(cmst, cmst) - poisson_lnl(cmst, mmst))
        sigma = np.sqrt(ts)
        sigma[excess < 0] *= -1

        sigma_map_file = utils.format_filename(self.config['fileio']['workdir'],
                                               'residmap_sigma.fits',
                                               prefix=[prefix, modelname])

        data_map_file = utils.format_filename(self.config['fileio']['workdir'],
                                              'residmap_data.fits',
                                              prefix=[prefix, modelname])

        model_map_file = utils.format_filename(self.config['fileio']['workdir'],
                                               'residmap_model.fits',
                                               prefix=[prefix, modelname])

        excess_map_file = utils.format_filename(self.config['fileio']['workdir'],
                                                'residmap_excess.fits',
                                                prefix=[prefix, modelname])

        emst /= np.max(emst)

        utils.write_fits_image(sigma, skywcs, sigma_map_file)
        utils.write_fits_image(cmst / emst, skywcs, data_map_file)
        utils.write_fits_image(mmst / emst, skywcs, model_map_file)
        utils.write_fits_image(excess / emst, skywcs, excess_map_file)

        files = {'sigma': os.path.basename(sigma_map_file),
                 'model': os.path.basename(model_map_file),
                 'data': os.path.basename(data_map_file),
                 'excess': os.path.basename(excess_map_file)}

        o = {'name': '%s_%s' % (prefix, modelname),
             'files': files,
             'wcs': skywcs,
             'sigma': Map(sigma, skywcs),
             'model': Map(mmst / emst, skywcs),
             'data': Map(cmst / emst, skywcs),
             'excess': Map(excess / emst, skywcs)}

        return o
