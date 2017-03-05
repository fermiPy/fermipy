# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
import os
import json
import numpy as np
import scipy.signal
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.fits_utils as fits_utils
import fermipy.plotting as plotting
from fermipy.skymap import Map
from fermipy.config import ConfigSchema


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


def convolve_map(m, k, cpix, threshold=0.001, imin=0, imax=None):
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
    islice = slice(imin, imax)

    o = np.zeros(m[islice, ...].shape)
    ix = int(cpix[0])
    iy = int(cpix[1])

    # Loop over energy
    for i in range(m[islice, ...].shape[0]):

        ks = k[islice, ...][i, ...]
        ms = m[islice, ...][i, ...]

        mx = ks[ix, :] > ks[ix, iy] * threshold
        my = ks[:, iy] > ks[ix, iy] * threshold

        nx = int(max(3, np.round(np.sum(mx) / 2.)))
        ny = int(max(3, np.round(np.sum(my) / 2.)))

        # Ensure that there is an odd number of pixels in the kernel
        # array
        if ix + nx + 1 >= ms.shape[0] or ix - nx < 0:
            nx -= 1
            ny -= 1

        sx = slice(ix - nx, ix + nx + 1)
        sy = slice(iy - ny, iy + ny + 1)

        ks = ks[sx, sy]

#        origin = [0, 0]
#        if ks.shape[0] % 2 == 0: origin[0] += 1
#        if ks.shape[1] % 2 == 0: origin[1] += 1
#        o[i,...] = ndimage.convolve(ms, ks, mode='constant',
#                                     origin=origin, cval=0.0)

        o[i, ...] = scipy.signal.fftconvolve(ms, ks, mode='same')

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


class ResidMapGenerator(object):
    """Mixin class for `~fermipy.gtanalysis.GTAnalysis` that generates
    spatial residual maps from the difference of data and model maps
    smoothed with a user-defined spatial/spectral template.  The map
    of residual significance can be interpreted in the same way as a
    TS map (the likelihood of a source at the given location)."""

    def residmap(self, prefix='', **kwargs):
        """Generate 2-D spatial residual maps using the current ROI
        model and the convolution kernel defined with the `model`
        argument.

        Parameters
        ----------
        prefix : str
            String that will be prefixed to the output residual map files.

        {options}

        Returns
        -------
        maps : dict
           A dictionary containing the `~fermipy.utils.Map` objects
           for the residual significance and amplitude.    

        """

        self.logger.info('Generating residual maps')

        schema = ConfigSchema(self.defaults['residmap'])

        config = schema.create_config(self.config['residmap'], **kwargs)

        # Defining default properties of test source model
        config['model'].setdefault('Index', 2.0)
        config['model'].setdefault('SpectrumType', 'PowerLaw')
        config['model'].setdefault('SpatialModel', 'PointSource')
        config['model'].setdefault('Prefactor', 1E-13)

        o = self._make_residual_map(prefix, **config)

        if config['make_plots']:
            plotter = plotting.AnalysisPlotter(self.config['plotting'],
                                               fileio=self.config['fileio'],
                                               logging=self.config['logging'])

            plotter.make_residmap_plots(o, self.roi)

        self.logger.info('Finished residual maps')

        outfile = utils.format_filename(self.workdir, 'residmap',
                                        prefix=[o['name']])

        if config['write_fits']:
            o['file'] = os.path.basename(outfile) + '.fits'
            self._make_residmap_fits(o, outfile + '.fits')

        if config['write_npy']:
            np.save(outfile + '.npy', o)

        return o

    def _make_residmap_fits(self, data, filename, **kwargs):

        maps = {'DATA_MAP': data['data'],
                'MODEL_MAP': data['model'],
                'EXCESS_MAP': data['excess']}

        hdu_images = []
        for k, v in sorted(maps.items()):
            if v is None:
                continue
            hdu_images += [v.create_image_hdu(k)]

        hdus = [data['sigma'].create_primary_hdu()] + hdu_images
        hdus[0].header['CONFIG'] = json.dumps(data['config'])
        hdus[1].header['CONFIG'] = json.dumps(data['config'])
        fits_utils.write_hdus(hdus, filename)

    def _make_residual_map(self, prefix, **kwargs):

        src_dict = copy.deepcopy(kwargs.setdefault('model', {}))
        exclude = kwargs.setdefault('exclude', None)
        loge_bounds = kwargs.setdefault('loge_bounds', None)

        if loge_bounds:
            if len(loge_bounds) != 2:
                raise Exception('Wrong size of loge_bounds array.')
            loge_bounds[0] = (loge_bounds[0] if loge_bounds[0] is not None
                              else self.log_energies[0])
            loge_bounds[1] = (loge_bounds[1] if loge_bounds[1] is not None
                              else self.log_energies[-1])
        else:
            loge_bounds = [self.log_energies[0], self.log_energies[-1]]

        # Put the test source at the pixel closest to the ROI center
        xpix, ypix = (np.round((self.npix - 1.0) / 2.),
                      np.round((self.npix - 1.0) / 2.))
        cpix = np.array([xpix, ypix])

        skywcs = self._skywcs
        skydir = wcs_utils.pix_to_skydir(cpix[0], cpix[1], skywcs)

        if src_dict is None:
            src_dict = {}
        src_dict['ra'] = skydir.ra.deg
        src_dict['dec'] = skydir.dec.deg
        src_dict.setdefault('SpatialModel', 'PointSource')
        src_dict.setdefault('SpatialWidth', 0.3)
        src_dict.setdefault('Index', 2.0)

        kernel = None

        if src_dict['SpatialModel'] == 'Gaussian':
            kernel = utils.make_gaussian_kernel(src_dict['SpatialWidth'],
                                                cdelt=self.components[0].binsz,
                                                npix=101)
            kernel /= np.sum(kernel)
            cpix = [50, 50]

        self.add_source('residmap_testsource', src_dict, free=True,
                        init_source=False, save_source_maps=False)
        src = self.roi.get_source_by_name('residmap_testsource')

        modelname = utils.create_model_name(src)
        npix = self.components[0].npix

        mmst = np.zeros((npix, npix))
        cmst = np.zeros((npix, npix))
        emst = np.zeros((npix, npix))

        sm = get_source_kernel(self, 'residmap_testsource', kernel)
        ts = np.zeros((npix, npix))
        sigma = np.zeros((npix, npix))
        excess = np.zeros((npix, npix))

        self.delete_source('residmap_testsource')

        for i, c in enumerate(self.components):

            imin = utils.val_to_edge(c.log_energies, loge_bounds[0])[0]
            imax = utils.val_to_edge(c.log_energies, loge_bounds[1])[0]

            mc = c.model_counts_map(exclude=exclude).counts.astype('float')
            cc = c.counts_map().counts.astype('float')
            ec = np.ones(mc.shape)

            ccs = convolve_map(cc, sm[i], cpix, imin=imin, imax=imax)
            mcs = convolve_map(mc, sm[i], cpix, imin=imin, imax=imax)
            ecs = convolve_map(ec, sm[i], cpix, imin=imin, imax=imax)

            cms = np.sum(ccs, axis=0)
            mms = np.sum(mcs, axis=0)
            ems = np.sum(ecs, axis=0)

            cmst += cms
            mmst += mms
            emst += ems

            # cts = 2.0 * (poisson_lnl(cms, cms) - poisson_lnl(cms, mms))
            excess += cms - mms

        ts = 2.0 * (poisson_lnl(cmst, cmst) - poisson_lnl(cmst, mmst))
        sigma = np.sqrt(ts)
        sigma[excess < 0] *= -1
        emst /= np.max(emst)

        sigma_map = Map(sigma, skywcs)
        model_map = Map(mmst / emst, skywcs)
        data_map = Map(cmst / emst, skywcs)
        excess_map = Map(excess / emst, skywcs)

        o = {'name': utils.join_strings([prefix, modelname]),
             'file': None,
             'sigma': sigma_map,
             'model': model_map,
             'data': data_map,
             'excess': excess_map,
             'config': kwargs}

        return o
