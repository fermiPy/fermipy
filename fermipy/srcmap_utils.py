# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import spline_filter
from scipy.ndimage.interpolation import shift
import astropy.io.fits as pyfits
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils


class MapInterpolator(object):
    """Object that can efficiently generate source maps by
    interpolation of a map object."""

    def __init__(self, data, pix_ref, shape_out, rebin):

        self._data = data
        self._data_spline = []
        for i in range(data.shape[0]):
            self._data_spline += [spline_filter(self._data[i], order=2)]

        self._axes = []
        for i in range(data.ndim):
            self._axes += [np.arange(0, data.shape[i], dtype=float)]

        #self._coords = np.meshgrid(*self._axes[1:], indexing='ij')
        self._rebin = rebin

        # Shape of global output array
        self._shape_out = shape_out

        self._shape = np.array(self.data.shape)
        for i in range(1, self.data.ndim):
            self._shape[i] = int(self._shape[i] / self.rebin)
        self._shape = tuple(self._shape)

        # Reference pixel coordinates
        self._pix_ref = pix_ref

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def shape_out(self):
        return self._shape_out

    @property
    def rebin(self):
        return self._rebin

    @property
    def ndim(self):
        return self._data.ndim

    def get_offsets(self, pix):
        """Get offset of the first pixel in each dimension in the
        global coordinate system.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Pixel coordinates in global coordinate system.
        """

        idx = []
        for i in range(self.ndim):

            if i == 0:
                idx += [0]
            else:
                npix1 = int(self.shape[i])
                pix0 = int(pix[i - 1]) - npix1 // 2
                idx += [pix0]

        return idx

    def shift_to_coords(self, pix, fill_value=np.nan):
        """Create a new map that is shifted to the pixel coordinates
        ``pix``."""

        pix_offset = self.get_offsets(pix)
        dpix = np.zeros(len(self.shape) - 1)
        for i in range(len(self.shape) - 1):
            x = self.rebin * (pix[i] - pix_offset[i + 1]
                              ) + (self.rebin - 1.0) / 2.
            dpix[i] = x - self._pix_ref[i]

        pos = [pix_offset[i] + self.shape[i] // 2
               for i in range(self.data.ndim)]
        s0, s1 = utils.overlap_slices(self.shape_out, self.shape, pos)

        k = np.zeros(self.data.shape)
        for i in range(k.shape[0]):
            k[i] = shift(self._data_spline[i], dpix, cval=np.nan,
                         order=2, prefilter=False)

        for i in range(1, len(self.shape)):
            k = utils.sum_bins(k, i, self.rebin)

        k0 = np.ones(self.shape_out) * fill_value

        if k[s1].size == 0 or k0[s0].size == 0:
            return k0
        k0[s0] = k[s1]
        return k0


class SourceMapCache(object):
    """Object generates source maps by interpolation of map
    templates."""

    def __init__(self, m0, m1):
        self._m0 = m0
        self._m1 = m1

    def create_map(self, pix):
        """Create a new map with reference pixel coordinates shifted
        to the pixel coordinates ``pix``.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Reference pixel of new map.

        Returns
        -------
        out_map : `~numpy.ndarray`
            The shifted map.        
        """
        k0 = self._m0.shift_to_coords(pix)
        k1 = self._m1.shift_to_coords(pix)

        k0[np.isfinite(k1)] = k1[np.isfinite(k1)]
        k0[~np.isfinite(k0)] = 0
        return k0

    @staticmethod
    def create(psf, spatial_model, spatial_width, shape_out, cdelt,
               rebin=4):

        npix = shape_out[1]
        pad_pix = npix // 2

        xpix = (npix + pad_pix - 1.0) / 2.
        ypix = (npix + pad_pix - 1.0) / 2.
        pix_ref = np.array([ypix, xpix])

        k0 = make_srcmap(psf, spatial_model, spatial_width,
                         npix=npix + pad_pix,
                         xpix=xpix, ypix=ypix,
                         cdelt=cdelt,
                         rebin=1)

        m0 = MapInterpolator(k0, pix_ref, shape_out, 1)

        npix1 = max(10, int(0.5 / cdelt)) * rebin
        xpix1 = (npix1 - 1.0) / 2.
        ypix1 = (npix1 - 1.0) / 2.
        pix_ref = np.array([ypix1, xpix1])

        k1 = make_srcmap(psf, spatial_model, spatial_width,
                         npix=npix1,
                         xpix=xpix1, ypix=ypix1,
                         cdelt=cdelt / rebin,
                         rebin=1)

        m1 = MapInterpolator(k1, pix_ref, shape_out, rebin)

        return SourceMapCache(m0, m1)


def make_srcmap(psf, spatial_model, sigma, npix=500, xpix=0.0, ypix=0.0,
                cdelt=0.01, rebin=1, psf_scale_fn=None):
    """Compute the source map for a given spatial model.

    Parameters
    ----------
    psf : `~fermipy.irfs.PSFModel`

    spatial_model : str
        Spatial model.

    sigma : float
        Spatial size parameter for extended models.

    xpix : float
        Source position in pixel coordinates in X dimension.

    ypix : float
        Source position in pixel coordinates in Y dimension.

    rebin : int    
        Factor by which the original map will be oversampled in the
        spatial dimension when computing the model.

    psf_scale_fn : callable        
        Function that evaluates the PSF scaling function.
        Argument is energy in MeV.

    """
    if rebin > 1:
        npix = npix * rebin
        xpix = xpix * rebin + (rebin - 1.0) / 2.
        ypix = ypix * rebin + (rebin - 1.0) / 2.
        cdelt = cdelt / rebin

    if spatial_model == 'RadialGaussian':
        k = utils.make_cgauss_kernel(psf, sigma, npix, cdelt,
                                     xpix, ypix, psf_scale_fn)
    elif spatial_model == 'RadialDisk':
        k = utils.make_cdisk_kernel(psf, sigma, npix, cdelt,
                                    xpix, ypix, psf_scale_fn)
    elif spatial_model == 'PointSource':
        k = utils.make_psf_kernel(psf, npix, cdelt,
                                  xpix, ypix, psf_scale_fn)
    else:
        raise Exception('Unsupported spatial model: %s', spatial_model)

    if rebin > 1:
        k = utils.sum_bins(k, 1, rebin)
        k = utils.sum_bins(k, 2, rebin)

    k *= psf.exp[:, np.newaxis, np.newaxis] * np.radians(cdelt) ** 2
    return k


def make_cgauss_mapcube(skydir, psf, sigma, outfile, npix=500, cdelt=0.01,
                        rebin=1):
    energies = psf.energies
    nebin = len(energies)

    k = utils.make_cgauss_kernel(psf, sigma, npix * rebin, cdelt / rebin)

    if rebin > 1:
        k = utils.rebin_map(k, nebin, npix, rebin)
    w = wcs_utils.create_wcs(skydir, cdelt=cdelt,
                             crpix=npix / 2. + 0.5, naxis=3)

    w.wcs.crpix[2] = 1
    w.wcs.crval[2] = 10 ** energies[0]
    w.wcs.cdelt[2] = energies[1] - energies[0]
    w.wcs.ctype[2] = 'Energy'

    ecol = pyfits.Column(name='Energy', format='D', array=10 ** energies)
    hdu_energies = pyfits.BinTableHDU.from_columns([ecol], name='ENERGIES')

    hdu_image = pyfits.PrimaryHDU(np.zeros((nebin, npix, npix)),
                                  header=w.to_header())

    hdu_image.data[...] = k

    hdu_image.header['CUNIT3'] = 'MeV'

    hdulist = pyfits.HDUList([hdu_image, hdu_energies])
    hdulist.writeto(outfile, clobber=True)


def make_psf_mapcube(skydir, psf, outfile, npix=500, cdelt=0.01, rebin=1):
    energies = psf.energies
    nebin = len(energies)

    k = utils.make_psf_kernel(psf, npix * rebin, cdelt / rebin)

    if rebin > 1:
        k = utils.rebin_map(k, nebin, npix, rebin)
    w = wcs_utils.create_wcs(skydir, cdelt=cdelt,
                             crpix=npix / 2. + 0.5, naxis=3)

    w.wcs.crpix[2] = 1
    w.wcs.crval[2] = 10 ** energies[0]
    w.wcs.cdelt[2] = energies[1] - energies[0]
    w.wcs.ctype[2] = 'Energy'

    ecol = pyfits.Column(name='Energy', format='D', array=10 ** energies)
    hdu_energies = pyfits.BinTableHDU.from_columns([ecol], name='ENERGIES')

    hdu_image = pyfits.PrimaryHDU(np.zeros((nebin, npix, npix)),
                                  header=w.to_header())

    hdu_image.data[...] = k

    hdu_image.header['CUNIT3'] = 'MeV'

    hdulist = pyfits.HDUList([hdu_image, hdu_energies])
    hdulist.writeto(outfile, clobber=True)


def make_gaussian_spatial_map(skydir, sigma, outfile, cdelt=None, npix=None):

    if cdelt is None:
        cdelt = sigma / 10.

    if npix is None:
        npix = int(np.ceil((6.0 * (sigma + cdelt)) / cdelt))

    w = wcs_utils.create_wcs(skydir, cdelt=cdelt, crpix=npix / 2. + 0.5)
    hdu_image = pyfits.PrimaryHDU(np.zeros((npix, npix)),
                                  header=w.to_header())

    hdu_image.data[:, :] = utils.make_gaussian_kernel(sigma, npix=npix,
                                                      cdelt=cdelt)
    hdulist = pyfits.HDUList([hdu_image])
    hdulist.writeto(outfile, clobber=True)


def make_disk_spatial_map(skydir, radius, outfile, cdelt=None, npix=None):

    if cdelt is None:
        cdelt = radius / 10.

    if npix is None:
        npix = int(np.ceil((2.0 * (radius + cdelt)) / cdelt))

    w = wcs_utils.create_wcs(skydir, cdelt=cdelt, crpix=npix / 2. + 0.5)
    hdu_image = pyfits.PrimaryHDU(np.zeros((npix, npix)),
                                  header=w.to_header())

    hdu_image.data[:, :] = utils.make_disk_kernel(radius, npix=npix,
                                                  cdelt=cdelt)
    hdulist = pyfits.HDUList([hdu_image])
    hdulist.writeto(outfile, clobber=True)


def delete_source_map(srcmap_file, names, logger=None):
    """Delete a map from a binned analysis source map file if it exists.

    Parameters
    ----------
    srcmap_file : str
       Path to the source map file.

    names : list
       List of HDU keys of source maps to be deleted.

    """
    hdulist = pyfits.open(srcmap_file)
    hdunames = [hdu.name.upper() for hdu in hdulist]

    if not isinstance(names, list):
        names = [names]

    for name in names:
        if not name.upper() in hdunames:
            continue
        del hdulist[name.upper()]

    hdulist.writeto(srcmap_file, clobber=True)


def update_source_maps(srcmap_file, srcmaps, logger=None):
    hdulist = pyfits.open(srcmap_file)
    hdunames = [hdu.name.upper() for hdu in hdulist]

    for name, data in srcmaps.items():

        if not name.upper() in hdunames:

            for hdu in hdulist[1:]:
                if hdu.header['XTENSION'] == 'IMAGE':
                    break

            newhdu = pyfits.ImageHDU(data, hdu.header, name=name)
            newhdu.header['EXTNAME'] = name
            hdulist.append(newhdu)

        if logger is not None:
            logger.debug('Updating source map for %s' % name)

        hdulist[name].data[...] = data

    hdulist.writeto(srcmap_file, clobber=True)
