# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import copy
import numpy as np
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.interpolation import map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import fermipy.utils as utils
import fermipy.wcs_utils as wcs_utils
import fermipy.hpx_utils as hpx_utils
import fermipy.fits_utils as fits_utils
from fermipy.hpx_utils import HPX, HpxToWcsMapping


def make_coadd_map(maps, proj, shape):

    if isinstance(proj, WCS):
        return make_coadd_wcs(maps, proj, shape)
    elif isinstance(proj, HPX):
        return make_coadd_hpx(maps, proj, shape)
    else:
        raise Exception("Can't co-add map of unknown type %s" % type(proj))


def make_coadd_wcs(maps, wcs, shape):
    data = np.zeros(shape)
    axes = wcs_utils.wcs_to_axes(wcs, shape)

    for m in maps:
        c = wcs_utils.wcs_to_coords(m.wcs, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes[::-1], weights=np.ravel(m.counts))[0]
        data += o

    return Map(data, copy.deepcopy(wcs))


def make_coadd_hpx(maps, hpx, shape):
    data = np.zeros(shape)
    axes = hpx_utils.hpx_to_axes(hpx, shape)
    for m in maps:
        c = hpx_utils.hpx_to_coords(m.hpx, m.counts.shape)
        o = np.histogramdd(c.T, bins=axes, weights=np.ravel(m.counts))[0]
        data += o
    return HpxMap(data, copy.deepcopy(hpx))


def read_map_from_fits(fitsfile, extname=None):
    """
    """
    proj, f, hdu = fits_utils.read_projection_from_fits(fitsfile, extname)
    if isinstance(proj, WCS):
        m = Map(hdu.data, proj)
    elif isinstance(proj, HPX):
        m = HpxMap.create_from_hdu(hdu, proj.ebins)
    else:
        raise Exception("Did not recognize projection type %s" % type(proj))
    return m, f


class Map_Base(object):
    """ Abstract representation of a 2D or 3D counts map."""

    def __init__(self, counts):
        self._counts = counts

    @property
    def counts(self):
        return self._counts

    @property
    def data(self):
        return self._counts

    @data.setter
    def data(self, val):
        if val.shape != self.data.shape:
            raise Exception('Wrong shape.')
        self._counts = val

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        raise NotImplementedError("MapBase.get_pixel_skydirs()")

    def get_pixel_indices(self, lats, lons):
        """Return the indices in the flat array corresponding to a set of coordinates """
        raise NotImplementedError("MapBase.get_pixel_indices()")

    def sum_over_energy(self):
        """Reduce a counts cube to a counts map by summing over the energy planes """
        raise NotImplementedError("MapBase.sum_over_energy()")

    def get_map_values(self, lons, lats, ibin=None):
        """Return the map values corresponding to a set of coordinates. """
        raise NotImplementedError("MapBase.get_map_values()")

    def interpolate(self, lon, lat, egy=None):
        """Return the interpolated map values corresponding to a set of coordinates. """
        raise NotImplementedError("MapBase.interpolate()")


class Map(Map_Base):
    """ Representation of a 2D or 3D counts map using WCS. """

    def __init__(self, counts, wcs, ebins=None):
        """
        Parameters
        ----------
        counts : `~numpy.ndarray`
            Counts array in row-wise ordering (LON is first dimension).
        """
        Map_Base.__init__(self, counts)
        self._wcs = wcs

        self._npix = counts.shape[::-1]

        if len(self._npix) == 3:
            self._xindex = 2
            self._yindex = 1
        elif len(self._npix) == 2:
            self._xindex = 1
            self._yindex = 0
        else:
            raise Exception('Wrong number of dimensions for Map object.')

        # if len(self._npix) != 3 and len(self._npix) != 2:
        #    raise Exception('Wrong number of dimensions for Map object.')

        self._width = np.array([np.abs(self.wcs.wcs.cdelt[0]) * self.npix[0],
                                np.abs(self.wcs.wcs.cdelt[1]) * self.npix[1]])
        self._pix_center = np.array([(self.npix[0] - 1.0) / 2.,
                                     (self.npix[1] - 1.0) / 2.])
        self._pix_size = np.array([np.abs(self.wcs.wcs.cdelt[0]),
                                   np.abs(self.wcs.wcs.cdelt[1])])

        self._skydir = SkyCoord.from_pixel(self._pix_center[0],
                                           self._pix_center[1],
                                           self.wcs)
        self._ebins = ebins
        if ebins is not None:
            self._ectr = np.exp(utils.edge_to_center(np.log(ebins)))
        else:
            self._ectr = None

    @property
    def wcs(self):
        return self._wcs

    @property
    def npix(self):
        return self._npix

    @property
    def skydir(self):
        """Return the sky coordinate of the image center."""
        return self._skydir

    @property
    def width(self):
        """Return the dimensions of the image."""
        return self._width

    @property
    def pix_size(self):
        """Return the pixel size along the two image dimensions."""
        return self._pix_size

    @property
    def pix_center(self):
        """Return the ROI center in pixel coordinates."""
        return self._pix_center

    @staticmethod
    def create_from_hdu(hdu, wcs):
        return Map(hdu.data.T, wcs)

    @staticmethod
    def create_from_fits(fitsfile, **kwargs):
        hdu = kwargs.get('hdu', 0)

        hdulist = fits.open(fitsfile)
        header = hdulist[hdu].header
        data = hdulist[hdu].data
        header = fits.Header.fromstring(header.tostring())
        wcs = WCS(header)

        ebins = None
        if 'ENERGIES' in hdulist:
            tab = Table.read(fitsfile, 'ENERGIES')
            ectr = np.array(tab.columns[0])
            ebins = np.exp(utils.center_to_edge(np.log(ectr)))
        elif 'EBOUNDS' in hdulist:
            tab = Table.read(fitsfile, 'EBOUNDS')
            emin = np.array(tab['E_MIN']) / 1E3
            emax = np.array(tab['E_MAX']) / 1E3
            ebins = np.append(emin, emax[-1])

        return Map(data, wcs, ebins)

    @staticmethod
    def create(skydir, cdelt, npix, coordsys='CEL', projection='AIT'):
        crpix = np.array([n / 2. + 0.5 for n in npix])
        wcs = wcs_utils.create_wcs(skydir, coordsys, projection,
                                   cdelt, crpix)
        return Map(np.zeros(npix).T, wcs)

    def create_image_hdu(self, name=None, **kwargs):
        return fits.ImageHDU(self.counts, header=self.wcs.to_header(),
                             name=name)

    def create_primary_hdu(self):
        return fits.PrimaryHDU(self.counts, header=self.wcs.to_header())

    def sum_over_energy(self):
        """ Reduce a 3D counts cube to a 2D counts map
        """
        # Note that the array is using the opposite convention from WCS
        # so we sum over axis 0 in the array, but drop axis 2 in the WCS object
        return Map(np.sum(self.counts, axis=0), self.wcs.dropaxis(2))

    def xypix_to_ipix(self, xypix, colwise=False):
        """Return the flattened pixel indices from an array multi-dimensional
        pixel indices.

        Parameters
        ----------
        xypix : list
            List of pixel indices in the order (LON,LAT,ENERGY).

        colwise : bool
            Use column-wise pixel indexing.
        """
        return np.ravel_multi_index(xypix, self.npix,
                                    order='F' if colwise else 'C',
                                    mode='raise')

    def ipix_to_xypix(self, ipix, colwise=False):
        """Return array multi-dimensional pixel indices from flattened index.

        Parameters
        ----------
        colwise : bool
            Use column-wise pixel indexing.
        """
        return np.unravel_index(ipix, self.npix,
                                order='F' if colwise else 'C')

    def ipix_swap_axes(self, ipix, colwise=False):
        """ Return the transposed pixel index from the pixel xy coordinates

        if colwise is True (False) this assumes the original index was
        in column wise scheme
        """
        xy = self.ipix_to_xypix(ipix, colwise)
        return self.xypix_to_ipix(xy, not colwise)

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel.

        """

        xpix = np.linspace(0, self.npix[0] - 1., self.npix[0])
        ypix = np.linspace(0, self.npix[1] - 1., self.npix[1])
        xypix = np.meshgrid(xpix, ypix, indexing='ij')
        return SkyCoord.from_pixel(np.ravel(xypix[0]),
                                   np.ravel(xypix[1]), self.wcs)

    def get_pixel_indices(self, lons, lats, ibin=None):
        """Return the indices in the flat array corresponding to a set of coordinates

        Parameters
        ----------
        lons  : array-like
           'Longitudes' (RA or GLON)

        lats  : array-like
           'Latitidues' (DEC or GLAT)

        ibin : int or array-like
           Extract data only for a given energy bin.  None -> extract data for all energy bins.

        Returns
        ----------
        pixcrd : list
           Pixel indices along each dimension of the map.
        """
        lons = np.array(lons, ndmin=1)
        lats = np.array(lats, ndmin=1)

        if len(lats) != len(lons):
            raise RuntimeError('Map.get_pixel_indices, input lengths '
                               'do not match %i %i' % (len(lons), len(lats)))
        if len(self._npix) == 2:
            pix_x, pix_y = self._wcs.wcs_world2pix(lons, lats, 0)
            pixcrd = [np.floor(pix_x).astype(int), np.floor(pix_y).astype(int)]
        elif len(self._npix) == 3:
            all_lons = np.expand_dims(lons, -1)
            all_lats = np.expand_dims(lats, -1)
            if ibin is None:
                all_bins = (np.expand_dims(
                    np.arange(self.npix[2]), -1) * np.ones(lons.shape)).T
            else:
                all_bins = ibin

            l = self.wcs.wcs_world2pix(all_lons, all_lats, all_bins, 0)
            pix_x = l[0]
            pix_y = l[1]
            pixcrd = [np.floor(l[0]).astype(int), np.floor(l[1]).astype(int),
                      all_bins.astype(int)]

        return pixcrd

    def get_map_values(self, lons, lats, ibin=None):
        """Return the map values corresponding to a set of coordinates.

        Parameters
        ----------
        lons  : array-like
           'Longitudes' (RA or GLON)

        lats  : array-like
           'Latitidues' (DEC or GLAT)

        ibin : int or array-like
           Extract data only for a given energy bin.  None -> extract data for all bins

        Returns
        ----------
        vals : numpy.ndarray((n))
           Values of pixels in the flattened map, np.nan used to flag
           coords outside of map
        """
        pix_idxs = self.get_pixel_indices(lons, lats, ibin)
        idxs = copy.copy(pix_idxs)

        m = np.empty_like(idxs[0], dtype=bool)
        m.fill(True)
        for i, p in enumerate(pix_idxs):
            m &= (pix_idxs[i] >= 0) & (pix_idxs[i] < self._npix[i])
            idxs[i][~m] = 0

        vals = self.counts.T[idxs]
        vals[~m] = np.nan
        return vals

    def interpolate(self, lon, lat, egy=None):

        if len(self.npix) == 2:
            pixcrd = self.wcs.wcs_world2pix(lon, lat, 0)
        else:
            if egy is None:
                egy = self._ectr

            pixcrd = self.wcs.wcs_world2pix(lon, lat, egy, 0)
            pixcrd[2] = np.array(utils.val_to_pix(np.log(self._ectr),
                                                  np.log(egy)), ndmin=1)

        points = []
        for npix in self.npix:
            points += [np.linspace(0, npix - 1., npix)]
        data = self.counts
        fn = RegularGridInterpolator(points, data.T,
                                     bounds_error=False,
                                     fill_value=None)
        return fn(np.column_stack(pixcrd))


class HpxMap(Map_Base):
    """ Representation of a 2D or 3D counts map using HEALPix. """

    def __init__(self, counts, hpx):
        """ C'tor, fill with a counts vector and a HPX object """
        Map_Base.__init__(self, counts)
        self._hpx = hpx
        self._wcs2d = None
        self._hpx2wcs = None

    @property
    def hpx(self):
        return self._hpx

    @staticmethod
    def create_from_hdu(hdu, ebins):
        """ Creates and returns an HpxMap object from a FITS HDU.

        hdu    : The FITS
        ebins  : Energy bin edges [optional]
        """
        hpx = HPX.create_from_header(hdu.header, ebins)
        colnames = hdu.columns.names
        cnames = []
        if hpx.conv.convname == 'FGST_SRCMAP_SPARSE':
            keys = hdu.data.field('KEY')
            vals = hdu.data.field('VALUE')
            nebin = len(ebins)
            data = np.zeros((nebin, hpx.npix))
            data.flat[keys] = vals
        else:
            for c in colnames:
                if c.find(hpx.conv.colstring) == 0:
                    cnames.append(c)
            nebin = len(cnames)
            data = np.ndarray((nebin, hpx.npix))
            for i, cname in enumerate(cnames):
                data[i, 0:] = hdu.data.field(cname)
        return HpxMap(data, hpx)

    @staticmethod
    def create_from_hdulist(hdulist, **kwargs):
        """ Creates and returns an HpxMap object from a FITS HDUList

        extname : The name of the HDU with the map data
        ebounds : The name of the HDU with the energy bin data
        """
        extname = kwargs.get('hdu', 'SKYMAP')
        ebins = fits_utils.find_and_read_ebins(hdulist)
        hpxMap = HpxMap.create_from_hdu(hdulist[extname], ebins)
        return hpxMap

    def create_image_hdu(self, name=None, **kwargs):
        kwargs['extname'] = name
        return self.hpx.make_hdu(self.counts, **kwargs)

    @staticmethod
    def create_from_fits(fitsfile, **kwargs):
        hdulist = fits.open(fitsfile)
        return HpxMap.create_from_hdulist(hdulist, **kwargs)

    def create_image_hdu(self, name=None, **kwargs):
        kwargs['extname'] = name
        return self.hpx.make_hdu(self.counts, **kwargs)

    def make_wcs_from_hpx(self, sum_ebins=False, proj='CAR', oversample=2,
                          normalize=True):
        """Make a WCS object and convert HEALPix data into WCS projection

        NOTE: this re-calculates the mapping, if you have already
        calculated the mapping it is much faster to use
        convert_to_cached_wcs() instead

        Parameters
        ----------
        sum_ebins  : bool
           sum energy bins over energy bins before reprojecting

        proj       : str
           WCS-projection

        oversample : int
           Oversampling factor for WCS map

        normalize  : bool
           True -> perserve integral by splitting HEALPix values between bins

        returns (WCS object, np.ndarray() with reprojected data)

        """
        self._wcs_proj = proj
        self._wcs_oversample = oversample
        self._wcs_2d = self.hpx.make_wcs(2, proj=proj, oversample=oversample)
        self._hpx2wcs = HpxToWcsMapping(self.hpx, self._wcs_2d)
        wcs, wcs_data = self.convert_to_cached_wcs(self.counts, sum_ebins,
                                                   normalize)
        return wcs, wcs_data

    def convert_to_cached_wcs(self, hpx_in, sum_ebins=False, normalize=True):
        """ Make a WCS object and convert HEALPix data into WCS projection

        Parameters
        ----------
        hpx_in     : `~numpy.ndarray`
           HEALPix input data
        sum_ebins  : bool
           sum energy bins over energy bins before reprojecting
        normalize  : bool
           True -> perserve integral by splitting HEALPix values between bins

        returns (WCS object, np.ndarray() with reprojected data)
        """
        if self._hpx2wcs is None:
            raise Exception('HpxMap.convert_to_cached_wcs() called '
                            'before make_wcs_from_hpx()')

        if len(hpx_in.shape) == 1:
            wcs_data = np.ndarray(self._hpx2wcs.npix)
            loop_ebins = False
            hpx_data = hpx_in
        elif len(hpx_in.shape) == 2:
            if sum_ebins:
                wcs_data = np.ndarray(self._hpx2wcs.npix)
                hpx_data = hpx_in.sum(0)
                loop_ebins = False
            else:
                wcs_data = np.ndarray((self.counts.shape[0],
                                       self._hpx2wcs.npix[0],
                                       self._hpx2wcs.npix[1]))
                hpx_data = hpx_in
                loop_ebins = True
        else:
            raise Exception('Wrong dimension for HpxMap %i' %
                            len(hpx_in.shape))

        if loop_ebins:
            for i in range(hpx_data.shape[0]):
                self._hpx2wcs.fill_wcs_map_from_hpx_data(
                    hpx_data[i], wcs_data[i], normalize)
                pass
            wcs_data.reshape((self.counts.shape[0], self._hpx2wcs.npix[
                             0], self._hpx2wcs.npix[1]))
            # replace the WCS with a 3D one
            wcs = self.hpx.make_wcs(3, proj=self._wcs_proj,
                                    energies=self.hpx.ebins,
                                    oversample=self._wcs_oversample)
        else:
            self._hpx2wcs.fill_wcs_map_from_hpx_data(
                hpx_data, wcs_data, normalize)
            wcs_data.reshape(self._hpx2wcs.npix)
            wcs = self._wcs_2d

        return wcs, wcs_data

    def get_pixel_skydirs(self):
        """Get a list of sky coordinates for the centers of every pixel. """
        sky_coords = self._hpx.get_sky_coords()
        if self.hpx.coordsys == 'GAL':
            frame = Galactic
        else:
            frame = ICRS
        return SkyCoord(sky_coords[0], sky_coords[1], frame=frame, unit='deg')

    def get_pixel_indices(self, lats, lons):
        """Return the indices in the flat array corresponding to a set of coordinates """
        return self._hpx.get_pixel_indices(lats, lons)

    def sum_over_energy(self):
        """ Reduce a counts cube to a counts map """
        # We sum over axis 0 in the array, and drop the energy binning in the
        # hpx object
        return HpxMap(np.sum(self.counts, axis=0), self.hpx.copy_and_drop_energy())

    def get_map_values(self, lons, lats, ibin=None):
        """Return the indices in the flat array corresponding to a set of coordinates

        Parameters
        ----------
        lons  : array-like
           'Longitudes' (RA or GLON)

        lats  : array-like
           'Latitidues' (DEC or GLAT)

        ibin : int or array-like
           Extract data only for a given energy bin.  None -> extract data for all bins

        Returns
        ----------
        vals : numpy.ndarray((n))
           Values of pixels in the flattened map, np.nan used to flag
           coords outside of map
        """
        theta = np.pi / 2. - np.radians(lats)
        phi = np.radians(lons)

        pix = hp.ang2pix(self.hpx.nside, theta, phi, nest=self.hpx.nest)

        if self.data.ndim == 2:
            return self.data[:, pix] if ibin is None else self.data[ibin, pix]
        else:
            return self.data[pix]

    def interpolate(self, lon, lat, egy=None, interp_log=True):
        """Interpolate map values.

        Parameters
        ----------
        interp_log : bool
            Interpolate the z-coordinate in logspace.

        """

        if self.data.ndim == 1:
            theta = np.pi / 2. - np.radians(lat)
            phi = np.radians(lon)
            return hp.pixelfunc.get_interp_val(self.counts, theta,
                                               phi, nest=self.hpx.nest)
        else:
            return self._interpolate_cube(lon, lat, egy, interp_log)

    def _interpolate_cube(self, lon, lat, egy=None, interp_log=True):
        """Perform interpolation on a healpix cube.  If egy is None
        then interpolation will be performed on the existing energy
        planes.

        """

        shape = np.broadcast(lon, lat, egy).shape
        lon = lon * np.ones(shape)
        lat = lat * np.ones(shape)
        theta = np.pi / 2. - np.radians(lat)
        phi = np.radians(lon)
        vals = []
        for i, _ in enumerate(self.hpx.evals):
            v = hp.pixelfunc.get_interp_val(self.counts[i], theta,
                                            phi, nest=self.hpx.nest)
            vals += [np.expand_dims(np.array(v, ndmin=1), -1)]

        vals = np.concatenate(vals, axis=-1)

        if egy is None:
            return vals.T

        egy = egy * np.ones(shape)

        if interp_log:
            xvals = utils.val_to_pix(np.log(self.hpx.evals), np.log(egy))
        else:
            xvals = utils.val_to_pix(self.hpx.evals, egy)

        vals = vals.reshape((-1, vals.shape[-1]))
        xvals = np.ravel(xvals)
        v = map_coordinates(vals, [np.arange(vals.shape[0]), xvals],
                            order=1)
        return v.reshape(shape)

    def swap_scheme(self):
        """
        """
        hpx_out = self.hpx.make_swapped_hpx()
        if self.hpx.nest:
            if self.data.ndim == 2:
                data_out = np.vstack([hp.pixelfunc.reorder(
                    self.data[i], n2r=True) for i in range(self.data.shape[0])])
            else:
                data_out = hp.pixelfunc.reorder(self.data, n2r=True)
        else:
            if self.data.ndim == 2:
                data_out = np.vstack([hp.pixelfunc.reorder(
                    self.data[i], r2n=True) for i in range(self.data.shape[0])])
            else:
                data_out = hp.pixelfunc.reorder(self.data, r2n=True)
        return HpxMap(data_out, hpx_out)

    def ud_grade(self, order, preserve_counts=False):
        """
        """
        new_hpx = self.hpx.ud_graded_hpx(order)
        nebins = len(new_hpx.evals)
        shape = self.counts.shape

        if preserve_counts:
            power = -2.
        else:
            power = 0

        if len(shape) == 1:
            new_data = hp.pixelfunc.ud_grade(self.counts,
                                             nside_out=new_hpx.nside,
                                             order_in=new_hpx.ordering,
                                             order_out=ew_hpx.ordering,
                                             power=power)
        else:
            new_data = np.vstack([hp.pixelfunc.ud_grade(self.counts[i],
                                                        nside_out=new_hpx.nside,
                                                        order_in=new_hpx.ordering,
                                                        order_out=new_hpx.ordering,
                                                        power=power) for i in range(shape[0])])
        return HpxMap(new_data, new_hpx)
