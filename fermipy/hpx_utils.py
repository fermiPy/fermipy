# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities for dealing with HEALPix projections and mappings
"""
from __future__ import absolute_import, division, print_function
import re
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic, ICRS

from fermipy.wcs_utils import WCSProj

# This is an approximation of the size of HEALPix pixels (in degrees)
# for a particular order.   It is used to convert from HEALPix to WCS-based
# projections
HPX_ORDER_TO_PIXSIZE = [32.0, 16.0, 8.0, 4.0, 2.0, 1.0,
                        0.50, 0.25, 0.1, 0.05, 0.025, 0.01,
                        0.005, 0.002]


class HPX_Conv(object):
    """ Data structure to define how a HEALPix map is stored to FITS """

    def __init__(self, convname, **kwargs):
        """
        """
        self.convname = convname
        self.colstring = kwargs.get('colstring', 'CHANNEL')
        self.firstcol = kwargs.get('firstcol', 1)
        self.extname = kwargs.get('extname', 'SKYMAP')
        self.energy_hdu = kwargs.get('energy_hdu', 'EBOUNDS')
        self.quantity_type = kwargs.get('quantity_type', 'integral')
        self.coordsys = kwargs.get('coordsys', 'COORDSYS')

    def colname(self, indx):
        return "%s%i" % (self.colstring, indx)


# Various conventions for storing HEALPix maps in FITS files
HPX_FITS_CONVENTIONS = {'FGST_CCUBE':HPX_Conv('FGST_CCUBE'),
                        'FGST_LTCUBE':HPX_Conv('FGST_LTCUBE', colstring='COSBINS', extname='EXPOSURE', energy_hdu='CTHETABOUNDS'),
                        'FGST_BEXPCUBE':HPX_Conv('FGST_BEXPCUBE', colstring='ENERGY', extname='HPXEXPOSURES', energy_hdu='ENERGIES'),
                        'FGST_SRCMAP':HPX_Conv('FGST_SRCMAP', extname=None, quantity_type='differential'),
                        'FGST_TEMPLATE':HPX_Conv('FGST_TEMPLATE', colstring='ENERGY', energy_hdu='ENERGIES'),
                        'FGST_SRCMAP_SPARSE':HPX_Conv('FGST_SRCMAP_SPARSE', colstring=None, extname=None, quantity_type='differential'),
                        'GALPROP':HPX_Conv('GALPROP', colstring='Bin', extname='SKYMAP2', 
                                           energy_hdu='ENERGIES', quantity_type='differential',
                                           coordsys='COORDTYPE'),
                        'GALPROP2':HPX_Conv('GALPROP', colstring='Bin', extname='SKYMAP2', 
                                            energy_hdu='ENERGIES', quantity_type='differential')}

def coords_to_vec(lon, lat):
    """ Converts longitute and latitude coordinates to a unit 3-vector

    return array(3,n) with v_x[i],v_y[i],v_z[i] = directional cosines
    """
    phi = np.radians(lon)
    theta = (np.pi / 2) - np.radians(lat)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    xVals = sin_t * np.cos(phi)
    yVals = sin_t * np.sin(phi)
    zVals = cos_t

    # Stack them into the output array
    out = np.vstack((xVals, yVals, zVals)).swapaxes(0, 1)
    return out


def get_pixel_size_from_nside(nside):
    """ Returns an estimate of the pixel size from the HEALPix nside coordinate

    This just uses a lookup table to provide a nice round number for each
    HEALPix order. 
    """
    order = int(np.log2(nside))
    if order < 0 or order > 13:
        raise ValueError('HEALPix order must be between 0 to 13 %i' % order)

    return HPX_ORDER_TO_PIXSIZE[order]


def hpx_to_axes(h, npix):
    """ Generate a sequence of bin edge vectors corresponding to the
    axes of a HPX object."""
    x = h.ebins
    z = np.arange(npix[-1] + 1)

    return x, z


def hpx_to_coords(h, shape):
    """ Generate an N x D list of pixel center coordinates where N is
    the number of pixels and D is the dimensionality of the map."""

    x, z = hpx_to_axes(h, shape)

    x = np.sqrt(x[0:-1] * x[1:])
    z = z[:-1] + 0.5

    x = np.ravel(np.ones(shape) * x[:, np.newaxis])
    z = np.ravel(np.ones(shape) * z[np.newaxis, :])

    return np.vstack((x, z))


def make_hpx_to_wcs_mapping_centers(hpx, wcs):
    """ Make the mapping data needed to from from HPX pixelization to a
    WCS-based array

    Parameters
    ----------
    hpx     : `~fermipy.hpx_utils.HPX`
       The healpix mapping (an HPX object)

    wcs     : `~astropy.wcs.WCS`
       The wcs mapping (a pywcs.wcs object)

    Returns
    -------
      ipixs    :  array(nx,ny) of HEALPix pixel indices for each wcs pixel 
                  -1 indicates the wcs pixel does not contain the center of a HEALpix pixel
      mult_val :  array(nx,ny) of 1.
      npix     :  tuple(nx,ny) with the shape of the wcs grid

    """
    npix = (int(wcs.wcs.crpix[0] * 2), int(wcs.wcs.crpix[1] * 2))
    mult_val = np.ones(npix).T.flatten()
    sky_crds = hpx.get_sky_coords()
    pix_crds = wcs.wcs_world2pix(sky_crds, 0).astype(int)
    ipixs = -1 * np.ones(npix, int).T.flatten()
    pix_index = npix[1] * pix_crds[0:, 0] + pix_crds[0:, 1]
    if hpx._ipix is None:
        for ipix, pix_crd in enumerate(pix_index):
            ipixs[pix_crd] = ipix
    else:
        for pix_crd, ipix in zip(pix_index, hpx._ipix):
            ipixs[pix_crd] = ipix
    ipixs = ipixs.reshape(npix).T.flatten()
    return ipixs, mult_val, npix


def make_hpx_to_wcs_mapping(hpx, wcs):
    """Make the mapping data needed to from from HPX pixelization to a
    WCS-based array

    Parameters
    ----------
    hpx     : `~fermipy.hpx_utils.HPX`
       The healpix mapping (an HPX object)

    wcs     : `~astropy.wcs.WCS`
       The wcs mapping (a pywcs.wcs object)

    Returns
    -------
      ipixs    :  array(nx,ny) of HEALPix pixel indices for each wcs pixel
      mult_val :  array(nx,ny) of 1./number of wcs pixels pointing at each HEALPix pixel
      npix     :  tuple(nx,ny) with the shape of the wcs grid

    """
    npix = (int(wcs.wcs.crpix[0] * 2), int(wcs.wcs.crpix[1] * 2))
    pix_crds = np.dstack(np.meshgrid(np.arange(npix[0]),
                                     np.arange(npix[1]))).swapaxes(0, 1).reshape((npix[0] * npix[1], 2))
    sky_crds = wcs.wcs_pix2world(pix_crds, 0)

    sky_crds *= np.radians(1.)
    sky_crds[0:, 1] = (np.pi / 2) - sky_crds[0:, 1]

    fullmask = np.isnan(sky_crds)
    mask = (fullmask[0:, 0] + fullmask[0:, 1]) == 0
    ipixs = -1 * np.ones(npix, int).T.flatten()
    ipixs[mask] = hp.pixelfunc.ang2pix(hpx.nside, sky_crds[0:, 1][mask],
                                       sky_crds[0:, 0][mask], hpx.nest)

    # Here we are counting the number of HEALPix pixels each WCS pixel points to;
    # this could probably be vectorized by filling a histogram.
    d_count = {}
    for ipix in ipixs:
        if ipix in d_count:
            d_count[ipix] += 1
        else:
            d_count[ipix] = 1

    # Here we are getting a multiplicative factor that tells use how to split up
    # the counts in each HEALPix pixel (by dividing the corresponding WCS pixels
    # by the number of associated HEALPix pixels).
    # This could also likely be vectorized.
    mult_val = np.ones(ipixs.shape)
    for i, ipix in enumerate(ipixs):
        mult_val[i] /= d_count[ipix]

    ipixs = ipixs.reshape(npix).flatten()
    mult_val = mult_val.reshape(npix).flatten()
    return ipixs, mult_val, npix


def match_hpx_pixel(nside, nest, nside_pix, ipix_ring):
    """
    """
    ipix_in = np.arange(12 * nside * nside)
    vecs = hp.pix2vec(nside, ipix_in, nest)
    pix_match = hp.vec2pix(nside_pix, vecs[0], vecs[1], vecs[2]) == ipix_ring
    return ipix_in[pix_match]


class HPX(object):
    """ Encapsulation of basic healpix map parameters """

    def __init__(self, nside, nest, coordsys, order=-1, region=None, ebins=None, conv=HPX_Conv('FGST_CCUBE')):
        """ C'tor

        nside     : HEALPix nside parameter, the total number of pixels is 12*nside*nside
        nest      : bool, True -> 'NESTED', False -> 'RING' indexing scheme
        coordsys  : Coordinate system, 'CEL' | 'GAL'
        """
        if nside >= 0:
            if order >= 0:
                raise Exception('Specify either nside or oder, not both.')
            else:
                self._nside = nside
                self._order = -1
        else:
            if order >= 0:
                self._nside = 2**order
                self._order = order
            else:
                raise Exception('Specify either nside or oder, not both.')
        self._nest = nest
        self._coordsys = coordsys
        self._region = region
        self._maxpix = 12 * self._nside * self._nside
        if self._region:
            self._ipix = self.get_index_list(
                self._nside, self._nest, self._region)
            self._rmap = {}
            self._npix = len(self._ipix)
        else:
            self._ipix = None
            self._rmap = None
            self._npix = self._maxpix

        self._ebins = ebins
        self._conv = conv
        if self._ebins is not None:
            self._evals = np.sqrt(self._ebins[0:-1] * self._ebins[1:])
        else:
            self._evals = None

        if self._ipix is not None:
            for i, ipixel in enumerate(self._ipix.flat):
                self._rmap[ipixel] = i

    def __getitem__(self, sliced):
        """ This implements the global-to-local lookup

        sliced:   An array of HEALPix pixel indices

        For all-sky maps it just returns the input array.
        For partial-sky maps in returns the local indices corresponding to the
        indices in the input array, and -1 for those pixels that are outside the 
        selected region.
        """

        if self._rmap is not None:
            retval = np.zeros((sliced.size), 'i')
            for i, v in enumerate(sliced.flat):
                if v in self._rmap:
                    retval[i] = self._rmap[v]
                else:
                    retval[i] = -1
            retval = retval.reshape(sliced.shape)
            return retval
        return sliced

    @property
    def ordering(self):
        if self._nest:
            return "NESTED"
        return "RING"

    @property
    def nside(self):
        return self._nside

    @property
    def order(self):
        return self._order

    @property
    def nest(self):
        return self._nest

    @property
    def npix(self):
        return self._npix

    @property
    def ebins(self):
        return self._ebins

    @property
    def conv(self):
        return self._conv

    @property
    def coordsys(self):
        return self._coordsys

    @property
    def evals(self):
        return self._evals

    @property
    def region(self):
        return self._region

    def ud_graded_hpx(self, order):
        """
        """
        if self._order < 0:
            raise RuntimeError(
                "Upgrade and degrade only implemented for standard maps")
        return HPX(-1, self.nest, self.coordsys, order, self.region, self.ebins, self.conv)

    def make_swapped_hpx(self):
        """
        """
        if self.order > 0:
            return HPX(-1, not self.nest, self.coordsys, self.order, self.region, self.ebins, self.conv)
        else:
            return HPX(self.nside, not self.nest, self.coordsys, -1, self.region, self.ebins, self.conv)

    def copy_and_drop_energy(self):
        """
        """
        if self.order > 0:
            return HPX(-1, not self.nest, self.coordsys, self.order, self.region, None, self.conv)
        else:
            return HPX(self.nside, not self.nest, self.coordsys, -1, self.region, None, self.conv)

    @staticmethod
    def create_hpx(nside, nest, coordsys='CEL', order=-1, region=None,
                   ebins=None, conv=HPX_Conv('FGST_CCUBE')):
        """Create a HPX object.

        Parameters
        ----------
        nside    : int
           HEALPix nside paramter

        nest     : bool
           True for HEALPix "NESTED" indexing scheme, False for "RING" scheme.

        coordsys : str
           "CEL" or "GAL"

        order    : int
           nside = 2**order

        region   : Allows for partial-sky mappings
        ebins    : Energy bin edges
        """
        return HPX(nside, nest, coordsys, order, region, ebins, conv)

    @staticmethod
    def identify_HPX_convention(header):
        """ Identify the convention used to write this file """
        # Hopefully the file contains the HPX_CONV keyword specifying
        # the convention used
        try:
            return header['HPX_CONV']
        except KeyError:
            pass

        # Try based on the EXTNAME keyword
        extname = header.get('EXTNAME', None)
        if extname == 'HPXEXPOSURES':
            return 'FGST_BEXPCUBE'
        elif extname == 'SKYMAP2':
            if 'COORDTYPE' in header.keys():
                return 'GALPROP'
            else:
                return 'GALPROP2'

        # Check the name of the first column
        colname = header['TTYPE1']
        if colname == 'PIX':
            colname = header['TTYPE2']

        if colname == 'KEY':
            return 'FGST_SRCMAP_SPARSE'
        elif colname == 'ENERGY1':
            return 'FGST_TEMPLATE'
        elif colname == 'COSBINS':
            return 'FGST_LTCUBE'
        elif colname == 'Bin0':
            return 'GALPROP'
        elif colname == 'CHANNEL1':
            if extname == 'SKYMAP':
                return 'FGST_CCUBE'
            else:
                return 'FGST_SRCMAP'
        else:
            raise ValueError("Could not identify HEALPix convention")

    @staticmethod
    def create_from_header(header, ebins=None):
        """ Creates an HPX object from a FITS header.

        header : The FITS header
        ebins  : Energy bin edges [optional]
        """
        convname = HPX.identify_HPX_convention(header)
        conv = HPX_FITS_CONVENTIONS[convname]

        if conv.convname != 'GALPROP':
            if header["PIXTYPE"] != "HEALPIX":
                raise Exception("PIXTYPE != HEALPIX")
        if header["PIXTYPE"] != "HEALPIX":
            raise Exception("PIXTYPE != HEALPIX")
        if header["ORDERING"] == "RING":
            nest = False
        elif header["ORDERING"] == "NESTED":
            nest = True
        else:
            raise Exception("ORDERING != RING | NESTED")

        try:
            order = header["ORDER"]
        except KeyError:
            order = -1

        if order < 0:
            nside = header["NSIDE"]
        else:
            nside = -1

        try:
            coordsys = header[conv.coordsys]
        except KeyError:
            coordsys = header['COORDSYS']
 
        try:
            region = header["HPX_REG"]
        except KeyError:
            try:
                region = header["HPXREGION"]
            except KeyError:
                region = None

        return HPX(nside, nest, coordsys, order, region, ebins=ebins, conv=conv)

    def make_header(self):
        """ Builds and returns FITS header for this HEALPix map """
        cards = [fits.Card("TELESCOP", "GLAST"),
                 fits.Card("INSTRUME", "LAT"),
                 fits.Card(self._conv.coordsys, self._coordsys),
                 fits.Card("PIXTYPE", "HEALPIX"),
                 fits.Card("ORDERING", self.ordering),
                 fits.Card("ORDER", self._order),
                 fits.Card("NSIDE", self._nside),
                 fits.Card("FIRSTPIX", 0),
                 fits.Card("LASTPIX", self._maxpix - 1),
                 fits.Card("HPX_CONV", self._conv.convname)]

        if self._coordsys == "CEL":
            cards.append(fits.Card("EQUINOX", 2000.0,
                                   "Equinox of RA & DEC specifications"))

        if self._region:
            cards.append(fits.Card("HPX_REG", self._region))

        header = fits.Header(cards)
        return header

    def make_hdu(self, data, **kwargs):
        """ Builds and returns a FITs HDU with input data

        data      : The data begin stored

        Keyword arguments
        -------------------
        extname   : The HDU extension name        
        colbase   : The prefix for column names
        """
        shape = data.shape
        extname = kwargs.get('extname', self.conv.extname)

        if shape[-1] != self._npix:
            raise Exception(
                "Size of data array does not match number of pixels")
        cols = []
        if self._region:
            cols.append(fits.Column("PIX", "J", array=self._ipix))

        if self.conv.convname == 'FGST_SRCMAP_SPARSE':
            nonzero = data.nonzero()
            nfilled = len(nonzero[0])
            print ('Nfilled ', nfilled)
            if len(shape) == 1:
                nonzero = nonzero[0]
                cols.append(fits.Column("KEY", "%iJ" %
                                        nfilled, array=nonzero.reshape(1, nfilled)))
                cols.append(fits.Column("VALUE", "%iE" % nfilled, array=data[
                            nonzero].astype(float).reshape(1, nfilled)))
            elif len(shape) == 2:
                nonzero = self._npix * nonzero[0] + nonzero[1]
                cols.append(fits.Column("KEY", "%iJ" %
                                        nfilled, array=nonzero.reshape(1, nfilled)))
                cols.append(fits.Column("VALUE", "%iE" % nfilled, array=data.flat[
                            nonzero].astype(float).reshape(1, nfilled)))
            else:
                raise Exception("HPX.write_fits only handles 1D and 2D maps")

        else:
            if len(shape) == 1:
                cols.append(fits.Column(self.conv.colname(
                    indx=i + self.conv.firstcol), "E", array=data.astype(float)))
            elif len(shape) == 2:
                for i in range(shape[0]):
                    cols.append(fits.Column(self.conv.colname(
                        indx=i + self.conv.firstcol), "E", array=data[i].astype(float)))
            else:
                raise Exception("HPX.write_fits only handles 1D and 2D maps")

        header = self.make_header()
        hdu = fits.BinTableHDU.from_columns(cols, header=header, name=extname)

        return hdu

    def make_energy_bounds_hdu(self, extname="EBOUNDS"):
        """ Builds and returns a FITs HDU with the energy bin boundries

        extname   : The HDU extension name            
        """
        if self._ebins is None:
            return None
        cols = [fits.Column("CHANNEL", "I", array=np.arange(1, len(self._ebins + 1))),
                fits.Column("E_MIN", "1E", unit='keV',
                            array=1000 * self._ebins[0:-1]),
                fits.Column("E_MAX", "1E", unit='keV', array=1000 * self._ebins[1:])]
        hdu = fits.BinTableHDU.from_columns(
            cols, self.make_header(), name=extname)
        return hdu

    def make_energies_hdu(self, extname="ENERGIES"):
        """ Builds and returns a FITs HDU with the energy bin boundries

        extname   : The HDU extension name            
        """
        if self._evals is None:
            return None
        cols = [fits.Column("ENERGY", "1E", unit='MeV',
                            array=self._evals)]
        hdu = fits.BinTableHDU.from_columns(
            cols, self.make_header(), name=extname)
        return hdu

    def write_fits(self, data, outfile, extname="SKYMAP", clobber=True):
        """ Write input data to a FITS file

        data      : The data begin stored
        outfile   : The name of the output file
        extname   : The HDU extension name        
        clobber   : True -> overwrite existing files
        """
        hdu_prim = fits.PrimaryHDU()
        hdu_hpx = self.make_hdu(data, extname=extname)
        hl = [hdu_prim, hdu_hpx]
        if self.conv.energy_hdu == 'EBOUNDS':
            hdu_energy = self.make_energy_bounds_hdu()
        elif self.conv.energy_hdu == 'ENERGIES':
            hdu_energy = self.make_energies_hdu()
        if hdu_energy is not None:
            hl.append(hdu_energy)
        hdulist = fits.HDUList(hl)
        hdulist.writeto(outfile, clobber=clobber)

    @staticmethod
    def get_index_list(nside, nest, region):
        """ Returns the list of pixels indices for all the pixels in a region

        nside    : HEALPix nside parameter
        nest     : True for 'NESTED', False = 'RING'
        region   : HEALPix region string
        """
        tokens = re.split('\(|\)|,', region)
        if tokens[0] == 'DISK':
            vec = coords_to_vec(float(tokens[1]), float(tokens[2]))
            ilist = hp.query_disc(nside, vec[0], np.radians(float(tokens[3])),
                                  inclusive=False, nest=nest)
        elif tokens[0] == 'DISK_INC':
            vec = coords_to_vec(float(tokens[1]), float(tokens[2]))
            ilist = hp.query_disc(nside, vec[0], np.radians(float(tokens[3])),
                                  inclusive=True, fact=int(tokens[4]),
                                  nest=nest)
        elif tokens[0] == 'HPX_PIXEL':
            nside_pix = int(tokens[2])
            if tokens[1] == 'NESTED':
                ipix_ring = hp.nest2ring(nside_pix, int(tokens[3]))
            elif tokens[1] == 'RING':
                ipix_ring = int(tokens[3])
            else:
                raise Exception(
                    "Did not recognize ordering scheme %s" % tokens[1])
            ilist = match_hpx_pixel(nside, nest, nside_pix, ipix_ring)
        else:
            raise Exception(
                "HPX.get_index_list did not recognize region type %s" % tokens[0])
        return ilist

    @staticmethod
    def get_ref_dir(region, coordsys):
        """ Finds and returns the reference direction for a given 
        HEALPix region string.   

        region   : a string describing a HEALPix region
        coordsys : coordinate system, GAL | CEL
        """
        if region is None:
            if coordsys == "GAL":
                c = SkyCoord(0., 0., frame=Galactic, unit="deg")
            elif coordsys == "CEL":
                c = SkyCoord(0., 0., frame=ICRS, unit="deg")
            return c
        tokens = re.split('\(|\)|,', region)
        if tokens[0] in ['DISK', 'DISK_INC']:
            if coordsys == "GAL":
                c = SkyCoord(float(tokens[1]), float(
                    tokens[2]), frame=Galactic, unit="deg")
            elif coordsys == "CEL":
                c = SkyCoord(float(tokens[1]), float(
                    tokens[2]), frame=ICRS, unit="deg")
            return c
        elif tokens[0] == 'HPX_PIXEL':
            nside_pix = int(tokens[2])
            ipix_pix = int(tokens[3])
            if tokens[1] == 'NESTED':
                nest_pix = True
            elif tokens[1] == 'RING':
                nest_pix = False
            else:
                raise Exception(
                    "Did not recognize ordering scheme %s" % tokens[1])
            theta, phi = hp.pix2ang(nside_pix, ipix_pix, nest_pix)
            lat = np.degrees((np.pi / 2) - theta)
            lon = np.degrees(phi)
            if coordsys == "GAL":
                c = SkyCoord(lon, lat, frame=Galactic, unit="deg")
            elif coordsys == "CEL":
                c = SkyCoord(lon, lat, frame=ICRS, unit="deg")
            return c
        else:
            raise Exception(
                "HPX.get_ref_dir did not recognize region type %s" % tokens[0])
        return None

    @staticmethod
    def get_region_size(region):
        """ Finds and returns the approximate size of region (in degrees)  
        from a HEALPix region string.   
        """
        if region is None:
            return 180.
        tokens = re.split('\(|\)|,', region)
        if tokens[0] in ['DISK', 'DISK_INC']:
            return float(tokens[3])
        elif tokens[0] == 'HPX_PIXEL':
            pixel_size = get_pixel_size_from_nside(int(tokens[2]))
            return 2. * pixel_size
        else:
            raise Exception(
                "HPX.get_region_size did not recognize region type %s" % tokens[0])
        return None

    def make_wcs(self, naxis=2, proj='CAR', energies=None, oversample=2):
        """ Make a WCS projection appropirate for this HPX pixelization
        """
        w = WCS(naxis=naxis)
        skydir = self.get_ref_dir(self._region, self.coordsys)

        if self.coordsys == 'CEL':
            w.wcs.ctype[0] = 'RA---%s' % (proj)
            w.wcs.ctype[1] = 'DEC--%s' % (proj)
            w.wcs.crval[0] = skydir.ra.deg
            w.wcs.crval[1] = skydir.dec.deg
        elif self.coordsys == 'GAL':
            w.wcs.ctype[0] = 'GLON-%s' % (proj)
            w.wcs.ctype[1] = 'GLAT-%s' % (proj)
            w.wcs.crval[0] = skydir.galactic.l.deg
            w.wcs.crval[1] = skydir.galactic.b.deg
        else:
            raise Exception('Unrecognized coordinate system.')

        pixsize = get_pixel_size_from_nside(self.nside)
        roisize = self.get_region_size(self._region)
        allsky = False
        if roisize > 45:
            roisize = 90
            allsky = True

        npixels = int(2. * roisize / pixsize) * oversample
        crpix = npixels / 2.

        if allsky:
            w.wcs.crpix[0] = 2 * crpix
            npix = (2 * npixels, npixels)
        else:
            w.wcs.crpix[0] = crpix
            npix = (npixels, npixels)

        w.wcs.crpix[1] = crpix
        w.wcs.cdelt[0] = -pixsize / oversample
        w.wcs.cdelt[1] = pixsize / oversample

        if naxis == 3:
            w.wcs.crpix[2] = 1
            w.wcs.ctype[2] = 'Energy'
            if energies is not None:
                w.wcs.crval[2] = 10 ** energies[0]
                w.wcs.cdelt[2] = 10 ** energies[1] - 10 ** energies[0]

        w = WCS(w.to_header())
        wcs_proj = WCSProj(w, npix)
        return wcs_proj

    def get_sky_coords(self):
        """ Get the sky coordinates of all the pixels in this pixelization """
        if self._ipix is None:
            theta, phi = hp.pix2ang(
                self._nside, xrange(self._npix), self._nest)
        else:
            theta, phi = hp.pix2ang(self._nside, self._ipix, self._nest)

        lat = np.degrees((np.pi / 2) - theta)
        lon = np.degrees(phi)
        return np.vstack([lon, lat]).T

    def get_sky_dirs(self):

        lonlat = self.get_sky_coords()
        return SkyCoord(ra=lonlat.T[0], dec=lonlat.T[1], unit='deg')

    def get_pixel_indices(self, lats, lons):
        """ "Return the indices in the flat array corresponding to a set of coordinates """
        theta = np.radians(90. - lats)
        phi = np.radians(lons)
        return hp.ang2pix(self.nside, theta, phi, self.nest)

    def skydir_to_pixel(self, skydir):
        """Return the pixel index of a SkyCoord object."""
        if self.coordsys in ['CEL', 'EQU']:
            skydir = skydir.transform_to('icrs')
            lon = skydir.ra.deg
            lat = skydir.dec.deg
        else:
            skydir = skydir.transform_to('galactic')
            lon = skydir.l.deg
            lat = skydir.b.deg

        return self.get_pixel_indices(lat, lon)


class HpxToWcsMapping(object):
    """ Stores the indices need to conver from HEALPix to WCS """

    def __init__(self, hpx, wcs, mapping_data=None):
        """
        """
        self._hpx = hpx
        self._wcs = wcs
        if mapping_data is None:
            self._ipixs, self._mult_val, self._npix = make_hpx_to_wcs_mapping(
                self.hpx, self.wcs.wcs)
        else:
            self._ipixs = mapping_data['ipixs']
            self._mult_val = mapping_data['mult_val']
            self._npix = mapping_data['npix']
        self._lmap = self._hpx[self._ipixs]
        self._valid = self._lmap > 0

    @property
    def hpx(self):
        """ The HEALPix projection """
        return self._hpx

    @property
    def wcs(self):
        """ The WCS projection """
        return self._wcs

    @property
    def ipixs(self):
        """An array(nx,ny) of the global HEALPix pixel indices for each WCS
        pixel"""
        return self._ipixs

    @property
    def mult_val(self):
        """An array(nx,ny) of 1/number of WCS pixels pointing at each HEALPix
        pixel"""
        return self._mult_val

    @property
    def npix(self):
        """ A tuple(nx,ny) of the shape of the WCS grid """
        return self._npix

    @property
    def lmap(self):
        """An array(nx,ny) giving the mapping of the local HEALPix pixel
        indices for each WCS pixel"""
        return self._lmap

    @property
    def valid(self):
        """An array(nx,ny) of bools giving if each WCS pixel in inside the
        HEALPix region"""
        return self._valid

    def write_to_fitsfile(self, fitsfile, clobber=True):
        """Write this mapping to a FITS file, to avoid having to recompute it
        """
        from fermipy.skymap import Map
        hpx_header = self._hpx.make_header()
        index_map = Map(self.ipixs, self.wcs)
        mult_map = Map(self.mult_val, self.wcs)
        prim_hdu = index_map.create_primary_hdu()
        mult_hdu = index_map.create_image_hdu()
        for key in ['COORDSYS', 'ORDERING', 'PIXTYPE',
                    'ORDERING', 'ORDER', 'NSIDE',
                    'FIRSTPIX', 'LASTPIX']:
            prim_hdu.header[key] = hpx_header[key]
            mult_hdu.header[key] = hpx_header[key]

        hdulist = fits.HDUList([prim_hdu, mult_dhu])
        hdulist.writeto(fitsfile, clobber=clobber)

    @staticmethod
    def create_from_fitsfile(self, fitsfile):
        """ Read a fits file and use it to make a mapping
        """
        from fermipy.skymap import Map
        index_map = Map.create_from_fits(fitsfile)
        mult_map = Map.create_from_fits(fitsfile, hdu=1)
        ff = fits.open(fitsfile)
        hpx = HPX.create_from_header(ff[0])
        mapping_data = dict(ipixs=index_map.counts,
                            mult_val=mult_map.counts,
                            npix=mult_map.counts.shape)
        return HpxToWcsMapping(hpx, index_map.wcs, mapping_data)

    def fill_wcs_map_from_hpx_data(self, hpx_data, wcs_data, normalize=True):
        """Fills the wcs map from the hpx data using the pre-calculated
        mappings

        hpx_data  : the input HEALPix data
        wcs_data  : the data array being filled
        normalize : True -> perserve integral by splitting HEALPix values between bins

        """
        # FIXME, there really ought to be a better way to do this
        hpx_data_flat = hpx_data.flatten()
        wcs_data_flat = np.zeros((wcs_data.size))
        lmap_valid = self._lmap[self._valid]
        wcs_data_flat[self._valid] = hpx_data_flat[lmap_valid]
        if normalize:
            wcs_data_flat *= self._mult_val
        wcs_data.flat = wcs_data_flat

    def make_wcs_data_from_hpx_data(self, hpx_data, wcs, normalize=True):
        """ Creates and fills a wcs map from the hpx data using the pre-calculated
        mappings

        hpx_data  : the input HEALPix data
        wcs       : the WCS object
        normalize : True -> perserve integral by splitting HEALPix values between bins
        """
        wcs_data = np.zeros(wcs.npix)
        self.fill_wcs_map_from_hpx_data(hpx_data, wcs_data, normalize)
        return wcs_data
