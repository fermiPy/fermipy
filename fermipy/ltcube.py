# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import glob
import re
import copy
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column

from fermipy import utils
from fermipy.utils import edge_to_center
from fermipy.utils import edge_to_width
from fermipy.utils import angle_to_cartesian
from fermipy.skymap import HpxMap
from fermipy.hpx_utils import HPX


def fill_livetime_hist(skydir, tab_sc, tab_gti, zmax, costh_edges):
    """Generate a sequence of livetime distributions at the sky
    positions given by ``skydir``.  The output of the method are two
    NxM arrays containing a sequence of histograms for N sky positions
    and M incidence angle bins where the bin edges are defined by
    ``costh_edges``.  This method uses the same algorithm as
    `gtltcube` with the exception that SC time intervals are assumed
    to be aligned with GTIs.

    Parameters
    ----------
    skydir : `~astropy.coordinates.SkyCoord`    
        Vector of sky directions for which livetime histograms will be
        accumulated.

    tab_sc : `~astropy.table.Table`    
        Spacecraft table.  Must contain the following columns: START,
        STOP, LIVETIME, RA_SCZ, DEC_SZ, RA_ZENITH, DEC_ZENITH.

    tab_gti : `~astropy.table.Table`
        Table of good time intervals (GTIs).

    zmax : float
        Zenith cut.

    costh_edges : `~numpy.ndarray`
        Incidence angle bin edges in cos(angle).

    Returns
    -------
    lt : `~numpy.ndarray`
        Array of livetime histograms.

    lt_wt : `~numpy.ndarray`    
        Array of histograms of weighted livetime (livetime x livetime
        fraction).
    """

    if len(tab_gti) == 0:
        shape = (len(costh_edges) - 1, len(skydir))
        return (np.zeros(shape), np.zeros(shape))

    m = (tab_sc['START'] < tab_gti['STOP'][-1])
    m &= (tab_sc['STOP'] > tab_gti['START'][0])
    tab_sc = tab_sc[m]

    cos_zmax = np.cos(np.radians(zmax))
    sc_t0 = np.array(tab_sc['START'].data)
    sc_t1 = np.array(tab_sc['STOP'].data)
    sc_live = np.array(tab_sc['LIVETIME'].data)
    sc_lfrac = sc_live / (sc_t1 - sc_t0)

    sc_xyz = angle_to_cartesian(np.radians(tab_sc['RA_SCZ'].data),
                                np.radians(tab_sc['DEC_SCZ'].data))
    zn_xyz = angle_to_cartesian(np.radians(tab_sc['RA_ZENITH'].data),
                                np.radians(tab_sc['DEC_ZENITH'].data))

    tab_gti_t0 = np.array(tab_gti['START'].data)
    tab_gti_t1 = np.array(tab_gti['STOP'].data)

    # Index of the closest GTI interval
    idx = np.digitize(sc_t0, tab_gti_t0) - 1

    # start/stop time of closest GTI interval
    gti_t0 = np.zeros_like(sc_t0)
    gti_t1 = np.zeros_like(sc_t1)
    gti_t0[idx >= 0] = tab_gti_t0[idx[idx >= 0]]
    gti_t1[idx >= 0] = tab_gti_t1[idx[idx >= 0]]

    nbin = len(costh_edges) - 1

    lt = np.zeros((nbin,) + skydir.shape)
    lt_wt = np.zeros((nbin,) + skydir.shape)

    m0 = (idx >= 0) & (sc_t0 >= gti_t0) & (sc_t1 <= gti_t1)

    xyz = angle_to_cartesian(skydir.ra.rad, skydir.dec.rad)

    for i, t in enumerate(xyz):
        cos_sep = utils.dot_prod(t, sc_xyz)
        cos_zn = utils.dot_prod(t, zn_xyz)
        m = m0 & (cos_zn > cos_zmax) & (cos_sep > 0.0)
        bins = np.digitize(cos_sep[m], bins=costh_edges) - 1
        bins = np.clip(bins, 0, nbin - 1)
        lt[:, i] = np.bincount(bins, weights=sc_live[m], minlength=nbin)
        lt_wt[:, i] = np.bincount(bins, weights=sc_live[m] * sc_lfrac[m],
                                  minlength=nbin)

    return lt, lt_wt


class LTCube(HpxMap):
    """Class for reading and manipulating livetime cubes generated with
    gtltcube.
    """

    def __init__(self, data, hpx, cth_edges, **kwargs):
        HpxMap.__init__(self, data, hpx)
        self._cth_edges = cth_edges
        self._cth_center = edge_to_center(self._cth_edges)
        self._cth_width = edge_to_width(self._cth_edges)
        self._domega = (self._cth_edges[1:] -
                        self._cth_edges[:-1]) * 2 * np.pi
        self._tstart = kwargs.get('tstart', None)
        self._tstop = kwargs.get('tstop', None)
        self._zmin = kwargs.get('zmin', 0.0)
        self._zmax = kwargs.get('zmax', 180.0)
        self._tab_gti = kwargs.get('tab_gti', None)
        self._header = kwargs.get('header', None)
        self._data_wt = kwargs.get('data_wt', None)

        if self._data_wt is None:
            self._data_wt = np.zeros_like(self.data)

        if self._tab_gti is None:
            cols = [Column(name='START', dtype='f8', unit='s'),
                    Column(name='STOP', dtype='f8', unit='s')]
            self._tab_gti = Table(cols)

    @property
    def data_wt(self):
        """Return the weighted livetime vector."""
        return self._data_wt

    @property
    def tstart(self):
        """Return start time."""
        return self._tstart

    @property
    def tstop(self):
        """Return stop time."""
        return self._tstop

    @property
    def zmin(self):
        """Return start time."""
        return self._zmin

    @property
    def zmax(self):
        """Return stop time."""
        return self._zmax

    @property
    def domega(self):
        """Return solid angle of incidence angle bins in steradians."""
        return self._domega

    @property
    def costh_edges(self):
        """Return edges of incidence angle bins in cosine of the incidence
        angle."""
        return self._cth_edges

    @property
    def costh_center(self):
        """Return centers of incidence angle bins in cosine of the incidence
        angle.
        """
        return self._cth_center

    @staticmethod
    def create(ltfile):
        """Create a livetime cube from a single file or list of
        files."""

        if not re.search('\.txt?', ltfile) is None:
            files = np.loadtxt(ltfile, unpack=True, dtype='str')
        elif not isinstance(ltfile, list):
            files = glob.glob(ltfile)

        ltc = LTCube.create_from_fits(files[0])
        for f in files[1:]:
            ltc.load_ltfile(f)

        return ltc

    @staticmethod
    def create_from_fits(ltfile):

        hdulist = fits.open(ltfile)
        data = hdulist['EXPOSURE'].data.field('COSBINS')
        data_wt = hdulist['WEIGHTED_EXPOSURE'].data.field('COSBINS')
        data = data.astype(float)
        data_wt = data_wt.astype(float)
        tstart = hdulist[0].header['TSTART']
        tstop = hdulist[0].header['TSTOP']
        zmin = hdulist['EXPOSURE'].header['ZENMIN']
        zmax = hdulist['EXPOSURE'].header['ZENMAX']

        cth_min = np.array(hdulist['CTHETABOUNDS'].data.field('CTHETA_MIN'))
        cth_max = np.array(hdulist['CTHETABOUNDS'].data.field('CTHETA_MAX'))
        cth_min = cth_min.astype(float)
        cth_max = cth_max.astype(float)
        cth_edges = np.concatenate((cth_max[:1], cth_min))[::-1]
        hpx = HPX.create_from_header(hdulist['EXPOSURE'].header, cth_edges)
        #header = dict(hdulist['EXPOSURE'].header)
        tab_gti = Table.read(ltfile, 'GTI')
        return LTCube(data[:, ::-1].T, hpx, cth_edges,
                      tstart=tstart, tstop=tstop,
                      zmin=zmin, zmax=zmax, tab_gti=tab_gti,
                      data_wt=data_wt[:, ::-1].T)

    @staticmethod
    def create_empty(tstart, tstop, fill=0.0, nside=64):
        """Create an empty livetime cube."""
        cth_edges = np.linspace(0, 1.0, 41)
        domega = utils.edge_to_width(cth_edges) * 2.0 * np.pi
        hpx = HPX(nside, True, 'CEL', ebins=cth_edges)
        data = np.ones((len(cth_edges) - 1, hpx.npix)) * fill
        return LTCube(data, hpx, cth_edges, tstart=tstart, tstop=tstop)

    @staticmethod
    def create_from_obs_time(obs_time, nside=64):

        tstart = 239557417.0
        tstop = tstart + obs_time
        ltc = LTCube.create_empty(tstart, tstop, obs_time, nside)
        ltc._counts *= ltc.domega[:, np.newaxis] / (4. * np.pi)
        return ltc

    @staticmethod
    def create_from_gti(skydir, tab_sc, tab_gti, zmax, **kwargs):

        radius = kwargs.get('radius', 180.0)
        cth_edges = kwargs.get('cth_edges', None)
        if cth_edges is None:
            cth_edges = 1.0 - np.linspace(0, 1.0, 41)**2
            cth_edges = cth_edges[::-1]

        hpx = HPX(2**4, True, 'CEL', ebins=cth_edges)

        hpx_skydir = hpx.get_sky_dirs()

        m = skydir.separation(hpx_skydir).deg < radius
        map_lt = HpxMap(np.zeros((40, hpx.npix)), hpx)
        map_lt_wt = HpxMap(np.zeros((40, hpx.npix)), hpx)

        lt, lt_wt = fill_livetime_hist(
            hpx_skydir[m], tab_sc, tab_gti, zmax, cth_edges)
        map_lt.data[:, m] = lt
        map_lt_wt.data[:, m] = lt_wt

        hpx2 = HPX(2**6, True, 'CEL', ebins=cth_edges)

        ltc = LTCube(np.zeros((len(cth_edges) - 1, hpx2.npix)),
                     hpx2, cth_edges)
        ltc_skydir = ltc.hpx.get_sky_dirs()
        m = skydir.separation(ltc_skydir).deg < radius

        ltc.data[:, m] = map_lt.interpolate(ltc_skydir[m].ra.deg,
                                            ltc_skydir[m].dec.deg,
                                            interp_log=False)
        ltc.data_wt[:, m] = map_lt_wt.interpolate(ltc_skydir[m].ra.deg,
                                                  ltc_skydir[m].dec.deg,
                                                  interp_log=False)
        return ltc

    def load_ltfile(self, ltfile):

        ltc = LTCube.create_from_fits(ltfile)
        self._counts += ltc.data
        self._tstart = min(self.tstart, ltc.tstart)
        self._tstop = max(self.tstop, ltc.tstop)

    def get_skydir_lthist(self, skydir, cth_bins):
        """Get the livetime distribution (observing profile) for a given sky
        direction with binning in incidence angle defined by
        ``cth_bins``.

        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord`
            Sky coordinate for which the observing profile will be
            computed.

        cth_bins : `~numpy.ndarray`
            Bin edges in cosine of the incidence angle.

        """
        ra = skydir.ra.deg
        dec = skydir.dec.deg

        npts = 1
        bins = utils.split_bin_edges(cth_bins, npts)

        center = edge_to_center(bins)
        width = edge_to_width(bins)
        ipix = hp.ang2pix(self.hpx.nside, np.pi / 2. - np.radians(dec),
                          np.radians(ra), nest=self.hpx.nest)
        lt = np.interp(center, self._cth_center,
                       self.data[:, ipix] / self._cth_width) * width
        lt = np.sum(lt.reshape(-1, npts), axis=1)
        return lt

    def create_skydir_ltcube(self, skydir, tab_sc, tab_gti, zmax):
        """Create a new livetime cube by scaling this one by the
        observing profile ratio in the direction ``skydir``.  This
        method can be used to generate an approximate livetime cube
        that is accurate in the vicinity of ``skydir``.

        Parameters
        ----------
        skydir :  `~astropy.coordinates.SkyCoord`

        tab_sc : `~astropy.table.Table`
            Spacecraft (FT2) table.

        tab_gti : `~astropy.table.Table`
            Table of GTIs.

        zmax : float
            Zenith angle cut.
        """

        skydir = SkyCoord(np.array([skydir.ra.deg]),
                          np.array([skydir.dec.deg]), unit='deg')

        lt, lt_wt = fill_livetime_hist(skydir, tab_sc, tab_gti, zmax,
                                       self.costh_edges)

        ipix = self.hpx.skydir_to_pixel(skydir)

        lt_scale = np.ones_like(lt)
        lt_wt_scale = np.ones_like(lt_wt)
        m = self.data[:, ipix] > 0.0

        lt_scale[m] = lt[m] / self.data[:, ipix][m]
        lt_wt_scale[m] = lt_wt[m] / self._data_wt[:, ipix][m]
        data = self.data * lt_scale
        data_wt = self._data_wt * lt_wt_scale
        return LTCube(data, copy.deepcopy(self.hpx), self.costh_edges,
                      # tstart=np.min(tab_gti_t0),
                      # tstop=np.max(tab_gti_t1),
                      zmax=zmax, data_wt=data_wt)

    def _create_exp_hdu(self, data):

        pix_skydir = self.hpx.get_sky_dirs()
        cols = [Column(name='COSBINS', unit='s', dtype='f4',
                       data=data.T[:, ::-1],
                       shape=(len(self.costh_center),)),
                Column(name='RA', unit='deg', dtype='f4',
                       data=pix_skydir.ra.deg),
                Column(name='DEC', unit='deg', dtype='f4',
                       data=pix_skydir.dec.deg)]

        hdu_exp = fits.table_to_hdu(Table(cols))
        hdu_exp.header['THETABIN'] = 'SQRT(1-COSTHETA)'
        hdu_exp.header['NBRBINS'] = len(self.costh_center)
        hdu_exp.header['COSMIN'] = self.costh_edges[0]
        hdu_exp.header['ZENMIN'] = self.zmin
        hdu_exp.header['ZENMAX'] = self.zmax
        hdu_exp.header['NDSKEYS'] = 0
        hdu_exp.header['PHIBINS'] = 0

        header = self.hpx.make_header()
        hdu_exp.header.update(header)

        return hdu_exp

    def write(self, outfile):
        """Write the livetime cube to a FITS file."""

        hdu_pri = fits.PrimaryHDU()

        hdu_exp = self._create_exp_hdu(self.data)
        hdu_exp.name = 'EXPOSURE'
        hdu_exp_wt = self._create_exp_hdu(self._data_wt)
        hdu_exp_wt.name = 'WEIGHTED_EXPOSURE'

        cols = [Column(name='CTHETA_MIN', dtype='f4',
                       data=self.costh_edges[:-1][::-1]),
                Column(name='CTHETA_MAX',  dtype='f4',
                       data=self.costh_edges[1:][::-1]), ]
        hdu_bnds = fits.table_to_hdu(Table(cols))
        hdu_bnds.name = 'CTHETABOUNDS'

        hdu_gti = fits.table_to_hdu(self._tab_gti)
        hdu_gti.name = 'GTI'

        hdus = [hdu_pri, hdu_exp, hdu_exp_wt,
                hdu_bnds, hdu_gti]

        for hdu in hdus:
            hdu.header['TSTART'] = self.tstart
            hdu.header['TSTOP'] = self.tstop

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(outfile, clobber=True)
