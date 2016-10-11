# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import glob
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import healpy as hp
from astropy.io import fits

import pyIrfLoader

pyIrfLoader.Loader_go()

from fermipy import utils
from fermipy.utils import edge_to_center
from fermipy.utils import edge_to_width
from fermipy.skymap import HpxMap
from fermipy.hpx_utils import HPX


evtype_string = {
    1: 'FRONT',
    2: 'BACK',
    4: 'PSF0',
    8: 'PSF1',
    16: 'PSF2',
    32: 'PSF3'
}


def bitmask_to_bits(mask):

    bits = []
    for i in range(32):
        if mask & (2**i):
            bits += [2**i]

    return bits


def poisson_log_like(c, m):
    return c * np.log(m) - m


def poisson_ts(sig, bkg):
    return 2 * (poisson_log_like(sig + bkg, sig + bkg) - poisson_log_like(sig + bkg, bkg))


def compute_ext_flux(egy, flux):
    pass


def compute_ps_loc(egy, flux):
    """Solve for the localization precision of a point source with a given flux."""
    pass


def compute_ps_counts(ebins, exp, psf, bkg, fn, egy_dim=0):
    """Calculate the observed signal and background counts given models
    for the exposure, background intensity, PSF, and source flux.

    Parameters
    ----------
    ebins : `~numpy.ndarray`
        Array of energy bin edges.

    exp : `~numpy.ndarray`
        Model for exposure.

    psf : `~fermipy.irfs.PSFModel`
        Model for average PSF.

    bkg : `~numpy.ndarray`
        Array of background intensities.

    fn : `~fermipy.spectrum.SpectralFunction`

    egy_dim : int
        Index of energy dimension in ``bkg`` and ``exp`` arrays.

    """
    ewidth = utils.edge_to_width(ebins)
    ectr = np.exp(utils.edge_to_center(np.log(ebins)))

    theta_edges = np.linspace(0.0, 3.0, 31)[
        np.newaxis, :] * np.ones((len(ectr), 31))
    theta_edges *= psf.containment_angle(ectr, fraction=0.68)[:, np.newaxis]
    theta = 0.5 * (theta_edges[:, :-1] + theta_edges[:, 1:])
    domega = np.pi * (theta_edges[:, 1:]**2 - theta_edges[:, :-1]**2)

    sig_pdf = domega * \
        psf.interp(ectr[:, np.newaxis], theta) * (np.pi / 180.)**2
    sig_flux = fn.flux(ebins[:-1], ebins[1:])

    # Background and signal counts
    bkgc = bkg[:, np.newaxis] * domega * exp[:, np.newaxis] * \
        ewidth[:, np.newaxis] * (np.pi / 180.)**2
    sigc = sig_pdf * sig_flux[:, np.newaxis] * exp[:, np.newaxis]

    return sigc, bkgc


def compute_norm(sig, bkg, ts_thresh, min_counts, sum_axes=None):
    """Solve for the normalization of the signal distribution at which the
    detection test statistic (twice delta-loglikelihood ratio) is >=
    ``ts_thresh`` AND the number of signal counts >= ``min_counts``.
    This function uses the Asimov method to calculate the median
    expected TS when the model for the background is fixed (no
    uncertainty on the background amplitude).

    Parameters
    ----------
    sig : `~numpy.ndarray`
        Array of signal amplitudes in counts.

    bkg : `~numpy.ndarray`
        Array of background amplitudes in counts.

    ts_thresh : float
        Test statistic threshold.

    min_counts : float
        Counts threshold.

    sum_axes : list
        Axes over which the source test statistic should be summed.
        By default the summation will be performed over all
        dimensions.

    """

    sig_scale = 10**np.linspace(0.0, 5.0, 101)
    if sum_axes is None:
        sum_axes = np.arange(sig.ndim)

    sig = np.expand_dims(sig, -1)
    bkg = np.expand_dims(bkg, -1)
    sig_sum = np.apply_over_axes(np.sum, sig, sum_axes)
    sig_scale = sig_scale * min_counts / sig_sum
    ts = np.apply_over_axes(np.sum, poisson_ts(sig * sig_scale, bkg), sum_axes)
    vals = np.ones(ts.shape[:-1])

    for idx, v in np.ndenumerate(ts[..., 0]):
        vals[idx] = np.interp(ts_thresh, ts[idx], sig_scale[idx])
    return vals


class Exposure(HpxMap):

    def __init__(self, data, hpx):
        HpxMap.__init__(self, data, hpx)

    @staticmethod
    def create(ltc, event_class, event_types, log_energies):

        exp = np.zeros((len(log_energies), ltc.hpx.npix))
        for et in event_types:
            aeff = create_aeff(event_class, et, log_energies, ltc.costh_center)
            exp += np.sum(aeff.T[:, :, np.newaxis] *
                          ltc.data[:, np.newaxis, :], axis=0)

        hpx = HPX(ltc.hpx.nside, ltc.hpx.nest,
                  ltc.hpx.coordsys, ebins=log_energies)
        return Exposure(exp, hpx)


class PSFModel(object):

    def __init__(self, skydir, ltc, event_class, event_types,
                 log_energies, cth_min=0.2, ndtheta=1000, ncth=40):

        if isinstance(event_types, int):
            event_types = bitmask_to_bits(event_types)

        self._dtheta = np.logspace(-4, 1.75, ndtheta)
        self._dtheta = np.insert(self._dtheta, 0, [0])
        self._log_energies = log_energies
        self._energies = 10**log_energies
        self._scale_fn = None

        self._exp = np.zeros(len(log_energies))
        self._psf = self.create_average_psf(skydir, ltc, event_class, event_types,
                                            self._dtheta, log_energies, cth_min, ncth)

        self._psf_fn = RegularGridInterpolator((self._dtheta, log_energies),
                                               np.log(self._psf),
                                               bounds_error=False,
                                               fill_value=None)

        cth_edge = np.linspace(cth_min, 1.0, ncth + 1)
        cth = edge_to_center(cth_edge)
        ltw = ltc.get_skydir_lthist(skydir, cth_edge)
        for et in event_types:
            aeff = create_aeff(event_class, et, log_energies, cth)
            self._exp += np.sum(aeff * ltw[np.newaxis, :], axis=1)

    def eval(self, ebin, dtheta, scale_fn=None):
        """Evaluate the PSF at one of the source map energies.

        Parameters
        ----------
        ebin : int
            Index of energy bin.

        dtheta : array_like
            Array of angular separations in degrees.

        scale_fn : callable        
            Function that evaluates the PSF scaling function.
            Argument is energy in MeV.
        """

        if scale_fn is None and self.scale_fn is not None:
            scale_fn = self.scale_fn

        if scale_fn is None:
            scale_factor = 1.0
        else:
            dtheta = dtheta / scale_fn(self.energies[ebin])
            scale_factor = 1. / scale_fn(self.energies[ebin])**2

        vals = 10**np.interp(dtheta, self.dtheta, np.log10(self.val[:, ebin]))
        return vals * scale_factor

    def interp(self, energies, dtheta, scale_fn=None):
        """Evaluate the PSF model at an array of energies and angular
        separations.

        Parameters
        ----------
        energies : array_like
            Array of energies in MeV.

        dtheta : array_like
            Array of angular separations in degrees.

        scale_fn : callable        
            Function that evaluates the PSF scaling function.
            Argument is energy in MeV.
        """

        if scale_fn is None and self.scale_fn:
            scale_fn = self.scale_fn

        log_energies = np.log10(energies)

        shape = (energies * dtheta).shape
        scale_factor = np.ones(shape)

        if scale_fn is not None:
            dtheta = dtheta / scale_fn(energies)
            scale_factor = 1. / scale_fn(energies)**2

        vals = np.exp(self._psf_fn((dtheta, log_energies)))
        return vals * scale_factor

    def containment_angle(self, energies=None, fraction=0.68, scale_fn=None):
        """Evaluate the PSF containment angle at a sequence of energies."""

        if scale_fn is None and self.scale_fn:
            scale_fn = self.scale_fn

        if energies is None:
            energies = self.energies

        vals = self.interp(energies[np.newaxis, :], self.dtheta[:, np.newaxis],
                           scale_fn=scale_fn)
        dtheta = np.radians(self.dtheta[:, np.newaxis] * np.ones(vals.shape))

        delta = dtheta[1:] - dtheta[:-1]
        ctr = 0.5 * (dtheta[1:] + dtheta[:-1])

        avg_val = 0.5 * (vals[1:, :] * np.sin(dtheta[1:, :]) +
                         vals[:-1, :] * np.sin(dtheta[:-1, :]))

        csum = delta * avg_val * 2 * np.pi
        csum = np.cumsum(csum, axis=0)
        theta = np.zeros(len(energies))

        for i in range(len(theta)):
            theta[i] = np.degrees(np.interp(fraction, csum[:, i],
                                            dtheta[1:, i]))

        return theta

    def set_scale_fn(self, scale_fn):
        self._scale_fn = scale_fn

    @property
    def scale_fn(self):
        return self._scale_fn

    @property
    def dtheta(self):
        return self._dtheta

    @property
    def log_energies(self):
        return self._log_energies

    @property
    def energies(self):
        return self._energies

    @property
    def val(self):
        return self._psf

    @property
    def exp(self):
        return self._exp

    @staticmethod
    def create_average_psf(skydir, ltc, event_class, event_types, dtheta, egy,
                           cth_min=0.2, ncth=40):

        if isinstance(event_types, int):
            event_types = bitmask_to_bits(event_types)

        cth_edge = np.linspace(cth_min, 1.0, ncth + 1)
        cth = edge_to_center(cth_edge)

        wpsf = np.zeros((len(dtheta), len(egy)))
        exps = np.zeros(len(egy))

        ltw = ltc.get_skydir_lthist(skydir, cth_edge)

        for et in event_types:
            psf = create_psf(event_class, et, dtheta, egy, cth)
            aeff = create_aeff(event_class, et, egy, cth)

            wpsf += np.sum(psf * aeff[np.newaxis, :, :] *
                           ltw[np.newaxis, np.newaxis, :], axis=2)
            exps += np.sum(aeff * ltw[np.newaxis, :], axis=1)

        wpsf /= exps[np.newaxis, :]

        return wpsf


def create_irf(event_class, event_type):
    if isinstance(event_type, int):
        event_type = evtype_string[event_type]

    irf_factory = pyIrfLoader.IrfsFactory.instance()
    irfname = '%s::%s' % (event_class, event_type)
    irf = irf_factory.create(irfname)
    return irf


def create_psf(event_class, event_type, dtheta, egy, cth):
    """This function creates a map of the PSF versus offset angle.  

    """
    irf = create_irf(event_class, event_type)
    theta = np.degrees(np.arccos(cth))
    m = np.zeros((len(dtheta), len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):
            m[:, i, j] = irf.psf().value(dtheta, 10**x, y, 0.0)

    return m


def create_aeff(event_class, event_type, egy, cth):
    """This function creates a map of effective area versus energy and
    incidence angle.  Binning in energy and incidence angle is
    controlled with the egy and cth input parameters.

    Parameters
    ----------
    event_class : str
        Event class string (e.g. P8R2_SOURCE_V6).

    event_type : list

    egy : array_like
        Evaluation points in energy (MeV).

    cth : array_like
        Evaluation points in cosine of the incidence angle.
    """
    irf = create_irf(event_class, event_type)
    irf.aeff().setPhiDependence(False)
    theta = np.degrees(np.arccos(cth))

    # Exposure Matrix
    # Dimensions are Etrue and incidence angle
    m = np.zeros((len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):
            m[i, j] = irf.aeff().value(10**x, y, 0.0)

    return m


class LTCube(HpxMap):
    """Class for reading and manipulating livetime cubes generated with
    gtltcube.
    """

    def __init__(self, data, hpx, cth_edges, tstart=None, tstop=None):
        HpxMap.__init__(self, data, hpx)
        self._cth_edges = cth_edges
        self._cth_center = edge_to_center(self._cth_edges)
        self._cth_width = edge_to_width(self._cth_edges)
        self._domega = (self._cth_edges[1:] -
                        self._cth_edges[:-1]) * 2 * np.pi
        self._tstart = tstart
        self._tstop = tstop

    @property
    def tstart(self):
        """Return start time."""
        return self._tstart

    @property
    def tstop(self):
        """Return stop time."""
        return self._tstop

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

        ltc = LTCube.create_from_file(files[0])
        for f in files[1:]:
            ltc.load_ltfile(f)

        return ltc

    @staticmethod
    def create_from_file(ltfile):

        hdulist = fits.open(ltfile)
        data = hdulist['EXPOSURE'].data.field(0)
        tstart = hdulist[0].header['TSTART']
        tstop = hdulist[0].header['TSTOP']
        cth_edges = np.array(hdulist['CTHETABOUNDS'].data.field(0))
        cth_edges = np.concatenate(([1], cth_edges))
        cth_edges = cth_edges[::-1]
        hpx = HPX.create_from_header(hdulist['EXPOSURE'].header, cth_edges)
        return LTCube(data[:, ::-1].T, hpx, cth_edges, tstart, tstop)

    @staticmethod
    def create_empty(tstart, tstop, fill=0.0, nside=64):
        """Create an empty livetime cube."""
        cth_edges = np.linspace(0, 1.0, 41)
        hpx = HPX(nside, True, 'CEL', ebins=cth_edges)
        data = np.ones((len(cth_edges) - 1, hpx.npix)) * fill
        return LTCube(data, hpx, cth_edges, tstart, tstop)

    def load_ltfile(self, ltfile):

        ltc = LTCube.create_from_file(ltfile)
        self._counts += ltc.data
        self._tstart = min(self.tstart, ltc.tstart)
        self._tstop = max(self.tstop, ltc.tstop)

    def get_skydir_lthist(self, skydir, cth_edges):
        """Get the livetime distribution (observing profile) for a
        given sky direction.

        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord`

        cth_edges : `~numpy.ndarray`
            Bin edges in cosine of the incidence angle.
        """
        ra = skydir.ra.deg
        dec = skydir.dec.deg

        edges = np.linspace(cth_edges[0], cth_edges[-1],
                            (len(cth_edges) - 1) * 4 + 1)
        center = edge_to_center(edges)
        width = edge_to_width(edges)
        ipix = hp.ang2pix(self.hpx.nside, np.pi / 2. - np.radians(dec),
                          np.radians(ra), nest=self.hpx.nest)
        lt = np.interp(center, self._cth_center,
                       self.data[:, ipix] / self._cth_width) * width
        lt = np.sum(lt.reshape(-1, 4), axis=1)
        return lt


def plot_hpxmap(hpxmap, **kwargs):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm, Normalize, LogNorm

    kwargs_imshow = {'norm': None,
                     'vmin': None, 'vmax': None}

    gamma = kwargs.get('gamma', 2.0)
    zscale = kwargs.get('zscale', None)
    cbar = kwargs.get('cbar', None)
    cbar_label = kwargs.get('cbar_label', '')
    title = kwargs.get('title', '')
    levels = kwargs.get('levels', None)
    rot = kwargs.get('rot', None)

    kwargs_imshow['vmin'] = kwargs.get('vmin', None)
    kwargs_imshow['vmax'] = kwargs.get('vmax', None)

    cmap = mpl.cm.get_cmap(kwargs.get('cmap', 'jet'))
    cmap.set_under('white')
    kwargs_imshow['cmap'] = cmap

    if zscale == 'pow':
        vmed = np.median(hpxmap.counts)
        vmax = max(hpxmap.counts)
        vmin = min(1.1 * hpxmap.counts[hpxmap.counts > 0])
        kwargs_imshow['norm'] = PowerNorm(gamma=gamma, clip=True)
    elif zscale == 'log':
        kwargs_imshow['norm'] = LogNorm()
    else:
        kwargs_imshow['norm'] = Normalize(clip=True)

    from healpy import projaxes as PA

    fig = plt.gcf()
    if 'sub' in kwargs:
        sub = kwargs['sub']
        nrows, ncols, idx = sub / 100, (sub % 100) / 10, (sub % 10)
        c, r = (idx - 1) % ncols, (idx - 1) / ncols
        margins = (0.01, 0.0, 0.0, 0.02)
        extent = (c * 1. / ncols + margins[0],
                  1. - (r + 1) * 1. / nrows + margins[1],
                  1. / ncols - margins[2] - margins[0],
                  1. / nrows - margins[3] - margins[1])
        extent = (extent[0] + margins[0],
                  extent[1] + margins[1],
                  extent[2] - margins[2] - margins[0],
                  extent[3] - margins[3] - margins[1])
    else:
        extent = (0.02, 0.05, 0.96, 0.9)

    ax = hp.projaxes.HpxMollweideAxes(fig, extent, coord=None, rot=rot,
                                      format='%g', flipconv='astro')

    ax.set_title(title)
    fig.add_axes(ax)

    if hpxmap.data.ndim == 2:
        data = np.sum(hpxmap.data, axis=0)
    else:
        data = hpxmap.data

    img0 = ax.projmap(data, nest=hpxmap.hpx.nest, xsize=1600, coord='C',
                      **kwargs_imshow)

    if levels:
        cs = ax.contour(img0, extent=ax.proj.get_extent(),
                        levels=levels, colors=['k'],
                        interpolation='nearest')

    hp.visufunc.graticule(verbose=False, lw=0.5, color='k')

    if cbar is not None:

        im = ax.get_images()[0]
        cb_kw = dict(orientation='vertical',
                     shrink=.6, pad=0.05)
        cb_kw.update(cbar)
        cb = fig.colorbar(im, **cb_kw)  # ,format='%.3g')
        cb.set_label(cbar_label)
        #, ticks=[min, max])
#            cb.ax.xaxis.set_label_text(cbar_label)
