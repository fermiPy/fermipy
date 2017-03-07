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
from fermipy import spectrum
from fermipy.utils import edge_to_center
from fermipy.utils import edge_to_width
from fermipy.utils import sum_bins
from fermipy.skymap import HpxMap
from fermipy.hpx_utils import HPX
from fermipy.ltcube import LTCube

evtype_string = {
    1: 'FRONT',
    2: 'BACK',
    4: 'PSF0',
    8: 'PSF1',
    16: 'PSF2',
    32: 'PSF3',
    64: 'EDISP0',
    128: 'EDISP1',
    256: 'EDISP2',
    512: 'EDISP3',
}


def loglog_quad(x, y, dim):

    ys0 = [slice(None)] * y.ndim
    ys1 = [slice(None)] * y.ndim

    xs0 = [None] * y.ndim
    xs1 = [None] * y.ndim

    ys0[dim] = slice(None, -1)
    ys1[dim] = slice(1, None)

    xs0[dim] = slice(None, -1)
    xs1[dim] = slice(1, None)
    log_ratio = np.log(x[xs1] / x[xs0])
    return 0.5 * (y[ys0] * x[xs0] + y[ys1] * x[xs1]) * log_ratio


def bins_per_dec(edges):
    return (len(edges) - 1) / np.log10(edges[-1] / edges[0])


def bitmask_to_bits(mask):

    bits = []
    for i in range(32):
        if mask & (2**i):
            bits += [2**i]

    return bits


def poisson_log_like(c, m):
    return c * np.log(m) - m


def poisson_ts(sig, bkg):
    return 2 * (poisson_log_like(sig + bkg, sig + bkg) -
                poisson_log_like(sig + bkg, bkg))


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


class ExposureMap(HpxMap):

    def __init__(self, data, hpx):
        HpxMap.__init__(self, data, hpx)

    @staticmethod
    def create(ltc, event_class, event_types, ebins):
        """Create an exposure map from a livetime cube.  This method will
        generate an exposure map with the same geometry as the
        livetime cube (nside, etc.).

        Parameters
        ----------
        ltc : `~fermipy.irfs.LTCube`
            Livetime cube object.

        event_class : str
            Event class string.

        event_types : list
            List of event type strings, e.g. ['FRONT','BACK'].

        ebins :  `~numpy.ndarray`
            Energy bin edges in MeV.

        """

        evals = np.sqrt(ebins[1:] * ebins[:-1])
        exp = np.zeros((len(evals), ltc.hpx.npix))
        for et in event_types:
            aeff = create_aeff(event_class, et, evals, ltc.costh_center)
            exp += np.sum(aeff.T[:, :, np.newaxis] *
                          ltc.data[:, np.newaxis, :], axis=0)

        hpx = HPX(ltc.hpx.nside, ltc.hpx.nest,
                  ltc.hpx.coordsys, ebins=ebins)
        return ExposureMap(exp, hpx)


class PSFModel(object):
    """Class that stores a pre-computed model of the PSF versus energy.  

    """

    def __init__(self, dtheta, energies, cth_bins, exp, psf, wts):
        """Create a PSFModel.

        Parameters
        ----------
        dtheta : `~numpy.ndarray`
            Array of angular offsets in degrees at which the PSF is
            evaluated.

        energies : `~numpy.ndarray`
            Array of energies in MeV at which the PSF is evaluated.

        cth_bins : `~numpy.ndarray`
            Interval in cosine of the incidence angle for which this
            model of the PSF was generated.

        exp : `~numpy.ndarray`
            Array of exposure vs. energy in cm^2 s.

        psf : `~numpy.ndarray`
            2D array of PSF values evaluated on an NxM grid of N
            offset angles and M energies (defined by ``dtheta`` and
            ``energies``).

        wts : `~numpy.ndarray`
            Array of weights vs. energy.  These are used to evaluate
            the bin-averaged PSF model.

        """

        self._dtheta = dtheta
        self._log_energies = np.log10(energies)
        self._energies = energies
        self._cth_bins = cth_bins
        self._cth = utils.edge_to_center(cth_bins)
        self._scale_fn = None
        self._exp = exp
        self._psf = psf
        self._wts = wts
        self._psf_fn = RegularGridInterpolator((self._dtheta, self._log_energies),
                                               np.log(self._psf),
                                               bounds_error=False,
                                               fill_value=None)
        self._wts_fn = RegularGridInterpolator((self._log_energies,),
                                               np.log(self._wts),
                                               bounds_error=False,
                                               fill_value=None)

    def eval(self, ebin, dtheta, scale_fn=None):
        """Evaluate the PSF at the given energy bin index.

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

    def interp_bin(self, egy_bins, dtheta, scale_fn=None):
        """Evaluate the bin-averaged PSF model over the energy bins ``egy_bins``.

        Parameters
        ----------
        egy_bins : array_like
            Energy bin edges in MeV.

        dtheta : array_like
            Array of angular separations in degrees.

        scale_fn : callable        
            Function that evaluates the PSF scaling function.
            Argument is energy in MeV.
        """

        npts = 4
        egy_bins = np.exp(utils.split_bin_edges(np.log(egy_bins), npts))
        egy = np.exp(utils.edge_to_center(np.log(egy_bins)))
        log_energies = np.log10(egy)

        vals = self.interp(egy[None, :], dtheta[:, None],
                           scale_fn=scale_fn)
        wts = np.exp(self._wts_fn((log_energies,)))
        wts = wts.reshape((1,) + wts.shape)
        vals = np.sum(
            (vals * wts).reshape((vals.shape[0], int(vals.shape[1] / npts), npts)), axis=2)
        vals /= np.sum(wts.reshape(wts.shape[0],
                                   int(wts.shape[1] / npts), npts), axis=2)
        return vals

    def containment_angle(self, energies=None, fraction=0.68, scale_fn=None):
        """Evaluate the PSF containment angle at a sequence of energies."""

        if energies is None:
            energies = self.energies

        vals = self.interp(energies[np.newaxis, :], self.dtheta[:, np.newaxis],
                           scale_fn=scale_fn)
        dtheta = np.radians(self.dtheta[:, np.newaxis] * np.ones(vals.shape))
        return self._calc_containment(dtheta, vals, fraction)

    def containment_angle_bin(self, egy_bins, fraction=0.68, scale_fn=None):
        """Evaluate the PSF containment angle averaged over energy bins."""

        vals = self.interp_bin(egy_bins, self.dtheta, scale_fn=scale_fn)
        dtheta = np.radians(self.dtheta[:, np.newaxis] * np.ones(vals.shape))
        return self._calc_containment(dtheta, vals, fraction)

    def _calc_containment(self, dtheta, vals, fraction=0.68):

        delta = dtheta[1:] - dtheta[:-1]
        ctr = 0.5 * (dtheta[1:] + dtheta[:-1])

        avg_val = 0.5 * (vals[1:, :] * np.sin(dtheta[1:, :]) +
                         vals[:-1, :] * np.sin(dtheta[:-1, :]))

        csum = delta * avg_val * 2 * np.pi
        csum = np.cumsum(csum, axis=0)
        theta = np.zeros(csum.shape[1])

        for i in range(csum.shape[1]):
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
    def create(skydir, ltc, event_class, event_types, energies, cth_bins=None,
               ndtheta=500, use_edisp=False, fn=None, nbin=64):
        """Create a PSFModel object.  This class can be used to evaluate the
        exposure-weighted PSF for a source with a given observing
        profile and energy distribution.

        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord`

        ltc : `~fermipy.irfs.LTCube`

        energies : `~numpy.ndarray`
            Grid of energies at which the PSF will be pre-computed.

        cth_bins : `~numpy.ndarray`
            Bin edges in cosine of the inclination angle.

        use_edisp : bool
            Generate the PSF model accounting for the influence of
            energy dispersion.

        fn : `~fermipy.spectrum.SpectralFunction`
            Model for the spectral energy distribution of the source.

        """

        if isinstance(event_types, int):
            event_types = bitmask_to_bits(event_types)

        if fn is None:
            fn = spectrum.PowerLaw([1E-13, -2.0])

        dtheta = np.logspace(-4, 1.75, ndtheta)
        dtheta = np.insert(dtheta, 0, [0])
        log_energies = np.log10(energies)
        egy_bins = 10**utils.center_to_edge(log_energies)

        if cth_bins is None:
            cth_bins = np.array([0.2, 1.0])

        if use_edisp:
            psf = create_wtd_psf(skydir, ltc, event_class, event_types,
                                 dtheta, egy_bins, cth_bins, fn, nbin=nbin)
            wts = calc_counts_edisp(skydir, ltc, event_class, event_types,
                                    egy_bins, cth_bins, fn, nbin=nbin)
        else:
            psf = create_avg_psf(skydir, ltc, event_class, event_types,
                                 dtheta, energies, cth_bins)
            wts = calc_counts(skydir, ltc, event_class, event_types,
                              egy_bins, cth_bins, fn)

        exp = calc_exp(skydir, ltc, event_class, event_types,
                       energies, cth_bins)

        return PSFModel(dtheta, energies, cth_bins, np.squeeze(exp), np.squeeze(psf),
                        np.squeeze(wts))


def create_irf(event_class, event_type):
    if isinstance(event_type, int):
        event_type = evtype_string[event_type]

    irf_factory = pyIrfLoader.IrfsFactory.instance()
    irfname = '%s::%s' % (event_class, event_type)
    irf = irf_factory.create(irfname)
    return irf


def create_psf(event_class, event_type, dtheta, egy, cth):
    """Create an array of PSF response values versus energy and
    inclination angle.

    Parameters
    ----------
    egy : `~numpy.ndarray`
        Energy in MeV.

    cth : `~numpy.ndarray`
        Cosine of the incidence angle.

    """
    irf = create_irf(event_class, event_type)
    theta = np.degrees(np.arccos(cth))
    m = np.zeros((len(dtheta), len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):
            m[:, i, j] = irf.psf().value(dtheta, x, y, 0.0)

    return m


def create_edisp(event_class, event_type, erec, egy, cth):
    """Create an array of energy response values versus energy and
    inclination angle.

    Parameters
    ----------
    egy : `~numpy.ndarray`
        Energy in MeV.

    cth : `~numpy.ndarray`
        Cosine of the incidence angle.

    """
    irf = create_irf(event_class, event_type)
    theta = np.degrees(np.arccos(cth))
    v = np.zeros((len(erec), len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):

            m = (erec / x < 3.0) & (erec / x > 0.333)
            v[m, i, j] = irf.edisp().value(erec[m], x, y, 0.0)

    return v


def create_aeff(event_class, event_type, egy, cth):
    """Create an array of effective areas versus energy and incidence
    angle.  Binning in energy and incidence angle is controlled with
    the egy and cth input parameters.

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
            m[i, j] = irf.aeff().value(x, y, 0.0)

    return m


def calc_exp(skydir, ltc, event_class, event_types,
             egy, cth_bins, npts=None):
    """Calculate the exposure on a 2D grid of energy and incidence angle.

    Parameters
    ----------
    npts : int    
        Number of points by which to sample the response in each
        incidence angle bin.  If None then npts will be automatically
        set such that incidence angle is sampled on intervals of <
        0.05 in Cos(Theta).

    Returns
    -------
    exp : `~numpy.ndarray`
        2D Array of exposures vs. energy and incidence angle.

    """

    if npts is None:
        npts = int(np.ceil(np.max(cth_bins[1:] - cth_bins[:-1]) / 0.05))

    exp = np.zeros((len(egy), len(cth_bins) - 1))
    cth_bins = utils.split_bin_edges(cth_bins, npts)
    cth = edge_to_center(cth_bins)
    ltw = ltc.get_skydir_lthist(skydir, cth_bins).reshape(-1, npts)
    for et in event_types:
        aeff = create_aeff(event_class, et, egy, cth)
        aeff = aeff.reshape(exp.shape + (npts,))
        exp += np.sum(aeff * ltw[np.newaxis, :, :], axis=-1)

    return exp


def create_avg_rsp(rsp_fn, skydir, ltc, event_class, event_types, x,
                   egy, cth_bins, npts=None):

    if npts is None:
        npts = int(np.ceil(np.max(cth_bins[1:] - cth_bins[:-1]) / 0.05))

    wrsp = np.zeros((len(x), len(egy), len(cth_bins) - 1))
    exps = np.zeros((len(egy), len(cth_bins) - 1))

    cth_bins = utils.split_bin_edges(cth_bins, npts)
    cth = edge_to_center(cth_bins)
    ltw = ltc.get_skydir_lthist(skydir, cth_bins)
    ltw = ltw.reshape(-1, npts)

    for et in event_types:
        rsp = rsp_fn(event_class, et, x, egy, cth)
        aeff = create_aeff(event_class, et, egy, cth)
        rsp = rsp.reshape(wrsp.shape + (npts,))
        aeff = aeff.reshape(exps.shape + (npts,))
        wrsp += np.sum(rsp * aeff[np.newaxis, :, :, :] *
                       ltw[np.newaxis, np.newaxis, :, :], axis=-1)
        exps += np.sum(aeff * ltw[np.newaxis, :, :], axis=-1)

    wrsp /= exps[np.newaxis, :, :]
    return wrsp


def create_avg_psf(skydir, ltc, event_class, event_types, dtheta,
                   egy, cth_bins, npts=None):
    """Generate model for exposure-weighted PSF averaged over incidence
    angle.

    Parameters
    ----------
    egy : `~numpy.ndarray`
        Energies in MeV.

    cth_bins : `~numpy.ndarray`
        Bin edges in cosine of the incidence angle.
    """

    return create_avg_rsp(create_psf, skydir, ltc,
                          event_class, event_types,
                          dtheta, egy,  cth_bins, npts)


def create_avg_edisp(skydir, ltc, event_class, event_types, erec,
                     egy, cth_bins, npts=None):
    """Generate model for exposure-weighted DRM averaged over incidence
    angle.

    Parameters
    ----------
    egy : `~numpy.ndarray`
        True energies in MeV.

    cth_bins : `~numpy.ndarray`
        Bin edges in cosine of the incidence angle.
    """
    return create_avg_rsp(create_edisp, skydir, ltc,
                          event_class, event_types,
                          erec, egy,  cth_bins, npts)


def create_wtd_psf(skydir, ltc, event_class, event_types, dtheta,
                   egy_bins, cth_bins, fn, nbin=64, npts=1):
    """Create an exposure- and dispersion-weighted PSF model for a source
    with spectral parameterization ``fn``.  The calculation performed
    by this method accounts for the influence of energy dispersion on
    the PSF.

    Parameters
    ----------
    dtheta : `~numpy.ndarray`

    egy_bins : `~numpy.ndarray`
        Bin edges in observed energy.

    cth_bins : `~numpy.ndarray`
        Bin edges in cosine of the true incidence angle.

    nbin : int
        Number of bins per decade in true energy.

    npts : int
        Number of points by which to oversample each energy bin.

    """
    #npts = int(np.ceil(32. / bins_per_dec(egy_bins)))
    egy_bins = np.exp(utils.split_bin_edges(np.log(egy_bins), npts))
    etrue_bins = 10**np.linspace(1.0, 6.5, nbin * 5.5 + 1)
    etrue = 10**utils.edge_to_center(np.log10(etrue_bins))

    psf = create_avg_psf(skydir, ltc, event_class, event_types, dtheta,
                         etrue, cth_bins)
    drm = calc_drm(skydir, ltc, event_class, event_types,
                   egy_bins, cth_bins, nbin=nbin)
    cnts = calc_counts(skydir, ltc, event_class, event_types,
                       etrue_bins, cth_bins, fn)

    wts = drm * cnts[None, :, :]
    wts_norm = np.sum(wts, axis=1)
    wts_norm[wts_norm == 0] = 1.0
    wts = wts / wts_norm[:, None, :]
    wpsf = np.sum(wts[None, :, :, :] * psf[:, None, :, :], axis=2)
    wts = np.sum(wts[None, :, :, :], axis=2)

    if npts > 1:
        shape = (wpsf.shape[0], int(wpsf.shape[1] / npts), npts, wpsf.shape[2])
        wpsf = np.sum((wpsf * wts).reshape(shape), axis=2)
        shape = (wts.shape[0], int(wts.shape[1] / npts), npts, wts.shape[2])
        wpsf = wpsf / np.sum(wts.reshape(shape), axis=2)

    return wpsf


def calc_drm(skydir, ltc, event_class, event_types,
             egy_bins, cth_bins, nbin=64):
    """Calculate the detector response matrix."""
    npts = int(np.ceil(128. / bins_per_dec(egy_bins)))
    egy_bins = np.exp(utils.split_bin_edges(np.log(egy_bins), npts))

    etrue_bins = 10**np.linspace(1.0, 6.5, nbin * 5.5 + 1)
    egy = 10**utils.edge_to_center(np.log10(egy_bins))
    egy_width = utils.edge_to_width(egy_bins)
    etrue = 10**utils.edge_to_center(np.log10(etrue_bins))
    edisp = create_avg_edisp(skydir, ltc, event_class, event_types,
                             egy, etrue, cth_bins)
    edisp = edisp * egy_width[:, None, None]
    edisp = sum_bins(edisp, 0, npts)
    return edisp


def calc_counts(skydir, ltc, event_class, event_types,
                egy_bins, cth_bins, fn, npts=1):
    """Calculate the expected counts vs. true energy and incidence angle
    for a source with spectral parameterization ``fn``.

    Parameters
    ----------
    skydir : `~astropy.coordinate.SkyCoord`

    ltc : `~fermipy.irfs.LTCube`

    egy_bins : `~numpy.ndarray`
        Bin edges in observed energy in MeV.

    cth_bins : `~numpy.ndarray`
        Bin edges in cosine of the true incidence angle.

    npts : int
        Number of points by which to oversample each energy bin.
    """
    #npts = int(np.ceil(32. / bins_per_dec(egy_bins)))
    egy_bins = np.exp(utils.split_bin_edges(np.log(egy_bins), npts))
    exp = calc_exp(skydir, ltc, event_class, event_types,
                   egy_bins, cth_bins)
    dnde = fn.dnde(egy_bins)
    cnts = loglog_quad(egy_bins, exp * dnde[:, None], 0)
    cnts = sum_bins(cnts, 0, npts)
    return cnts


def calc_counts_edisp(skydir, ltc, event_class, event_types,
                      egy_bins, cth_bins, fn, nbin=64, npts=1):
    """Calculate the expected counts vs. observed energy and true
    incidence angle for a source with spectral parameterization ``fn``.

    Parameters
    ----------
    skydir : `~astropy.coordinate.SkyCoord`

    ltc : `~fermipy.irfs.LTCube`

    egy_bins : `~numpy.ndarray`
        Bin edges in observed energy in MeV.

    cth_bins : `~numpy.ndarray`
        Bin edges in cosine of the true incidence angle.

    npts : int
        Number of points by which to oversample each energy bin.

    """
    #npts = int(np.ceil(32. / bins_per_dec(egy_bins)))

    # Split energy bins
    egy_bins = np.exp(utils.split_bin_edges(np.log(egy_bins), npts))
    etrue_bins = 10**np.linspace(1.0, 6.5, nbin * 5.5 + 1)
    drm = calc_drm(skydir, ltc, event_class, event_types,
                   egy_bins, cth_bins, nbin=nbin)
    cnts_etrue = calc_counts(skydir, ltc, event_class, event_types,
                             etrue_bins, cth_bins, fn)

    cnts = np.sum(cnts_etrue[None, :, :] * drm[:, :, :], axis=1)
    cnts = sum_bins(cnts, 0, npts)
    return cnts


def calc_wtd_exp(skydir, ltc, event_class, event_types,
                 egy_bins, cth_bins, fn):
    """Calculate the effective exposure.

    Parameters
    ----------
    skydir : `~astropy.coordinates.SkyCoord`

    ltc : `~fermipy.irfs.LTCube`

    """
    cnts = calc_counts_edisp(skydir, ltc, event_class, event_types,
                             egy_bins, cth_bins, fn)
    flux = fn.flux(egy_bins[:-1], egy_bins[1:])
    return cnts / flux[:, None]


def plot_hpxmap(hpxmap, **kwargs):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm, Normalize, LogNorm

    zidx = kwargs.pop('zidx', None)

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

    if zidx is not None:
        data = hpxmap.data[zidx]
    elif hpxmap.data.ndim == 2:
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
