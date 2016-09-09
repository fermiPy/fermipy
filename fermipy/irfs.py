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

from fermipy.utils import edge_to_center
from fermipy.utils import edge_to_width

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


class PSFModel(object):

    def __init__(self, skydir, ltc, event_class, event_types,
                 log_energies, cth_min=0.2):

        if isinstance(event_types, int):
            event_types = bitmask_to_bits(event_types)

        self._dtheta = np.logspace(-4, 1.75, 1000)
        self._dtheta = np.insert(self._dtheta, 0, [0])
        self._log_energies = log_energies
        self._energies = 10**log_energies
        self._scale_fn = None

        self._exp = np.zeros(len(log_energies))
        self._psf = self.create_average_psf(skydir, ltc, event_class, event_types,
                                            self._dtheta, log_energies, cth_min)

        self._psf_fn = RegularGridInterpolator((self._dtheta, log_energies),
                                               np.log(self._psf),
                                               bounds_error=False,
                                               fill_value=None)

        cth_edge = np.linspace(cth_min, 1.0, 41)
        cth = edge_to_center(cth_edge)
        ltw = ltc.get_src_lthist(skydir, cth_edge)
        for et in event_types:
            aeff = create_exposure(event_class, et, log_energies, cth)
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

        if scale_fn is None:
            dtheta = dtheta[:, np.newaxis] * \
                np.ones((len(dtheta), len(energies)))
            scale_factor = np.ones(len(energies))
        else:
            dtheta = dtheta[:, np.newaxis] / scale_fn(energies)
            scale_factor = 1. / scale_fn(energies)**2

        vals = np.exp(self._psf_fn((dtheta, log_energies[None, :])))
        return vals * scale_factor

    def containment_angle(self, energies=None, fraction=0.68, scale_fn=None):
        """Evaluate the PSF containment angle at a sequence of energies."""

        if scale_fn is None and self.scale_fn:
            scale_fn = self.scale_fn

        if energies is None:
            energies = self.energies

        vals = self.interp(energies, self.dtheta, scale_fn=scale_fn)
        dtheta = np.radians(self.dtheta[:, np.newaxis] * np.ones(vals.shape))

        delta = dtheta[1:] - dtheta[:-1]
        ctr = 0.5 * (dtheta[1:] + dtheta[:-1])

        avg_val = 0.5 * (vals[1:, :] * np.sin(dtheta[1:, :]) +
                         vals[:-1, :] * np.sin(dtheta[:-1, :]))

        csum = delta * avg_val * 2 * np.pi
        csum = np.cumsum(csum, axis=0)
        theta = np.zeros(len(energies))

        for i in range(len(theta)):
            theta[i] = np.degrees(
                np.interp(fraction, csum[:, i], dtheta[1:, i]))

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
                           cth_min=0.2):

        if isinstance(event_types, int):
            event_types = bitmask_to_bits(event_types)

        cth_edge = np.linspace(cth_min, 1.0, 41)
        cth = edge_to_center(cth_edge)

        wpsf = np.zeros((len(dtheta), len(egy)))
        exps = np.zeros(len(egy))

        ltw = ltc.get_src_lthist(skydir, cth_edge)

        for et in event_types:
            psf = create_psf(event_class, et, dtheta, egy, cth)
            aeff = create_exposure(event_class, et, egy, cth)

            wpsf += np.sum(psf * aeff[np.newaxis, :, :] *
                           ltw[np.newaxis, np.newaxis, :], axis=2)
            exps += np.sum(aeff * ltw[np.newaxis, :], axis=1)

        wpsf /= exps[np.newaxis, :]

        return wpsf


def create_psf(event_class, event_type, dtheta, egy, cth):
    """This function creates a sequence of DRMs versus incidence
    angle.  The output is returned as a single 3-dimensional numpy
    array with dimensions of etrue,erec, and incidence angle."""

    if isinstance(event_type, int):
        event_type = evtype_string[event_type]

    irfname = '%s::%s' % (event_class, event_type)
    irf_factory = pyIrfLoader.IrfsFactory.instance()
    irf = irf_factory.create(irfname)

    theta = np.degrees(np.arccos(cth))
    m = np.zeros((len(dtheta), len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):
            m[:, i, j] = irf.psf().value(dtheta, 10**x, y, 0.0)

    return m


def create_exposure(event_class, event_type, egy, cth):
    """This function creates a map of exposure versus energy and
    incidence angle.  Binning in energy and incidence angle is
    controlled with the ebin_edge and cth_edge input parameters."""

    if isinstance(event_type, int):
        event_type = evtype_string[event_type]

    irf_factory = pyIrfLoader.IrfsFactory.instance()
    irf = irf_factory.create('%s::%s' % (event_class, event_type))

    irf.aeff().setPhiDependence(False)

    theta = np.degrees(np.arccos(cth))

    # Exposure Matrix
    # Dimensions are Etrue and incidence angle
    m = np.zeros((len(egy), len(cth)))

    for i, x in enumerate(egy):
        for j, y in enumerate(theta):
            m[i, j] = irf.aeff().value(10**x, y, 0.0)

    return m


class LTCube(object):

    def __init__(self, ltfile=None):

        self._ltmap = None

        if ltfile is None:
            return
        elif isinstance(ltfile, list):
            for f in ltfile:
                self.load_ltfile(f)
        elif not re.search('\.txt?', ltfile) is None:
            files = np.loadtxt(ltfile, unpack=True, dtype='str')
            for f in files:
                self.load_ltfile(f)
        else:
            self.load_ltfile(ltfile)

    @staticmethod
    def create(ltfile):

        ltc = LTCube()
        if not isinstance(ltfile, list):
            ltfile = glob.glob(ltfile)
        for f in ltfile:
            ltc.load_ltfile(f)

        return ltc

    def load_ltfile(self, ltfile):

        hdulist = fits.open(ltfile)

        if self._ltmap is None:
            self._ltmap = hdulist[1].data.field(0)
            self._tstart = hdulist[0].header['TSTART']
            self._tstop = hdulist[0].header['TSTOP']
        else:
            self._ltmap += hdulist[1].data.field(0)
            self._tstart = min(self._tstart, hdulist[0].header['TSTART'])
            self._tstop = max(self._tstop, hdulist[0].header['TSTOP'])

        cth_edges = np.array(hdulist[3].data.field(0))
        cth_edges = np.concatenate(([1], cth_edges))
        self._cth_edges = cth_edges[::-1]
        self._cth_center = edge_to_center(self._cth_edges)
        self._cth_width = edge_to_width(self._cth_edges)

#        self._domega = (self._cth_axis.edges[1:]-
#                        self._cth_axis.edges[:-1])*2*np.pi

    def get_src_lthist(self, skydir, cth_edges):

        ra = skydir.ra.deg
        dec = skydir.dec.deg

        edges = np.linspace(
            cth_edges[0], cth_edges[-1], (len(cth_edges) - 1) * 4 + 1)
        center = edge_to_center(edges)
        width = edge_to_width(edges)

        ipix = hp.ang2pix(64, np.pi / 2. - np.radians(dec),
                          np.radians(ra), nest=True)

        lt = np.interp(center, self._cth_center,
                       self._ltmap[ipix, ::-1] / self._cth_width) * width
        lt = np.sum(lt.reshape(-1, 4), axis=1)
        return lt
