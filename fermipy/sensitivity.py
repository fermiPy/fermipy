# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import pyLikelihood as pyLike

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from astropy.io import fits

from fermipy import utils
from fermipy import spectrum
from fermipy import irfs
from fermipy import skymap
from fermipy.ltcube import LTCube


class SensitivityCalc(object):
    """Class for evaluating LAT source flux sensitivity.  

    Parameters
    ----------
    gdiff : `~fermipy.skymap.SkyMap`
        Galactic diffuse map cube object.

    iso : `~numpy.ndarray`
        Array of background isotropic intensity vs. energy.

    ltc : `~fermipy.ltcube.LTCube`

    ebins : `~numpy.ndarray`
        Energy bin edges in MeV used for differential sensitivity.

    event_class : str
        Name of the IRF/event class (e.g. P8R2_SOURCE_V6).

    event_types : list
        List of lists of event type strings defining the event type
        selection to be used.  Each event type list will be combined.
        A selection for a combined FRONT/BACK analysis is defined with
        [['FRONT','BACK']].  A selection for joint FRONT/BACK analysis
        is defined with [['FRONT'],['BACK']].

    """

    def __init__(self, gdiff, iso, ltc, ebins, event_class, event_types=None,
                 gdiff_fit=None, iso_fit=None, spatial_model='PointSource',
                 spatial_size=None):

        self._gdiff = gdiff
        self._gdiff_fit = gdiff_fit
        self._iso = iso
        self._iso_fit = iso_fit
        self._ltc = ltc
        self._ebins = ebins
        self._log_ebins = np.log10(ebins)
        self._ectr = np.exp(utils.edge_to_center(np.log(self._ebins)))
        self._event_class = event_class
        self._spatial_model = spatial_model
        self._spatial_size = spatial_size
        if event_types is None:
            self._event_types = [['FRONT'], ['BACK']]
        else:
            self._event_types = event_types

        self._psf = []
        self._exp = []

        ebins = 10**np.linspace(1.0, 6.0, 5 * 8 + 1)
        skydir = SkyCoord(0.0, 0.0, unit='deg')
        for et in self._event_types:
            self._psf += [irfs.PSFModel.create(skydir.icrs, self._ltc,
                                               self._event_class, et,
                                               ebins)]
            self._exp += [irfs.ExposureMap.create(self._ltc,
                                                  self._event_class, et,
                                                  ebins)]

    @property
    def ebins(self):
        return self._ebins

    @property
    def ectr(self):
        return self._ectr

    @property
    def spatial_model(self):
        return self._spatial_model

    @property
    def spatial_size(self):
        return self._spatial_size

    def compute_counts(self, skydir, fn, ebins=None):
        """Compute signal and background counts for a point source at
        position ``skydir`` with spectral parameterization ``fn``.

        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord`

        ebins : `~numpy.ndarray`

        Returns
        -------
        sig : `~numpy.ndarray`
            Signal counts array.  Dimensions are energy, angular
            separation, and event type.

        bkg : `~numpy.ndarray`
            Background counts array.  Dimensions are energy, angular
            separation, and event type.

        """

        if ebins is None:
            ebins = self.ebins
            ectr = self.ectr
        else:
            ectr = np.exp(utils.edge_to_center(np.log(ebins)))

        skydir_cel = skydir.transform_to('icrs')
        skydir_gal = skydir.transform_to('galactic')

        sig = []
        bkg = []
        bkg_fit = None
        if self._gdiff_fit is not None:
            bkg_fit = []


        for psf, exp in zip(self._psf, self._exp):

            coords0 = np.meshgrid(*[skydir_cel.ra.deg, ectr], indexing='ij')
            coords1 = np.meshgrid(*[skydir_cel.dec.deg, ectr], indexing='ij')

            # expv = exp.interpolate(skydir_cel.icrs.ra.deg,
            #                       skydir_cel.icrs.dec.deg,
            #                       ectr)

            expv = exp.interpolate(coords0[0], coords1[0], coords0[1])

            coords0 = np.meshgrid(*[skydir_gal.l.deg, ectr], indexing='ij')
            coords1 = np.meshgrid(*[skydir_gal.b.deg, ectr], indexing='ij')

            bkgv = self._gdiff.interpolate(np.ravel(coords0[0]),
                                           np.ravel(coords1[0]),
                                           np.ravel(coords0[1]))
            bkgv = bkgv.reshape(expv.shape)
            

            # bkgv = self._gdiff.interpolate(
            #    skydir_gal.l.deg, skydir_gal.b.deg, ectr)

            isov = np.exp(np.interp(np.log(ectr), np.log(self._iso[0]),
                                    np.log(self._iso[1])))
            bkgv += isov

            
            s0, b0 = irfs.compute_ps_counts(ebins, expv, psf, bkgv, fn,
                                            egy_dim=1,
                                            spatial_model=self.spatial_model,
                                            spatial_size=self.spatial_size)

            sig += [s0]
            bkg += [b0]

            if self._iso_fit is not None:
                isov_fit = np.exp(np.interp(np.log(ectr), np.log(self._iso_fit[0]),
                                            np.log(self._iso_fit[1])))
            else:
                isov_fit = isov

            if self._gdiff_fit is not None:
                bkgv_fit = self._gdiff_fit.interpolate(np.ravel(coords0[0]),
                                                       np.ravel(coords1[0]),
                                                       np.ravel(coords0[1]))
                bkgv_fit = bkgv_fit.reshape(expv.shape)
                bkgv_fit += isov_fit
                s0, b0 = irfs.compute_ps_counts(ebins, expv, psf,
                                                bkgv_fit, fn, egy_dim=1,
                                                spatial_model=self.spatial_model,
                                                spatial_size=self.spatial_size)
                bkg_fit += [b0]

        sig = np.concatenate([np.expand_dims(t, -1) for t in sig])
        bkg = np.concatenate([np.expand_dims(t, -1) for t in bkg])
        if self._gdiff_fit is not None:
            bkg_fit = np.concatenate([np.expand_dims(t, -1) for t in bkg_fit])

        return sig, bkg, bkg_fit

    def diff_flux_threshold(self, skydir, fn, ts_thresh, min_counts):
        """Compute the differential flux threshold for a point source at
        position ``skydir`` with spectral parameterization ``fn``.

        Parameters
        ----------
        skydir : `~astropy.coordinates.SkyCoord`
            Sky coordinates at which the sensitivity will be evaluated.

        fn : `~fermipy.spectrum.SpectralFunction`

        ts_thresh : float
            Threshold on the detection test statistic (TS).

        min_counts : float
            Threshold on the minimum number of counts.

        """

        sig, bkg, bkg_fit = self.compute_counts(skydir, fn)
        norms = irfs.compute_norm(sig, bkg, ts_thresh,
                                  min_counts, sum_axes=[2, 3],
                                  rebin_axes=[10, 1],
                                  bkg_fit=bkg_fit)

        npred = np.squeeze(np.apply_over_axes(np.sum, norms * sig, [2, 3]))
        norms = np.squeeze(norms)

        flux = norms * fn.flux(self.ebins[:-1], self.ebins[1:])
        eflux = norms * fn.eflux(self.ebins[:-1], self.ebins[1:])
        dnde = norms * fn.dnde(self.ectr)
        e2dnde = self.ectr**2 * dnde

        return dict(e_min=self.ebins[:-1], e_max=self.ebins[1:],
                    e_ref=self.ectr,
                    npred=npred, flux=flux, eflux=eflux,
                    dnde=dnde, e2dnde=e2dnde)

    def int_flux_threshold(self, skydir, fn, ts_thresh, min_counts):
        """Compute the integral flux threshold for a point source at
        position ``skydir`` with spectral parameterization ``fn``.

        """

        ebins = 10**np.linspace(np.log10(self.ebins[0]),
                                np.log10(self.ebins[-1]), 33)
        ectr = np.sqrt(ebins[0] * ebins[-1])

        sig, bkg, bkg_fit = self.compute_counts(skydir, fn, ebins)

        norms = irfs.compute_norm(sig, bkg, ts_thresh,
                                  min_counts, sum_axes=[1, 2, 3], bkg_fit=bkg_fit,
                                  rebin_axes=[4, 10, 1])

        npred = np.squeeze(np.apply_over_axes(np.sum, norms * sig, [1, 2, 3]))
        npred = np.array(npred, ndmin=1)
        flux = np.squeeze(norms) * fn.flux(ebins[0], ebins[-1])
        eflux = np.squeeze(norms) * fn.eflux(ebins[0], ebins[-1])
        dnde = np.squeeze(norms) * fn.dnde(ectr)
        e2dnde = ectr**2 * dnde

        o = dict(e_min=self.ebins[0], e_max=self.ebins[-1], e_ref=ectr,
                 npred=npred, flux=flux, eflux=eflux,
                 dnde=dnde, e2dnde=e2dnde)

        sig, bkg, bkg_fit = self.compute_counts(skydir, fn)

        npred = np.squeeze(np.apply_over_axes(np.sum, norms * sig,
                                              [2, 3]))
        flux = np.squeeze(np.squeeze(norms, axis=(1, 2, 3))[:, None] *
                          fn.flux(self.ebins[:-1], self.ebins[1:]))
        eflux = np.squeeze(np.squeeze(norms, axis=(1, 2, 3))[:, None] *
                           fn.eflux(self.ebins[:-1], self.ebins[1:]))
        dnde = np.squeeze(np.squeeze(norms, axis=(1, 2, 3))
                          [:, None] * fn.dnde(self.ectr))
        e2dnde = ectr**2 * dnde

        o['bins'] = dict(npred=npred,
                         flux=flux,
                         eflux=eflux,
                         dnde=dnde,
                         e2dnde=e2dnde,
                         e_min=self.ebins[:-1], e_max=self.ebins[1:],
                         e_ref=self.ectr)

        return o
