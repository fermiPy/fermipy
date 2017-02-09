# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
from fermipy.tests.utils import requires_dependency
from fermipy import spectrum

from fermipy.scripts.flux_sensitivity import SensitivityCalc
from fermipy.ltcube import LTCube
from fermipy.skymap import Map

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')


def test_calc_diff_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')
    ebins = 10**np.linspace(1.5, 5.5, 8*4+1)

    gdiff = Map.create_from_fits(os.path.join(os.path.expandvars('$FERMI_DIFFUSE_DIR'),
                                              'gll_iem_v06.fits'))
    iso = np.loadtxt(os.path.expandvars('$FERMIPY_ROOT/data/iso_P8R2_SOURCE_V6_v06.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R2_SOURCE_V6', [['FRONT','BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.diff_flux_threshold(c, fn, 25.0, 3.0)
    assert_allclose(o['eflux'],
                    np.array([  1.83523931e-06,   1.23992086e-06,   8.92862763e-07,
                                6.79246072e-07,   5.39633059e-07,   4.61561248e-07,
                                3.94428366e-07,   3.42830114e-07,   2.93784668e-07,
                                2.55375514e-07,   2.18164833e-07,   1.90617174e-07,
                                1.63641518e-07,   1.45586177e-07,   1.31183937e-07,
                                1.22851233e-07,   1.18761865e-07,   1.19363597e-07,
                                1.22792814e-07,   1.32543078e-07,   1.45680389e-07,
                                1.62884742e-07,   1.84840136e-07,   2.09536197e-07,
                                2.44267208e-07,   2.89379741e-07,   3.49333271e-07,
                                4.49697981e-07,   5.99889350e-07,   7.98921204e-07,
                                1.06255031e-06,   1.41224660e-06]), rtol=1E-3)


def test_calc_int_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')
    ebins = 10**np.linspace(2.0, 5.0, 8*3+1)

    gdiff = Map.create_from_fits(os.path.join(os.path.expandvars('$FERMI_DIFFUSE_DIR'),
                                              'gll_iem_v06.fits'))
    iso = np.loadtxt(os.path.expandvars('$FERMIPY_ROOT/data/iso_P8R2_SOURCE_V6_v06.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R2_SOURCE_V6', [['FRONT','BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.int_flux_threshold(c, fn, 25.0, 3.0)

    assert_allclose(o['eflux'], 6.185618604617161e-07, rtol=1E-3)
    assert_allclose(o['flux'], 8.9456454903872717e-10, rtol=1E-3)
    assert_allclose(o['npred'], 294.95620464723726, rtol=1E-3)
    assert_allclose(o['dnde'], 8.9546000904777507e-15, rtol=1E-3)
    assert_allclose(o['e2dnde'], 8.9546000904777526e-08, rtol=1E-3)

    assert_allclose(o['bins']['flux'],
                    np.array([  2.23959734e-10,   1.67946107e-10,   1.25941813e-10,
                                9.44430366e-11,   7.08222862e-11,   5.31092223e-11,
                                3.98262983e-11,   2.98655105e-11,   2.23959734e-11,
                                1.67946107e-11,   1.25941813e-11,   9.44430366e-12,
                                7.08222862e-12,   5.31092223e-12,   3.98262983e-12,
                                2.98655105e-12,   2.23959734e-12,   1.67946107e-12,
                                1.25941813e-12,   9.44430366e-13,   7.08222862e-13,
                                5.31092223e-13,   3.98262983e-13,   2.98655105e-13]),
                    rtol=1E-3)
