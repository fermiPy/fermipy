# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
from fermipy.tests.utils import requires_dependency
from fermipy import spectrum

try:
    from fermipy import irfs
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')

def test_psfmodel():

    ltc = irfs.LTCube.create_empty(239557417.0, 428902995.0, 1.0)
    log_energies = np.linspace(2.0, 6.0, 17)
    c = SkyCoord(10.0, 10.0, unit='deg')
    psf = irfs.PSFModel(c, ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'], log_energies,
                        ndtheta=400, ncth=20)

    dtheta = np.array([0.0, 0.5, 1.0, 2.0])

    assert_allclose(psf.eval(0, dtheta),
                    np.array([107.174, 99.559, 81.714, 46.279]), rtol=1E-3)
    assert_allclose(psf.eval(0, dtheta),
                    psf.interp(10**log_energies[0], dtheta), rtol=1E-3)
    assert_allclose(psf.containment_angle(10**log_energies),
                    np.array([5.25865286,  3.39134629,  2.17670848,  1.35427338,  0.83850761,
                              0.52326098,  0.3359308,  0.22469995,  0.16011368,  0.12431046,
                              0.10854828,  0.10223939,  0.10147908,  0.101332,  0.097641,
                              0.09068469,  0.08329654]))


def test_ltcube():

    ltc = irfs.LTCube.create_empty(239557417.0, 428902995.0, 1.0)
    c = SkyCoord(10.0, 10.0, unit='deg')

    cth_edges = np.linspace(0, 1.0, 11)
    lthist0 = ltc.get_skydir_lthist(c, cth_edges)
    lthist1 = ltc.get_skydir_lthist(c, ltc.costh_edges)
    assert_allclose(np.sum(lthist0), np.sum(lthist1))
