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
    psf = irfs.PSFModel.create(c, ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'], 10**log_energies,
                               ndtheta=400)

    dtheta = np.array([0.0, 0.5, 1.0, 2.0])

    assert_allclose(psf.eval(0, dtheta),
                    np.array([ 107.12557515,   99.51633488,   81.6831002 ,   46.26852089]),
                    rtol=1E-3)
    assert_allclose(psf.eval(0, dtheta),
                    psf.interp(10**log_energies[0], dtheta), rtol=1E-3)
    assert_allclose(psf.containment_angle(10**log_energies),
                    np.array([ 5.26027904,  3.39225156,  2.17733773,  1.3545119 ,  0.8388698 ,
                               0.52372673,  0.33596259,  0.22476962,  0.16023516,  0.12436676,
                               0.10858581,  0.1022676 ,  0.10153258,  0.10137515,  0.09766877,
                               0.09070837,  0.08331364]), rtol=1E-3)
    

def test_psfmodel_edisp():

    ltc = irfs.LTCube.create_empty(239557417.0, 428902995.0, 1.0)
    log_energies = np.linspace(2.0, 6.0, 17)
    c = SkyCoord(10.0, 10.0, unit='deg')
    psf = irfs.PSFModel.create(c, ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'], 10**log_energies,
                               ndtheta=100, use_edisp=True, nbin=64)

    dtheta = np.array([0.0, 0.5, 1.0, 2.0])

    assert_allclose(psf.eval(0, dtheta),
                    np.array([ 114.79301341,  104.53823003,   83.30678261,   45.99297095]),
                    rtol=1E-3)
    assert_allclose(psf.eval(0, dtheta),
                    psf.interp(10**log_energies[0], dtheta), rtol=1E-3)
    assert_allclose(psf.containment_angle(10**log_energies),
                    np.array([ 5.27388871,  3.35209634,  2.13896084,  1.32429852,  0.81903267,
                               0.51230932,  0.33203214,  0.22337945,  0.15933116,  0.12412639,
                               0.10849143,  0.10226171,  0.10160706,  0.10133984,  0.0975222 ,
                               0.09030107,  0.08276633]), rtol=1E-3)


def test_ltcube():

    ltc = irfs.LTCube.create_empty(239557417.0, 428902995.0, 1.0)
    c = SkyCoord(10.0, 10.0, unit='deg')

    cth_edges = np.linspace(0, 1.0, 11)
    lthist0 = ltc.get_skydir_lthist(c, cth_edges)
    lthist1 = ltc.get_skydir_lthist(c, ltc.costh_edges)
    assert_allclose(np.sum(lthist0), np.sum(lthist1))
