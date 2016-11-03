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


def test_calc_exp():

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')
    egy = 10**np.linspace(1.0, 6.0, 6)
    cth_bins = np.array([0.2,0.6,1.0])

    exp = irfs.calc_exp(c,ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'],
                        egy, cth_bins)

    assert_allclose(exp,
                    np.array([[  1.99488297e+07,   1.19873696e+08],
                              [  2.62408989e+10,   1.45160418e+11],
                              [  1.48892140e+11,   4.16821553e+11],
                              [  1.54396951e+11,   4.62333934e+11],
                              [  1.64080383e+11,   4.69147489e+11],
                              [  1.26298212e+11,   4.12446229e+11]]),
                    rtol=1E-3)


def test_create_avg_psf():

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')
    egy = 10**np.linspace(1.0, 6.0, 6)
    cth_bins = np.array([0.2,0.6,1.0])
    dtheta = np.array([0.0, 1.0])

    psf = irfs.create_avg_psf(c,ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'],
                              dtheta, egy, cth_bins)
    
    assert_allclose(psf,
                    np.array([[[  1.18130643e+00,   3.28292468e+00],
                               [  6.84688131e+01,   1.14113625e+02],
                               [  3.38398660e+03,   6.28834012e+03],
                               [  9.53575281e+04,   1.32718447e+05],
                               [  4.34116995e+05,   5.66676868e+05],
                               [  7.38865869e+05,   7.17838392e+05]],              
                              [[  1.17856092e+00,   3.26537299e+00],
                               [  5.72311084e+01,   8.61070947e+01],
                               [  1.95186605e+02,   1.49160620e+02],
                               [  6.48954807e+01,   2.29121961e+01],
                               [  1.47953386e+01,   7.85470899e+00],
                               [  5.79479621e+00,   2.80795367e+00]]]),
                    rtol=1E-3)

    
def test_create_avg_edisp():

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')
    egy = 10**np.array([3.5])
    erec = 10**np.linspace(3.0, 4.0, 6)
    cth_bins = np.array([0.2,0.6,1.0])

    edisp = irfs.create_avg_edisp(c,ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'],
                                  erec, egy, cth_bins)

    assert_allclose(edisp,
                    np.array([[[  0.00000000e+00,   0.00000000e+00]],
                              [[  1.18478609e-05,   1.19762369e-05]],
                              [[  2.01589611e-04,   1.53937204e-04]],
                              [[  9.66320297e-06,   2.22120074e-05]],
                              [[  1.35586410e-08,   7.26040089e-09]],
                              [[  0.00000000e+00,   0.00000000e+00]]]),
                    rtol=1E-3)
                    
    
def test_psfmodel():

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
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

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
    log_energies = np.linspace(2.0, 6.0, 17)
    c = SkyCoord(10.0, 10.0, unit='deg')
    psf = irfs.PSFModel.create(c, ltc, 'P8R2_SOURCE_V6', ['FRONT', 'BACK'], 10**log_energies,
                               ndtheta=100, use_edisp=True, nbin=64)

    dtheta = np.array([0.0, 0.5, 1.0, 2.0])

    assert_allclose(psf.eval(0, dtheta),
                    np.array([ 118.75080503,  107.16226208,   84.17906815,   45.74811465]),
                    rtol=1E-3)
    assert_allclose(psf.eval(8, dtheta),
                    np.array([ 1.22250413e+05,   2.50655759e+02,   3.38000338e+01, 1.61790809e+00]),
                    rtol=1E-3)
    assert_allclose(psf.eval(0, dtheta),
                    psf.interp(10**log_energies[0], dtheta), rtol=1E-3)
    assert_allclose(psf.eval(8, dtheta),
                    psf.interp(10**log_energies[8], dtheta), rtol=1E-3)
    assert_allclose(psf.containment_angle(10**log_energies),
                    np.array([ 5.29456712,  3.38903389,  2.17160528,  1.35119099,  0.83606193,
                               0.52268297,  0.33887066,  0.22721536,  0.16208684,  0.12568082,
                               0.10895818,  0.10243248,  0.10157182,  0.10136436,  0.09777351,
                               0.0906484 ,  0.08328053]), rtol=1E-3)


def test_ltcube():

    ltc = irfs.LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg')

    cth_edges = np.linspace(0, 1.0, 11)
    lthist0 = ltc.get_skydir_lthist(c, cth_edges)
    lthist1 = ltc.get_skydir_lthist(c, ltc.costh_edges)
    assert_allclose(np.sum(lthist0), np.sum(lthist1))
