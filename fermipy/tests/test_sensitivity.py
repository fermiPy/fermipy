# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from fermipy.tests.utils import requires_dependency, requires_file
from fermipy import spectrum
from fermipy.ltcube import LTCube
from fermipy.skymap import Map

try:
    from fermipy.scripts import flux_sensitivity
    from fermipy.sensitivity import SensitivityCalc
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')

galdiff_path = os.path.join(os.path.expandvars('$FERMI_DIFFUSE_DIR'),
                            'gll_iem_v06.fits')


@requires_file(galdiff_path)
def test_calc_diff_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg', frame='galactic')
    ebins = 10**np.linspace(2.0, 5.0, 8 * 3 + 1)

    gdiff = Map.create_from_fits(galdiff_path)
    iso = np.loadtxt(os.path.expandvars('$FERMIPY_ROOT/data/iso_P8R2_SOURCE_V6_v06.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R2_SOURCE_V6', [['FRONT', 'BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.diff_flux_threshold(c, fn, 25.0, 3.0)
    assert_allclose(o['eflux'],
                    np.array([9.46940878e-07,   8.18350327e-07,   7.08228859e-07,
                              6.25785181e-07,   5.45241744e-07,   4.80434705e-07,
                              4.16747935e-07,   3.68406513e-07,   3.16719850e-07,
                              2.79755007e-07,   2.50862769e-07,   2.31349646e-07,
                              2.16286209e-07,   2.08381335e-07,   2.02673929e-07,
                              2.05372045e-07,   2.14355673e-07,   2.29584275e-07,
                              2.53220397e-07,   2.80878974e-07,   3.19989251e-07,
                              3.74686765e-07,   4.49321103e-07,   5.48865583e-07]),
                    rtol=3E-3)

    assert_allclose(o['flux'],
                    np.array([8.22850449e-09,   5.33257909e-09,   3.46076145e-09,
                              2.29310172e-09,   1.49825986e-09,   9.89993669e-10,
                              6.43978699e-10,   4.26899204e-10,   2.75215778e-10,
                              1.82295486e-10,   1.22584132e-10,   8.47748218e-11,
                              5.94328933e-11,   4.29394879e-11,   3.13181379e-11,
                              2.37979404e-11,   1.86265760e-11,   1.49602959e-11,
                              1.23736187e-11,   1.02924147e-11,   8.79292634e-12,
                              7.72087281e-12,   6.94312304e-12,   6.36010146e-12]),
                    rtol=3E-3)


@requires_file(galdiff_path)
def test_calc_int_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg', frame='galactic')
    ebins = 10**np.linspace(2.0, 5.0, 8 * 3 + 1)

    gdiff = Map.create_from_fits(galdiff_path)
    iso = np.loadtxt(os.path.expandvars('$FERMIPY_ROOT/data/iso_P8R2_SOURCE_V6_v06.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R2_SOURCE_V6', [['FRONT', 'BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.int_flux_threshold(c, fn, 25.0, 3.0)

    assert_allclose(o['eflux'], 1.0719847971553671e-06, rtol=3E-3)
    assert_allclose(o['flux'], 1.550305083355546e-09, rtol=3E-3)
    assert_allclose(o['npred'], 511.16725330021416, rtol=3E-3)
    assert_allclose(o['dnde'], 1.5518569402958423e-14, rtol=3E-3)
    assert_allclose(o['e2dnde'], 1.5518569402958427e-07, rtol=3E-3)

    assert_allclose(o['bins']['flux'],
                    np.array([3.88128407e-10,   2.91055245e-10,   2.18260643e-10,
                              1.63672392e-10,   1.22736979e-10,   9.20397499e-11,
                              6.90200755e-11,   5.17577549e-11,   3.88128407e-11,
                              2.91055245e-11,   2.18260643e-11,   1.63672392e-11,
                              1.22736979e-11,   9.20397499e-12,   6.90200755e-12,
                              5.17577549e-12,   3.88128407e-12,   2.91055245e-12,
                              2.18260643e-12,   1.63672392e-12,   1.22736979e-12,
                              9.20397499e-13,   6.90200755e-13,   5.17577549e-13]),
                    rtol=1E-3)


@requires_file(galdiff_path)
def test_flux_sensitivity_script(tmpdir):

    p = tmpdir.mkdir("sub").join('output.fits')
    outpath = str(p)
    outpath = 'test.fits'
    isodiff = os.path.expandvars(
        '$FERMIPY_ROOT/data/iso_P8R2_SOURCE_V6_v06.txt')
    flux_sensitivity.run_flux_sensitivity(ltcube=None, galdiff=galdiff_path,
                                          isodiff=isodiff,
                                          glon=10.0, glat=10.0,
                                          emin=10**2.0, emax=10**5.0, nbin=24, output=outpath,
                                          obs_time_yr=10.0, ts_thresh=25.0, min_counts=3.0)

    tab = Table.read(outpath, 'DIFF_FLUX')
    assert_allclose(tab['flux'],
                    np.array([8.22850449e-09,   5.33257909e-09,   3.46076145e-09,
                              2.29310172e-09,   1.49825986e-09,   9.89993669e-10,
                              6.43978699e-10,   4.26899204e-10,   2.75215778e-10,
                              1.82295486e-10,   1.22584132e-10,   8.47748218e-11,
                              5.94328933e-11,   4.29394879e-11,   3.13181379e-11,
                              2.37979404e-11,   1.86265760e-11,   1.49602959e-11,
                              1.23736187e-11,   1.02924147e-11,   8.79292634e-12,
                              7.72087281e-12,   6.94312304e-12,   6.36010146e-12]),
                    rtol=3E-3)
