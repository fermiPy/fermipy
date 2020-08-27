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

from fermipy.tests.utils import requires_st_version
try:
    from fermipy.scripts import flux_sensitivity
    from fermipy.sensitivity import SensitivityCalc
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't up to date
pytestmark = requires_st_version('01-03-00')

galdiff_path = os.path.join(os.path.expandvars('$FERMI_DIFFUSE_DIR'),
                            'gll_iem_v07.fits')


@requires_file(galdiff_path)
def test_calc_diff_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg', frame='galactic')
    ebins = 10**np.linspace(2.0, 5.0, 8 * 3 + 1)

    gdiff = Map.create_from_fits(galdiff_path)
    iso = np.loadtxt(os.path.expandvars('$FERMI_DIFFUSE_DIR/iso_P8R3_SOURCE_V3_v1.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R3_SOURCE_V3', [['FRONT', 'BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.diff_flux_threshold(c, fn, 25.0, 3.0)
    
    assert_allclose(o['eflux'],
                    np.array([9.93077714e-07, 8.59371115e-07, 7.44896026e-07, 6.61397166e-07,
                                  5.67842613e-07, 4.93802411e-07, 4.22501374e-07, 3.69299411e-07,
                                  3.16648674e-07, 2.81050363e-07, 2.52941925e-07, 2.35549281e-07,
                                  2.27269823e-07, 2.33109741e-07, 2.28212061e-07, 2.22473996e-07,
                                  2.26623213e-07, 2.46539227e-07, 2.61423960e-07, 3.04766386e-07,
                                  3.35688695e-07, 4.09265051e-07, 4.79776973e-07, 5.93001330e-07]),
                    rtol=3E-3)

    assert_allclose(o['flux'],
                    np.array([8.62941353e-09, 5.59988099e-09, 3.63993562e-09, 2.42359683e-09,
                                  1.56036437e-09, 1.01753944e-09, 6.52869186e-10, 4.27933870e-10,
                                  2.75153929e-10, 1.83139573e-10, 1.23600112e-10, 8.63137188e-11,
                                  6.24510606e-11, 4.80350743e-11, 3.52644113e-11, 2.57796669e-11,
                                  1.96925718e-11, 1.60651237e-11, 1.27744860e-11, 1.11677353e-11,
                                  9.22432852e-12, 8.43340011e-12, 7.41374161e-12, 6.87153420e-12]),
                    rtol=3E-3)


@requires_file(galdiff_path)
def test_calc_int_flux_sensitivity():

    ltc = LTCube.create_from_obs_time(3.1536E8)
    c = SkyCoord(10.0, 10.0, unit='deg', frame='galactic')
    ebins = 10**np.linspace(2.0, 5.0, 8 * 3 + 1)

    gdiff = Map.create_from_fits(galdiff_path)
    iso = np.loadtxt(os.path.expandvars('$FERMI_DIFFUSE_DIR/iso_P8R3_SOURCE_V3_v1.txt'),
                     unpack=True)
    scalc = SensitivityCalc(gdiff, iso, ltc, ebins,
                            'P8R3_SOURCE_V3', [['FRONT', 'BACK']])

    fn = spectrum.PowerLaw([1E-13, -2.0], scale=1E3)
    o = scalc.int_flux_threshold(c, fn, 25.0, 3.0)

    assert_allclose(o['eflux'], 1.15296479181e-06, rtol=3E-3)
    assert_allclose(o['flux'], 1.66741840222e-09, rtol=3E-3)
    assert_allclose(o['npred'], 549.71066228, rtol=3E-3)
    assert_allclose(o['dnde'], 1.66908748971e-14, rtol=3E-3)
    assert_allclose(o['e2dnde'], 1.66908748971e-07, rtol=3E-3)

    assert_allclose(o['bins']['flux'],
                    np.array([4.180875e-10, 3.135214e-10, 2.351078e-10, 1.763060e-10,
                              1.322108e-10, 9.914417e-11, 7.434764e-11, 5.575286e-11,
                              4.180876e-11, 3.135216e-11, 2.351078e-11, 1.763060e-11,
                              1.322108e-11, 9.914417e-12, 7.434764e-12, 5.575286e-12,
                              4.180875e-12, 3.135214e-12, 2.351078e-12, 1.763060e-12,
                              1.322108e-12, 9.914417e-13, 7.434764e-13, 5.575286e-13]), rtol=1E-3)



@requires_file(galdiff_path)
def test_flux_sensitivity_script(tmpdir):

    p = tmpdir.mkdir("sub").join('output.fits')
    outpath = str(p)
    outpath = 'test.fits'
    isodiff = os.path.expandvars(
        '$FERMI_DIFFUSE_DIR/iso_P8R3_SOURCE_V3_v1.txt')
    flux_sensitivity.run_flux_sensitivity(ltcube=None, galdiff=galdiff_path,
                                          isodiff=isodiff,
                                          glon=10.0, glat=10.0,
                                          emin=10**2.0, emax=10**5.0, nbin=24, output=outpath,
                                          obs_time_yr=10.0, ts_thresh=25.0, min_counts=3.0)

    tab = Table.read(outpath, 'DIFF_FLUX')

    assert_allclose(tab['flux'],
                    np.array([8.62850665e-09, 5.59920518e-09, 3.63950511e-09, 2.42326614e-09,
                                  1.56015165e-09, 1.01737776e-09, 6.52779820e-10, 4.27876477e-10,
                                  2.75077234e-10, 1.83076832e-10, 1.23542694e-10, 8.62621145e-11,
                                  6.24062445e-11, 4.79971674e-11, 3.52281635e-11, 2.57478300e-11,
                                  1.96615495e-11, 1.60323578e-11, 1.27440221e-11, 1.11360816e-11,
                                  9.19157532e-12, 8.39957289e-12, 7.37949945e-12, 6.83355191e-12]),
                    rtol=3E-3)
