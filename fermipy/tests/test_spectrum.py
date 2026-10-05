# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
import pytest
from fermipy import spectrum


def test_powerlaw_spectrum():

    params = [1E-13, -2.3]
    fn = spectrum.PowerLaw(params, scale=2E3)


def test_logparabola_spectrum():

    params = [1E-13, -2.3, 0.5]
    fn = spectrum.LogParabola(params, scale=2E3)


def test_plexpcutoff_spectrum():

    params = [1E-13, -2.3, 1E3]
    fn = spectrum.PLExpCutoff(params, scale=2E3)


def test_dmfitfunction_spectrum():

    sigmav = 3E-26
    mass = 100.  # Mass in GeV
    params = [sigmav, mass]

    fn0 = spectrum.DMFitFunction(params, chan='bb', tablepath='legacy')
    fn1 = spectrum.DMFitFunction(params, chan='tautau', tablepath='legacy')

    loge = np.linspace(2, 4, 5)

    # Test energy scalar evaluation
    assert_allclose(fn0.dnde(1E3), 1.15754e-14, rtol=1E-3)
    assert_allclose(fn1.dnde(1E3), 2.72232e-16, rtol=1E-3)

    fn0.flux(1E3, 1E4)
    fn1.flux(1E3, 1E4)

    fn0.eflux(1E3, 1E4)
    fn1.eflux(1E3, 1E4)

    # Test energy vector evaluation
    assert_allclose(fn0.dnde(10**loge),
                    [5.39894e-14, 3.26639e-14, 1.15754e-14,
                     2.13262e-15, 1.79554e-16], rtol=1E-3)
    assert_allclose(fn1.dnde(10**loge),
                    [7.12808e-16, 3.79861e-16, 2.72232e-16,
                     1.96952e-16, 9.49478e-17], rtol=1E-3)

    fn0.flux(loge[:-1], loge[1:])
    fn1.flux(loge[:-1], loge[1:])

    fn0.eflux(loge[:-1], loge[1:])
    fn1.eflux(loge[:-1], loge[1:])

    # Test energy vector + parameter vector evaluation
    dnde0 = fn0.dnde(10**loge, params=[sigmav, [100E3, 200E3]])
    dnde1 = fn1.dnde(10**loge, params=[sigmav, [100E3, 200E3]])

    assert_allclose(dnde0[:, 0], fn0.dnde(10**loge, params=[sigmav, 100E3]))
    assert_allclose(dnde0[:, 1], fn0.dnde(10**loge, params=[sigmav, 200E3]))
    assert_allclose(dnde1[:, 0], fn1.dnde(10**loge, params=[sigmav, 100E3]))
    assert_allclose(dnde1[:, 1], fn1.dnde(10**loge, params=[sigmav, 200E3]))


def test_dmfitfunction_tables():

    assert spectrum.get_dmfit_tablepath() == \
        os.path.join('$FERMIPY_DATA_DIR', 'gammamc_dif_CosmiXs.dat')
    assert spectrum.get_dmfit_tablepath('legacy') == \
        os.path.join('$FERMIPY_DATA_DIR', 'gammamc_dif.dat')
    assert spectrum.get_dmfit_tablepath('/some/table.dat') == \
        '/some/table.dat'

    sigmav = 3E-26
    mass = 100.  # Mass in GeV
    params = [sigmav, mass]

    fn0 = spectrum.DMFitFunction(params, chan='bb')
    fn1 = spectrum.DMFitFunction(params, chan='tautau')

    loge = np.linspace(2, 4, 5)

    assert_allclose(fn0.dnde(10**loge),
                    [5.08337e-14, 3.08786e-14, 1.12428e-14,
                     2.14891e-15, 1.83371e-16], rtol=1E-3)
    assert_allclose(fn1.dnde(10**loge),
                    [7.49400e-16, 4.08160e-16, 2.80641e-16,
                     1.97219e-16, 9.62359e-17], rtol=1E-3)


def test_dmfitfunction_pylike():

    pyLike = pytest.importorskip('pyLikelihood')

    params = [3E-26, 100.]
    loge = np.linspace(2, 4, 5)
    for table in spectrum.DMFIT_TABLES:
        tablepath = os.path.expandvars(spectrum.get_dmfit_tablepath(table))
        fn_st = pyLike.DMFitFunction()
        fn_st.readFunction(tablepath)
        fn_st.setParam('sigmav', params[0])
        fn_st.setParam('mass', params[1])
        fn_st.setParam('channel0', 4)
        fn_st.setParam('norm', 1E19)
        fn = spectrum.DMFitFunction(params, chan='bb', tablepath=table)
        assert_allclose(fn.dnde(10**loge),
                        [fn_st(pyLike.dArg(10**x)) for x in loge], rtol=1E-5)
