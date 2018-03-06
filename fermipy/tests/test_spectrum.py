# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
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

    fn0 = spectrum.DMFitFunction(params, chan='bb')
    fn1 = spectrum.DMFitFunction(params, chan='tautau')

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
