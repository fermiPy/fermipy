# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from fermipy import castro
from fermipy.spectrum import DMFitFunction


@pytest.fixture(scope='module')
def sedfile(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    outfile = 'sed.fits'
    url = 'https://raw.githubusercontent.com/fermiPy/fermipy-extra/master/data/sed.fits'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


def test_castro_test_spectra_sed(sedfile):
    c = castro.CastroData.create_from_sedfile(sedfile)
    test_dict = c.test_spectra()

    assert_allclose(test_dict['PowerLaw']['TS'][0], 28.20377, atol=0.01)
    assert_allclose(test_dict['LogParabola']['TS'][0], 28.2466, atol=0.01)
    assert_allclose(test_dict['PLExpCutoff']['TS'][0], 28.32140, atol=0.01)

    assert_allclose(test_dict['PowerLaw']['Result'],
                    np.array([1.12719216e-12, -2.60937952e+00]), rtol=1E-4)
    assert_allclose(test_dict['LogParabola']['Result'],
                    np.array([1.01468889e-12, -2.43974568e+00, 6.66682177e-02]),
                    rtol=1E-4)
    assert_allclose(test_dict['PLExpCutoff']['Result'],
                    np.array([1.04931290e-12, -2.50472567e+00, 6.35096327e+04]),
                    rtol=1E-4)


def test_castro_test_spectra_castro(tmpdir):
    castrofile = str(tmpdir.join('castro.fits'))
    url = 'https://raw.githubusercontent.com/fermiPy/fermipy-extra/master/data/castro.fits'
    os.system('curl -o %s -OL %s' % (castrofile, url))
    c = castro.CastroData.create_from_fits(castrofile, irow=19)
    test_dict = c.test_spectra()

    assert_allclose(test_dict['PowerLaw']['TS'][0], 0.00, atol=0.01)
    assert_allclose(test_dict['LogParabola']['TS'][0], 0.00, atol=0.01)
    assert_allclose(test_dict['PLExpCutoff']['TS'][0], 2.71, atol=0.01)


def test_fit_dmfitfunction(sedfile):

    cd = castro.CastroData.create_from_sedfile(sedfile)

    init_pars = np.array([1E-26, 100.])
    jfactor = 1E19
    dmfn_bb = DMFitFunction(init_pars, chan='bb', jfactor=jfactor)
    dmfn_mumu = DMFitFunction(init_pars, chan='mumu', jfactor=jfactor)
    dmfn_tautau = DMFitFunction(init_pars, chan='tautau', jfactor=jfactor)

    init_pars = np.array([1E-26, 100.])
    spec_func = cd.create_functor(dmfn_bb, init_pars)
    fit_out = cd.fit_spectrum(spec_func, init_pars, freePars=[True, False])

    assert_allclose(fit_out['ts_spec'], 23.96697127, atol=0.01)
    assert_allclose(fit_out['params'][0], 8.02000000e-25)


    init_pars = np.array([1E-26, 30.])
    spec_func = cd.create_functor(dmfn_mumu, init_pars)
    fit_out = cd.fit_spectrum(spec_func, init_pars, freePars=[True, False])

    assert_allclose(fit_out['ts_spec'], 14.66041109, atol=0.01)
    assert_allclose(fit_out['params'][0], 2.45750000e-24, rtol=0.05)


    init_pars = np.array([1E-26, 30.])
    spec_func = cd.create_functor(dmfn_tautau, init_pars)
    fit_out = cd.fit_spectrum(spec_func, init_pars, freePars=[True, False])

    assert_allclose(fit_out['ts_spec'], 17.14991598, atol=0.01)
    assert_allclose(fit_out['params'][0], 2.98000000e-25, rtol=0.05)
