# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from fermipy.tests.utils import requires_dependency
from fermipy import spectrum

try:
    from fermipy import gtanalysis
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')


@pytest.fixture(scope='module')
def setup(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    print('\ndownload')
    url = 'https://www.dropbox.com/s/8a9ebwolxmif1n6/fermipy_test0_small.tar.gz'
    #    'https://www.dropbox.com/s/b5zln7ku780xvzq/fermipy_test0.tar.gz'

    #    dirname = os.path.abspath(os.path.dirname(__file__))
    outfile = path.join('fermipy_test0_small.tar.gz')
    dirname = path.join()
    # os.system('wget -nc %s -O %s' % (url, outfile))
    os.system('curl -o %s -OL %s' % (outfile, url))
    os.system('cd %s;tar xzf %s' % (dirname, outfile))

    os.system('touch %s' % path.join('test.txt'))
    request.addfinalizer(lambda: path.remove(rec=1))

    cfgfile = path.join('fermipy_test0_small', 'config.yaml')

    gta = gtanalysis.GTAnalysis(str(cfgfile))
    gta.setup()

    return gta


def test_gtanalysis_setup(setup):
    gta = setup
    gta.print_roi()


def test_gtanalysis_write_roi(setup):
    gta = setup
    gta.write_roi('test')


def test_gtanalysis_load_roi(setup):
    gta = setup
    gta.load_roi('fit0')


def test_gtanalysis_optimize(setup):
    gta = setup
    gta.load_roi('fit0')
    gta.optimize()


def test_gtanalysis_fit(setup):
    gta = setup
    gta.load_roi('fit0')
    gta.free_sources(distance=3.0, pars='norm')
    gta.write_xml('fit_test')
    fit_output0 = gta.fit(optimizer='MINUIT')
    gta.load_xml('fit_test')
    fit_output1 = gta.fit(optimizer='NEWMINUIT')

    assert (np.abs(fit_output0['loglike'] - fit_output1['loglike']) < 0.01)


def test_gtanalysis_tsmap(setup):
    gta = setup
    gta.load_roi('fit1')
    gta.tsmap(model={})


def test_gtanalysis_residmap(setup):
    gta = setup
    gta.load_roi('fit1')
    gta.residmap(model={})


def test_gtanalysis_sed(setup):
    gta = setup
    gta.load_roi('fit1')
    np.random.seed(1)
    gta.simulate_roi()

    prefactor = 3E-12
    index = 1.9
    scale = gta.roi['draco'].params['Scale'][0]

    emin = gta.energies[:-1]
    emax = gta.energies[1:]

    flux_true = spectrum.PowerLaw.eval_flux(emin, emax,
                                            [prefactor, -index], scale)

    gta.simulate_source({'SpatialModel': 'PointSource',
                         'Index': index,
                         'Scale': scale,
                         'Prefactor': prefactor})

    gta.free_source('draco')
    gta.fit()

    o = gta.sed('draco')

    flux_resid = (flux_true - o['flux']) / o['flux_err']
    assert_allclose(flux_resid, 0, atol=3.0)

    params = gta.roi['draco'].params
    index_resid = (-params['Index'][0] - index) / params['Index'][1]
    assert_allclose(index_resid, 0, atol=3.0)

    prefactor_resid = (params['Prefactor'][0] - prefactor) / params['Prefactor'][1]
    assert_allclose(prefactor_resid, 0, atol=3.0)

    gta.simulate_roi(restore=True)


def test_gtanalysis_extension_gaussian(setup):
    gta = setup
    gta.simulate_roi(restore=True)
    gta.load_roi('fit1')
    np.random.seed(1)
    spatial_width = 0.5

    gta.simulate_source({'SpatialModel': 'GaussianSource',
                         'SpatialWidth': spatial_width,
                         'Prefactor': 3E-12})

    o = gta.extension('draco',
                      width=[0.4, 0.45, 0.5, 0.55, 0.6],
                      spatial_model='GaussianSource')

    assert_allclose(o['ext'], spatial_width, atol=0.1)

    gta.simulate_roi(restore=True)


def test_gtanalysis_localization(setup):
    gta = setup
    gta.simulate_roi(restore=True)
    gta.load_roi('fit1')
    np.random.seed(1)

    src_dict = {'SpatialModel': 'PointSource',
                'Prefactor': 4E-12,
                'glat': 36.0, 'glon': 86.0}

    gta.simulate_source(src_dict)

    src_dict['glat'] = 36.05
    src_dict['glon'] = 86.05

    gta.add_source('testloc', src_dict, free=True)
    gta.fit()

    result = gta.localize('testloc', nstep=5, dtheta_max=0.5, update=True)

    assert result['fit_success'] is True
    assert_allclose(result['glon'], 86.0, atol=0.02)
    assert_allclose(result['glat'], 36.0, atol=0.02)
    gta.delete_source('testloc')

    gta.simulate_roi(restore=True)
