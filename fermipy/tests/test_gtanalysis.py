from __future__ import absolute_import, division, print_function, \
    unicode_literals

from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from .. import gtanalysis
from .. import utils
from .. import spectrum
import pytest
import os


@pytest.fixture(scope='module')
def setup(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    print('\ndownload')
    url = 'https://www.dropbox.com/s/8a9ebwolxmif1n6/fermipy_test0_small.tar.gz'
#    'https://www.dropbox.com/s/b5zln7ku780xvzq/fermipy_test0.tar.gz'

#    dirname = os.path.abspath(os.path.dirname(__file__))
    outfile = path.join('fermipy_test0_small.tar.gz')
    dirname = path.join()
    os.system('wget -nc %s -O %s' % (url, outfile))
    os.system('cd %s;tar xzf %s' % (dirname, outfile))

    os.system('touch %s' % path.join('test.txt'))
    request.addfinalizer(lambda: path.remove(rec=1))

    cfgfile = path.join('fermipy_test0_small', 'config.yaml')

    print(cfgfile)

    gta = gtanalysis.GTAnalysis(str(cfgfile))
    gta.setup()

    return gta


#@pytest.mark.skipif("True")
def test_gtanalysis_setup(setup):

    gta = setup
    gta.print_roi()


#@pytest.mark.skipif("True")
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
    
    flux_true = spectrum.PowerLaw.eval_flux(10**emin,10**emax,
                                            [prefactor,-index],scale)
    
    gta.simulate_source({'SpatialModel': 'PointSource',
                         'Index' : index,
                         'Scale' : scale,
                         'Prefactor': prefactor})

    gta.free_source('draco')
    gta.fit()
    
    o = gta.sed('draco')

    flux_resid = np.abs((flux_true - o['flux'])/o['flux_err'])
    params = gta.roi['draco'].params
    assert(np.all(flux_resid<3.0))
    assert(np.abs(-params['Index'][0]-index)/params['Index'][1] < 3.0)
    assert(np.abs(params['Prefactor'][0]-prefactor)/params['Prefactor'][1] < 3.0)
    
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

    assert(np.abs(o['ext']-spatial_width) < 0.1)

    gta.simulate_roi(restore=True)


def test_gtanalysis_localization(setup):

    gta = setup
    gta.simulate_roi(restore=True)
    gta.load_roi('fit1')
    np.random.seed(1)

    src_dict = {'SpatialModel': 'PointSource',
                'Prefactor': 2E-12,
                'glat': 36.0, 'glon': 86.0}

    gta.simulate_source(src_dict)

    src_dict['glat'] = 36.05
    src_dict['glon'] = 86.05

    gta.add_source('testloc', src_dict, free=True)
    gta.fit()

    o = gta.localize('testloc', nstep=5, dtheta_max=0.5,
                     update=True)

    import pprint
    pprint.pprint(o)

    assert(np.abs(o['glon']-86.0) < 0.025)
    assert(np.abs(o['glat']-36.0) < 0.025)
    gta.delete_source('testloc')

    gta.simulate_roi(restore=True)
