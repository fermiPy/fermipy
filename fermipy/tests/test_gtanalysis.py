# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from fermipy.tests.utils import requires_dependency, requires_st_version
from fermipy import spectrum

try:
    from fermipy import gtanalysis
except ImportError:
    pass

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')


@pytest.fixture(scope='module')
def create_draco_analysis(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('draco')
    url = 'https://raw.githubusercontent.com/fermiPy/fermipy-extras/master/data/fermipy_test_draco.tar.gz'
    outfile = path.join('fermipy_test_draco.tar.gz')
    dirname = path.join()
    os.system('curl -o %s -OL %s' % (outfile, url))
    os.system('cd %s;tar xzf %s' % (dirname, outfile))
    request.addfinalizer(lambda: path.remove(rec=1))

    cfgfile = path.join('fermipy_test_draco', 'config.yaml')
    gta = gtanalysis.GTAnalysis(str(cfgfile))
    gta.setup()

    return gta


@pytest.fixture(scope='module')
def create_pg1553_analysis(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('pg1553')
    url = 'https://raw.githubusercontent.com/fermiPy/fermipy-extras/master/data/fermipy_test_pg1553.tar.gz'
    outfile = path.join('fermipy_test_pg1553.tar.gz')
    dirname = path.join()
    os.system('curl -o %s -OL %s' % (outfile, url))
    os.system('cd %s;tar xzf %s' % (dirname, outfile))

    ft2_files = ['P8_P302_TRANSIENT020E_239557414_242187214_ft2.fits',
                 'P8_P302_TRANSIENT020E_247446814_250076614_ft2.fits',
                 'P8_P302_TRANSIENT020E_255336214_257966014_ft2.fits',
                 'P8_P302_TRANSIENT020E_242187214_244817014_ft2.fits',
                 'P8_P302_TRANSIENT020E_250076614_252706414_ft2.fits',
                 'P8_P302_TRANSIENT020E_257966014_260595814_ft2.fits',
                 'P8_P302_TRANSIENT020E_244817014_247446814_ft2.fits',
                 'P8_P302_TRANSIENT020E_252706414_255336214_ft2.fits',
                 'P8_P302_TRANSIENT020E_260595814_263225614_ft2.fits']

    for f in ft2_files:
        url = 'https://raw.githubusercontent.com/fermiPy/fermipy-extras/master/data/ft2/%s' % f
        outfile = path.join('fermipy_test_pg1553', f)
        os.system('curl -o %s -OL %s' % (outfile, url))

    #request.addfinalizer(lambda: path.remove(rec=1))

    cfgfile = path.join('fermipy_test_pg1553', 'config.yaml')

    gta = gtanalysis.GTAnalysis(str(cfgfile))
    gta.setup()

    return gta


def test_gtanalysis_setup(create_draco_analysis):
    gta = create_draco_analysis
    gta.print_roi()


def test_print_model(create_draco_analysis):
    gta = create_draco_analysis
    gta.print_model()


def test_print_params(create_draco_analysis):
    gta = create_draco_analysis
    gta.print_params(True)


def test_gtanalysis_write_roi(create_draco_analysis):
    gta = create_draco_analysis
    gta.write_roi('test', make_plots=True)


def test_gtanalysis_load_roi(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit0')
    src = gta.roi['3FGL J1725.3+5853']

    prefactor = src.spectral_pars['Prefactor']
    index = src.spectral_pars['Index']
    assert_allclose(prefactor['value'] * prefactor['scale'],
                    1.6266779e-13, rtol=1E-3)
    assert_allclose(index['value'] * index['scale'], -2.17892, rtol=1E-3)
    assert_allclose(src['flux'], 4.099648e-10, rtol=1E-3)
    assert_allclose(src['flux_err'], np.nan, rtol=1E-3)
    assert_allclose(src['eflux'], 9.76762e-07, rtol=1E-3)
    assert_allclose(src['eflux_err'], np.nan, rtol=1E-3)

    gta.load_roi('fit1')
    src = gta.roi['3FGL J1725.3+5853']
    prefactor = src.spectral_pars['Prefactor']
    index = src.spectral_pars['Index']
    assert_allclose(prefactor['value'] *
                    prefactor['scale'], 2.0878036e-13, rtol=1E-3)
    assert_allclose(index['value'] * index['scale'], -2.053723, rtol=1E-3)
    assert_allclose(src['flux'], 5.377593e-10, rtol=1E-3)
    assert_allclose(src['flux_err'], 6.40203e-11, rtol=1E-3)
    assert_allclose(src['eflux'], 1.34617749e-06, rtol=1E-3)
    assert_allclose(src['eflux_err'], 1.584814e-07, rtol=1E-3)
    assert_allclose(src['ts'], 200.604, rtol=1E-3)
    assert_allclose(src['npred'], 170.258, rtol=1E-3)


def test_gtanalysis_optimize(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit0')
    gta.optimize()


def test_gtanalysis_fit(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit0')
    gta.free_sources(distance=3.0, pars='norm')
    gta.write_xml('fit_test')
    fit_output0 = gta.fit(optimizer='MINUIT')
    gta.load_xml('fit_test')
    fit_output1 = gta.fit(optimizer='NEWMINUIT')

    assert (np.abs(fit_output0['loglike'] - fit_output1['loglike']) < 0.01)


@requires_st_version('11-04-00')
def test_gtanalysis_fit_newton(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit0')
    gta.free_sources(distance=3.0, pars='norm')
    gta.write_xml('fit_test')
    fit_output0 = gta.fit(optimizer='MINUIT')
    gta.load_xml('fit_test')
    fit_output1 = gta.fit(optimizer='NEWTON')

    assert (np.abs(fit_output0['loglike'] - fit_output1['loglike']) < 0.01)


def test_gtanalysis_tsmap(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit1')
    gta.tsmap(model={}, make_plots=True)


@requires_st_version('11-04-00')
def test_gtanalysis_tscube(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit1')
    gta.tscube(model={}, make_plots=True)


def test_gtanalysis_residmap(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit1')
    gta.residmap(model={}, make_plots=True)


def test_gtanalysis_find_sources(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit1')
    np.random.seed(1)

    src0 = {'SpatialModel': 'PointSource',
            'Index': 2.0, 'offset_glon': 0.0, 'offset_glat': 2.0,
            'Prefactor': 1E-12}

    src1 = {'SpatialModel': 'PointSource',
            'Index': 2.0, 'offset_glon': 0.0, 'offset_glat': -2.0,
            'Prefactor': 1E-12}

    gta.add_source('src0', src0)
    gta.add_source('src1', src1)
    gta.simulate_roi()
    src0 = gta.delete_source('src0')
    src1 = gta.delete_source('src1')

    gta.find_sources()

    diff_sources = [s.name for s in gta.roi.sources if s.diffuse]
    newsrcs0 = gta.get_sources(skydir=src0.skydir, distance=0.3,
                               exclude=diff_sources)
    newsrcs1 = gta.get_sources(skydir=src1.skydir, distance=0.3,
                               exclude=diff_sources)

    assert(len(newsrcs0) == 1)
    assert(len(newsrcs1) == 1)

    newsrc0 = newsrcs0[0]
    newsrc1 = newsrcs1[0]

    sep0 = src0.skydir.separation(newsrc0.skydir).deg
    sep1 = src1.skydir.separation(newsrc1.skydir).deg

    assert(sep0 < newsrc0['pos_r99'])
    assert(sep1 < newsrc1['pos_r99'])

    flux_diff0 = (np.abs(src0['flux'] - newsrc0['flux']) /
                  newsrc0['flux_err'])
    flux_diff1 = (np.abs(src1['flux'] - newsrc1['flux']) /
                  newsrc1['flux_err'])

    assert(flux_diff0 < 3.0)
    assert(flux_diff1 < 3.0)


def test_gtanalysis_sed(create_draco_analysis):
    gta = create_draco_analysis
    gta.load_roi('fit1')
    np.random.seed(1)
    gta.simulate_roi()

    params = gta.roi['draco'].params

    prefactor = 3E-12
    index = 1.9
    scale = params['Scale']['value']
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

    o = gta.sed('draco', make_plots=True)

    flux_resid = (flux_true - o['flux']) / o['flux_err']
    assert_allclose(flux_resid, 0, atol=3.0)

    params = gta.roi['draco'].params
    index_resid = (-params['Index']['value'] - index) / \
        params['Index']['error']
    assert_allclose(index_resid, 0, atol=3.0)

    prefactor_resid = (params['Prefactor']['value'] -
                       prefactor) / params['Prefactor']['error']
    assert_allclose(prefactor_resid, 0, atol=3.0)

    gta.simulate_roi(restore=True)


def test_gtanalysis_extension_gaussian(create_draco_analysis):
    gta = create_draco_analysis
    gta.simulate_roi(restore=True)
    gta.load_roi('fit1')
    np.random.seed(1)
    spatial_width = 0.5

    gta.simulate_source({'SpatialModel': 'RadialGaussian',
                         'SpatialWidth': spatial_width,
                         'Prefactor': 3E-12})

    o = gta.extension('draco',
                      width=[0.4, 0.45, 0.5, 0.55, 0.6],
                      spatial_model='RadialGaussian')

    assert_allclose(o['ext'], spatial_width, atol=0.1)

    gta.simulate_roi(restore=True)


def test_gtanalysis_localization(create_draco_analysis):
    gta = create_draco_analysis
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

    result = gta.localize('testloc', nstep=4, dtheta_max=0.5, update=True,
                          make_plots=True)

    assert result['fit_success'] is True
    assert_allclose(result['glon'], 86.0, atol=0.02)
    assert_allclose(result['glat'], 36.0, atol=0.02)
    gta.delete_source('testloc')

    gta.simulate_roi(restore=True)


def test_gtanalysis_lightcurve(create_pg1553_analysis):
    gta = create_pg1553_analysis
    gta.load_roi('fit1')
    o = gta.lightcurve('3FGL J1555.7+1111', nbins=2,
                       free_radius=3.0)

    rtol = 0.01
    flux = np.array([2.917568e-08,
                     2.359114e-08])
    flux_err = np.array([1.931940e-09,
                         1.822694e-09])
    ts = np.array([1463.066,
                   1123.160])

    assert_allclose(o['flux'], flux, rtol=rtol)
    assert_allclose(o['flux_err'], flux_err, rtol=rtol)
    assert_allclose(o['ts'], ts, rtol=rtol)

    tab = Table.read(os.path.join(gta.workdir, o['file']))
    assert_allclose(tab['flux'], flux, rtol=rtol)
    assert_allclose(tab['flux_err'], flux_err, rtol=rtol)
    assert_allclose(tab['ts'], ts, rtol=rtol)
