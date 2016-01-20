from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from .. import gtanalysis
import pytest
import os

@pytest.fixture(scope="module")
def setup(request):
    print('download')
    url = 'https://www.dropbox.com/s/1lpkincrj04c8m2/draco.tar.gz'
    dirname = os.path.abspath(os.path.dirname(__file__))
    outfile = os.path.join(dirname,'draco.tar.gz')
    os.system('wget -nc %s -O %s'%(url,outfile))
    os.system('cd %s;tar xzf %s'%(dirname,outfile))
    def fin():        
        print("delete")
        os.system('rm %s'%(outfile))      
        os.system('rm -rf %s'%(os.path.join(dirname,'draco'))) 

    request.addfinalizer(fin)
    return 

@pytest.fixture(scope='module')
def setup2(request,tmpdir_factory):    
    path =  tmpdir_factory.mktemp('data')

    print('\ndownload')
    url = 'https://www.dropbox.com/s/b5zln7ku780xvzq/fermipy_test0.tar.gz'
#    url = 'https://www.dropbox.com/s/1lpkincrj04c8m2/draco.tar.gz'
#    dirname = os.path.abspath(os.path.dirname(__file__))
#    dirname = path.dirpath()
 #   outfile = os.path.join(dirname,'draco.tar.gz')
    outfile = path.join('fermipy_test0.tar.gz')
    dirname = path.join()
    os.system('wget -nc %s -O %s'%(url,outfile))
    os.system('cd %s;tar xzf %s'%(dirname,outfile))

    os.system('touch %s'%path.join('test.txt'))
    request.addfinalizer(lambda: path.remove(rec=1))

    cfgfile = path.join('fermipy_test0','config.yaml')

    print(cfgfile)

    gta = gtanalysis.GTAnalysis(str(cfgfile))
    gta.setup()

    return gta
    
#@pytest.mark.skipif("True")
def test_gtanalysis_setup(setup2):

    gta = setup2
    gta.print_roi()

#@pytest.mark.skipif("True")
def test_gtanalysis_write_roi(setup2):

    gta = setup2
    gta.write_roi('test')

def test_gtanalysis_load_roi(setup2):

    gta = setup2
    gta.load_roi('fit0')

def test_gtanalysis_optimize(setup2):

    gta = setup2
    gta.load_roi('fit0')
    gta.optimize()

def test_gtanalysis_tsmap(setup2):

    gta = setup2
    gta.load_roi('fit0')
    gta.tsmap(model={})

def test_gtanalysis_residmap(setup2):

    gta = setup2
    gta.load_roi('fit0')
    gta.residmap(model={})

def test_gtanalysis_extension_gaussian(setup2):

    gta = setup2
    gta.load_roi('fit0')

    spatial_width = 0.5

    gta.simulate_source({'SpatialModel' : 'GaussianSource',
                         'SpatialWidth' : spatial_width, 
                         'Prefactor' : 3E-12})

    o = gta.extension('draco',width=[0.4,0.45,0.5,0.55,0.6],
                      spatial_model='GaussianSource')

    

    assert(np.abs(o['ext']-spatial_width) < 0.1)

    gta.restore_counts_maps()

def test_gtanalysis_localization(setup2):

    gta = setup2
    gta.load_roi('fit0')

    spatial_width = 0.5

    src_dict = {'SpatialModel' : 'PointSource',
                'Prefactor' : 2E-12,
                'glat' : 36.0, 'glon' : 86.0}

    gta.simulate_source(src_dict)

    src_dict['glat'] = 36.05
    src_dict['glon'] = 86.05

    gta.add_source('testloc',src_dict,free=True)

    o = gta.localize('testloc',nstep=5,dtheta_max=0.1,
                     update=True)

    assert(np.abs(o['glon']-86.0) < 0.025)
    assert(np.abs(o['glat']-36.0) < 0.025)

    gta.delete_source('testloc')

    gta.restore_counts_maps()
