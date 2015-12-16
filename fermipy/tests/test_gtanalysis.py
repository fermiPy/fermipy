from numpy.testing import assert_allclose
import astropy.units as u
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
    print('here is test one')
    print(os.getcwd())

    gta = setup2
    gta.print_roi()

#@pytest.mark.skipif("True")
def test_gtanalysis_write_roi(setup2):
    print('here is test two')
    print(os.getcwd())
    print(setup2)

    gta = setup2
    gta.write_roi('test')
