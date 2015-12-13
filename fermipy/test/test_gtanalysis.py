from numpy.testing import assert_allclose
import astropy.units as u
#from .. import gtanalysis
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

    print('download')
    url = 'https://www.dropbox.com/s/1lpkincrj04c8m2/draco.tar.gz'
#    dirname = os.path.abspath(os.path.dirname(__file__))
#    dirname = path.dirpath()
 #   outfile = os.path.join(dirname,'draco.tar.gz')
    outfile = path.join('draco.tar.gz')
    dirname = path.join()
    os.system('wget -nc %s -O %s'%(url,outfile))
    os.system('cd %s;tar xzf %s'%(dirname,outfile))

    os.system('touch %s'%path.join('test.txt'))
    request.addfinalizer(lambda: path.remove(rec=1))
    return path
    
def test_gtanalysis_one(setup2):
    print('here is test one')
    cfgfile = setup2.join('draco','config.yaml')
    os.system('cat %s'%cfgfile)
    print(os.getcwd())
    print(setup2)
    

def test_gtanalysis_two(setup2):
    print('here is test two')
    print(os.getcwd())
    print(setup2)
