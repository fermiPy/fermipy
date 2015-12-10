from numpy.testing import assert_allclose
import astropy.units as u
from .. import roi_model


def test_load_3fgl_catalog_fits():
    
    rm = roi_model.ROIModel(catalogs=['3FGL'])
    assert(len(rm.sources)==3034)

def test_load_3fgl_catalog_xml():
    
    rm = roi_model.ROIModel(catalogs=['gll_psc_v16.xml'])
    assert(len(rm.sources)==3034)

def test_load_2fhl_catalog_fits():
    
    rm = roi_model.ROIModel(catalogs=['2FHL'])
    assert(len(rm.sources)==360)

