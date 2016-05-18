import pytest
import xml.etree.cElementTree as ElementTree

from numpy.testing import assert_allclose

from fermipy import roi_model
from fermipy.roi_model import Source


@pytest.fixture(scope='module')
def tmppath(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('tmpdir')
    return path


def test_load_3fgl_catalog_fits():

    rm = roi_model.ROIModel(catalogs=['3FGL'])
    assert(len(rm.sources) == 3034)

    rm = roi_model.ROIModel(catalogs=['gll_psc_v16.fit'])
    assert(len(rm.sources) == 3034)

def test_load_3fgl_catalog_xml():

    rm = roi_model.ROIModel(catalogs=['gll_psc_v16.xml'],
                            extdir='Extended_archive_v15')
    assert(len(rm.sources) == 3034)


def test_load_2fhl_catalog_fits():

    rm = roi_model.ROIModel(catalogs=['2FHL'])
    assert(len(rm.sources) == 360)


def test_load_roi_from_dict(tmppath):

    sources = [{'name': 'ptsrc0', 'SpectrumType': 'PowerLaw',
                'ra': 121.0, 'dec': 32.56,
                'Index': 2.3, 'Prefactor': 1.3E-15, 'Scale': 1452.0},
               {'name': 'ptsrc1', 'SpectrumType': 'PowerLaw',
                'ra': 121.0, 'dec': 32.56,
                'Index': {'value': 2.3, 'scale': 2.4, 'min': 0.3, 'max': 5.7},
                'Prefactor': {'value': 1.2, 'scale': 3E-12, 'min': 0.0, 'max': 5.0},
                'Scale': {'value': 134.1, 'scale': 1.2, 'min': 0.0, 'max': 2000.0},
               } ]

    roi = roi_model.ROIModel(sources=sources)
    src = roi['ptsrc0']

    assert_allclose(src['dec'], sources[0]['dec'])
    assert_allclose(src['ra'], sources[0]['ra'])

    sp = src['spectral_pars']

    assert_allclose(sp['Index']['value'],sources[0]['Index'])
    assert_allclose(sp['Scale']['value'],1.452)
    assert_allclose(sp['Prefactor']['value'],1.3)

    for p in ['Index', 'Prefactor', 'Scale']:
        if p == 'Index':
            assert_allclose(sources[0][p], -src.params[p][0])
        else:
            assert_allclose(sources[0][p], src.params[p][0])

    src = roi['ptsrc1']
    sp = src['spectral_pars']
    for p in ['Index', 'Prefactor', 'Scale']:
        assert_allclose(sources[1][p]['value']*sources[1][p]['scale'],
                        src.params[p][0])


def test_load_source_from_xml(tmppath):

    values = {'ptsrc_dec': 52.6356, 'ptsrc_ra': 252.367,
              'ptsrc_index_value': 2.21879, 'ptsrc_index_scale': 1.2,
              'ptsrc_index_min': 1.3, 'ptsrc_index_max': 3.5,
              'ptsrc_prefactor_value': 0.727, 'ptsrc_prefactor_scale': 1e-12,
              'ptsrc_prefactor_min': 0.01, 'ptsrc_prefactor_max': 10.0,
              'ptsrc_scale_value': 1.3, 'ptsrc_scale_scale': 1e3,
              'ptsrc_scale_min': 0.01, 'ptsrc_scale_max': 113.0}

    xmlmodel = """
    <source_library title="source_library">
    <source name="3FGL J1649.4+5238" type="PointSource">
    <spatialModel type="SkyDirFunction">
    <parameter free="0" max="90.0" min="-90.0" name="DEC" scale="1.0" value="{ptsrc_dec}"/>
    <parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="{ptsrc_ra}"/>
    </spatialModel>
    <spectrum type="PowerLaw">
    <parameter free="0" max="{ptsrc_index_max}" min="{ptsrc_index_min}" name="Index" scale="{ptsrc_index_scale}" value="{ptsrc_index_value}"/>
    <parameter free="0" max="{ptsrc_scale_max}" min="{ptsrc_scale_min}" name="Scale" scale="{ptsrc_scale_scale}" value="{ptsrc_scale_value}"/>
    <parameter free="0" max="{ptsrc_prefactor_max}" min="{ptsrc_prefactor_min}" name="Prefactor" scale="{ptsrc_prefactor_scale}" value="{ptsrc_prefactor_value}"/>
    </spectrum>
    </source>
    <source name="isodiff" type="DiffuseSource">
    <spectrum ctype="-1" file="$(FERMIPY_ROOT)/data/iso_P8R2_SOURCE_V6_v06.txt" type="FileFunction">
    <parameter free="0" max="1000.0" min="0.001" name="Normalization" scale="1.0" value="1.2"/>
    </spectrum>
    <spatialModel type="ConstantValue">
    <parameter free="0" max="10" min="0" name="Value" scale="1" value="1"/>
    </spatialModel>
    </source>
    <source name="galdiff" type="DiffuseSource">
    <spectrum type="PowerLaw">
    <parameter free="0" max="1.0" min="-1.0" name="Index" scale="-1" value="0.0"/>
    <parameter free="0" max="1000.0" min="1000.0" name="Scale" scale="1" value="1000.0"/>
    <parameter free="0" max="10.0" min="0.1" name="Prefactor" scale="1" value="1.0"/>
    </spectrum>
    <spatialModel file="$(FERMIPY_WORKDIR)/gll_iem_v06_extracted.fits" type="MapCubeFunction">
    <parameter free="0" max="10" min="0" name="Normalization" scale="1" value="1"/>
    </spatialModel>
    </source>
    </source_library>
    """.format(**values)

    root = ElementTree.fromstring(xmlmodel)
    xmlfile = str(tmppath.join('test.xml'))
    print xmlfile
    ElementTree.ElementTree(root).write(xmlfile)

    roi = roi_model.ROIModel(config={'catalogs': [xmlfile]})

    src = roi['3FGL J1649.4+5238']

    assert_allclose(src['dec'],values['ptsrc_dec'])
    assert_allclose(src['ra'],values['ptsrc_ra'])
    assert(src['SpectrumType'] == 'PowerLaw')
    assert(src['SpatialType'] == 'SkyDirFunction')
    assert(src['SpatialModel'] == 'PointSource')
    assert(src['SourceType'] == 'PointSource')

    sp = src['spectral_pars']

    attribs = ['value', 'scale', 'min', 'max']

    for x in attribs:
        assert_allclose(sp['Index'][x],values['ptsrc_index_%s'%x])

    for x in attribs:
        assert_allclose(sp['Prefactor'][x],values['ptsrc_prefactor_%s'%x])

    for x in attribs:
        assert_allclose(sp['Scale'][x],values['ptsrc_scale_%s'%x])
        
    assert_allclose(src.params['Prefactor'][0],
                    values['ptsrc_prefactor_value']*values['ptsrc_prefactor_scale'])
    assert_allclose(src.params['Index'][0],
                    values['ptsrc_index_value']*values['ptsrc_index_scale'])
    assert_allclose(src.params['Scale'][0],
                    values['ptsrc_scale_value']*values['ptsrc_scale_scale'])
    
    src = roi['galdiff']
    assert(src['SpatialType'] == 'MapCubeFunction')
    assert(src['SpatialModel'] == 'MapCubeFunction')

    src = roi['isodiff']
    assert(src['SpatialType'] == 'ConstantValue')
    assert(src['SpatialModel'] == 'ConstantValue')


def test_create_source_from_dict(tmppath):

    ra = 252.367
    dec = 52.6356
    
    src = Source.create_from_dict({'name' : 'testsrc',
                                   'SpatialModel' : 'PointSource',
                                   'SpectrumType' : 'PowerLaw',
                                   'Index' : 2.3,
                                   'ra' : ra, 'dec' : dec})

    assert_allclose(src['ra'],ra)
    assert_allclose(src['dec'],dec)
    assert(src['SpatialModel'] == 'PointSource')
    assert(src['SpatialType'] == 'SkyDirFunction')
    assert(src['SourceType'] == 'PointSource')
    assert(src.extended is False)
    
    src = Source.create_from_dict({'name' : 'testsrc',
                                   'SpatialModel' : 'GaussianSource',
                                   'SpectrumType' : 'PowerLaw',
                                   'Index' : 2.3,
                                   'ra' : ra, 'dec' : dec})

    assert_allclose(src['ra'],ra)
    assert_allclose(src['dec'],dec)
    assert(src['SpatialModel'] == 'GaussianSource')
    assert(src['SpatialType'] == 'SpatialMap')
    assert(src['SourceType'] == 'DiffuseSource')
    assert(src.extended is True)
    
    src = Source.create_from_dict({'name' : 'testsrc',
                                   'SpatialModel' : 'RadialGaussian',
                                   'SpectrumType' : 'PowerLaw',
                                   'Index' : 2.3, 'Sigma' : 0.5,
                                   'ra' : ra, 'dec' : dec})

    assert_allclose(src['ra'],ra)
    assert_allclose(src['dec'],dec)
    assert_allclose(src['Sigma'],0.5)
    if src['SpatialType'] == 'RadialGaussian':
        assert_allclose(src.spatial_pars['Sigma']['value'],0.5)
    
    assert(src['SpatialModel'] == 'RadialGaussian')
    assert(src['SourceType'] == 'DiffuseSource')
    assert(src.extended is True)

    src = Source.create_from_dict({'name' : 'testsrc',
                                   'SpatialModel' : 'RadialDisk',
                                   'SpectrumType' : 'PowerLaw',
                                   'Index' : 2.3, 'Radius' : 0.5,
                                   'ra' : ra, 'dec' : dec})

    assert_allclose(src['ra'],ra)
    assert_allclose(src['dec'],dec)
    assert_allclose(src['Radius'],0.5)
    if src['SpatialType'] == 'RadialDisk':
        assert_allclose(src.spatial_pars['Radius']['value'],0.5)
    
    assert(src['SpatialModel'] == 'RadialDisk')
    assert(src['SourceType'] == 'DiffuseSource')
    assert(src.extended is True)
    

def test_create_source(tmppath):

    ra = 252.367
    dec = 52.6356
    sigma = 0.5
    
    src_dict = {'SpatialModel' : 'GaussianSource', 'ra' : ra, 'dec' : dec, 'Sigma' : sigma}    
    src = Source('testsrc',src_dict)

    assert_allclose(src['ra'],ra)
    assert_allclose(src['dec'],dec)
    assert_allclose(src['Sigma'],sigma)
    assert(src['SpatialModel'] == 'GaussianSource')


def test_set_spatial_model(tmppath):

    ra = 252.367
    dec = 52.6356
    
    src_dict = {'SpatialModel' : 'GaussianSource', 'ra' : ra, 'dec' : dec}    
    src = Source('testsrc',src_dict)

    src.set_spatial_model('PointSource')
    assert(src['SpatialModel'] == 'PointSource')
    assert(src['SpatialType'] == 'SkyDirFunction')
    assert(src['SourceType'] == 'PointSource')
    
