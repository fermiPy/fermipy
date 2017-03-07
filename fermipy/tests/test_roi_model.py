# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import xml.etree.cElementTree as ElementTree
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
from fermipy.tests.utils import requires_dependency
from fermipy import roi_model
from fermipy.roi_model import Source, ROIModel

# Skip tests in this file if Fermi ST aren't available
#pytestmark = requires_dependency('Fermi ST')


@pytest.fixture(scope='module')
def tmppath(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('tmpdir')
    return path


def test_load_3fgl_catalog_fits():
    skydir = SkyCoord(0.0, 0.0, unit='deg', frame='galactic').icrs
    rm = ROIModel(catalogs=['3FGL'], skydir=skydir, src_radius=20.0)
    assert len(rm.sources) == 175

    rm = ROIModel(catalogs=['gll_psc_v16.fit'], skydir=skydir,
                            src_radius=20.0)
    assert len(rm.sources) == 175


def test_load_3fgl_catalog_xml():
    skydir = SkyCoord(0.0, 0.0, unit='deg', frame='galactic').icrs
    rm = ROIModel(catalogs=['gll_psc_v16.xml'],
                            extdir='Extended_archive_v15',
                            skydir=skydir, src_radius=20.0)
    assert len(rm.sources) == 175


def test_load_2fhl_catalog_fits():
    rm = ROIModel(catalogs=['2FHL'])
    assert len(rm.sources) == 360


def test_create_roi_from_source():

    rm = ROIModel.create_from_source('3FGL J2021.0+4031e',
                                     {'catalogs' : ['3FGL'], 'src_radius' : 2.0})
    assert len(rm.sources) == 9
    src = rm.sources[0]    
    assert src.name == '3FGL J2021.0+4031e'
    assert src['SpatialType'] == 'SpatialMap'
    assert src['SourceType'] == 'DiffuseSource'
    assert src['SpectrumType'] == 'PowerLaw'
    assert_allclose(src['ra'], 305.26999, rtol=1E-5)
    assert_allclose(src['dec'], 40.52, rtol=1E-5)
    assert_allclose(src.spectral_pars['Index']['value'], 1.53, rtol=1E-4)
    assert_allclose(src.spectral_pars['Prefactor']['value']*
                    src.spectral_pars['Prefactor']['scale'], 0.4003659112E-12, rtol=1E-4)
    assert_allclose(src.spatial_pars['Prefactor']['value'], 1.0, rtol=1E-4)
    assert_allclose(src.spatial_pars['Prefactor']['scale'], 1.0, rtol=1E-4)


def test_load_roi_from_dict(tmppath):
    sources = [{'name': 'ptsrc0', 'SpectrumType': 'PowerLaw',
                'ra': 121.0, 'dec': 32.56,
                'Index': 2.3, 'Prefactor': 1.3E-15, 'Scale': 1452.0},
               {'name': 'ptsrc1', 'SpectrumType': 'PowerLaw',
                'ra': 121.0, 'dec': 32.56,
                'Index': {'value': 2.3, 'scale': 2.4, 'min': 0.3, 'max': 5.7},
                'Prefactor': {'value': 1.2, 'scale': 3E-12, 'min': 0.0, 'max': 5.0},
                'Scale': {'value': 134.1, 'scale': 1.2, 'min': 0.0, 'max': 2000.0},
                }]

    roi = ROIModel(sources=sources)
    src = roi['ptsrc0']

    assert_allclose(src['dec'], sources[0]['dec'])
    assert_allclose(src['ra'], sources[0]['ra'])

    sp = src['spectral_pars']

    assert_allclose(sp['Index']['value'], sources[0]['Index'])
    assert_allclose(sp['Scale']['value'], 1.452)
    assert_allclose(sp['Prefactor']['value'], 1.3)

    for p in ['Index', 'Prefactor', 'Scale']:
        if p == 'Index':
            assert_allclose(sources[0][p], -src.params[p]['value'])
        else:
            assert_allclose(sources[0][p], src.params[p]['value'])

    src = roi['ptsrc1']
    sp = src['spectral_pars']
    for p in ['Index', 'Prefactor', 'Scale']:
        assert_allclose(sources[1][p]['value'] * sources[1][p]['scale'],
                        src.params[p]['value'])


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
    ElementTree.ElementTree(root).write(xmlfile)

    roi = ROIModel(config={'catalogs': [xmlfile]})

    src = roi['3FGL J1649.4+5238']

    assert_allclose(src['dec'], values['ptsrc_dec'])
    assert_allclose(src['ra'], values['ptsrc_ra'])
    assert (src['SpectrumType'] == 'PowerLaw')
    assert (src['SpatialType'] == 'SkyDirFunction')
    assert (src['SpatialModel'] == 'PointSource')
    assert (src['SourceType'] == 'PointSource')

    sp = src['spectral_pars']

    attribs = ['value', 'scale', 'min', 'max']

    for x in attribs:
        assert_allclose(sp['Index'][x], values['ptsrc_index_%s' % x])

    for x in attribs:
        assert_allclose(sp['Prefactor'][x], values['ptsrc_prefactor_%s' % x])

    for x in attribs:
        assert_allclose(sp['Scale'][x], values['ptsrc_scale_%s' % x])

    assert_allclose(src.params['Prefactor']['value'],
                    values['ptsrc_prefactor_value'] * values['ptsrc_prefactor_scale'])
    assert_allclose(src.params['Index']['value'],
                    values['ptsrc_index_value'] * values['ptsrc_index_scale'])
    assert_allclose(src.params['Scale']['value'],
                    values['ptsrc_scale_value'] * values['ptsrc_scale_scale'])

    src = roi['galdiff']
    assert (src['SpatialType'] == 'MapCubeFunction')
    assert (src['SpatialModel'] == 'MapCubeFunction')

    src = roi['isodiff']
    assert (src['SpatialType'] == 'ConstantValue')
    assert (src['SpatialModel'] == 'ConstantValue')


def test_load_composite_source_from_xml(tmppath):
    values = {'ptsrc_dec': 52.6356, 'ptsrc_ra': 252.367,
              'ptsrc_Index_value': 2.21879, 'ptsrc_Index_scale': 1.2,
              'ptsrc_Index_min': 1.3, 'ptsrc_Index_max': 3.5,
              'ptsrc_Prefactor_value': 0.727, 'ptsrc_Prefactor_scale': 1e-12,
              'ptsrc_Prefactor_min': 0.01, 'ptsrc_Prefactor_max': 10.0,
              'ptsrc_Scale_value': 1.3, 'ptsrc_Scale_scale': 1e3,
              'ptsrc_Scale_min': 0.01, 'ptsrc_Scale_max': 113.0}

    xmlmodel = """
    <source_library title="source_library">
    <source name="CompositeSource" type="CompositeSource">
    <spectrum type="ConstantValue">
    <parameter free="1" max="3.402823466e+38" min="-3.402823466e+38" name="Value" scale="1" value="1" />
    </spectrum>
    <source_library>
    <source name="SourceA" type="PointSource">
    <spatialModel type="SkyDirFunction">
    <parameter free="0" max="90.0" min="-90.0" name="DEC" scale="1.0" value="{ptsrc_dec}"/>
    <parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="{ptsrc_ra}"/>
    </spatialModel>
    <spectrum type="PowerLaw">
    <parameter free="0" max="{ptsrc_Index_max}" min="{ptsrc_Index_min}" name="Index" scale="{ptsrc_Index_scale}" value="{ptsrc_Index_value}"/>
    <parameter free="0" max="{ptsrc_Scale_max}" min="{ptsrc_Scale_min}" name="Scale" scale="{ptsrc_Scale_scale}" value="{ptsrc_Scale_value}"/>
    <parameter free="0" max="{ptsrc_Prefactor_max}" min="{ptsrc_Prefactor_min}" name="Prefactor" scale="{ptsrc_Prefactor_scale}" value="{ptsrc_Prefactor_value}"/>
    </spectrum>
    </source>
    <source name="SourceB" type="PointSource">
    <spatialModel type="SkyDirFunction">
    <parameter free="0" max="90.0" min="-90.0" name="DEC" scale="1.0" value="{ptsrc_dec}"/>
    <parameter free="0" max="360.0" min="-360.0" name="RA" scale="1.0" value="{ptsrc_ra}"/>
    </spatialModel>
    <spectrum type="PowerLaw">
    <parameter free="0" max="{ptsrc_Index_max}" min="{ptsrc_Index_min}" name="Index" scale="{ptsrc_Index_scale}" value="{ptsrc_Index_value}"/>
    <parameter free="0" max="{ptsrc_Scale_max}" min="{ptsrc_Scale_min}" name="Scale" scale="{ptsrc_Scale_scale}" value="{ptsrc_Scale_value}"/>
    <parameter free="0" max="{ptsrc_Prefactor_max}" min="{ptsrc_Prefactor_min}" name="Prefactor" scale="{ptsrc_Prefactor_scale}" value="{ptsrc_Prefactor_value}"/>
    </spectrum>
    </source>
    </source_library>
    </source>
    </source_library>
    """.format(**values)

    root = ElementTree.fromstring(xmlmodel)
    xmlfile = str(tmppath.join('test.xml'))
    ElementTree.ElementTree(root).write(xmlfile)

    roi = ROIModel(config={'catalogs': [xmlfile]})

    src = roi['CompositeSource']

    assert(type(src) == roi_model.CompositeSource)
    assert (src['SpectrumType'] == 'ConstantValue')
    assert (src['SpatialType'] == 'CompositeSource')
    assert (src['SpatialModel'] == 'CompositeSource')
    assert (src['SourceType'] == 'CompositeSource')

    attribs = ['value', 'scale', 'min', 'max']
    par_names = ['Index', 'Prefactor', 'Scale']

    for s in src.nested_sources:
        pars = s.spectral_pars
        for par_name in par_names:
            for x in attribs:
                assert_allclose(pars[par_name][x], values[
                                'ptsrc_%s_%s' % (par_name, x)])


def test_create_source_from_dict(tmppath):
    ra = 252.367
    dec = 52.6356

    src = Source.create_from_dict({'name': 'testsrc',
                                   'SpatialModel': 'PointSource',
                                   'SpectrumType': 'PowerLaw',
                                   'Index': 2.3,
                                   'Prefactor': 1.3E-9,
                                   'ra': ra, 'dec': dec},
                                  rescale=True)

    assert_allclose(src['ra'], ra)
    assert_allclose(src['dec'], dec)
    assert_allclose(src.spectral_pars['Prefactor']['value'], 1.3)
    assert_allclose(src.spectral_pars['Prefactor']['scale'], 1E-9)
    assert_allclose(src.spectral_pars['Index']['value'], 2.3)
    assert src['SpatialModel'] == 'PointSource'
    assert src['SpatialType'] == 'SkyDirFunction'
    assert src['SourceType'] == 'PointSource'
    assert src.extended is False
    assert 'Prefactor' not in src.data
    assert 'Index' not in src.data

    src = Source.create_from_dict({'name': 'testsrc',
                                   'SpatialModel': 'PointSource',
                                   'SpectrumType': 'PowerLaw',                                   
                                   'Index': 2.3,
                                   'Prefactor': {'value' : 1.3, 'scale' : 1E-8,
                                                 'min' : 0.15, 'max' : 10.0,
                                                 'free' : False},
                                   'ra': ra, 'dec': dec},
                                  rescale=True)

    assert_allclose(src.spectral_pars['Prefactor']['value'], 1.3)
    assert_allclose(src.spectral_pars['Prefactor']['scale'], 1E-8)
    assert_allclose(src.spectral_pars['Prefactor']['min'], 0.15)
    assert_allclose(src.spectral_pars['Prefactor']['max'], 10.0)
    assert src.spectral_pars['Prefactor']['free'] is False
    
    
    src = Source.create_from_dict({'name': 'testsrc',
                                   'SpatialModel': 'RadialGaussian',
                                   'SpectrumType': 'PowerLaw',
                                   'Index': 2.3,
                                   'ra': ra, 'dec': dec})

    assert_allclose(src['ra'], ra)
    assert_allclose(src['dec'], dec)
    assert src['SpatialModel'] == 'RadialGaussian'
    assert (src['SpatialType'] == 'SpatialMap') or (src['SpatialType'] == 'RadialGaussian')
    assert src['SourceType'] == 'DiffuseSource'
    assert src.extended is True

    #src = Source.create_from_dict({'name': 'testsrc',
    #                               'SpatialModel': 'RadialGaussian',
    #                               'SpectrumType': 'PowerLaw',
    #                               'Index': 2.3, 'Sigma': 0.5,
    #                               'ra': ra, 'dec': dec})

    #assert_allclose(src['ra'], ra)
    #assert_allclose(src['dec'], dec)
    #assert_allclose(src['SpatialWidth'], 0.5*1.5095921854516636)
    #if src['SpatialType'] == 'RadialGaussian':
    #    assert_allclose(src.spatial_pars['Sigma']['value'], 0.5)

    src = Source.create_from_dict({'name': 'testsrc',
                                   'SpatialModel': 'RadialDisk',
                                   'SpectrumType': 'PowerLaw',
                                   'Index': 2.3, 'SpatialWidth': 0.5,
                                   'ra': ra, 'dec': dec})

    assert_allclose(src['ra'], ra)
    assert_allclose(src['dec'], dec)
    assert_allclose(src['SpatialWidth'], 0.5)
    if src['SpatialType'] == 'RadialDisk':
        assert_allclose(src.spatial_pars['Radius']['value'],
                        0.5 / 0.8246211251235321)


def test_create_point_source(tmppath):
    ra = 252.367
    dec = 52.6356
    prefactor = 2.3

    src_dict = {'SpatialModel': 'PointSource',
                'SpectrumType': 'PowerLaw',
                'ra': ra, 'dec': dec,
                'spectral_pars' : {'Prefactor': {'value' : prefactor} } }
    src = Source('testsrc', src_dict)

    assert_allclose(src['ra'], ra)
    assert_allclose(src['dec'], dec)
    assert_allclose(src.spatial_pars['RA']['value'], ra)
    assert_allclose(src.spatial_pars['DEC']['value'], dec)
    assert_allclose(src.spectral_pars['Prefactor']['value'], prefactor)
    assert src['SpatialModel'] == 'PointSource'
    assert src['SpatialType'] == 'SkyDirFunction'
    assert src['SpectrumType'] == 'PowerLaw'

    
def test_create_gaussian_source(tmppath):
    ra = 252.367
    dec = 52.6356
    sigma = 0.5

    src_dict = {'SpatialModel': 'RadialGaussian',
                'ra': ra, 'dec': dec, 'spatial_pars' : {'Sigma': {'value' : sigma} } }
    src = Source('testsrc', src_dict)

    assert_allclose(src['ra'], ra)
    assert_allclose(src['dec'], dec)
    if src['SpatialType'] == 'RadialGaussian':
        assert_allclose(src.spatial_pars['RA']['value'], ra)
        assert_allclose(src.spatial_pars['DEC']['value'], dec)
        assert_allclose(src.spatial_pars['Sigma']['value'], sigma)
    assert src['SpatialModel'] == 'RadialGaussian'


def test_set_spatial_model(tmppath):
    ra = 252.367
    dec = 52.6356

    src_dict = {'SpatialModel': 'RadialGaussian', 'ra': ra, 'dec': dec}
    src = Source('testsrc', src_dict)

    src.set_spatial_model('PointSource',{'ra' : 1.0, 'dec' : 2.0})
    assert_allclose(src.spatial_pars['RA']['value'], 1.0)
    assert_allclose(src.spatial_pars['DEC']['value'], 2.0)
    assert_allclose(src['ra'], 1.0)
    assert_allclose(src['dec'], 2.0)    
    assert src['SpatialModel'] == 'PointSource'
    assert src['SpatialType'] == 'SkyDirFunction'
    assert src['SourceType'] == 'PointSource'

    src_dict = {'SpatialModel': 'RadialGaussian', 'ra': ra, 'dec': dec}
    src = Source('testsrc', src_dict)
    src.set_spatial_model('PointSource',{'RA' : {'value' : 1.0}, 'DEC' : {'value' : 2.0}})
    assert_allclose(src.spatial_pars['RA']['value'], 1.0)
    assert_allclose(src.spatial_pars['DEC']['value'], 2.0)
    assert_allclose(src['ra'], 1.0)
    assert_allclose(src['dec'], 2.0)

    src_dict = {'SpatialModel': 'RadialGaussian', 'RA': ra, 'DEC': dec}
    src = Source('testsrc', src_dict)
    src.set_spatial_model('PointSource',{'RA' : {'value' : 1.0}, 'DEC' : {'value' : 2.0}})
    assert_allclose(src.spatial_pars['RA']['value'], 1.0)
    assert_allclose(src.spatial_pars['DEC']['value'], 2.0)
    assert_allclose(src['ra'], 1.0)
    assert_allclose(src['dec'], 2.0)

    src_dict = {'SpatialModel': 'RadialGaussian', 'RA': ra, 'DEC': dec}
    src = Source('testsrc', src_dict)
    src.set_spatial_model('RadialDisk',{'ra' : 2.0, 'dec' : 3.0, 'Radius' : 1.0})

    if src['SpatialType'] == 'RadialDisk':
        assert_allclose(src.spatial_pars['RA']['value'], 2.0)
        assert_allclose(src.spatial_pars['DEC']['value'], 3.0)
        assert_allclose(src.spatial_pars['Radius']['value'], 1.0)
    assert_allclose(src['ra'], 2.0)
    assert_allclose(src['dec'], 3.0)    
    assert src['SpatialModel'] == 'RadialDisk'
    assert src['SourceType'] == 'DiffuseSource'

    src_dict = {'SpatialModel': 'RadialGaussian', 'RA': ra, 'DEC': dec}
    src = Source('testsrc', src_dict)
    src.set_spatial_model('RadialDisk',{'ra' : 2.0, 'dec' : 3.0, 'SpatialWidth' : 2.0})

    if src['SpatialType'] == 'RadialDisk':
        assert_allclose(src.spatial_pars['RA']['value'], 2.0)
        assert_allclose(src.spatial_pars['DEC']['value'], 3.0)
    assert_allclose(src['ra'], 2.0)
    assert_allclose(src['dec'], 3.0)
    assert_allclose(src['SpatialWidth'], 2.0)
    assert src['SpatialModel'] == 'RadialDisk'
    assert src['SourceType'] == 'DiffuseSource'
    
