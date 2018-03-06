# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os

from fermipy import PACKAGE_ROOT
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse.diffuse_src_manager import make_ring_dicts, make_diffuse_comp_info_dict
from fermipy.diffuse.model_manager import make_library


def test_catalog_src_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    library = os.path.join(basedir, 'models', 'library.yaml')
    ret_dict = make_catalog_comp_dict(library=library, basedir=basedir)

    # spot check results

    # Test the dictionary of catalogs
    assert(ret_dict['catalog_info_dict']['3FGL'].catalog_name == '3FGL')
    assert(ret_dict['catalog_info_dict']['3FGL'].catalog_type == '3FGL')
    assert(len(ret_dict['catalog_info_dict']['3FGL'].catalog.table) == 63)
    assert(len(ret_dict['catalog_info_dict'][
           '3FGL'].roi_model.sources) == 63)

    # Test the split dictionary
    assert(len(ret_dict['comp_info_dict']['3FGL_v00'].keys()) == 3)
    assert(len(ret_dict['comp_info_dict']['3FGL_v00']
               ['extended'].roi_model.sources) == 2)
    assert(len(ret_dict['comp_info_dict']['3FGL_v00']
               ['faint'].roi_model.sources) == 60)
    assert(len(ret_dict['comp_info_dict']['3FGL_v00']
               ['remain'].roi_model.sources) == 1)

    # Test the CatalogSourceManager
    assert(len(ret_dict['CatalogSourceManager'].splitkeys()) == 1)
    assert(len(ret_dict['CatalogSourceManager'].catalogs()) == 1)


def test_galprop_ring_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    kwargs = dict(basedir=basedir,
                  library=os.path.join(basedir, 'models', 'library.yaml'),
                  comp=os.path.join(basedir, 'binning.yaml'))
    gmm = make_ring_dicts(**kwargs)

    # spot check results
    assert(len(gmm.galkeys()) == 1)
    assert(gmm.galkeys()[0] == 'ref')
    assert(len(gmm.ring_dict('ref')) == 21)

    assert(gmm.ring_dict('ref')['merged_CO_5_ref'].ring == 5)
    assert(gmm.ring_dict('ref')['merged_CO_5_ref'].source_name == 'merged_CO')
    assert(len(gmm.ring_dict('ref')['merged_CO_5_ref'].files) == 4)

    assert(gmm.ring_dict('ref')['merged_IC_1_ref'].ring == 1)
    assert(gmm.ring_dict('ref')['merged_IC_1_ref'].source_name == 'merged_IC')
    assert(len(gmm.ring_dict('ref')['merged_IC_1_ref'].files) == 12)


def test_diffuse_src_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    kwargs = dict(basedir=basedir,
                  library=os.path.join(basedir, 'models', 'library.yaml'),
                  comp=os.path.join(basedir, 'binning.yaml'))
    ret_dict = make_diffuse_comp_info_dict(**kwargs)

    # spot check results
    assert(len(ret_dict['comp_info_dict'].keys()) == 29)

    # sun
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].model_type == 'MapCubeSource')
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].moving)
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].selection_dependent is False)
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].source_name == 'sun-ic')
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].sourcekey == 'sun-ic_v2r0')
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].source_ver == 'v2r0')

    assert(len(ret_dict['comp_info_dict']['sun-ic_v2r0'].components) == 4)
    assert(ret_dict['comp_info_dict']['sun-ic_v2r0'].components[
           'zmax100'].comp_key == 'zmax100')


    # isotropic
    assert(ret_dict['comp_info_dict'][
           'isotropic_P8R3_SOURCE_V2'].model_type == 'IsoSource')
    assert(ret_dict['comp_info_dict']['isotropic_P8R3_SOURCE_V2'].moving is False)
    assert(ret_dict['comp_info_dict'][
           'isotropic_P8R3_SOURCE_V2'].selection_dependent is False)
    assert(ret_dict['comp_info_dict'][
           'isotropic_P8R3_SOURCE_V2'].source_name == 'isotropic')
    assert(ret_dict['comp_info_dict'][
           'isotropic_P8R3_SOURCE_V2'].sourcekey == 'isotropic_P8R3_SOURCE_V2')
    assert(ret_dict['comp_info_dict']['isotropic_P8R3_SOURCE_V2'].source_ver == 'P8R3_SOURCE_V2')
    assert(ret_dict['comp_info_dict']['isotropic_P8R3_SOURCE_V2'].components is None)

    # galprop ring
    assert(ret_dict['comp_info_dict'][
           'merged_HI_5_ref'].model_type == 'MapCubeSource')
    assert(ret_dict['comp_info_dict']['merged_HI_5_ref'].moving is False)
    assert(ret_dict['comp_info_dict'][
           'merged_HI_5_ref'].selection_dependent is False)
    assert(ret_dict['comp_info_dict'][
           'merged_HI_5_ref'].source_name == 'merged_HI_5')
    assert(ret_dict['comp_info_dict'][
           'merged_HI_5_ref'].sourcekey == 'merged_HI_5_ref')
    assert(ret_dict['comp_info_dict']['merged_HI_5_ref'].source_ver == 'ref')
    assert(ret_dict['comp_info_dict']['merged_HI_5_ref'].components is None)

    assert(len(ret_dict['DiffuseModelManager'].sourcekeys()) == 8)


def test_model_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    kwargs = dict(basedir=basedir,
                  library=os.path.join(basedir, 'models', 'library.yaml'),
                  comp=os.path.join(basedir, 'binning.yaml'))
    ret_dict = make_library(**kwargs)

    # spot check results
    assert(len(ret_dict['model_comp_dict'].keys()) == 30)
 
   # sun
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].model_type == 'MapCubeSource')
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].moving)
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].selection_dependent is False)
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].source_name == 'sun-ic')
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].sourcekey == 'sun-ic_v2r0')
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].source_ver == 'v2r0')

    assert(len(ret_dict['model_comp_dict']['sun-ic_v2r0'].components) == 4)
    assert(ret_dict['model_comp_dict']['sun-ic_v2r0'].components[
           'zmax100'].comp_key == 'zmax100')

    # isotropic
    assert(ret_dict['model_comp_dict'][
           'isotropic_P8R3_SOURCE_V2'].model_type == 'IsoSource')
    assert(ret_dict['model_comp_dict']['isotropic_P8R3_SOURCE_V2'].moving is False)
    assert(ret_dict['model_comp_dict'][
           'isotropic_P8R3_SOURCE_V2'].selection_dependent is False)
    assert(ret_dict['model_comp_dict'][
           'isotropic_P8R3_SOURCE_V2'].source_name == 'isotropic')
    assert(ret_dict['model_comp_dict'][
           'isotropic_P8R3_SOURCE_V2'].sourcekey == 'isotropic_P8R3_SOURCE_V2')
    assert(ret_dict['model_comp_dict']['isotropic_P8R3_SOURCE_V2'].source_ver == 'P8R3_SOURCE_V2')
    assert(ret_dict['model_comp_dict']['isotropic_P8R3_SOURCE_V2'].components is None)

    # galprop ring
    assert(ret_dict['model_comp_dict'][
           'merged_HI_5_ref'].model_type == 'MapCubeSource')
    assert(ret_dict['model_comp_dict']['merged_HI_5_ref'].moving is False)
    assert(ret_dict['model_comp_dict'][
           'merged_HI_5_ref'].selection_dependent is False)
    assert(ret_dict['model_comp_dict'][
           'merged_HI_5_ref'].source_name == 'merged_HI_5')
    assert(ret_dict['model_comp_dict'][
           'merged_HI_5_ref'].sourcekey == 'merged_HI_5_ref')
    assert(ret_dict['model_comp_dict']['merged_HI_5_ref'].source_ver == 'ref')
    assert(ret_dict['model_comp_dict']['merged_HI_5_ref'].components is None)

    model_manager = ret_dict['ModelManager']

    baseline_model = model_manager.make_model_info('baseline')
    assert(len(baseline_model.model_components) == 30)

