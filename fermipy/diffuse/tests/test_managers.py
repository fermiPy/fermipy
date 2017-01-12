# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os

from fermipy import PACKAGE_ROOT
from fermipy.diffuse.catalog_src_manager import make_catalog_comp_dict
from fermipy.diffuse.diffuse_src_manager import make_ring_dicts, make_diffuse_comp_info_dict
from fermipy.diffuse.model_manager import make_library


def test_catalog_src_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    sources = os.path.join(basedir, 'catalog_components.yaml')
    ret_dict = make_catalog_comp_dict(sources=sources, basedir=basedir)

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
                  diffuse=os.path.join(basedir, 'diffuse_components.yaml'),
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
                  diffuse=os.path.join(basedir, 'diffuse_components.yaml'),
                  comp=os.path.join(basedir, 'binning.yaml'))
    ret_dict = make_diffuse_comp_info_dict(**kwargs)

    # spot check results
    assert(len(ret_dict['comp_info_dict'].keys()) == 28)

    # sun
    assert(ret_dict['comp_info_dict']['sun_v00'].model_type == 'MapCubeSource')
    assert(ret_dict['comp_info_dict']['sun_v00'].moving)
    assert(ret_dict['comp_info_dict']['sun_v00'].selection_dependent is False)
    assert(ret_dict['comp_info_dict']['sun_v00'].source_name == 'sun')
    assert(ret_dict['comp_info_dict']['sun_v00'].sourcekey == 'sun_v00')
    assert(ret_dict['comp_info_dict']['sun_v00'].source_ver == 'v00')

    assert(len(ret_dict['comp_info_dict']['sun_v00'].components) == 4)
    assert(ret_dict['comp_info_dict']['sun_v00'].components[
           'zmax100'].comp_key == 'zmax100')

    # residual cr
    assert(ret_dict['comp_info_dict'][
           'residual_cr_v00'].model_type == 'MapCubeSource')
    assert(ret_dict['comp_info_dict']['residual_cr_v00'].moving is False)
    assert(ret_dict['comp_info_dict']['residual_cr_v00'].selection_dependent)
    assert(ret_dict['comp_info_dict'][
           'residual_cr_v00'].source_name == 'residual_cr')
    assert(ret_dict['comp_info_dict'][
           'residual_cr_v00'].sourcekey == 'residual_cr_v00')
    assert(ret_dict['comp_info_dict']['residual_cr_v00'].source_ver == 'v00')

    assert(len(ret_dict['comp_info_dict']['residual_cr_v00'].components) == 10)
    assert(ret_dict['comp_info_dict']['residual_cr_v00'].components[
           'E0_PSF3'].comp_key == 'E0_PSF3')

    # isotropic
    assert(ret_dict['comp_info_dict'][
           'isotropic_v00'].model_type == 'IsoSource')
    assert(ret_dict['comp_info_dict']['isotropic_v00'].moving is False)
    assert(ret_dict['comp_info_dict'][
           'isotropic_v00'].selection_dependent is False)
    assert(ret_dict['comp_info_dict'][
           'isotropic_v00'].source_name == 'isotropic')
    assert(ret_dict['comp_info_dict'][
           'isotropic_v00'].sourcekey == 'isotropic_v00')
    assert(ret_dict['comp_info_dict']['isotropic_v00'].source_ver == 'v00')
    assert(ret_dict['comp_info_dict']['isotropic_v00'].components is None)

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

    assert(len(ret_dict['DiffuseModelManager'].sourcekeys()) == 7)


def test_model_manager():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    kwargs = dict(basedir=basedir,
                  sources=os.path.join(basedir, 'catalog_components.yaml'),
                  diffuse=os.path.join(basedir, 'diffuse_components.yaml'),
                  comp=os.path.join(basedir, 'binning.yaml'))
    ret_dict = make_library(**kwargs)

    # spot check results
    assert(len(ret_dict['model_comp_dict'].keys()) == 29)

    # sun
    assert(ret_dict['model_comp_dict'][
           'sun_v00'].model_type == 'MapCubeSource')
    assert(ret_dict['model_comp_dict']['sun_v00'].moving)
    assert(ret_dict['model_comp_dict']['sun_v00'].selection_dependent is False)
    assert(ret_dict['model_comp_dict']['sun_v00'].source_name == 'sun')
    assert(ret_dict['model_comp_dict']['sun_v00'].sourcekey == 'sun_v00')
    assert(ret_dict['model_comp_dict']['sun_v00'].source_ver == 'v00')

    assert(len(ret_dict['model_comp_dict']['sun_v00'].components) == 4)
    assert(ret_dict['model_comp_dict']['sun_v00'].components[
           'zmax100'].comp_key == 'zmax100')

    # residual cr
    assert(ret_dict['model_comp_dict'][
           'residual_cr_v00'].model_type == 'MapCubeSource')
    assert(ret_dict['model_comp_dict']['residual_cr_v00'].moving is False)
    assert(ret_dict['model_comp_dict']['residual_cr_v00'].selection_dependent)
    assert(ret_dict['model_comp_dict'][
           'residual_cr_v00'].source_name == 'residual_cr')
    assert(ret_dict['model_comp_dict'][
           'residual_cr_v00'].sourcekey == 'residual_cr_v00')
    assert(ret_dict['model_comp_dict']['residual_cr_v00'].source_ver == 'v00')

    assert(len(ret_dict['model_comp_dict'][
           'residual_cr_v00'].components) == 10)
    assert(ret_dict['model_comp_dict']['residual_cr_v00'].components[
           'E0_PSF3'].comp_key == 'E0_PSF3')

    # isotropic
    assert(ret_dict['model_comp_dict'][
           'isotropic_v00'].model_type == 'IsoSource')
    assert(ret_dict['model_comp_dict']['isotropic_v00'].moving is False)
    assert(ret_dict['model_comp_dict'][
           'isotropic_v00'].selection_dependent is False)
    assert(ret_dict['model_comp_dict'][
           'isotropic_v00'].source_name == 'isotropic')
    assert(ret_dict['model_comp_dict'][
           'isotropic_v00'].sourcekey == 'isotropic_v00')
    assert(ret_dict['model_comp_dict']['isotropic_v00'].source_ver == 'v00')
    assert(ret_dict['model_comp_dict']['isotropic_v00'].components is None)

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
    assert(len(baseline_model.model_components) == 29)
