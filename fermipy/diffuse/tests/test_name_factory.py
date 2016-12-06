# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.diffuse.name_policy import NameFactory


def test_name_factory():
    names = dict(data_pass='P8',
                 data_ver='P302',
                 evclass='source',
                 data_time='8years',
                 zcut='zmax105',
                 psftype='PSF3',
                 ebin='E3',
                 coordsys='GAL',
                 irf_ver='V6',
                 sourcekey='ptsrc')

    name_fact = NameFactory(**names)

    test_dict = name_fact.make_filenames(**names)
    test_irfs = name_fact.irfs(**names)

    assert(test_dict['ltcube'] == 'lt_cubes/ltcube_8years_zmax105.fits')
    assert(test_dict['ft1file'] == 'P8_P302_8years_source_zmax105.lst')
    assert(test_dict['bexpcube'] ==
           'bexp_cubes/bexcube_P8_P302_8years_source_zmax105_E3_PSF3_GAL_V6.fits')
    assert(test_dict[
           'srcmaps'] == 'srcmaps/srcmaps_ptsrc_P8_P302_8years_source_zmax105_E3_PSF3_GAL_V6.fits')
    assert(test_dict[
           'ccube'] == 'counts_cubes/ccube_P8_P302_8years_source_zmax105_E3_PSF3_GAL.fits')
    assert(test_dict[
           'mcube'] == 'model_cubes/mcube_ptsrc_P8_P302_8years_source_zmax105_E3_PSF3_GAL_V6.fits')
    assert(test_irfs == 'P8R2_SOURCE_V6')
