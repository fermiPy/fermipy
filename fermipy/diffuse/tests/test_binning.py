# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from fermipy.diffuse.binning import Component


def test_binning():
    the_yaml = """
E0:
  log_emin : 1.5
  log_emax : 2.0
  enumbins : 4
  zmax : 80.
  psf_types :
    PSF3 : 
      hpx_order : 5
E1:
  log_emin : 2.0
  log_emax : 2.5
  enumbins : 4
  zmax : 90.
  psf_types :
    PSF2 : 
      hpx_order : 6
    PSF3 : 
      hpx_order : 6
E2:
  log_emin : 2.5
  log_emax : 3.0
  enumbins : 4
  zmax : 100.
  psf_types :
    PSF1 : 
      hpx_order : 6
    PSF2 : 
      hpx_order : 7
    PSF3 : 
      hpx_order : 8
E3:
  log_emin : 3.0
  log_emax : 6.0
  enumbins : 12
  zmax : 105.
  psf_types :
    PSF0 : 
      hpx_order : 7
    PSF1 : 
      hpx_order : 8
    PSF2 : 
      hpx_order : 9
    PSF3 : 
      hpx_order : 9
"""
    components = Component.build_from_yamlstr(the_yaml)

    assert(len(components) == 10)
    # spot check first and last components
    assert(components[0].log_emin == 2.0)
    assert(components[0].log_emax == 2.5)
    assert(components[0].enumbins == 4)
    assert(components[0].hpx_order == 6)
    assert(components[0].zmax == 90)

    assert(components[-1].log_emin == 2.5)
    assert(components[-1].log_emax == 3.0)
    assert(components[-1].enumbins == 4)
    assert(components[-1].hpx_order == 7)
    assert(components[-1].zmax == 100)
