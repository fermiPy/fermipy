# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

import os

from fermipy import PACKAGE_ROOT
from fermipy.diffuse.binning import Component


def test_binning():
    basedir = os.path.join(PACKAGE_ROOT, 'diffuse', 'tests', 'data')
    the_yaml = os.path.join(basedir, 'binning.yaml')

    components = Component.build_from_yamlfile(the_yaml)

    assert(len(components) == 10)
    # spot check first and last components

    assert(components[0].log_emin == 1.5)
    assert(components[0].log_emax == 2.0)
    assert(components[0].enumbins == 4)
    assert(components[0].hpx_order == 5)
    assert(components[0].zmax == 80)

    assert(components[-1].log_emin == 3.0)
    assert(components[-1].log_emax == 6.0)
    assert(components[-1].enumbins == 12)
    assert(components[-1].hpx_order == 9)
    assert(components[-1].zmax == 105)
