# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function

from astropy.tests.helper import pytest
from fermipy.tests.utils import requires_dependency

from fermipy.jobs.target_analysis import AnalyzeROI, AnalyzeSED
from fermipy.jobs.target_analysis import AnalyzeROI_SG, AnalyzeSED_SG

# Skip tests in this file if Fermi ST aren't available
pytestmark = requires_dependency('Fermi ST')

def test_analysis_link_classes():
    """ Test that we can create `Link` classes """    
    AnalyzeROI.create()
    AnalyzeSED.create()


def test_analysis_sg_classes():
    """ Test that we can create `ScatterGather` classes """    
    AnalyzeROI_SG.create()
    AnalyzeSED_SG.create()


if __name__ == '__main__':
    test_analysis_link_classes()
    test_analysis_sg_classes()
