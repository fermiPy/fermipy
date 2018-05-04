# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function


def test_analysis_link_classes():
    """ Test that we can create `Link` classes """
    from fermipy.jobs.target_analysis import AnalyzeROI, AnalyzeSED
    AnalyzeROI.create()
    AnalyzeSED.create()


def test_analysis_sg_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.jobs.target_analysis import AnalyzeROI_SG, AnalyzeSED_SG
    AnalyzeROI_SG.create()
    AnalyzeSED_SG.create()


if __name__ == '__main__':
    test_analysis_link_classes()
    test_analysis_sg_classes()
