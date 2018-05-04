# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function


def test_sim_link_classes():
    """ Test that we can create `Link` classes """
    from fermipy.jobs.target_sim import CopyBaseROI, RandomDirGen, SimulateROI
    CopyBaseROI.create()
    RandomDirGen.create()
    SimulateROI.create()


def test_sim_sg_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.jobs.target_sim import CopyBaseROI_SG, RandomDirGen_SG
    CopyBaseROI_SG.create()
    RandomDirGen_SG.create()


if __name__ == '__main__':
    test_sim_link_classes()
    test_sim_sg_classes()
