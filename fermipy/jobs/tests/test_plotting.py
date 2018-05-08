# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function


def test_plotting_link_classes():
    """ Test that we can create `Link` classes """
    from fermipy.jobs.target_plotting import PlotCastro
    PlotCastro.create()


def test_plotting_sg_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.jobs.target_plotting import PlotCastro_SG
    PlotCastro_SG.create()

if __name__ == '__main__':
    test_plotting_link_classes()
    test_plotting_sg_classes()
