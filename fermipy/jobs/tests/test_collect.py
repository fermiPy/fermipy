# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function


def test_collect_link_classes():
    """ Test that we can create `Link` classes """
    from fermipy.jobs.target_collect import CollectSED
    CollectSED.create()


def test_collect_sg_classes():
    """ Test that we can create `ScatterGather` classes """
    from fermipy.jobs.target_collect import CollectSED_SG
    CollectSED_SG.create()


if __name__ == '__main__':
    test_collect_link_classes()
    test_collect_sg_classes()
