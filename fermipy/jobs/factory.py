#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A factory class to build links
"""
from __future__ import absolute_import, division, print_function


class LinkFactory(object):
    """Static Factory class used by build `Link` objects.

    The `Link` objects are registerd and accessed by
    their appname data member
    """

    _class_dict = {}

    @staticmethod
    def register(appname, cls):
        """Register a class with this factory """
        LinkFactory._class_dict[appname] = cls

    @staticmethod
    def create(appname, **kwargs):
        """Create a `Link` of a particular class, using the kwargs as options"""

        if appname in LinkFactory._class_dict:
            return LinkFactory._class_dict[appname].create(**kwargs)
        else:
            raise KeyError(
                "Could not create object associated to app %s" % appname)
