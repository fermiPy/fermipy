#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A factory class to build links
"""
from __future__ import absolute_import, division, print_function

class LinkFactory(object):
    _class_dict = {}

    @staticmethod
    def register(appname, cls):
        LinkFactory._class_dict[appname] = cls

    @staticmethod
    def create(appname, **kwargs):
        if LinkFactory._class_dict.has_key(appname):
            return LinkFactory._class_dict[appname].create(**kwargs)
        else:
            raise KeyError("Could not create object associated to app %s"%appname)

    
