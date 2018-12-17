# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Small helper class to represent the selection of mktime filters used in the analysis
"""
from __future__ import absolute_import, division, print_function

import yaml


class MktimeFilterDict(object):
    """Small helper class toselection of mktime filters used in the analysis
    """

    def __init__(self, aliases, selections):
        """C'tor: copies keyword arguments to data members
        """
        self.aliases = aliases
        self.selections = {}
        for k, v in selections.items():
            self.selections[k] = v.format(**self.aliases)

    def __getitem__(self, key):
        """Return a filter string by key """
        return self.selections[key]

    def keys(self):
        """Return the iterator over keys """
        return self.selections.keys()

    def items(self):
        """Return the itratetor over key, value pairs """
        return self.selections.items()

    def values(self):
        """Return the itratetor over values """
        return self.selections.values()

    @staticmethod
    def build_from_yamlfile(yamlfile):
        """ Build a list of components from a yaml file
        """
        d = yaml.load(open(yamlfile))
        return MktimeFilterDict(d['aliases'], d['selections'])
