# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities that manage spectral model specific to diffuse analyses
"""
from __future__ import absolute_import, division, print_function

import yaml


class SpectralLibrary(object):
    """ A small helper class that serves as an alias dictionary for spectral models
    """

    def __init__(self, spectral_dict):
        """C'tor, loads the dictionary
        """
        self.spectral_dict = spectral_dict

    def update(self, spectral_dict):
        """Update the dictionary """
        self.spectral_dict.update(spectral_dict)

    def __getitem__(self, key):
        """Get an item from the dictionary """
        return self.spectral_dict.get(key, {})

    @classmethod
    def create_from_yamlstr(cls, yamlstr):
        """Create the dictionary for a yaml file
        """
        spectral_dict = yaml.load(yamlstr)
        return cls(spectral_dict)

    @classmethod
    def create_from_yaml(cls, yamlfile):
        """Create the dictionary for a yaml file
        """
        return cls.create_from_yamlstr(open(yamlfile))
