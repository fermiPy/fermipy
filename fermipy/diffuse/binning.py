# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Small helper class to represent the binning used for a single component
of a summed likelihood in diffuse analysis
"""
from __future__ import absolute_import, division, print_function

import math
import yaml

# Map event type 'key' to evtype bin mask value
EVT_TYPE_DICT = dict(PSF0=4, PSF1=8, PSF2=16, PSF3=32,
                     PSF12=24, PSF012=28,
                     PSF23=48, PSF123=56, PSF0123=60)


class Component(object):
    """Small helper class to represent the binning used for a single component
    of a summed likelihood in diffuse analysis

    Parameters
    ----------

    log_emin : float
        Log base 10 of minimum energy for this component
    log_emax : float
        Log base 10 of maximum energy for this component
    enumbins : int
        Number of energy bins for this component
    zmax : float
        Maximum zenith angle cube for this component in degrees
    mktimefilters : list
        Filters for gtmktime.
    hpx_order : int
        HEALPix order to use for this component
    coordsys : str
        Coodinate system, 'CEL' or 'GAL'
    """

    def __init__(self, **kwargs):
        """C'tor: copies keyword arguments to data members
        """
        self.log_emin = kwargs['log_emin']
        self.log_emax = kwargs['log_emax']
        self.enumbins = kwargs['enumbins']
        self.zmax = kwargs['zmax']
        self.mktimefilters = kwargs.get('mktimefilters', [])
        self.evtclasses = kwargs.get('evtclasses', [])
        self.hpx_order = kwargs['hpx_order']
        self.evtype_name = kwargs['evtype_name']
        self.ebin_name = kwargs['ebin_name']
        self.coordsys = kwargs['coordsys']

    def __repr__(self):
        retstr = "Binning component %s_%s\n" % (self.ebin_name, self.evtype_name)
        retstr += "  log_10(E/MeV) : %.2f %.2f %i bins\n" % (self.log_emin,
                                                             self.log_emax, self.enumbins)
        retstr += "  HPX order     : %i\n" % self.hpx_order
        retstr += "  Coordsys      : %s\n" % self.coordsys
        retstr += "  Zenith cut    : %.0f\n" % self.zmax
        retstr += "  Mktime filters: %s\n" % str(self.mktimefilters)
        retstr += "  Evt classes   : %s" % str(self.evtclasses)
        return retstr

    def make_key(self, format_str):
        """ Make a key to identify this compoment

        format_str is formatted using object __dict__
        """
        return format_str.format(**self.__dict__)

    @property
    def emin(self):
        """ Minimum energy for this component """
        return math.pow(10., self.log_emin)

    @property
    def emax(self):
        """ Maximum energy for this component """
        return math.pow(10., self.log_emax)

    @property
    def evtype(self):
        """ Event type bit mask for this component """
        return EVT_TYPE_DICT[self.evtype_name]

    @classmethod
    def build_from_energy_dict(cls, ebin_name, input_dict):
        """ Build a list of components from a dictionary for a single energy range
        """
        psf_types = input_dict.pop('psf_types')
        output_list = []
        for psf_type, val_dict in sorted(psf_types.items()):
            fulldict = input_dict.copy()
            fulldict.update(val_dict)
            fulldict['evtype_name'] = psf_type
            fulldict['ebin_name'] = ebin_name
            component = cls(**fulldict)
            output_list += [component]
        return output_list

    @classmethod
    def build_from_yamlstr(cls, yamlstr):
        """ Build a list of components from a yaml string
        """
        top_dict = yaml.safe_load(yamlstr)
        coordsys = top_dict.pop('coordsys')
        output_list = []
        for e_key, e_dict in sorted(top_dict.items()):
            if e_key == 'coordsys':
                continue
            e_dict = top_dict[e_key]
            e_dict['coordsys'] = coordsys
            output_list += cls.build_from_energy_dict(e_key, e_dict)
        return output_list

    @classmethod
    def build_from_yamlfile(cls, yamlfile):
        """ Build a list of components from a yaml file
        """
        return cls.build_from_yamlstr(open(yamlfile))
