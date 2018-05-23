# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Handle the naming conventions for DM pipeline analysis
"""
from __future__ import absolute_import, division, print_function

import sys
import yaml

from fermipy.jobs.utils import is_null, is_not_null

class NameFactory(object):
    """ Helper class to define file names and keys consistently. """

    # Input configuration file
    ttypeconfig_format = 'config/config_{target_type}.yaml'

    # Random target direction configruation
    randconfig_format = 'config/random_{target_type}.yaml'

    # target keys, these are how we specify various files associated with
    # particular targets

    # Directory for a particular target
    targetdir_format = '{target_type}/{target_name}'

    # Directory for simulations for a particular target
    sim_targetdir_format = '{target_type}_sim/sim_{sim_name}/{target_name}'

    # Targetlist file format
    targetfile_format = '{target_type}/{targetlist}'

    # Roster file format
    sim_targetfile_format = '{target_type}_sim/sim_{sim_name}/{targetlist}'

    # Information about a particular target profile
    profilefile_format = '{target_type}/{target_name}/profile_{profile}.yaml'

    # Information about a particular target profile
    sim_profilefile_format = '{target_type}_sim/sim_{sim_name}/{target_name}/profile_{profile}.yaml'

    # SED file for a particular target
    sedfile_format = '{target_type}/{target_name}/sed_{profile}.fits'

    # Simulated SED file for a particular target
    sim_sedfile_format = '{target_type}_sim/sim_{sim_name}/{target_name}/sed_{profile}_{seed}.fits'

    # Stamp files from scatter gather jobs
    stamp_format = 'stamps/{linkname}.stamp'

    # Full filepath
    fullpath_format = '{basedir}/{localpath}'

    def __init__(self, **kwargs):
        """C'tor.  Set baseline dictionary used to resolve names
        """
        self.base_dict = kwargs.copy()

    def update_base_dict(self, yamlfile):
        """Update the values in baseline dictionary used to resolve names
        """
        self.base_dict.update(**yaml.safe_load(open(yamlfile)))

    def _format_from_dict(self, format_string, **kwargs):
        """Return a formatted file name dictionary components """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        localpath = format_string.format(**kwargs_copy)
        if kwargs.get('fullpath', False):
            return self.fullpath(localpath=localpath)
        return localpath

    def ttypeconfig(self, **kwargs):
        """Return the name of the input configuration file
        """
        return self._format_from_dict(NameFactory.ttypeconfig_format, **kwargs)

    def randconfig(self, **kwargs):
        """Return the name of the random direction configuration file
        """
        return self._format_from_dict(NameFactory.randconfig_format, **kwargs)

    def targetdir(self, **kwargs):
        """Return the name for the directory for a particular target
        """
        return self._format_from_dict(NameFactory.targetdir_format, **kwargs)

    def sim_targetdir(self, **kwargs):
        """Return the name for the directory for a particular target
        """
        return self._format_from_dict(NameFactory.sim_targetdir_format, **kwargs)

    def targetfile(self, **kwargs):
        """Return the name for the Target list file
        """
        return self._format_from_dict(NameFactory.targetfile_format, **kwargs)

    def sim_targetfile(self, **kwargs):
        """Return the name for the Target list file for simulation
        """
        return self._format_from_dict(NameFactory.sim_targetfile_format, **kwargs)

    def profilefile(self, **kwargs):
        """Return the name of the yaml file with information about a partiuclar profile
        """
        return self._format_from_dict(NameFactory.profilefile_format, **kwargs)

    def sim_profilefile(self, **kwargs):
        """Return the name of the yaml file with information about a partiuclar profile
        """
        return self._format_from_dict(NameFactory.sim_profilefile_format, **kwargs)

    def sedfile(self, **kwargs):
        """Return the name for the SED file for a particular target
        """
        return self._format_from_dict(NameFactory.sedfile_format, **kwargs)

    def sim_sedfile(self, **kwargs):
        """Return the name for the simulated SED file for a particular target
        """
        if 'seed' not in kwargs:
            kwargs['seed'] = 'SEED'
        return self._format_from_dict(NameFactory.sim_sedfile_format, **kwargs)

    def stamp(self, **kwargs):
        """Return the path for a stamp file for a scatter gather job"""
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        return NameFactory.stamp_format.format(**kwargs_copy)

    def fullpath(self, **kwargs):
        """Return a full path name for a given file
        """
        kwargs_copy = self.base_dict.copy()
        kwargs_copy.update(**kwargs)
        return NameFactory.fullpath_format.format(**kwargs_copy)

    def resolve_targetfile(self, args, require_sim_name=False):  # x
        """Get the name of the targetfile based on the job arguments"""
        ttype = args.get('ttype')
        if is_null(ttype):
            sys.stderr.write('Target type must be specified')
            return (None, None)

        sim = args.get('sim')
        if is_null(sim):
            if require_sim_name:
                sys.stderr.write('Simulation scenario must be specified')
                return (None, None)
            else:
                sim = None

        name_keys = dict(target_type=ttype,
                         targetlist='target_list.yaml',
                         sim_name=sim,
                         fullpath=True)
        if sim is None:
            targetfile = self.targetfile(**name_keys)
        else:
            targetfile = self.sim_targetfile(**name_keys)

        targets_override = args.get('targetfile')
        if is_not_null(targets_override):
            targetfile = targets_override

        return (targetfile, sim)

    def resolve_randconfig(self, args):
        """Get the name of the specturm file based on the job arguments"""
        ttype = args.get('ttype')
        if is_null(ttype):
            sys.stderr.write('Target type must be specified')
            return None
        name_keys = dict(target_type=ttype,
                         fullpath=True)
        randconfig = self.randconfig(**name_keys)
        rand_override = args.get('rand_config')
        if is_not_null(rand_override):
            randconfig = rand_override
        return randconfig
