# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Handle the naming conventions for stacking pipeline analysis
"""
from __future__ import absolute_import, division, print_function

import sys

from fermipy.jobs.utils import is_null, is_not_null
from fermipy.jobs.name_policy import NameFactory as NameFactory_Base


class NameFactory(NameFactory_Base):
    """ Helper class to define file names and keys consistently. """

    # Stacking spectral file target configruation
    specconfig_format = 'config/stack_spectra_{target_type}.yaml'

    # Stacking spectral file
    specfile_format = 'stack_spectra_{target_type}.fits'

    # Stack variable likelilood file for a particular target (and astro-factor prior)
    stack_likefile_format = '{target_type}/{target_name}/stacklike_{profile}_{astro_prior}.fits'

    # Stack variable limits file for a particular target (and astro-factor prior)
    stack_limitsfile_format = '{target_type}/{target_name}/stacklimits_{profile}_{astro_prior}.fits'

    # Stacked variable limits file for a particular roster (and astro-factor prior)
    resultsfile_format = '{target_type}/stacked/results_{roster_name}_{astro_prior}.fits'

    # Stacked variable limits file for a particular roster (and astro-factor prior)
    stackedlimitsfile_format = '{target_type}/stacked/limits_{roster_name}_{astro_prior}.fits'

    # Simulated stack variable likelilood file for a particular target (and astro-factor prior)
    sim_stack_likefile_format =\
      '{target_type}_sim/sim_{sim_name}/{target_name}/stacklike_{profile}_{astro_prior}_{seed}.fits'

    # Simulated stack variable limits file for a particular target (and astro-factor prior)
    sim_stack_limitsfile_format =\
      '{target_type}_sim/sim_{sim_name}/{target_name}/stacklimits_{profile}_{astro_prior}_{seed}.fits'

    # Simulated Stacked variable limits file for a particular roster (and astro-factor
    # prior)
    sim_resultsfile_format =\
      '{target_type}_sim/sim_{sim_name}/stacked/results_{roster_name}_{astro_prior}_{seed}.fits'

    # Stacked staked variable limits file for a particular roster (and astro-factor prior)
    sim_stackedlimitsfile_format =\
      '{target_type}_sim/sim_{sim_name}/stacked/limits_{roster_name}_{astro_prior}_{seed}.fits'

    def specconfig(self, **kwargs):
        """ return the name of the input configuration file
        """
        return self._format_from_dict(NameFactory.specconfig_format, **kwargs)

    def specfile(self, **kwargs):
        """ return the name of stacking spectral file
        """
        return self._format_from_dict(NameFactory.specfile_format, **kwargs)


    def stack_likefile(self, **kwargs):
        """ return the name for the stack variable likelilood file for a particular target
        """
        return self._format_from_dict(NameFactory.stack_likefile_format, **kwargs)

    def stack_limitsfile(self, **kwargs):
        """ return the name for the stacking variable limits file for a particular target
        """
        return self._format_from_dict(
            NameFactory.stack_limitsfile_format, **kwargs)

    def resultsfile(self, **kwargs):
        """ return the name for the stacked results file for a particular roster
        """
        return self._format_from_dict(NameFactory.resultsfile_format, **kwargs)

    def stackedlimitsfile(self, **kwargs):
        """ return the name for the stacked limits file for a particular roster
        """
        return self._format_from_dict(
            NameFactory.stackedlimitsfile_format, **kwargs)

    def sim_stack_likefile(self, **kwargs):
        """ return the name for the simulated stack variable likelilood file for a particular target
        """
        if 'seed' not in kwargs:
            kwargs['seed'] = 'SEED'
        return self._format_from_dict(
            NameFactory.sim_stack_likefile_format, **kwargs)

    def sim_stack_limitsfile(self, **kwargs):
        """ return the name for the simulated stack variable limits file for a particular target
        """
        if 'seed' not in kwargs:
            kwargs['seed'] = 'SEED'
        return self._format_from_dict(
            NameFactory.sim_stack_limitsfile_format, **kwargs)

    def sim_resultsfile(self, **kwargs):
        """ return the name for the stacked results file for a particular roster
        """
        if 'seed' not in kwargs:
            kwargs['seed'] = 'SEED'
        return self._format_from_dict(
            NameFactory.sim_resultsfile_format, **kwargs)

    def sim_stackedlimitsfile(self, **kwargs):
        """ return the name for the stacked limits file for a particular roster
        """
        if 'seed' not in kwargs:
            kwargs['seed'] = 'SEED'
        return self._format_from_dict(
            NameFactory.sim_stackedlimitsfile_format, **kwargs)

    def resolve_rosterfile(self, args, require_sim_name=False):
        """Get the name of the roster based on the job arguments"""
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
                         rosterlist='roster_list.yaml',
                         sim_name=sim,
                         fullpath=True)
        if sim is None:
            rosterfile = self.rosterfile(**name_keys)
        else:
            rosterfile = self.sim_rosterfile(**name_keys)

        roster_override = args.get('rosterfile')
        if is_not_null(roster_override):
            rosterfile = roster_override

        return (rosterfile, sim)

    def resolve_specfile(self, args):
        """Get the name of the specturm file based on the job arguments"""
        ttype = args.get('ttype')
        if is_null(ttype):
            sys.stderr.write('Target type must be specified')
            return None
        name_keys = dict(target_type=ttype,
                         fullpath=True)
        specfile = self.specfile(**name_keys)
        spec_override = args.get('specfile')
        if is_not_null(spec_override):
            specfile = spec_override
        return specfile

    def resolve_specconfig(self, args):
        """Get the name of the specturm file based on the job arguments"""
        ttype = args.get('ttype')
        if is_null(ttype):
            sys.stderr.write('Target type must be specified')
            return None
        name_keys = dict(target_type=ttype,
                         fullpath=True)
        specconfig = self.specconfig(**name_keys)
        spec_override = args.get('specconfig')
        if is_not_null(spec_override):
            specconfig = spec_override
        return specconfig
